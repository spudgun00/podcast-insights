#!/usr/bin/env python3
"""
Podcast back-fill & metadata-rich pipeline
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import re
import signal
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import certifi
import yaml
from bs4 import BeautifulSoup               # NEW  ← rss <podcast:person>
from dateutil import parser as dtparse
from tenacity import retry, stop_after_attempt, wait_exponential
import csv
from io import StringIO
from urllib.parse import urlparse
import feedparser # ADDED IMPORT
import spacy # ADDED IMPORT
from sentence_transformers import SentenceTransformer # ADDED IMPORT

# Define _NoAWS shim class at a higher scope so it's always available
class _NoAWS:
    def __getattr__(self, *_):
        return lambda *a, **k: None

from podcast_insights.audio_utils import (
    calculate_audio_hash,
    download_with_retry,
    get_audio_tech,
    estimate_speech_music_ratio,
    verify_audio,
    check_timestamp_support,
)
from podcast_insights.meta_utils import (
    enrich_meta,
    process_transcript,
    load_json_config,
    tidy_people,
    generate_podcast_slug,
    make_slug,
    save_segments_file,
    load_stopwords,
)
from podcast_insights.transcribe import transcribe_audio
from podcast_insights.db_utils import upsert_episode, init_db
from podcast_insights.kpi_utils import extract_kpis
from podcast_insights.settings import (
    S3_BUCKET_RAW, 
    S3_BUCKET_STAGE, 
    BASE_PREFIX, # Assuming this is still relevant for S3 object key structure
    LAYOUT, # For S3 object key structure
    layout_fn as get_s3_prefix # Renamed for clarity, it now ONLY returns S3 prefix string
)
from podcast_insights.feed_utils import feed_items, entry_dt # ADDED entry_dt
from podcast_insights.const import SCHEMA_VERSION # ADDED IMPORT
from podcast_insights.db_utils_dynamo import init_dynamo_db_table, update_episode_status, get_episode_status # ADDED DYNAMO
from podcast_insights.metrics import put_metric_data_value # ADDED METRICS

# --------------------------------------------------------------------------- SSL
os.environ["SSL_CERT_FILE"] = certifi.where()

# ----------------------------------------------------------------------- AWS shim
USE_AWS = not os.getenv("NO_AWS")
AWS_REGION_ENV = os.getenv("AWS_REGION")
AWS_PROFILE_ENV = os.getenv("AWS_PROFILE")

S3 = None # Will be initialized later if USE_AWS
boto_session = None # Shared Boto3 session

if USE_AWS:
    import boto3
    if not AWS_REGION_ENV:
        print("ERROR: AWS_REGION environment variable is not set, but USE_AWS is true. Boto3 clients will likely fail.")
        # Allow to proceed, but expect errors if region is truly needed and not found by Boto3 defaults.
    
    try:
        print(f"Attempting to create Boto3 session. Profile: {AWS_PROFILE_ENV}, Region: {AWS_REGION_ENV}")
        if AWS_PROFILE_ENV and AWS_REGION_ENV:
            boto_session = boto3.Session(profile_name=AWS_PROFILE_ENV, region_name=AWS_REGION_ENV)
        elif AWS_REGION_ENV: # Profile might be default, but region is specified
            boto_session = boto3.Session(region_name=AWS_REGION_ENV)
        elif AWS_PROFILE_ENV: # Region might be in profile's config, but profile is specified
            boto_session = boto3.Session(profile_name=AWS_PROFILE_ENV)
        else:
            boto_session = boto3.Session() # Rely on SDK defaults for credentials and region
        
        S3 = boto_session.client("s3")
        print(f"Boto3 session and S3 client created successfully. Session region: {boto_session.region_name}")

    except Exception as e_session:
        print(f"ERROR: Failed to create Boto3 session or S3 client: {e_session}", file=sys.stderr)

if not S3: # If S3 is still None (either USE_AWS was false, or session creation failed)
    # _NoAWS class is now defined globally
    S3 = _NoAWS()

# -------------------------------------------------------------------------- CLI
pa = argparse.ArgumentParser(description="Back-fill podcast transcripts")
pa.add_argument("--mode", choices=["fetch", "transcribe"], 
                help="Processing mode: 'fetch' (download audio, create minimal meta to S3 raw bucket) "
                     "or 'transcribe' (process audio from raw, enrich, save to S3 stage bucket)")
pa.add_argument("--manifest", type=str, 
                help="Path to a local CSV manifest file or an S3 URI (s3://bucket/key.csv) to process. "
                     "Required for 'fetch' mode. Optional for 'transcribe' mode (can scan S3 raw if not provided).")
pa.add_argument("--dry_run", action="store_true")
pa.add_argument("--since")                  # YYYY-MM-DD override for feed processing if not using manifest
pa.add_argument("--feed")                  # single RSS URL for ad-hoc processing if not using manifest
pa.add_argument("--limit", type=int, default=0, help="Limit the number of episodes to process from manifest or feed list")
pa.add_argument("--model_size",
                choices=["tiny", "base", "small", "medium", "large"],
                help="Whisper model size for transcribe mode.") # Default can be handled by CFG if not specified
pa.add_argument("--episode_details_json",
                    help="JSON string describing one episode to process (full pipeline, bypasses modes for single item)")
args = pa.parse_args()
DRY_RUN = bool(args.dry_run)

# ---------------------------------------------------------------------- logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)7s | %(message)s")
log = logging.getLogger("backfill")
DEBUG_FEED = os.getenv("DEBUG_FEED") == "1"

# ------------------------------------------------------------------- config/YAML
CFG = yaml.safe_load(Path("tier1_feeds.yaml").read_text())   # backfill_test.py uses test-five.yaml
SINCE_DATE = dt.datetime.strptime(
    args.since or CFG.get("since_date", "2025-01-01"), "%Y-%m-%d"
)
DOWNLOAD_PATH = Path(os.getenv("DOWNLOAD_PATH",
                               CFG.get("download_path", "/tmp/audio")))
TRANSCRIPT_PATH = Path(os.getenv("TRANSCRIPT_PATH",
                                 CFG.get("transcript_path", "/tmp/transcripts")))
MODEL_VERSION = f"{args.model_size or CFG.get('model_size','base')}"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# --- Define Local Base Paths --- 
LOCAL_DATA_ROOT = Path(os.getenv("LOCAL_DATA_ROOT", CFG.get("local_data_root", "./data")))
LOCAL_AUDIO_PATH = LOCAL_DATA_ROOT / "audio"
LOCAL_TRANSCRIPTS_PATH = LOCAL_DATA_ROOT / "transcripts"
LOCAL_METADATA_PATH = LOCAL_DATA_ROOT / "metadata"
LOCAL_SEGMENTS_PATH = LOCAL_DATA_ROOT / "segments"
LOCAL_KPIS_PATH = LOCAL_DATA_ROOT / "kpis"
LOCAL_EMBEDDINGS_PATH = LOCAL_DATA_ROOT / "embeddings"
LOCAL_ENTITIES_PATH = LOCAL_DATA_ROOT / "entities"

# --------------------------------------------------------------------- helpers
def md5_8(x: str) -> str:
    return hashlib.md5(x.encode()).hexdigest()[:8]       # noqa: S324

def mark_processed(guid: str, audio_hash: str) -> None:
    """Marks an episode as processed by adding its GUID and audio hash to global sets."""
    if guid: # Ensure guid is not empty or None
        processed_guids.add(guid)
    if audio_hash: # Ensure audio_hash is not empty or None
        processed_hashes.add(audio_hash)

processed_guids: set[str] = set()
processed_hashes: set[str] = set()
TERMINATE = False
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
signal.signal(signal.SIGINT, lambda *_: sys.exit(130))

# --- SpaCy and SentenceTransformer Model Loading (INSERTED HERE) ---
NLP_MODEL_NAME = CFG.get("spacy_model", "en_core_web_sm")
ST_MODEL_NAME = CFG.get("sentence_transformer_model", "all-MiniLM-L6-v2")
NLP_MODEL = None
ST_MODEL = None
MODELS_LOADED_SUCCESSFULLY = False
try:
    log.info(f"Loading SpaCy model: {NLP_MODEL_NAME}...")
    NLP_MODEL = spacy.load(NLP_MODEL_NAME)
    log.info("SpaCy model loaded successfully.")
    log.info(f"Loading SentenceTransformer model: {ST_MODEL_NAME}...")
    ST_MODEL = SentenceTransformer(ST_MODEL_NAME)
    log.info("SentenceTransformer model loaded successfully.")
    MODELS_LOADED_SUCCESSFULLY = NLP_MODEL is not None and ST_MODEL is not None
except Exception as e:
    log.error(f"Failed to load NLP/SentenceTransformer models in backfill.py: {e}. Caching in enrich_meta might be affected if models are not passed.")
# --- End Model Loading ---

# ------------------------------------------------------------------- pipeline
def process(entry, when: dt.datetime, podcast: str, feed_url: str) -> bool:
    guid = entry.get("id", "")
    if guid in processed_guids:
        log.info("skip (guid)")
        return True

    # Generate podcast_slug from the podcast title (feed title)
    podcast_slug = generate_podcast_slug(podcast)

    title = entry.get("title", "")
    # Create a simple slug from the title for the filename component
    simple_title_slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
    # Further truncate if necessary
    simple_title_slug = simple_title_slug[:60] 
    # episode_file_identifier = f"{simple_title_slug.strip('-')}_{md5_8(guid)}" # OLD: uses md5_8(guid)
    # NEW: uses the first 8 characters of the GUID string itself
    guid_prefix = guid[:8] if guid and len(guid) >= 8 else md5_8(guid) # Fallback to md5_8 if guid is short/missing
    episode_file_identifier = f"{simple_title_slug.strip('-')}_{guid_prefix}"
    audio_mp3_fname = f"{episode_file_identifier}.mp3"

    mp3_url = entry.enclosures[0]["href"] if entry.enclosures else ""
    if not mp3_url:
        log.warning(f"skip {title!r} (no .mp3 url in feed entry: {entry})")
        return True

    # ---- define paths & create directories ----------------------------------
    # Ensure base directories exist (handled by get_local_artifact_path now)
    # DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True) # Old way

    audio_file = get_local_artifact_path(
        base_path=LOCAL_AUDIO_PATH, 
        podcast_slug=podcast_slug, 
        episode_identifier=episode_file_identifier, 
        file_name_suffix="audio.mp3"
    )
    # audio_file.parent.mkdir(parents=True, exist_ok=True) # Handled by get_local_artifact_path

    # Calculate audio_hash can now be done AFTER download, or if URL hashing is preferred, keep as is.
    # For consistency with having the file first, let's assume download then hash.
    # If pre-download hashing is critical, this needs adjustment.

    # ---- download -----------------------------------------------------------
    log.info(f"Downloading to: {audio_file}")
    download_successful = download_with_retry(mp3_url, audio_file)
    if not download_successful:
        log.error(f"Failed to download {mp3_url} for {title!r}. Skipping episode.")
        if audio_file.exists(): # Cleanup partial download
            audio_file.unlink()
        return False # Critical failure if download fails

    valid_audio, reason = verify_audio(audio_file)
    if not valid_audio:
        log.error(f"Failed audio verification for {audio_file}: {reason}")
        # Consider cleanup of downloaded file if invalid
        audio_file.unlink(missing_ok=True) # Clean up invalid audio file
        return False

    # ---- Calculate audio_hash from downloaded file content ----
    audio_hash = calculate_audio_hash(str(audio_file)) # Changed from mp3_url
    log.info(f"Calculated audio_hash (from content): {audio_hash} for {audio_file.name}")

    # ---- Check if already processed (using content hash) ----
    if guid in processed_guids or (audio_hash and audio_hash in processed_hashes):
        existing_reason = "guid" if guid in processed_guids else "audio_hash"
        log.info(f"Skipping {title!r} (already processed - {existing_reason})")
        # No need to return True for dry run if we are skipping based on processed_hashes with content hash
        # as the file would not have been downloaded again for a real run.
        return True

    if DRY_RUN:
        log.info(f"DRY RUN: Would process {title!r} (Audio hash if downloaded: {audio_hash})")
        # In a dry run, we can simulate marking it processed with the content hash
        # mark_processed(guid, audio_hash)
        return True # For dry run, after simulating hash and check, we typically stop here for this item.

    # ---- tech metadata (post-download, as it needs the local file) ----------
    tech_meta = get_audio_tech(audio_file)
    speech_music_ratio = estimate_speech_music_ratio(str(audio_file))
    tech_meta["speech_to_music_ratio"] = speech_music_ratio
    tech_meta["timestamp_support"] = check_timestamp_support(str(audio_file))

    # ---- Initialize raw_meta structure --------------------------------------
    # This is the initial metadata extracted directly from the feed and basic audio properties.
    raw_meta = {
        "podcast_title_slug": podcast_slug, # Added for easier reference
        "podcast_title": podcast,
        "episode_title": title,
        "guid": guid,
        "published_date": entry.get("published", ""), # Keep original format
        "parsed_published_date": when.isoformat() if when else None,
        "episode_url": entry.get("link"),
        "audio_url": mp3_url,
        "audio_local_path": str(audio_file), # Path to the downloaded audio
        "audio_hash": audio_hash, # Now using content-based hash
        "duration_seconds": tech_meta.get("duration_seconds"),
        "file_size_bytes": tech_meta.get("file_size_bytes"),
        "mime_type": tech_meta.get("mime_type"),
        "audio_tech_details": tech_meta, # Includes format, channels, sample_rate, bitrate
        "speech_to_music_ratio": speech_music_ratio,
        "supports_timestamp_processing": tech_meta["timestamp_support"],
        "itunes_episode_type": entry.get("itunes_episode_type"),
        "itunes_explicit": entry.get("itunes_explicit") or feedparser.parse(feed_url).feed.get("itunes_explicit"),
        "rights_copyright": entry.get("copyright") or feedparser.parse(feed_url).feed.get("copyright"),
        "feed_categories": [t["term"] for t in entry.get("tags", []) if t.get("term")],
        "feed_hosts": [], # To be populated by parsing <podcast:person>
        "feed_guests": [], # To be populated by parsing <podcast:person>
        "raw_summary": getattr(entry, "summary_detail", {}).get("value", ""),
        "processed_utc": dt.datetime.utcnow().isoformat(),
        "asr_model_name": MODEL_VERSION, # Just the model size like "base" or "large"
        "compute_type": COMPUTE_TYPE, # e.g., int8, float16
        "schema_version": "1.0", # Initial schema version
        # Placeholder for where final meta file will be saved
        "final_meta_json_path": None, 
        # Placeholder for where final transcript (with segments) will be saved
        "final_transcript_json_path": None, 
        "cleaned_entities_path": None, # Will be updated by meta_utils
        "sentence_embeddings_path": None, # Will be updated by meta_utils
        "episode_kpis_path": None, # Will be updated by meta_utils
        # Fields to be populated by transcribe.py and meta_utils.py
        "detected_language": None,
        "transcription_info": {},
        "word_count": 0,
        "char_count": 0,
        "segment_count": 0,
        "avg_words_per_segment": 0,
        "keywords": [],
        "hosts": [],
        "guests": [],
        "entities": [],
        "entity_counts": {},
        "highlights": [], # from KPI extraction
    }

    # ---- Parse <podcast:person> from raw_summary --------------------------
    # Using BeautifulSoup to parse <podcast:person> tags from RSS summary if present
    # This populates raw_meta["feed_hosts"] and raw_meta["feed_guests"]
    if raw_meta["raw_summary"]:
        soup = BeautifulSoup(raw_meta["raw_summary"], "xml") # Use "xml" parser for <podcast:*> tags
        for tag in soup.find_all("podcast:person"):
            person_info = {
                "name": tag.text.strip(),
                "role": (tag.get("role") or "guest").lower(),
                "href": tag.get("href"),
                "img": tag.get("img"), # Capture image if available
                "group": tag.get("group") # Capture group if available
            }
            if person_info["role"] in ("host", "co-host", "presenter"):
                raw_meta["feed_hosts"].append(person_info)
            else:
                raw_meta["feed_guests"].append(person_info)
    
    # ---- Call transcribe_audio (from transcribe.py) -------------------------
    # This function should handle chunking internally if needed and return the path to the main transcript JSON
    # It also needs raw_meta to pass some initial metadata along.
    # The output of transcribe_audio will be a path to a JSON file containing segments and detected language.
    
    # Define where the raw transcript (output of transcribe.py) will be stored.
    raw_transcript_file = get_local_artifact_path(
        base_path=LOCAL_TRANSCRIPTS_PATH,
        podcast_slug=podcast_slug,
        episode_identifier=episode_file_identifier,
        file_name_suffix="raw_transcript.json"
    )
    # raw_transcript_file.parent.mkdir(parents=True, exist_ok=True) # Handled

    log.info(f"Starting transcription for {title} (GUID: {guid})")
    transcription_output_path = transcribe_audio(
        audio_file=audio_file,
        output_dir=raw_transcript_file.parent, # Pass the directory for output
        output_filename_stem=raw_transcript_file.stem, # Pass just the stem for transcribe_audio to append .json
        model_name=MODEL_VERSION,
        language=None,  # Let Whisper detect, or specify if known e.g. CFG.get("language_override").get(podcast_slug)
        compute_type=COMPUTE_TYPE,
        initial_meta=raw_meta, # Pass the initial metadata
        podcast_slug=podcast_slug, # For caching purposes within transcribe_audio
        guid=guid, # For caching and context
        vad_filter=True, # Apply VAD by default
        word_timestamps=True, # Get word timestamps
        # cleanup_audio_file=True # Set to True if you want to delete audio after transcription
    )

    if not transcription_output_path or not transcription_output_path.exists():
        log.error(f"Transcription failed or output not found for {audio_file}. Expected at {transcription_output_path}")
        return False
    
    raw_meta["final_transcript_json_path"] = str(transcription_output_path)
    log.info(f"Raw transcript saved to: {raw_transcript_file}")

    # ---- Enrich Metadata (enrich_meta) ----
    # This step takes the raw_meta and the transcript data, processes entities, keywords, etc.
    # and saves the final rich metadata JSON.
    # It also patches the transcript JSON with additional info.
    
    # Define where the final rich metadata file will be stored.
    # Using the new local path helper, with meta_style_guid_naming for meta_{guid}_...json structure
    final_meta_file = get_local_artifact_path(
        base_path=LOCAL_METADATA_PATH,
        podcast_slug=podcast_slug,
        episode_identifier=episode_file_identifier, # simple_title_slug_guidprefix part
        file_name_suffix="details.json", # The part after meta_{guid}_
        guid=guid,
        meta_style_guid_naming=True
    )
    # final_meta_file.parent.mkdir(parents=True, exist_ok=True) # Handled
    raw_meta["final_meta_json_path"] = str(final_meta_file)

    log.info(f"Enriching metadata for {title} (GUID: {guid})")
    final_meta, updated_transcript_path = enrich_meta(
        raw_meta_path=None, # Not needed if raw_meta_dict is provided
        transcript_path=transcription_output_path,
        people_aliases_path="config/people_aliases.yml",
        known_hosts_path="config/known_hosts.yml",
        output_meta_path=final_meta_file,
        # base_data_dir=TRANSCRIPT_PATH.parent, # for resolving relative paths if any inside meta, though paths should be absolute
        base_data_dir=Path("."), # Current working directory as base for resolving relative paths from config files
        raw_meta_dict=raw_meta # Pass the dictionary directly
    )

    if not final_meta or not final_meta_file.exists():
        log.error(f"Metadata enrichment failed for {title}. Expected meta at {final_meta_file}")
        return False

    log.info(f"Final rich metadata saved to: {final_meta_file}")
    if updated_transcript_path and updated_transcript_path.exists():
        log.info(f"Patched transcript (if changed) at: {updated_transcript_path}")
        raw_meta["final_transcript_json_path"] = str(updated_transcript_path) # Update if path changed (e.g. due to patching)
    else:
        log.warning(f"Updated transcript path not returned or does not exist: {updated_transcript_path}")


    # ---- Generate Segments file (using meta_utils.save_segments_file) -------
    # This uses the *final_meta* because it might contain refined segment information or GUID.
    # The segments file is now named using the GUID.
    segments_file = get_local_artifact_path(
        base_path=LOCAL_SEGMENTS_PATH,
        podcast_slug=podcast_slug,
        episode_identifier=guid, # Using guid directly as the unique part for segments file
        file_name_suffix="segments.json"
    )
    # segments_file.parent.mkdir(parents=True, exist_ok=True) # Handled

    transcript_data_for_segments = json.loads(updated_transcript_path.read_text())
    saved_segments_path = save_segments_file(
        guid=guid, # Use the episode GUID
        podcast_slug=podcast_slug,
        segments=transcript_data_for_segments.get("segments", []),
        # base_data_dir=TRANSCRIPT_PATH.parent # Base for resolving output path if it were relative
        base_data_dir=Path(".") # Segments path template in settings.py is relative to this
    )
    if saved_segments_path and saved_segments_path.exists():
        log.info(f"Segments file saved to: {saved_segments_path}")
        final_meta["segments_path"] = str(saved_segments_path) # Add to final_meta
    else:
        log.error(f"Failed to save segments file for GUID {guid}. Expected at {segments_file}")
        # Decide if this is a critical failure

    # ---- Extract KPIs (using kpi_utils.extract_kpis) -----------------------
    log.info(f"Extracting KPIs for {title} (GUID: {guid})")
    kpi_output_path = get_local_artifact_path(
        base_path=LOCAL_KPIS_PATH,
        podcast_slug=podcast_slug,
        episode_identifier=episode_file_identifier, # Or use guid if preferred for KPI files
        file_name_suffix="kpis.json",
        guid=guid # Available if needed for meta_style_guid_naming option
    )
    # kpi_output_path.parent.mkdir(parents=True, exist_ok=True) # Handled

    kpis = extract_kpis(
        meta_file_path=final_meta_file, 
        # transcript_file_path=updated_transcript_path # transcript content is usually in meta or linked from it
    )
    if kpis:
        kpi_output_path.write_text(json.dumps(kpis, indent=2)) # CHANGED kpi_output_file to kpi_output_path
        log.info(f"KPIs saved to: {kpi_output_path}") # CHANGED kpi_output_file to kpi_output_path
        final_meta["episode_kpis_path"] = str(kpi_output_path) # CHANGED kpi_output_file to kpi_output_path
        # Also update highlights in final_meta if extract_kpis returns them in a structured way
        if "highlights" in kpis:
            final_meta["highlights"] = kpis["highlights"]
    else:
        log.warning(f"KPI extraction produced no output for {final_meta_file}")

    # ---- Update and Resave Final Meta with paths populated by other steps ----
    # This ensures paths like segments_path, kpis_path are in the definitive meta JSON.
    final_meta["final_transcript_json_path"] = str(updated_transcript_path) # ensure it's the patched one
    # final_meta["final_meta_json_path"] is already correct as it's the file we're writing to.
    # Embeddings paths are already set within enrich_meta if generated.

    log.info(f"Resaving final meta for {guid} with all generated artifact paths at {final_meta_file}")
    final_meta_file.write_text(json.dumps(final_meta, indent=2, ensure_ascii=False))

    # ---- Database Upsert (using db_utils.upsert_episode) -------------------
    # Ensure DB is initialized (e.g., tables created)
    # init_db() # Call this once at the start of your application, not per episode.
    log.info(f"Upserting episode to DB: {guid} - {title}")
    try:
        upsert_episode(meta_json_path=final_meta_file)
        log.info(f"Successfully upserted to DB: {guid}")
    except Exception as e:
        log.error(f"Failed to upsert episode {guid} to DB: {e}", exc_info=True)
        # Decide if this is a critical error. For now, log and continue.

    # ---- S3 upload (optional, for final artifacts) --------------------------
    if USE_AWS:
        # Use the get_s3_prefix function (originally layout_fn from settings)
        # This now returns a prefix like "podcast_slug/guid/" because BASE_PREFIX is ""
        s3_base_key_prefix = get_s3_prefix(guid=guid, podcast_slug=podcast_slug)
        
        # Upload the original audio file to S3_BUCKET_RAW
        s3_audio_key = f"{s3_base_key_prefix}audio/{audio_file.name}" # Runbook: <feed_slug>/<guid>/audio/episode.mp3
        log.info(f"Uploading audio to S3: s3://{S3_BUCKET_RAW}/{s3_audio_key}")
        S3.upload_file(str(audio_file), S3_BUCKET_RAW, s3_audio_key)
        final_meta["s3_audio_path"] = f"s3://{S3_BUCKET_RAW}/{s3_audio_key}"

        # All subsequent artifacts go to S3_BUCKET_STAGE
        # Update final_meta with the s3_artifacts_prefix_stage for reference
        final_meta["s3_artifacts_prefix_stage"] = f"s3://{S3_BUCKET_STAGE}/{s3_base_key_prefix}"

        # Upload final meta.json to S3_BUCKET_STAGE into /meta/ subdirectory
        s3_meta_key = f"{s3_base_key_prefix}meta/{final_meta_file.name}" # Runbook: <feed_slug>/<guid>/meta/meta.json
        log.info(f"Uploading final meta to S3: s3://{S3_BUCKET_STAGE}/{s3_meta_key}")
        S3.upload_file(str(final_meta_file), S3_BUCKET_STAGE, s3_meta_key)

        # Upload final transcript.json (patched) to S3_BUCKET_STAGE into /transcripts/ subdirectory
        if updated_transcript_path and updated_transcript_path.exists():
            s3_transcript_key = f"{s3_base_key_prefix}transcripts/{updated_transcript_path.name}" # Runbook: <feed_slug>/<guid>/transcripts/transcript.json
            log.info(f"Uploading final transcript to S3: s3://{S3_BUCKET_STAGE}/{s3_transcript_key}")
            S3.upload_file(str(updated_transcript_path), S3_BUCKET_STAGE, s3_transcript_key)
        
        # Upload segments.json to S3_BUCKET_STAGE into /segments/ subdirectory
        if saved_segments_path and saved_segments_path.exists():
            s3_segments_key = f"{s3_base_key_prefix}segments/{saved_segments_path.name}" # Runbook: <feed_slug>/<guid>/segments/segments.json
            log.info(f"Uploading segments to S3: s3://{S3_BUCKET_STAGE}/{s3_segments_key}")
            S3.upload_file(str(saved_segments_path), S3_BUCKET_STAGE, s3_segments_key)

        # Upload cleaned_entities.json to S3_BUCKET_STAGE into /entities/ subdirectory
        cleaned_entities_local_path_str = final_meta.get("cleaned_entities_path")
        if cleaned_entities_local_path_str:
            cleaned_entities_local_path = Path(cleaned_entities_local_path_str)
            if cleaned_entities_local_path.exists():
                s3_entities_key = f"{s3_base_key_prefix}entities/{cleaned_entities_local_path.name}" # Runbook: <feed_slug>/<guid>/entities/*.json
                log.info(f"Uploading cleaned entities to S3: s3://{S3_BUCKET_STAGE}/{s3_entities_key}")
                S3.upload_file(str(cleaned_entities_local_path), S3_BUCKET_STAGE, s3_entities_key)

        # Upload sentence_embeddings.npy to S3_BUCKET_STAGE into /embeddings/ subdirectory
        sentence_embeddings_local_path_str = final_meta.get("sentence_embeddings_path")
        if sentence_embeddings_local_path_str:
            sentence_embeddings_local_path = Path(sentence_embeddings_local_path_str)
            if sentence_embeddings_local_path.exists():
                s3_embeddings_key = f"{s3_base_key_prefix}embeddings/{sentence_embeddings_local_path.name}" # Runbook: <feed_slug>/<guid>/embeddings/*.npy
                log.info(f"Uploading sentence embeddings to S3: s3://{S3_BUCKET_STAGE}/{s3_embeddings_key}")
                S3.upload_file(str(sentence_embeddings_local_path), S3_BUCKET_STAGE, s3_embeddings_key)

        # Upload kpis.json to S3_BUCKET_STAGE into /kpis/ subdirectory
        episode_kpis_local_path_str = final_meta.get("episode_kpis_path")
        if episode_kpis_local_path_str:
            episode_kpis_local_path = Path(episode_kpis_local_path_str)
            if episode_kpis_local_path.exists():
                s3_kpis_key = f"{s3_base_key_prefix}kpis/{episode_kpis_local_path.name}" # Runbook: <feed_slug>/<guid>/kpis/kpis.json
                log.info(f"Uploading KPIs to S3: s3://{S3_BUCKET_STAGE}/{s3_kpis_key}")
                S3.upload_file(str(episode_kpis_local_path), S3_BUCKET_STAGE, s3_kpis_key)
        
        # Re-save final_meta with updated S3 paths (s3_audio_path, s3_artifacts_prefix_stage)
        final_meta_file.write_text(json.dumps(final_meta, indent=2, ensure_ascii=False))
        log.info(f"Re-saved final meta with S3 paths: {final_meta_file}")

    # ---- Mark as processed (locally) ----
    mark_processed(guid, audio_hash) # audio_hash is now content-based
    log.info(f"Successfully processed and marked: {podcast} - {title} (GUID: {guid})")

    # ---- Optional: Cleanup downloaded audio file ----
    if CFG.get("cleanup_downloaded_audio", False) and audio_file.exists():
        log.info(f"Cleaning up downloaded audio file: {audio_file}")
        # audio_file.unlink(missing_ok=True)
        # Be careful with cleanup if other processes might need the audio, 
        # e.g. if S3 upload happens much later or in a different step.
        # For now, keeping it commented out. Deletion is handled by transcribe_audio if enabled there.

        return True

def process_feed_url(url: str, limit_per_feed: int | None = None, total_processed_count: int = 0, overall_limit: int = 0) -> int:
    if TERMINATE:
        return total_processed_count

    # Call the new feed_items function, passing SINCE_DATE and DEBUG_FEED
    items = feed_items(url, SINCE_DATE, DEBUG_FEED)
    
    count_this_feed = 0
    try:
        for entry, when, podcast_title, feed_url_val in items:
            if TERMINATE:
                log.info("Termination signal received, stopping feed processing.")
                break 
            if overall_limit and total_processed_count >= overall_limit:
                log.info(f"Overall processing limit ({overall_limit}) reached.")
                break
            if limit_per_feed is not None and count_this_feed >= limit_per_feed:
                log.info(f"Per-feed limit ({limit_per_feed}) reached for {url}.")
                break

            log.info(f"Processing item: {podcast_title} - {entry.get('title')} (Published: {when})")
            try:
                if process(entry, when, podcast_title, feed_url_val):
                    count_this_feed += 1
                    total_processed_count +=1 # Increment overall counter passed by reference or managed by caller
            except Exception as e:
                guid = entry.get("id", "[unknown_guid]")
                ep_title = entry.get("title", "[unknown_title]")
                log.error(f"Failed to process episode {ep_title} (GUID: {guid}) from feed {url}: {e}", exc_info=True)
                # Optionally, write error to a separate log or database for tracking failures
                # with open("failed_episodes.log", "a") as f:
                #     f.write(f"{dt.datetime.utcnow().isoformat()} | {guid} | {ep_title} | {url} | {e}\n{traceback.format_exc()}\n")
    except Exception as e:
        log.error(f"Failed to fetch or iterate feed {url}: {e}", exc_info=True)
    return count_this_feed

def process_single_episode(episode_details_json: str) -> bool:
    """Processes a single episode based on JSON details."""
    try:
        details = json.loads(episode_details_json)
        # Basic validation of required fields
        required_fields = ["id", "title", "enclosures", "published_parsed", "podcast_title", "feed_url"]
        if not all(field in details for field in required_fields):
            log.error(f"Missing one or more required fields in --episode_details_json: {required_fields}")
            log.error(f"Provided details: {details}")
            return False
        
        # Reconstruct a feedparser-like entry object
        entry = {
            "id": details["id"],
            "title": details["title"],
            "link": details.get("link"),
            "published": details.get("published"), # Original string form
            "published_parsed": tuple(dtparse.parse(details["published_parsed"]).timetuple()) if details.get("published_parsed") else None,
            "updated_parsed": tuple(dtparse.parse(details["updated_parsed"]).timetuple()) if details.get("updated_parsed") else None,
            "summary_detail": {"value": details.get("summary", "")},
            "itunes_episode_type": details.get("itunes_episode_type"),
            "itunes_explicit": details.get("itunes_explicit"),
            "copyright": details.get("copyright"),
            "tags": details.get("tags", []), # e.g. [{"term": "Technology"}]
            "enclosures": details["enclosures"] # e.g. [{"href": "url.mp3", "type": "audio/mpeg"}]
        }
        if not entry["enclosures"] or not isinstance(entry["enclosures"], list) or not entry["enclosures"][0].get("href"):
            log.error("Invalid or missing enclosures in --episode_details_json. Need at least one enclosure with an href.")
            return False

        when = entry_dt(entry) # Parse datetime object
        if not when:
            log.error(f"Could not parse valid date from 'published_parsed' or other date fields for episode {details['id']}")
            return False

        podcast_title = details["podcast_title"]
        feed_url = details["feed_url"]

        log.info(f"Processing single episode: {podcast_title} - {entry['title']} (GUID: {entry['id']})")
        return process(entry, when, podcast_title, feed_url)

    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON provided for --episode_details_json: {e}")
        return False
    except Exception as e:
        log.error(f"Error processing single episode from JSON: {e}", exc_info=True)
        return False

# NEW: Placeholder functions for mode-specific logic
def run_fetch_mode(args_fetch: argparse.Namespace):
    log.info(f"Running in FETCH mode. Manifest: {args_fetch.manifest}, Limit: {args_fetch.limit}, DryRun: {args_fetch.dry_run}")
    if not args_fetch.manifest:
        log.error("--manifest is required for fetch mode.")
        return 1 # Indicate error

    try:
        episodes_to_fetch = _read_manifest_csv(args_fetch.manifest)
    except Exception as e:
        # Error already logged by _read_manifest_csv or _parse_s3_uri
        log.error(f"Could not proceed with fetch mode due to manifest reading error.")
        return 1

    if not episodes_to_fetch:
        log.warning("Manifest is empty or could not be read. Nothing to fetch.")
        return 0

    processed_count = 0
    for i, episode_data in enumerate(episodes_to_fetch):
        if args_fetch.limit and processed_count >= args_fetch.limit:
            log.info(f"Fetch limit ({args_fetch.limit}) reached. Stopping.")
            break

        # Extract data from manifest row (ensure these keys match your manifest.csv headers)
        try:
            guid = episode_data["episode_guid"]
            podcast_title = episode_data["podcast_title"]
            episode_title = episode_data["episode_title"]
            mp3_url = episode_data["mp3_url"]
            feed_url = episode_data["feed_url"]
            published_date_iso = episode_data["published_date_iso"]
        except KeyError as e:
            log.error(f"Manifest row {i+1} is missing expected key: {e}. Row data: {episode_data}. Skipping.")
            continue

        log.info(f"Fetching episode {i+1}/{len(episodes_to_fetch)}: '{episode_title}' (GUID: {guid}) from {mp3_url}")

        # 1. Generate podcast_slug
        # Assuming generate_podcast_slug is available from meta_utils
        # If not, we might need a simpler slugify here or ensure it's imported.
        # For now, let's assume a simple slugification for podcast_title if generate_podcast_slug is not directly usable here
        # or if podcast_title from manifest is already slug-like. The runbook implies <feed_slug> which might be available directly.
        # For simplicity, if manifest provides a podcast_slug, use it, else slugify podcast_title.
        podcast_slug = episode_data.get("podcast_slug") # Check if manifest provides it
        if not podcast_slug:
            podcast_slug = generate_podcast_slug(podcast_title) # from meta_utils

        # Create a unique identifier for the episode file, similar to what's in process()
        # This ensures consistency if the same episode is processed by different paths.
        guid_prefix_for_filename = guid[:8] if guid and len(guid) >= 8 else md5_8(guid)
        episode_file_identifier = f"{make_slug(episode_title=episode_title[:60], published_iso_date=published_date_iso, podcast_name_or_slug=podcast_slug)}_{guid_prefix_for_filename}"

        if DRY_RUN:
            log.info(f"[DRY RUN] Would process '{episode_title}' (GUID: {guid}). Slug: {podcast_slug}, FileID: {episode_file_identifier}")
            log.info(f"[DRY RUN]  -> Audio download from: {mp3_url}")
            log.info(f"[DRY RUN]  -> Audio S3 Target: s3://{S3_BUCKET_RAW}/{podcast_slug}/{guid}/audio/{episode_file_identifier}_audio.mp3")
            log.info(f"[DRY RUN]  -> Minimal Meta S3 Target: s3://{S3_BUCKET_RAW}/{podcast_slug}/{guid}/meta/meta.json")
            processed_count += 1
            continue

        # 2. Determine local audio download path
        with tempfile.TemporaryDirectory(prefix="podcast_fetch_") as tmpdir:
            local_audio_download_path = Path(tmpdir) / f"{episode_file_identifier}_audio.mp3"

            # --- Metrics for Download Attempt ---
            if USE_AWS:
                put_metric_data_value(
                    metric_name="DownloadAttempt", value=1, unit="Count",
                    dimensions=[{'Name': 'FeedSlug', 'Value': podcast_slug}, {'Name': 'Mode', 'Value': 'Fetch'}]
                )
                try:
                    parsed_url = urlparse(mp3_url)
                    hostname = parsed_url.hostname if parsed_url.hostname else "unknown-host"
                    put_metric_data_value(metric_name="DownloadAttemptPerHost", value=1, unit="Count", dimensions=[{'Name': 'Hostname', 'Value': hostname}])
                except Exception as e_metric_host:
                    log.warning(f"Failed to parse hostname for DownloadAttemptPerHost metric: {e_metric_host}") 

            # 3. Download audio
            log.info(f"Downloading audio for '{episode_title}' to {local_audio_download_path}")
            download_successful = False # Initialize
            download_error_code = None # To store potential HTTP error like 429
            
            # Modify download_with_retry or its call site to return error info if possible
            # For now, we assume download_with_retry uses tenacity and might have an exception containing response
            # This is a simplified way to check for 429. A more robust way would be for download_with_retry
            # to explicitly return a status or specific exception for rate limiting.
            try:
                # Start timer for download
                download_start_time = time.perf_counter()
                download_successful = download_with_retry(mp3_url, local_audio_download_path)
                download_duration_ms = (time.perf_counter() - download_start_time) * 1000
            except Exception as e_download: # Catching broad exception to check for status_code
                log.error(f"Exception during download attempt for {mp3_url}: {e_download}")
                if hasattr(e_download, 'response') and hasattr(e_download.response, 'status_code'):
                    download_error_code = e_download.response.status_code
                download_successful = False # Ensure it's false on exception
                download_duration_ms = (time.perf_counter() - download_start_time) * 1000 # Record duration even on failure

            if USE_AWS:
                dimensions_feed_mode = [{'Name': 'FeedSlug', 'Value': podcast_slug}, {'Name': 'Mode', 'Value': 'Fetch'}]
                hostname_dim = []
                try:
                    parsed_url = urlparse(mp3_url)
                    hostname = parsed_url.hostname if parsed_url.hostname else "unknown-host"
                    hostname_dim = [{'Name': 'Hostname', 'Value': hostname}]
                except Exception as e_metric_host_detail:
                     log.warning(f"Failed to parse hostname for detailed download metrics: {e_metric_host_detail}")

                if download_successful:
                    put_metric_data_value(metric_name="DownloadSuccess", value=1, unit="Count", dimensions=dimensions_feed_mode)
                    if hostname_dim: put_metric_data_value(metric_name="DownloadSuccessPerHost", value=1, unit="Count", dimensions=hostname_dim)
                    put_metric_data_value(metric_name="DownloadDuration", value=download_duration_ms, unit="Milliseconds", dimensions=dimensions_feed_mode)
                    if local_audio_download_path.exists():
                        file_size_bytes = local_audio_download_path.stat().st_size
                        put_metric_data_value(metric_name="AvgDownloadMB", value=(file_size_bytes / (1024*1024)), unit="Megabytes", dimensions=dimensions_feed_mode)

                else: # Download failed
                    put_metric_data_value(metric_name="DownloadFailure", value=1, unit="Count", dimensions=dimensions_feed_mode)
                    if hostname_dim: put_metric_data_value(metric_name="DownloadFailurePerHost", value=1, unit="Count", dimensions=hostname_dim)
                    if download_error_code == 429:
                        log.warning(f"Download for {mp3_url} failed with 429 (rate-limiting). Recording PerHost429 metric.")
                        if hostname_dim: put_metric_data_value(metric_name="PerHost429", value=1, unit="Count", dimensions=hostname_dim)
                    elif download_error_code:
                        log.warning(f"Download for {mp3_url} failed with HTTP error code: {download_error_code}.")
                        if hostname_dim: put_metric_data_value(metric_name=f"PerHostHttpError{download_error_code}", value=1, unit="Count", dimensions=hostname_dim)
            else:
                        log.warning(f"Download for {mp3_url} failed with an unknown error (not an HTTP error with status code).")

            if download_successful:
                # 4a. Calculate audio_content_hash
                audio_content_hash = calculate_audio_hash(str(local_audio_download_path))
                log.info(f"Calculated content hash: {audio_content_hash} for {local_audio_download_path.name}")

                s3_episode_key_prefix = get_s3_prefix(guid=guid, podcast_slug=podcast_slug)

                # 4b. Upload audio to S3_BUCKET_RAW
                s3_audio_key = f"{s3_episode_key_prefix}audio/{local_audio_download_path.name}"
                log.info(f"Uploading audio to S3: s3://{S3_BUCKET_RAW}/{s3_audio_key}")
                try:
                    S3.upload_file(str(local_audio_download_path), S3_BUCKET_RAW, s3_audio_key)
                    log.info(f"Audio uploaded successfully to {s3_audio_key}")
                except Exception as e:
                    log.error(f"Failed to upload audio {local_audio_download_path.name} to S3: {e}", exc_info=True)
                    continue # Skip to next episode if S3 upload fails

                # 4c. Create minimal_meta_content
                minimal_meta_content = {
                    "guid": guid,
                    "podcast_slug": podcast_slug,
                    "podcast_title": podcast_title,
                    "episode_title": episode_title,
                    "feed_url": feed_url,
                    "published_date_iso": published_date_iso,
                    "mp3_url_original": mp3_url,
                    "s3_audio_path_raw": f"s3://{S3_BUCKET_RAW}/{s3_audio_key}",
                    "audio_content_hash": audio_content_hash,
                    "fetch_processed_utc": dt.datetime.utcnow().isoformat(),
                    "processing_status": "fetched",
                    "schema_version": "1.0_minimal_fetch"
                }

                # 4d. Determine local path for minimal_meta.json (in the same temp dir for atomicity with upload)
                local_minimal_meta_path = Path(tmpdir) / "meta.json" # Standardized name for S3
                
                # 4e. Save minimal_meta_content to local temporary path
                try:
                    with open(local_minimal_meta_path, 'w', encoding='utf-8') as f_meta:
                        json.dump(minimal_meta_content, f_meta, indent=2)
                    log.info(f"Minimal meta saved locally: {local_minimal_meta_path}")
                except IOError as e:
                    log.error(f"Failed to write local minimal meta {local_minimal_meta_path}: {e}", exc_info=True)
                    continue # Skip if local write fails

                # 4f. Upload minimal_meta.json to S3_BUCKET_RAW
                s3_minimal_meta_key = f"{s3_episode_key_prefix}meta/meta.json" # Standardized name in S3
                log.info(f"Uploading minimal meta to S3: s3://{S3_BUCKET_RAW}/{s3_minimal_meta_key}")
                try:
                    S3.upload_file(str(local_minimal_meta_path), S3_BUCKET_RAW, s3_minimal_meta_key)
                    log.info(f"Minimal meta uploaded successfully to {s3_minimal_meta_key}")
                except Exception as e:
                    log.error(f"Failed to upload minimal meta {local_minimal_meta_path.name} to S3: {e}", exc_info=True)
                    # Continue, but the episode is in a partial state (audio uploaded, meta not)
                    # This might require a cleanup or reconciliation step later.
                    # For now, we log and proceed to next episode.
                    continue

                # After successful S3 uploads in fetch mode, update DynamoDB
                if USE_AWS: # Only attempt DynamoDB update if AWS is enabled
                    s3_minimal_meta_path_raw = f"s3://{S3_BUCKET_RAW}/{s3_minimal_meta_key}"
                    dynamo_attrs_to_update = {
                        "podcast_slug": podcast_slug,
                        "episode_title": episode_title,
                        "podcast_title": podcast_title,
                        "feed_url": feed_url,
                        "published_date_iso": published_date_iso,
                        "mp3_url_original": mp3_url,
                        "s3_audio_path_raw": minimal_meta_content["s3_audio_path_raw"],
                        "s3_minimal_meta_path_raw": s3_minimal_meta_path_raw,
                        "audio_content_hash": audio_content_hash,
                        "processing_status": "fetched",
                        "fetch_timestamp_utc": minimal_meta_content["fetch_processed_utc"],
                        "last_updated_utc": dt.datetime.utcnow().isoformat(),
                        "schema_version_dynamo": "1.0_status"
                    }
                    log.info(f"Updating DynamoDB status for GUID {guid} to 'fetched'.")
                    if not update_episode_status(episode_guid=guid, attributes_to_update=dynamo_attrs_to_update):
                        log.warning(f"Failed to update DynamoDB status for GUID {guid}. Continuing...")
                    else:
                        log.info(f"DynamoDB status updated successfully for GUID {guid}.")
                else:
                    log.info("USE_AWS is false, skipping DynamoDB status update for fetch mode.")
                
                # 4h. mark_processed (optional for fetch mode, depends on how you track progress across modes)
                # If using a global processed list that spans fetch and transcribe, then mark it.
                # Hashing by content is good. Note that `processed_hashes` set is in-memory for this run.
                # If Batch jobs are independent, this in-memory set won't persist across them.
                # DynamoDB in runbook is better for distributed state.
                # mark_processed(guid, audio_content_hash)
                # log.info(f"Marked as processed (fetch): {guid} with hash {audio_content_hash}")

            else: # Download failed
                log.error(f"Download failed for '{episode_title}' (GUID: {guid}) from {mp3_url}. Skipping.")
                # local_audio_download_path is in tempdir, will be cleaned up.
                continue # Skip to next episode
            
            # 4g. local_audio_download_path and local_minimal_meta_path are in tmpdir and will be auto-cleaned up.

        processed_count += 1
    
    log.info(f"Fetch mode finished. Processed {processed_count} episodes.")
    return 0

# --- Placeholder for the core transcription and enrichment logic ---
# This will be populated by refactoring parts of the original process() function.
def _transcribe_and_enrich_episode(
    guid: str,
    podcast_slug: str,
    episode_title: str, # For logging/context
    local_audio_path: Path,
    initial_minimal_meta: Dict[str, Any], # Content of meta.json from S3_BUCKET_RAW
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Path]]]:
    """
    Takes a downloaded audio file and its minimal metadata, performs transcription,
    enrichment, KPI extraction, etc., saves artifacts locally (in a temp dir managed by caller),
    and returns the final enriched metadata and a dict of paths to local artifacts.
    """
    log.info(f"Starting transcription and enrichment for GUID: {guid}, Episode: {episode_title} at {local_audio_path}")
    
    temp_base_path = local_audio_path.parent

    guid_prefix_for_filename = guid[:8] if guid and len(guid) >= 8 else md5_8(guid)
    published_date_iso_for_slug = initial_minimal_meta.get("published_date_iso", "")
    if not published_date_iso_for_slug:
        log.warning(f"published_date_iso not found in initial_minimal_meta for GUID {guid} for slug generation. Using current date as fallback.")
        published_date_iso_for_slug = dt.datetime.utcnow().isoformat()
    episode_file_identifier_stem = f"{make_slug(episode_title=episode_title[:60], published_iso_date=published_date_iso_for_slug, podcast_name_or_slug=podcast_slug)}_{guid_prefix_for_filename}"

    local_artifacts: Dict[str, Path] = {}
    final_enriched_meta: Optional[Dict[str, Any]] = None

    try:
        tech_meta_dict = get_audio_tech(local_audio_path)
        speech_music_ratio = estimate_speech_music_ratio(str(local_audio_path))
        tech_meta_dict["speech_to_music_ratio"] = speech_music_ratio
        tech_meta_dict["timestamp_support"] = check_timestamp_support(str(local_audio_path))

        current_meta = initial_minimal_meta.copy()
        current_meta.update({
            "podcast_title_slug": podcast_slug,
            "episode_title": episode_title,
            "audio_local_path": str(local_audio_path),
            "duration_seconds": tech_meta_dict.get("duration_seconds"),
            "file_size_bytes": tech_meta_dict.get("file_size_bytes"),
            "mime_type": tech_meta_dict.get("mime_type"),
            "audio_tech_details": tech_meta_dict, # This will be the 'tech' arg for enrich_meta
            "speech_to_music_ratio": speech_music_ratio,
            "supports_timestamp_processing": tech_meta_dict["timestamp_support"],
            "processed_utc_transcribe_enrich_start": dt.datetime.utcnow().isoformat(),
            "asr_model_name": MODEL_VERSION,
            "compute_type": COMPUTE_TYPE,
            "detected_language": None,
            "transcription_info": {},
            "word_count": 0,
            "char_count": 0,
            "segment_count": 0,
            "avg_words_per_segment": 0,
            "keywords": [],
            "hosts": [],
            "guests": [],
            "entities": [],
            "entity_counts": {},
            "highlights": [],
            "final_meta_json_path": None,
            "final_transcript_json_path": None,
            "cleaned_entities_path": None,
            "sentence_embeddings_path": None,
            "episode_kpis_path": None,
            "segments_file_path": None,
        })
        if "feed_hosts" not in current_meta: current_meta["feed_hosts"] = []
        if "feed_guests" not in current_meta: current_meta["feed_guests"] = []
        if "raw_summary" not in current_meta: current_meta["raw_summary"] = ""


        raw_transcript_file = temp_base_path / f"{episode_file_identifier_stem}_raw_transcript.json"
        
        log.info(f"Starting transcription for {episode_title} (GUID: {guid}) output to {raw_transcript_file}")
        # transcribe_audio now returns a dictionary of information, including paths it created.
        # The primary transcript JSON path is raw_transcript_file.
        transcription_result_info = transcribe_audio( # RENAMED from transcription_output_path
            audio_file=local_audio_path,
            output_file=str(raw_transcript_file),
            model_size=current_meta["asr_model_name"],
            compute_type=current_meta["compute_type"],
            metadata=current_meta,
            podcast_slug=podcast_slug,
            enable_word_timestamps=True,
            perform_entity_embedding_caching=True # transcribe_audio itself can cache if models are passed
        )

        if not transcription_result_info:
            log.error(f"Transcription failed for {local_audio_path}. transcribe_audio returned None or empty.")
            return None, None
        
        if not raw_transcript_file.exists():
            log.error(f"Transcription output file not found at {raw_transcript_file} despite transcribe_audio success.")
            return None, None

        current_meta["final_transcript_json_path"] = str(raw_transcript_file)
        local_artifacts["transcript"] = raw_transcript_file
        log.info(f"Raw transcript saved to: {raw_transcript_file}")

        # ---- Enrich Metadata (enrich_meta) ----
        # This step takes the raw_meta and the transcript data, processes entities, keywords, etc.
        # and saves the final rich metadata JSON.
        # It also patches the transcript JSON with additional info.
        
        # Define where the final rich metadata file will be stored.
        # Using the new local path helper, with meta_style_guid_naming for meta_{guid}_...json structure
        final_meta_file = temp_base_path / f"meta_{guid}_details.json"
        current_meta["final_meta_json_path"] = str(final_meta_file)

        log.info(f"Enriching metadata for {episode_title} (GUID: {guid}), output to {final_meta_file}")
        
        # Prepare arguments for enrich_meta
        # 'entry' is the core episode metadata, from initial_minimal_meta
        # 'feed_details' is broader podcast-level info, also largely from initial_minimal_meta
        # 'tech' is audio technical details
        
        # Load transcript text and segments from raw_transcript_file
        transcript_content = {}
        if raw_transcript_file.exists():
            try:
                transcript_content = json.loads(raw_transcript_file.read_text(encoding='utf-8'))
            except Exception as e_json:
                log.error(f"Failed to load transcript JSON from {raw_transcript_file} for enrich_meta: {e_json}")
                return None, None
        
        transcript_text_for_enrich = transcript_content.get("text")
        transcript_segments_for_enrich = transcript_content.get("segments")

        # 'entry' data for enrich_meta can be initial_minimal_meta itself
        # 'feed_details' can also be largely derived from initial_minimal_meta fields like 'podcast_title', 'podcast_feed_url' etc.
        # For simplicity, pass initial_minimal_meta for both and let enrich_meta pick what it needs.
        feed_details_for_enrich = {
            "title": initial_minimal_meta.get("podcast_title"),
            "url": initial_minimal_meta.get("feed_url"), # Assuming this was in initial_minimal_meta
            "generator": initial_minimal_meta.get("podcast_generator"),
            "language": initial_minimal_meta.get("podcast_language"),
            "itunes_author": initial_minimal_meta.get("podcast_itunes_author"),
            "copyright": initial_minimal_meta.get("podcast_copyright"),
            "itunes_explicit": initial_minimal_meta.get("podcast_itunes_explicit") # podcast level explicit
        }

        enriched_meta_dict = enrich_meta(
            entry=initial_minimal_meta, # Contains episode specific details like title, guid, published date, summary
            feed_details=feed_details_for_enrich, # Contains podcast level details
            tech=tech_meta_dict, # Contains audio technical details
            transcript_text=transcript_text_for_enrich,
            transcript_segments=transcript_segments_for_enrich,
            perform_caching=True, # Default, but explicit. enrich_meta uses its globally loaded models if NLP_MODEL/ST_MODEL not passed.
            nlp_model=NLP_MODEL, # Pass loaded model
            st_model=ST_MODEL,   # Pass loaded model
            base_data_dir=str(temp_base_path), # For path construction within enrich_meta if it saves files
            podcast_slug=podcast_slug,
            word_timestamps_enabled=current_meta.get("supports_timestamp_processing", False) 
            # enrich_meta does not return updated_transcript_path_from_enrich
        )

        if not enriched_meta_dict:
            log.error(f"Metadata enrichment failed for {episode_title}. enrich_meta returned None/empty.")
            return None, None

        # enrich_meta returns the dictionary. The caller is responsible for saving it.
        try:
            final_meta_file.write_text(json.dumps(enriched_meta_dict, indent=2, ensure_ascii=False))
            log.info(f"Enriched metadata dictionary saved to {final_meta_file}")
        except Exception as e_save:
            log.error(f"Failed to save enriched metadata to {final_meta_file}: {e_save}")
            return None, None # Critical failure if we can't save it

        final_enriched_meta = enriched_meta_dict
        local_artifacts["final_meta"] = final_meta_file
        log.info(f"Final rich metadata processed and file path noted: {final_meta_file}")

        # The transcript path should not change during enrich_meta unless enrich_meta itself modifies and re-saves it
        # under a new name, which its current signature does not suggest.
        # The raw_transcript_file is the source of truth for transcript text/segments.
        # final_enriched_meta might contain an updated "final_transcript_json_path" if enrich_meta set it.
        
        # Use the path from current_meta which points to raw_transcript_file,
        # as enrich_meta doesn't return a new transcript path.
        # Patches to transcript (like adding cleaned_entities_path) happen inside enrich_meta itself if it loads/saves.
        # The 'updated_transcript_path_from_enrich' variable is removed as enrich_meta doesn't return it.
        # The raw_transcript_file (pointed by current_meta["final_transcript_json_path"]) is what enrich_meta might patch.
        
        # Update local_artifacts with paths from final_enriched_meta for entities/embeddings
        # These paths are constructed within enrich_meta relative to its base_data_dir (temp_base_path)
        # We need to ensure these are Path objects for consistency in local_artifacts.
        # enrich_meta should return string paths that are relative to its base_data_dir or absolute within it.
        
        # Example: if final_enriched_meta["cleaned_entities_path"] is "entities_cleaned/podcast_slug/guid_clean.json"
        # then local_artifacts["entities"] becomes temp_base_path / "entities_cleaned/podcast_slug/guid_clean.json"
        
        # Check paths from enrich_meta. They should be absolute paths within temp_base_path or just filenames.
        # enrich_meta seems to return resolved absolute paths now.
        
        cleaned_entities_path_str = final_enriched_meta.get("cleaned_entities_path")
        if cleaned_entities_path_str:
            # Ensure it's a Path object and it exists. enrich_meta should have saved it.
            cleaned_entities_p = Path(cleaned_entities_path_str)
            if cleaned_entities_p.exists() and cleaned_entities_p.is_file():
                 local_artifacts["entities_cleaned"] = cleaned_entities_p # Changed key from "entities"
            else:
                log.warning(f"Cleaned entities path from enrich_meta does not exist or not a file: {cleaned_entities_path_str}")
        
        sentence_embeddings_path_str = final_enriched_meta.get("sentence_embeddings_path") # Corrected key
        if sentence_embeddings_path_str:
            sentence_embeddings_p = Path(sentence_embeddings_path_str)
            if sentence_embeddings_p.exists() and sentence_embeddings_p.is_file():
                local_artifacts["embeddings"] = sentence_embeddings_p
            else:
                log.warning(f"Sentence embeddings path from enrich_meta does not exist or not a file: {sentence_embeddings_path_str}")

        # Raw entities path (if generated by transcribe_audio's call to _generate_spacy_entities_file)
        # transcribe_audio stores this in transcription_result_info['entities_output_path']
        raw_entities_path_str = transcription_result_info.get("entities_output_path")
        if raw_entities_path_str:
            raw_entities_p = Path(raw_entities_path_str)
            if raw_entities_p.exists() and raw_entities_p.is_file():
                local_artifacts["entities_raw"] = raw_entities_p # New key for raw entities
            else:
                log.warning(f"Raw entities path from transcribe_audio does not exist or not a file: {raw_entities_path_str}")
        
        # Raw embeddings path (if generated by transcribe_audio's call to _generate_sentence_embedding_file)
        # transcribe_audio stores this in transcription_result_info['embedding_output_path']
        raw_embeddings_path_str = transcription_result_info.get("embedding_output_path")
        if raw_embeddings_path_str:
            raw_embeddings_p = Path(raw_embeddings_path_str)
            if raw_embeddings_p.exists() and raw_embeddings_p.is_file():
                local_artifacts["embeddings_raw"] = raw_embeddings_p # New key for raw embeddings (though typically same as 'embeddings' if transcribe_audio does it)
            else:
                log.warning(f"Raw embeddings path from transcribe_audio does not exist or not a file: {raw_embeddings_path_str}")


        # ---- Generate Segments file (save_segments_file) ----
        # Uses final_enriched_meta and its transcript path.
        # Segments file named using GUID, stored in temp_base_path.
        segments_file = temp_base_path / f"segments_{guid}.json"
        
        # Load the (potentially patched) transcript data for creating segments file
        transcript_for_segments_path = Path(final_enriched_meta["final_transcript_json_path"])
        if transcript_for_segments_path.exists():
            transcript_data_for_segments = json.loads(transcript_for_segments_path.read_text())
            # save_segments_file from meta_utils now needs output_dir and filename parameters
            saved_segments_path = save_segments_file(
                guid=guid,
                podcast_slug=podcast_slug,
                segments=transcript_data_for_segments.get("segments", []),
                output_dir=temp_base_path, # Explicitly pass the output directory
                filename_override=segments_file.name # Explicitly pass the desired filename
            )

            if saved_segments_path and saved_segments_path.exists():
                log.info(f"Segments file saved to: {saved_segments_path}")
                final_enriched_meta["segments_file_path"] = str(saved_segments_path) # Path within temp_base_path
                local_artifacts["segments"] = saved_segments_path
            else:
                log.error(f"Failed to save segments file for GUID {guid}. Expected at {segments_file}")
                # Non-critical, log and continue
        else:
            log.error(f"Transcript file for segments not found at {transcript_for_segments_path}")

        # ---- Extract KPIs (extract_kpis) ----
        log.info(f"Extracting KPIs for {episode_title} (GUID: {guid}) from {final_meta_file}")
        kpi_output_file = temp_base_path / f"kpis_{guid}.json"
        
        kpis_dict = extract_kpis(meta_file_path=final_meta_file)
        if kpis_dict:
            kpi_output_file.write_text(json.dumps(kpis_dict, indent=2))
            log.info(f"KPIs saved to: {kpi_output_file}")
            final_enriched_meta["episode_kpis_path"] = str(kpi_output_file) # Path within temp_base_path
            if "highlights" in kpis_dict: # Update highlights in the main meta
                final_enriched_meta["highlights"] = kpis_dict["highlights"]
            local_artifacts["kpis"] = kpi_output_file
        else:
            log.warning(f"KPI extraction produced no output for {final_meta_file}")

        # ---- Final update to the enriched_meta dict before returning ----
        # Ensure all local artifact paths within final_enriched_meta are relative to temp_base_path or absolute temp paths
        # The S3 upload logic in run_transcribe_mode will handle placing them correctly in S3_STAGE.
        final_enriched_meta["processed_utc_transcribe_enrich_end"] = dt.datetime.utcnow().isoformat()
        final_enriched_meta["processing_status"] = "enriched_locally" # Indicates successful local processing

        # Re-save the final_meta_file with all updated paths and statuses, if it changed.
        # This ensures the version uploaded to S3 is the most complete for this stage.
        final_meta_file.write_text(json.dumps(final_enriched_meta, indent=2, ensure_ascii=False))
        log.info(f"Final meta (with all local artifact paths) re-saved at: {final_meta_file}")

        # Remove absolute paths from dicts if they were constructed using temp_base_path for entities/embeddings by enrich_meta directly in the dict
        # The local_artifacts dict should hold the true Path objects for upload.
        # final_enriched_meta should store relative string paths if that's the convention from enrich_meta.
        # For now, assuming enrich_meta stores relative paths and they are correctly resolved by temp_base_path / path_str for local_artifacts.

        return final_enriched_meta, local_artifacts

    except Exception as e:
        log.error(f"Error during _transcribe_and_enrich_episode for GUID {guid}: {e}", exc_info=True)
        # Cleanup any partially created files in temp_base_path might be tricky here,
        # but TemporaryDirectory from caller should handle overall cleanup.
        return None, None # Indicate failure

def run_transcribe_mode(args_transcribe: argparse.Namespace):
    log.info(f"Running in TRANSCRIBE mode. Manifest: {args_transcribe.manifest}, Limit: {args_transcribe.limit}, Model: {args_transcribe.model_size or MODEL_VERSION}, DryRun: {args_transcribe.dry_run}")

    if not args_transcribe.manifest:
        # TODO: Implement S3 scan logic as an alternative if manifest is not provided.
        # For now, manifest is required for transcribe mode.
        log.error("--manifest is required for transcribe mode (S3 scan not yet implemented).")
        return 1

    try:
        episodes_to_process_from_manifest = _read_manifest_csv(args_transcribe.manifest)
    except Exception as e:
        log.error(f"Could not proceed with transcribe mode due to manifest reading error: {e}")
        return 1

    if not episodes_to_process_from_manifest:
        log.warning("Manifest for transcribe mode is empty or could not be read. Nothing to transcribe.")
        return 0

    processed_count = 0
    for i, manifest_row in enumerate(episodes_to_process_from_manifest):
        if args_transcribe.limit and processed_count >= args_transcribe.limit:
            log.info(f"Transcribe limit ({args_transcribe.limit}) reached. Stopping.")
            break

        # Extract necessary identifiers from the manifest row.
        # This manifest might provide direct S3 paths or just identifiers to construct them.
        # Assuming it provides at least guid and podcast_slug to find items in S3_BUCKET_RAW.
        try:
            guid = manifest_row["episode_guid"] # Must match generate_manifest.py output
            podcast_slug = manifest_row.get("podcast_slug") # Should be there from fetch or generate_manifest
            if not podcast_slug:
                 # Attempt to get from s3_audio_path_raw if available and parse, or error out
                s3_audio_path_raw = manifest_row.get("s3_audio_path_raw")
                if s3_audio_path_raw:
                    # s3://bucket/podcast_slug/guid/audio/file.mp3 -> extract podcast_slug
                    path_parts = s3_audio_path_raw.replace(f"s3://{S3_BUCKET_RAW}/", "").split("/")
                    if len(path_parts) > 1:
                        podcast_slug = path_parts[0]
                if not podcast_slug:
                    log.error(f"Manifest row {i+1} missing 'podcast_slug' and cannot derive it. GUID: {guid}. Skipping.")
                    continue
            
            # Determine S3 paths for inputs from S3_BUCKET_RAW
            s3_episode_key_prefix_raw = get_s3_prefix(guid=guid, podcast_slug=podcast_slug) # e.g. podcast_slug/guid/
            
            # Attempt to get the audio filename from manifest if provided (e.g., from fetch mode's output meta)
            # Otherwise, we might need a convention or to list S3 dir (less ideal).
            # Let's assume minimal_meta.json from RAW will have `s3_audio_path_raw` which includes the filename.
            s3_minimal_meta_key_raw = f"{s3_episode_key_prefix_raw}meta/meta.json"
            # Audio path will be derived after fetching and parsing minimal_meta_raw.

        except KeyError as e:
            log.error(f"Manifest row {i+1} is missing expected basic key (e.g., episode_guid): {e}. Row: {manifest_row}. Skipping.")
            continue
        
        episode_title_for_log = manifest_row.get("episode_title", guid) # For logging
        log.info(f"To transcribe episode {i+1}/{len(episodes_to_process_from_manifest)}: '{episode_title_for_log}' (GUID: {guid})")

        # Check DynamoDB status before proceeding (if USE_AWS)
        if USE_AWS:
            existing_status_item = get_episode_status(episode_guid=guid)
            if existing_status_item:
                current_ddb_status = existing_status_item.get("processing_status")
                skippable_statuses = ["completed", "transcribed"] 
                if current_ddb_status in skippable_statuses:
                    log.info(f"GUID {guid} already has status '{current_ddb_status}' in DynamoDB. Skipping transcribe mode processing.")
                    continue 
                elif current_ddb_status == "fetched":
                    log.info(f"GUID {guid} has status 'fetched'. Proceeding with transcription.")
                else:
                    log.info(f"GUID {guid} has status '{current_ddb_status}'. Proceeding as it's not a final skippable state for transcribe mode.")
            else:
                log.info(f"No existing DynamoDB status found for GUID {guid}. Proceeding with transcription.")
        else:
            log.info("USE_AWS is false, not checking DynamoDB status before transcription.")

        if DRY_RUN:
            log.info(f"[DRY RUN] Would download audio & minimal_meta for GUID {guid} from s3://{S3_BUCKET_RAW}/{s3_episode_key_prefix_raw}")
            log.info(f"[DRY RUN]  -> Then transcribe, enrich, and upload artifacts to s3://{S3_BUCKET_STAGE}/{s3_episode_key_prefix_raw}")
            processed_count += 1
            continue
        
        with tempfile.TemporaryDirectory(prefix=f"podcast_transcribe_{guid[:8]}_") as tmp_episode_dir_str:
            tmp_episode_dir = Path(tmp_episode_dir_str)
            local_minimal_meta_path = tmp_episode_dir / "minimal_meta.json"
            # local_audio_path will be set after parsing minimal_meta

            # 1. Download minimal_meta.json from S3_BUCKET_RAW
            try:
                log.info(f"Downloading minimal meta from s3://{S3_BUCKET_RAW}/{s3_minimal_meta_key_raw} to {local_minimal_meta_path}")
                S3.download_file(S3_BUCKET_RAW, s3_minimal_meta_key_raw, str(local_minimal_meta_path))
                initial_minimal_meta = json.loads(local_minimal_meta_path.read_text())
                log.info(f"Minimal meta downloaded and parsed for GUID {guid}")
            except Exception as e:
                log.error(f"Failed to download or parse minimal_meta.json for GUID {guid} from s3://{S3_BUCKET_RAW}/{s3_minimal_meta_key_raw}: {e}", exc_info=True)
                continue # Skip to next episode

            # 2. Download audio file from S3_BUCKET_RAW (path from minimal_meta)
            s3_audio_path_raw_str = initial_minimal_meta.get("s3_audio_path_raw")
            if not s3_audio_path_raw_str or not s3_audio_path_raw_str.startswith(f"s3://{S3_BUCKET_RAW}/"):
                log.error(f"Invalid or missing 's3_audio_path_raw' in minimal_meta for GUID {guid}. Found: {s3_audio_path_raw_str}. Skipping.")
                continue
            
            s3_audio_key_raw = s3_audio_path_raw_str.replace(f"s3://{S3_BUCKET_RAW}/", "")
            audio_filename = Path(s3_audio_key_raw).name
            local_audio_path = tmp_episode_dir / audio_filename

            try:
                log.info(f"Downloading audio from s3://{S3_BUCKET_RAW}/{s3_audio_key_raw} to {local_audio_path}")
                S3.download_file(S3_BUCKET_RAW, s3_audio_key_raw, str(local_audio_path))
                log.info(f"Audio downloaded for GUID {guid}: {local_audio_path}")
            except Exception as e:
                log.error(f"Failed to download audio for GUID {guid} from s3://{S3_BUCKET_RAW}/{s3_audio_key_raw}: {e}", exc_info=True)
                continue # Skip to next episode

            # 3. Call _transcribe_and_enrich_episode
            transcribe_enrich_start_time = time.perf_counter()
            final_enriched_meta, local_artifacts = _transcribe_and_enrich_episode(
                guid=guid,
                podcast_slug=podcast_slug, 
                episode_title=initial_minimal_meta.get("episode_title", episode_title_for_log),
                local_audio_path=local_audio_path,
                initial_minimal_meta=initial_minimal_meta
            )

            if not final_enriched_meta or not local_artifacts:
                log.error(f"Transcription and enrichment failed for GUID {guid}. Skipping S3 stage upload.")
                if USE_AWS:
                    put_metric_data_value(
                        metric_name="TranscribeEnrichFailure", value=1, unit="Count",
                        dimensions=[{'Name': 'FeedSlug', 'Value': podcast_slug}, {'Name': 'GUID', 'Value': guid}]
                    )
                continue 
            
            transcribe_enrich_duration_ms = (time.perf_counter() - transcribe_enrich_start_time) * 1000
            log.info(f"Transcription and enrichment successful for GUID {guid} in {transcribe_enrich_duration_ms:.2f} ms. Artifacts generated in {tmp_episode_dir}")
            if USE_AWS:
                put_metric_data_value(
                    metric_name="TranscribeEnrichSuccess", value=1, unit="Count",
                    dimensions=[{'Name': 'FeedSlug', 'Value': podcast_slug}]
                )
                put_metric_data_value(
                    metric_name="EpisodeProcessingTime", value=transcribe_enrich_duration_ms, unit="Milliseconds",
                    dimensions=[{'Name': 'FeedSlug', 'Value': podcast_slug}, {'Name': 'Guid', 'Value': guid}]
                )

            # 4. Upload all generated artifacts to S3_BUCKET_STAGE
            s3_episode_key_prefix_stage = get_s3_prefix(guid=guid, podcast_slug=podcast_slug) # e.g. podcast_slug/guid/
            
            # This loop needs to be robust for all artifact types defined in RUNBOOK.md for STAGE
            upload_success_all_artifacts = True
            s3_artifact_paths_for_meta: Dict[str, str] = {}

            for artifact_type, local_path in local_artifacts.items():
                if local_path and local_path.exists():
                    s3_stage_subdir = ""
                    # Field name in final_enriched_meta to update with S3 path
                    meta_field_key_for_s3_path = None 

                    if artifact_type == "final_meta": 
                        s3_stage_subdir = "meta/"
                        # This artifact (final_meta itself) is handled specially after loop
                    elif artifact_type == "transcript": 
                        s3_stage_subdir = "transcripts/"
                        meta_field_key_for_s3_path = "final_transcript_json_path"
                    elif artifact_type == "segments": 
                        s3_stage_subdir = "segments/"
                        meta_field_key_for_s3_path = "segments_file_path"
                    elif artifact_type == "kpis": 
                        s3_stage_subdir = "kpis/"
                        meta_field_key_for_s3_path = "episode_kpis_path"
                    elif artifact_type == "entities": 
                        s3_stage_subdir = "entities/"
                        meta_field_key_for_s3_path = "cleaned_entities_path"
                    elif artifact_type == "embeddings": 
                        s3_stage_subdir = "embeddings/"
                        meta_field_key_for_s3_path = "sentence_embeddings_path"
                    else:
                        log.warning(f"Unknown artifact type '{artifact_type}' for S3 STAGE upload. Skipping {local_path.name}")
                        continue
                    
                    # Skip special handling for final_meta within this loop, it's uploaded last with updated S3 paths.
                    if artifact_type == "final_meta":
                        continue

                    s3_artifact_key_stage = f"{s3_episode_key_prefix_stage}{s3_stage_subdir}{local_path.name}"
                    log.info(f"Uploading {artifact_type} ('{local_path.name}') to s3://{S3_BUCKET_STAGE}/{s3_artifact_key_stage}")
                    try:
                        S3.upload_file(str(local_path), S3_BUCKET_STAGE, s3_artifact_key_stage)
                        s3_artifact_paths_for_meta[artifact_type] = f"s3://{S3_BUCKET_STAGE}/{s3_artifact_key_stage}"
                        if meta_field_key_for_s3_path:
                             # Store the S3 path for updating the final_meta dict later
                            s3_artifact_paths_for_meta[meta_field_key_for_s3_path] = f"s3://{S3_BUCKET_STAGE}/{s3_artifact_key_stage}"
                    except Exception as e:
                        log.error(f"Failed to upload {artifact_type} {local_path.name} to S3 STAGE: {e}", exc_info=True)
                        upload_success_all_artifacts = False
                else:
                    log.warning(f"Local artifact path for '{artifact_type}' not found or invalid: {local_path}. Skipping upload.")
            
            if upload_success_all_artifacts:
                log.info(f"All dependent artifacts for GUID {guid} uploaded to s3://{S3_BUCKET_STAGE}/{s3_episode_key_prefix_stage}")
                
                # Now, prepare and upload the final_meta.json with updated S3 paths
                if "final_meta" in local_artifacts and local_artifacts["final_meta"].exists():
                    final_meta_local_path = local_artifacts["final_meta"]
                    # Create a new dict for the S3 version of meta, starting from the locally enriched one.
                    s3_stage_meta_content = final_enriched_meta.copy() 

                    # Update paths in s3_stage_meta_content to point to S3 STAGE locations
                    for field_key, s3_url in s3_artifact_paths_for_meta.items():
                        if field_key in s3_stage_meta_content: # Ensure the key exists before updating
                            s3_stage_meta_content[field_key] = s3_url
                    
                    # Ensure s3_audio_path points to RAW bucket (should be inherited from initial_minimal_meta)
                    if "s3_audio_path_raw" in initial_minimal_meta:
                        s3_stage_meta_content["s3_audio_path"] = initial_minimal_meta["s3_audio_path_raw"]
                    elif "s3_audio_path" not in s3_stage_meta_content: # If somehow missing
                        log.warning(f"s3_audio_path not found in meta for {guid}, attempting to reconstruct from initial_minimal_meta.")
                        # This case should ideally not happen if fetch_mode populates s3_audio_path_raw

                    s3_stage_meta_content["s3_artifacts_prefix_stage"] = f"s3://{S3_BUCKET_STAGE}/{s3_episode_key_prefix_stage}"
                    s3_stage_meta_content["processing_status"] = "completed" # Final status
                    s3_stage_meta_content["schema_version"] = SCHEMA_VERSION # Use global schema from const

                    # Define S3 key for the final_meta.json in STAGE
                    s3_final_meta_key_stage = f"{s3_episode_key_prefix_stage}meta/{final_meta_local_path.name}"
                    
                    # Save this modified s3_stage_meta_content to a new temporary local file before upload
                    # to ensure we upload the version with S3 paths.
                    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=tmp_episode_dir, delete=False, suffix='_s3_meta.json') as tmp_s3_meta_file:
                        json.dump(s3_stage_meta_content, tmp_s3_meta_file, indent=2, ensure_ascii=False)
                        tmp_s3_meta_file_path = tmp_s3_meta_file.name
                    
                    log.info(f"Uploading final meta with S3 paths to s3://{S3_BUCKET_STAGE}/{s3_final_meta_key_stage}")
                    try:
                        S3.upload_file(tmp_s3_meta_file_path, S3_BUCKET_STAGE, s3_final_meta_key_stage)
                        log.info(f"Final meta for {guid} successfully uploaded to STAGE.")
                        # TODO: Optional - Update status in S3_RAW meta.json or DynamoDB
                    except Exception as e:
                        log.error(f"Failed to upload final meta for {guid} to S3 STAGE: {e}", exc_info=True)
                        upload_success_all_artifacts = False # Mark as overall failure if this crucial step fails
                    finally:
                        Path(tmp_s3_meta_file_path).unlink(missing_ok=True) # Clean up the temp S3 meta file
                else:
                    log.error(f"Final meta artifact not found locally for {guid}. Cannot upload to STAGE.")
                    upload_success_all_artifacts = False
            
            if not upload_success_all_artifacts:
                log.error(f"One or more artifact uploads (or final meta update) failed for GUID {guid} to S3 STAGE.")

            # Temporary files in tmp_episode_dir are cleaned up automatically by TemporaryDirectory context manager.

            # Update DynamoDB status to "completed"
            if USE_AWS:
                dynamo_attrs_completed = {
                    "processing_status": "completed",
                    "s3_artifacts_prefix_stage": s3_stage_meta_content.get("s3_artifacts_prefix_stage"),
                    "s3_final_meta_path_stage": f"s3://{S3_BUCKET_STAGE}/{s3_final_meta_key_stage}",
                    "transcribe_enrich_timestamp_utc": final_enriched_meta.get("processed_utc_transcribe_enrich_end"), # from _t_a_e
                    "last_updated_utc": dt.datetime.utcnow().isoformat(),
                    "schema_version_dynamo": "1.0_status", # Consistent schema version
                    # Include other key paths from s3_stage_meta_content if desired for quick access in Dynamo
                    "s3_transcript_path_stage": s3_stage_meta_content.get("final_transcript_json_path"),
                    "s3_segments_path_stage": s3_stage_meta_content.get("segments_file_path"),
                    "s3_kpis_path_stage": s3_stage_meta_content.get("episode_kpis_path"),
                }
                # Remove None values to avoid writing them to DynamoDB, or ensure they are handled by update_episode_status
                dynamo_attrs_completed = {k: v for k, v in dynamo_attrs_completed.items() if v is not None}
                
                log.info(f"Updating DynamoDB status for GUID {guid} to 'completed'.")
                if not update_episode_status(episode_guid=guid, attributes_to_update=dynamo_attrs_completed):
                    log.warning(f"Failed to update DynamoDB status for GUID {guid} to 'completed'.")
                else:
                    log.info(f"DynamoDB status for GUID {guid} updated to 'completed' successfully.")
            else:
                log.info("USE_AWS is false, skipping DynamoDB status update for transcribe mode completion.")
        # End of the 'with tempfile.TemporaryDirectory' block

        processed_count += 1 # This should be the last line in the for loop body
    # End of the main 'for' loop in run_transcribe_mode

    log.info(f"Transcribe mode finished. Attempted to process {processed_count} episodes.")
    return 0

# ------------------------------------------------------------------------ main
def main() -> int:
    try:
        # init_db() # This is for SQLite, keep if still used for legacy path
        # Initialize DynamoDB table
        if USE_AWS and boto_session: # Only attempt DynamoDB init if AWS is enabled AND session was created
            log.info("Initializing DynamoDB table (if not exists)...")
            if not init_dynamo_db_table(session=boto_session): # Pass the session
                log.error("Failed to initialize DynamoDB table. Operations requiring DynamoDB might fail.")
                # Depending on strictness, might exit: return 1
            else:
                log.info("DynamoDB table initialized successfully or already exists.")
        elif USE_AWS and not boto_session:
            log.error("USE_AWS is true, but Boto3 session was not created. Skipping DynamoDB initialization.")
        else:
            log.info("USE_AWS is false, skipping DynamoDB initialization.")
            
        log.info("Database initialized (if required).")
    except Exception as e:
        log.error(f"Error during DB initialization: {e}", exc_info=True)
        # Depending on DB importance, might exit: return 1

    # Ensure LOCAL_DATA_ROOT and its main subdirectories exist at startup
    paths_to_create = [
        LOCAL_DATA_ROOT,
        LOCAL_AUDIO_PATH,
        LOCAL_TRANSCRIPTS_PATH,
        LOCAL_METADATA_PATH,
        LOCAL_SEGMENTS_PATH,
        LOCAL_KPIS_PATH,
        LOCAL_EMBEDDINGS_PATH,
        LOCAL_ENTITIES_PATH
    ]
    for p_init in paths_to_create:
        p_init.mkdir(parents=True, exist_ok=True)
    log.info(f"Ensured local data directories exist under {LOCAL_DATA_ROOT}")

    # Handle single episode processing first (bypasses modes)
    if args.episode_details_json:
        log.info(f"Processing single episode from JSON details: {args.episode_details_json}")
        if process_single_episode(args.episode_details_json):
            log.info("Single episode processed successfully.")
            return 0
        else:
            log.error("Single episode processing failed.")
            return 1

    # Mode-driven processing
    if args.mode == "fetch":
        # Before running fetch, check if we should skip based on DynamoDB status?
        # This would require reading the manifest first, then checking each GUID.
        # For now, fetch mode processes all manifest items unless limit is hit.
        # If an item is re-fetched, its DynamoDB record will be updated.
        return run_fetch_mode(args)
    elif args.mode == "transcribe":
        # In transcribe_mode, we *definitely* want to check DynamoDB before processing.
        # This will be added when implementing the main loop of run_transcribe_mode.
        return run_transcribe_mode(args)
    elif args.mode: # Should not happen if choices are set correctly
        log.error(f"Invalid mode specified: {args.mode}. Choose from 'fetch', 'transcribe'.")
        return 1
    else: # Default behavior if no mode is specified (legacy or direct feed processing)
        log.info("No specific --mode provided. Proceeding with legacy feed processing (if --feed is used) or default behavior.")
        # Legacy feed processing (original main loop)
        feeds_to_process = [args.feed] if args.feed else CFG.get("feeds", [])
        if not feeds_to_process:
            log.warning("No feeds specified via --feed or in config, and no --mode given. Nothing to do.")
            return 0

        total_processed_count = 0
        limit_per_feed = CFG.get("limit_per_feed")

        for url in feeds_to_process:
            if TERMINATE:
                log.info("Termination signal received, stopping main loop.")
                break
            if args.limit and total_processed_count >= args.limit:
                log.info(f"Overall processing limit ({args.limit}) reached before processing all feeds.")
                break
            log.info(f"Starting legacy processing for feed: {url}")
            processed_for_this_feed = process_feed_url(url, limit_per_feed, total_processed_count, args.limit)
            total_processed_count += processed_for_this_feed
            log.info(f"Processed {processed_for_this_feed} episodes from feed: {url}. Total processed so far: {total_processed_count}")
        
        log.info(f"Legacy feed processing finished. Total episodes processed: {total_processed_count}")
    return 0

    # Fallback if no conditions met, though argparse choices should prevent unknown modes.
    # If mode is required, argparse will handle it if `required=True` is set on --mode.
    # For now, if --mode is not given, it falls through to legacy feed processing or does nothing if no feeds.
    # If --mode is a required part of the new workflow, this else block for legacy processing might be removed
    # or argparse for --mode made `required=True`.
    # log.warning("No valid operation mode or legacy feed parameters specified. Exiting.")
    # return 0

# NEW: Helper function for constructing local file paths consistently
def get_local_artifact_path(
    base_path: Path, 
    podcast_slug: str, 
    episode_identifier: str, # guid_prefix or similar unique part for the episode
    file_name_suffix: str, # e.g., "audio.mp3", "meta.json", "segments.json"
    guid: str | None = None, # Optional, for specific naming like meta_{guid}_...
    meta_style_guid_naming: bool = False # If true, use meta_{guid}_{suffix}.json style
) -> Path:
    """Constructs a local path for an artifact."""
    # Example: data/metadata/podcast-slug/episode-identifier/meta_guid_....json
    #          data/audio/podcast-slug/episode-identifier_audio.mp3 
    
    dir_path = base_path / podcast_slug
    if not meta_style_guid_naming: # Standard naming: {episode_identifier}_{suffix}
        # For audio, raw transcript, etc.
        # Ensure episode_identifier itself doesn't contain problematic chars for filenames
        # (slugify_title in meta_utils helps for the simple_title_slug part of episode_identifier)
        filename = f"{episode_identifier}_{file_name_suffix}"
    else:
        # For meta files that might use GUID directly in name like meta_{guid}_details.json
        if not guid:
            raise ValueError("GUID is required for meta_style_guid_naming")
        filename = f"meta_{guid}_{file_name_suffix}" # was: final_meta_file in backfill.py for meta_{guid}_...

    # Ensure the directory exists
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path / filename

# --- Helper for manifest reading ---
def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}. Must start with s3://")
    parts = s3_uri[5:].split("/", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid S3 URI: {s3_uri}. Must be s3://bucket/key")
    return parts[0], parts[1]

def _read_manifest_csv(manifest_path_or_uri: str) -> List[Dict[str, str]]:
    """Reads a CSV manifest from a local path or S3 URI."""
    records = []
    if manifest_path_or_uri.startswith("s3://"):
        if not USE_AWS or not S3 or isinstance(S3, _NoAWS) or not boto_session:
            log.error("S3 client or Boto3 session not available, cannot read S3 manifest. Ensure AWS is configured and session initialized.")
            raise RuntimeError("S3 client or Boto3 session not available for S3 manifest.")
        bucket_name, key = _parse_s3_uri(manifest_path_or_uri)
        try:
            log.info(f"Downloading manifest from s3://{bucket_name}/{key} using S3 client from session.")
            # S3 client is already initialized from boto_session if USE_AWS is true and session succeeded.
            response = S3.get_object(Bucket=bucket_name, Key=key)
            content = response["Body"].read().decode("utf-8")
            # Use StringIO to treat the string content as a file for csv.DictReader
            csvfile = StringIO(content)
            reader = csv.DictReader(csvfile)
            for row in reader:
                records.append(row)
        except Exception as e:
            log.error(f"Failed to read or parse manifest from S3 s3://{bucket_name}/{key}: {e}", exc_info=True)
            raise # Re-raise after logging
    else:
        local_manifest_path = Path(manifest_path_or_uri)
        if not local_manifest_path.exists():
            log.error(f"Local manifest file not found: {local_manifest_path}")
            raise FileNotFoundError(f"Local manifest file not found: {local_manifest_path}")
        try:
            log.info(f"Reading manifest from local file: {local_manifest_path}")
            with open(local_manifest_path, mode='r', encoding='utf-8', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    records.append(row)
        except Exception as e:
            log.error(f"Failed to read or parse manifest from local file {local_manifest_path}: {e}", exc_info=True)
            raise # Re-raise after logging
    log.info(f"Read {len(records)} records from manifest: {manifest_path_or_uri}")
    return records

if __name__ == "__main__":
    # The main() function now returns an exit code, which sys.exit will use.
    # Initialization of local paths was moved into main().
    exit_code = main()
    sys.exit(exit_code) 