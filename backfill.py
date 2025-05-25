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
from typing import Any, Dict, List, Tuple

import certifi
import feedparser
import yaml
from bs4 import BeautifulSoup               # NEW  ← rss <podcast:person>
from dateutil import parser as dtparse
from tenacity import retry, stop_after_attempt, wait_exponential

from podcast_insights.audio_utils import (
    calculate_audio_hash,
    chunk_long_audio,
    download_with_retry,
    get_audio_tech,
    estimate_speech_music_ratio,
    verify_audio,
    verify_s3_upload,
    check_timestamp_support,
)
from podcast_insights.meta_utils import (
    enrich_meta,
    process_transcript,
    load_json_config,
    tidy_people,
    generate_podcast_slug,
    make_slug,
    cleanup_old_meta,
    save_segments_file,
    load_stopwords,
)
from podcast_insights.transcribe import transcribe_audio
from podcast_insights.db_utils import upsert_episode, init_db
from podcast_insights.kpi_utils import extract_kpis
from podcast_insights.settings import layout_fn, BUCKET

# --------------------------------------------------------------------------- SSL
os.environ["SSL_CERT_FILE"] = certifi.where()

# ----------------------------------------------------------------------- AWS shim
USE_AWS = not os.getenv("NO_AWS")
if USE_AWS:
    import boto3

    S3 = boto3.client("s3")
else:

    class _NoAWS:
        def __getattr__(self, *_):          # every attribute → no-op fn
            return lambda *a, **k: None

    S3 = _NoAWS()                           # type: ignore

# -------------------------------------------------------------------------- CLI
pa = argparse.ArgumentParser(description="Back-fill podcast transcripts")
pa.add_argument("--dry_run", action="store_true")
pa.add_argument("--since")                  # YYYY-MM-DD override
pa.add_argument("--feed")                  # single RSS URL
pa.add_argument("--limit", type=int, default=0)
pa.add_argument("--model_size",
                choices=["tiny", "base", "small", "medium", "large"])
pa.add_argument("--episode_details_json",
                    help="JSON string describing one episode to process")
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

processed_guids: set[str] = set()
processed_hashes: set[str] = set()
TERMINATE = False
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
signal.signal(signal.SIGINT, lambda *_: sys.exit(130))

# --------------------------------------------------------------------- helpers
def mark_processed(guid: str, audio_hash: str) -> None:
    """Marks an episode as processed by adding its GUID and audio hash to global sets."""
    if guid: # Ensure guid is not empty or None
        processed_guids.add(guid)
    if audio_hash: # Ensure audio_hash is not empty or None
        processed_hashes.add(audio_hash)

def md5_8(x: str) -> str:
    return hashlib.md5(x.encode()).hexdigest()[:8]       # noqa: S324


def entry_dt(e) -> dt.datetime | None:
    p = e.get("published_parsed") or e.get("updated_parsed")
    if p:
        return dt.datetime(*p[:6])
    for k in ("published", "updated", "pubDate"):
        if raw := e.get(k):
            try:
                return dtparse.parse(raw)
            except Exception:
                pass
    return None


def feed_items(url: str) -> list[Tuple[Any, dt.datetime, str, str]]:
    log.info(f"Fetch {url}")
    d = feedparser.parse(url)
    title = getattr(d.feed, "title", url)
    out = []
    log.info(f"Found {len(d.entries)} entries in {title}")
    for ent in d.entries:
        when = entry_dt(ent)
        if DEBUG_FEED:
            log.debug(f"  ↳ {ent.get('title')!r} {when}")
        if when and when >= SINCE_DATE:
            out.append((ent, when, title, url))
            log.info(f"Added {ent.get('title')!r} from {when}")
    if not out:
        log.warning(f"No items ≥{SINCE_DATE:%Y-%m-%d} in {url}")
    return out


@retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(6))
def run_transcribe(chunk: Path, out_json: Path, meta: dict, podcast_slug: str) -> None:
    import subprocess, sys, json
    # Determine the base data directory (e.g., "data/" if transcripts are in "data/transcripts/")
    # This assumes TRANSCRIPT_PATH is something like /path/to/project/data/transcripts
    base_data_dir_for_caching = TRANSCRIPT_PATH.parent

    command = [
        sys.executable,
        "-m",
        "podcast_insights.transcribe",
        "--file",
        str(chunk),
        "--output",
        str(out_json),
        "--model_size",
        meta["asr_model"],
        "--metadata_json", # Changed from --metadata
        json.dumps(meta),
        "--enable_caching", # Added for new functionality
        "--base_data_dir", # Added for new functionality
        str(base_data_dir_for_caching), # Added for new functionality
        "--podcast_slug", # ADDED: Pass podcast_slug to transcribe.py
        podcast_slug      # ADDED: Pass podcast_slug to transcribe.py
    ]
    # VAD filter is no longer an explicit CLI argument for transcribe.py
    # if "vad_filter" in meta: # Example: only add if relevant, but it's removed from transcribe.py
    #    command.extend(["--vad_filter", str(meta["vad_filter"])])
    
    log.info(f"Running transcription command: {' '.join(command)}")
    subprocess.run(
        command,
        check=True,
    )

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
    # Ensure base directories exist
    DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
    # TRANSCRIPT_PATH.mkdir(parents=True, exist_ok=True) # Not used directly for final output, meta_utils handles paths

    # audio_file = DOWNLOAD_PATH / podcast_slug / audio_mp3_fname # OLD structure
    audio_file = layout_fn("audio_path_template",
                            podcast_slug=podcast_slug,
                            episode_slug=episode_file_identifier,
                            file_ext="mp3")
    audio_file.parent.mkdir(parents=True, exist_ok=True) # Ensure specific episode audio dir exists


    # Calculate audio_hash using the URL before downloading (or after if preferred, but URL is good for early check)
    # We use the *actual* audio URL for the hash, not a constructed filename.
    audio_hash = calculate_audio_hash(mp3_url)
    log.info(f"Calculated audio_hash: {audio_hash} for {mp3_url}")

    if guid in processed_guids or audio_hash in processed_hashes:
        existing_reason = "guid" if guid in processed_guids else "audio_hash"
        log.info(f"Skipping {title!r} (already processed - {existing_reason})")
        return True

    if DRY_RUN:
        log.info(f"DRY RUN: Would process {title!r}")
        # In a dry run, we might still want to mark as processed to simulate the run accurately
        # However, for testing, it might be better not to mark, to allow repeated dry runs on same data.
        # For now, let's not mark during dry run, assuming tests might re-run.
        # mark_processed(guid, audio_hash) # Optional: mark as processed even in dry run
        return True

    # ---- download -----------------------------------------------------------
    log.info(f"Downloading to: {audio_file}")
    download_with_retry(mp3_url, audio_file)
    valid_audio, reason = verify_audio(audio_file)
    if not valid_audio:
        log.error(f"Failed audio verification for {audio_file}: {reason}")
        # Consider cleanup of downloaded file if invalid
        # audio_file.unlink(missing_ok=True)
        return False

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
        "audio_hash": audio_hash,
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
    # This is before full metadata enrichment.
    # episode_transcript_identifier = f"{simple_title_slug.strip('-')}_{guid_prefix}" # Same as audio
    raw_transcript_file = layout_fn("transcript_path_template",
                                    podcast_slug=podcast_slug,
                                    episode_slug=episode_file_identifier, # Use the same identifier as audio
                                    file_ext="json")
    raw_transcript_file.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Starting transcription for {title} (GUID: {guid})")
    transcription_output_path = transcribe_audio(
        audio_file_path=audio_file,
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
    log.info(f"Raw transcript saved to: {transcription_output_path}")

    # ---- Load stopwords for keyword extraction (once per run or based on config)
    # This should ideally be loaded once. For now, loading here for simplicity in process().
    stopwords_config = load_json_config("config/stopwords.json") # Using a JSON config for stopwords
    stopwords = load_stopwords(stopwords_config.get("files", ["config/stopwords.txt"]))

    # ---- Enrich Metadata (using meta_utils.enrich_meta) ---------------------
    # This step takes the raw_meta and the transcript data, processes entities, keywords, etc.
    # and saves the final rich metadata JSON.
    # It also patches the transcript JSON with additional info.
    
    # Define where the final rich metadata file will be stored.
    final_meta_file = layout_fn("meta_path_template",
                                podcast_slug=podcast_slug,
                                episode_slug=episode_file_identifier, # Consistent identifier
                                guid=guid, # Pass guid for potential use in naming (e.g., meta_{guid}_...)
                                file_ext="json")
    final_meta_file.parent.mkdir(parents=True, exist_ok=True)
    raw_meta["final_meta_json_path"] = str(final_meta_file) # Update placeholder

    log.info(f"Enriching metadata for {title} (GUID: {guid})")
    final_meta, updated_transcript_path = enrich_meta(
        raw_meta_path=None, # Not needed if raw_meta_dict is provided
        transcript_path=transcription_output_path,
        people_aliases_path="config/people_aliases.yml",
        known_hosts_path="config/known_hosts.yml",
        stopwords_list=stopwords, # Pass loaded stopwords
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
    segments_file = layout_fn("segments_path_template",
                              podcast_slug=podcast_slug,
                              episode_slug=None, # Not using episode_slug here as per new naming
                              guid=guid, # Use GUID for the filename
                              file_ext="json")
    segments_file.parent.mkdir(parents=True, exist_ok=True)

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
    kpi_output_path = layout_fn("kpis_path_template",
                                podcast_slug=podcast_slug,
                                episode_slug=episode_file_identifier, # Or use guid if kpi files are guid-named
                                guid=guid,
                                file_ext="json")
    kpi_output_path.parent.mkdir(parents=True, exist_ok=True)

    kpis = extract_kpis(
        meta_file_path=final_meta_file, 
        # transcript_file_path=updated_transcript_path # transcript content is usually in meta or linked from it
    )
    if kpis:
        kpi_output_path.write_text(json.dumps(kpis, indent=2))
        log.info(f"KPIs saved to: {kpi_output_path}")
        final_meta["episode_kpis_path"] = str(kpi_output_path) # Add to final_meta
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
        s3_upload_prefix = layout_fn("s3_prefix_template", podcast_slug=podcast_slug, guid=guid)
        
        # Upload final meta.json
        s3_meta_key = f"{s3_upload_prefix}/{final_meta_file.name}"
        log.info(f"Uploading final meta to S3: s3://{BUCKET}/{s3_meta_key}")
        S3.upload_file(str(final_meta_file), BUCKET, s3_meta_key)
        # verify_s3_upload(s3_meta_key, final_meta_file.stat().st_size, S3, BUCKET)

        # Upload final transcript.json (patched)
        if updated_transcript_path and updated_transcript_path.exists():
            s3_transcript_key = f"{s3_upload_prefix}/{updated_transcript_path.name}"
            log.info(f"Uploading final transcript to S3: s3://{BUCKET}/{s3_transcript_key}")
            S3.upload_file(str(updated_transcript_path), BUCKET, s3_transcript_key)
            # verify_s3_upload(s3_transcript_key, updated_transcript_path.stat().st_size, S3, BUCKET)
        
        # Upload segments.json
        if saved_segments_path and saved_segments_path.exists():
            s3_segments_key = f"{s3_upload_prefix}/{saved_segments_path.name}"
            log.info(f"Uploading segments to S3: s3://{BUCKET}/{s3_segments_key}")
            S3.upload_file(str(saved_segments_path), BUCKET, s3_segments_key)

        # Upload cleaned_entities.json (if path is in final_meta and file exists)
        cleaned_entities_s3_path = final_meta.get("cleaned_entities_path")
        if cleaned_entities_s3_path and Path(cleaned_entities_s3_path).exists():
            s3_entities_key = f"{s3_upload_prefix}/{Path(cleaned_entities_s3_path).name}"
            log.info(f"Uploading cleaned entities to S3: s3://{BUCKET}/{s3_entities_key}")
            S3.upload_file(cleaned_entities_s3_path, BUCKET, s3_entities_key)

        # Upload sentence_embeddings.npy (if path is in final_meta and file exists)
        sentence_embeddings_s3_path = final_meta.get("sentence_embeddings_path")
        if sentence_embeddings_s3_path and Path(sentence_embeddings_s3_path).exists():
            s3_embeddings_key = f"{s3_upload_prefix}/{Path(sentence_embeddings_s3_path).name}"
            log.info(f"Uploading sentence embeddings to S3: s3://{BUCKET}/{s3_embeddings_key}")
            S3.upload_file(sentence_embeddings_s3_path, BUCKET, s3_embeddings_key)

        # Upload kpis.json (if path is in final_meta and file exists)
        episode_kpis_s3_path = final_meta.get("episode_kpis_path")
        if episode_kpis_s3_path and Path(episode_kpis_s3_path).exists():
            s3_kpis_key = f"{s3_upload_prefix}/{Path(episode_kpis_s3_path).name}"
            log.info(f"Uploading KPIs to S3: s3://{BUCKET}/{s3_kpis_key}")
            S3.upload_file(episode_kpis_s3_path, BUCKET, s3_kpis_key)
        
        # Upload the original audio file (already downloaded)
        s3_audio_key = f"{s3_upload_prefix}/audio/{audio_file.name}" # Store audio in a subfolder
        log.info(f"Uploading audio to S3: s3://{BUCKET}/{s3_audio_key}")
        S3.upload_file(str(audio_file), BUCKET, s3_audio_key)
        # Update meta with S3 audio path if it differs from a simple prefix/name structure
        final_meta["s3_audio_path"] = f"s3://{BUCKET}/{s3_audio_key}"
        final_meta["s3_artifacts_prefix"] = f"s3://{BUCKET}/{s3_upload_prefix}"
        final_meta_file.write_text(json.dumps(final_meta, indent=2, ensure_ascii=False)) # Resave with S3 paths
        log.info(f"Re-saved final meta with S3 paths: {final_meta_file}")

    # ---- Mark as processed (locally) ----
    mark_processed(guid, audio_hash)
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
    """Processes a single feed URL, respecting limits."""
    processed_in_this_feed = 0
    try:
        items = feed_items(url)
        if not items:
            log.info(f"No suitable items found in feed: {url}")
            return 0
            
        # Sort items by published date (oldest first) to process chronologically
        # feed_items already returns items that are >= SINCE_DATE
        # If you need a specific order not guaranteed by feed, sort here.
        # items.sort(key=lambda x: x[1]) # x[1] is 'when' (datetime object)

        for entry, when, podcast_title, feed_url_val in items:
            if TERMINATE:
                log.info("Termination signal received, stopping feed processing.")
                break 
            if overall_limit and total_processed_count >= overall_limit:
                log.info(f"Overall processing limit ({overall_limit}) reached.")
                break
            if limit_per_feed is not None and processed_in_this_feed >= limit_per_feed:
                log.info(f"Per-feed limit ({limit_per_feed}) reached for {url}.")
                break

            log.info(f"Processing item: {podcast_title} - {entry.get('title')} (Published: {when})")
            try:
                if process(entry, when, podcast_title, feed_url_val):
                    processed_in_this_feed += 1
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
    return processed_in_this_feed

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

# ------------------------------------------------------------------------ main
def main() -> int:
    try:
        # Initialize DB (ensure tables exist etc.)
        # Consider making this conditional or configurable if DB is optional
        # init_db() 
        # log.info("Database initialized (if required).")

        # Make sure essential paths exist
        DOWNLOAD_PATH.mkdir(exist_ok=True, parents=True)
        # TRANSCRIPT_PATH.mkdir(exist_ok=True, parents=True) # meta_utils and transcribe.py handle their own paths now

        # Handle single episode processing first if argument is provided
        if args.episode_details_json:
            log.info(f"Processing single episode from JSON details: {args.episode_details_json}")
            if process_single_episode(args.episode_details_json):
                log.info("Single episode processed successfully.")
                return 0
            else:
                log.error("Single episode processing failed.")
                return 1

        # Proceed with feed processing if not a single episode run
        feeds_to_process = [args.feed] if args.feed else CFG.get("feeds", [])
        if not feeds_to_process:
            log.warning("No feeds specified either via --feed argument or in the config file. Nothing to do.")
            return 0

        total_processed_count = 0
        limit_per_feed = CFG.get("limit_per_feed") # Optional: limit per feed in config

        for url in feeds_to_process:
            if TERMINATE:
                log.info("Termination signal received, stopping main loop.")
                break
            if args.limit and total_processed_count >= args.limit:
                log.info(f"Overall processing limit ({args.limit}) reached before processing all feeds.")
                break
            
            log.info(f"Starting processing for feed: {url}")
            # Pass the overall limit to the feed processing function to check within its loop
            # And update total_processed_count based on how many were done for this feed.
            processed_for_this_feed = process_feed_url(url, limit_per_feed, total_processed_count, args.limit)
            total_processed_count += processed_for_this_feed
            log.info(f"Processed {processed_for_this_feed} episodes from feed: {url}. Total processed so far: {total_processed_count}")

        log.info(f"All feeds processed. Total episodes processed in this run: {total_processed_count}")
        return 0

    except Exception:
        log.critical("An unhandled exception occurred in main.", exc_info=True)
        return 1
    finally:
        log.info("Backfill script finished.")
        # Cleanup logic if any (e.g. closing DB connections if opened in main)

if __name__ == "__main__":
    # Consider initializing DB connection here if it's to be shared globally and needs explicit setup/teardown
    # For example: 
    # if not DRY_RUN: # Assuming DB operations are skipped in dry run
    #    init_db()
    #    log.info("Database initialized for the run.")

    exit_code = main()
    sys.exit(exit_code) 