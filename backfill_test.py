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
CFG = yaml.safe_load(Path("test-five.yaml").read_text())   # backfill_test.py uses test-five.yaml
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

    # When entry is a dict (from --episode_details_json), use dict access
    enclosures = entry.get("enclosures")
    if not enclosures:
        log.warning("no enclosure in entry")
        return False
    audio_url = enclosures[0].get("href")
    if not audio_url:
        log.warning("no href in enclosure")
        return False
        
    # slug = re.sub(r"\\W+", "_", podcast.lower()).strip("_") # old slug for fname
    # fname = f"{slug}_{md5_8(guid)}.mp3" # old fname

    # Create podcast-specific directory for audio
    audio_podcast_dir = DOWNLOAD_PATH / podcast_slug
    audio_podcast_dir.mkdir(parents=True, exist_ok=True)
    mp3 = audio_podcast_dir / audio_mp3_fname # Use new fname and path

    # Ensure parent directories for these top-level types still exist
    DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True) # e.g. data/audio
    TRANSCRIPT_PATH.mkdir(parents=True, exist_ok=True) # e.g. data/transcripts

    if DRY_RUN:
        print(f"[DRY] {podcast} | {title} | {audio_url}")
        return True

    if not download_with_retry(audio_url, mp3):
        return False
    if not verify_audio(mp3):
        mp3.unlink(missing_ok=True)
        return False

    # Get audio tech info
    tech = get_audio_tech(mp3)
    tech["speech_music_ratio"] = estimate_speech_music_ratio(mp3)

    audio_hash = calculate_audio_hash(mp3)
    if audio_hash in processed_hashes:
        mp3.unlink(missing_ok=True)
        return True

    # Parse feed for metadata
    feed = feedparser.parse(feed_url)

    # Initialize base metadata
    is_dict_entry = isinstance(entry, dict)

    # Use .get() for dictionaries, attribute access for feedparser objects
    meta = {
        # Core IDs & URLs
        "podcast": podcast,
        "episode": title,
        "guid": entry.get("id") if is_dict_entry else getattr(entry, "id", None),
        "published": entry.get("published", ""),
        "episode_url": entry.get("link"),
        "audio_url": audio_url,
        
        # Processing info
        "asr_model": MODEL_VERSION,
        "processed_date": dt.datetime.utcnow().isoformat(),
        "processed_by": "backfill_test.py",
        
        # Audio tech info
        **tech,
        
        # Episode metadata
        "itunes_episodeType": entry.get("itunes_episode_type"),
        "categories": [t["term"] for t in entry.get("tags", [])] if entry.get("tags") else [],
        "rights": {
            "copyright": entry.get("copyright"),
            "explicit": entry.get("itunes_explicit")
        }
    }

    # Get transcript text for meta enrichment
    transcript_text = ""
    try:
        with tempfile.TemporaryDirectory() as td:
            chunks = chunk_long_audio(mp3, Path(td))
            outs: list[Path] = []
            # Create podcast-specific directory for transcripts
            transcript_podcast_dir = TRANSCRIPT_PATH / podcast_slug
            transcript_podcast_dir.mkdir(parents=True, exist_ok=True)

            for i, ch in enumerate(chunks):
                # Use episode_file_identifier for transcript part names
                transcript_part_fname = f"{episode_file_identifier}.mp3.part{i}.json"
                out_json = transcript_podcast_dir / transcript_part_fname # New path
                run_transcribe(ch, out_json,
                               meta | {"chunk_index": i, "total_chunks": len(chunks)},
                               podcast_slug=podcast_slug) # MODIFIED: Pass podcast_slug
                outs.append(out_json)

            final = outs[0] # This is now, e.g., data/transcripts/podcast-slug/ep_id.mp3.part0.json
            if len(outs) > 1:
                combo = {"segments": [], "meta": meta}
                for p in outs:
                    combo["segments"].extend(json.loads(p.read_text())["segments"])
                final.write_text(json.dumps(combo, ensure_ascii=False))
                transcript_text = " ".join(seg.get("text", "") for seg in combo["segments"] if "text" in seg)
                transcript_data = combo
            else:
                transcript_data = json.loads(final.read_text())
                transcript_text = " ".join(seg.get("text", "") for seg in transcript_data.get("segments", []) if "text" in seg)

            # Enrich metadata with transcript and feed data
            # Ensure feed is parsed before calling enrich_meta
            feed_parsed_data = feedparser.parse(feed_url) # Moved feed parsing here to ensure it's available
            enriched_meta_obj = enrich_meta(
                entry=entry, 
                feed_details={
                    "title": podcast,
                    "url": feed_url,
                    "generator": getattr(feed.feed, 'generator', None),
                    "language": getattr(feed.feed, 'language', None),
                    "itunes_explicit": getattr(feed.feed, 'itunes_explicit', None),
                    "itunes_author": getattr(feed.feed, 'itunes_author', None)
                },
                tech=tech, 
                transcript_text=transcript_text, 
                transcript_segments=transcript_data.get("segments"),
                nlp_model=None,
                base_data_dir=TRANSCRIPT_PATH.parent,
                perform_caching=True,
                podcast_slug=podcast_slug,
                word_timestamps_enabled=check_timestamp_support(transcript_data)
            )
            
            # Process transcript and add timestamp info
            enriched_meta_obj = process_transcript(transcript_data, enriched_meta_obj)
            
            # --- Define canonical transcript path EARLIER ---
            final_transcript_filename = f"{guid}.json" # Example: {guid}.json
            canonical_transcript_path = transcript_podcast_dir / final_transcript_filename
            log.info(f"Defined canonical_transcript_path: {canonical_transcript_path}")

            # --- Save segments file using the new utility function ---
            # Ensure transcript_data["segments"] exists and is a list
            segments_to_save = transcript_data.get("segments")
            if isinstance(segments_to_save, list) and guid and podcast_slug:
                # Use the correct base_data_dir, which is TRANSCRIPT_PATH.parent
                # This corresponds to the "data/" directory if TRANSCRIPT_PATH is "data/transcripts"
                base_data_dir_for_segments = TRANSCRIPT_PATH.parent
                
                # The save_segments_file function expects the actual GUID, not audio_hash
                # The 'guid' variable should hold the correct episode GUID at this point
                saved_segments_path = save_segments_file(
                    segments=segments_to_save,
                    podcast_slug=podcast_slug,
                    guid=guid, # Uses the episode GUID
                    base_data_dir=str(base_data_dir_for_segments)
                )
                if saved_segments_path:
                    enriched_meta_obj["segments_path"] = saved_segments_path
                    log.info(f"Segments file saved via utility function to: {saved_segments_path} for GUID {guid}")
                else:
                    log.warning(f"Failed to save segments file via utility function for GUID {guid}.")
            else:
                log.warning(f"Could not save segments for GUID {guid}: Segments data missing, or guid/podcast_slug missing.")
            # --- End save segments file ---
            
            # Extract entities_path and embedding_path from the transcript data (if transcribe.py cached them)
            entities_path_from_transcript_json = transcript_data.get("meta", {}).get("entities_path")
            embedding_path_from_transcript_json = transcript_data.get("meta", {}).get("embedding_path")

            if entities_path_from_transcript_json:
                enriched_meta_obj["entities_path"] = entities_path_from_transcript_json
            if embedding_path_from_transcript_json:
                enriched_meta_obj["embedding_path"] = embedding_path_from_transcript_json
            
            # Add any additional metadata, now using the defined canonical_transcript_path
            enriched_meta_obj.update({
                "segment_count": len(transcript_data.get("segments", [])),
                "chunk_count": len(chunks) if len(chunks) > 1 else 1,
                "audio_hash": audio_hash,
                "download_path": str(mp3.resolve()), 
                "transcript_path": str(canonical_transcript_path.resolve()) # Use defined canonical path
            })

            # Update the transcript_data's internal 'meta' block with the fully enriched metadata
            # This is crucial BEFORE writing transcript_data to the canonical_transcript_path
            transcript_data["meta"] = enriched_meta_obj
            
            # Ensure complete paths are preserved in JSON output
            class PathEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Path):
                        return str(obj.absolute())
                    return super().default(obj)
            
            # Write transcript_data to this new canonical path
            with open(canonical_transcript_path, "w", encoding='utf-8') as f:
                json.dump(transcript_data, f, cls=PathEncoder, ensure_ascii=False, indent=2)
            log.info(f"Transcript data saved to canonical path: {canonical_transcript_path}") 
            
            # If outs[0] (original final path) was different and chunking happened, or single part renamed, 
            # remove the old transcript file pointed to by 'final'.
            if final != canonical_transcript_path:
                log.info(f"Removing old transcript file: {final}")
                final.unlink(missing_ok=True) 

            # --- Update main tech dict with ALL paths and critical info from transcript meta / first enrich pass ---
            # This ensures the final enrich_meta call gets these.
            
            # Paths from transcribe.py's internal caching (these paths include podcast_slug)
            if entities_path_from_transcript_json:
                tech["entities_path"] = entities_path_from_transcript_json
                log.info(f"Updated tech dict with entities_path: {tech['entities_path']}")
            if embedding_path_from_transcript_json:
                tech["embedding_path"] = embedding_path_from_transcript_json
                log.info(f"Updated tech dict with embedding_path: {tech['embedding_path']}")

            # Other critical fields from enriched_meta_obj (which is transcript_data["meta"] at this point)
            if enriched_meta_obj and isinstance(enriched_meta_obj, dict):
                if "transcript_length" in enriched_meta_obj:
                    tech["transcript_length"] = enriched_meta_obj["transcript_length"]
                    log.info(f"Updated tech with transcript_length: {tech['transcript_length']}")
                
                if "avg_confidence" in enriched_meta_obj: # From process_transcript
                    tech["confidence"] = { 
                        "avg_confidence": enriched_meta_obj["avg_confidence"],
                        "wer_estimate": enriched_meta_obj.get("wer_estimate", 1.0 - enriched_meta_obj["avg_confidence"])
                    }
                    log.info(f"Updated tech with confidence: {tech['confidence']}")

                if 'keywords' in enriched_meta_obj: # From process_transcript
                    tech['keywords'] = enriched_meta_obj['keywords']
                    log.info(f"Updated tech with keywords, count: {len(tech['keywords']) if isinstance(tech.get('keywords'), list) else 0}")

                if 'supports_timestamp' in enriched_meta_obj: # From process_transcript
                    tech['supports_timestamp'] = enriched_meta_obj['supports_timestamp']
                    log.info(f"Updated tech with supports_timestamp: {tech['supports_timestamp']}")
                
                if 'segment_count' in enriched_meta_obj: # From the .update() call
                    tech['segment_count'] = enriched_meta_obj['segment_count']
                    log.info(f"Updated tech with segment_count: {tech['segment_count']}")
                
                if 'chunk_count' in enriched_meta_obj: # From the .update() call
                    tech['chunk_count'] = enriched_meta_obj['chunk_count']
                    log.info(f"Updated tech with chunk_count: {tech['chunk_count']}")
                
                # audio_hash and download_path might have been updated in enriched_meta_obj.
                # Also, transcript_path was set in enriched_meta_obj.update()
                if 'audio_hash' in enriched_meta_obj:
                     tech['audio_hash'] = enriched_meta_obj['audio_hash']
                     log.info(f"Updated tech with audio_hash from enriched_meta_obj: {tech['audio_hash']}")
                if 'download_path' in enriched_meta_obj:
                     tech['download_path'] = enriched_meta_obj['download_path']
                     log.info(f"Updated tech with download_path from enriched_meta_obj: {tech['download_path']}")
                if 'transcript_path' in enriched_meta_obj:
                     tech['transcript_path'] = enriched_meta_obj['transcript_path']
                     log.info(f"Updated tech with transcript_path from enriched_meta_obj: {tech['transcript_path']}")

            # Final safety net: ensure primary paths are in tech if somehow missed from enriched_meta_obj
            # (though they should be there from earlier steps like get_audio_tech or enriched_meta_obj.update())
            if mp3 and mp3.exists() and not tech.get('download_path'):
                 tech['download_path'] = str(mp3.resolve())
                 log.info(f"Ensured tech has download_path (safety net): {tech['download_path']}")
            # Use canonical_transcript_path for the safety net check for transcript_path
            if canonical_transcript_path and canonical_transcript_path.exists() and not tech.get('transcript_path'):
                tech['transcript_path'] = str(canonical_transcript_path.resolve())
                log.info(f"Ensured tech has transcript_path (safety net using canonical): {tech['transcript_path']}")

    except Exception as e:
        log.error("transcribe error", exc_info=e)
        return False

    if USE_AWS:
        # Generate S3 prefix using the new layout function
        s3_prefix_for_transcript_upload = layout_fn(guid, podcast_slug) # MODIFIED: added podcast_slug
        key_transcript_json = f"{s3_prefix_for_transcript_upload}{canonical_transcript_path.name}"
        try:
            S3.upload_file(str(canonical_transcript_path), BUCKET, key_transcript_json)
            verify_s3_upload(S3, BUCKET, key_transcript_json, canonical_transcript_path)
            # The transcript_s3_path will be added to the *final* meta object later.
        except Exception as e:
            log.error(f"Failed to upload transcript JSON {canonical_transcript_path.name} to S3: {e}")

    processed_guids.add(guid)
    processed_hashes.add(audio_hash)

    # Enrich metadata with transcript info and entities
    # Construct feed_details dictionary for the new enrich_meta signature
    feed_details_for_enrich = {
        "title": podcast, # This was feed_title (original podcast title string)
        "url": feed_url,  # The feed URL string
        "generator": getattr(feed.feed, 'generator', None),
        "language": getattr(feed.feed, 'language', None),
        "itunes_explicit": getattr(feed.feed, 'itunes_explicit', None),
        "itunes_author": getattr(feed.feed, 'itunes_author', None),
        # 'rights' could also be sourced from feed.feed.get('rights') if needed by enrich_meta via feed_details
    }
    
    # The 'feed' object itself is no longer passed directly.
    # Its relevant parts are now in feed_details_for_enrich.
    # transcript_segments should be passed from transcript_data.
    # word_timestamps_enabled can be determined from check_timestamp_support(transcript_data)
    
    meta = enrich_meta(
        entry=entry, 
        feed_details=feed_details_for_enrich,
        tech=tech, 
        transcript_text=transcript_text,
        transcript_segments=transcript_data.get("segments", []), # Pass the actual segments
        perform_caching=True, # Assuming True, adjust if needed
        nlp_model=None,       # Assuming None for this call, adjust if an NLP model instance should be passed
        base_data_dir=TRANSCRIPT_PATH.parent, # e.g. "data/"
        podcast_slug=podcast_slug,
        word_timestamps_enabled=check_timestamp_support(transcript_data) # Check from actual data
    )

    # --- Extract and Save KPIs ---
    kpis_list = []
    kpis_file_path_obj = None
    if transcript_text: # Ensure there is text to process
        kpis_list = extract_kpis(transcript_text)
        if kpis_list:
            kpis_base_dir = TRANSCRIPT_PATH.parent / "kpis" # data/kpis
            kpis_podcast_dir = kpis_base_dir / podcast_slug
            kpis_podcast_dir.mkdir(parents=True, exist_ok=True)

            kpis_file_name = f"{guid}_kpis.json"
            kpis_file_path_obj = kpis_podcast_dir / kpis_file_name # New path
            with open(kpis_file_path_obj, "w", encoding='utf-8') as kf:
                json.dump(kpis_list, kf, ensure_ascii=False, indent=2)
            log.info(f"Saved {len(kpis_list)} KPIs to {kpis_file_path_obj}")
            meta["kpis_path"] = str(kpis_file_path_obj.resolve())
        else:
            log.info(f"No KPIs extracted for GUID {guid}.")
            meta["kpis_path"] = None
    else:
        log.info(f"Transcript text is empty for GUID {guid}. Skipping KPI extraction.")
        meta["kpis_path"] = None

    # --- Save final combined metadata to JSON (locally) ---
    # Use a descriptive local name, but a canonical name on S3.
    
    # Generate S3 prefix using the new layout function BEFORE S3 paths are constructed for meta
    s3_prefix = layout_fn(guid, podcast_slug) # MODIFIED: added podcast_slug. Ensure guid and podcast_slug are valid
    meta["s3_prefix"] = s3_prefix # Store for transparency/DB

    # --- Define S3 paths for all artifacts and add to meta ---
    # Canonical S3 path for the metadata file itself
    s3_meta_key = f"{s3_prefix}meta.json"
    meta["meta_s3_path"] = f"s3://{BUCKET}/{s3_meta_key}"

    # S3 path for transcript (final is the local transcript json Path object)
    if canonical_transcript_path and canonical_transcript_path.exists():
        s3_transcript_key = f"{s3_prefix}{canonical_transcript_path.name}"
        meta["transcript_s3_path"] = f"s3://{BUCKET}/{s3_transcript_key}"
    else:
        # If canonical_transcript_path doesn't exist, ensure path is None or handled
        meta["transcript_s3_path"] = meta.get("transcript_s3_path") 

    # S3 path for audio (mp3 is the local audio file Path object)
    if mp3 and mp3.exists():
        s3_audio_key = f"{s3_prefix}{mp3.name}"
        meta["audio_s3_path"] = f"s3://{BUCKET}/{s3_audio_key}"
        # audio_expected_size_bytes & audio_hash are already in meta
    else:
        meta["audio_s3_path"] = meta.get("audio_s3_path")

    # S3 path for cleaned entities
    if meta.get("cleaned_entities_path"):
        cleaned_entities_local_path = Path(meta["cleaned_entities_path"])
        if cleaned_entities_local_path.exists():
            s3_cleaned_entities_key = f"{s3_prefix}{cleaned_entities_local_path.name}"
            meta["cleaned_entities_s3_path"] = f"s3://{BUCKET}/{s3_cleaned_entities_key}"
        else:
            meta["cleaned_entities_s3_path"] = None # Or keep existing if any logic sets it earlier
    else:
         meta["cleaned_entities_s3_path"] = None


    # S3 path for KPIs
    if meta.get("kpis_path"):
        kpis_local_path = Path(meta["kpis_path"])
        if kpis_local_path.exists():
            s3_kpis_key = f"{s3_prefix}{kpis_local_path.name}"
            meta["kpis_s3_path"] = f"s3://{BUCKET}/{s3_kpis_key}" # Add to meta
        else:
            meta["kpis_s3_path"] = None
    else:
        meta["kpis_s3_path"] = None


    # Add expected sizes for JSON artifacts before final save
    if canonical_transcript_path and canonical_transcript_path.exists(): # Use canonical_transcript_path
        meta["transcript_expected_size_bytes"] = canonical_transcript_path.stat().st_size
    
    if meta.get("cleaned_entities_path"):
        cleaned_entities_file = Path(meta["cleaned_entities_path"])
        if cleaned_entities_file.exists():
            meta["cleaned_entities_expected_size_bytes"] = cleaned_entities_file.stat().st_size
            
    if meta.get("kpis_path"):
        kpis_file = Path(meta["kpis_path"])
        if kpis_file.exists():
            meta["kpis_expected_size_bytes"] = kpis_file.stat().st_size

    # Use the now ISO-formatted 'published' date for slug generation
    raw_episode_slug_part = make_slug(
        podcast_name_or_slug="", 
        episode_title=meta.get("episode", "unknown_episode"),
        # meta.get("published") should now be the ISO string from published_iso
        published_iso_date=meta.get("published", "nodate") 
    )

    # The episode slug part is no longer needed for the local meta filename itself
    # if len(raw_episode_slug_part) > 40:
    #     # Truncate to near 40 chars, then rsplit to avoid cutting mid-word
    #     short_episode_slug_for_file = raw_episode_slug_part[:40].rsplit('-', 1)[0]
    # else:
    #     short_episode_slug_for_file = raw_episode_slug_part
    # # Ensure it doesn't end with a hyphen if rsplit resulted in that or original was short
    # short_episode_slug_for_file = short_episode_slug_for_file.strip('-')

    # Correct local_meta_filename to {guid}.json
    local_meta_filename = f"{guid}.json"
    meta_base_dir = Path(TRANSCRIPT_PATH.parent) / "meta" # data/meta
    meta_podcast_dir = meta_base_dir / podcast_slug
    meta_podcast_dir.mkdir(parents=True, exist_ok=True)
    local_meta_file_path = meta_podcast_dir / local_meta_filename # New path

    # Save the local meta file with all S3 paths now embedded
    with open(local_meta_file_path, "w", encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info(f"SUCCESS: Final combined metadata saved locally to {local_meta_file_path} for GUID {guid}")
    # Clean up old meta files for this GUID, ensuring keep_path is correct
    # The cleanup_old_meta function expects a glob pattern that includes "meta_" prefix.
    # We need to adjust either cleanup_old_meta or how we call it if the primary file no longer has "meta_".
    # For now, let's assume cleanup_old_meta needs to be more flexible or the saved file needs a prefix for cleanup to work.
    # Given the request for {guid}.json, cleanup_old_meta might need adjustment or a different strategy.
    # Let's simplify the cleanup call assuming we only want to keep THE {guid}.json file.
    # The existing cleanup_old_meta is: glob.glob(f"{meta_dir}/meta_{guid}_*.json")
    # This will no longer match if the file is just {guid}.json.
    # Option 1: Modify cleanup_old_meta to handle various patterns (more complex).
    # Option 2: Keep a consistent prefix for all meta files that might be generated, even the final one, if cleanup relies on it.
    # Option 3: Change cleanup to find all *.json for the guid and keep the newest.
    # Let's adjust the saved filename to include "meta_" for now to keep cleanup_old_meta working as is,
    # OR adjust cleanup_old_meta.
    # The user asked for <guid>.json for the final meta. Let's try to achieve that and adjust cleanup.

    # The filename for the final, canonical meta file is now {guid}.json
    # The cleanup function `cleanup_old_meta` uses glob pattern `meta_{guid}_*.json`
    # This means it won't see the new `{guid}.json` as a file to potentially keep OR remove if it's the newest.
    # It will try to remove ALL `meta_{guid}_*.json` files.
    # This is problematic if an old `meta_{guid}_slug.json` exists and the new canonical is just `{guid}.json`
    #
    # Let's adjust `cleanup_old_meta` to be more robust.
    # For now, the `local_meta_file_path` is `.../{guid}.json`.
    # The `cleanup_old_meta` will be called with this path as `keep_path`.
    # It will glob for `meta_{guid}_*.json`.
    # If `local_meta_file_path` is indeed just `{guid}.json`, then `os.path.abspath(f) != os.path.abspath(keep_path)` will always be true for the globbed files.
    # This means all old `meta_{guid}_*.json` files *should* be removed, and the new `{guid}.json` is safe. This seems acceptable.

    cleanup_old_meta(str(meta_podcast_dir), guid, str(local_meta_file_path))

    # --- S3 Upload Steps ---
    # Upload the metadata file itself
    if USE_AWS and BUCKET and s3_meta_key: # s3_meta_key defined above
        try:
            S3.upload_file(
                Filename=str(local_meta_file_path),
                Bucket=BUCKET,
                Key=s3_meta_key
            )
            log.info(f"Uploaded metadata for {guid} to s3://{BUCKET}/{s3_meta_key}")
        except Exception as e:
            log.error(f"Failed to upload metadata for {guid} to S3: {e}")

    # Upload other artifacts (audio, transcript, entities, kpis)
    # Transcript (final) was uploaded earlier in the script, path already in meta.
    # Audio (mp3)
    if USE_AWS and BUCKET and meta.get("audio_s3_path") and mp3 and mp3.exists():
        s3_audio_key_for_upload = meta["audio_s3_path"].split(f"s3://{BUCKET}/")[1]
        try:
            S3.upload_file(Filename=str(mp3), Bucket=BUCKET, Key=s3_audio_key_for_upload)
            log.info(f"Uploaded audio for {guid} to {meta['audio_s3_path']}")
        except Exception as e:
            log.error(f"Failed to upload audio for {guid} to S3: {e}")

    # Cleaned Entities
    if USE_AWS and BUCKET and meta.get("cleaned_entities_s3_path") and meta.get("cleaned_entities_path"):
        cleaned_entities_local_file = Path(meta["cleaned_entities_path"])
        if cleaned_entities_local_file.exists():
            s3_cleaned_entities_key_for_upload = meta["cleaned_entities_s3_path"].split(f"s3://{BUCKET}/")[1]
            try:
                S3.upload_file(Filename=str(cleaned_entities_local_file), Bucket=BUCKET, Key=s3_cleaned_entities_key_for_upload)
                log.info(f"Uploaded cleaned entities for {guid} to {meta['cleaned_entities_s3_path']}")
            except Exception as e:
                log.error(f"Failed to upload cleaned entities for {guid} to S3: {e}")
    
    # KPIs
    if USE_AWS and BUCKET and meta.get("kpis_s3_path") and meta.get("kpis_path"):
        kpis_local_file = Path(meta["kpis_path"])
        if kpis_local_file.exists():
            s3_kpis_key_for_upload = meta["kpis_s3_path"].split(f"s3://{BUCKET}/")[1]
            try:
                S3.upload_file(Filename=str(kpis_local_file), Bucket=BUCKET, Key=s3_kpis_key_for_upload)
                log.info(f"Uploaded KPIs for {guid} to {meta['kpis_s3_path']}")
            except Exception as e:
                log.error(f"Failed to upload KPIs for {guid} to S3: {e}")


    # --- Upsert episode to database ---
    # Ensure published_date is in YYYY-MM-DD format for the DB
    published_date_db_format = ""
    # meta.get("published") should now hold the ISO string from published_iso
    current_published_date_str = meta.get("published")

    if current_published_date_str:
        try:
            # Attempt to parse assuming it's an ISO format (potentially with Z or offset)
            # Remove Z and replace with +00:00 for fromisoformat compatibility if Z is present
            if current_published_date_str.endswith('Z'):
                current_published_date_str = current_published_date_str[:-1] + '+00:00'
            
            date_obj = dt.datetime.fromisoformat(current_published_date_str)
            published_date_db_format = date_obj.strftime("%Y-%m-%d")
        except ValueError as ve_iso:
            # Fallback for dates that might not be full ISO but YYYY-MM-DD (though unlikely if from published_iso)
            try:
                dt.datetime.strptime(current_published_date_str, "%Y-%m-%d") # validates format
                published_date_db_format = current_published_date_str
            except ValueError as ve_simple:
                log.warning(f"Could not parse published date '{current_published_date_str}' to YYYY-MM-DD for DB. ISO attempt error: {ve_iso}. Simple date attempt error: {ve_simple}. Skipping DB upsert for this field or using placeholder.")
                # Log specific error for DB handling
                log.error(f"Published date '{current_published_date_str}' is invalid, cannot upsert to DB due to NOT NULL constraint. Skipping upsert for GUID {guid}.")
    else:
        log.error(f"Published date is missing in meta for GUID {guid}. Skipping DB upsert for this field.")
        log.error(f"Published date is missing, cannot upsert to DB due to NOT NULL constraint. Skipping upsert for GUID {guid}.")
        
    # Check if published_date_db_format was successfully set
    if not published_date_db_format:
        log.error(f"Critical field published_date_db_format is missing for GUID {guid}. Skipping DB upsert.")
    else:
        podcast_s = generate_podcast_slug(meta.get("podcast", "Unknown Podcast"))
        # Use the same short_episode_slug for consistency if desired for the DB slug field
        # or generate a full one using make_slug with podcast_s.
        # The DB `slug` column is for the full episode slug, not just the short file part.
        full_episode_slug = make_slug(
            podcast_s, 
            meta.get("episode", "Unknown Episode"), 
            published_date_db_format
        )

        # Construct asr_engine string (example)
        # MODEL_VERSION should be like 'base', 'small', etc. COMPUTE_TYPE like 'int8'
        asr_engine_str = f"whisperx|{MODEL_VERSION}|{COMPUTE_TYPE}" 

        db_row = {
            "guid": guid,
            "podcast_slug": podcast_s,
            "podcast_title": meta.get("podcast"),
            "episode_title": meta.get("episode"),
            "published_date": published_date_db_format, 
            "slug": full_episode_slug, # Full episode slug for DB
            "s3_prefix": meta.get("s3_prefix"), # From layout_fn, stored in meta
            "meta_s3_path": meta.get("meta_s3_path"), 
            "transcript_s3_path": meta.get("transcript_s3_path"),
            "cleaned_entities_s3_path": meta.get("cleaned_entities_s3_path"),
            # Add kpis_s3_path to db_row if it's a field in your 'episodes' table
            # "kpis_s3_path": meta.get("kpis_s3_path"), 
            "duration_sec": int(meta.get("duration_sec", 0)),
            "asr_engine": asr_engine_str,
            "local_audio_path": str(mp3.resolve()) if mp3 and mp3.exists() else None,
            "local_transcript_path": str(canonical_transcript_path.resolve()) if canonical_transcript_path and canonical_transcript_path.exists() else None,
            "local_entities_path": meta.get("entities_path"),
            "local_cleaned_entities_path": meta.get("cleaned_entities_path"),
            # Path to the descriptively named local meta file is now just {guid}.json
            "meta_path_local": str(local_meta_file_path.resolve()) 
        }
        upsert_episode(db_row)

    return True


def main() -> int:
    log.info(f"since={SINCE_DATE:%Y-%m-%d} dry={DRY_RUN} aws={USE_AWS}")
    
    # ---- Single episode processing mode ----
    if args.episode_details_json:
        ep = json.loads(args.episode_details_json)
        log.info(f"Processing single episode via --episode_details_json: {ep.get('guid')}")
        # ---- build the minimal mock objects the existing process() expects ----

        # published_parsed is not strictly needed for mock_entry if raw 'published' string is available
        # entry_dt function will handle parsing the raw 'published' string.
        mock_entry = {
            "id": ep.get("guid"), # process() uses entry.get("id", "") for guid
            "guid": ep.get("guid"), # also make it available directly
            "title": ep.get("episode_title"),
            "link": ep.get("episode_url"),
            "published": ep.get("published"), # entry_dt will use this
            "published_parsed": None, # Explicitly None, entry_dt will ignore this and parse the raw 'published' string
            "enclosures":  [{ "href": ep.get("audio_url") }] if ep.get("audio_url") else [], # process() expects "href"
            "itunes_explicit": ep.get("itunes_explicit"),
            "summary": ep.get("summary"),
            "tags": [{ "term": t } for t in ep.get("tags", [])],
            "itunes_author": ep.get("itunes_author"),
            # Add any other fields that enrich_meta or process might expect from entry
        }
        
        # entry_dt expects a dict-like object with 'published_parsed' or date strings
        when = entry_dt(mock_entry)
        if not when:
            log.error(f"Could not determine 'when' for episode {ep.get('guid')}. Exiting.")
            return 1 # Indicate error
            
        success = process(mock_entry, when,
                podcast=ep.get("podcast_title", "Unknown Podcast"), # Ensure a default for podcast title
                feed_url=ep.get("feed_url", "")) # Ensure a default for feed_url
        return 0 if success else 1
    # ---- End single episode processing mode ----

    # Clear processed sets at start of run (only for multi-episode mode)
    processed_guids.clear()
    processed_hashes.clear()
    
    feeds = [args.feed] if args.feed else CFG["feeds"]

    items: list[Tuple[Any, dt.datetime, str, str]] = []
    for f in feeds:
        items.extend(feed_items(f))
    items.sort(key=lambda t: t[1], reverse=True)
    if args.limit:
        items = items[: args.limit]
    log.info(f"Total items → {len(items)}")
    for item in items:
        log.info(f"Processing: {item[0].get('title')} from {item[1]}")

    ok = bad = 0
    for ent, when, pod, url in items:
        if TERMINATE:
            break
        (ok := ok + 1) if process(ent, when, pod, url) else (bad := bad + 1)
        time.sleep(0.5)

    log.info(f"Done ✓{ok} ✗{bad}")
    return 0


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Determine if running in single episode (worker) mode
    is_worker_process = bool(args.episode_details_json)
    db_disabled_by_env = os.getenv("DISABLE_DB_OPERATIONS", "false").lower() == "true"

    if db_disabled_by_env:
        log.info("Database operations explicitly disabled by DISABLE_DB_OPERATIONS environment variable.")
    elif is_worker_process:
        # This case is for parallel worker processes
        log.info("Skipping DB initialization in single-episode (worker) mode.")
    else:
        # This is a standalone run (not a parallel worker) and DB is not disabled by env var
        log.info("Attempting database initialization for standalone run...")
        init_db() # init_db() is idempotent and also checks DISABLE_DB_OPERATIONS
            
    sys.exit(main())
