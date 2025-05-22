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
)
from podcast_insights.nlp_utils import load_nlp_models
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
def run_transcribe(chunk: Path, out_json: Path, meta: dict) -> None:
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
        meta["asr_model"].split("-")[1],  # tiny/base/…
        "--metadata_json", # Changed from --metadata
        json.dumps(meta),
        "--enable_caching", # Added for new functionality
        "--base_data_dir", # Added for new functionality
        str(base_data_dir_for_caching) # Added for new functionality
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

    title = entry.get("title", "")
    if not entry.get("enclosures"):
        log.warning("no enclosure")
        return False
    audio_url = entry.enclosures[0]["href"]
    slug = re.sub(r"\W+", "_", podcast.lower()).strip("_")
    fname = f"{slug}_{md5_8(guid)}.mp3"
    mp3 = DOWNLOAD_PATH / fname
    DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_PATH.mkdir(parents=True, exist_ok=True)

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
    meta = {
        # Core IDs & URLs
        "podcast": podcast,
        "episode": title,
        "guid": guid,
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
            for i, ch in enumerate(chunks):
                out_json = TRANSCRIPT_PATH / f"{fname}.part{i}.json"
                run_transcribe(ch, out_json,
                               meta | {"chunk_index": i, "total_chunks": len(chunks)})
                outs.append(out_json)

            final = outs[0]
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
            enriched_meta_obj = enrich_meta(entry, podcast, feed_url, tech, transcript_text, feed_parsed_data, nlp_model=None, st_model=None, base_data_dir=TRANSCRIPT_PATH.parent, perform_caching=True) # Added nlp/st model placeholders, base_data_dir, and perform_caching

            # Process transcript and add timestamp info
            enriched_meta_obj = process_transcript(transcript_data, enriched_meta_obj)
            
            # --- Save segment timestamps ---
            segments_data_dir = TRANSCRIPT_PATH.parent / "segments"
            segments_data_dir.mkdir(parents=True, exist_ok=True)
            segments_file_path = segments_data_dir / f"{audio_hash}.json"
            
            segment_timestamps = []
            if "segments" in transcript_data and isinstance(transcript_data["segments"], list):
                segment_timestamps = [(s.get("start"), s.get("end")) for s in transcript_data["segments"] if isinstance(s, dict)]
            
            with open(segments_file_path, "w", encoding='utf-8') as sf:
                json.dump(segment_timestamps, sf, ensure_ascii=False, indent=2)
            enriched_meta_obj["segments_path"] = str(segments_file_path.resolve())
            # --- End save segment timestamps ---

            # Extract entities_path and embedding_path from the transcript data (if transcribe.py cached them)
            entities_path_from_transcript_json = transcript_data.get("meta", {}).get("entities_path")
            embedding_path_from_transcript_json = transcript_data.get("meta", {}).get("embedding_path")

            if entities_path_from_transcript_json:
                enriched_meta_obj["entities_path"] = entities_path_from_transcript_json
            if embedding_path_from_transcript_json:
                enriched_meta_obj["embedding_path"] = embedding_path_from_transcript_json
            
            # Add any additional metadata
            enriched_meta_obj.update({
                "segment_count": len(transcript_data.get("segments", [])),
                "chunk_count": len(chunks) if len(chunks) > 1 else 1,
                "audio_hash": audio_hash,
                "download_path": str(mp3),
                "transcript_path": str(final.absolute())
            })

            # Update the final JSON with enriched metadata
            transcript_data["meta"] = enriched_meta_obj
            
            # Ensure complete paths are preserved in JSON output
            class PathEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Path):
                        return str(obj.absolute())
                    return super().default(obj)
            
            # Write JSON with custom encoder to preserve paths
            final.write_text(json.dumps(transcript_data, cls=PathEncoder, ensure_ascii=False, indent=2))

    except Exception as e:
        log.error("transcribe error", exc_info=e)
        return False

    if USE_AWS:
        # Generate S3 prefix using the new layout function
        s3_prefix = layout_fn(guid) # guid is available here
        key = f"{s3_prefix}{final.name}" # final is the transcript JSON file
        S3.upload_file(str(final), BUCKET, key)
        verify_s3_upload(S3, BUCKET, key, final)
        # enriched_meta_obj was the dict being built up, let's assume 'meta' is the one we want to populate
        # meta["s3_key"] = key # This was for the transcript JSON, now covered by transcript_s3_path
        meta["transcript_s3_path"] = f"s3://{BUCKET}/{key}"

    processed_guids.add(guid)
    processed_hashes.add(audio_hash)

    # Enrich metadata with transcript info and entities
    # Assuming nlp and st_model are loaded if perform_caching_global is True
    meta = enrich_meta(
        entry=entry, 
        feed_title=title, 
        feed_url=feed_url, 
        tech=tech, 
        transcript_text=transcript_text, 
        feed=feed, # pass the whole feed object for rights etc.
        nlp_model=None,
        st_model=None,
        base_data_dir=TRANSCRIPT_PATH.parent,
        perform_caching=True
    )

    # Ensure 'published' date is in ISO 8601 format in the final meta object
    if meta.get("published_iso"):
        meta["published"] = meta["published_iso"]
        # To keep meta lean, we can remove published_iso if 'published' is now the canonical ISO version
        # For now, let's keep it for traceability, can be decided later if it should be removed.
        # if "published_iso" in meta:
        #     del meta["published_iso"]

    # --- Extract and Save KPIs ---
    kpis_list = []
    kpis_file_path_obj = None
    if transcript_text: # Ensure there is text to process
        kpis_list = extract_kpis(transcript_text)
        if kpis_list:
            kpis_data_dir = TRANSCRIPT_PATH.parent / "kpis"
            kpis_data_dir.mkdir(parents=True, exist_ok=True)
            kpis_file_name = f"{guid}_kpis.json"
            kpis_file_path_obj = kpis_data_dir / kpis_file_name
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
    s3_prefix = layout_fn(guid) # Ensure guid is valid
    meta["s3_prefix"] = s3_prefix # Store for transparency/DB

    # Use the potentially updated (ISO formatted) 'published' date for slug generation
    raw_episode_slug_part = make_slug(
        podcast_name_or_slug="", 
        episode_title=meta.get("episode", "unknown_episode"),
        published_iso_date=meta.get("published", "nodate") # This should now be ISO if available
    )

    if len(raw_episode_slug_part) > 40:
        # Truncate to near 40 chars, then rsplit to avoid cutting mid-word
        short_episode_slug_for_file = raw_episode_slug_part[:40].rsplit('-', 1)[0]
    else:
        short_episode_slug_for_file = raw_episode_slug_part
    # Ensure it doesn't end with a hyphen if rsplit resulted in that or original was short
    short_episode_slug_for_file = short_episode_slug_for_file.strip('-')

    local_meta_filename = f"meta_{guid}_{short_episode_slug_for_file}.json"
    meta_output_dir = Path(TRANSCRIPT_PATH.parent) / "meta"
    meta_output_dir.mkdir(parents=True, exist_ok=True)
    local_meta_file_path = meta_output_dir / local_meta_filename

    # --- Define S3 paths for all artifacts and add to meta ---
    # Canonical S3 path for the metadata file itself
    s3_meta_key = f"{s3_prefix}meta.json"
    meta["meta_s3_path"] = f"s3://{BUCKET}/{s3_meta_key}"

    # S3 path for transcript (final is the local transcript json Path object)
    if final and final.exists():
        s3_transcript_key = f"{s3_prefix}{final.name}"
        meta["transcript_s3_path"] = f"s3://{BUCKET}/{s3_transcript_key}"
    else:
        # If final doesn't exist, ensure path is None or handled
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
    if final and final.exists(): # final is the transcript JSON path
        meta["transcript_expected_size_bytes"] = final.stat().st_size
    
    if meta.get("cleaned_entities_path"):
        cleaned_entities_file = Path(meta["cleaned_entities_path"])
        if cleaned_entities_file.exists():
            meta["cleaned_entities_expected_size_bytes"] = cleaned_entities_file.stat().st_size
            
    if meta.get("kpis_path"):
        kpis_file = Path(meta["kpis_path"])
        if kpis_file.exists():
            meta["kpis_expected_size_bytes"] = kpis_file.stat().st_size

    # Save the local meta file with all S3 paths now embedded
    with open(local_meta_file_path, "w", encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info(f"SUCCESS: Final combined metadata saved locally to {local_meta_file_path} for GUID {guid}")

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
    if meta.get("published"):
        try:
            # Handle full datetime strings or just YYYY-MM-DD
            published_full_str = meta["published"]
            if 'T' in published_full_str:
                 # Parse then reformat to ensure only date part is used
                date_obj = dt.datetime.fromisoformat(published_full_str.replace('Z', '+00:00'))
                published_date_db_format = date_obj.strftime("%Y-%m-%d")
            else:
                # If it's already YYYY-MM-DD or similar, ensure it's valid date string
                dt.datetime.strptime(published_full_str, "%Y-%m-%d") # validates format
                published_date_db_format = published_full_str
        except ValueError as ve:
            log.warning(f"Could not parse published date '{meta['published']}' to YYYY-MM-DD for DB: {ve}. Skipping DB upsert for this field or using placeholder.")
            # Decide on fallback: either skip upsert or use a placeholder/None
            # For now, we might still upsert with other fields if this is not critical for PK
            # However, published_date is NOT NULL in DB schema
            log.error(f"Published date '{meta['published']}' is invalid, cannot upsert to DB due to NOT NULL constraint. Skipping upsert for GUID {guid}.")
            # continue # or return, depending on loop structure
            # Let's assume we have a valid date from now on for this example, error handling above is key
        
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
            "local_transcript_path": str(final.resolve()) if final and final.exists() else None,
            "local_entities_path": meta.get("entities_path"),
            "local_cleaned_entities_path": meta.get("cleaned_entities_path"),
            "meta_path_local": str(local_meta_file_path.resolve()) # Path to the descriptively named local meta file
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
        # Ensure published_parsed is handled correctly
        published_parsed_val = None
        if ep.get("published"):
            try:
                # feedparser uses _parse_date internally, attempting to replicate its output format (time.struct_time)
                published_parsed_val = feedparser.common.parse_date(ep["published"])
            except Exception as e:
                log.warning(f"Could not parse 'published' date '{ep['published']}' for mock_entry: {e}")

        mock_entry = {
            "id": ep["guid"], # process() uses entry.get("id", "") for guid
            "guid": ep["guid"], # also make it available directly
            "title": ep["episode_title"],
            "link": ep.get("episode_url"),
            "published": ep.get("published"),
            "published_parsed": published_parsed_val,
            "enclosures":  [{ "href": ep["audio_url"] }] if ep.get("audio_url") else [], # process() expects "href"
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
                podcast=ep["podcast_title"], # process() expects 'podcast' not 'podcast_title'
                feed_url=ep["feed_url"])
        return 0 if success else 1
    # ---- End single episode processing mode ----

    # Clear processed sets at start of run (only for multi-episode mode)
    processed_guids.clear()
    processed_hashes.clear()
    
    # Initialize DB_INITIALIZED flag if not present
    global DB_INITIALIZED
    try:
        DB_INITIALIZED
    except NameError:
        DB_INITIALIZED = False

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

    # Initialize the database (create table if not exists)
    # This should ideally be run once, or ensured it's safe to run multiple times (e.g., IF NOT EXISTS)
    # For a script like this, doing it at the start is okay.
    if not DB_INITIALIZED and not os.getenv("DISABLE_DB_OPERATIONS", "false").lower() == "true":
        if not args.episode_details_json: # Initialize DB only if not in single episode mode to avoid repeated inits by parallel
            init_db()
            DB_INITIALIZED = True
        else:
            # In single episode mode, assume DB is initialized by a master process or not needed for this specific call pattern
            # (e.g. if only file generation is desired and DB operations are disabled via ENV var)
            pass 

    sys.exit(main())
