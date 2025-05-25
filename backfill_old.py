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

from podcast_insights.audio_utils import (                    # all live in audio_utils.py
    calculate_audio_hash,
    chunk_long_audio,
    download_with_retry,
    get_audio_tech,
    estimate_speech_music_ratio,
    verify_audio,
    verify_s3_upload,
    check_timestamp_support,
)

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

BUCKET = "startupaudio-transcripts"

# -------------------------------------------------------------------------- CLI
pa = argparse.ArgumentParser(description="Back-fill podcast transcripts")
pa.add_argument("--dry_run", action="store_true")
pa.add_argument("--since")                  # YYYY-MM-DD override
pa.add_argument("--feed")                  # single RSS URL
pa.add_argument("--limit", type=int, default=0)
pa.add_argument("--model_size",
                choices=["tiny", "base", "small", "medium", "large"])
pa.add_argument("--parallel", action="store_true", help="Enable parallel processing (AWS only)")
pa.add_argument("--thread_safe", action="store_true", help="Enable thread safety for AWS")
args = pa.parse_args()
DRY_RUN = bool(args.dry_run)
PARALLEL = bool(args.parallel)
THREAD_SAFE = bool(args.thread_safe)

# ---------------------------------------------------------------------- logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)7s | %(message)s")
log = logging.getLogger("backfill")
DEBUG_FEED = os.getenv("DEBUG_FEED") == "1"

# ------------------------------------------------------------------- config/YAML
CFG = yaml.safe_load(Path("all24.yaml").read_text())   # backfill_test.py uses test-five.yaml
SINCE_DATE = dt.datetime.strptime(
    args.since or CFG.get("since_date", "2025-01-01"), "%Y-%m-%d"
)
DOWNLOAD_PATH = Path(os.getenv("DOWNLOAD_PATH",
                               CFG.get("download_path", "/tmp/audio")))
TRANSCRIPT_PATH = Path(os.getenv("TRANSCRIPT_PATH",
                                 CFG.get("transcript_path", "/tmp/transcripts")))
MODEL_VERSION = f"whisper-{args.model_size or CFG.get('model_size','base')}-{dt.datetime.now():%Y-%m-%d}"

# Thread-safe tracking if enabled
if THREAD_SAFE:
    from threading import Lock
    processed_guids_lock = Lock()
    processed_hashes_lock = Lock()
    processed_guids: set[str] = set()
    processed_hashes: set[str] = set()
else:
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


# ----------------------------- rich-metadata (rights, persons, categories …)
def enrich_meta(entry, feed_title: str, feed_url: str, tech: dict[str, Any]) -> dict:
    # ---- rights + explicit --------------------------------------------------
    d = feedparser.parse(feed_url)
    rights = entry.get("copyright") or d.feed.get("copyright")
    explicit = entry.get("itunes_explicit") or d.feed.get("itunes_explicit")

    # ---- <podcast:person> ---------------------------------------------------
    hosts, guests = [], []
    raw_summary = getattr(entry, "summary_detail", {}).get("value", "")
    if raw_summary:
        soup = BeautifulSoup(raw_summary, "xml")
        for tag in soup.find_all("podcast:person"):
            person = {
                "name": tag.text.strip(),
                "role": (tag.get("role") or "guest").lower(),
                "href": tag.get("href"),
            }
            (hosts if person["role"] in ("host", "presenter") else guests).append(person)

    # ---- categories ---------------------------------------------------------
    cats = [t["term"] for t in entry.get("tags", [])] if entry.get("tags") else []

    return {
        "podcast": feed_title,
        "episode": entry.get("title", ""),
        "guid": entry.get("id", ""),
        "published": entry.get("published", ""),
        "episode_url": entry.get("link"),
        "audio_url": entry.enclosures[0]["href"] if entry.enclosures else "",
        "itunes_episodeType": entry.get("itunes_episode_type"),
        "categories": cats,
        "hosts": hosts,
        "guests": guests,
        "rights": {"copyright": rights, "explicit": explicit},
        "asr_model": MODEL_VERSION,
        "processed_date": dt.datetime.utcnow().isoformat(),
        **tech,
    }


def feed_items(url: str) -> list[Tuple[Any, dt.datetime, str, str]]:
    log.info(f"Fetch {url}")
    d = feedparser.parse(url)
    title = getattr(d.feed, "title", url)
    out = []
    for ent in d.entries:
        when = entry_dt(ent)
        if DEBUG_FEED:
            log.debug(f"  ↳ {ent.get('title')!r} {when}")
        if when and when >= SINCE_DATE:
            out.append((ent, when, title, url))
    if not out:
        log.warning(f"No items ≥{SINCE_DATE:%Y-%m-%d} in {url}")
    return out


@retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(6))
def run_transcribe(chunk: Path, out_json: Path, meta: dict) -> None:
    import subprocess, sys, json
    subprocess.run(
        [
            sys.executable,
            "-m", "podcast_insights.transcribe",
            "--file",
            str(chunk),
            "--output",
            str(out_json),
            "--model_size",
            meta["asr_model"].split("-")[1],     # tiny/base/…
            "--vad_filter",
            "True",
            "--metadata",
            json.dumps(meta),
        ],
        check=True,
    )

# ------------------------------------------------------------------- pipeline
def is_processed(guid: str, audio_hash: str) -> bool:
    if THREAD_SAFE:
        with processed_guids_lock:
            if guid in processed_guids:
                return True
        with processed_hashes_lock:
            if audio_hash in processed_hashes:
                return True
    else:
        if guid in processed_guids:
            return True
        if audio_hash in processed_hashes:
            return True
    return False

def mark_processed(guid: str, audio_hash: str) -> None:
    if THREAD_SAFE:
        with processed_guids_lock:
            processed_guids.add(guid)
        with processed_hashes_lock:
            processed_hashes.add(audio_hash)
    else:
        processed_guids.add(guid)
        processed_hashes.add(audio_hash)


def process(entry, when: dt.datetime, podcast: str, feed_url: str) -> bool:
    guid = entry.get("id", "")
    title = entry.get("title", "")
    mp3 = entry.enclosures[0]["href"] if entry.enclosures else ""
    if not mp3:
        log.warning(f"skip {title!r} (no .mp3 url)")
        return True

    # ---- calculate audio hash for caching/identification --------------------
    try:
        audio_hash = calculate_audio_hash(mp3)
    except Exception as e:
        log.error(f"Failed to calculate audio hash for {mp3}: {e}")
        return False # Or True if you want to skip problematic files

    # ---- check if already processed -----------------------------------------
    if is_processed(guid, audio_hash):
        log.info(f"skip {title!r} (already processed guid or audio_hash)")
        return True

    # ---- file paths setup ---------------------------------------------------
    DOWNLOAD_PATH.mkdir(exist_ok=True, parents=True)
    TRANSCRIPT_PATH.mkdir(exist_ok=True, parents=True)

    ts_prefix = when.strftime("%Y%m%d")
    audio_file = DOWNLOAD_PATH / f"{ts_prefix}_{md5_8(mp3)}_{Path(mp3).name}"
    json_file = TRANSCRIPT_PATH / f"{ts_prefix}_{md5_8(mp3)}_{Path(mp3).stem}.json"

    if not DRY_RUN:
        # ---- download -----------------------------------------------------------
        log.info(f"  → {audio_file.name}")
        download_with_retry(mp3, audio_file)
        valid_audio, reason = verify_audio(audio_file)
        if not valid_audio:
            log.error(f"Failed audio verification for {audio_file}: {reason}")
            return False

        # ---- tech metadata ------------------------------------------------------
        tech_meta = get_audio_tech(audio_file)
        meta = enrich_meta(entry, podcast, feed_url, tech_meta)

        # ---- chunking for long audio ------------------------------------------
        # Note: chunk_long_audio returns a list of paths to chunks or the original file
        # if no chunking is needed. run_transcribe will be called for each chunk.
        # For simplicity, this example assumes run_transcribe can handle a list of files
        # or you would loop here.

        audio_chunks = chunk_long_audio(audio_file, meta.get('language', 'en'))
        all_segment_data = [] # To store segment data from all chunks

        for i, chunk_path in enumerate(audio_chunks):
            chunk_json_file = json_file.with_name(f"{json_file.stem}_chunk_{i}{json_file.suffix}")
            log.info(f"Transcribing chunk {i+1}/{len(audio_chunks)}: {chunk_path.name} to {chunk_json_file.name}")
            run_transcribe(chunk_path, chunk_json_file, meta)
            
            # If transcribe.py saves segments directly, you might need to load them here
            # to aggregate. For now, let's assume the main json_file is what we care about
            # or that process_transcript will handle finding chunked results.
            if i == 0 and len(audio_chunks) == 1: # if only one chunk, it's the main file
                json_file = chunk_json_file # Use the actual output file name

        # If multiple chunks, you'd need a strategy to combine their JSON outputs or segments.
        # This example assumes process_transcript can handle it or it's handled later.

        # ---- S3 upload (optional) ---------------------------------------------
        if USE_AWS:
            s3_path = f"transcripts/{podcast}/{when.year}/{audio_file.name}"
            S3.upload_file(str(audio_file), BUCKET, s3_path)
            verify_s3_upload(s3_path, audio_file.stat().st_size, S3, BUCKET) # Verify after upload
            meta["s3_audio_path"] = f"s3://{BUCKET}/{s3_path}"

            s3_transcript_path = f"transcripts/{podcast}/{when.year}/{json_file.name}"
            S3.upload_file(str(json_file), BUCKET, s3_transcript_path)
            verify_s3_upload(s3_transcript_path, json_file.stat().st_size, S3, BUCKET) # Verify after upload
            meta["s3_transcript_path"] = f"s3://{BUCKET}/{s3_transcript_path}"

        # ---- save final metadata ----------------------------------------------
        class PathEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Path):
                    return str(obj)
                return json.JSONEncoder.default(self, obj)
        json_file.write_text(json.dumps(meta, indent=2, cls=PathEncoder))
        log.info(f"Done: {json_file.name}")

    mark_processed(guid, audio_hash) # Mark as processed even in DRY_RUN to avoid re-picking
    return True

# ------------------------------------------------------------------------ main
def main() -> int:
    try:
        DOWNLOAD_PATH.mkdir(exist_ok=True, parents=True)
        TRANSCRIPT_PATH.mkdir(exist_ok=True, parents=True)
        feeds = [args.feed] if args.feed else CFG["feeds"]
        count = 0
        for url in feeds:
            try:
                for entry, when, podcast, feed_url_val in feed_items(url):
                    if TERMINATE:
                        return 1
                    if args.limit and count >= args.limit:
                        log.warning(f"Limit {args.limit} reached.")
                        return 0
                    log.info(f"{podcast}: {entry.get('title')} {when}")
                    if process(entry, when, podcast, feed_url_val):
                        count += 1
            except Exception:
                log.error(f"Feed {url} failed: {traceback.format_exc()}")
        return 0
    except Exception:
        log.critical(traceback.format_exc())
        return 1
    finally:
        if PARALLEL:
            # Ensure any background workers are cleaned up if you were using them
            log.info("Parallel processing cleanup (if any).")

if __name__ == "__main__":
    exit_code = main()
    # input("Press Enter to exit...") # For debugging standalone console
    sys.exit(exit_code) 