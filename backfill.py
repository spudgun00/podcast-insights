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
    """Thread-safe check if episode is processed."""
    if THREAD_SAFE:
        with processed_guids_lock:
            if guid in processed_guids:
                return True
        with processed_hashes_lock:
            if audio_hash in processed_hashes:
                return True
        return False
    return guid in processed_guids or audio_hash in processed_hashes

def mark_processed(guid: str, audio_hash: str) -> None:
    """Thread-safe mark episode as processed."""
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
    if is_processed(guid, ""):  # Check guid first
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
    tech = get_audio_tech(mp3)
    tech["speech_music_ratio"] = estimate_speech_music_ratio(mp3)

    audio_hash = calculate_audio_hash(mp3)
    if is_processed("", audio_hash):  # Check hash
        mp3.unlink(missing_ok=True)
        return True

    meta = enrich_meta(entry, podcast, feed_url, tech)
    meta["supports_timestamp"] = check_timestamp_support(audio_url)

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
                
                # Add transcript_path to metadata
                meta.update({
                    "segment_count": len(combo["segments"]),
                    "chunk_count": len(chunks),
                    "download_path": str(mp3.absolute()),
                    "transcript_path": str(final.absolute())
                })
                combo["meta"] = meta
                
                # Custom JSON encoder to handle Path objects
                class PathEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, Path):
                            return str(obj.absolute())
                        return super().default(obj)
                
                # Use custom encoder to ensure full paths are preserved
                final.write_text(json.dumps(combo, cls=PathEncoder, ensure_ascii=False, indent=2))
            else:
                # Single chunk case - add transcript_path
                meta.update({
                    "segment_count": len(json.loads(final.read_text()).get("segments", [])),
                    "chunk_count": 1,
                    "download_path": str(mp3.absolute()),
                    "transcript_path": str(final.absolute())
                })
                
                # Update JSON with full path information
                final_data = json.loads(final.read_text())
                final_data["meta"] = meta
                
                # Custom JSON encoder to handle Path objects
                class PathEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, Path):
                            return str(obj.absolute())
                        return super().default(obj)
                
                # Use custom encoder to ensure full paths are preserved
                final.write_text(json.dumps(final_data, cls=PathEncoder, ensure_ascii=False, indent=2))
    except Exception as e:
        log.error("transcribe error", exc_info=e)
        return False

    if USE_AWS:
        key = f"json/{final.name}"
        S3.upload_file(str(final), BUCKET, key)
        verify_s3_upload(S3, BUCKET, key, final)

    mark_processed(guid, audio_hash)
    return True


def main() -> int:
    if PARALLEL and not USE_AWS:
        log.error("Parallel processing only available on AWS")
        return 1

    if PARALLEL:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for feed_url in CFG["feeds"]:
                for entry, when, podcast, url in feed_items(feed_url):
                    if args.limit and len(futures) >= args.limit:
                        break
                    futures.append(executor.submit(process, entry, when, podcast, url))
            for future in futures:
                future.result()
    else:
        for feed_url in CFG["feeds"]:
            for entry, when, podcast, url in feed_items(feed_url):
                if args.limit and len(processed_guids) >= args.limit:
                    break
                process(entry, when, podcast, url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
