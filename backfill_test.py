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
MODEL_VERSION = f"whisper-{args.model_size or CFG.get('model_size','base')}-{dt.datetime.now():%Y-%m-%d}"

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
    subprocess.run(
        [
            sys.executable,
            "-m",
            "podcast_insights.transcribe",
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
            meta = enrich_meta(entry, podcast, feed_url, tech, transcript_text, feed)
            
            # Process transcript and add timestamp info
            meta = process_transcript(transcript_data, meta)
            
            # Add any additional metadata
            meta.update({
                "segment_count": len(transcript_data.get("segments", [])),
                "chunk_count": len(chunks) if len(chunks) > 1 else 1,
                "audio_hash": audio_hash,
                "download_path": str(mp3),
                "transcript_path": str(final.absolute())
            })

            # Update the final JSON with enriched metadata
            final_data = json.loads(final.read_text())
            final_data["meta"] = meta
            
            # Ensure complete paths are preserved in JSON output
            class PathEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Path):
                        return str(obj.absolute())
                    return super().default(obj)
            
            # Write JSON with custom encoder to preserve paths
            final.write_text(json.dumps(final_data, cls=PathEncoder, ensure_ascii=False, indent=2))

    except Exception as e:
        log.error("transcribe error", exc_info=e)
        return False

    if USE_AWS:
        key = f"json/{final.name}"
        S3.upload_file(str(final), BUCKET, key)
        verify_s3_upload(S3, BUCKET, key, final)
        meta["s3_key"] = key

    processed_guids.add(guid)
    processed_hashes.add(audio_hash)
    return True


def main() -> int:
    log.info(f"since={SINCE_DATE:%Y-%m-%d} dry={DRY_RUN} aws={USE_AWS}")
    
    # Clear processed sets at start of run
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
    sys.exit(main())
