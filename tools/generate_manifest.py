#!/usr/bin/env python3
"""
Generates a CSV manifest of podcast episodes from RSS feeds.
"""
import argparse
import csv
import datetime as dt
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Dict
import yaml
import boto3

# Adjust import path assuming generate_manifest.py is in tools/
# and podcast_insights is at the same level as tools/
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from podcast_insights.feed_utils import feed_items, entry_dt # Assuming entry_dt is also in feed_utils
# If feedparser.USER_AGENT needs to be set, it should be done in feed_utils.py globally

# --- AWS S3 Upload (Optional) ---
USE_AWS = not os.getenv("NO_AWS")
S3 = None
if USE_AWS:
    try:
        import boto3
        S3 = boto3.client("s3")
    except ImportError:
        logging.warning("boto3 library not found, S3 upload will be disabled.")
        S3 = None
        USE_AWS = False
else:
    logging.info("NO_AWS environment variable set, S3 upload is disabled.")

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)7s | %(module)s | %(message)s")
log = logging.getLogger("generate_manifest")

# --- Helper Function (moved from original process() in backfill for mp3_url extraction if needed here) ---
def get_mp3_url(entry) -> str:
    """Extracts the MP3 URL from a feed entry."""
    if entry.enclosures:
        for enc in entry.enclosures:
            if enc.get("type", "").startswith("audio"):
                return enc.href
    return ""

def generate_manifest(config_path: str, output_csv_path: str, since_date_override: Optional[str], 
                      upload_to_s3: bool, s3_bucket_name: str):
    try:
        config = yaml.safe_load(Path(config_path).read_text())
    except FileNotFoundError:
        log.error(f"Configuration file not found: {config_path}")
        return False
    except yaml.YAMLError as e:
        log.error(f"Error parsing YAML configuration file {config_path}: {e}")
        return False

    feeds_to_process = config.get("feeds", [])
    if not feeds_to_process:
        log.warning(f"No feeds found in {config_path}. Manifest will be empty.")
        # Create empty manifest with headers if it's expected
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["feed_url", "podcast_title", "episode_guid", "episode_title", 
                             "mp3_url", "published_date_iso", "podcast_slug"])
        log.info(f"Empty manifest with headers written to {output_csv_path}")
        return True # Successfully created an empty manifest

    if since_date_override:
        try:
            since_dt = dt.datetime.strptime(since_date_override, "%Y-%m-%d")
        except ValueError:
            log.error(f"Invalid --since date format: {since_date_override}. Use YYYY-MM-DD.")
            return False
    else:
        since_dt_str = config.get("since_date", "2000-01-01") # Default to very old if not in config
        since_dt = dt.datetime.strptime(since_dt_str, "%Y-%m-%d")
    
    log.info(f"Processing feeds listed in {config_path}, getting items since {since_dt.strftime('%Y-%m-%d')}")

    csv_header = ["feed_url", "podcast_title", "episode_guid", "episode_title", 
                  "mp3_url", "published_date_iso", "podcast_slug"]
    manifest_records = []

    for feed_url_config_item in feeds_to_process:
        current_feed_url = ""
        podcast_slug_override = None # For future use if config provides slug per feed

        if isinstance(feed_url_config_item, str):
            current_feed_url = feed_url_config_item
        elif isinstance(feed_url_config_item, dict):
            current_feed_url = feed_url_config_item.get("url")
            podcast_slug_override = feed_url_config_item.get("slug") # Example of per-feed config
        
        if not current_feed_url:
            log.warning(f"Skipping invalid feed entry in config: {feed_url_config_item}")
            continue

        log.info(f"Fetching items from feed: {current_feed_url}")
        # feed_items expects since_date as datetime object, debug_feed as bool
        try:
            # feed_items yields: entry, when_dt, podcast_title_from_feed, feed_url_from_feed
            for entry, when_dt, podcast_title_from_feed, _ in feed_items(current_feed_url, since_dt, debug_feed=False):
                guid = entry.get("id", "")
                episode_title = entry.get("title", "")
                mp3_url = get_mp3_url(entry)
                published_iso = when_dt.isoformat() if when_dt else ""
                
                # Determine podcast_slug: use override if available, else generate from title
                # For manifest generation, we might not have meta_utils.generate_podcast_slug easily
                # So, either it comes from config, or we might need a simpler slug for the manifest
                # or the manifest only stores the podcast_title and slug is generated later.
                # For now, let's try to use the override or a very simple slug of the podcast title from feed.
                actual_podcast_slug = podcast_slug_override
                if not actual_podcast_slug:
                    # Basic slugification - replace non-alphanum with hyphen, lowercase
                    actual_podcast_slug = re.sub(r'[^a-z0-9]+', '-', podcast_title_from_feed.lower()).strip('-')
                    if not actual_podcast_slug: # Fallback if title was all special chars
                        actual_podcast_slug = "unknown-podcast"

                if not guid:
                    log.warning(f"Episode '{episode_title}' from '{podcast_title_from_feed}' has no GUID. Skipping.")
                    continue
                if not mp3_url:
                    log.warning(f"Episode '{episode_title}' (GUID: {guid}) from '{podcast_title_from_feed}' has no MP3 URL. Skipping.")
                    continue

                manifest_records.append({
                    "feed_url": current_feed_url, 
                    "podcast_title": podcast_title_from_feed,
                    "episode_guid": guid,
                    "episode_title": episode_title,
                    "mp3_url": mp3_url,
                    "published_date_iso": published_iso,
                    "podcast_slug": actual_podcast_slug # Added podcast_slug
                })
                log.debug(f"Added to manifest: {podcast_title_from_feed} - {episode_title} ({guid})")
        except Exception as e:
            log.error(f"Failed to process feed {current_feed_url}: {e}", exc_info=True)
            # Continue to next feed

    # Write to local CSV
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            writer.writeheader()
            writer.writerows(manifest_records)
        log.info(f"Manifest with {len(manifest_records)} records written to {output_csv_path}")
    except IOError as e:
        log.error(f"Failed to write manifest CSV to {output_csv_path}: {e}")
        return False

    # Upload to S3 if enabled and possible
    if upload_to_s3 and USE_AWS and S3:
        s3_filename = f"{dt.datetime.utcnow().strftime('%Y%m%d')}_{Path(output_csv_path).name}"
        log.info(f"Attempting to upload manifest to s3://{s3_bucket_name}/{s3_filename}")
        try:
            S3.upload_file(str(output_csv_path), s3_bucket_name, s3_filename)
            log.info(f"Manifest successfully uploaded to s3://{s3_bucket_name}/{s3_filename}")
            # Optionally, print the S3 path for podrun to capture if needed in a cloud context
            # print(f"s3://{s3_bucket_name}/{s3_filename}") 
        except Exception as e:
            log.error(f"Failed to upload manifest to S3 bucket {s3_bucket_name}: {e}", exc_info=True)
            # Do not return False here, local manifest generation was successful.
    elif upload_to_s3 and (not USE_AWS or not S3):
        log.warning("S3 upload requested but not possible (AWS not configured or boto3 missing).")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a CSV manifest from podcast RSS feeds.")
    parser.add_argument("--config", type=str, default="tier1_feeds.yaml", 
                        help="Path to the YAML feed configuration file. Default: tier1_feeds.yaml in CWD.")
    parser.add_argument("--output-csv", type=str, default="manifest.csv", 
                        help="Path to save the generated CSV manifest file. Default: manifest.csv in CWD.")
    parser.add_argument("--since", type=str, default=None, 
                        help="Override since_date (YYYY-MM-DD) from config. Processes episodes published after this date.")
    parser.add_argument("--upload-to-s3", action="store_true", 
                        help="Upload the generated manifest to S3.")
    parser.add_argument("--s3-bucket", type=str, default="pod-insights-manifests", 
                        help="S3 bucket name for manifest upload. Default: pod-insights-manifests.")

    args = parser.parse_args()

    # Resolve paths relative to project root if they are not absolute, for consistency when called by podrun
    # CWD for generate_manifest.py when called by podrun will be PROJECT_ROOT.
    config_file_path = Path(args.config)
    if not config_file_path.is_absolute():
        config_file_path = project_root / config_file_path

    output_csv_file_path = Path(args.output_csv)
    if not output_csv_file_path.is_absolute():
        output_csv_file_path = project_root / output_csv_file_path

    success = generate_manifest(
        config_path=str(config_file_path),
        output_csv_path=str(output_csv_file_path),
        since_date_override=args.since,
        upload_to_s3=args.upload_to_s3,
        s3_bucket_name=args.s3_bucket
    )

    if success:
        log.info("Manifest generation process completed.")
        # When podrun.py calls this, it relies on local_manifest_path.exists().
        # If S3 upload is the primary path, podrun might need to get the S3 URI.
        # For now, the local file is the primary output for podrun.
        sys.exit(0)
    else:
        log.error("Manifest generation process failed.")
        sys.exit(1) 