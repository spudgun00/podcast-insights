#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import sys
import feedparser # Ensure this is imported

# Helper function to parse entry dates, similar to backfill_test.py
# This needs to be defined or imported if collect_episodes.py is standalone
# and doesn't share utils with backfill_test.py directly.
# For now, defining it here for simplicity.
def entry_dt(e) -> dt.datetime | None:
    from dateutil import parser as dtparse # Local import to keep it self-contained if needed
    p = e.get("published_parsed") or e.get("updated_parsed")
    if p:
        # feedparser stores dates as time.struct_time
        return dt.datetime(*p[:6])
    for k in ("published", "updated", "pubDate"): # Check common date fields
        if raw_date_str := e.get(k):
            try:
                return dtparse.parse(raw_date_str)
            except Exception:
                pass # Try next key or return None
    return None


parser = argparse.ArgumentParser(description="Collects episode details from RSS feeds and outputs JSONL.")
parser.add_argument("--feed", action="append", required=True,
                    help="One or more RSS URLs (e.g., --feed URL1 --feed URL2)")
parser.add_argument("--since", default="2025-01-01", # Default to a future date to avoid accidental large pulls
                    help="ISO date (YYYY-MM-DD); only emit episodes on/after this date.")
args = parser.parse_args()

# Convert SINCE from string to datetime.date object for comparison
# It's important that published dates are also converted to date objects for comparison
# or that SINCE is a datetime object if comparing with datetime objects.
# The example uses .date(), so entry dates should also be .date().
SINCE_DATE_OBJ = dt.datetime.fromisoformat(args.since).date()

for feed_url in args.feed:
    print(f"Fetching feed: {feed_url}", file=sys.stderr) # Log to stderr
    feed = feedparser.parse(feed_url)
    podcast_title = feed.feed.get("title", "Unknown Podcast").strip()

    if not feed.entries:
        print(f"No entries found in feed: {feed_url}", file=sys.stderr)
        continue

    print(f"Found {len(feed.entries)} entries in {podcast_title}", file=sys.stderr)

    for item in feed.entries:
        pub_datetime_obj = entry_dt(item) # Returns a datetime.datetime object or None
        
        if not pub_datetime_obj:
            print(f"Skipping entry, no parsable date: {item.get('title', '[No Title]')}", file=sys.stderr)
            continue
        
        # Compare date parts only
        if pub_datetime_obj.date() < SINCE_DATE_OBJ:
            # print(f"Skipping older entry: {item.get('title')} published {pub_datetime_obj.date()}", file=sys.stderr)
            continue
        
        # Ensure audio URL exists
        audio_url_val = None
        if item.enclosures and isinstance(item.enclosures, list) and len(item.enclosures) > 0:
            # Common patterns for audio URL in enclosures
            if "href" in item.enclosures[0]:
                audio_url_val = item.enclosures[0]["href"]
            elif "url" in item.enclosures[0]: # Some feeds might use 'url'
                 audio_url_val = item.enclosures[0]["url"]

        if not audio_url_val:
            print(f"Skipping entry, no audio URL: {item.get('title', '[No Title]')}", file=sys.stderr)
            continue

        payload = {
            "guid":           item.get("id") or item.get("guid"), # Use item.id as fallback for guid
            "episode_title":  item.get("title", "Unknown Episode"),
            "audio_url":      audio_url_val,
            "published":      item.get("published"), # Raw published string
            "episode_url":    item.get("link"),
            "podcast_title":  podcast_title,
            "feed_url":       feed_url,
            "itunes_author":  item.get("itunes_author"),
            "itunes_explicit": item.get("itunes_explicit"),
            "summary":        item.get("summary"),
            "tags":           [t.term for t in item.get("tags", []) if hasattr(t, 'term')],
        }
        
        print(json.dumps(payload, ensure_ascii=False))

print("Finished collecting episodes.", file=sys.stderr) 