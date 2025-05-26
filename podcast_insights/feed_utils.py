#!/usr/bin/env python3
"""
Feed parsing utilities.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, List, Optional, Tuple

import feedparser
from dateutil import parser as dtparse

# Set a global User-Agent for feedparser
feedparser.USER_AGENT = "PodInsight-MVP/1.0 (https://github.com/your-repo/podcast-insights; mailto:your-email@example.com)"
# Please update the USER_AGENT string with your actual project details.

logger = logging.getLogger(__name__)

def entry_dt(entry: feedparser.FeedParserDict) -> Optional[dt.datetime]:
    """
    Extracts and parses the publication or update date from a feed entry.
    Tries 'published_parsed', 'updated_parsed', then raw 'published', 'updated', 'pubDate'.
    Returns a datetime object or None if no valid date is found.
    """
    parsed_date = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed_date:
        # feedparser.FeedParserDict.published_parsed is a time.struct_time
        return dt.datetime(*parsed_date[:6])
    
    for key in ("published", "updated", "pubDate"):
        raw_date_str = entry.get(key)
        if raw_date_str:
            try:
                return dtparse.parse(raw_date_str)
            except (dtparse.ParserError, TypeError, ValueError) as e:
                logger.debug(f"Failed to parse date string '{raw_date_str}' for key '{key}': {e}")
                pass  # Try the next key
    return None


def feed_items(
    url: str, since_date: dt.datetime, debug_feed: bool = False
) -> list[Tuple[feedparser.FeedParserDict, dt.datetime, str, str]]:
    """
    Fetches and parses a podcast feed, returning entries published on or after since_date.

    Args:
        url: The URL of the RSS feed.
        since_date: The earliest publication date for entries to include.
        debug_feed: If True, logs verbose details of each entry.

    Returns:
        A list of tuples, where each tuple contains:
        - The feedparser entry object.
        - The publication datetime.
        - The podcast title.
        - The feed URL.
    """
    logger.info(f"Fetching feed: {url}")
    try:
        feed_data = feedparser.parse(url)
    except Exception as e:
        logger.error(f"Error parsing feed {url}: {e}")
        return []

    if feed_data.bozo:
        bozo_msg = f"Feed {url} may be ill-formed. Bozo reason: {feed_data.bozo_exception}"
        if isinstance(feed_data.bozo_exception, feedparser.NonXMLContentType):
            logger.error(bozo_msg)
            return []
        logger.warning(bozo_msg)

    podcast_title = getattr(feed_data.feed, "title", url)
    entries_out: List[Tuple[feedparser.FeedParserDict, dt.datetime, str, str]] = []

    logger.info(f"Found {len(feed_data.entries)} entries in feed: {podcast_title}")
    for entry in feed_data.entries:
        published_datetime = entry_dt(entry)

        if debug_feed:
            entry_title_debug = entry.get("title", "N/A")
            logger.debug(f"  -> Entry: '{entry_title_debug}', Published: {published_datetime}")

        if published_datetime and published_datetime.replace(tzinfo=None) >= since_date.replace(tzinfo=None):
            entries_out.append((entry, published_datetime, podcast_title, url))
            entry_title_log = entry.get("title", "N/A")
            logger.info(f"  + Added entry: '{entry_title_log}' (Published: {published_datetime:%Y-%m-%d})")
        elif not published_datetime:
            entry_title_log = entry.get("title", "N/A")
            logger.warning(f"  - Skipped entry (no valid date): '{entry_title_log}' from feed {url}")


    if not entries_out:
        logger.warning(
            f"No new entries found since {since_date:%Y-%m-%d} in feed: {url} ({podcast_title})"
        )
    
    return entries_out 