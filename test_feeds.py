#!/usr/bin/env python3
"""
Test all RSS feeds in config.yaml
"""

import yaml, feedparser

CFG = yaml.safe_load(open("config.yaml"))

for rss in CFG["feeds"]:
    print(f"Testing: {rss}")
    feed = feedparser.parse(rss)
    if not feed.entries:
        print(f"❌ No entries found!")
    else:
        print(f"✅ Found {len(feed.entries)} entries")
        latest = feed.entries[0]
        print(f"   Latest: {latest.title}")
