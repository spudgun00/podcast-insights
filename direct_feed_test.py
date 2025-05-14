#!/usr/bin/env python3
"""
Test RSS feeds by directly fetching them with requests
"""

import requests
import xml.etree.ElementTree as ET
import yaml

CFG = yaml.safe_load(open("config.yaml"))

def test_feed(url):
    print(f"Testing: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Print the first 100 characters to see what we got
        print(f"Response begins with: {response.text[:100]}...")
        
        try:
            # Try to parse as XML
            root = ET.fromstring(response.text)
            # Look for items or entries
            items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
            if items:
                print(f"✅ Found {len(items)} items/entries")
                if len(items) > 0:
                    # Try to extract title
                    title_elem = items[0].find('.//title') or items[0].find('.//{http://www.w3.org/2005/Atom}title')
                    if title_elem is not None and title_elem.text:
                        print(f"   Latest title: {title_elem.text[:80]}...")
            else:
                print("❌ No items/entries found in XML")
        except Exception as e:
            print(f"❌ Error parsing XML: {e}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

# Alternative, verified feeds
alternative_feeds = [
    "https://rss.art19.com/my-first-million",                    # My First Million
    "https://mfmpod.libsyn.com/rss",                            # Alternative My First Million
    "https://20vc.libsyn.com/rss",                              # Alternative Twenty Minute VC
    "https://feeds.transistor.fm/lenny-s-podcast-1",            # Alternative Lenny's Podcast
    "https://feeds.simplecast.com/4MvgQ73R",                    # All-In (alternative)
    "https://podcast.thinkinbusiness.ca/feed.xml",              # Random working podcast to test
]

# Test configured feeds
for rss in CFG["feeds"]:
    test_feed(rss)

print("\nTrying alternative feeds:")
for rss in alternative_feeds:
    test_feed(rss)
