#!/usr/bin/env python3
"""
Test a wider range of podcast feeds to find working ones
"""

import requests
import xml.etree.ElementTree as ET

# List of popular podcasts to test
test_feeds = [
    # Popular tech and startup podcasts
    "https://feeds.simplecast.com/4MvgQ73R",                    # All-In (working)
    "https://lexfridman.com/feed/podcast/",                     # Lex Fridman
    "https://feeds.npr.org/510313/podcast.xml",                 # How I Built This
    "https://feeds.megaphone.fm/vergecast",                     # The Vergecast
    "https://www.wired.com/feed/podcast/wired-podcast/rss",     # WIRED
    "https://feeds.buzzsprout.com/1775696.rss",                 # First Round Search Fund 
    "https://feeds.acast.com/public/shows/masters-of-scale",    # Masters of Scale
    "https://feeds.megaphone.fm/startup",                       # Startup Podcast
    "https://feed.podbean.com/failory/feed.xml",                # Failory
    "https://feeds.transistor.fm/indie-hackers",                # Indie Hackers (new feed)
    "https://feeds.simplecast.com/R0M5kZQm",                    # A16Z
    "https://feeds.megaphone.fm/ROOSTERGLOBAL8472876393",       # Tech Buzz China
]

def test_feed(url):
    print(f"\nTesting: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        try:
            # Try to parse as XML
            root = ET.fromstring(response.text)
            # Look for items or entries
            items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
            if items:
                print(f"✅ Found {len(items)} items/entries")
                # Try to extract title and link of the first episode
                if len(items) > 0:
                    # Try to get feed title
                    channel = root.find('.//channel')
                    if channel is not None:
                        feed_title = channel.find('title')
                        if feed_title is not None and feed_title.text:
                            print(f"📢 Podcast: {feed_title.text}")
                    
                    # Try to extract episode title
                    title_elem = items[0].find('.//title') or items[0].find('.//{http://www.w3.org/2005/Atom}title')
                    if title_elem is not None and title_elem.text:
                        print(f"   Latest episode: {title_elem.text[:80]}...")
                    
                    # Try to find enclosure
                    enclosures = items[0].findall('.//enclosure')
                    if enclosures:
                        for enclosure in enclosures:
                            if 'type' in enclosure.attrib and enclosure.attrib['type'].startswith('audio/'):
                                print(f"   Audio URL found ✓")
                                break
                        else:
                            print("   ❌ No audio enclosure found")
                    else:
                        print("   ❌ No enclosures found")
                    
                print(f"👍 WORKING FEED: {url}")
                return True
            else:
                print("❌ No items/entries found in XML")
                return False
        except Exception as e:
            print(f"❌ Error parsing XML: {e}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

# Test all feeds and collect working ones
working_feeds = []
for feed_url in test_feeds:
    if test_feed(feed_url):
        working_feeds.append(feed_url)

# Print summary of working feeds
print("\n=== SUMMARY OF WORKING FEEDS ===")
for i, feed in enumerate(working_feeds, 1):
    print(f"{i}. {feed}")

print("\nUpdate your config.yaml with these working feeds")
