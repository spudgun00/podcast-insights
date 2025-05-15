#!/usr/bin/env python3
"""
Test multiple podcast feeds and verify they're working
"""

import os
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import time
import html

# Create data directories if they don't exist
os.makedirs("data/audio", exist_ok=True)
os.makedirs("data/transcripts", exist_ok=True)

# List of feeds to test - including all from the table
feeds_to_test = [
    # All feeds from your table with some organized information
    {"name": "All-In", "url": "https://allinchamathjason.libsyn.com/rss", "priority": "p1"},
    {"name": "Twenty Minute VC", "url": "https://thetwentyminutevc.libsyn.com/rss", "priority": "p2"},
    {"name": "Acquired", "url": "https://feeds.transistor.fm/acquired", "priority": "p5"},
    {"name": "a16z Podcast", "url": "https://feeds.simplecast.com/JGE3yC0V", "priority": "p5"},
    {"name": "Lenny's Podcast", "url": "https://api.substack.com/feed/podcast/10845.rss", "priority": "p4"},
    {"name": "This Week in Startups", "url": "https://anchor.fm/s/7c624c84/podcast/rss", "priority": "p3"},
    {"name": "SaaStr Podcast", "url": "https://saastr.libsyn.com/rss", "priority": "p10"},
    {"name": "Indie Hackers", "url": "https://feeds.transistor.fm/the-indie-hackers-podcast", "priority": "p9"},
    {"name": "Founders Podcast", "url": "https://feeds.simplecast.com/3hnxp7yk", "priority": "-"},
    {"name": "TechCrunch Equity", "url": "https://feeds.megaphone.fm/YFL6537156961", "priority": "p8"},
    {"name": "Y Combinator Podcast", "url": "https://anchor.fm/s/8c1524bc/podcast/rss", "priority": "-"},
    {"name": "Masters of Scale", "url": "https://rss.art19.com/masters-of-scale", "priority": "-"},
    {"name": "My First Million", "url": "https://feeds.megaphone.fm/HS2300184645", "priority": "p7"},
    {"name": "How I Built This", "url": "https://feeds.npr.org/510313/podcast.xml", "priority": "-"},
    {"name": "The Pitch", "url": "https://feeds.megaphone.fm/thepitch", "priority": "-"},
    {"name": "Startups for the Rest of Us", "url": "https://feeds.castos.com/mqv6", "priority": "-"},
    {"name": "The Bootstrapped Founder", "url": "https://feeds.transistor.fm/bootstrapped-founder", "priority": "-"},
    {"name": "The SaaS Podcast", "url": "https://feeds.megaphone.fm/AHARO1075645324", "priority": "-"},
    {"name": "Turpentine VC", "url": "https://feeds.megaphone.fm/turpentinevc", "priority": "-"},
    {"name": "Fintech Insider", "url": "https://feeds.megaphone.fm/FS9665566819", "priority": "-"},
    {"name": "UI Breakfast", "url": "https://feeds.simplecast.com/4MvgQ73R", "priority": "-"},
    {"name": "Huberman Lab", "url": "https://feeds.megaphone.fm/hubermanlab", "priority": "-"},
    {"name": "Planet Money", "url": "https://feeds.npr.org/510051/podcast.xml", "priority": "-"},
    {"name": "A Product Market Fit Show", "url": "https://feeds.buzzsprout.com/1889238.rss", "priority": "-"}
]

def get_latest_episode_info(feed_url, podcast_name):
    """Get the audio URL and title from the latest episode"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
        'Accept': 'application/rss+xml, application/xml, text/xml'
    }
    print(f"Testing: {podcast_name} ({feed_url})")
    
    try:
        response = requests.get(feed_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse XML
        try:
            root = ET.fromstring(response.content)
            
            # Find podcast title to confirm
            podcast_title = podcast_name
            channel = root.find('.//channel/title')
            if channel is not None and channel.text:
                podcast_title = channel.text
                print(f"Confirmed podcast title: {podcast_title}")
            
            # Find items/episodes
            items = root.findall('.//item')
            
            if not items:
                print(f"❌ No episodes found in feed")
                return None, None, None
                
            print(f"✅ Found {len(items)} episodes")
            
            # Get the first (latest) item
            latest_item = items[0]
            
            # Try to find the episode title
            episode_title = "Unknown Episode"
            title_elem = latest_item.find('./title')
            if title_elem is not None and title_elem.text:
                episode_title = title_elem.text
                print(f"   Latest episode: {episode_title}")
            
            # Find the enclosure (audio file)
            enclosure = latest_item.find('./enclosure')
            if enclosure is not None:
                audio_url = enclosure.get('url')
                if audio_url:
                    print(f"   ✅ Audio URL found: {audio_url}")
                    return audio_url, episode_title, podcast_title
            
            print("   ❌ No audio URL found in latest episode")
            return None, episode_title, podcast_title
            
        except ET.ParseError as e:
            print(f"❌ XML parsing error: {e}")
            return None, None, None
            
    except Exception as e:
        print(f"❌ Error fetching feed: {str(e)}")
        return None, None, None

# Test each feed and track results
working_feeds = []
results = []

for feed in feeds_to_test:
    podcast_name = feed["name"]
    feed_url = feed["url"]
    priority = feed["priority"]
    
    audio_url, episode_title, podcast_title = get_latest_episode_info(feed_url, podcast_name)
    
    if audio_url:
        working_feeds.append({
            "name": podcast_name,
            "url": feed_url,
            "priority": priority
        })
        results.append({
            "name": podcast_name,
            "url": feed_url,
            "priority": priority,
            "status": "Working ✅",
            "latest_episode": episode_title
        })
    else:
        results.append({
            "name": podcast_name,
            "url": feed_url,
            "priority": priority,
            "status": "Failed ❌",
            "latest_episode": episode_title if episode_title else "Unknown"
        })
    
    # Add a small delay between requests to avoid rate limiting
    time.sleep(2)

# Sort results by priority (putting p1, p2, etc. first)
def sort_key(item):
    priority = item["priority"]
    if priority.startswith('p'):
        try:
            return int(priority[1:])
        except:
            return 999
    return 999

# Sort by priority
sorted_results = sorted(results, key=sort_key)

# Print results table
print("\n=== FEED TESTING RESULTS ===")
print(f"{'Priority':<10} | {'Podcast Name':<30} | {'Status':<15} | {'Latest Episode':<50}")
print("-" * 110)
for result in sorted_results:
    episode = result["latest_episode"]
    if episode and len(episode) > 47:
        episode = episode[:47] + "..."
    print(f"{result['priority']:<10} | {result['name']:<30} | {result['status']:<15} | {episode:<50}")

# Sort working feeds by priority
sorted_working_feeds = sorted(working_feeds, key=sort_key)

# Print config for working feeds
print("\n=== YAML CONFIG ===")
print("feeds:")
for feed in sorted_working_feeds:
    print(f"  - {feed['url']}  # {feed['name']}")
print("download_path: data/audio")
print("transcript_path: data/transcripts")
print("model_size: base")

# Also create the YAML file directly
with open("config_updated.yaml", "w") as f:
    f.write("feeds:\n")
    for feed in sorted_working_feeds:
        f.write(f"  - {feed['url']}  # {feed['name']}\n")
    f.write("download_path: data/audio\n")
    f.write("transcript_path: data/transcripts\n")
    f.write("model_size: base\n")

print("\nA new config file has been created at config_updated.yaml with all working feeds")
