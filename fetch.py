#!/usr/bin/env python3
"""
Fetch latest episode MP3 from each RSS feed in config.yaml
using direct XML parsing (no feedparser)
"""

import os
import yaml
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from tqdm import tqdm

# Load configuration
CFG = yaml.safe_load(open("config.yaml"))
AUDIO_ROOT = CFG["download_path"]
os.makedirs(AUDIO_ROOT, exist_ok=True)

def get_latest_episode_info(feed_url):
    """Get the audio URL and title from the latest episode"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    }
    print(f"Fetching feed: {feed_url}")
    
    try:
        response = requests.get(feed_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        # Find podcast title
        podcast_title = "Unknown Podcast"
        channel = root.find('.//channel')
        if channel is not None:
            title_elem = channel.find('./title')
            if title_elem is not None and title_elem.text:
                podcast_title = title_elem.text
        
        # Find items/episodes
        items = root.findall('.//item')
        
        if not items:
            print(f"No episodes found in feed!")
            return None, None, None
            
        print(f"Found {len(items)} episodes in {podcast_title}")
        
        # Get the first (latest) item
        latest_item = items[0]
        
        # Try to find the episode title
        episode_title = "Unknown Episode"
        title_elem = latest_item.find('./title')
        if title_elem is not None and title_elem.text:
            episode_title = title_elem.text
            print(f"Latest episode: {episode_title}")
        
        # Find the enclosure (audio file)
        enclosure = latest_item.find('./enclosure')
        if enclosure is not None:
            audio_url = enclosure.get('url')
            if audio_url:
                return audio_url, episode_title, podcast_title
        
        # If standard enclosure not found, try other approaches
        # Look for media:content
        namespaces = {'media': 'http://search.yahoo.com/mrss/'}
        media_content = latest_item.find('./media:content', namespaces)
        if media_content is not None:
            audio_url = media_content.get('url')
            if audio_url:
                return audio_url, episode_title, podcast_title
                
        print("No audio URL found in the latest episode")
        return None, episode_title, podcast_title
        
    except Exception as e:
        print(f"Error fetching feed: {str(e)}")
        return None, None, None

def download_file(url, output_path):
    """Download a file with progress bar"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    }
    print(f"Downloading from: {url}")
    try:
        with requests.get(url, stream=True, headers=headers, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            pbar.update(len(chunk))
            return True
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return False

# Process each feed in config
successful_downloads = 0
for feed_url in CFG["feeds"]:
    # Get the latest episode info
    audio_url, episode_title, podcast_title = get_latest_episode_info(feed_url)
    
    if not audio_url:
        print(f"Skipping {feed_url} - no audio URL found")
        continue
    
    # Generate filename from URL or title
    filename = os.path.basename(urlparse(audio_url).path)
    if not filename or '.' not in filename:
        # Clean title for use as filename
        clean_title = episode_title.replace(' ', '_').replace('/', '_')[:50]
        filename = f"{clean_title}.mp3"
    
    # Add podcast name prefix for better organization
    clean_podcast = podcast_title.replace(' ', '_').replace('/', '_')[:20]
    filename = f"{clean_podcast}_{filename}"
    
    output_path = os.path.join(AUDIO_ROOT, filename)
    
    # Check if file already exists
    if os.path.exists(output_path):
        print(f"✔ Skipping (already exists): {filename}")
        continue
    
    # Download the file
    success = download_file(audio_url, output_path)
    
    if success:
        # Verify file size
        file_size = os.path.getsize(output_path)
        if file_size < 1000000:  # Less than 1MB
            print(f"⚠️ Warning: File seems small ({file_size} bytes). Might be incomplete.")
        else:
            print(f"✅ Successfully downloaded: {filename}")
            successful_downloads += 1

print(f"\nSummary: Downloaded {successful_downloads} new episodes")
