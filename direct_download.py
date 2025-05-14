#!/usr/bin/env python3
"""
Directly download a podcast episode using requests without feedparser
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

def download_file(url, output_path):
    """Download a file with progress bar"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    }
    print(f"Downloading from: {url}")
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        pbar.update(len(chunk))

def get_latest_episode_url(feed_url):
    """Get the audio URL from the latest episode using requests directly"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    }
    print(f"Fetching feed: {feed_url}")
    
    try:
        response = requests.get(feed_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        # Find items/episodes (handle different XML namespaces)
        items = root.findall('.//item')
        
        if not items:
            print(f"No items found in feed!")
            return None
            
        print(f"Found {len(items)} episodes")
        
        # Get the first (latest) item
        latest_item = items[0]
        
        # Try to find the title
        title_elem = latest_item.find('.//title')
        if title_elem is not None and title_elem.text:
            print(f"Latest episode: {title_elem.text}")
        
        # Find the enclosure (audio file)
        enclosure = latest_item.find('.//enclosure')
        if enclosure is not None:
            audio_url = enclosure.get('url')
            if audio_url:
                return audio_url
                
        print("No audio URL found in the latest episode")
        return None
        
    except Exception as e:
        print(f"Error fetching feed: {str(e)}")
        return None

# Try to download from How I Built This podcast (a reliable feed)
feed_url = "https://feeds.npr.org/510313/podcast.xml"
audio_url = get_latest_episode_url(feed_url)

if audio_url:
    # Generate filename from URL
    filename = os.path.basename(urlparse(audio_url).path)
    if not filename or '.' not in filename:
        filename = "how_i_built_this_latest.mp3"
        
    output_path = os.path.join(AUDIO_ROOT, filename)
    
    # Download the file
    try:
        download_file(audio_url, output_path)
        print(f"✅ Successfully downloaded to {output_path}")
    except Exception as e:
        print(f"Error downloading: {str(e)}")
else:
    print("Could not find audio URL in feed")
