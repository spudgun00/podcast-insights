#!/usr/bin/env python3
"""
Rebuild missing podcast metadata without re-transcription.

This script restores metadata fields that may have been lost, including:
- podcast, episode, guid, published, episode_url, audio_url from RSS
- categories/tags from RSS <itunes:category>
- rights/explicit flags from RSS
- hosts/guests from podcast:person tags or NER on intro segments
- optional embeddings and entities for semantic search
"""
import os
import sys
import json
import glob
import logging
import argparse
from pathlib import Path
import hashlib
import numpy as np
import re

import feedparser
import spacy
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s | %(levelname)7s | %(message)s")
logger = logging.getLogger(__name__)

def compute_embedding(text, model_name="all-MiniLM-L6-v2"):
    """
    Compute embedding for text using sentence-transformers
    
    Args:
        text: Text to compute embedding for
        model_name: Sentence transformer model name
        
    Returns:
        Numpy array containing the embedding
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model.encode(text)
    except ImportError:
        logger.warning("sentence-transformers not installed, skipping embedding generation")
        return None
    except Exception as e:
        logger.error(f"Error computing embedding: {e}")
        return None

def extract_entities(text, model=None):
    """
    Extract named entities from text using spaCy
    
    Args:
        text: Text to extract entities from
        model: Loaded spaCy model or None
        
    Returns:
        List of entity dictionaries with text, type, and count
    """
    if not text or len(text) < 50:
        return []
    
    # Load model if not provided
    if model is None:
        try:
            model = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            return []
    
    # Process text (limit to first 50K chars for performance)
    doc = model(text[:50000])
    
    # Extract entities
    entity_counts = {}
    for ent in doc.ents:
        # Skip very short entities and common false positives
        if len(ent.text) < 3 or ent.text.lower() in ["the", "this", "that", "these", "those"]:
            continue
            
        key = (ent.text, ent.label_)
        if key in entity_counts:
            entity_counts[key] += 1
        else:
            entity_counts[key] = 1
    
    # Convert to list format
    entities = [
        {"text": ent[0], "type": ent[1], "count": count}
        for ent, count in entity_counts.items()
    ]
    
    # Sort by count descending
    entities.sort(key=lambda x: x["count"], reverse=True)
    
    return entities[:100]  # Limit to top 100 entities

def enrich_from_rss(feed_url, transcript_path, meta=None, compute_embeddings=False):
    """
    Enrich metadata from RSS feed without re-transcription
    
    Args:
        feed_url: URL of the RSS feed
        transcript_path: Path to the transcript JSON file
        meta: Existing metadata dictionary or None
        compute_embeddings: Whether to compute embeddings and extract entities
        
    Returns:
        Enriched metadata dictionary
    """
    if meta is None:
        meta = {}
    
    # Parse feed
    logger.info(f"Parsing feed: {feed_url}")
    feed = feedparser.parse(feed_url)
    
    # Load transcript JSON
    with open(transcript_path) as f:
        transcript_data = json.load(f)
    
    # Get audio hash from filename
    audio_hash = os.path.basename(transcript_path).replace(".json", "")
    
    # Try to find matching item in feed by guid if we have it
    matching_item = None
    
    # If we have a guid, use it
    if "guid" in meta and meta["guid"]:
        matching_item = next((i for i in feed.entries if i.id == meta["guid"]), None)
    
    # If we couldn't find by guid, try matching by title
    if not matching_item and "episode" in meta and meta["episode"]:
        matching_item = next((i for i in feed.entries if i.title == meta["episode"]), None)
    
    # If still no match, we can't enrich from RSS
    if not matching_item:
        logger.warning(f"Could not find matching item in feed for {transcript_path}")
        return meta
    
    # Extract basic metadata
    meta.update({
        "podcast": feed.feed.title,
        "episode": matching_item.title,
        "guid": matching_item.id,
        "published": matching_item.get("published", ""),
        "episode_url": matching_item.get("link", ""),
        "audio_url": matching_item.enclosures[0]["href"] if matching_item.enclosures else "",
    })
    
    # Extract categories
    categories = []
    
    # Check entry-level tags
    if hasattr(matching_item, "tags") and matching_item.tags:
        categories = [tag.get("term", "") for tag in matching_item.tags if tag.get("term")]
    
    # Check iTunes categories
    if hasattr(matching_item, "itunes_categories"):
        categories.extend(matching_item.itunes_categories)
    
    # If no categories at entry level, try feed level
    if not categories:
        if hasattr(feed.feed, "tags") and feed.feed.tags:
            categories = [tag.get("term", "") for tag in feed.feed.tags if tag.get("term")]
        
        if hasattr(feed.feed, "itunes_categories"):
            categories.extend(feed.feed.itunes_categories)
    
    # Filter out duplicates while preserving order
    seen = set()
    meta["categories"] = [cat for cat in categories if cat and cat not in seen and not seen.add(cat)]
    
    # Extract rights
    meta["rights"] = {
        "copyright": matching_item.get("copyright") or feed.feed.get("copyright", ""),
        "explicit": (matching_item.get("itunes_explicit", "no").lower() in ("yes", "true", "1"))
    }
    
    # Try to extract hosts/guests
    # First look for podcast:person tags (modern feeds)
    hosts = []
    if hasattr(matching_item, "podcast_person"):
        for person in matching_item.podcast_person:
            hosts.append({
                "name": person.get("name", ""),
                "role": person.get("role", "host")
            })
    
    # Extract full transcript text
    transcript_text = ""
    if "text" in transcript_data:
        transcript_text = transcript_data["text"]
    elif "segments" in transcript_data:
        transcript_text = " ".join(seg.get("text", "") for seg in transcript_data["segments"] if "text" in seg)
    
    # If no hosts found, try NER on first segments
    if not hosts and "segments" in transcript_data:
        # Load spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
            
            # Get text from first 3 segments (intro usually mentions hosts)
            intro_text = " ".join(seg["text"] for seg in transcript_data["segments"][:3] 
                                if "text" in seg)
            
            # Run NER
            doc = nlp(intro_text)
            
            # Extract PERSON entities
            person_names = [e.text for e in doc.ents if e.label_ == "PERSON"]
            
            # Add as hosts
            hosts = [{"name": name, "role": "host"} for name in person_names]
            
        except Exception as e:
            logger.error(f"Error extracting hosts with spaCy: {e}")
    
    meta["hosts"] = hosts
    
    # Ensure we have an empty keywords list if it doesn't exist
    if "keywords" not in meta:
        meta["keywords"] = []
    
    # Optional: Compute embeddings and extract entities
    if compute_embeddings and transcript_text:
        try:
            # Compute embedding
            embedding = compute_embedding(transcript_text)
            if embedding is not None:
                # Generate embedding file path
                embedding_dir = os.path.join(os.path.dirname(transcript_path), "embeddings")
                os.makedirs(embedding_dir, exist_ok=True)
                embedding_path = os.path.join(embedding_dir, f"{audio_hash}.npy")
                
                # Save embedding
                np.save(embedding_path, embedding)
                
                # Add embedding path to metadata
                meta["embedding_path"] = os.path.relpath(embedding_path, os.path.dirname(transcript_path))
                logger.info(f"Created embedding for {audio_hash}")
            
            # Extract entities
            nlp = spacy.load("en_core_web_sm")
            entities = extract_entities(transcript_text, nlp)
            if entities:
                meta["entities"] = entities
                logger.info(f"Extracted {len(entities)} entities for {audio_hash}")
        
        except Exception as e:
            logger.error(f"Error generating embeddings or entities: {e}")
    
    # In enrich_meta, after categories are collected:
    # meta["categories"] = [cat.lower() for cat in categories if cat and cat not in seen and not seen.add(cat)]
    
    return meta

def process_transcripts(config, feeds_mapping=None, compute_embeddings=False):
    """
    Process all transcript files and rebuild metadata
    
    Args:
        config: Dictionary with configuration
        feeds_mapping: Dictionary mapping podcast names to feed URLs
        compute_embeddings: Whether to compute embeddings and extract entities
    """
    if feeds_mapping is None:
        feeds_mapping = {}
    
    # Find all transcript files
    transcript_dir = config.get("transcript_path", "data/transcripts")
    transcript_files = glob.glob(os.path.join(transcript_dir, "*.json"))
    
    if not transcript_files:
        logger.error(f"No transcript files found in {transcript_dir}")
        return
    
    logger.info(f"Found {len(transcript_files)} transcript files to process")
    
    # Process each transcript
    for transcript_path in tqdm(transcript_files):
        try:
            # Load transcript
            with open(transcript_path) as f:
                transcript_data = json.load(f)
            
            # Get existing metadata
            meta = transcript_data.get("meta", {})
            
            # If we have a podcast name and it's in the mapping, use that feed URL
            feed_url = None
            if "podcast" in meta and meta["podcast"] in feeds_mapping:
                feed_url = feeds_mapping[meta["podcast"]]
            
            # If we have a feed URL, enrich from RSS
            if feed_url:
                meta = enrich_from_rss(feed_url, transcript_path, meta, compute_embeddings)
                
                # Update transcript with enhanced metadata
                transcript_data["meta"] = meta
                
                # Save updated transcript
                with open(transcript_path, "w") as f:
                    json.dump(transcript_data, f, indent=2)
                
                logger.info(f"Updated metadata for {os.path.basename(transcript_path)}")
            else:
                logger.warning(f"No feed URL for {meta.get('podcast', 'unknown')}")
        
        except Exception as e:
            logger.error(f"Error processing {transcript_path}: {e}")

def load_feed_mapping(config_path):
    """Load feed mapping from config file"""
    import yaml
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        feeds = config.get("feeds", [])
        return {feed["title"]: feed["feed_url"] for feed in feeds if "title" in feed and "feed_url" in feed}
    
    except Exception as e:
        logger.error(f"Error loading feed mapping from {config_path}: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Rebuild podcast metadata without re-transcription")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--feeds", help="Path to YAML file with feed mapping")
    parser.add_argument("--embeddings", action="store_true", help="Compute embeddings and extract entities")
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load feed mapping
    feeds_mapping = {}
    if args.feeds:
        feeds_mapping = load_feed_mapping(args.feeds)
    elif "feeds" in config:
        # Use feeds from main config
        feeds_mapping = {feed["title"]: feed["feed_url"] for feed in config["feeds"]
                         if "title" in feed and "feed_url" in feed}
    
    # Process transcripts
    process_transcripts(config, feeds_mapping, args.embeddings)
    
    logger.info("Done rebuilding metadata")

if __name__ == "__main__":
    main() 