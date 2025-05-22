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
import re

import feedparser
import spacy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from podcast_insights.meta_utils import enrich_meta

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s | %(levelname)7s | %(message)s")
logger = logging.getLogger(__name__)

def process_transcripts(config, feeds_mapping=None, nlp_model=None, st_model=None, compute_embeddings_flag=False):
    """
    Process all transcripts in the configured directory.
    Calls the centralized enrich_meta function from podcast_insights.meta_utils.
    """
    transcript_dir = Path(config.get("transcript_path", "data/transcripts"))
    # output_dir is not strictly needed if we update in place, but good for config
    # output_dir = Path(config.get("output_path", "data/processed_meta")) 
    base_data_dir = Path(config.get("base_data_path", "data")) # For entities/embeddings

    # output_dir.mkdir(parents=True, exist_ok=True) # Only if saving to a new location
    
    if not feeds_mapping:
        logger.warning("No feed mapping provided. RSS-based enrichment will be limited.")
        feeds_mapping = {}

    transcript_files = glob.glob(str(transcript_dir / "*.json"))
    if not transcript_files:
        logger.warning(f"No transcript files found in {transcript_dir}")
        return

    logger.info(f"Found {len(transcript_files)} transcripts to process.")

    for transcript_file_path_str in tqdm(transcript_files, desc="Processing transcripts"):
        transcript_file_path = Path(transcript_file_path_str)
        try:
            with open(transcript_file_path, 'r', encoding='utf-8') as f: # Added encoding
                original_transcript_data = json.load(f)
            
            current_meta = original_transcript_data.get("meta", {})

            transcript_text = ""
            if "text" in original_transcript_data:
                transcript_text = original_transcript_data["text"]
            elif "segments" in original_transcript_data:
                transcript_text = " ".join(seg.get("text", "").strip() for seg in original_transcript_data["segments"] if seg.get("text"))
            else:
                logger.warning(f"No parsable text found in {transcript_file_path}. Skipping text-based enrichment.")
                # continue # Decide if to skip or proceed with limited data

            # Prepare tech_info dict for enrich_meta
            # enrich_meta now takes most of these directly or calculates them.
            # We primarily need to pass what's intrinsic to the *transcription process output* or *audio file itself*
            # if not already in current_meta.
            audio_hash_from_filename = transcript_file_path.stem.split('.')[0] # Example: guid.mp3 -> guid
            
            tech_info = {
                # Base essential IDs that enrich_meta might need if not in feed entry
                "audio_hash": current_meta.get("audio_hash", audio_hash_from_filename),
                "transcript_path": str(transcript_file_path.resolve()),
                "download_path": current_meta.get("download_path"), # enrich_meta does not set this

                # These are often part of original transcript output or existing meta
                "supports_timestamp": current_meta.get("supports_timestamp", original_transcript_data.get("word_timestamps") is not False),
                "speech_music_ratio": current_meta.get("speech_music_ratio", 0.0),
                "transcript_length": len(transcript_text), # enrich_meta calculates this from text too
                "sample_rate_hz": current_meta.get("sample_rate_hz"), 
                "bitrate_kbps": current_meta.get("bitrate_kbps"), 
                "duration_sec": current_meta.get("duration_sec"), 
                "avg_confidence": current_meta.get("avg_confidence", original_transcript_data.get("confidence")), # whisperx might put it top-level
                "wer_estimate": current_meta.get("wer_estimate"),
                "segment_count": len(original_transcript_data.get("segments", [])),
                "chunk_count": original_transcript_data.get("chunk_count", 1), # If applicable
                # audio_url, episode_url etc. will be pulled from feed by enrich_meta
            }
            # enrich_meta also expects 'guid' in its 'entry' argument.
            # It has fallback for missing guid in entry, but better to provide.
            # It also expects 'title' in 'entry'.

            # --- Find corresponding feed entry --- 
            feed_title_guess = current_meta.get("podcast")
            if not feed_title_guess: # Try to infer from path if possible, very rough
                # e.g. data/transcripts/the_showname_series_guid.mp3.json -> the_showname_series
                parts = transcript_file_path.stem.split('_') 
                if len(parts) > 1: # Avoid using just GUID as show name
                    feed_title_guess = "_".join(parts[:-1])


            feed_url = feeds_mapping.get(feed_title_guess) 
            feed_parsed = None
            entry_data_for_enrich = {} # Default to empty dict

            if feed_url:
                logger.debug(f"Parsing feed {feed_url} for {transcript_file_path.name}")
                feed_parsed = feedparser.parse(feed_url)
                if not feed_parsed or feed_parsed.bozo:
                    logger.warning(f"Failed to parse feed or bad feed data for {feed_url}. Bozo: {feed_parsed.bozo_exception if feed_parsed and hasattr(feed_parsed, 'bozo_exception') else 'N/A'}")
                    feed_parsed = {"feed": {"title": feed_title_guess or "Unknown Podcast"}, "entries": []} # Mock structure
            else:
                logger.warning(f"No feed_url found in mapping for podcast title guess: '{feed_title_guess}' from file {transcript_file_path.name}")
                feed_parsed = {"feed": {"title": feed_title_guess or "Unknown Podcast"}, "entries": []} # Mock structure
            
            # Try to find the entry in the parsed feed
            guid_to_find = current_meta.get("guid", tech_info["audio_hash"]) # Use audio_hash as fallback GUID
            title_to_find = current_meta.get("episode")

            if feed_parsed and feed_parsed.entries:
                if guid_to_find:
                    entry_data_for_enrich = next((e for e in feed_parsed.entries if e.get('id') == guid_to_find), None)
                if not entry_data_for_enrich and title_to_find: # Try by title if GUID match failed
                    entry_data_for_enrich = next((e for e in feed_parsed.entries if e.get('title') == title_to_find), None)
            
            if not entry_data_for_enrich:
                logger.warning(f"Could not find matching entry in feed {feed_url or 'N/A'} for GUID '{guid_to_find}' or title '{title_to_find}'. Proceeding with minimal entry data.")
                # Create a minimal entry dict for enrich_meta
                entry_data_for_enrich = {
                    "title": title_to_find or "Unknown Episode", 
                    "id": guid_to_find # enrich_meta needs an ID for GUID
                    # enrich_meta will fill in other fields like link, published from this if found, or use N/A
                }
            
            # --- Call Centralized enrich_meta --- 
            updated_meta = enrich_meta(
                entry=entry_data_for_enrich, 
                feed_title=feed_parsed.feed.get("title", feed_title_guess or "Unknown Podcast"),
                feed_url=feed_url if feed_url else "",
                tech=tech_info, # Pass the prepared tech dictionary
                transcript_text=transcript_text,
                feed=feed_parsed, # Pass the full feedparser object
                nlp_model=nlp_model,
                st_model=st_model,
                base_data_dir=base_data_dir,
                perform_caching=compute_embeddings_flag
            )

            original_transcript_data["meta"] = updated_meta

            # Save the updated transcript JSON file (overwriting existing)
            output_transcript_path = transcript_file_path 
            with open(output_transcript_path, 'w', encoding='utf-8') as f: # Added encoding
                json.dump(original_transcript_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Processed and saved updated meta to {output_transcript_path}")

        except Exception as e:
            logger.error(f"Failed to process {transcript_file_path}: {e}", exc_info=True)
            continue

def load_feed_mapping(config_path_str):
    """Loads feed mapping from a JSON or YAML file."""
    config_path = Path(config_path_str)
    if not config_path.exists():
        logger.warning(f"Feed mapping file {config_path} not found. Returning empty map.")
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f: # Added encoding
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                mapping = yaml.safe_load(f)
                logger.info(f"Loaded YAML feed mapping from {config_path}")
                return mapping
            elif config_path.suffix.lower() == '.json':
                mapping = json.load(f)
                logger.info(f"Loaded JSON feed mapping from {config_path}")
                return mapping
            else:
                logger.error(f"Unknown feed mapping file format: {config_path}. Needs .json or .yaml")
                return {}
    except ImportError:
        logger.error("PyYAML library not found. Please install it to load YAML feed mapping (pip install PyYAML).")
        return {}
    except Exception as e:
        logger.error(f"Error loading feed mapping from {config_path}: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Rebuild podcast metadata.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML or JSON file")
    parser.add_argument("--compute_embeddings", action="store_true", help="Compute and save embeddings and entities")
    # The --feeds argument is replaced by feed_mapping_path in config.yaml
    args = parser.parse_args()

    config_data = {}
    config_file_path = Path(args.config)
    if not config_file_path.exists():
        logger.error(f"Config file {config_file_path} not found.")
        sys.exit(1)
    
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f: # Added encoding
            if config_file_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml 
                config_data = yaml.safe_load(f)
                logger.info(f"Loaded YAML config from {config_file_path}")
            elif config_file_path.suffix.lower() == '.json':
                config_data = json.load(f)
                logger.info(f"Loaded JSON config from {config_file_path}")
            else:
                logger.error(f"Unsupported config file format: {config_file_path}. Must be .json or .yaml.")
                sys.exit(1)
    except ImportError:
        logger.error("PyYAML not found. Please install for YAML config (pip install PyYAML).")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading config {config_file_path}: {e}")
        sys.exit(1)

    feeds_mapping = load_feed_mapping(config_data.get("feed_mapping_path", "feed_mapping.json"))

    nlp = None
    st = None
    if args.compute_embeddings:
        logger.info("Loading NLP and Sentence Transformer models as --compute_embeddings is enabled...")
        try:
            nlp_model_name = config_data.get("spacy_model", "en_core_web_sm")
            nlp = spacy.load(nlp_model_name)
            logger.info(f"SpaCy model '{nlp_model_name}' loaded.")
        except Exception as e:
            logger.error(f"Failed to load SpaCy model: {e}. Entity extraction will be skipped.")
        
        try:
            st_model_name = config_data.get("sentence_transformer_model", "all-MiniLM-L6-v2")
            st = SentenceTransformer(st_model_name)
            logger.info(f"SentenceTransformer model '{st_model_name}' loaded.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}. Embeddings will be skipped.")
    else:
        logger.info("--compute_embeddings flag not set. Skipping model loading and caching.")

    process_transcripts(
        config_data, 
        feeds_mapping,
        nlp_model=nlp,
        st_model=st,
        compute_embeddings_flag=args.compute_embeddings
    )
    logger.info("Finished rebuilding metadata.")

if __name__ == "__main__":
    main() 