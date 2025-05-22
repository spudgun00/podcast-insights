#!/usr/bin/env python3
"""
Simple script to download required spaCy models for the rebuild_meta.py script.
"""
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s | %(levelname)7s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    """Download required spaCy models"""
    try:
        logger.info("Downloading spaCy English model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        logger.info("✅ Successfully installed spaCy English model")
        return 0
    except Exception as e:
        logger.error(f"❌ Error installing spaCy model: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 