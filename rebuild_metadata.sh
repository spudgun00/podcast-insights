#!/bin/bash
set -e

echo "=== Podcast Insights Metadata Rebuild ==="
echo "This script will restore missing metadata fields without re-transcription"

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔄 Setting up virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
else
    echo "✅ Using existing virtual environment: $VIRTUAL_ENV"
fi

# Install requirements
echo "🔄 Installing required packages..."
python3.11 -m pip install --no-cache-dir --force-reinstall -r rebuild_meta_requirements.txt

# Install spaCy model
echo "🔄 Installing spaCy English model..."
python3.11 install_spacy_model.py

# Run metadata rebuild
echo "🔄 Rebuilding metadata..."
python3.11 rebuild_meta.py "$@"

echo "✅ Metadata rebuild process complete!" 