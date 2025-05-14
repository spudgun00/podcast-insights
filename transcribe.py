#!/usr/bin/env python3
"""
Transcribe all MP3s that lack a JSON transcript
"""

import os
import glob
import yaml
import json
import subprocess
import sys

# Check if ffmpeg is installed
try:
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    print("✅ ffmpeg is installed")
except (subprocess.SubprocessError, FileNotFoundError):
    print("❌ Error: ffmpeg is not installed or not in PATH")
    print("Please install ffmpeg:")
    print("macOS: brew install ffmpeg")
    print("Ubuntu/Debian: sudo apt-get install ffmpeg")
    sys.exit(1)

# Load configuration
CFG = yaml.safe_load(open("config.yaml"))
AUDIO_ROOT = CFG["download_path"]
OUT_ROOT = CFG["transcript_path"]
os.makedirs(OUT_ROOT, exist_ok=True)

# Now import whisperx after ffmpeg check
try:
    import whisperx
    print("✅ whisperx is installed")
except ImportError:
    print("❌ Error: whisperx is not installed")
    print("Please install: pip install whisperx")
    sys.exit(1)

# Configure model for CPU
print("Loading whisperx model...")
try:
    model = whisperx.load_model(
        CFG["model_size"], 
        device="cpu",
        compute_type="int8"  # Use int8 for CPU compatibility
    )
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    sys.exit(1)

def transcript_path(mp3_path: str) -> str:
    base = os.path.splitext(os.path.basename(mp3_path))[0]
    return os.path.join(OUT_ROOT, f"{base}.json")

# Find all MP3 files
mp3_files = glob.glob(os.path.join(AUDIO_ROOT, "*.mp3"))
if not mp3_files:
    print("No MP3 files found in", AUDIO_ROOT)
    exit(0)

print(f"Found {len(mp3_files)} MP3 files to process")

# Process each file
successful_transcriptions = 0
for mp3 in mp3_files:
    out_json = transcript_path(mp3)
    if os.path.exists(out_json):
        print(f"✔ Already transcribed {os.path.basename(mp3)}")
        continue

    print(f"📝 Transcribing {os.path.basename(mp3)}")
    try:
        # Transcribe with error handling
        result = model.transcribe(mp3)
        
        # Save result
        with open(out_json, "w") as f:
            json.dump(result, f)
        
        print(f"✅ Successfully transcribed {os.path.basename(mp3)}")
        successful_transcriptions += 1
        
    except Exception as e:
        print(f"❌ Error transcribing {os.path.basename(mp3)}: {str(e)}")
        import traceback
        print(traceback.format_exc())

print(f"\nSummary: Transcribed {successful_transcriptions} of {len(mp3_files)} files")
