#!/usr/bin/env python3
"""
Transcription module for podcast-insights

This module provides functions for transcribing audio files using WhisperX.
It's designed to be imported as a module or called via command line.
"""

import os
import sys
import json
import argparse
import logging
import tempfile
import subprocess
from pathlib import Path

import whisperx

logger = logging.getLogger(__name__)

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("ffmpeg is not installed or not in PATH")
        return False

def load_whisper_model(model_size="base", device="cpu", compute_type="int8"):
    """Load whisperx model with appropriate settings"""
    try:
        model = whisperx.load_model(
            model_size, 
            device=device,
            compute_type=compute_type
        )
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def transcribe_audio(audio_file, model_size="base", output_file=None, vad_filter=True, metadata=None):
    """
    Transcribe an audio file using WhisperX
    
    Args:
        audio_file: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        output_file: Path to save output JSON
        vad_filter: Whether to use VAD filtering
        metadata: Dictionary of metadata to include in output
        
    Returns:
        Dictionary with transcription results
    """
    # Check dependencies
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is required for transcription")
    
    # Load model
    model = load_whisper_model(model_size)
    
    try:
        # Transcribe audio
        logger.info(f"Transcribing {audio_file}")
        audio = whisperx.load_audio(str(audio_file))
        
        # Perform initial transcription (segment-level timestamps)
        result = model.transcribe(audio)
        logger.info("Transcription complete. Word-level alignment is currently disabled.")

        # --- WORD-LEVEL ALIGNMENT DISABLED --- 
        # language_code = result.get("language", "en") 
        # try:
        #     model_a, metadata_a = whisperx.load_align_model(language_code=language_code, device="cpu") 
        # except Exception as e:
        #     logger.warning(f"Failed to load alignment model for language '{language_code}'. Error: {e}")
        #     # Fallback: use original result without alignment if alignment model fails
        #     aligned_result = result 
        # else:
        #     logger.info(f"Aligning transcript for {audio_file} (language: {language_code})")
        #     try:
        #         aligned_result = whisperx.align(result["segments"], model_a, metadata_a, audio, "cpu", return_char_alignments=False)
        #     except Exception as e:
        #         logger.warning(f"Failed to align transcript for {audio_file}. Error: {e}")
        #         aligned_result = result # Fallback to unaligned result
        # final_result_payload = aligned_result
        # --- END OF WORD-LEVEL ALIGNMENT DISABLED ---
        
        final_result_payload = result # Use the direct output of model.transcribe()
        
        if metadata:
            final_result_payload["meta"] = metadata # Attach original metadata passed in
        
        # Save to output file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(final_result_payload, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved transcript to {output_file}")
        
        return final_result_payload
        
    except Exception as e:
        logger.error(f"Error transcribing {audio_file}: {str(e)}")
        raise

def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(description="Transcribe audio file with WhisperX")
    parser.add_argument("--file", required=True, help="Path to audio file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--model_size", default="base", help="Whisper model size")
    parser.add_argument("--vad_filter", default="True", help="Use VAD filtering")
    parser.add_argument("--metadata", default="{}", help="Metadata JSON string")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s | %(levelname)7s | %(message)s")
    
    # Parse metadata
    try:
        metadata = json.loads(args.metadata) if args.metadata else {}
    except json.JSONDecodeError:
        logger.error("Invalid metadata JSON")
        return 1
    
    # Convert string to boolean
    vad_filter = args.vad_filter.lower() in ("true", "yes", "1")
    
    # Run transcription
    try:
        transcribe_audio(
            args.file, 
            model_size=args.model_size,
            output_file=args.output,
            vad_filter=vad_filter,
            metadata=metadata
        )
        return 0
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 