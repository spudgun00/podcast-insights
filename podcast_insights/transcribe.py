#!/usr/bin/env python3
"""
Transcription module for podcast-insights

This module provides functions for transcribing audio files using WhisperX.
It now also handles inline generation of NER entities and sentence embeddings.
"""

import os
import sys
import json
import argparse
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any # Added typing imports

import whisperx
import spacy
from sentence_transformers import SentenceTransformer

# Import helper functions from meta_utils
# Assuming meta_utils.py is in the same directory or Python path
try:
    from .meta_utils import _generate_spacy_entities_file, _generate_sentence_embedding_file
except ImportError:
    # Fallback for direct execution if podcast_insights is not installed as a package
    from meta_utils import _generate_spacy_entities_file, _generate_sentence_embedding_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load NLP Models Globally (once per module load) ---
# These can be made configurable, e.g., via environment variables or a config file read here
NLP_MODEL_NAME = os.environ.get("SPACY_MODEL", "en_core_web_sm")
ST_MODEL_NAME = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
DEFAULT_BASE_DATA_DIR = Path(os.environ.get("DEFAULT_BASE_DATA_DIR", "data"))

NLP_MODEL: Optional[spacy.Language] = None
ST_MODEL: Optional[SentenceTransformer] = None
MODELS_LOADED_SUCCESSFULLY = False

try:
    logger.info(f"Loading SpaCy model: {NLP_MODEL_NAME}...")
    NLP_MODEL = spacy.load(NLP_MODEL_NAME)
    logger.info("SpaCy model loaded successfully.")
    logger.info(f"Loading SentenceTransformer model: {ST_MODEL_NAME}...")
    ST_MODEL = SentenceTransformer(ST_MODEL_NAME)
    logger.info("SentenceTransformer model loaded successfully.")
    MODELS_LOADED_SUCCESSFULLY = NLP_MODEL is not None and ST_MODEL is not None
except Exception as e:
    logger.error(f"Failed to load NLP/SentenceTransformer models globally: {e}. Caching will be disabled by default if models are not passed explicitly.")
    NLP_MODEL = None
    ST_MODEL = None
# --- End Model Loading ---

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("ffmpeg is not installed or not in PATH")
        return False

def load_whisper_model(model_size="base", device="auto", compute_type="auto"):
    """
    Load whisperx model with GPU detection
    
    device="auto": cuda if available else cpu
    compute_type="auto": float16 on GPU, int8 on CPU
    """
    import torch
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
        
    logger.info(f"Loading Whisper model on device={device} with compute_type={compute_type}")
    if device == "cuda":
        logger.info(f"CUDA devices available: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            gpu_props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU memory: {gpu_props.total_memory // 1024**3} GB")
            logger.info(f"CUDA version: {torch.version.cuda}")
    
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

def transcribe_audio(
    audio_file: str,
    model_size: str = "base", 
    output_file: Optional[str] = None, 
    compute_type: str = "int8",
    metadata: Optional[Dict[str, Any]] = None,
    perform_entity_embedding_caching: bool = True,
    nlp_model_override: Optional[spacy.Language] = None,
    st_model_override: Optional[SentenceTransformer] = None,
    base_data_dir_override: Optional[Path] = None,
    enable_word_timestamps: bool = False,
    podcast_slug: Optional[str] = None
) -> Dict[str, Any]:
    """
    Transcribe an audio file using WhisperX and optionally cache entities/embeddings.
    
    Args:
        audio_file: Path to audio file
        model_size: Whisper model size
        output_file: Path to save output JSON
        compute_type: Compute type for the whisper model
        metadata: Dictionary of metadata (e.g., containing GUID) to include in output
        perform_entity_embedding_caching: Whether to perform inline caching.
        nlp_model_override: Optionally pass a loaded SpaCy model.
        st_model_override: Optionally pass a loaded SentenceTransformer model.
        base_data_dir_override: Optionally specify base directory for entities/embeddings.
        enable_word_timestamps: Whether to perform word-level alignment.
        podcast_slug: Optional podcast slug for path construction during caching.
        
    Returns:
        Dictionary with transcription results and paths to cached files if generated.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is required for transcription")
    
    model = load_whisper_model(model_size=model_size, compute_type=compute_type)
    
    # Initialize paths for cached files to None
    raw_entities_output_path_val: Optional[str] = None
    raw_embedding_output_path_val: Optional[str] = None

    try:
        logger.info(f"Transcribing {audio_file} with model_size={model_size}, compute_type={compute_type}")
        audio = whisperx.load_audio(str(audio_file))
        result = model.transcribe(audio)

        if enable_word_timestamps:
            logger.info("Transcription complete. Performing word-level alignment.")
            # Align whisper output
            # TODO: Add error handling for alignment model loading
            # TODO: Consider making language_code configurable or detecting it more robustly
            model_a, metadata_a = whisperx.load_align_model(language_code=result.get("language", "en"), device=model.device)
            aligned_result = whisperx.align(result["segments"], model_a, metadata_a, audio, model.device, return_char_alignments=False)
            logger.info("Word-level alignment complete.")
            final_result_payload = aligned_result
        else:
            logger.info("Transcription complete. Word-level alignment is disabled.")
        final_result_payload = result 

        # --- Inline Entity and Embedding Caching --- 
        if perform_entity_embedding_caching and MODELS_LOADED_SUCCESSFULLY or (nlp_model_override and st_model_override):
            transcript_text = ""
            if "text" in result:
                transcript_text = result["text"]
            elif "segments" in result:
                transcript_text = " ".join(seg.get("text", "").strip() for seg in result["segments"] if seg.get("text"))

            guid = metadata.get("guid") if metadata else None
            current_nlp_model = nlp_model_override if nlp_model_override else NLP_MODEL
            current_st_model = st_model_override if st_model_override else ST_MODEL
            current_base_data_dir = base_data_dir_override if base_data_dir_override else DEFAULT_BASE_DATA_DIR

            entities_path = None
            embedding_path = None

            # Get segments for embedding
            segments_for_embedding = result.get("segments", [])
            segment_texts_for_embedding = [s.get('text', '').strip() for s in segments_for_embedding if s.get('text', '').strip()]

            if transcript_text and guid and current_nlp_model: # NER still uses full text
                logger.info(f"Performing inline NER caching for GUID: {guid} in dir {current_base_data_dir}")
                entities_path = _generate_spacy_entities_file(
                    transcript_text, guid, current_base_data_dir, current_nlp_model,
                    podcast_slug=podcast_slug
                )
                if entities_path:
                    raw_entities_output_path_val = entities_path
            else:
                logger.warning(f"Skipping inline NER caching for {audio_file} due to missing text, GUID, or NLP model.")

            if segment_texts_for_embedding and guid and current_st_model: # Embeddings use segment texts
                logger.info(f"Performing inline embedding caching for GUID: {guid} using {len(segment_texts_for_embedding)} segments in dir {current_base_data_dir}")
                embedding_path = _generate_sentence_embedding_file(
                    segment_texts_for_embedding, guid, current_base_data_dir, current_st_model,
                    podcast_slug=podcast_slug
                )
                if embedding_path:
                    raw_embedding_output_path_val = embedding_path
            else:
                logger.warning(f"Skipping inline embedding caching for {audio_file} due to missing segment texts, GUID, or ST model.")
            
            # Add paths to the metadata if they were generated
            if metadata is None: metadata = {} # Ensure metadata dict exists
            if "meta" not in final_result_payload: final_result_payload["meta"] = metadata
            
            # Ensure final_result_payload["meta"] is a dictionary
            if not isinstance(final_result_payload.get("meta"), dict):
                 final_result_payload["meta"] = {}
            
            if entities_path:
                final_result_payload["meta"]["entities_path"] = entities_path
            if embedding_path:
                final_result_payload["meta"]["embedding_path"] = embedding_path
        else:
            logger.info(f"Inline entity/embedding caching skipped for {audio_file} (flag disabled or models not loaded).")
            if metadata and "meta" not in final_result_payload: # still attach original meta if passed
                 final_result_payload["meta"] = metadata
            elif metadata and not isinstance(final_result_payload.get("meta"), dict): # if meta exists but not dict
                 final_result_payload["meta"] = {}


        # Attach original metadata if not already part of the payload's meta section
        if metadata:
            if "meta" not in final_result_payload:
                final_result_payload["meta"] = metadata
            elif isinstance(final_result_payload.get("meta"), dict):
                # Merge, giving preference to keys already in final_result_payload["meta"] from caching
                original_meta_copy = metadata.copy()
                original_meta_copy.update(final_result_payload["meta"])
                final_result_payload["meta"] = original_meta_copy
            # else: it means final_result_payload["meta"] was set to something else, which shouldn't happen here.

        # Ensure top-level "text" field exists in the final payload
        if "text" not in final_result_payload or not final_result_payload["text"]:
            if "segments" in final_result_payload and final_result_payload["segments"]:
                full_text = " ".join(seg.get("text", "").strip() for seg in final_result_payload["segments"] if seg.get("text"))
                final_result_payload["text"] = full_text
                logger.info("Constructed top-level 'text' field from segments.")
            else:
                logger.warning("Could not construct top-level 'text' field: 'segments' are missing or empty.")
                final_result_payload["text"] = "" # Ensure the key exists, even if empty

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding='utf-8') as f:
                json.dump(final_result_payload, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved transcript to {output_file}")
        
        # ADDED: Populate main return dictionary with cached paths
        final_return_dict = final_result_payload.copy() # Start with transcription payload
        if raw_entities_output_path_val:
            final_return_dict["raw_entities_output_path"] = raw_entities_output_path_val
        if raw_embedding_output_path_val:
            final_return_dict["raw_embedding_output_path"] = raw_embedding_output_path_val
            
        return final_return_dict
        
    except Exception as e:
        logger.error(f"Error transcribing {audio_file}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio file with WhisperX, optionally caching entities/embeddings.")
    parser.add_argument("--file", required=True, help="Path to audio file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--model_size", default="base", help="Whisper model size")
    parser.add_argument("--compute_type", default="int8", help="Compute type for the whisper model")
    parser.add_argument("--metadata_json", default="{}", help="Metadata JSON string (e.g., '{\"guid\": \"your-guid\"}')")
    parser.add_argument("--enable_caching", action="store_true", help="Enable inline entity and embedding caching.")
    parser.add_argument("--base_data_dir", default=None, help=f"Override base data directory for caching (default: {DEFAULT_BASE_DATA_DIR})")
    parser.add_argument("--word_timestamps", action="store_true", help="Enable word-level timestamps in transcription.") # Re-added
    parser.add_argument("--podcast_slug", default=None, help="Podcast slug for path construction during caching.") # ADDED
    
    args = parser.parse_args()
    
    try:
        metadata_arg = json.loads(args.metadata_json) if args.metadata_json else {}
    except json.JSONDecodeError:
        logger.error("Invalid metadata_json string.")
        return 1
    
    base_data_path_override = Path(args.base_data_dir) if args.base_data_dir else None

    try:
        transcribe_audio(
            args.file, 
            model_size=args.model_size,
            output_file=args.output,
            compute_type=args.compute_type,
            metadata=metadata_arg,
            perform_entity_embedding_caching=args.enable_caching,
            # Global models NLP_MODEL, ST_MODEL are used by default if not overridden here
            base_data_dir_override=base_data_path_override,
            enable_word_timestamps=args.word_timestamps, # Re-added
            podcast_slug=args.podcast_slug # ADDED
        )
        return 0
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 