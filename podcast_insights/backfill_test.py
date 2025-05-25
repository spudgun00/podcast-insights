import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ... existing code ... 

# Make sure to import slugify and other necessary functions at the top of the file:
# from .meta_utils import slugify, generate_podcast_slug # (or wherever slugify is)
# import os
# from pathlib import Path
# import logging # Already added, but ensure logger is available
# logger = logging.getLogger(__name__)

from pathlib import Path
import os # Though not directly used in the snippet, it's often useful.
from .meta_utils import slugify_title as slugify # Assuming slugify_title is the intended slugify

logger = logging.getLogger(__name__)

# Assume episode_details, guid, podcast_slug, title are defined
# Placeholder initializations for testing:
episode_details = {"title": "Test Episode Title"}
guid = "test-guid-123"
podcast_slug = "test-podcast-slug"
# title = "Test Episode Title" # This was mentioned in comments but episode_details.get('title') is used

# --- Transcript File Path (A-1) ---
# OLD LOGIC (example, will be replaced):
# transcript_file_name = f"{guid}_{some_hash_or_full_title}.json"
# transcript_path = Path("data/transcripts") / podcast_slug / transcript_file_name

# NEW LOGIC for transcript path:
title_slug_truncated = slugify(episode_details.get('title', 'unknown-episode'))[:80]
transcript_filename = f"{title_slug_truncated}_{guid}.json"
transcript_path_local = Path("data/transcripts") / podcast_slug / transcript_filename
transcript_path_local.parent.mkdir(parents=True, exist_ok=True)
logger.info(f"Set transcript path for {guid} to: {transcript_path_local}")

# When calling transcribe.py script:
# cmd = [
# sys.executable, "podcast_insights/transcribe.py",
# "--file", str(audio_file_path),
# "--output", str(transcript_path_local), # <--- THIS IS THE KEY CHANGE
# ... other args ...
# ]
# logger.info(f"Running transcription: {' '.join(cmd)}")
# subprocess.run(cmd, check=True)


# --- Segments File Path (C-6, C-1 related) ---
# This path is where enrich_meta will be told to SAVE the segments IF they are extracted
# or where other parts of the system will EXPECT to find them.
segments_filename = f"{guid}.json" # C-1: Segments file uses GUID
segments_path_local = Path("data/segments") / podcast_slug / segments_filename
segments_path_local.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists for writing
logger.info(f"Set expected segments path for {guid} to: {segments_path_local}")

# This segments_path_local would then be passed to enrich_meta or used when updating meta.
# e.g., meta['segments_path'] = str(segments_path_local)


# --- Meta File Path (A-1) ---
# OLD LOGIC (example):
# meta_filename_local = f"meta_{guid}.json"
# meta_path_local = Path("data/meta") / podcast_slug / meta_filename_local

# NEW LOGIC for local meta file path:
# title_slug_truncated is already defined above
meta_filename_local = f"meta_{title_slug_truncated}_{guid}.json"
local_meta_path = Path("data/meta") / podcast_slug / meta_filename_local
local_meta_path.parent.mkdir(parents=True, exist_ok=True)
logger.info(f"Set local meta path for {guid} to: {local_meta_path}")

# The S3 path for meta was already guid/meta.json, which is fine.

# ... existing code ...

# --- Meta Cleanup (A-2) ---
# Ensure cleanup_old_meta is imported: from .meta_utils import cleanup_old_meta
# In the main processing loop, likely within a try/finally for each episode:
# try:
    # ... main processing for an episode ...
    # meta_to_save = enrich_meta(...)
    # with open(local_meta_path, "w") as f:
    #     json.dump(meta_to_save, f, indent=2)
    # logger.info(f"Saved final metadata to {local_meta_path} for {guid}")
# finally:
    # if 'local_meta_path' in locals() and 'guid' in locals() and 'podcast_slug' in locals():
    #     try:
    #         logger.info(f"Attempting to clean up old meta files for {guid} in {local_meta_path.parent}")
    #         cleanup_old_meta(local_meta_path.parent, guid, title_slug_truncated) # Adjusted call
    #         logger.info(f"Meta cleanup executed for {guid}.")
    #     except Exception as e_clean:
    #         logger.error(f"Error during meta cleanup for {guid}: {e_clean}")
    # else:
    #     logger.warning(f"Skipping meta cleanup for {guid} due to missing path/guid/slug info.")

# ... existing code ... 