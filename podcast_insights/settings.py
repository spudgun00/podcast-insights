# podcast_insights/settings.py
# ------------------------------------------------------------------
# GLOBAL STORAGE SETTINGS — CHANGE THESE ONLY WITH A DB MIGRATION!
# ------------------------------------------------------------------

BUCKET          = "my-podcast-data"          # <- your S3 bucket name
BASE_PREFIX     = "processed_episodes/"      # trailing slash required
LAYOUT          = "flat-guid"                # locked choice: "flat-guid"

def layout_fn(guid: str, *_ignored) -> str:
    """
    Returns the S3 prefix folder for a given episode.
    For flat-guid layout that is just '<BASE_PREFIX><guid>/'.
    """
    if not guid:
        raise ValueError("GUID cannot be empty for layout_fn")
    return f"{BASE_PREFIX}{guid}/" 