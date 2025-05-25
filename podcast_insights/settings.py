# podcast_insights/settings.py
# ------------------------------------------------------------------
# GLOBAL STORAGE SETTINGS — CHANGE THESE ONLY WITH A DB MIGRATION!
# ------------------------------------------------------------------

BUCKET          = "my-podcast-data"          # <- your S3 bucket name
BASE_PREFIX     = "processed_episodes/"      # trailing slash required
LAYOUT          = "podcast-guid"                # Changed from "flat-guid"

def layout_fn(guid: str, podcast_slug: str | None = None, *args, **kwargs) -> str:
    """
    Determines the S3 prefix for storing episode artifacts.
    Requires guid. If LAYOUT is 'podcast-guid', podcast_slug is also required.
    """
    if LAYOUT == "flat-guid":
        return f"{BASE_PREFIX}{guid}/"
    elif LAYOUT == "podcast-guid":
        if not podcast_slug:
            raise ValueError("podcast_slug is required for 'podcast-guid' layout")
        return f"{BASE_PREFIX}{podcast_slug}/{guid}/"
    else:
        raise ValueError(f"Unsupported LAYOUT: {LAYOUT}")

# Example usage (illustrative, not run):
# if __name__ == '__main__':
#     print(f"Layout: {LAYOUT}")
#     # For flat-guid:
#     # print(f"Example S3 prefix (flat-guid): {layout_fn(guid='some-guid-123')}")
#     # For podcast-guid:
#     print(f"Example S3 prefix (podcast-guid): {layout_fn(guid='some-guid-123', podcast_slug='my-podcast-show')}") 