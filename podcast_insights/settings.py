# podcast_insights/settings.py
# ------------------------------------------------------------------
# GLOBAL STORAGE SETTINGS
# ------------------------------------------------------------------

# S3 Bucket names for different stages of data processing
S3_BUCKET_RAW          = "pod-insights-raw"      # For raw audio (MP3s) and minimal initial meta
S3_BUCKET_STAGE        = "pod-insights-stage"    # For transcripts, KPIs, enriched meta, embeddings, etc.
S3_BUCKET_MANIFESTS    = "pod-insights-manifests" # For generated manifest files (used by generate_manifest.py)

BASE_PREFIX     = ""      # Common base prefix within S3 buckets. Set to empty to match Runbook paths like <feed_slug>/<guid>/.
LAYOUT          = "podcast-guid"                # Layout structure within the BASE_PREFIX

def layout_fn(guid: str, podcast_slug: str | None = None) -> str:
    """
    Determines the S3 key prefix for storing episode artifacts based on LAYOUT.
    Requires guid. If LAYOUT is 'podcast-guid', podcast_slug is also required.
    If BASE_PREFIX is empty, returns a string like "podcast_slug/guid/" (if podcast-guid layout)
    or "guid/" (if flat-guid layout).
    The returned prefix ends with a slash.
    """
    # Determine the initial part of the prefix based on BASE_PREFIX
    current_base = BASE_PREFIX
    if current_base and not current_base.endswith("/"):
        current_base = f"{current_base}/"
    # If current_base is empty, it remains empty, so paths start directly with slug/guid or guid.

    if LAYOUT == "flat-guid":
        if not guid:
            raise ValueError("guid is required for 'flat-guid' S3 layout")
        return f"{current_base}{guid}/"
    elif LAYOUT == "podcast-guid":
        if not podcast_slug:
            raise ValueError("podcast_slug is required for 'podcast-guid' layout")
        if not guid:
            raise ValueError("guid is required for 'podcast-guid' layout")
        return f"{current_base}{podcast_slug}/{guid}/"
    else:
        raise ValueError(f"Unsupported LAYOUT: {LAYOUT}")

# Example usage (illustrative, not run):
# if __name__ == '__main__':
#     from pathlib import Path # Needed for this example block if run directly
#     print(f"Layout: {LAYOUT}")
#     print(f"S3 Raw Bucket: {S3_BUCKET_RAW}")
#     print(f"S3 Stage Bucket: {S3_BUCKET_STAGE}")
#     print(f"Current BASE_PREFIX: '{BASE_PREFIX}'")
#     print(f"Example S3 prefix (podcast-guid): {layout_fn(guid='some-guid-123', podcast_slug='my-podcast-show')}")
#     # To test flat-guid, you'd temporarily change LAYOUT above
#     # LAYOUT = "flat-guid"
#     # print(f"Example S3 prefix (flat-guid): {layout_fn(guid='some-guid-123')}") 