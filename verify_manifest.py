import argparse
import boto3
import logging
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from podcast_insights
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from podcast_insights.s3_utils import (
    parse_s3_uri,
    list_s3_objects_by_suffix,
    download_s3_json,
    get_s3_object_md5,
    s3_object_exists,
    get_s3_object_size
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_episode_manifest(s3_client, meta_json_s3_uri: str) -> dict:
    """
    Verifies a single episode's manifest (meta.json) and its associated S3 artifacts.

    Args:
        s3_client: Initialized Boto3 S3 client.
        meta_json_s3_uri: Full S3 URI to the meta.json file.

    Returns:
        A dictionary containing verification results and any issues found.
    """
    results = {
        "meta_json_uri": meta_json_s3_uri,
        "guid": None,
        "checks": [],
        "issues": [],
        "overall_status": "VERIFIED"
    }

    bucket_name, meta_key = parse_s3_uri(meta_json_s3_uri)
    if not bucket_name or not meta_key:
        results["issues"].append(f"Invalid meta.json S3 URI: {meta_json_s3_uri}")
        results["overall_status"] = "ERROR"
        return results

    logger.info(f"Processing manifest: {meta_json_s3_uri}")
    meta_data = download_s3_json(s3_client, bucket_name, meta_key)

    if not meta_data:
        results["issues"].append(f"Failed to download or parse meta.json: {meta_json_s3_uri}")
        results["overall_status"] = "ERROR"
        return results

    results["guid"] = meta_data.get("guid", "Unknown GUID")
    episode_identifier = f"GUID: {results['guid']} (meta: {meta_json_s3_uri})"

    # Expected fields and their checks
    # (s3_path_key_in_meta, expected_hash_key_in_meta, expected_size_key_in_meta, friendly_name)
    files_to_check = [
        ("audio_s3_path", "audio_hash", "audio_expected_size_bytes", "Audio File"),
        ("transcript_s3_path", None, None, "Transcript File"),
        ("cleaned_entities_path", None, None, "Cleaned Entities File"),
        # Add other files if needed, e.g., raw_entities_s3_path
    ]

    for s3_path_key, hash_key, size_key, friendly_name in files_to_check:
        check_result = {"file": friendly_name, "s3_path_key": s3_path_key, "status": "OK", "details": []}
        s3_path_value = meta_data.get(s3_path_key)

        if not s3_path_value:
            check_result["status"] = "MISSING_PATH_IN_META"
            check_result["details"].append(f"S3 path key '{s3_path_key}' not found in meta.json.")
            results["issues"].append(f"{episode_identifier}: {friendly_name} - S3 path not in meta.json.")
            results["checks"].append(check_result)
            results["overall_status"] = "ERROR"
            continue # Cannot proceed with this file if path is missing

        file_bucket, file_key = parse_s3_uri(s3_path_value)
        if not file_bucket or not file_key:
            check_result["status"] = "INVALID_S3_URI_IN_META"
            check_result["details"].append(f"Invalid S3 URI for {friendly_name}: {s3_path_value}")
            results["issues"].append(f"{episode_identifier}: {friendly_name} - Invalid S3 URI in meta.json: {s3_path_value}")
            results["checks"].append(check_result)
            results["overall_status"] = "ERROR"
            continue
        
        check_result["actual_s3_uri"] = s3_path_value

        if not s3_object_exists(s3_client, file_bucket, file_key):
            check_result["status"] = "NOT_FOUND_ON_S3"
            check_result["details"].append(f"File not found on S3: {s3_path_value}")
            results["issues"].append(f"{episode_identifier}: {friendly_name} - File not found on S3: {s3_path_value}")
            results["overall_status"] = "ERROR"
        else:
            check_result["details"].append(f"File found on S3: {s3_path_value}")
            
            # Size check (if expected size is provided)
            if size_key and meta_data.get(size_key) is not None:
                expected_size = meta_data.get(size_key)
                actual_size = get_s3_object_size(s3_client, file_bucket, file_key)
                check_result["expected_size"] = expected_size
                check_result["actual_size"] = actual_size
                if actual_size is None:
                    check_result["status"] = "SIZE_CHECK_FAILED"
                    check_result["details"].append(f"Could not retrieve actual size for {s3_path_value}.")
                    results["issues"].append(f"{episode_identifier}: {friendly_name} - Could not get actual size from S3.")
                    results["overall_status"] = "ERROR"
                elif actual_size != expected_size:
                    check_result["status"] = "SIZE_MISMATCH"
                    check_result["details"].append(f"Size mismatch: Expected {expected_size}, got {actual_size}.")
                    results["issues"].append(f"{episode_identifier}: {friendly_name} - Size mismatch (expected {expected_size}, got {actual_size}).")
                    results["overall_status"] = "ERROR"
                else:
                    check_result["details"].append(f"Size matches expected: {actual_size} bytes.")

            # Hash check (if expected hash is provided)
            if hash_key and meta_data.get(hash_key):
                expected_hash = meta_data.get(hash_key)
                actual_hash = get_s3_object_md5(s3_client, file_bucket, file_key)
                check_result["expected_hash"] = expected_hash
                check_result["actual_hash"] = actual_hash
                if actual_hash is None:
                    check_result["status"] = "HASH_CHECK_FAILED"
                    check_result["details"].append(f"Could not calculate actual MD5 hash for {s3_path_value}.")
                    results["issues"].append(f"{episode_identifier}: {friendly_name} - Could not calculate MD5 hash from S3.")
                    results["overall_status"] = "ERROR"
                elif actual_hash != expected_hash:
                    check_result["status"] = "HASH_MISMATCH"
                    check_result["details"].append(f"MD5 Hash mismatch: Expected {expected_hash}, got {actual_hash}.")
                    results["issues"].append(f"{episode_identifier}: {friendly_name} - MD5 Hash mismatch.")
                    results["overall_status"] = "ERROR"
                else:
                    check_result["details"].append(f"MD5 Hash matches expected: {actual_hash}.")
        
        results["checks"].append(check_result)

    if results["overall_status"] == "VERIFIED":
        logger.info(f"{episode_identifier}: All checks passed.")
    else:
        logger.warning(f"{episode_identifier}: One or more checks failed. Issues: {results['issues']}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Verify integrity of podcast episode artifacts on S3 based on meta.json manifests.")
    parser.add_argument("s3_path_glob", 
                        help="S3 path glob to find meta.json files (e.g., 's3://bucket-name/path/to/episodes/**/meta.json'). "
                             "The part before '**' is the prefix, the part after is the suffix.")
    parser.add_argument("--profile", help="AWS CLI profile to use for credentials.")
    parser.add_argument("--region", help="AWS region for the S3 client.")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop processing further manifests if an error is found in one.")

    args = parser.parse_args()

    # Parse the s3_path_glob
    # Example: s3://my-bucket/data/processed/**/meta.json
    # prefix_part = data/processed/
    # suffix_part = meta.json (if glob is simple like */meta.json or **/meta.json)
    # For more complex globs like **/something_*_meta.json, this suffix logic would need to be smarter.
    # For now, assume the glob ends with the exact suffix we are looking for.
    
    glob_parts = args.s3_path_glob.split('/**/')
    if len(glob_parts) != 2:
        logger.error("Invalid s3_path_glob format. Expected format: s3://bucket/prefix/**/suffix (e.g., s3://my-bucket/data/**/meta.json)")
        sys.exit(1)

    s3_base_uri_prefix = glob_parts[0]
    search_suffix = glob_parts[1]

    bucket_name, s3_prefix = parse_s3_uri(s3_base_uri_prefix)
    if not bucket_name:
        logger.error(f"Could not parse bucket from s3_path_glob: {args.s3_path_glob}")
        sys.exit(1)
    if s3_prefix is None: # Root of bucket is a valid prefix
        s3_prefix = ""
    
    # Initialize S3 client
    session_args = {}
    if args.profile:
        session_args['profile_name'] = args.profile
    if args.region:
        session_args['region_name'] = args.region
    
    try:
        session = boto3.Session(**session_args)
        s3_client = session.client('s3')
    except Exception as e:
        logger.error(f"Failed to initialize AWS S3 client: {e}")
        sys.exit(1)

    logger.info(f"Searching for manifests in s3://{bucket_name}/{s3_prefix} with suffix '{search_suffix}'")
    manifest_objects = list_s3_objects_by_suffix(s3_client, bucket_name, s3_prefix, search_suffix)

    if not manifest_objects:
        logger.warning(f"No manifests found matching criteria.")
        sys.exit(0)

    total_manifests = len(manifest_objects)
    logger.info(f"Found {total_manifests} manifests to verify.")

    all_results = []
    errors_found_overall = False

    for i, s3_obj in enumerate(manifest_objects):
        manifest_s3_uri = f"s3://{bucket_name}/{s3_obj['Key']}"
        logger.info(f"--- Verifying manifest {i+1}/{total_manifests}: {manifest_s3_uri} ---")
        verification_result = verify_episode_manifest(s3_client, manifest_s3_uri)
        all_results.append(verification_result)
        if verification_result["overall_status"] != "VERIFIED":
            errors_found_overall = True
            if args.stop_on_error:
                logger.error(f"Stopping due to error in {manifest_s3_uri} as --stop_on_error is set.")
                break
        logger.info("---") # Separator

    # Summary
    logger.info("======== Verification Summary ========")
    verified_count = sum(1 for r in all_results if r["overall_status"] == "VERIFIED")
    error_count = total_manifests - verified_count # Based on processed manifests

    logger.info(f"Total manifests processed: {len(all_results)}")
    logger.info(f"Manifests VERIFIED: {verified_count}")
    logger.info(f"Manifests with ERRORS: {error_count}")

    if error_count > 0:
        logger.error("Verification finished with errors. Details:")
        for result in all_results:
            if result["overall_status"] != "VERIFIED":
                logger.error(f"  Manifest: {result['meta_json_uri']} (GUID: {result['guid']})")
                for issue in result["issues"]:
                    logger.error(f"    - {issue}")
        sys.exit(1)
    else:
        logger.info("All processed manifests verified successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main() 