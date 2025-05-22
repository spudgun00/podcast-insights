import boto3
import re
import hashlib
import tempfile
import json
import logging
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)

def parse_s3_uri(s3_uri: str) -> tuple[str | None, str | None]:
    """Parses an S3 URI (s3://bucket/key) into bucket and key.
    Returns (None, None) if URI is invalid.
    """
    match = re.match(r"s3://([^/]+)/?(.*)", s3_uri)
    if match:
        bucket_name = match.group(1)
        key = match.group(2)
        return bucket_name, key
    logger.warning(f"Invalid S3 URI: {s3_uri}")
    return None, None

def list_s3_objects_by_suffix(s3_client, bucket: str, prefix: str, suffix: str) -> list[dict]:
    """Lists S3 objects under a given prefix that end with a specific suffix."""
    paginator = s3_client.get_paginator('list_objects_v2')
    matching_objects = []
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    if obj["Key"].endswith(suffix):
                        matching_objects.append(obj)
        logger.info(f"Found {len(matching_objects)} objects in s3://{bucket}/{prefix} ending with '{suffix}'.")
    except Exception as e:
        logger.error(f"Error listing S3 objects in bucket {bucket} with prefix {prefix}: {e}")
    return matching_objects

def download_s3_json(s3_client, bucket: str, key: str) -> dict | None:
    """Downloads a JSON file from S3 and parses it."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode('utf-8')
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error downloading or parsing JSON from s3://{bucket}/{key}: {e}")
        return None

def get_s3_object_md5(s3_client, bucket: str, key: str) -> str | None:
    """Downloads an S3 object to a temporary file and calculates its MD5 hash."""
    with tempfile.NamedTemporaryFile() as tmp_file:
        try:
            s3_client.download_file(bucket, key, tmp_file.name)
            hash_md5 = hashlib.md5() # noqa: S324
            with open(tmp_file.name, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating MD5 for s3://{bucket}/{key}: {e}")
            return None

def s3_object_exists(s3_client, bucket: str, key: str) -> bool:
    """Checks if an S3 object exists."""
    if not bucket or not key:
        logger.warning("s3_object_exists: Bucket or key is empty.")
        return False
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:
        # Check if the exception is because the object does not exist (e.g., ClientError with 404)
        if hasattr(e, 'response') and e.response.get('Error', {}).get('Code') == '404':
            logger.debug(f"Object s3://{bucket}/{key} not found (404).")
        else:
            logger.warning(f"Error checking existence of s3://{bucket}/{key}: {e}")
        return False

def get_s3_object_size(s3_client, bucket: str, key: str) -> int | None:
    """Gets the size of an S3 object in bytes."""
    if not bucket or not key:
        logger.warning("get_s3_object_size: Bucket or key is empty.")
        return None
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return response.get('ContentLength')
    except Exception as e:
        logger.warning(f"Error getting size for s3://{bucket}/{key}: {e}")
        return None 