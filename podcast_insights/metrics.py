# podcast_insights/metrics.py
import os
import boto3
from botocore.exceptions import ClientError
import logging
from typing import Optional, List, Dict, Any, Union

log = logging.getLogger(__name__)

# Configuration
CLOUDWATCH_NAMESPACE = os.getenv("CLOUDWATCH_NAMESPACE", "PodInsights")
AWS_REGION = os.getenv("AWS_REGION") # Let boto3 use its default resolution if not set

_cloudwatch_client = None

def _get_cloudwatch_client():
    global _cloudwatch_client
    if _cloudwatch_client is None:
        if AWS_REGION:
            _cloudwatch_client = boto3.client('cloudwatch', region_name=AWS_REGION)
        else:
            _cloudwatch_client = boto3.client('cloudwatch')
    return _cloudwatch_client

def put_metric_data_value(
    metric_name: str,
    value: float,
    dimensions: Optional[List[Dict[str, str]]] = None,
    unit: str = 'Count', # See AWS StandardUnit for valid values
    namespace: str = CLOUDWATCH_NAMESPACE
) -> bool:
    """
    Puts a single metric data point to CloudWatch.

    Args:
        metric_name: The name of the metric.
        value: The value for the metric.
        dimensions: A list of dimensions for the metric, e.g., 
                    [{'Name': 'FeedSlug', 'Value': 'some-slug'}].
        unit: The unit of the metric (e.g., 'Seconds', 'Milliseconds', 'Bytes', 'Percent', 'Count').
        namespace: The CloudWatch namespace for the metric.

    Returns:
        True if the metric was sent successfully, False otherwise.
    """
    if dimensions is None:
        dimensions = []
    
    client = _get_cloudwatch_client()
    try:
        log.debug(f"Putting metric to CloudWatch: Namespace={namespace}, Name={metric_name}, Value={value}, Unit={unit}, Dimensions={dimensions}")
        client.put_metric_data(
            Namespace=namespace,
            MetricData=[
                {
                    'MetricName': metric_name,
                    'Dimensions': dimensions,
                    'Value': value,
                    'Unit': unit
                    # 'Timestamp': datetime.utcnow() # Optional, CW uses receive time if not specified
                },
            ]
        )
        log.info(f"Successfully sent metric '{metric_name}' to CloudWatch. Value: {value} {unit}")
        return True
    except ClientError as e:
        log.error(f"Error sending metric '{metric_name}' to CloudWatch: {e}", exc_info=True)
        return False
    except Exception as e:
        # Catch any other unexpected errors, e.g., during client initialization if not caught before
        log.error(f"Unexpected error sending metric '{metric_name}': {e}", exc_info=True)
        return False

# Example usage (for testing this module standalone):
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)7s | %(module)s | %(message)s')
    log.info("Testing CloudWatch metrics utils...")

    # Ensure AWS credentials and region are configured in your environment for boto3.
    # This will attempt to send actual metrics to your AWS account.

    log.info("--- Test 1: Simple Count Metric ---")
    success1 = put_metric_data_value(
        metric_name="TestDownloadSuccess",
        value=1,
        unit="Count"
    )
    log.info(f"Test 1 Succeeded: {success1}")

    log.info("--- Test 2: Metric with Dimensions ---")
    success2 = put_metric_data_value(
        metric_name="TestPerHostDownloadTime",
        value=150.5,
        dimensions=[
            {'Name': 'Hostname', 'Value': 'test.example.com'},
            {'Name': 'FeedSlug', 'Value': 'test-feed'}
        ],
        unit="Milliseconds"
    )
    log.info(f"Test 2 Succeeded: {success2}")

    log.info("--- Test 3: Failure Metric (e.g., 429 error count) ---")
    success3 = put_metric_data_value(
        metric_name="TestPerHost429Errors",
        value=5,
        dimensions=[
            {'Name': 'Hostname', 'Value': 'another-host.com'}
        ],
        unit="Count"
    )
    log.info(f"Test 3 Succeeded: {success3}")
    
    log.info("--- Test 4: Percentage Metric ---")
    success4 = put_metric_data_value(
        metric_name="TestSuccessRate",
        value=98.5,
        dimensions=[
            {'Name': 'FeedSlug', 'Value': 'my-critical-feed'}
        ],
        unit="Percent"
    )
    log.info(f"Test 4 Succeeded: {success4}")

    log.info("CloudWatch metrics utils test run finished. Check your CloudWatch console for metrics in the 'PodInsights' (or custom) namespace.") 