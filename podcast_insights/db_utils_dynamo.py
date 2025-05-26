# podcast_insights/db_utils_dynamo.py
import os
import boto3
from botocore.exceptions import ClientError
import logging
import time
from typing import Dict, Any, Optional, List

log = logging.getLogger(__name__)

# Configuration - ideally from settings or environment variables
DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_STATUS_TABLE", "podinsights-status")
AWS_REGION = os.getenv("AWS_REGION") # Let boto3 use its default resolution if not set

_dynamodb_client = None
_dynamodb_resource = None
_boto_session = None # ADDED: Global variable for the session

# ADDED: Function to get or create a shared Boto3 session
def _get_boto_session(session: Optional[boto3.Session] = None) -> boto3.Session:
    global _boto_session
    if session:
        return session
    if _boto_session is None:
        profile_name = os.getenv("AWS_PROFILE")
        region_name = AWS_REGION
        if profile_name and region_name:
            log.info(f"Creating new Boto3 session with profile: {profile_name}, region: {region_name}")
            _boto_session = boto3.Session(profile_name=profile_name, region_name=region_name)
        elif region_name:
            log.info(f"Creating new Boto3 session with region: {region_name} (default profile)")
            _boto_session = boto3.Session(region_name=region_name)
        else:
            log.info("Creating new Boto3 session with default profile and region.")
            _boto_session = boto3.Session()
    return _boto_session

def _get_dynamodb_resource(session: Optional[boto3.Session] = None):
    global _dynamodb_resource
    # Ensure resource is recreated if session changes or not using the global one initially
    # This logic might need refinement if session is passed sometimes and not others.
    # For now, let's assume if a session is passed, we use it, otherwise global session then global resource.
    current_session = _get_boto_session(session)
    if _dynamodb_resource is None or (_dynamodb_resource.meta.client.meta.config.region_name != current_session.region_name):
        log.info(f"Initializing DynamoDB resource with session (region: {current_session.region_name})")
        _dynamodb_resource = current_session.resource('dynamodb')
    return _dynamodb_resource

def _get_dynamodb_client(session: Optional[boto3.Session] = None):
    global _dynamodb_client
    current_session = _get_boto_session(session)
    if _dynamodb_client is None or (_dynamodb_client.meta.region_name != current_session.region_name):
        log.info(f"Initializing DynamoDB client with session (region: {current_session.region_name})")
        _dynamodb_client = current_session.client('dynamodb')
    return _dynamodb_client

def init_dynamo_db_table(table_name: str = DYNAMODB_TABLE_NAME, read_capacity_units: int = 1, write_capacity_units: int = 1, session: Optional[boto3.Session] = None) -> bool:
    """
    Initializes the DynamoDB table if it doesn't exist.
    Uses provisioned throughput by default, can be changed to PAY_PER_REQUEST.
    Returns True if table exists or was created, False on error.
    """
    client = _get_dynamodb_client(session)
    try:
        client.describe_table(TableName=table_name)
        log.info(f"DynamoDB table '{table_name}' already exists.")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            log.info(f"DynamoDB table '{table_name}' not found. Creating table...")
            try:
                client.create_table(
                    TableName=table_name,
                    KeySchema=[
                        {
                            'AttributeName': 'episode_guid',
                            'KeyType': 'HASH'  # Primary Key
                        },
                        # Add GSI for podcast_slug and processing_status if needed for queries
                        # {
                        #     'AttributeName': 'podcast_slug',
                        #     'KeyType': 'RANGE' # Example sort key if needed
                        # }
                    ],
                    AttributeDefinitions=[
                        {
                            'AttributeName': 'episode_guid',
                            'AttributeType': 'S'
                        },
                        # {
                        #     'AttributeName': 'podcast_slug',
                        #     'AttributeType': 'S'
                        # },
                    ],
                    BillingMode='PAY_PER_REQUEST' # Preferred for unpredictable workloads
                    # ProvisionedThroughput={
                    #     'ReadCapacityUnits': read_capacity_units,
                    #     'WriteCapacityUnits': write_capacity_units
                    # }
                )
                log.info(f"Waiting for table '{table_name}' to be created...")
                waiter = client.get_waiter('table_exists')
                waiter.wait(TableName=table_name, WaiterConfig={'Delay': 5, 'MaxAttempts': 12})
                log.info(f"DynamoDB table '{table_name}' created successfully.")
                return True
            except ClientError as ce:
                log.error(f"Error creating DynamoDB table '{table_name}': {ce}", exc_info=True)
                return False
        else:
            log.error(f"Error describing DynamoDB table '{table_name}': {e}", exc_info=True)
            return False

def update_episode_status(episode_guid: str, attributes_to_update: Dict[str, Any], table_name: str = DYNAMODB_TABLE_NAME, session: Optional[boto3.Session] = None) -> bool:
    """
    Updates an episode's status and other attributes in DynamoDB.
    Uses UpdateItem with SET action for each attribute.

    Args:
        episode_guid: The GUID of the episode.
        attributes_to_update: Dictionary of attributes to set/update.
                              e.g., {'processing_status': 'fetched', 's3_audio_path_raw': 's3://...'}
                              Nested dictionaries are supported by DynamoDB.
        table_name: Name of the DynamoDB table.

    Returns:
        True if update was successful, False otherwise.
    """
    if not episode_guid or not attributes_to_update:
        log.warning("episode_guid or attributes_to_update is empty. Skipping DynamoDB update.")
        return False

    dynamodb = _get_dynamodb_resource(session)
    table = dynamodb.Table(table_name)
    
    update_expression_parts = []
    expression_attribute_values = {}
    expression_attribute_names = {}

    for i, (key, value) in enumerate(attributes_to_update.items()):
        # DynamoDB attribute names cannot contain special characters like '.' or '-'
        # Use ExpressionAttributeNames for any such keys.
        # For simplicity, let's assume keys are simple for now, or use placeholders if needed.
        # Placeholder for attribute name: #attr_name
        # Placeholder for attribute value: :attr_val
        attr_name_placeholder = f"#k{i}"
        attr_value_placeholder = f":v{i}"
        
        update_expression_parts.append(f"{attr_name_placeholder} = {attr_value_placeholder}")
        expression_attribute_names[attr_name_placeholder] = key
        expression_attribute_values[attr_value_placeholder] = value

    if not update_expression_parts:
        log.warning("No valid attributes to update. Skipping DynamoDB update.")
        return False

    update_expression = "SET " + ", ".join(update_expression_parts)

    try:
        log.info(f"Updating DynamoDB item for GUID '{episode_guid}' in table '{table_name}'. Attributes: {list(attributes_to_update.keys())}")
        # log.debug(f"UpdateExpression: {update_expression}")
        # log.debug(f"ExpressionAttributeValues: {expression_attribute_values}")
        # log.debug(f"ExpressionAttributeNames: {expression_attribute_names}")
        
        table.update_item(
            Key={
                'episode_guid': episode_guid
            },
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_attribute_values,
            ExpressionAttributeNames=expression_attribute_names
        )
        log.info(f"Successfully updated DynamoDB item for GUID '{episode_guid}'.")
        return True
    except ClientError as e:
        log.error(f"Error updating DynamoDB item for GUID '{episode_guid}': {e}", exc_info=True)
        return False

def get_episode_status(episode_guid: str, table_name: str = DYNAMODB_TABLE_NAME, session: Optional[boto3.Session] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieves an episode's item (status and attributes) from DynamoDB.

    Args:
        episode_guid: The GUID of the episode.
        table_name: Name of the DynamoDB table.

    Returns:
        A dictionary containing the episode's attributes if found, None otherwise.
    """
    if not episode_guid:
        log.warning("episode_guid is empty. Cannot get status from DynamoDB.")
        return None

    dynamodb = _get_dynamodb_resource(session)
    table = dynamodb.Table(table_name)

    try:
        log.debug(f"Getting DynamoDB item for GUID '{episode_guid}' from table '{table_name}'.")
        response = table.get_item(
            Key={
                'episode_guid': episode_guid
            }
        )
        item = response.get('Item')
        if item:
            log.debug(f"Found DynamoDB item for GUID '{episode_guid}': {item}")
            return item
        else:
            log.debug(f"No DynamoDB item found for GUID '{episode_guid}'.")
            return None
    except ClientError as e:
        log.error(f"Error getting DynamoDB item for GUID '{episode_guid}': {e}", exc_info=True)
        return None


# Example usage (for testing this module standalone):
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)7s | %(module)s | %(message)s')
    log.info("Testing DynamoDB utils...")

    # Ensure AWS credentials and region are configured in your environment for boto3
    # For local testing, you might use DynamoDB Local (https://aws.amazon.com/dynamodb/local/)

    # Create a session for testing if AWS_PROFILE and AWS_REGION are set
    test_session = None
    test_profile = os.getenv("AWS_PROFILE")
    test_region = os.getenv("AWS_REGION")
    if test_profile and test_region:
        log.info(f"Creating test session with profile {test_profile} and region {test_region}")
        test_session = boto3.Session(profile_name=test_profile, region_name=test_region)
    else:
        log.info("Using default Boto3 session for testing (ensure AWS_REGION is set if not using a profile with a default region).")
        # If AWS_REGION is set, default session will pick it up. If not, _get_boto_session will try to create one.
        # This is just for the __main__ block test.
        test_session = _get_boto_session() # Use the utility to get a session

    test_table_name = "podinsights-status-test"
    
    log.info(f"Initializing test table: {test_table_name}")
    if init_dynamo_db_table(table_name=test_table_name, session=test_session):
        log.info("Test table initialized or already exists.")

        test_guid_1 = "test-guid-123"
        test_guid_2 = "test-guid-456"

        log.info(f"--- Testing update_episode_status for {test_guid_1} (new item) ---")
        attrs_1 = {
            "processing_status": "fetched",
            "podcast_slug": "test-podcast",
            "s3_audio_path_raw": "s3://bucket/raw/test-podcast/test-guid-123/audio.mp3",
            "fetch_timestamp": time.time(),
            "retries": 0,
            "nested_info": {"detail1": "value1", "count": 5}
        }
        update_episode_status(test_guid_1, attrs_1, table_name=test_table_name, session=test_session)

        log.info(f"--- Testing get_episode_status for {test_guid_1} ---")
        status_1 = get_episode_status(test_guid_1, table_name=test_table_name, session=test_session)
        if status_1:
            log.info(f"Status for {test_guid_1}: {status_1}")
            assert status_1['processing_status'] == "fetched"
            assert status_1['nested_info']['count'] == 5
        else:
            log.error(f"Could not retrieve status for {test_guid_1}")

        log.info(f"--- Testing update_episode_status for {test_guid_1} (update existing) ---")
        attrs_1_updated = {
            "processing_status": "transcribed",
            "s3_transcript_path_stage": "s3://bucket/stage/test-podcast/test-guid-123/transcript.json",
            "transcribe_timestamp": time.time(),
            "retries": 1, # Increment retries
            "nested_info": {"detail1": "value1_updated", "count": 10, "new_field": "added"} 
        }
        update_episode_status(test_guid_1, attrs_1_updated, table_name=test_table_name, session=test_session)
        status_1_updated = get_episode_status(test_guid_1, table_name=test_table_name, session=test_session)
        if status_1_updated:
            log.info(f"Updated status for {test_guid_1}: {status_1_updated}")
            assert status_1_updated['processing_status'] == "transcribed"
            assert status_1_updated['retries'] == 1
            assert status_1_updated['nested_info']['count'] == 10
            assert status_1_updated['nested_info']['new_field'] == "added"

        log.info(f"--- Testing get_episode_status for {test_guid_2} (non-existent) ---")
        status_2 = get_episode_status(test_guid_2, table_name=test_table_name, session=test_session)
        if status_2 is None:
            log.info(f"Status for {test_guid_2} is None, as expected.")
        else:
            log.error(f"Status for {test_guid_2} was found, which is unexpected: {status_2}")

        # log.info(f"Deleting test table: {test_table_name} (manual step for now if needed)")
        # client = _get_dynamodb_client()
        # client.delete_table(TableName=test_table_name)
        # log.info(f"Table {test_table_name} deletion initiated.")

    else:
        log.error(f"Failed to initialize test table {test_table_name}. Aborting tests.")

    log.info("DynamoDB utils test run finished.") 