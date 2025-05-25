import unittest
import json
from pathlib import Path

# This test would typically run from the project root, so paths to data need to be relative from there.
# For a real test suite, you might use a fixture or a known good meta.json file.

class TestMetaJsonIntegrity(unittest.TestCase):

    def test_segment_count_implies_timestamps(self):
        # This is a placeholder for how you might load a meta.json file.
        # In a real test, you would generate one or have a sample file.
        # For now, we'll use a mock dictionary.
        
        mock_meta_files_data = [
            {
                "name": "meta_with_segments_and_timestamps",
                "data": {
                    "guid": "test-guid-1",
                    "segment_count": 150,
                    "supports_timestamp": True,
                    "other_field": "value"
                }
            },
            {
                "name": "meta_with_segments_no_timestamps_flag_FAIL", # This should fail the implication
                "data": {
                    "guid": "test-guid-2",
                    "segment_count": 100,
                    "supports_timestamp": False 
                }
            },
            {
                "name": "meta_with_zero_segments_no_timestamps_flag",
                "data": {
                    "guid": "test-guid-3",
                    "segment_count": 0,
                    "supports_timestamp": False
                }
            },
            {
                "name": "meta_with_zero_segments_with_timestamps_flag_OK_VACUOUS", # Implication holds vacuously
                "data": {
                    "guid": "test-guid-4",
                    "segment_count": 0,
                    "supports_timestamp": True 
                }
            }
        ]

        for test_case in mock_meta_files_data:
            meta_content = test_case["data"]
            case_name = test_case["name"]
            with self.subTest(case_name=case_name):
                segment_count = meta_content.get("segment_count", 0)
                supports_timestamp = meta_content.get("supports_timestamp", False)

                if segment_count > 0:
                    self.assertTrue(supports_timestamp, 
                                    f"For {case_name} (GUID: {meta_content.get('guid')}), segment_count ({segment_count}) > 0 but supports_timestamp is False.")
                # If segment_count is 0, the implication (0 > 0 -> supports_timestamp) is vacuously true, 
                # or rather, the condition for the assert is not met, so no assertion is made on supports_timestamp specifically based on this rule.
                # We are only testing: IF segment_count > 0 THEN supports_timestamp MUST BE TRUE.

if __name__ == '__main__':
    unittest.main() 