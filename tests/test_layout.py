import unittest
from pathlib import Path
import sys

# Ensure the module path is correct for imports
# This assumes your test script is in /tests and settings.py is in /podcast_insights
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from podcast_insights.settings import layout_fn, LAYOUT, BASE_PREFIX

class TestStorageLayout(unittest.TestCase):

    def test_layout_function_flat_guid(self):
        # Temporarily set LAYOUT for this test if possible, or ensure settings reflects this
        # For this example, we assume LAYOUT can be tested directly based on its value in settings.py
        if LAYOUT == "flat-guid":
            guid = "test-guid-123"
            expected_prefix = f"{BASE_PREFIX}{guid}/"
            self.assertEqual(layout_fn(guid), expected_prefix)
            print(f"Test flat-guid passed: {layout_fn(guid)}")
        else:
            print(f"Skipping flat-guid test as LAYOUT is {LAYOUT}")

    def test_layout_function_podcast_guid(self):
        if LAYOUT == "podcast-guid":
            guid = "test-guid-456"
            podcast_slug = "test-podcast-slug"
            expected_prefix = f"{BASE_PREFIX}{podcast_slug}/{guid}/"
            self.assertEqual(layout_fn(guid, podcast_slug), expected_prefix)
            print(f"Test podcast-guid passed: {layout_fn(guid, podcast_slug)}")
        else:
            print(f"Skipping podcast-guid test as LAYOUT is {LAYOUT}")

    def test_layout_function_podcast_guid_missing_slug(self):
        if LAYOUT == "podcast-guid":
            guid = "test-guid-789"
            with self.assertRaisesRegex(ValueError, "podcast_slug is required for 'podcast-guid' layout"):
                layout_fn(guid) # Missing podcast_slug
            print(f"Test podcast-guid missing slug passed.")
        else:
            print(f"Skipping podcast-guid missing slug test as LAYOUT is {LAYOUT}")

if __name__ == '__main__':
    unittest.main() 