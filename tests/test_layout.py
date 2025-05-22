from podcast_insights.settings import layout_fn, BASE_PREFIX

def test_flat_guid_layout():
    guid = "abcdef123456"
    # Use BASE_PREFIX from settings to make the test robust to its changes
    assert layout_fn(guid) == f"{BASE_PREFIX}{guid}/"

def test_layout_fn_with_empty_guid():
    import pytest
    with pytest.raises(ValueError, match="GUID cannot be empty for layout_fn"):
        layout_fn("")

def test_layout_fn_with_none_guid():
    import pytest
    with pytest.raises(ValueError, match="GUID cannot be empty for layout_fn"):
        layout_fn(None) # type: ignore 