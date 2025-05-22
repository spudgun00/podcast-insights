import pytest
from podcast_insights.meta_utils import tidy_people, ALIAS_MAP, KNOWN_HOSTS

# Sample KNOWN_HOSTS and ALIAS_MAP for testing
# In a real scenario, these might be mocked or loaded from test-specific files

@pytest.fixture(autouse=True)
def setup_test_configs(monkeypatch):
    # Monkeypatch the global ALIAS_MAP and KNOWN_HOSTS for isolated testing
    test_alias_map = {
        "mark zuckerberg": "Mark Zuckerberg",
        "zuck": "Mark Zuckerberg",
        "mark zuck": "Mark Zuckerberg",
        "fb ceo": "Mark Zuckerberg", # Example of an alias
        "sbf": "Sam Bankman-Fried",
        "harry": "Harry Stebbings"
    }
    test_known_hosts = {
        "Some Podcast": {"Mark Zuckerberg", "Harry Stebbings"},
        "Another Show": {"Sam Bankman-Fried"}
    }
    monkeypatch.setattr("podcast_insights.meta_utils.ALIAS_MAP", test_alias_map)
    monkeypatch.setattr("podcast_insights.meta_utils.KNOWN_HOSTS", test_known_hosts)

def test_tidy_people_zuckerberg_aliasing_and_dedupe():
    meta_input = {"podcast": "Some Podcast"}
    initial_people_list = [
        {"name": "mark zuckerberg", "role": "guest", "org": "Meta"},
        {"name": "Zuck", "role": "guest", "title": "CEO"},
        {"name": "Mark Zuck", "role": "guest", "href": "http://example.com/zuck"},
        {"name": "FB CEO", "role": "guest", "org": "Facebook (old)"},
        {"name": "Some Other Person", "role": "guest", "org": "OtherCorp"}
    ]
    
    hosts, guests = tidy_people(meta_input, initial_people_list)
    
    # Expected: Mark Zuckerberg should be identified as a host due to KNOWN_HOSTS
    # All variants should be merged into one host entry.
    
    assert len(hosts) == 1
    assert hosts[0]["name"] == "Mark Zuckerberg" # KNOWN_HOSTS might have specific casing
    assert hosts[0]["role"] == "host"
    assert hosts[0]["org"] == "Meta"  # "Meta" should be preferred due to merge logic (if it's first non-empty) or length
    assert hosts[0]["title"] == "CEO"
    assert hosts[0]["href"] == "http://example.com/zuck"
    
    assert len(guests) == 1
    assert guests[0]["name"] == "Some Other Person"

def test_tidy_people_alias_to_guest():
    meta_input = {"podcast": "Tech Talks"} # A podcast where SBF is not a known host
    initial_people_list = [
        {"name": "sbf", "role": "guest", "org": "FTX"},
        {"name": "Sam Bankman-Fried", "role": "guest", "title": "Former CEO"}
    ]
    hosts, guests = tidy_people(meta_input, initial_people_list)
    
    assert len(hosts) == 0
    assert len(guests) == 1
    assert guests[0]["name"] == "Sam Bankman-Fried"
    assert guests[0]["org"] == "FTX"
    assert guests[0]["title"] == "Former CEO"

def test_tidy_people_known_host_and_alias():
    # Test case where an alias points to someone who is also a known host
    meta_input = {"podcast": "Some Podcast"} # Harry Stebbings is a known host
    initial_people_list = [
        {"name": "harry", "role": "guest", "org": "20VC"}, # 'harry' is an alias for 'Harry Stebbings'
        {"name": "Another Guest", "role": "guest"}
    ]
    hosts, guests = tidy_people(meta_input, initial_people_list)

    assert len(hosts) == 1
    # KNOWN_HOSTS for "Some Podcast" has "Harry Stebbings", check if it's title cased.
    # The actual KNOWN_HOSTS has "Harry Stebbings".
    # The alias map has "harry": "Harry Stebbings".
    # The tidy_people logic should title case host names if they aren't already.
    assert hosts[0]["name"] == "Harry Stebbings" # Expect canonical and title-cased name
    assert hosts[0]["role"] == "host"
    assert hosts[0]["org"] == "20VC"
    
    assert len(guests) == 1
    assert guests[0]["name"] == "Another Guest"

def test_merge_people_logic():
    from podcast_insights.meta_utils import merge_people
    
    # Test basic merge: fill empty fields
    p1 = {"name": "Test Person", "role": "guest", "org": "Org A", "title": None, "href": None}
    p2 = {"name": "Test Person", "role": "guest", "org": None, "title": "Title B", "href": "http://b.com"}
    merged = merge_people(p1, p2)
    assert merged["org"] == "Org A"
    assert merged["title"] == "Title B"
    assert merged["href"] == "http://b.com"

    # Test merge: prefer existing non-empty over new empty
    p1_rev = {"name": "Test Person", "role": "guest", "org": None, "title": "Title B", "href": "http://b.com"}
    p2_rev = {"name": "Test Person", "role": "guest", "org": "Org A", "title": None, "href": None}
    merged_rev = merge_people(p1_rev, p2_rev)
    assert merged_rev["org"] == "Org A"
    assert merged_rev["title"] == "Title B"
    assert merged_rev["href"] == "http://b.com"

    # Test merge: conflict, prefer longer string
    p_conflict1 = {"name": "Test Person", "role": "guest", "org": "Short Org"}
    p_conflict2 = {"name": "Test Person", "role": "guest", "org": "Much Longer Organization Name"}
    merged_conflict = merge_people(p_conflict1, p_conflict2)
    assert merged_conflict["org"] == "Much Longer Organization Name"
    
    merged_conflict_rev = merge_people(p_conflict2, p_conflict1) # Order shouldn't matter for this rule if implemented robustly
    assert merged_conflict_rev["org"] == "Much Longer Organization Name"

    # Test role merging: host preference
    p_host = {"name": "Test Person", "role": "host"}
    p_guest = {"name": "Test Person", "role": "guest"}
    merged_role_hg = merge_people(p_host, p_guest)
    assert merged_role_hg["role"] == "host"
    merged_role_gh = merge_people(p_guest, p_host)
    assert merged_role_gh["role"] == "host"
    
    p_guest_no_role = {"name": "Test Person"} # No role specified
    merged_role_gr = merge_people(p_guest, p_guest_no_role)
    assert merged_role_gr["role"] == "guest"

    # Test that other fields are preserved
    p_full = {"name": "Test Full", "role": "guest", "org":"FullOrg", "title":"FullTitle", "href":"full.com", "extra":"data"}
    p_empty_fields = {"name": "Test Full", "role":"guest"}
    merged_extra = merge_people(p_full, p_empty_fields)
    assert merged_extra["extra"] == "data"
    assert merged_extra["org"] == "FullOrg"

    merged_extra_rev = merge_people(p_empty_fields, p_full)
    assert merged_extra_rev["extra"] is None # 'extra' is not in ('href', 'org', 'title') so it's not merged from p_full
    assert merged_extra_rev["org"] == "FullOrg" # This should be picked up

# Minimal main for running tests with pytest
if __name__ == "__main__":
    pytest.main() 