# tests/test_kpi_utils.py
import pytest
from podcast_insights.kpi_utils import extract_kpis, parse_monetary_value

SAMPLE_TEXT_FULL = """
  In Q1 we hit $2.5 M ARR with gross margins of 78 %. Our MRR is $200K.
  Series A brought in $12 million at a $60 million post-money valuation.
  Marketing spend is 40 % of revenue; CAC sits at $450 (a general money amount).
  This is a simple $100. Another percentage is 15.5%.
  The company aims for an ARR of $5M by next year.
  A seed round of $500k was also mentioned.
  Valued at $1B in the last round.
"""

def test_basic_extraction_and_values():
    kpis = extract_kpis(SAMPLE_TEXT_FULL)
    
    # For easier assertion, create a dictionary mapping type to a list of found KPIs of that type
    kpis_by_type = {}
    for kpi in kpis:
        kpis_by_type.setdefault(kpi["type"], []).append(kpi)

    # --- MONEY (general, not part of ARR/MRR/Funding/Valuation) ---
    assert "MONEY" in kpis_by_type
    money_values = sorted([k["value"] for k in kpis_by_type["MONEY"]])
    # Expected: $450 -> 450.0, $100 -> 100.0
    assert money_values == [100.0, 450.0]
    # Check one instance for text
    assert any(k["text"] == "$450" and k["value"] == 450.0 for k in kpis_by_type["MONEY"]) 

    # --- PERCENTAGE ---
    assert "PERCENTAGE" in kpis_by_type
    percentage_values = sorted([k["value"] for k in kpis_by_type["PERCENTAGE"]])
    # Expected: 78 % -> 78.0, 40 % -> 40.0, 15.5% -> 15.5
    assert percentage_values == [15.5, 40.0, 78.0]
    assert any(k["text"] == "78 %" and k["value"] == 78.0 for k in kpis_by_type["PERCENTAGE"])

    # --- ARR ---    
    assert "ARR" in kpis_by_type
    arr_kpis = kpis_by_type["ARR"]
    # Expected: $2.5 M ARR -> 2,500,000; ARR of $5M -> 5,000,000
    arr_values = sorted([k["value"] for k in arr_kpis])
    assert arr_values == [2_500_000.0, 5_000_000.0]
    assert any(k["text"] == "$2.5 M ARR" and k["value"] == 2_500_000.0 for k in arr_kpis)
    assert any(k["text"] == "ARR of $5M" and k["value"] == 5_000_000.0 for k in arr_kpis)

    # --- MRR ---
    assert "MRR" in kpis_by_type
    mrr_kpis = kpis_by_type["MRR"]
    assert len(mrr_kpis) == 1
    assert mrr_kpis[0]["value"] == 200_000.0
    assert mrr_kpis[0]["text"] == "MRR is $200K"

    # --- FUNDING_ROUND_MONEY ---
    assert "FUNDING_ROUND_MONEY" in kpis_by_type
    funding_kpis = kpis_by_type["FUNDING_ROUND_MONEY"]
    funding_values = sorted([k["value"] for k in funding_kpis])
    # Expected: $12 million (Series A) -> 12,000,000; $500k (seed round) -> 500,000
    assert funding_values == [500_000.0, 12_000_000.0]
    assert any(k["text"] == "$12 million" and k["type"]=="FUNDING_ROUND_MONEY" for k in funding_kpis) # Regex captures more, check specific
    assert any(k["text"] == "$500k" and k["type"]=="FUNDING_ROUND_MONEY" for k in funding_kpis)

    # --- VALUATION ---
    assert "VALUATION" in kpis_by_type
    valuation_kpis = kpis_by_type["VALUATION"]
    valuation_values = sorted([k["value"] for k in valuation_kpis])
    # Expected: $60 million post-money valuation -> 60,000,000; Valued at $1B -> 1,000,000,000
    assert valuation_values == [60_000_000.0, 1_000_000_000.0]
    assert any(k["text"] == "$60 million post-money valuation" and k["value"] == 60_000_000.0 for k in valuation_kpis)
    assert any(k["text"] == "Valued at $1B" and k["value"] == 1_000_000_000.0 for k in valuation_kpis)

    # Ensure spans are increasing (no overlaps / duplicates that were not handled by prefering specific KPIs)
    starts = [k["start_char"] for k in kpis]
    assert starts == sorted(list(set(starts))), "KPI start characters are not unique and sorted, implies overlap or ordering issue."

def test_no_kpis_returns_empty():
    text = "We chatted about product-market fit and culture today."
    assert extract_kpis(text) == []

def test_empty_string_returns_empty():
    assert extract_kpis("") == []

def test_none_input_returns_empty():
    assert extract_kpis(None) == [] # type: ignore

def test_monetary_parser_edge_cases():
    assert parse_monetary_value("$1.2.3M") is None
    assert parse_monetary_value("100K USD") == 100_000.0 # Assumes K is at end after cleaning currency symbols
    assert parse_monetary_value("€50,000.75") == 50000.75
    assert parse_monetary_value("text only") is None
    assert parse_monetary_value("$M") is None # Number part is missing
    assert parse_monetary_value("1000") == 1000.0
    assert parse_monetary_value("2.5b") == 2_500_000_000.0

def test_kpi_ordering_and_deduplication():
    # Test that more specific KPIs (ARR, MRR) prevent generic MONEY KPIs for the same value string
    # and that the order of KPI_PATTERNS definition doesn't negatively impact this.
    text = "Our ARR is $5M and that $5M is great."
    kpis = extract_kpis(text)
    
    arr_count = 0
    money_count = 0
    found_arr_text = None

    for kpi in kpis:
        if kpi["type"] == "ARR":
            arr_count += 1
            assert kpi["value"] == 5_000_000.0
            found_arr_text = kpi["text"]
        elif kpi["type"] == "MONEY":
            money_count +=1 
            # This MONEY entry should not be the one from ARR
            assert kpi["text"] != found_arr_text # Ensure the text isn't the exact ARR phrase
            # More robustly, check if the span of this MONEY kpi is within the span of the ARR kpi
            # This requires spans of ARR to be known when checking MONEY, current logic tries this.
            
    assert arr_count == 1, f"Expected 1 ARR KPI, got {arr_count}. KPIs: {kpis}"
    # Depending on how MONEY regex and the dedupe logic works with context, 
    # the second "$5M" might be caught as MONEY or not. 
    # Current MONEY regex: r"([$€£¥]?\s*\d[\d,]*\.?\d*\s*[KMBkmb]?)(?!\s*%(?:\s|$))"
    # It will match "$5M". The dedupe logic `is_sub_match_of_specific_kpi` should prevent it if it's inside the ARR span.
    # The text "Our ARR is $5M" will be one ARR kpi. Span for this is (e.g. 12, 24)
    # The text "that $5M is great." will be one MONEY kpi. Span is (e.g. 29, 32)
    # So, they are distinct. money_count should be 1.
    assert money_count == 1, f"Expected 1 standalone MONEY KPI, got {money_count}. KPIs: {kpis}"

    # Test case: Percentage shouldn't be misidentified as money
    text_perc = "Increase of 20% not $20M."
    kpis_perc = extract_kpis(text_perc)
    assert any(k["type"] == "PERCENTAGE" and k["value"] == 20.0 for k in kpis_perc)
    assert not any(k["type"] == "MONEY" and k["text"] == "20%" for k in kpis_perc)
    assert any(k["type"] == "MONEY" and k["text"] == "$20M" for k in kpis_perc) 