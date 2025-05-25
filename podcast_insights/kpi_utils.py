import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def parse_monetary_value(value_str: str) -> Optional[float]:
    """Converts a monetary string (e.g., "$10.5M", "250K", "€1,200.50") to a float.
       Returns None if parsing fails.
    """
    if not isinstance(value_str, str):
        return None

    # Remove currency symbols and commas
    cleaned_str = re.sub(r"[$,€£¥]", "", value_str)
    cleaned_str = cleaned_str.replace(",", "")

    multiplier = 1.0
    if cleaned_str.upper().endswith('K'):
        multiplier = 1_000.0
        cleaned_str = cleaned_str[:-1]
    elif cleaned_str.upper().endswith('M'):
        multiplier = 1_000_000.0
        cleaned_str = cleaned_str[:-1]
    elif cleaned_str.upper().endswith('B'):
        multiplier = 1_000_000_000.0
        cleaned_str = cleaned_str[:-1]
    
    try:
        return float(cleaned_str) * multiplier
    except ValueError:
        logger.debug(f"Could not parse monetary value: {value_str}")
        return None

KPI_PATTERNS = [
    {
        "type": "MONEY",
        "regex": r"([$€£¥]?\s*\d[\d,]*\.?\d*\s*[KMBkmb]?)(?!\s*%(?:\s|$))", # Exclude if followed by %
        "value_parser": parse_monetary_value,
        "context_words": None 
    },
    {
        "type": "PERCENTAGE",
        "regex": r"(\d[\d,]*\.?\d*\s*%)",
        "value_parser": lambda x: float(x.replace("%", "").replace(",", "").strip()) if x else None,
        "context_words": None
    },
    {
        "type": "ARR", 
        "regex": r"(?i)((?:\$[\€\£\¥]?\s*\d[\d,]*\.?\d*\s*[KMBkmb]?|\d[\d,]*\.?\d*\s*[KMBkmb]?\s*dollars|euros|pounds|yen)\s*(?:in|of)?\s*ARR|ARR\s*(?:of|is|was|at)?\s*\$[\€\£\¥]?\s*\d[\d,]*\.?\d*\s*[KMBkmb]?)",
        "value_extractor_regex": r"[\$€£¥]?\s*\d[\d,]*\.?\d*\s*[KMBkmb]?", 
        "value_parser": parse_monetary_value,
        "context_words": ["ARR", "Annual Recurring Revenue"]
    },
    {
        "type": "MRR", 
        "regex": r"(?i)((?:\$[\€\£\¥]?\s*\d[\d,]*\.?\d*\s*[KMBkmb]?|\d[\d,]*\.?\d*\s*[KMBkmb]?\s*dollars|euros|pounds|yen)\s*(?:in|of)?\s*MRR|MRR\s*(?:of|is|was|at)?\s*\$[\€\£\¥]?\s*\d[\d,]*\.?\d*\s*[KMBkmb]?)",
        "value_extractor_regex": r"[\$€£¥]?\s*\d[\d,]*\.?\d*\s*[KMBkmb]?",
        "value_parser": parse_monetary_value,
        "context_words": ["MRR", "Monthly Recurring Revenue"]
    },
    {
        "type": "FUNDING_ROUND_MONEY", 
        "regex": r"(?i)((?:Series\s+[A-Za-z](?:\+\+?)?|Seed(?:\s+Round)?|Angel(?:\s+Round)?)(?:\s+round)?(?:\s+of|\s+worth|\s+at)?\s+\$[\d,]+\.?\d*\s*[KMBkmb]?|\$[\d,]+\.?\d*\s*[KMBkmb]?\s+(?:in\s+)?(?:Series\s+[A-Za-z](?:\+\+?)?|Seed(?:\s+Round)?|Angel(?:\s+Round)?)(?:\s+round)?)",
        "value_extractor_regex": r"\$[\d,]+\.?\d*\s*[KMBkmb]?",
        "value_parser": parse_monetary_value,
        "context_words": ["series", "seed", "angel", "round", "funding"]
    },
    {
        "type": "VALUATION",
        "regex": r"(?i)((?:valued\s+at|valuation\s+of|a\s+\$[\d,]+\.?\d*\s*[KMBkmb]?\s+valuation))",
        "value_extractor_regex": r"\$[\d,]+\.?\d*\s*[KMBkmb]?",
        "value_parser": parse_monetary_value,
        "context_words": ["valuation", "valued at"]
    }
]

def extract_kpis(transcript_text: str) -> List[Dict[str, Any]]:
    """Extracts Key Performance Indicators (KPIs) from transcript text using regex patterns."""
    if not transcript_text or not isinstance(transcript_text, str):
        logger.warning("Transcript text is empty or not a string. Skipping KPI extraction.")
        return []

    extracted_kpis: List[Dict[str, Any]] = []
    matched_spans: List[Tuple[int, int]] = [] 

    for kpi_def in KPI_PATTERNS:
        try:
            pattern = re.compile(kpi_def["regex"])
            for match in pattern.finditer(transcript_text):
                start_char, end_char = match.span()
                
                is_sub_match_of_specific_kpi = False
                if kpi_def["type"] == "MONEY": 
                    for ms_start, ms_end in matched_spans:
                        if ms_start <= start_char and end_char <= ms_end:
                            is_sub_match_of_specific_kpi = True
                            break
                if is_sub_match_of_specific_kpi:
                    continue

                matched_text = match.group(1)
                value_to_parse = matched_text
                
                if kpi_def.get("value_extractor_regex"):
                    value_match = re.search(kpi_def["value_extractor_regex"], matched_text)
                    if value_match:
                        value_to_parse = value_match.group(0)
                    else:
                        logger.debug(f"Could not extract value part for {kpi_def['type']} from '{matched_text}'. Skipping value parsing for this instance.")
                        value_to_parse = None 
                
                parsed_value = None
                if value_to_parse and kpi_def['value_parser']:
                    parsed_value = kpi_def['value_parser'](value_to_parse)
                
                if parsed_value is None and kpi_def['type'] not in ['OTHER_NON_NUMERIC_KPI'] :
                    logger.debug(f"Parsed value is None for matched text '{matched_text}' of type '{kpi_def['type']}'. Skipping this KPI instance.")
                    continue

                kpi_entry = {
                    "text": matched_text.strip(),
                    "type": kpi_def['type'],
                    "value": parsed_value,
                    "start_char": start_char,
                    "end_char": end_char,
                    "raw_value_parsed": value_to_parse 
                }
                extracted_kpis.append(kpi_entry)
                
                if kpi_def['type'] not in ['MONEY', 'PERCENTAGE']:
                    matched_spans.append((start_char, end_char))

        except re.error as e:
            logger.error(f"Regex error for KPI type {kpi_def.get('type', 'UNKNOWN')}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during KPI extraction for type {kpi_def.get('type', 'UNKNOWN')}: {e}")

    extracted_kpis.sort(key=lambda x: x['start_char'])
    
    # --- Post-filtering based on ChatGPT review ---
    final_kpis = []
    currency_symbols = "$€£¥"
    for kpi in extracted_kpis:
        if kpi["type"] == "MONEY":
            parsed_value = kpi.get("value")
            raw_text_matched = kpi.get("text", "") # The full text matched by the regex for this KPI
            
            has_currency_symbol = any(symbol in raw_text_matched for symbol in currency_symbols)
            
            # Condition: Drop MONEY if value < 1000 AND no currency symbol was in the matched text.
            if parsed_value is not None and parsed_value < 1000 and not has_currency_symbol:
                logger.debug(f"Post-filtering MONEY KPI: '{raw_text_matched}' (value: {parsed_value}) because value < 1000 and no currency symbol.")
                continue # Skip this KPI
        final_kpis.append(kpi)
    # --- End Post-filtering ---

    logger.info(f"Extracted {len(final_kpis)} KPIs after post-filtering (originally {len(extracted_kpis)}).")
    return final_kpis


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    sample_text = """
    The company announced $10.5M in ARR and a 25% growth. Our MRR is now $900k.
    They raised a $5M Series A round. Last year, revenue was 1,200,000 dollars.
    This cost $50. Their valuation is $100M. We hit 50% profit margin. And a Series B of $20M.
    Also, plain numbers like 50000 dollars. Or €60K. Or £7.5B.
    A simple $100. And 15%. MRR of $1.2M. $500K ARR.
    This is just $5, not $5% or $5M%.
    Series C funding of $50 million was great.
    """
    
    kpis = extract_kpis(sample_text)
    for kpi in kpis:
        print(kpi)

    print("\n--- Test monetary parser ---")
    test_values = ["$10.5M", "250K", "€1,200.50", "$1B", "100", "£5k", "¥1000000", "text", None, "$1.2.3M"]
    for tv in test_values:
        print(f"'{tv}' -> {parse_monetary_value(tv)}")
 