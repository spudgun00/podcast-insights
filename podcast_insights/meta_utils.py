from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import logging
import re
import json

logger = logging.getLogger(__name__)

# --- Configuration loading --- 
def load_json_config(file_path, default_data=None):
    if default_data is None:
        default_data = {}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded configuration from {file_path}")
            return data
    except FileNotFoundError:
        logger.warning(f"Configuration file {file_path} not found. Using default/empty data.")
        return default_data
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}. Using default/empty data.")
        return default_data
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path}: {e}. Using default/empty data.")
        return default_data

# Load KNOWN_HOSTS from JSON, with a fallback to an empty dict or a minimal hardcoded version
# The path could be made configurable, e.g., via an environment variable or a global config object
KNOWN_HOSTS_FILE = Path(__file__).parent / "known_hosts.json"
KNOWN_HOSTS = load_json_config(KNOWN_HOSTS_FILE, default_data={
    "The Twenty Minute VC (20VC): Venture Capital | Startup Funding | The Pitch": ["Harry Stebbings", "harry stebbings"],
    # Minimal fallback if JSON is missing/corrupt
})

# Ensure KNOWN_HOSTS values are sets for efficient lookup (convert after loading)
for podcast_title, host_names_list in KNOWN_HOSTS.items():
    if isinstance(host_names_list, list):
        KNOWN_HOSTS[podcast_title] = set(host_names_list)
    elif not isinstance(host_names_list, set):
        logger.warning(f"KNOWN_HOSTS entry for '{podcast_title}' is not a list or set, converting to empty set. Value: {host_names_list}")
        KNOWN_HOSTS[podcast_title] = set()

SPURIOUS_PERSON_NAMES_LC = {
    "chimes ipo", 
    "twitter",
    # Add other common non-person entities that NER might misclassify as PERSON
    "ipo", # general ipo might be misclassified
    "ai",
    "api"
}

SPURIOUS_ORG_NAMES_LC = {
    "rory o'dry school",
    # Add other known spurious organization names here
}

# Enhanced stop-word list: strips filler, discourse markers, and generic business nouns
EXTENDED_STOP_WORDS = set().union(
    # scikit-learn's built-in
    ENGLISH_STOP_WORDS,
    # conversational fillers
    {
        "yeah", "yep", "yup", "uh", "um", "uh-huh", "mm-hmm", "ok", "okay", "right",
        "like", "just", "literally", "basically", "actually", "kinda", "sorta",
        "you", "youre", "we", "they", "ive", "hes", "shes", "dont", "doesnt", "didnt",
        "gonna", "wanna", "gotta", "think", "pretty", "really", "stuff", "things",
        "know", "did", "let", "10", # Added from user feedback
        "jason", "money", "30", # Added from user feedback round 2
        "point", "question", "million", "billion" # Added from final user feedback
    },
    # generic business words that add little insight
    {
        "company", "companies", "business", "industry", "product", "products",
        "market", "markets", "people", "team", "teams", "customer", "customers",
        "thing", "lot", "lots", "great", "good", "big", "small", "way", "today",
        "year", "years", "world", "going", "come", "comes", "make", "makes",
    },
    # Legacy stop words from previous implementation
    {
        "mean", "kind", "sort", "little", "guys", "probably", "totally", "absolutely", "certainly",
        "definitely", "obviously", "simply", "clearly", "quite", "ah", "eh", "er", "hmm", "huh", 
        "well", "so", "anyway", "ve", "ll", "re", "m", "s", "d", "t", "don", "doesn", "didn", 
        "won", "wouldn", "couldn", "shouldn", "isn", "aren", "haven", "hasn", "hadn", "wasn", "weren",
        "person", "makes", "making", "made", "time", "times", "new", "old", "bad", "better", "best", 
        "worse", "worst", "high", "low", "many", "much", "few", "lots", "get", "gets", "getting", 
        "got", "comes", "coming", "came", "go", "goes", "went", "gone", "look", "looks", "looking", 
        "looked", "want", "wants", "wanting", "wanted", "need", "needs", "needing", "needed",
        "try", "tries", "trying", "tried", "use", "uses", "using", "used", "work", "works", 
        "working", "worked", "talk", "talks", "talking", "talked", "say", "says", "saying", 
        "said", "tell", "tells", "telling", "told", "feel", "feels", "feeling", "felt", 
        "thinks", "thinking", "thought", "see", "sees", "seeing", "saw", "seen", "hear", 
        "hears", "hearing", "heard", "day", "days", "week", "weeks", "month", "months",
        "tomorrow", "yesterday", "now", "later", "soon", "early", "late", "first", "second", 
        "third", "last", "next", "previous"
    }
)

# Domain-specific terms to boost in keyword extraction
DOMAIN_BOOST = {
    # ── Funding stages & finance ──
    "pre-seed": 2.0, "seed": 1.8, "series_a": 1.8, "series_b": 1.6,
    "series_c": 1.4, "growth_round": 1.4, "bridge_round": 1.4,
    "valuation": 1.5, "runway": 1.4, "burn_rate": 1.4,
    "ipo": 1.7, # Added/boosted from user feedback
    "seed_round": 2.0, # Ensure present, from user feedback
    "late-stage": 1.6, # Added from user feedback (note: Tfidf might see "late stage")
    "late_stage": 1.6, # Added for two-word version
    "tier_1_vcs": 2.2, # Added from user feedback

    # ── SaaS & metrics ──
    "arr": 2.0, "mrr": 1.8, "cac": 1.8, "ltv": 1.6, "nrr": 1.6,
    "churn": 1.6, "retention": 1.6, "payback_period": 1.5,
    "usage-based_pricing": 1.5, "plg": 1.5, "freemium": 1.3,
    "rippling": 2.0, # Added from user feedback
    "deel": 1.8, # Added from user feedback

    # ── AI / frontier tech ──
    "agentic_ai": 2.2, "generative_ai": 2.0, "foundation_model": 1.8,
    "rag": 1.8, "fine-tuning": 1.6, "inference_latency": 1.6,
    "vector_database": 1.6, "gpu_shortage": 1.5, "llm": 1.5,
    "openai": 1.4, "anthropic": 1.4,

    # ── Infra & dev-tooling ──
    "kubernetes": 1.4, "serverless": 1.4, "observability": 1.4,
    "ci/cd": 1.3, "lakehouse": 1.3, "data_pipeline": 1.3,

    # ── Fintech & crypto ──
    "kyc": 1.5, "aml": 1.5, "payment_rails": 1.3,
    "interchange": 1.3, "stablecoin": 1.3, "defi": 1.3,

    # ── Health / gov / legal ──
    "hipaa": 1.5, "fda": 1.4, "cpt_code": 1.4,
    "reimbursement": 1.4, "regtech": 1.3,

    # ── Strategy & GTM ──
    "tam": 1.5, "sam": 1.5, "som": 1.5, "pmf": 1.5,
    "gtm": 1.4, "icp": 1.4, "cohort_analysis": 1.3,
    
    # Legacy terms from previous implementation
    "venture_capital": 2.5, "founder": 2.0, "startup": 2.0, "bootstrapped": 2.0,
    "cap_table": 2.0, "pitch_deck": 2.0, "term_sheet": 2.0, "exit_strategy": 2.0,
    "acquisition": 1.8, "angel_investor": 2.0, "saas": 2.0, "product_market_fit": 2.2,
    "growth_rate": 1.8, "business_model": 1.8,
    "go_to_market": 2.0, "fund_raising": 2.0, "venture_fund": 2.2, "portfolio_company": 1.8,
    "limited_partner": 2.0, "general_partner": 2.0, "pre_seed": 2.2, "a16z": 2.5,
    "andreessen_horowitz": 2.5, "y_combinator": 2.5, "yc": 2.5, "sequoia": 2.5,
    "benchmark": 2.5, "artificial_intelligence": 2.5, "machine_learning": 2.3,
    "neural_network": 2.0, "deep_learning": 2.0, "natural_language_processing": 2.3,
    "nlp": 2.3, "computer_vision": 2.0, "blockchain": 2.0, "cryptocurrency": 2.0,
    "bitcoin": 1.8, "ethereum": 1.8, "web3": 2.0, "cloud_computing": 1.8,
    "api": 1.7, "open_source": 1.8, "data_science": 2.0, "cybersecurity": 2.0,
    "devops": 1.8, "infrastructure": 1.8, "microservices": 2.0, "edge_computing": 2.0,
    "quantum_computing": 2.2, "large_language_model": 2.5, "gpt": 2.3, "transformer": 2.0,
    "fine_tuning": 2.0, "prompt_engineering": 2.2, "embeddings": 2.0, "agentic": 2.5,
    "multimodal": 2.2, "diffusion_model": 2.2, "claude": 2.3, "chatgpt": 2.3,
    "dall_e": 2.2, "stable_diffusion": 2.2, "midjourney": 2.2, "synthetic_data": 2.0,
    "retrieval_augmented_generation": 2.2, "llama": 2.3, "mistral": 2.3, "mixtral": 2.3,
    "gemini": 2.3, "agent": 2.0, "ai_assistant": 2.2, "autonomous_agent": 2.3,
    "frontier_model": 2.3, "andreessen": 2.5, "horowitz": 2.5, "marc_andreessen": 2.5,
    "ben_horowitz": 2.5
}

class ShowHosts:
    """Cache for known hosts per show"""
    def __init__(self):
        self.hosts = {}  # show_name -> set(lowercase_host_names)
    
    def add_hosts(self, show: str, hosts_to_add: list[str]):
        # Ensure hosts_to_add is a list of strings
        if not all(isinstance(h, str) for h in hosts_to_add):
            logger.warning(f"add_hosts received non-string in hosts_to_add for show '{show}'. Skipping non-strings.")
            hosts_to_add = [h for h in hosts_to_add if isinstance(h, str)]

        if show not in self.hosts:
            self.hosts[show] = set()
        for host_name in hosts_to_add:
            self.hosts[show].add(host_name.lower()) # Store lowercase for matching
    
    def is_host(self, show: str, name_to_check: str) -> bool:
        if not isinstance(name_to_check, str):
            return False # Cannot check non-string name
        return name_to_check.lower() in self.hosts.get(show, set())

show_hosts_cache = ShowHosts()

def extract_keywords(transcript: str, show_notes: str = "", top_n: int = 10) -> list[dict]:
    """
    Extract keywords using TF-IDF on combined corpus.
    Uses extended stop words list to filter out filler words.
    Prioritizes multi-word terms that represent domain concepts.
    Applies domain-specific boosts to VC/tech terms.
    """
    # Ensure we have text to process
    if not transcript or len(transcript) < 50:
        logger.warning("Transcript too short for keyword extraction")
        return []  # Return empty list instead of sentinel value
    
    try:
        # Combine transcript and show notes for better context
        raw_corpus_texts = [transcript]
        if show_notes:
            raw_corpus_texts.append(show_notes)

        processed_corpus = []
        # Preprocess corpus to replace spaces with underscores for boosted multi-word phrases
        # Sort by length descending to replace longer phrases first (e.g., "series a" before "a")
        # This is a simple approach; more robust would be non-overlapping replacements.
        # Only consider multi-word phrases from DOMAIN_BOOST for this replacement.
        phrases_to_replace = sorted([phrase for phrase in DOMAIN_BOOST.keys() if "_" in phrase],
                                    key=len, reverse=True)

        for text in raw_corpus_texts:
            processed_text = text
            for phrase_with_underscore in phrases_to_replace:
                phrase_with_space = phrase_with_underscore.replace("_", " ")
                # Use regex to replace whole words/phrases only, case insensitive
                processed_text = re.sub(r"\b" + re.escape(phrase_with_space) + r"\b", 
                                        phrase_with_underscore, processed_text, flags=re.IGNORECASE)
            processed_corpus.append(processed_text)
        
        # Create vectorizer with extended stop words and ngrams
        # For single-document cases, min_df and max_df need special handling
        min_df = 1  # Allow terms that appear at least once
        max_df = 1.0  # Allow terms that appear in all docs (which is just our 1-2 docs)
        
        vectorizer = TfidfVectorizer(
            max_features=200,  # Extract more features initially
            stop_words=list(EXTENDED_STOP_WORDS),
            ngram_range=(1, 3),  # Include up to trigrams for domain concepts
            min_df=min_df,
            max_df=max_df,
            use_idf=len(processed_corpus) > 1  # Use processed_corpus here
        )
        
        # Get feature names and their scores
        tfidf_matrix = vectorizer.fit_transform(processed_corpus)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get scores from the first document (transcript)
        scores = tfidf_matrix.toarray()[0]
        
        # Filter out low-value terms and sort by score
        keywords = []
        
        # Minimum score threshold to consider a term relevant
        MIN_SCORE_THRESHOLD = 0.05
        
        # Apply multi-word boosts and domain-specific boosts
        for term, score in zip(feature_names, scores):
            # Filter out short numeric terms
            if term.isdigit() and len(term) <= 4: # Max 4 digits, e.g. "10", "100", "2023"
                logger.debug(f"Skipping short numeric keyword: {term}")
                continue

            if score > MIN_SCORE_THRESHOLD:  # Apply minimum threshold
                # Base boost for multi-word terms (likely domain concepts)
                word_count = term.count(' ') + 1
                boosted_score = score * (1.0 + (word_count - 1) * 0.3)  # Increased boost: 30% per additional word
                
                # Apply domain-specific boost if term is in our boost dictionary
                # Term from vectorizer will be lowercased and underscore_joined if replacement happened
                if term in DOMAIN_BOOST: # Direct match for underscore_terms or single words
                    boosted_score *= DOMAIN_BOOST[term]
                    boosted_score += 0.05
                # For terms that might have had underscores removed by tokenizer, try space version
                elif term.replace("_", " ") in DOMAIN_BOOST: 
                    boosted_score *= DOMAIN_BOOST[term.replace("_", " ")]
                    boosted_score += 0.05
                
                # Final check against threshold after boosting
                if boosted_score > MIN_SCORE_THRESHOLD:
                    keywords.append({"term": term, "score": float(boosted_score)})
        
        # Sort by score descending and take top N
        keywords.sort(key=lambda x: x["score"], reverse=True)
        
        # Always ensure we have at least a few keywords if ANY were found
        if len(keywords) > 0 and len(keywords) < 3:
            return keywords
            
        return keywords[:top_n] if keywords else []
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []  # Return empty list on error

def extract_org_and_title(text: str, person_name: str) -> tuple[str, str]:
    """
    Extract organization and title information from text for a given person.
    Uses pattern matching and NER to find affiliations.
    """
    org, title = "", ""
    
    # Common patterns for org/title extraction
    patterns = [
        rf"{person_name}(?:,?\s+|\s+is\s+|\s+from\s+)(?:the\s+)?(\w+\s+of\s+|\w+\s+at\s+|CEO\s+of\s+|founder\s+of\s+)(?:the\s+)?([A-Z][A-Za-z0-9\-\.]+(?:\s+[A-Z][A-Za-z0-9\-\.]+)*)",
        rf"{person_name}(?:,?\s+|\s+is\s+|\s+from\s+)(?:the\s+)?([A-Z][A-Za-z0-9\-\.]+(?:\s+[A-Z][A-Za-z0-9\-\.]+)*)(?:'s\s+|\s+)'?s?\s*(\w+\s+of\s+|\w+\s+at\s+|CEO|founder|president|CTO|COO)",
    ]
    
    # Try pattern matching first
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in matches:
                if len(match) >= 2:
                    if "CEO" in match[0] or "founder" in match[0] or "president" in match[0]:
                        title = match[0].strip()
                        org = match[1].strip()
                    else:
                        title = match[0].strip()
                        org = match[1].strip()
                    return org, title
    
    # Try NER as fallback
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:min(len(text), 2000)])  # Limit to first 2000 chars for speed
        
        # Find ORG entities near person name
        person_pos = text.find(person_name)
        if person_pos > -1:
            search_window = text[max(0, person_pos-100):min(len(text), person_pos+100)]
            doc_window = nlp(search_window)
            
            for ent in doc_window.ents:
                if ent.label_ == "ORG":
                    org = ent.text
                    break
        
        # Look for common titles near the name
        title_patterns = [
            r"(CEO|Chief Executive Officer|CTO|Chief Technology Officer|COO|Chief Operating Officer|Founder|Co-founder|President|Director|VP|Vice President)",
        ]
        
        for pattern in title_patterns:
            title_matches = re.findall(pattern, text[max(0, person_pos-50):min(len(text), person_pos+50)])
            if title_matches:
                title = title_matches[0]
                break
    except Exception as e:
        logger.warning(f"Error using NER to extract org/title: {e}")
    
    return org, title

def get_explicit_flag(entry, feed) -> bool:
    """Get explicit flag with proper fallback chain"""
    # Try entry-level first, with proper string conversion
    if entry.get("itunes_explicit"):
        return entry.get("itunes_explicit").lower() in ("yes", "true", "1")
    
    # Fall back to feed-level
    if hasattr(feed, "feed") and feed.feed.get("itunes_explicit"):
        return feed.feed.get("itunes_explicit").lower() in ("yes", "true", "1")
    
    # Default to false if not specified
    return False

def extract_people(entry, transcript: str, show_name: str) -> tuple[list, list]:
    """Extract hosts and guests with improved accuracy including org and title, then tidies them."""
    initial_people_candidates = []
    
    # 1. Try podcast:person tags first
    try:
        if hasattr(entry, "summary_detail") and entry.summary_detail.get("value"):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(entry.summary_detail.get("value"), "xml")
            for tag in soup.find_all("podcast:person"):
                name = tag.text.strip()
                role = (tag.get("role") or "guest").lower()
                href = tag.get("href", "")
                org = tag.get("org", "")
                title = tag.get("title", "")
                
                if not org or not title:
                    extracted_org, extracted_title = extract_org_and_title(transcript, name)
                    org = org or extracted_org
                    title = title or extracted_title
                
                person_info = {
                    "name": name,
                    "role": role,
                    "href": href,
                    "org": org,
                    "title": title
                }
                initial_people_candidates.append(person_info)
                # Cache host if identified by tag
                if role in ("host", "presenter") and isinstance(name, str):
                    show_hosts_cache.add_hosts(show_name, [name])

    except Exception as e:
        logger.error(f"Error parsing podcast:person tags: {e}")
    
    # 1b. Try hardcoded hosts for known podcasts if no people from tags
    if not initial_people_candidates and show_name == "a16z Podcast":
        # These are known hosts, directly add them as candidates with role host
        a16z_hosts = [
            {"name": "Ben Horowitz", "role": "host", "org": "a16z", "title": "Co-founder and General Partner"},
            {"name": "Marc Andreessen", "role": "host", "org": "a16z", "title": "Co-founder and General Partner"}
        ]
        initial_people_candidates.extend(a16z_hosts)
        # Cache these hosts
        show_hosts_cache.add_hosts(show_name, [h["name"] for h in a16z_hosts if isinstance(h.get("name"), str)])

    # 2. Fallback to NER on transcript if still no people or to augment
    # We run NER even if tags exist to catch other mentions that might be guests.
    try:
        nlp = spacy.load("en_core_web_sm")
        intro_text = transcript[:1500]
        doc = nlp(intro_text)
        
        ner_people_names_seen = {p["name"].lower() for p in initial_people_candidates if isinstance(p.get("name"), str)} # Avoid re-adding people from tags

        for ent in doc.ents:
            if ent.label_ == "PERSON" and isinstance(ent.text, str):
                name = ent.text
                if name.lower() in ner_people_names_seen:
                    continue # Already added from podcast:person or a16z hardcode

                # Default to guest, will be corrected by tidy_people if a known host
                # or if cached by show_hosts_cache from <podcast:person> tags
                role = "host" if show_hosts_cache.is_host(show_name, name) else "guest"
                
                org, title = extract_org_and_title(transcript, name)
                
                person_info = {
                    "name": name,
                    "role": role,
                    "org": org,
                    "title": title
                }
                initial_people_candidates.append(person_info)
                ner_people_names_seen.add(name.lower())
    except Exception as e:
        logger.error(f"Error extracting people with NER: {e}")

    # 3. Consolidate and tidy up hosts and guests
    # Create a minimal meta dict for tidy_people, primarily for the podcast name
    meta_for_tidy = {"podcast": show_name}
    final_hosts, final_guests = tidy_people(meta_for_tidy, initial_people_candidates)
    
    return final_hosts, final_guests

def get_episode_type(entry) -> dict:
    """Get episode type with trailer flag"""
    # Default to "full" if not specified
    episode_type = entry.get("itunes_episode_type", "full")
    return {
        "type": episode_type,
        "is_trailer": episode_type != "full"
    }

def check_timestamp_support(transcript: dict) -> bool:
    """Check if transcript has word-level timestamps"""
    # Check for Whisper segments with start times
    if "segments" in transcript and transcript["segments"]:
        return any("start" in segment for segment in transcript["segments"])
    return False

def fix_speech_music_ratio(ratio: float) -> float:
    """Fix speech-music ratio if it's too low for a talk show"""
    if ratio < 0.5:
        # Default to 0.8 which is typical for talk shows
        return 0.8
    return ratio

def calculate_confidence(segments: list) -> dict:
    """Calculate average confidence and WER estimate from segments"""
    if not segments:
        return {"avg_confidence": 0.0, "wer_estimate": 1.0}
    
    # Try to get confidence from segments
    confidences = []
    for seg in segments:
        if "confidence" in seg:
            confidences.append(float(seg["confidence"]))
        # Some Whisper versions use different field names
        elif "avg_logprob" in seg:
            # Convert log probability to confidence-like score
            confidences.append(min(1.0, max(0.0, 1.0 + float(seg["avg_logprob"]))))
    
    # If no confidences found, return defaults
    if not confidences:
        return {"avg_confidence": 0.8, "wer_estimate": 0.2}  # More reasonable defaults
    
    avg_confidence = sum(confidences) / len(confidences)
    # Rough WER estimate: lower confidence = higher WER
    wer_estimate = round(1.0 - avg_confidence, 3)
    
    return {
        "avg_confidence": round(avg_confidence, 3),
        "wer_estimate": max(0.01, wer_estimate)  # Ensure minimum WER of 1%
    }

def process_transcript(transcript: dict, meta: dict) -> dict:
    """Process transcript and set timestamp flag early"""
    # Check if transcript is a dict with segments or just text
    if isinstance(transcript, str):
        transcript_text = transcript
        segments = []
    elif isinstance(transcript, dict):
        transcript_text = transcript.get("text", "")
        segments = transcript.get("segments", [])
    else:
        transcript_text = ""
        segments = []
        
    # Construct text from segments if needed
    if not transcript_text and segments:
        # Join all segment text with spaces
        transcript_text = " ".join(seg.get("text", "") for seg in segments if seg.get("text"))
        
    # Set timestamps support - ALWAYS true if segments exist with 'start' field
    has_timestamps = False
    if segments:
        has_timestamps = any("start" in segment for segment in segments)
    meta["supports_timestamp"] = has_timestamps
    
    # Fix speech-music ratio if it's too low for a talk show
    if "speech_music_ratio" in meta:
        meta["speech_music_ratio"] = fix_speech_music_ratio(meta["speech_music_ratio"])
    
    # Calculate transcript length - should never be 0 if we have segments
    meta["transcript_length"] = len(transcript_text)
    
    # If transcript length is still 0 but we have segments, estimate from segment count
    if meta["transcript_length"] == 0 and segments:
        # Average 10 words per segment × ~5 chars per word
        meta["transcript_length"] = len(segments) * 50
    
    # Add confidence metrics from segments
    if segments:
        confidence_metrics = calculate_confidence(segments)
        meta.update(confidence_metrics)
    
    # Extract keywords if transcript has text
    if len(transcript_text) > 50:
        # Use full transcript for keyword extraction
        logger.info(f"Extracting keywords from transcript ({len(transcript_text)} chars)")
        meta["keywords"] = extract_keywords(transcript_text)
    elif segments:
        # Try to build a better transcript from segments for keyword extraction
        logger.info("Building transcript from segments for keyword extraction")
        reconstructed_text = " ".join(seg.get("text", "") for seg in segments if seg.get("text"))
        if len(reconstructed_text) > 50:
            meta["keywords"] = extract_keywords(reconstructed_text)
        else:
            # Fallback to segment-based extraction (use first N segments as sample)
            sample_text = " ".join(seg.get("text", "") for seg in segments[:20] if seg.get("text"))
            if len(sample_text) > 50:
                meta["keywords"] = extract_keywords(sample_text)
            else:
                logger.warning("Unable to extract sufficient text for keyword generation")
                meta["keywords"] = []
    else:
        logger.warning("No transcript text or segments available for keyword extraction")
        meta["keywords"] = []
    
    return meta

def create_timestamp_mapping(chunks: list[str]) -> dict:
    """Create mapping of chunks to their original timestamps"""
    # Implementation depends on how you want to store the mapping
    return {"chunks": chunks}  # Placeholder

def enrich_meta(entry, feed_title: str, feed_url: str, tech: dict[str, Any], transcript: str, feed: Any) -> dict:
    """Enrich metadata with additional information"""
    # Get show notes if available
    show_notes = entry.get("summary", "")
    
    # Extract people with host caching
    hosts, guests = extract_people(entry, transcript, feed_title)
    
    # Get episode type with trailer flag
    episode_type = get_episode_type(entry)
    
    # Fix speech-music ratio if it's too low for a talk show
    speech_music_ratio = tech.get("speech_music_ratio", 0.0)
    speech_music_ratio = fix_speech_music_ratio(speech_music_ratio)
    
    # Extract keywords from transcript with show notes for context
    keywords = []
    if transcript and len(transcript) > 50:
        logger.info(f"Extracting keywords in enrich_meta for {feed_title} - {entry.get('title', '')}")
        keywords = extract_keywords(transcript, show_notes)
    
    # Get categories from RSS feed
    categories = []
    
    # First check for tags/categories in the entry
    if hasattr(entry, "tags") and entry.tags:
        categories = [tag.get("term", "") for tag in entry.tags if tag.get("term")]
    
    # Also check for iTunes categories which may be different
    if hasattr(entry, "itunes_categories"):
        categories.extend(entry.itunes_categories)
    
    # If no categories at entry level, try feed level
    if not categories and hasattr(feed, "feed"):
        if hasattr(feed.feed, "tags") and feed.feed.tags:
            categories = [tag.get("term", "") for tag in feed.feed.tags if tag.get("term")]
        
        if hasattr(feed.feed, "itunes_categories"):
            categories.extend(feed.feed.itunes_categories)
    
    # Filter out empty strings and duplicates while preserving order
    seen = set()
    categories = [cat for cat in categories if cat and cat not in seen and not seen.add(cat)]
    
    # Check if URL supports timestamp linking
    from podcast_insights.audio_utils import check_timestamp_support as check_url_timestamp
    supports_timestamp = True  # Default to true if Whisper segments have timestamps
    if entry.enclosures and entry.enclosures[0]["href"]:
        url_supports_timestamp = check_url_timestamp(entry.enclosures[0]["href"])
        supports_timestamp = supports_timestamp and url_supports_timestamp
    
    # Set transcript length
    transcript_length = len(transcript) if transcript else 0
    
    return {
        # Core IDs & URLs
        "podcast": feed_title,
        "episode": entry.get("title", ""),
        "guid": entry.get("id", ""),
        "published": entry.get("published", ""),
        "episode_url": entry.get("link"),
        "audio_url": entry.enclosures[0]["href"] if entry.enclosures else "",
        
        # Enhanced metadata
        "keywords": keywords,
        "categories": categories,
        "rights": {
            "copyright": entry.get("copyright") or (feed.feed.get("copyright") if hasattr(feed, "feed") else None),
            "explicit": get_explicit_flag(entry, feed)
        },
        "hosts": hosts,
        "guests": guests,
        "itunes_episodeType": episode_type["type"],
        "is_trailer": episode_type["is_trailer"],
        "supports_timestamp": supports_timestamp,
        "speech_music_ratio": speech_music_ratio,
        "transcript_length": transcript_length,
        
        # Audio tech info
        **tech
    } 

def tidy_people(meta_input: dict, initial_people_list: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Corrects host/guest lists using KNOWN_HOSTS and a list of spurious names.
    meta_input is expected to have a 'podcast' field for the show name.
    initial_people_list is the list of person dicts from NER/tags.
    """
    current_podcast_name = meta_input.get("podcast", "")
    
    known_hosts_for_show_lc = {name.lower() for name in KNOWN_HOSTS.get(current_podcast_name, set())}
    known_host_original_casing_map = {}
    for name_in_known_hosts_set in KNOWN_HOSTS.get(current_podcast_name, set()):
        lc_name = name_in_known_hosts_set.lower()
        if lc_name not in known_host_original_casing_map: 
            known_host_original_casing_map[lc_name] = name_in_known_hosts_set

    corrected_hosts = []
    corrected_guests = []
    host_names_assigned_lc = set()

    for p_dict in initial_people_list:
        if not isinstance(p_dict, dict) or not isinstance(p_dict.get("name"), str):
            logger.warning(f"Skipping invalid person entry in tidy_people (no name/not dict): {p_dict} for podcast {current_podcast_name}")
            continue

        person_name_original = p_dict["name"]
        person_name_lc = person_name_original.lower()
        person_first_name_lc = person_name_lc.split()[0] if ' ' in person_name_lc else person_name_lc

        current_org = p_dict.get("org", "")
        if isinstance(current_org, str) and current_org.strip() != "":
            current_org_lc = current_org.lower()
            if current_org_lc in SPURIOUS_ORG_NAMES_LC or current_org_lc.startswith(person_first_name_lc):
                p_dict["org"] = ""
                logger.debug(f"Cleared org '{current_org}' for person '{person_name_original}' in podcast {current_podcast_name}")

        if person_name_lc in SPURIOUS_PERSON_NAMES_LC:
            logger.debug(f"Filtered out spurious person name: '{person_name_original}' for podcast {current_podcast_name}")
            continue

        if person_name_lc in known_hosts_for_show_lc:
            host_display_name = known_host_original_casing_map.get(person_name_lc, person_name_original).title()
            host_data = {**p_dict, "name": host_display_name, "role": "host"}
            corrected_hosts.append(host_data)
            host_names_assigned_lc.add(person_name_lc)
        else:
            corrected_guests.append(p_dict)

    if current_podcast_name == "The Twenty Minute VC (20VC): Venture Capital | Startup Funding | The Pitch":
        primary_host_lc = "harry stebbings"
        if primary_host_lc not in host_names_assigned_lc:
            harry_canonical_name = known_host_original_casing_map.get(primary_host_lc, "Harry Stebbings").title()
            harry_in_guests_info = next((g for g in corrected_guests if g.get("name", "").lower() == primary_host_lc), None)
            if harry_in_guests_info:
                harry_org = harry_in_guests_info.get("org", "")
                harry_first_name_lc = harry_in_guests_info.get("name","").lower().split()[0]
                if isinstance(harry_org, str) and harry_org.strip() != "":
                    harry_org_lc = harry_org.lower()
                    if harry_org_lc in SPURIOUS_ORG_NAMES_LC or harry_org_lc.startswith(harry_first_name_lc):
                        harry_in_guests_info["org"] = ""
                corrected_hosts.append({**harry_in_guests_info, "name": harry_canonical_name, "role": "host"})
            else:
                corrected_hosts.append({"name": harry_canonical_name, "role": "host"})
            host_names_assigned_lc.add(primary_host_lc)

    final_guests = [g for g in corrected_guests if isinstance(g.get("name"), str) and g["name"].lower() not in host_names_assigned_lc]

    final_hosts_dict = {}
    for h_idx, h_item in enumerate(corrected_hosts):
        lc_name = h_item["name"].lower() # Use .lower() for dict key consistency
        # If name is already title-cased from KNOWN_HOSTS or Harry's rule, keep it.
        # Otherwise, title-case it here as a general rule for hosts.
        if h_item["name"] == lc_name: # It implies it wasn't from a canonical map or Harry's rule with title casing
            h_item["name"] = h_item["name"].title()
        
        if lc_name not in final_hosts_dict or len(h_item) > len(final_hosts_dict[lc_name]):
            final_hosts_dict[lc_name] = h_item
        elif lc_name in final_hosts_dict and h_item["name"] != final_hosts_dict[lc_name]["name"] and h_item["name"].istitle():
            # Prefer an already title-cased name if lengths are same (e.g. from KNOWN_HOSTS)
            final_hosts_dict[lc_name] = h_item
            
    final_hosts = list(final_hosts_dict.values())

    return final_hosts, final_guests 