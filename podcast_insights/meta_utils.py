from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import logging
import re
import json
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import unicodedata
import datetime as dt # Already imported as datetime by psycopg2, ensure no conflict or use one consistently

logger = logging.getLogger(__name__)

# --- Helper function for SpaCy Entity Caching ---
def _generate_spacy_entities_file(
    transcript_text: str, 
    guid: str, 
    base_data_dir: Path, 
    nlp_model: spacy.Language
) -> Optional[str]:
    """Generates and saves SpaCy entities to a JSON file, returns the file path or None."""
    if not transcript_text or not guid or not nlp_model:
        logger.warning("Skipping SpaCy entity generation due to missing text, GUID, or model.")
        return None
    try:
        doc = nlp_model(transcript_text)
        entities = [{"text":e.text, "type":e.label_, "start_char":e.start_char, "end_char":e.end_char} for e in doc.ents]
        
        entities_dir = base_data_dir / "entities"
        entities_dir.mkdir(parents=True, exist_ok=True)
        entities_path = entities_dir / f"{guid}.json"
        with open(entities_path, "w", encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved NER entities to {entities_path}")
        return str(entities_path.resolve())
    except Exception as e:
        logger.error(f"Failed to generate/save SpaCy entities for GUID {guid}: {e}")
        return None

# --- Helper function for Sentence Embedding Caching ---
def _generate_sentence_embedding_file(
    transcript_text: str, 
    guid: str, 
    base_data_dir: Path, 
    st_model: SentenceTransformer
) -> Optional[str]:
    """Generates and saves sentence embedding to a .npy file, returns the file path or None."""
    if not transcript_text or not guid or not st_model:
        logger.warning("Skipping sentence embedding generation due to missing text, GUID, or model.")
        return None
    try:
        embedding = st_model.encode(transcript_text, convert_to_numpy=True)
        
        embeddings_dir = base_data_dir / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = embeddings_dir / f"{guid}.npy"
        np.save(embedding_path, embedding)
        logger.info(f"Saved sentence embedding to {embedding_path}")
        return str(embedding_path.resolve())
    except Exception as e:
        logger.error(f"Failed to generate/save sentence embedding for GUID {guid}: {e}")
        return None

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

ALIAS_MAP_FILE = Path(__file__).parent / "alias_map.json"
ALIAS_MAP = load_json_config(ALIAS_MAP_FILE, default_data={})

# TODO: Future enhancement for context-aware aliasing:
# If an alias (e.g., "jason") can map to multiple canonical names 
# depending on the show or other context (e.g., co-occurring org/company words 
# in the same sentence/segment), the current static ALIAS_MAP will not suffice.
# This would require a more sophisticated aliasing mechanism, potentially involving:
# 1. Storing multiple canonical options for an alias.
# 2. Passing more context (like show name, surrounding text) to the aliasing logic.
# 3. Using heuristics or a model to disambiguate based on context.

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
    "api",
    "kajabi", # Added from entity list feedback
    "mode" # Added from entity list feedback
}

SPURIOUS_ORG_NAMES_LC = {
    "rory o'dry school",
    # Add other known spurious organization names here
    "ipo" # Added from entity list feedback
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
        "point", "question", "million", "billion", # Added from final user feedback
        "doing", # Added from latest feedback to remove keyword noise
        "interesting", # Added from latest feedback
        "didn", # Added from new keyword list
        "mm", # Added to address sklearn warning
        "does" # Added from new keyword list
    },
    # generic business words that add little insight
    {
        "company", "companies", "business", "industry", "product", "products",
        "market", "markets", "people", "team", "teams", "customer", "customers",
        "thing", "lot", "lots", "great", "good", "big", "small", "way", "today",
        "year", "years", "world", "going", "come", "comes", "make", "makes",
        "deal", "capital" # Added from new feedback to reduce keyword noise
    },
    # Legacy stop words from previous implementation
    {
        "mean", "kind", "sort", "little", "guys", "probably", "totally", "absolutely", "certainly",
        "definitely", "obviously", "simply", "clearly", "quite", "ah", "eh", "er", "hmm", "huh", 
        "well", "so", "anyway", "ve", "ll", "re", "m", "s", "d", "t", "don", "doesn", "didnt", 
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
        "third", "last", "next", "previous",
        "deals" # Added based on latest keyword output
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
    explicit_val = entry.get("itunes_explicit")
    if explicit_val is not None:
        return explicit_val.lower() == "yes"
    
    # Fall back to feed-level
    if hasattr(feed, "feed") and feed.feed.get("itunes_explicit"):
        feed_explicit_val = feed.feed.get("itunes_explicit")
        if feed_explicit_val is not None:
            return feed_explicit_val.lower() == "yes"
    
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

def enrich_meta(
    entry: Dict[str, Any], 
    feed_title: str, 
    feed_url: str, 
    tech: Dict[str, Any], 
    transcript_text: str,
    feed: Any,
    nlp_model: spacy.Language | None = None,
    st_model: SentenceTransformer | None = None,
    base_data_dir: Path | str = "data",
    perform_caching: bool = True
) -> Dict[str, Any]:
    """
    Enriches episode metadata with information from feed, tech details, and transcript.
    Optionally caches NER entities and sentence embeddings if models are provided and perform_caching is True.
    """
    meta = {}
    
    # Ensure base_data_dir is a Path object
    if isinstance(base_data_dir, str):
        base_data_dir = Path(base_data_dir)

    # Basic feed info (ensure keys exist in 'entry' or provide defaults)
    meta['podcast'] = feed_title
    meta['episode'] = entry.get('title', 'N/A')
    # Use entry.id for guid if available, otherwise entry.guid or fallback to new uuid
    # meta['guid'] = entry.get('id', entry.get('guid', str(uuid.uuid4()))) # uuid needs import
    # For now, assume guid is one of these keys or handle missing guid appropriately
    if 'id' in entry and entry.id:
        meta['guid'] = entry.id
    elif 'guid' in entry and entry.guid:
        meta['guid'] = entry.guid
    else:
        # Fallback, though a stable GUID is preferred. This might occur if entry lacks common ID fields.
        # Consider logging a warning if a GUID has to be generated.
        # For now, let's assume a GUID will be found or this part needs robust handling.
        logger.warning(f"No stable 'id' or 'guid' found for entry: {entry.get('title')}. Placeholder used.")
        meta['guid'] = f"missing_guid_{entry.get('title', 'unknown_episode').replace(' ', '_')[:20]}"


    meta['published'] = entry.get('published', 'N/A')
    meta['episode_url'] = entry.get('link', 'N/A')
    
    # Get audio_url directly from the entry (parsed feed item) if possible
    audio_url_from_entry = None
    if entry.get("enclosures") and len(entry["enclosures"]) > 0 and entry["enclosures"][0].get("href"):
        audio_url_from_entry = entry["enclosures"][0]["href"]
    
    meta['audio_url'] = audio_url_from_entry or tech.get('audio_url', 'N/A') # Prioritize feed, fallback to tech

    # Keywords from transcript (using existing function)
    # Assuming extract_keywords and tidy_people are defined in this file
    # And that `transcript_text` is the full transcript string
    show_notes_text = entry.get('summary', '')
    meta['keywords'] = extract_keywords(transcript_text, show_notes_text)
    
    # Categories from feed
    meta['categories'] = sorted(list(set(term['term'].lower() for term in entry.get('tags', []) if term['term']))) 

    # Rights and explicit flag
    meta['rights'] = {
        'copyright': feed.feed.get('rights', 'N/A'),
        'explicit': get_explicit_flag(entry, feed.feed)
    }

    # Hosts and Guests (using existing function)
    # Ensure KNOWN_HOSTS is populated correctly if used by extract_people
    # Need to confirm show_name source; using feed_title for now
    initial_hosts, initial_guests = extract_people(entry, transcript_text, feed_title) 
    meta['hosts'], meta['guests'] = tidy_people(meta, initial_hosts + initial_guests)

    # iTunes specific
    itunes_info = get_episode_type(entry)
    meta['itunes_episodeType'] = itunes_info['type']
    meta['is_trailer'] = itunes_info['is_trailer']

    # Transcript specific details (from tech dictionary or process_transcript)
    meta['supports_timestamp'] = tech.get('supports_timestamp', False) 
    meta['speech_music_ratio'] = fix_speech_music_ratio(tech.get('speech_music_ratio', 0.0))
    meta['transcript_length'] = tech.get('transcript_length', 0)
    meta['sample_rate_hz'] = tech.get('sample_rate_hz', 0)
    meta['bitrate_kbps'] = tech.get('bitrate_kbps', 0)
    meta['duration_sec'] = tech.get('duration_sec', 0.0)
    
    # Confidence and WER (from tech or calculate_confidence if segments available)
    confidence_info = tech.get('confidence', {})
    meta['avg_confidence'] = confidence_info.get('avg_confidence', 0.0)
    meta['wer_estimate'] = confidence_info.get('wer_estimate', 0.0)
    
    meta['segment_count'] = tech.get('segment_count', 0)
    meta['chunk_count'] = tech.get('chunk_count', 0)

    # Technical metadata like file paths and hashes
    meta['audio_hash'] = tech.get('audio_hash', 'N/A')
    meta['download_path'] = tech.get('download_path', 'N/A')
    meta['transcript_path'] = tech.get('transcript_path', 'N/A')

    # --- Cache NER Entities and Sentence Embeddings --- 
    meta["entities_path"] = None # Initialize to None
    meta["embedding_path"] = None # Initialize to None
    meta["cleaned_entities_path"] = None # Initialize for cleaned entities
    meta["entity_stats"] = {} # Initialize for entity stats

    if perform_caching and meta.get('guid') and meta['guid'] != f"missing_guid_{entry.get('title', 'unknown_episode').replace(' ', '_')[:20]}":
        # 1. SpaCy Entities (Raw)
        if nlp_model:
            raw_entities_path = _generate_spacy_entities_file(
                transcript_text, meta['guid'], base_data_dir, nlp_model
            )
            meta["entities_path"] = raw_entities_path

            # ---- NEW: clean & persist cleaned entities ----
            if raw_entities_path:
                try:
                    with open(raw_entities_path, 'r', encoding='utf-8') as f: # Added encoding
                        raw_entities_list = json.load(f)
                    
                    # Ensure hosts and guests are lists of strings (names)
                    host_names_for_cleaning = [h.get("name") for h in meta.get("hosts", []) if isinstance(h, dict) and h.get("name")]
                    guest_names_for_cleaning = [g.get("name") for g in meta.get("guests", []) if isinstance(g, dict) and g.get("name")]

                    cleaned_entities_list = clean_entities(
                        raw_entities_list,
                        host_names_for_cleaning,
                        guest_names_for_cleaning
                    )
                    
                    # Construct cleaned entities path
                    # entities_dir should be defined or accessible if _generate_spacy_entities_file uses it.
                    # Assuming base_data_dir / "entities" is the dir.
                    entities_dir = base_data_dir / "entities" # Re-affirm or get from raw_entities_path
                    if isinstance(raw_entities_path, str): # Ensure raw_entities_path is a string for Path manipulation
                        clean_entities_file_name = Path(raw_entities_path).stem + "_clean.json"
                        cleaned_entities_path_obj = entities_dir / clean_entities_file_name
                        meta["cleaned_entities_path"] = str(cleaned_entities_path_obj.resolve())

                        with open(cleaned_entities_path_obj, "w", encoding='utf-8') as fout: # Added encoding
                            json.dump(cleaned_entities_list, fout, ensure_ascii=False, indent=2) # Added indent
                        logger.info(f"Saved cleaned NER entities to {meta['cleaned_entities_path']}")
                        
                        # Calculate entity stats
                        stats = defaultdict(int)
                        for e in cleaned_entities_list:
                            stats[e.get("label", "UNKNOWN_LABEL")] += 1
                        meta["entity_stats"] = dict(stats) # Convert back to dict for JSON serializability
                        
                    else:
                        logger.error(f"Could not determine cleaned entities path because raw_entities_path was not a string: {raw_entities_path}")

                except FileNotFoundError:
                    logger.error(f"Raw entities file not found at {raw_entities_path} for cleaning.")
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON from raw entities file: {raw_entities_path}")
                except Exception as e:
                    logger.error(f"Error processing or saving cleaned entities for {meta.get('guid')}: {e}")
            else:
                logger.warning(f"Skipping entity cleaning for {meta.get('guid')} as raw_entities_path was not generated.")
        else:
            logger.warning(f"Skipping SpaCy entity generation and cleaning for {meta.get('guid')} as nlp_model was not provided to enrich_meta.")

        # 2. Sentence Embeddings
        if st_model:
            meta["embedding_path"] = _generate_sentence_embedding_file(
                transcript_text, meta['guid'], base_data_dir, st_model
            )
        else:
            logger.warning(f"Skipping sentence embedding generation for {meta.get('guid')} as st_model was not provided to enrich_meta.")
            
    elif perform_caching:
        # This condition implies guid might be missing or is a placeholder
        logger.warning(f"Skipping entity/embedding caching for {meta.get('guid', entry.get('title', 'unknown_episode'))} due to missing/placeholder GUID, or models not provided to enrich_meta.")
        
    return meta

def tidy_people(meta_input: dict, initial_people_list: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Corrects host/guest lists using KNOWN_HOSTS, ALIAS_MAP, and a list of spurious names.
    meta_input is expected to have a 'podcast' field for the show name.
    initial_people_list is the list of person dicts from NER/tags.
    """
    current_podcast_name = meta_input.get("podcast", "")
    
    # Apply aliases first
    canonicalized_people_list = []
    for p_dict_orig in initial_people_list:
        p_dict = p_dict_orig.copy() # Work on a copy
        if not isinstance(p_dict, dict) or not isinstance(p_dict.get("name"), str):
            logger.warning(f"Skipping invalid person entry in tidy_people (pre-alias): {p_dict_orig}")
            continue
        
        original_name_lc = p_dict["name"].lower()
        canonical_name = ALIAS_MAP.get(original_name_lc, p_dict["name"]) # Get canonical name or use original
        p_dict["name"] = canonical_name # Update name to canonical form
        canonicalized_people_list.append(p_dict)

    known_hosts_for_show_lc = {name.lower() for name in KNOWN_HOSTS.get(current_podcast_name, set())}
    known_host_original_casing_map = {}
    for name_in_known_hosts_set in KNOWN_HOSTS.get(current_podcast_name, set()):
        lc_name = name_in_known_hosts_set.lower()
        if lc_name not in known_host_original_casing_map: 
            known_host_original_casing_map[lc_name] = name_in_known_hosts_set

    corrected_hosts = []
    corrected_guests = []
    host_names_assigned_lc = set()

    # Process the canonicalized list
    for p_dict in canonicalized_people_list:
        if not isinstance(p_dict, dict) or not isinstance(p_dict.get("name"), str):
            logger.warning(f"Skipping invalid person entry in tidy_people (post-alias): {p_dict} for podcast {current_podcast_name}")
            continue

        person_name_original = p_dict["name"] # This is now the canonical name if an alias was found
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
        
        if lc_name not in final_hosts_dict:
            final_hosts_dict[lc_name] = h_item
        else:
            # Duplicate host found, merge attributes
            final_hosts_dict[lc_name] = merge_people(final_hosts_dict[lc_name], h_item)
            
    final_hosts = list(final_hosts_dict.values())

    # Deduplicate and merge guests as well
    final_guests_dict = {}
    for g_item in final_guests:
        if not isinstance(g_item, dict) or not isinstance(g_item.get("name"), str):
            logger.warning(f"Skipping invalid guest item during final deduplication: {g_item}")
            continue
        lc_name = g_item["name"].lower()
        # Guests are generally not title-cased unless specifically formatted that way
        # (e.g. from <podcast:person> with specific casing)

        if lc_name not in final_guests_dict:
            final_guests_dict[lc_name] = g_item
        else:
            final_guests_dict[lc_name] = merge_people(final_guests_dict[lc_name], g_item)
            
    final_guests = list(final_guests_dict.values())

    return final_hosts, final_guests

def merge_people(existing_person_dict: dict, new_person_dict: dict) -> dict:
    """Return a single dict merging two duplicate people records."""
    merged = existing_person_dict.copy()
    for key in ("href", "org", "title"):
        if not merged.get(key) and new_person_dict.get(key): # If existing doesn't have it, but new one does
            merged[key] = new_person_dict.get(key)
        elif merged.get(key) and new_person_dict.get(key) and merged.get(key) != new_person_dict.get(key):
            # If both have different non-empty values, prefer the longer one (simplistic merge)
            # Or one could concatenate, or log a conflict. For now, prefer longer.
            if len(str(new_person_dict.get(key))) > len(str(merged.get(key))):
                merged[key] = new_person_dict.get(key)
    # Ensure 'role' is preserved, preferring 'host' if one is 'host' and other is 'guest'
    if existing_person_dict.get("role") == "host" or new_person_dict.get("role") == "host":
        merged["role"] = "host"
    elif not merged.get("role") and new_person_dict.get("role"): # existing has no role, new one does
        merged["role"] = new_person_dict.get("role")
    return merged 

# ---------------------------------------------------------------------------
# Entity post-processor
# ---------------------------------------------------------------------------

def clean_entities(
    raw_entities: List[Dict],
    host_names: List[str],
    guest_names: List[str],
) -> List[Dict]:
    """
    Deduplicate and filter spaCy entities.
    Returns a compact list suitable for KPI extraction & graphs.
    Uses globally defined SPURIOUS_PERSON_NAMES_LC and SPURIOUS_ORG_NAMES_LC.
    """
    # Normalise host/guest names to lowercase for quick membership checks
    people_skip_lc = {name.lower() for name in host_names + guest_names}

    seen = set()               # (text_lc, label)  tuples
    cleaned = []

    for ent in raw_entities:
        text = ent.get("text", "").strip()
        label = ent.get("label", "") # spaCy typically uses 'label_', but our stored format is 'label' or 'type'
        
        # Adjust to use 'type' if that's what _generate_spacy_entities_file produces
        # Based on _generate_spacy_entities_file, the key is "type" for label
        if not label and ent.get("type"):
            label = ent.get("type")


        text_lc = text.lower()

        # --------- filtering ----------
        if not text or not label: # Skip if essential fields are missing
            logger.debug(f"Skipping entity due to missing text or label: {ent}")
            continue
        if len(text) < 2:                                # too short
            logger.debug(f"Skipping entity (too short): '{text}' ({label})")
            continue
        if label == "PERSON" and text_lc in SPURIOUS_PERSON_NAMES_LC:
            logger.debug(f"Skipping spurious PERSON entity: '{text}'")
            continue
        if label == "ORG" and text_lc in SPURIOUS_ORG_NAMES_LC:
            logger.debug(f"Skipping spurious ORG entity: '{text}'")
            continue
        if label == "PERSON" and text_lc in people_skip_lc:
            logger.debug(f"Skipping PERSON entity (host/guest): '{text}'")
            continue

        # --------- canonicalisation ----------
        # Standardize casing for certain entity types
        if label in {"PERSON", "ORG", "PRODUCT", "EVENT", "LOC", "GPE"}: # Added LOC, GPE
            text = text.title()          # "jason lemkin" → "Jason Lemkin"
        # leave MONEY, DATE, etc. as-is, unless specific rules are needed

        # Deduplication based on canonicalized text and label
        # Use the canonicalized text for the signature
        sig = (text.lower(), label) # Use .lower() of the now title-cased text for broader matching
        if sig in seen:
            logger.debug(f"Skipping duplicate entity (post-canonicalization): '{text}' ({label})")
            continue

        cleaned.append({
            "text": text,  # Store the canonicalized text
            "label": label,
            "start_char": ent.get("start_char"), # Corrected from 'start'
            "end_char": ent.get("end_char"),   # Corrected from 'end'
        })
        seen.add(sig)
    
    logger.info(f"Cleaned {len(raw_entities)} raw entities down to {len(cleaned)} unique entities.")
    return cleaned 

def generate_podcast_slug(podcast_title: str) -> str:
    """Generates a simple slug from a podcast title."""
    if not podcast_title:
        return "unknown-podcast"
    # Normalize to lowercase, replace non-alphanumeric with hyphens
    s = podcast_title.lower()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    s = s.strip('-') # Remove leading/trailing hyphens
    return s if s else "podcast"

def make_slug(podcast_name_or_slug: str, episode_title: str, published_iso_date: str) -> str:
    """Generates a unique slug for an episode.
    Args:
        podcast_name_or_slug: The podcast title or a pre-generated podcast slug.
        episode_title: The title of the episode.
        published_iso_date: The publication date in ISO format (YYYY-MM-DD or full datetime string).
    """
    # Generate podcast_slug if full title is given
    if ' ' in podcast_name_or_slug: # Heuristic: if it has spaces, it's likely a title not a slug
        podcast_s = generate_podcast_slug(podcast_name_or_slug)
    else:
        podcast_s = podcast_name_or_slug # Assume it's already a slug

    # Clean episode title part
    ep_title_normalized = unicodedata.normalize('NFKD', episode_title).encode('ascii', 'ignore').decode('utf-8')
    ep_title_slug_part = re.sub(r'[^a-z0-9]+', '-', ep_title_normalized.lower()).strip('-')
    ep_title_slug_part = ep_title_slug_part[:60].strip('-') # Max length for episode part

    # Format date
    try:
        # Handle full datetime strings or just YYYY-MM-DD
        if 'T' in published_iso_date:
            date_obj = dt.datetime.fromisoformat(published_iso_date.replace('Z', '+00:00'))
        else:
            date_obj = dt.date.fromisoformat(published_iso_date)
        date_slug_part = date_obj.strftime("%Y-%m-%d")
    except ValueError:
        # Fallback if date parsing fails, though published_iso_date should be reliable
        logger.warning(f"Could not parse date '{published_iso_date}' for slug generation. Using raw value.")
        date_slug_part = published_iso_date # Or a fixed string like 'unknown-date'

    return f"{podcast_s}-{date_slug_part}-{ep_title_slug_part}".strip('-')

# --- Configuration loading ---
# ... existing code ...