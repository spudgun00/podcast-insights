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
from dateutil import parser as dtparse # Correct import for dateutil
import hashlib
import uuid
import yaml
from unidecode import unidecode
from podcast_insights.const import SCHEMA_VERSION, BUCKET # ADDED BUCKET IMPORT
import glob, os

logger = logging.getLogger(__name__)

# --- Helper function for SpaCy Entity Caching ---
def _generate_spacy_entities_file(
    transcript_text: str, 
    guid: str, 
    base_data_dir: Path, 
    nlp_model: spacy.Language,
    podcast_slug: Optional[str] = None
) -> Optional[str]:
    """Generates and saves SpaCy entities to a JSON file, returns the file path or None."""
    if not transcript_text or not guid or not nlp_model:
        logger.warning("Skipping SpaCy entity generation due to missing text, GUID, or model.")
        return None
    try:
        doc = nlp_model(transcript_text)
        entities = [{'text':e.text, 'type':e.label_, 'start_char':e.start_char, 'end_char':e.end_char} for e in doc.ents]
        
        entities_base_dir = base_data_dir / "entities_raw"
        target_dir = entities_base_dir
        if podcast_slug:
            target_dir = entities_base_dir / podcast_slug
        
        target_dir.mkdir(parents=True, exist_ok=True)
        entities_path = target_dir / f"{guid}.json"
        with open(entities_path, "w", encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved NER entities to {entities_path}")
        return str(entities_path.resolve())
    except Exception as e:
        logger.error(f"Failed to generate/save SpaCy entities for GUID {guid}: {e}")
        return None

# --- Helper function for Sentence Embedding Caching ---
def _generate_sentence_embedding_file(
    segment_texts: List[str],
    guid: str, 
    base_data_dir: Path, 
    st_model: SentenceTransformer,
    podcast_slug: Optional[str] = None
) -> Optional[str]:
    """Generates and saves sentence/segment embeddings to a .npy file, returns the file path or None."""
    if not segment_texts or not guid or not st_model:
        logger.warning("Skipping sentence embedding generation due to missing segment texts, GUID, or model.")
        return None
    try:
        # Encode list of segment texts
        embeddings_array = st_model.encode(segment_texts, convert_to_numpy=True)
        # Ensure the array is a numpy array first, then convert to float16 for saving
        if not isinstance(embeddings_array, np.ndarray):
            embeddings_array = np.array(embeddings_array) # Default dtype, then cast
        
        embeddings_to_save = embeddings_array.astype(np.float16)  # Shrink to float16
        
        embeddings_base_dir = base_data_dir / "embeddings"
        target_dir = embeddings_base_dir
        if podcast_slug:
            target_dir = embeddings_base_dir / podcast_slug
            
        target_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = target_dir / f"{guid}.npy"
        np.save(embedding_path, embeddings_to_save)
        logger.info(f"Saved sentence embeddings to {embedding_path} with dtype={embeddings_to_save.dtype}, shape={embeddings_to_save.shape}")
        return str(embedding_path.resolve())
    except Exception as e: # This outer except catches errors from st_model.encode or path operations or np.save
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
_raw_known_hosts_data = load_json_config(KNOWN_HOSTS_FILE, default_data={}) # Load raw data

KNOWN_HOSTS: Dict[str, set[str]] = {} # Ensure type hint for clarity
for show_title, host_names_list in _raw_known_hosts_data.items():
    if isinstance(host_names_list, list) and all(isinstance(name, str) for name in host_names_list):
        KNOWN_HOSTS[show_title] = {name.lower() for name in host_names_list}
    else:
        # If the default_data from load_json_config was triggered and it contains pre-formatted sets, handle that too.
        # Example: default_data={"Show A": {"host1", "host2"}}
        if isinstance(host_names_list, set) and all(isinstance(name, str) for name in host_names_list):
             KNOWN_HOSTS[show_title] = {name.lower() for name in host_names_list} # Ensure lowercase even if already a set
        else:
            logger.warning(f"Invalid host list format for '{show_title}' in {KNOWN_HOSTS_FILE} or its default. Expected list/set of strings. Skipping this entry.")

# Fallback for critical shows if JSON is missing or malformed for them.
# This ensures core functionality for these specific podcasts even with bad JSON.
# The primary source should be the JSON file.
_CRITICAL_SHOWS_FALLBACK = {
    "The Twenty Minute VC (20VC): Venture Capital | Startup Funding | The Pitch": {"harry stebbings"},
    # "a16z Podcast": {"ben horowitz", "marc andreessen"}  # COMMENTED OUT: Use YAML config instead
}
for critical_show, default_hosts_set in _CRITICAL_SHOWS_FALLBACK.items():
    if critical_show not in KNOWN_HOSTS:
        logger.warning(f"Adding hardcoded fallback for '{critical_show}' hosts as it was not found or invalid in {KNOWN_HOSTS_FILE}.")
        KNOWN_HOSTS[critical_show] = default_hosts_set

# --- Load alias map from config/people_aliases.yml ---
# Single source: config/people_aliases.yml – both tidy_people and role assignment read it.
PEOPLE_ALIASES_FILE = Path("config/people_aliases.yml") # Ensure this path is correct relative to runtime CWD

def load_yaml_config(file_path: Path, default_data=None): # Type hint Path
    if default_data is None:
        default_data = {}
    try:
        # Ensure file_path is treated as Path object for .open()
        with file_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg is None: # File was empty or contained only comments/null
            logger.warning(f"ALIAS_MAP_DEBUG: Alias file {file_path} is empty or invalid YAML (parsed as None). Loaded 0 aliases. Using default: {default_data}")
            return default_data # Return default if file is empty or just 'null'
        logger.info(f"ALIAS_MAP_DEBUG: Loaded {len(cfg)} aliases from {file_path}")
        return cfg
    except FileNotFoundError:
        logger.error(f"ALIAS_MAP_DEBUG: CRITICAL - Alias file {file_path} not found. People aliasing WILL NOT function. Please create this file or check the path. Using empty alias map as default.")
        return default_data # Return default so program MIGHT continue, but with clear error
    except yaml.YAMLError as ye: # Catch YAML-specific errors
        logger.error(f"ALIAS_MAP_DEBUG: CRITICAL - YAML parse error in alias file {file_path}: {ye}. People aliasing will LIKELY FAIL or be incorrect. Re-raising.")
        raise # Re-raise to make the problem visible and stop execution if YAML is broken
    except Exception as e: # Catch other potential errors during file operations
        logger.error(f"ALIAS_MAP_DEBUG: CRITICAL - Unexpected error loading alias config from {file_path}: {e}. People aliasing will LIKELY FAIL. Re-raising.")
        raise # Re-raise to make the problem visible

ALIAS_MAP = load_yaml_config(PEOPLE_ALIASES_FILE, default_data={})
print(f"META_UTILS_PRINT_DEBUG: ALIAS_MAP directly after definition in meta_utils.py: {{len(ALIAS_MAP) if ALIAS_MAP else 'None or Empty'}} keys: {list(ALIAS_MAP.keys())[:5]}")
logger.info(f"META_UTILS_LOGGER_DEBUG: ALIAS_MAP directly after definition in meta_utils.py: {{len(ALIAS_MAP) if ALIAS_MAP else 'None or Empty'}} keys: {list(ALIAS_MAP.keys())[:5]}")

# --- Load stopwords from config/stopwords.txt ---
# Extractor reads config/stopwords.txt at runtime; no code deploy needed for updates.
STOPWORDS_FILE = Path("config/stopwords.txt")
def load_stopwords(file_path):
    try:
        with open(file_path, 'r') as f:
            stopwords = set(line.strip() for line in f if line.strip())
            logger.info(f"Loaded {len(stopwords)} stopwords from {file_path}")
            return stopwords
    except FileNotFoundError:
        logger.warning(f"Stopwords file {file_path} not found. Using EXTENDED_STOP_WORDS only.")
        return set()
    except Exception as e:
        logger.error(f"Error loading stopwords from {file_path}: {e}")
        return set()

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
    },
    {
        "like", "it's", "ve", "really", "actual", "actually", "lot", "that's", "one", "two", "three",
        "okay", "im", "don't", "gon", "na", "yeah", "um", "uh", "hm", "er", "ll", "re", "yep", "yes", "no", "oh", "got",
        "did", "does", "do", "was", "is", "are", "were", "am", "been", "being", "have", "has", "had", 
        "make", "get", "go", "going", "let", "let's", "say", "says", "said", "see", "look", "thing", "things", "think", "know", "just",
        "percent", "million", "billion", "thousand", "hundred", "dollars", "usd", "eur", "gbp",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100",
        "podcast", "episode", "today", "week", "month", "year", "welcome", "everyone", "guys", "folks"
    }
)

EXTENDED_STOP_WORDS = EXTENDED_STOP_WORDS.union(load_stopwords(STOPWORDS_FILE))

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
    
    # Load stopwords at runtime
    current_stopwords = set(ENGLISH_STOP_WORDS) # Start with base
    current_stopwords.update({
        "yeah", "yep", "yup", "uh", "um", "uh-huh", "mm-hmm", "ok", "okay", "right",
        "like", "just", "literally", "basically", "actually", "kinda", "sorta",
        "you", "youre", "we", "they", "ive", "hes", "shes", "dont", "doesnt", "didnt",
        "gonna", "wanna", "gotta", "think", "pretty", "really", "stuff", "things",
        "know", "did", "let", "10",
        "jason", "money", "30",
        "point", "question", "million", "billion",
        "doing",
        "interesting",
        "didn",
        "mm",
        "does"
    })
    current_stopwords.update({
        "company", "companies", "business", "industry", "product", "products",
        "market", "markets", "people", "team", "teams", "customer", "customers",
        "thing", "lot", "lots", "great", "good", "big", "small", "way", "today",
        "year", "years", "world", "going", "come", "comes", "make", "makes",
        "deal", "capital"
    })
    current_stopwords.update({
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
        "deals"
    })
    current_stopwords.update({
        "like", "it's", "ve", "really", "actual", "actually", "lot", "that's", "one", "two", "three",
        "okay", "im", "don't", "gon", "na", "yeah", "um", "uh", "hm", "er", "ll", "re", "yep", "yes", "no", "oh", "got",
        "did", "does", "do", "was", "is", "are", "were", "am", "been", "being", "have", "has", "had", 
        "make", "get", "go", "going", "let", "let's", "say", "says", "said", "see", "look", "thing", "things", "think", "know", "just",
        "percent", "million", "billion", "thousand", "hundred", "dollars", "usd", "eur", "gbp",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100",
        "podcast", "episode", "today", "week", "month", "year", "welcome", "everyone", "guys", "folks"
    })
    current_stopwords.update(load_stopwords(STOPWORDS_FILE))
    
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
            stop_words=list(current_stopwords),  # Use the runtime loaded stopwords
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

def get_explicit_flag(entry_explicit_val, feed_level_explicit_val):
    """
    Determines the explicit flag status.
    Prioritizes entry-level, then feed-level.
    Strictly checks for "yes" (case-insensitive) or boolean True for True.
    Handles None values gracefully.
    """
    # Try entry level first
    if entry_explicit_val is not None: # Check the passed-in value
        if isinstance(entry_explicit_val, str) and entry_explicit_val.lower() == "yes":
            return True
        if isinstance(entry_explicit_val, bool) and entry_explicit_val is True: # Handle boolean True
            return True
        # Any other string (like "no", "clean") or boolean False means not explicit
        return False

    # Try feed level if entry level was None or not definitively True
    if feed_level_explicit_val is not None: # Check the passed-in feed level value
        if isinstance(feed_level_explicit_val, str) and feed_level_explicit_val.lower() == "yes":
            return True
        if isinstance(feed_level_explicit_val, bool) and feed_level_explicit_val is True: # Handle boolean True
            return True
    return False
        
    return None # Default if neither is available or definitively "yes"

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
    # COMMENTED OUT: a16z hardcoded hack - use YAML config instead
    # if not initial_people_candidates and show_name == "a16z Podcast":
    #     # These are known hosts, directly add them as candidates with role host
    #     a16z_hosts = [
    #         {"name": "Ben Horowitz", "role": "host", "org": "a16z", "title": "Co-founder and General Partner"},
    #         {"name": "Marc Andreessen", "role": "host", "org": "a16z", "title": "Co-founder and General Partner"}
    #     ]
    #     initial_people_candidates.extend(a16z_hosts)
    #     # Cache these hosts
    #     show_hosts_cache.add_hosts(show_name, [h["name"] for h in a16z_hosts if isinstance(h.get("name"), str)])

    # 2. Fallback to NER on transcript if still no people or to augment
    # We run NER even if tags exist to catch other mentions that might be guests.
    try:
        nlp = spacy.load("en_core_web_sm")
        intro_text = transcript[:1500]
        doc = nlp(intro_text)
        
        ner_people_names_seen = {p["name"].lower() for p in initial_people_candidates if isinstance(p.get("name"), str)} # Avoid re-adding people from tags

        for ent in doc.ents:
            if ent.label_ == "PERSON" and isinstance(ent.text, str):
                name = ent.text # <--- This is the raw text from SpaCy NER
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

def enrich_meta(entry, feed_details, tech, transcript_text=None, transcript_segments=None, perform_caching=True, nlp_model=None, st_model=None, base_data_dir="data", podcast_slug=None, word_timestamps_enabled=False):
    # Initialize meta dictionary
    meta = {"raw_entry_original_feed": entry if isinstance(entry, dict) else entry.__dict__ if hasattr(entry, '__dict__') else str(entry)}
    meta["schema_version"] = SCHEMA_VERSION

    # --- Basic Feed and Episode Info ---
    meta["podcast_title"] = feed_details.get("title", "N/A")
    meta["podcast_feed_url"] = feed_details.get("url", "N/A")
    meta["podcast_generator"] = feed_details.get("generator")
    meta["podcast_language"] = feed_details.get("language")
    meta["podcast_itunes_author"] = feed_details.get("itunes_author")
    meta["podcast_copyright"] = feed_details.get("copyright")

    is_dict_entry = isinstance(entry, dict)

    # Use .get() for dictionaries, attribute access for feedparser objects
    # OLD GUID LOGIC:
    # meta["guid"] = entry.get("id", "missing_guid_") if is_dict_entry else getattr(entry, "id", f"missing_guid_{str(uuid.uuid4())}")
    # if not meta["guid"] and is_dict_entry: # Fallback for mock_entry if 'id' wasn't guid
    #     meta["guid"] = entry.get("guid")
    # NEW GUID LOGIC from o3 feedback:
    if is_dict_entry:
        extracted_guid = entry.get("guid") or entry.get("id")
    else: # feedparser object
        extracted_guid = getattr(entry, "guid", None) or getattr(entry, "id", None)

    if not extracted_guid:
        logger.warning(f"Could not find 'guid' or 'id' in entry. Generating a new UUID. Entry keys: {list(entry.keys()) if is_dict_entry else dir(entry)}")
        extracted_guid = f"missing_guid_{str(uuid.uuid4())}"
    meta["guid"] = extracted_guid
    # END NEW GUID LOGIC

    meta["episode_title"] = entry.get("title") if is_dict_entry else getattr(entry, "title", "N/A")
    meta["episode_summary_original"] = entry.get("summary") if is_dict_entry else getattr(entry, "summary", "N/A")
    meta["episode_copyright"] = entry.get("rights") if is_dict_entry else getattr(entry, "rights", None)
    
    published_original_format = entry.get("published") if is_dict_entry else getattr(entry, "published", None)
    published_parsed_dt = entry.get("published_parsed") if is_dict_entry else getattr(entry, "published_parsed", None)

    if published_parsed_dt:
        try:
            # Ensure it's a datetime object if it came from parsed feedparser field
            if not isinstance(published_parsed_dt, dt.datetime ):
                 # It's likely a time.struct_time from feedparser, convert to datetime
                published_parsed_dt = dt.datetime(*published_parsed_dt[:6])
            meta["published"] = published_parsed_dt.isoformat()
        except Exception:
            logger.warning(f"Could not convert published_parsed ({published_parsed_dt}) to datetime object for {meta.get('guid')}. Falling back to 'published' string.")
            published_parsed_dt = None # Reset to ensure fallback
            meta["published"] = None # Initialize to None

    if not meta.get("published") and published_original_format:
        try:
            # If 'published_parsed' failed or wasn't there, parse the original string
            dt_obj = dtparse.parse(published_original_format)
            meta["published"] = dt_obj.isoformat()
        except (dtparse.ParserError, OverflowError, TypeError) as e:
            logger.warning(f"Could not parse 'published' date string '{published_original_format}' for GUID {meta.get('guid')}: {e}")
            meta["published"] = None # Set to None if parsing fails
    
    meta["published_original_format"] = published_original_format

    meta["categories"] = [{"term": t.get("term"), "scheme": t.get("scheme"), "label": t.get("label")} for t in entry.get("tags", [])] if is_dict_entry else [{"term": t.term, "scheme": t.scheme, "label": t.label} for t in getattr(entry, "tags", []) if t]
    meta["episode_url"] = entry.get("link") if is_dict_entry else getattr(entry, "link", None)
    
    enclosures = entry.get("enclosures") if is_dict_entry else getattr(entry, "enclosures", [])
    if enclosures:
        meta["audio_url_original_feed"] = enclosures[0].get("href") if isinstance(enclosures[0], dict) else getattr(enclosures[0], "href", None)
        meta["audio_type_original_feed"] = enclosures[0].get("type") if isinstance(enclosures[0], dict) else getattr(enclosures[0], "type", None)
        meta["audio_length_bytes_original_feed"] = enclosures[0].get("length") if isinstance(enclosures[0], dict) else getattr(enclosures[0], "length", None)
        try:
            meta["audio_length_bytes_original_feed"] = int(meta["audio_length_bytes_original_feed"]) if meta["audio_length_bytes_original_feed"] else None
        except (ValueError, TypeError):
            logger.warning(f"Could not convert audio_length_bytes_original_feed '{meta['audio_length_bytes_original_feed']}' to int for {meta.get('guid')}")
            meta["audio_length_bytes_original_feed"] = None
    else:
        meta["audio_url_original_feed"] = None
        meta["audio_type_original_feed"] = None
        meta["audio_length_bytes_original_feed"] = None
        if is_dict_entry and entry.get("audio_url"): # from mock_entry
             meta["audio_url_original_feed"] = entry.get("audio_url")


    # iTunes specific
    itunes_explicit_val = entry.get("itunes_explicit") if is_dict_entry else getattr(entry, "itunes_explicit", None)
    meta["itunes_explicit"] = get_explicit_flag(itunes_explicit_val, feed_details.get("itunes_explicit"))
    meta["itunes_episode_type"] = entry.get("itunes_episodetype") if is_dict_entry else getattr(entry, "itunes_episodetype", None)
    meta["itunes_subtitle"] = entry.get("itunes_subtitle") if is_dict_entry else getattr(entry, "itunes_subtitle", None)
    meta["itunes_summary"] = entry.get("itunes_summary") if is_dict_entry else getattr(entry, "itunes_summary", None)

    # -- People and Guests --
    # Call extract_people which itself calls tidy_people internally.
    # extract_people needs the original entry object, the transcript text, and the podcast title.
    current_podcast_title = meta.get("podcast_title", "N/A") # Get podcast title from meta
    
    # Ensure transcript_text is a string. If it's None, pass an empty string.
    text_for_people_extraction = transcript_text if transcript_text is not None else ""
    
    # Initial extraction
    # Ensure extract_people returns two lists: hosts, guests
    raw_hosts, raw_guests = extract_people(entry, text_for_people_extraction, current_podcast_title)

    # Refine roles
    # Ensure KNOWN_HOSTS and SPURIOUS_GUESTS_LC are accessible here (defined globally in the module)
    refined_hosts, refined_guests = refine_people_roles(
        raw_hosts, 
        raw_guests, 
        current_podcast_title, 
        KNOWN_HOSTS, 
        SPURIOUS_GUESTS_LC
    )
    meta["hosts"] = refined_hosts
    meta["guests"] = refined_guests
    
    # -- Technical Metadata --
    # Extract from tech dictionary
    tech_metadata = tech.get('metadata', {})
    meta.update(tech_metadata)

    # --- Canonical Segments Path Construction and other Path Initializations ---
    guid_for_caching = meta.get('guid')
    # podcast_slug is an argument to enrich_meta and is available in this scope.

    # Construct the canonical segments_path if possible.
    # This will be the primary value for meta['segments_path'] if guid and slug are known.
    # If not, meta['segments_path'] will retain the value from tech_metadata (if any), or be None.
    if guid_for_caching and podcast_slug:
        segments_base_dir = Path(base_data_dir) / "segments"
        segments_target_dir = segments_base_dir / podcast_slug
        expected_segments_filename = f"{guid_for_caching}.json"
        constructed_segments_path = str((segments_target_dir / expected_segments_filename).resolve())
        
        # Log how the segments_path was determined for clarity
        tech_provided_segments_path = tech.get('segments_path') or tech_metadata.get('segments_path')
        if tech_provided_segments_path and tech_provided_segments_path != constructed_segments_path:
            logger.info(f"tech provided segments_path '{tech_provided_segments_path}' which differs from constructed canonical path '{constructed_segments_path}'. Prioritizing constructed path for meta output.")
        elif not tech_provided_segments_path:
            logger.info(f"No segments_path provided by tech. Using constructed canonical path: {constructed_segments_path}")
        # If tech_provided_segments_path == constructed_segments_path, it will just use it.
            
        meta["segments_path"] = constructed_segments_path
        logger.info(f"Final meta['segments_path'] set to: {meta['segments_path']}")

    elif not meta.get('segments_path'): # if not constructible (no guid/slug) AND not already set by tech_metadata
        logger.warning(
            f"Cannot construct canonical segments_path (GUID='{guid_for_caching}', podcast_slug='{podcast_slug}' are not both valid) "
            f"and no segments_path was provided in tech_metadata. meta['segments_path'] will be None."
        )
        meta['segments_path'] = None # Explicitly set to None
    else: # Was set by tech_metadata, and not constructible, so use the tech-provided one
        logger.info(f"Using segments_path from tech_metadata as canonical construction was not possible: {meta.get('segments_path')}")

    # Initialize other paths from tech. These might overwrite if tech_metadata also had them (which is fine).
    meta["entities_path"] = tech.get("entities_path")
    meta["embedding_path"] = tech.get("embedding_path")
    meta["cleaned_entities_path"] = None # Initialize for cleaned entities
    meta["entity_stats"] = {} # Initialize for entity stats

    # --- Cache NER Entities and Sentence Embeddings --- 
    # guid_for_caching is defined above. podcast_slug is from function arguments.
    can_cache = guid_for_caching and podcast_slug # ADDED THIS LINE

    if can_cache:
        # 1. SpaCy Entities Caching Logic
        if nlp_model and perform_caching: # We have means and permission to generate
            if not meta.get("entities_path"): # Check if path wasn't already in tech
                logger.info(f"Generating raw entities for {guid_for_caching} as nlp_model is present and no entities_path found in meta (podcast_slug: {podcast_slug}).")
                generated_raw_entities_path = _generate_spacy_entities_file(
                    transcript_text, guid_for_caching, Path(base_data_dir), nlp_model,
                    podcast_slug=podcast_slug
                )
                if generated_raw_entities_path:
                    meta["entities_path"] = generated_raw_entities_path
            else: # entities_path already exists in meta
                logger.info(f"Using existing entities_path from meta for {guid_for_caching} (nlp_model was present but path already existed): {meta.get('entities_path')}")
        elif meta.get("entities_path"): # No nlp_model or caching disabled, but path exists
             logger.info(f"Using existing entities_path from meta for {guid_for_caching} (nlp_model not provided or caching disabled): {meta.get('entities_path')}")
        else: # No nlp_model/caching disabled AND no pre-existing path
            logger.warning(f"Raw entities cannot be generated or loaded for {guid_for_caching}: nlp_model not provided (or caching disabled) AND no entities_path was found in meta/tech.")

        # 2. Sentence Embeddings Caching Logic
        if st_model and perform_caching and transcript_segments:
            if not meta.get("embedding_path"): # Check if path wasn't already in tech
                logger.info(f"Generating sentence embeddings for {guid_for_caching} as st_model is present, segments are available, and no embedding_path found in meta (podcast_slug: {podcast_slug}).")
                segment_texts = [segment.get('text', '') for segment in transcript_segments]
                generated_embedding_path = _generate_sentence_embedding_file(
                    segment_texts, guid_for_caching, Path(base_data_dir), st_model,
                    podcast_slug=podcast_slug
                )
                if generated_embedding_path:
                    meta["embedding_path"] = generated_embedding_path # MODIFIED KEY back to embedding_path
            else: # embedding_path already exists in meta
                logger.info(f"Using existing embedding_path from meta for {guid_for_caching} (st_model was present but path already existed): {meta.get('embedding_path')}")
        elif meta.get("embedding_path"): # No st_model/caching disabled/no segments, but path exists
            logger.info(f"Using existing embedding_path from meta for {guid_for_caching} (st_model not provided, caching disabled, or no segments): {meta.get('embedding_path')}") # This log might also need update if key changes everywhere
        else: # No st_model/caching disabled/no segments AND no pre-existing path
            logger.warning(f"Sentence embeddings cannot be generated or loaded for {guid_for_caching}: st_model not provided, caching disabled, no segments, AND no embedding_path was found in meta/tech.")
    else:
        logger.warning(f"Cannot perform NER/Embedding caching for episode: guid_for_caching={guid_for_caching}, podcast_slug={podcast_slug}. Check if GUID and podcast_slug are available in meta.")

    meta["title"] = entry.get("title") if is_dict_entry else getattr(entry, "title", None)
    meta["link"] = entry.get("link") if is_dict_entry else getattr(entry, "link", None)
    meta["episode_copyright"] = entry.get("rights") if is_dict_entry else getattr(entry, "rights", None)
    meta["summary"] = entry.get("summary") if is_dict_entry else getattr(entry, "summary", None)

    # --- Clean and Save Entities ---
    loaded_raw_entities = []
    if meta.get("entities_path") and Path(meta["entities_path"]).exists():
        try:
            with open(meta["entities_path"], 'r', encoding='utf-8') as f:
                loaded_raw_entities = json.load(f)
            if not isinstance(loaded_raw_entities, list): # Basic validation
                logger.warning(f"Raw entities file {meta['entities_path']} did not contain a list. Skipping entity cleaning.")
                loaded_raw_entities = []
        except Exception as e:
            logger.error(f"Failed to load raw entities from {meta['entities_path']}: {e}. Skipping entity cleaning.")
            loaded_raw_entities = []
    elif not meta.get("entities_path"):
        logger.info(f"No raw entities_path available for {guid_for_caching}, cannot perform entity cleaning.")
    else: # Path exists in meta but file does not
        logger.warning(f"Raw entities file {meta['entities_path']} not found. Skipping entity cleaning.")

    if loaded_raw_entities:
        host_names_for_cleaning = [h.get('name') for h in refined_hosts if isinstance(h, dict) and h.get('name')]
        guest_names_for_cleaning = [g.get('name') for g in refined_guests if isinstance(g, dict) and g.get('name')]
        
        cleaned_entities_list = clean_entities(
            loaded_raw_entities, 
            host_names_for_cleaning, 
            guest_names_for_cleaning
        )
        
        if cleaned_entities_list:
            # Path construction for cleaned_entities_dir is already correct here
            cleaned_entities_dir = Path(base_data_dir) / "entities_cleaned" / podcast_slug 
            cleaned_entities_dir.mkdir(parents=True, exist_ok=True)
            # Use guid_for_caching which is already defined earlier in enrich_meta
            resolved_cleaned_entities_path = str((cleaned_entities_dir / f"{guid_for_caching}_clean.json").resolve())
            # Only keep cleaned_entities_path in meta; derive S3 path at runtime if needed.
            meta['cleaned_entities_path'] = resolved_cleaned_entities_path
            meta['entity_stats'] = { # Basic stats
                "raw_entity_count": len(loaded_raw_entities),
                "cleaned_entity_count": len(cleaned_entities_list)
            }
            try:
                with open(resolved_cleaned_entities_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_entities_list, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(cleaned_entities_list)} cleaned entities to {resolved_cleaned_entities_path}")
                # Patch transcript JSON after writing cleaned entities
                transcript_path = tech.get('transcript_path')
                if transcript_path and Path(transcript_path).exists():
                    try:
                        import portalocker
                        with open(transcript_path, 'r+', encoding='utf-8') as tf:
                            portalocker.lock(tf, portalocker.LOCK_EX)
                            data = json.load(tf)
                            # Ensure 'meta' block exists and get a reference to it
                            if "meta" not in data:
                                data["meta"] = {}
                            transcript_meta_block_ref = data["meta"]
                            
                            logger.info(f"DEBUG_PATCH (portalocker): Attempting to patch transcript_meta_block for {transcript_path} with cleaned_entities_path: {resolved_cleaned_entities_path}, schema_version: {SCHEMA_VERSION}, entity_stats: {meta.get('entity_stats')}, segments_path: {{tech.get('segments_path') or meta.get('segments_path')}}")
                            
                            transcript_meta_block_ref["cleaned_entities_path"] = resolved_cleaned_entities_path
                            transcript_meta_block_ref["schema_version"] = SCHEMA_VERSION
                            if meta.get("entity_stats"):
                                transcript_meta_block_ref["entity_stats"] = meta["entity_stats"]
                            
                            # For segments_path in transcript: prefer tech's value if available (original from transcription),
                            # else use meta's value (which is now the canonical GUID-based path).
                            # The transcript_meta_block_ref already has its segments_path updated by the 'meta.update(tech_metadata)'
                            # if it was in tech['metadata']['segments_path'].
                            # Or, if tech['segments_path'] was directly provided.
                            # We only want to overwrite it with meta['segments_path'] (the canonical one) if it wasn't in tech.
                            if not tech.get("segments_path") and not tech.get('metadata', {}).get('segments_path') and meta.get("segments_path"):
                                transcript_meta_block_ref["segments_path"] = meta["segments_path"]
                            elif tech.get("segments_path"): # Explicit tech.segments_path takes precedence for transcript
                                transcript_meta_block_ref["segments_path"] = tech["segments_path"]
                            # If tech.get('metadata', {}).get('segments_path') was set, it's already in transcript_meta_block_ref.
                            # If meta.get("segments_path") is the only one available, it's also fine.

                            tf.seek(0)
                            json.dump(data, tf, indent=2)
                            tf.truncate()
                            logger.info(f"Patched cleaned_entities_path and schema_version into transcript: {transcript_path}")
                            portalocker.unlock(tf)
                    except ImportError:
                        # Fallback: no lock, just patch
                        logger.warning(f"portalocker not found. Patching transcript {transcript_path} without file lock.")
                        with open(transcript_path, 'r+', encoding='utf-8') as tf:
                            data = json.load(tf)
                            # Ensure 'meta' block exists and get a reference to it
                            if "meta" not in data:
                                data["meta"] = {}
                            transcript_meta_block_ref = data["meta"]

                            logger.info(f"DEBUG_PATCH (no lock): Attempting to patch transcript_meta_block for {transcript_path} with cleaned_entities_path: {resolved_cleaned_entities_path}, schema_version: {SCHEMA_VERSION}, entity_stats: {meta.get('entity_stats')}, segments_path: {{tech.get('segments_path') or meta.get('segments_path')}}")
                            
                            transcript_meta_block_ref["cleaned_entities_path"] = resolved_cleaned_entities_path
                            transcript_meta_block_ref["schema_version"] = SCHEMA_VERSION
                            if meta.get("entity_stats"):
                                transcript_meta_block_ref["entity_stats"] = meta["entity_stats"]
                            
                            # For segments_path in transcript (no lock):
                            if not tech.get("segments_path") and not tech.get('metadata', {}).get('segments_path') and meta.get("segments_path"):
                                transcript_meta_block_ref["segments_path"] = meta["segments_path"]
                            elif tech.get("segments_path"):
                                transcript_meta_block_ref["segments_path"] = tech["segments_path"]
                                
                            tf.seek(0)
                            json.dump(data, tf, indent=2)
                            tf.truncate()
                            logger.info(f"Patched cleaned_entities_path and schema_version into transcript (no lock): {transcript_path}")
                    except Exception as e:
                        logger.error(f"Failed to patch transcript JSON {transcript_path}: {e}")
            except Exception as e:
                logger.error(f"Failed to save cleaned entities to {resolved_cleaned_entities_path}: {e}")
        else:
            logger.info(f"No entities remained after cleaning for {guid_for_caching}.")
            meta['cleaned_entities_path'] = None # Ensure it's None if no file saved
            meta['entity_stats'] = {"raw_entity_count": len(loaded_raw_entities), "cleaned_entity_count": 0}
    else:
        # No raw entities were loaded, so can't clean. Ensure path is None.
        meta['cleaned_entities_path'] = None
        meta['entity_stats'] = {"raw_entity_count": 0, "cleaned_entity_count": 0}

    # After all enrichment, sync segment_count and supports_timestamp
    # Prioritize transcript_segments if passed directly to this enrich_meta call
    if transcript_segments is not None:
        meta["segment_count"] = len(transcript_segments)
        logger.info(f"Set segment_count in meta to {len(transcript_segments)} from provided transcript_segments for GUID {guid_for_caching}.")
        # If we have segments, we can determine timestamp support directly
        meta["supports_timestamp"] = any("start" in segment for segment in transcript_segments)
        logger.info(f"Set supports_timestamp in meta to {meta['supports_timestamp']} from provided transcript_segments for GUID {guid_for_caching}.")
    elif tech.get('transcript_path') and Path(tech['transcript_path']).exists():
        transcript_path_for_sync = tech['transcript_path']
        try:
            with open(transcript_path_for_sync, 'r', encoding='utf-8') as tf_sync:
                data_sync = json.load(tf_sync)
                if "meta" in data_sync:
                    if "segment_count" in data_sync["meta"]:
                        meta["segment_count"] = data_sync["meta"]["segment_count"]
                        logger.info(f"Synced segment_count in meta from {transcript_path_for_sync}: {meta['segment_count']}.")
                    if "supports_timestamp" in data_sync["meta"]:
                        meta["supports_timestamp"] = data_sync["meta"]["supports_timestamp"]
                        logger.info(f"Synced supports_timestamp in meta from {transcript_path_for_sync}: {meta['supports_timestamp']}.")
        except Exception as e:
            logger.error(f"Failed to sync segment_count/supports_timestamp from transcript {transcript_path_for_sync}: {e}")
    else:
        logger.warning(f"Could not determine segment_count or supports_timestamp for {guid_for_caching} as no transcript_segments provided and transcript_path missing or invalid.")
        meta.setdefault("segment_count", 0)
        meta.setdefault("supports_timestamp", False)
        
    # Construct transcript_s3_path using transcript_path from the tech dictionary
    # and other variables already defined in this function's scope (podcast_slug, guid_for_caching, BUCKET).
    local_transcript_path_str = tech.get('transcript_path')

    if local_transcript_path_str and podcast_slug and guid_for_caching and BUCKET:
        transcript_filename = Path(local_transcript_path_str).name
        # The s3_prefix for transcripts should match the one used for other artifacts if layout_fn is the source of truth.
        # layout_fn(guid, podcast_slug) produces "{podcast_slug}/{guid}/"
        s3_prefix = f"{podcast_slug}/{guid_for_caching}/" 
        meta["transcript_s3_path"] = f"s3://{BUCKET}/{s3_prefix}{transcript_filename}"
        logger.info(f"Constructed transcript_s3_path: {meta['transcript_s3_path']}")
    elif not local_transcript_path_str:
        logger.warning(f"Cannot construct transcript_s3_path for GUID {guid_for_caching}: 'transcript_path' not found in tech dictionary. Tech keys: {list(tech.keys())}")
        meta["transcript_s3_path"] = None
    else:
        # This case means local_transcript_path_str was found, but one of the other components was missing.
        missing_components = []
        if not podcast_slug: missing_components.append("podcast_slug")
        if not guid_for_caching: missing_components.append("guid_for_caching")
        if not BUCKET: missing_components.append("BUCKET")
        logger.warning(f"Cannot construct transcript_s3_path for GUID {guid_for_caching}: Missing component(s): {', '.join(missing_components)}. transcript_path was {local_transcript_path_str}")
        meta["transcript_s3_path"] = None

    return meta

def tidy_people(meta_input: dict, initial_people_list: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Corrects host/guest lists using KNOWN_HOSTS, ALIAS_MAP, and a list of spurious names.
    meta_input is expected to have a 'podcast' field for the show name.
    initial_people_list is the list of person dicts from NER/tags.
    """
    global ALIAS_MAP_KEYS_LOGGED # Use the flag
    if not ALIAS_MAP_KEYS_LOGGED:
        logger.info(f"TIDY_PEOPLE_DEBUG: ALIAS_MAP.get('harry'): {ALIAS_MAP.get('harry')}")
        logger.info(f"TIDY_PEOPLE_DEBUG: ALIAS_MAP.get('sbf'): {ALIAS_MAP.get('sbf')}")
        logger.info(f"TIDY_PEOPLE_DEBUG: ALIAS_MAP.get('anjane mita'): {ALIAS_MAP.get('anjane mita')}")
        ALIAS_MAP_KEYS_LOGGED = True

    current_podcast_name = meta_input.get("podcast", "")
    
    # Apply aliases first, and filter spurious names early
    canonicalized_and_filtered_people_list = []
    for p_dict_orig in initial_people_list: # CORRECTED: Use the function parameter initial_people_list
        p_dict = p_dict_orig.copy() # Work on a copy
        if not isinstance(p_dict, dict) or not isinstance(p_dict.get("name"), str):
            logger.warning(f"Skipping invalid person entry in tidy_people (pre-alias/filter): {p_dict_orig}")
            continue
        
        original_name_for_alias_lookup = p_dict["name"]
        original_name_lc_for_alias = unidecode(original_name_for_alias_lookup).lower()
        
        # DEBUG LOGGING START
        if original_name_lc_for_alias in ["anjane mita", "guido", "appenzeller"]:
            logger.info(f"TIDY_PEOPLE_DEBUG: Original name: '{original_name_for_alias_lookup}', lc_for_alias: '{original_name_lc_for_alias}'")
        # DEBUG LOGGING END
            
        canonical_name = ALIAS_MAP.get(original_name_lc_for_alias, original_name_for_alias_lookup) # Get canonical name or use original
        
        # DEBUG LOGGING START
        if original_name_lc_for_alias in ["anjane mita", "guido", "appenzeller"]:
            logger.info(f"TIDY_PEOPLE_DEBUG: Canonical name from ALIAS_MAP: '{canonical_name}' for input '{original_name_lc_for_alias}'")
        # DEBUG LOGGING END

        # If alias result is None (e.g., "a16z": null in YAML), discard this person entry
        if canonical_name is None:
            logger.debug(f"Discarding person entry for '{original_name_for_alias_lookup}' due to None alias result for podcast {current_podcast_name}.")
            continue
            
        p_dict["name"] = canonical_name # Update name to canonical form
        
        # DEBUG LOGGING START
        if original_name_lc_for_alias in ["anjane mita", "guido", "appenzeller"]:
            logger.info(f"TIDY_PEOPLE_DEBUG: p_dict after name update: {p_dict}")
        # DEBUG LOGGING END
            
        # Filter SPURIOUS names (using the now canonicalized name) before appending
        # This check is on the canonical_name's lowercased version.
        canonical_name_lc_for_filter = unidecode(canonical_name).lower()
        if canonical_name_lc_for_filter in SPURIOUS_PERSON_NAMES_LC:
            logger.debug(f"Filtered out spurious person name (after aliasing, before list append): '{canonical_name}' for podcast {current_podcast_name}")
            continue
        
        canonicalized_and_filtered_people_list.append(p_dict)
    
    known_hosts_for_show_lc = {name.lower() for name in KNOWN_HOSTS.get(current_podcast_name, set())}
    known_host_original_casing_map = {}
    for name_in_known_hosts_set in KNOWN_HOSTS.get(current_podcast_name, set()):
        lc_name = name_in_known_hosts_set.lower()
        if lc_name not in known_host_original_casing_map: 
            known_host_original_casing_map[lc_name] = name_in_known_hosts_set

    corrected_hosts = []
    corrected_guests = []
    host_names_assigned_lc = set()

    # Process the canonicalized and pre-filtered list
    for p_dict in canonicalized_and_filtered_people_list:
        if not isinstance(p_dict, dict) or not isinstance(p_dict.get("name"), str):
            logger.warning(f"Skipping invalid person entry in tidy_people (post-alias/filter): {p_dict} for podcast {current_podcast_name}")
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

        # Spurious name check is now done earlier, but double check canonicalized name just in case alias created a new spurious one (unlikely)
        if person_name_lc in SPURIOUS_PERSON_NAMES_LC:
            logger.debug(f"Filtered out spurious canonicalized person name: '{person_name_original}' for podcast {current_podcast_name}")
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
        
        # DEBUG LOGGING START
        if "anjney midha" in lc_name or "guido appenzeller" in lc_name : # Check against expected canonical names
            logger.info(f"TIDY_PEOPLE_DEBUG (final_guests_dict): Processing g_item: {g_item}, lc_name for dict key: '{lc_name}'")
        # DEBUG LOGGING END
            
        # Remove guest if name is substring or Levenshtein distance < 2 from any host
        if any(lc_name in h["name"].lower() or h["name"].lower() in lc_name or simple_levenshtein(lc_name, h["name"].lower()) < 2 for h in final_hosts):
            logger.debug(f"Filtered guest '{g_item['name']}' due to similarity to host.")
            continue
        # Remove guest if org/title is in noise list
        org = g_item.get("org", "").lower()
        title = g_item.get("title", "").lower()
        if org in SPURIOUS_ORG_NAMES_LC or title in SPURIOUS_ORG_NAMES_LC:
            logger.debug(f"Filtered guest '{g_item['name']}' due to noisy org/title.")
            continue
        if lc_name not in final_guests_dict:
            final_guests_dict[lc_name] = g_item
        else:
            # DEBUG LOGGING START
            if "anjney midha" in lc_name or "guido appenzeller" in lc_name:
                 logger.info(f"TIDY_PEOPLE_DEBUG (final_guests_dict): Merging. Existing: {final_guests_dict[lc_name]}, New: {g_item}")
            # DEBUG LOGGING END
            final_guests_dict[lc_name] = merge_people(final_guests_dict[lc_name], g_item)
    final_guests = list(final_guests_dict.values())

    # Removed default host logic for a16z Podcast, as per latest feedback.
    # Hosts will be determined by KNOWN_HOSTS or <podcast:person> tags.
    # if current_podcast_name == "a16z Podcast" and not final_hosts:
    #     final_hosts.append({"name": "a16z", "role": "host", "org": "a16z", "title": "Host"})
    #     logger.info(f"Added default host 'a16z' for podcast '{current_podcast_name}' as no other hosts were identified.")

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
        original_text_from_ner = ent.get("text", "").strip()
        label = ent.get("label", "") 
        if not label and ent.get("type"):
            label = ent.get("type")

        # Apply ALIAS_MAP first to get canonical text
        # Use lowercased original text for alias lookup, but apply alias to original-cased text if needed
        # For entities, usually we want the alias result as is, or title-cased for PERSON/ORG.
        canonical_text = ALIAS_MAP.get(original_text_from_ner.lower(), original_text_from_ner)
        text_lc = canonical_text.lower() # Use lower of canonical for filtering and seen set

        # --------- filtering (uses canonical_text and text_lc) ----------
        if not canonical_text or not label:
            logger.debug(f"Skipping entity due to missing canonical text or label: {ent} (original: '{original_text_from_ner}')")
            continue
        if len(canonical_text) < 2:
            logger.debug(f"Skipping entity (too short): '{canonical_text}' ({label})")
            continue
        if label == "PERSON" and text_lc in SPURIOUS_PERSON_NAMES_LC:
            logger.debug(f"Skipping spurious PERSON entity: '{canonical_text}'")
            continue
        if label == "ORG" and text_lc in SPURIOUS_ORG_NAMES_LC:
            logger.debug(f"Skipping spurious ORG entity: '{canonical_text}'")
            continue
        if label == "PERSON" and text_lc in people_skip_lc: # people_skip_lc contains hosts/guests
            logger.debug(f"Skipping PERSON entity (host/guest): '{canonical_text}'")
            continue

        # --------- canonicalisation of casing (on the potentially aliased text) ----------
        display_text = canonical_text # Start with the (aliased) canonical text
        if label in {"PERSON", "ORG", "PRODUCT", "EVENT", "LOC", "GPE"}:
            display_text = canonical_text.title() 
        # For other labels like MONEY, DATE, etc., display_text remains canonical_text (original casing from alias or NER)

        # Deduplication based on canonicalized text (lower) and label
        sig = (text_lc, label) # text_lc is already lowercased canonical_text
        if sig in seen:
            logger.debug(f"Skipping duplicate entity (post-canonicalization): '{display_text}' ({label})")
            continue

        cleaned.append({
            "text": display_text,  # Store the canonicalized and cased text
            "label": label,
            "start_char": ent.get("start_char"),
            "end_char": ent.get("end_char"),
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

MIN_DURATION_FOR_SUBTITLE_SEARCH = 120 # seconds, e.g. 2 minutes
MAX_SUBTITLE_SEARCH_DURATION = 600 # seconds, e.g. 10 minutes

# For refining hosts and guests after initial extraction
# KNOWN_HOSTS = {  <-- THIS BLOCK TO BE DELETED
#     "The Twenty Minute VC (20VC): Venture Capital | Startup Funding | The Pitch": {
#         "harry stebbings",
#     },
# REMOVED: a16z hardcoded mapping - use YAML config instead
# } <-- END OF BLOCK TO BE DELETED

SPURIOUS_GUESTS_LC = {
    "chimes ipo", 
    "twitter",
    # Add other common non-person entities that might appear as guests
}

# ALIAS_MAP = {} # <-- THIS LINE TO BE DELETED
# ALIAS_MAP_PATH = Path(__file__).parent / "alias_map.json" # <-- THIS LINE TO BE DELETED

def refine_people_roles(hosts_list, guests_list, podcast_title_str, known_hosts_map, spurious_guests_set_lc):
    """
    Refines hosts and guests lists.
    - Promotes known guests to hosts based on podcast_title.
    - Removes spurious guests.
    - Ensures there's at least one default host if none are identified.
    """
    refined_hosts = []
    potential_guests = [] # Start with all original guests

    # First, consolidate original hosts and guests into a single list to check against KNOWN_HOSTS
    current_people = hosts_list + guests_list
    
    processed_host_names = set()

    for person_entry in current_people:
        if not isinstance(person_entry, dict) or "name" not in person_entry:
            potential_guests.append(person_entry) # Keep malformed entries as potential guests for now
            continue

        name_lower = person_entry["name"].lower()
        
        # Check if this person is a known host for THIS podcast
        if podcast_title_str in known_hosts_map and name_lower in known_hosts_map[podcast_title_str]:
            if name_lower not in processed_host_names:
                refined_hosts.append({"name": person_entry["name"], "role": "host"})
                processed_host_names.add(name_lower)
        else:
            potential_guests.append(person_entry) # If not a known host, add to potential guests

    # Now filter potential_guests to remove spurious ones
    final_guests = []
    processed_guest_names = set()
    for person_entry in potential_guests:
        if not isinstance(person_entry, dict) or "name" not in person_entry:
            final_guests.append(person_entry) # Keep malformed
            continue
        
        name_lower = person_entry["name"].lower()
        # Ensure they are not already listed as a host (by name) and not spurious
        if name_lower not in processed_host_names and name_lower not in spurious_guests_set_lc:
            if name_lower not in processed_guest_names:
                 # Ensure role is 'guest' if it was something else or missing
                final_guests.append({"name": person_entry["name"], "role": "guest"})
                processed_guest_names.add(name_lower)

    # Safety net: if no hosts were identified for "The Twenty Minute VC", add Harry Stebbings.
    # This can be generalized if needed for other podcasts.
    if not refined_hosts and podcast_title_str == "The Twenty Minute VC (20VC): Venture Capital | Startup Funding | The Pitch":
        if "harry stebbings" not in processed_host_names: # Check again in case he was added as guest
            refined_hosts.append({"name": "Harry Stebbings", "role": "host"})
            # If Harry was in final_guests, remove him
            final_guests = [g for g in final_guests if g["name"].lower() != "harry stebbings"]

    return refined_hosts, final_guests

def cleanup_old_meta(meta_dir, guid, keep_path):
    """Delete all meta files for a GUID except the one at keep_path. Thread-safe for parallel runs."""
    old_files = glob.glob(f"{meta_dir}/meta_{guid}_*.json")
    for f in old_files:
        # If two workers race, the loser's os.remove will raise FileNotFoundError – safe to ignore
        try:
            if os.path.abspath(f) != os.path.abspath(keep_path):
                os.remove(f)
                logger.info(f"Removed old meta file: {f}")
        except FileNotFoundError:
            logger.warning(f"Attempted to remove old meta file {f}, but it was not found (possibly removed by another process).")
        except Exception as e:
            logger.error(f"Error removing old meta file {f}: {e}")
    logger.info(f"Meta cleanup process completed for GUID {guid} around path {keep_path}.")

SPURIOUS_PERSON_NAMES_LC = {
    "chimes ipo", 
    "twitter",
    # Add other common non-person entities that NER might misclassify as PERSON
    "ipo", # general ipo might be misclassified
    "ai",
    "api",
    "kajabi", # Added from entity list feedback
    "mode", # Added from entity list feedback
    "secureframe", # ADDED from ChatGPT review
    "angel list",   # ADDED from ChatGPT review
    "previously luke heldrolls", # ADDED from new review
    "runner", # ADDED from new review
    "bat", # Added from latest review
    "cliner perkins" # Added from latest review to ensure it's removed if alias doesn't catch it
}

SPURIOUS_ORG_NAMES_LC = {
    "rory o'dry school",
    # Add other known spurious organization names here
    "ipo" # Added from entity list feedback
}

# For Levenshtein distance in guest deduplication
def simple_levenshtein(a, b):
    if a == b:
        return 0
    if not a: return len(b)
    if not b: return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr_row = [i]
        for j, cb in enumerate(b, 1):
            insertions = prev_row[j] + 1
            deletions = curr_row[j - 1] + 1
            substitutions = prev_row[j - 1] + (ca != cb)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def save_meta_file(meta, meta_podcast_dir, guid):
    # Use slugify_title for the filename
    # Assuming meta contains 'title' or 'episode_title' for the slug
    episode_title_for_slug = meta.get('episode_title', meta.get('title', 'unknown-episode'))
    # Ensure guid is a string, as slugify_title expects it.
    # meta.get('guid') should already provide this but defensive check.
    guid_str = str(guid) if guid is not None else str(uuid.uuid4()) 

    filename = slugify_title(episode_title_for_slug, guid_str, "json")
    local_meta_file_path = meta_podcast_dir / filename
    logger.info(f"Attempting to save metadata to: {local_meta_file_path}")

    try:
        with open(local_meta_file_path, "w", encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"SUCCESS: Final combined metadata saved locally to {local_meta_file_path} for GUID {guid}")
    finally:
        logger.info(f"Initiating cleanup of old meta files for GUID {guid} in directory {meta_podcast_dir}, keeping {local_meta_file_path}")
        cleanup_old_meta(str(meta_podcast_dir), guid_str, str(local_meta_file_path))

def slugify_title(title: str, guid: str, ext: str) -> str:
    import re
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:80]
    return f"{slug}_{guid}.{ext}"

def save_segments_file(segments, podcast_slug, guid, base_data_dir="data"):
    from pathlib import Path
    segments_dir = Path(base_data_dir) / "segments" / podcast_slug
    segments_dir.mkdir(parents=True, exist_ok=True)
    # Filename is just {guid}.json as per new requirement
    seg_path = segments_dir / f"{guid}.json" # <--- THIS LOOKS RIGHT
    with open(seg_path, "w", encoding='utf-8') as f:
        import json
        json.dump(segments, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved segments to {seg_path}") # Added logging
    return str(seg_path.resolve())

def md5_8(s):
    return hashlib.md5(s.encode()).hexdigest()[:8]

def episode_file_identifier(simple_title_slug, guid):
    return f"{simple_title_slug.strip('-')}_{md5_8(guid)}"

def audio_mp3_fname(episode_file_identifier):
    return f"{episode_file_identifier}.mp3"

ALIAS_MAP_KEYS_LOGGED = False