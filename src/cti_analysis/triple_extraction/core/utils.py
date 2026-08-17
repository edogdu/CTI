"""
UTILITY FUNCTIONS
JSON repair, embedding, text processing

File: core/utils.py
"""
import json
import re
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# JSON REPAIR
# ============================================================================

def repair_json(text: str) -> str:
    """
    Repair malformed JSON.
    """
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Try to extract JSON array or object
    # Look for [...] or {...}
    array_match = re.search(r'\[.*\]', text, re.DOTALL)
    if array_match:
        text = array_match.group()
    else:
        object_match = re.search(r'\{.*\}', text, re.DOTALL)
        if object_match:
            text = object_match.group()

    # Fix common issues
    # Unquoted keys
    text = re.sub(r'(\w+):', r'"\1":', text)

    # Single quotes to double quotes
    text = text.replace("'", '"')

    # Remove trailing commas
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    return text


# ============================================================================
# EMBEDDING (sentence-transformers)
# ============================================================================

_st_model = None


def _get_st_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load sentence-transformers model."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(model_name)
    return _st_model


def embed_text(text: str, model: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate text embedding using sentence-transformers."""
    try:
        st = _get_st_model(model)
        embedding = st.encode(text, normalize_embeddings=True)
        return np.array(embedding, dtype=float)
    except Exception as e:
        logger.warning("Embedding error: %s", e)
        return np.zeros(384)  # all-MiniLM-L6-v2 dimension


def embed_triples(triples: List[tuple], model: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a list of (subject, predicate, object) tuples."""
    texts = []
    for s, p, o in triples:
        text = f"{str(s).lower()} {str(p).lower()} {str(o).lower()}"
        texts.append(text)

    if not texts:
        return np.zeros((0, 384))

    st = _get_st_model(model)
    return st.encode(texts, normalize_embeddings=True)


# ============================================================================
# TEXT PROCESSING
# ============================================================================

def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception:
        # Fallback: regex split
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def count_tokens(text: str) -> int:
    """Simple token counter (approximate)."""
    return len(text.split())


# ============================================================================
# HASHING
# ============================================================================

def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of file."""
    import hashlib

    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================================
# SAFE FILENAME
# ============================================================================

def safe_filename(name: str) -> str:
    """Make filename safe for filesystem."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)
