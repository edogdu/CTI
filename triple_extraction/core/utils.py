"""
UTILITY FUNCTIONS
From your notebook - LLM interaction, JSON repair, embedding

File: core/utils.py
"""
import json
from multiprocessing.util import debug
import re
import requests
from typing import List, Optional
import numpy as np


# ============================================================================
# LLM INTERACTION (From Notebook)
# ============================================================================

def generate_response(prompt: str, 
                     model: str = "gemma2:9b",
                     base_url: str = "http://localhost:11434",
                     max_tokens: int = 1200,
                     temperature: float = 0.1,
                     debug: bool = False) -> str:
    """
    Generate LLM response (from your notebook).
    """
    
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        if debug:
            print(f"[LLM] Prompt length: {len(prompt)} chars")
            print(f"[LLM] Response length: {len(result.get('response', ''))} chars")
        
        return result.get("response", "")
    
    except Exception as e:
        print(f"[LLM Error] {e}")
        return ""


# ============================================================================
# JSON REPAIR (From Notebook)
# ============================================================================

def repair_json(text: str) -> str:
    """
    Repair malformed JSON (from your notebook).
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
# EMBEDDING (From Notebook)
# ============================================================================

def embed_text(text: str, 
               model: str = "nomic-embed-text",
               base_url: str = "http://localhost:11434") -> np.ndarray:
    """
    Generate text embedding using Ollama (from your notebook).
    """
    try:
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        
        embedding = np.array(response.json()["embedding"], dtype=float)
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return np.zeros(768)  # Default dimension


def embed_triples(triples: List[tuple]) -> np.ndarray:
    """
    Embed a list of (subject, predicate, object) tuples.
    From your notebook_eval.py
    """
    texts = []
    for s, p, o in triples:
        text = f"{str(s).lower()} {str(p).lower()} {str(o).lower()}"
        texts.append(text)
    
    if not texts:
        return np.zeros((0, 768))
    
    embeddings = []
    for text in texts:
        emb = embed_text(text)
        embeddings.append(emb)
    
    return np.vstack(embeddings)


# ============================================================================
# TEXT PROCESSING (From Notebook)
# ============================================================================

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences (from your notebook Cell 11).
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except:
        # Fallback: regex split
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def count_tokens(text: str) -> int:
    """Simple token counter (approximate)."""
    return len(text.split())


# ============================================================================
# OLLAMA MODEL MANAGEMENT
# ============================================================================

def ensure_ollama_model(model_name: str, base_url: str = "http://localhost:11434"):
    """
    Ensure Ollama model is available (from your extraction.py).
    """
    try:
        # Check available models
        resp = requests.get(f"{base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        
        if model_name not in models:
            print(f"Pulling model '{model_name}'...")
            pull_resp = requests.post(
                f"{base_url}/api/pull",
                json={"name": model_name},
                timeout=300
            )
            pull_resp.raise_for_status()
            print(f"✓ Model '{model_name}' ready")
        else:
            print(f"✓ Model '{model_name}' already available")
    
    except Exception as e:
        print(f"Warning: Could not ensure model: {e}")


# ============================================================================
# HASHING (From your extraction.py)
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
# SAFE FILENAME (From your extraction.py)
# ============================================================================

def safe_filename(name: str) -> str:
    """Make filename safe for filesystem."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)


if __name__ == "__main__":
    print("CTI Utils - Testing")
    print("=" * 60)
    
    # Test JSON repair
    bad_json = "```json\n{'key': 'value',}\n```"
    fixed = repair_json(bad_json)
    print(f"JSON repair: {fixed}")
    
    # Test sentence split
    text = "APT29 used Cobalt Strike. They targeted government networks."
    sentences = split_sentences(text)
    print(f"Sentences: {sentences}")
    
    print("\n✓ Utils ready")
