# --- maxmin_chunking.py (or inline in extraction_updated_ontology.py) ---

import logging
import math
import numpy as np
from typing import List, Optional

from cti_analysis.models.documents import NormalizedDocument, Chunk, chunk_from_text

logger = logging.getLogger(__name__)

# Lazy-loaded sentence-transformers model
_st_model = None


def _get_st_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load sentence-transformers model."""
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers model: %s", model_name)
            _st_model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for semantic chunking. "
                "Install with: pip install sentence-transformers"
            )
    return _st_model


def _embed(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[np.ndarray]:
    """Embed texts using sentence-transformers (local, no server needed)."""
    model = _get_st_model(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True)
    return [embeddings[i] for i in range(len(texts))]


def _min_pairwise_cos(embs):
    """Return the minimum pairwise cosine similarity in a set (fast enough for small windows)."""
    if len(embs) <= 1:
        return 1.0
    M = np.vstack(embs)
    S = M @ M.T
    # ignore diagonal
    mins = []
    for i in range(S.shape[0]):
        row = np.delete(S[i], i)
        mins.append(row.min())
    return float(min(mins))

def maxmin_semantic_chunks(
    sentences: list[str],
    *,
    embed_model: str = "all-MiniLM-L6-v2",
    split_threshold: float = 0.35,  # split when adjacent cosine drops below this
    max_words: int = 300,           # hard size cap per chunk
) -> list[str]:
    """
    Adjacent-cosine semantic chunking.

    Computes cosine similarity between each consecutive sentence pair.
    Merges consecutive sentences into a chunk as long as:
      - the adjacent cosine stays above split_threshold
      - the cumulative word count stays below max_words

    When either condition fails, a boundary is placed and a new chunk starts.
    No overlap — each sentence appears in exactly one chunk.
    """
    if not sentences:
        return []
    if len(sentences) == 1:
        return list(sentences)

    # Embed all sentences once
    embs = _embed(sentences, model_name=embed_model)

    # Compute adjacent cosine similarities
    M = np.vstack(embs)
    adj_cos = []
    for i in range(len(sentences) - 1):
        adj_cos.append(float(M[i] @ M[i + 1]))

    # Greedy chunking: merge while adjacent similarity is high enough
    chunks: list[str] = []
    current = [sentences[0]]
    current_words = len(sentences[0].split())

    for i in range(1, len(sentences)):
        next_words = len(sentences[i].split())
        sim = adj_cos[i - 1]

        if sim >= split_threshold and (current_words + next_words) <= max_words:
            # Continue current chunk
            current.append(sentences[i])
            current_words += next_words
        else:
            # Boundary: save current chunk, start new one
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_words = next_words

    # Don't forget the last chunk
    if current:
        chunks.append(" ".join(current))

    return chunks


def run_chunking(cfg, docs: List[NormalizedDocument], sentence_chunks: Optional[List[Chunk]] = None) -> List[Chunk]:
    """Run semantic chunking: merge pre-sentencized chunks into semantic groups.

    If sentence_chunks are provided, merges them based on semantic similarity.
    Otherwise, falls back to splitting the document text on periods.
    """
    embed_model = getattr(cfg, "embed_model", "all-MiniLM-L6-v2")

    if sentence_chunks and len(sentence_chunks) > 1:
        # Use existing sentence-level chunks — merge adjacent similar ones
        sentences = [c.text for c in sentence_chunks]
        doc = sentence_chunks[0].source_doc if hasattr(sentence_chunks[0], "source_doc") else docs[0] if docs else None

        chunk_texts = maxmin_semantic_chunks(
            sentences,
            embed_model=embed_model,
        )

        merged_chunks: List[Chunk] = []
        for i, ct in enumerate(chunk_texts):
            doc_id = sentence_chunks[0].chunk_id.rsplit("_", 2)[0] if sentence_chunks else "unknown"
            merged_chunks.append(chunk_from_text(
                doc if doc else docs[0],
                f"{doc_id}_schunk{i}",
                ct,
            ))
        return merged_chunks

    # Fallback: split document text on periods
    all_chunks: List[Chunk] = []
    for doc in docs:
        sentences = [s.strip() for s in doc.text.split(".") if s.strip()]
        if len(sentences) <= 1:
            all_chunks.append(chunk_from_text(doc, f"{doc.doc_id}_chunk0", doc.text))
            continue

        chunk_texts = maxmin_semantic_chunks(
            sentences,
            embed_model=embed_model,
        )

        for i, ct in enumerate(chunk_texts):
            all_chunks.append(chunk_from_text(doc, f"{doc.doc_id}_chunk{i}", ct))

    return all_chunks
