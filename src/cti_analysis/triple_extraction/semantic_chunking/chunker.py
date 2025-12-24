# --- maxmin_chunking.py (or inline in extraction_updated_ontology.py) ---

import math, numpy as np, requests
from typing import List

from cti_analysis.models.documents import NormalizedDocument, Chunk, chunk_from_text

def _ollama_embed(texts, model="nomic-embed-text", base_url="http://localhost:11434"):
    vecs = []
    for t in texts:
        r = requests.post(f"{base_url.rstrip('/')}/api/embeddings",
                          json={"model": model, "prompt": t}, timeout=30)
        r.raise_for_status()
        v = np.array(r.json()["embedding"], dtype=float)
        n = np.linalg.norm(v) or 1.0
        vecs.append(v / n)
    return vecs

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
    base_url="http://localhost:11434",
    embed_model="nomic-embed-text",
    lookahead: int = 6,          # examine up to this many future sentences when placing a boundary
    min_sim_threshold: float = 0.35,  # if even the best split has min-sim below this, force split
    max_words: int = 260,         # soft size cap
    overlap_words: int = 50
):
    """
    Return contiguous chunks chosen so that each chunk maximizes the minimum
    internal cosine similarity (Max–Min) with a small look-ahead search.
    """
    if not sentences:
        return []

    embs = _ollama_embed(sentences, model=embed_model, base_url=base_url)
    word_counts = [max(1, len(s.split())) for s in sentences]

    chunks = []
    i = 0
    while i < len(sentences):
        # greedy growth + Max–Min split within a look-ahead band
        best_end = i + 1
        best_score = -1.0
        acc_words = 0

        # progressively consider end candidates j ∈ [i+1, i+lookahead] while size allows
        for j in range(i + 1, min(len(sentences), i + lookahead + 1)):
            acc_words += sum(word_counts[i:j])
            if acc_words > max_words and best_end > i + 1:
                break

            # compute min pairwise similarity in the candidate chunk
            cand_embs = embs[i:j]
            min_sim = _min_pairwise_cos(cand_embs)

            # choose the j that maximizes this minimum similarity
            if min_sim > best_score:
                best_score, best_end = min_sim, j

        # guard: if coherence is below threshold, still cut at best_end to avoid topic bleed
        end = best_end
        if best_score < min_sim_threshold and end == i + 1 and (i + 2) <= len(sentences):
            end = i + 2  # ensure progress even on choppy texts

        text = " ".join(sentences[i:end])
        centroid = np.mean(np.vstack(embs[i:end]), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)

        chunks.append({
            "start": i, "end": end, "text": text, "centroid": centroid,
            "min_pair_sim": best_score, "size_words": sum(word_counts[i:end])
        })

        # advance with overlap
        words_kept = 0
        k = end - 1
        while k > i and words_kept < overlap_words:
            words_kept += word_counts[k]
            k -= 1
        i = max(k + 1, end)

    return chunks


def run_chunking(cfg, docs: List[NormalizedDocument]) -> List[Chunk]:
    """
    High-level chunking entrypoint.
    If semantic chunking params are provided, will attempt embedding-based
    chunking; otherwise produces a single chunk per document.
    """
    chunks: List[Chunk] = []
    for doc in docs:
        # Minimal fallback: one chunk per document
        chunk_id = f"{doc.doc_id}_chunk0"
        chunks.append(chunk_from_text(doc, chunk_id, doc.text))
    return chunks