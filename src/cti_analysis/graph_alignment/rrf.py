"""Reciprocal Rank Fusion (RRF) for combining ranked result lists.

Merges multiple ranked retrieval results (e.g., vector similarity + BM25)
into a single ranked list using the RRF formula:

    score(item) = sum(1 / (k + rank + 1)) across all lists containing the item

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
individual Rank Learning Methods", SIGIR 2009.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple


def reciprocal_rank_fusion(
    *ranked_lists: List[str],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Fuse multiple ranked lists of IDs using RRF.

    Args:
        *ranked_lists: Each list is a ranked sequence of ATT&CK IDs
                       (index 0 = rank 1, highest priority).
        k: RRF constant (default 60, from the original paper).

    Returns:
        List of (id, rrf_score) tuples sorted by score descending.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fuse_similarity_and_bm25(
    similarity_dir,
    token_search_dir,
    output_dir,
    k: int = 60,
) -> Dict:
    """Fuse vector similarity and BM25 results for a single document.

    Reads per-node similarity results and token search results,
    extracts ranked ATT&CK ID lists from each, and applies RRF.

    Args:
        similarity_dir: Path to similarity results dir (has similarity_per_node.json)
        token_search_dir: Path to token search dir (has token_search.json)
        output_dir: Path to write hybrid results
        k: RRF constant

    Returns:
        Dict with hybrid results (ranked IDs, scores, etc.)
    """
    import json
    import re
    from pathlib import Path

    sim_dir = Path(similarity_dir)
    ts_dir = Path(token_search_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ATTACK_LABELS = {"UcoexMITREATTACK", "UcoexTACTICS", "UcoexSOFTWARE"}

    # --- Extract ranked IDs from vector similarity ---
    sim_ranked: List[str] = []
    per_node = sim_dir / "similarity_per_node.json"
    if per_node.exists():
        obj = json.loads(per_node.read_text(encoding="utf-8"))

        # Collect all (clean_id, cosine) pairs across all queries
        all_hits = []
        for entry in (obj.values() if isinstance(obj, dict) else obj):
            if not isinstance(entry, dict):
                continue
            for hit in (entry.get("top_k") or []):
                target = hit.get("target") or {}
                labels = set(target.get("labels") or [])
                if not (ATTACK_LABELS & labels):
                    continue
                cid = (target.get("clean_id") or "").strip().upper()
                cosine = (hit.get("scores") or {}).get("cosine", 0.0)
                if cid:
                    all_hits.append((cid, cosine))

        # Deduplicate: keep best score per ID, then rank by score
        best_scores: Dict[str, float] = {}
        for cid, score in all_hits:
            if cid not in best_scores or score > best_scores[cid]:
                best_scores[cid] = score
        sim_ranked = [cid for cid, _ in sorted(best_scores.items(), key=lambda x: x[1], reverse=True)]

    # --- Extract ranked IDs from BM25/token search ---
    bm25_ranked: List[str] = []
    ts_file = ts_dir / "token_search.json"
    if ts_file.exists():
        ts = json.loads(ts_file.read_text(encoding="utf-8"))

        # Collect all (clean_id, bm25_score) pairs from entity matches
        all_bm25 = []
        for entry in ts.get("per_entity", []):
            for hit in entry.get("hits", []):
                cid = hit.get("clean_id", "").strip().upper()
                score = hit.get("score", 0.0)
                if cid:
                    all_bm25.append((cid, score))

        # Deduplicate: keep best score per ID, then rank by score
        best_bm25: Dict[str, float] = {}
        for cid, score in all_bm25:
            if cid not in best_bm25 or score > best_bm25[cid]:
                best_bm25[cid] = score
        bm25_ranked = [cid for cid, _ in sorted(best_bm25.items(), key=lambda x: x[1], reverse=True)]

        # Also add regex IDs at the top (they're exact matches, highest confidence)
        regex_ids = ts.get("regex_matched_ids", [])
        if regex_ids:
            bm25_ranked = regex_ids + [cid for cid in bm25_ranked if cid not in set(regex_ids)]

    # --- RRF fusion ---
    fused = reciprocal_rank_fusion(sim_ranked, bm25_ranked, k=k)

    results = {
        "method": "hybrid-rrf",
        "k": k,
        "n_sim_ids": len(sim_ranked),
        "n_bm25_ids": len(bm25_ranked),
        "n_fused": len(fused),
        "fused_ranked": [{"id": cid, "rrf_score": round(score, 6)} for cid, score in fused],
        "fused_ids": [cid for cid, _ in fused],
        "sim_only_ids": sorted(set(sim_ranked) - set(bm25_ranked)),
        "bm25_only_ids": sorted(set(bm25_ranked) - set(sim_ranked)),
        "overlap_ids": sorted(set(sim_ranked) & set(bm25_ranked)),
    }

    out_path = out_dir / "hybrid_rrf.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def fuse_all_sources(
    similarity_dir,
    token_search_dir,
    sentence_search_dir,
    output_dir,
    k: int = 60,
) -> Dict:
    """Fuse three retrieval sources: entity vector, BM25, and sentence vector.

    Reads results from each source, extracts ranked ATT&CK ID lists,
    and applies RRF across all three.
    """
    import json
    from pathlib import Path

    sim_dir = Path(similarity_dir)
    ts_dir = Path(token_search_dir)
    ss_dir = Path(sentence_search_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ATTACK_LABELS = {"UcoexMITREATTACK", "UcoexTACTICS", "UcoexSOFTWARE"}

    # --- Source 1: Entity-neighbors vector similarity ---
    sim_ranked: List[str] = []
    per_node = sim_dir / "similarity_per_node.json"
    if per_node.exists():
        obj = json.loads(per_node.read_text(encoding="utf-8"))
        best_scores: Dict[str, float] = {}
        for entry in (obj.values() if isinstance(obj, dict) else obj):
            if not isinstance(entry, dict):
                continue
            for hit in (entry.get("top_k") or []):
                target = hit.get("target") or {}
                labels = set(target.get("labels") or [])
                if not (ATTACK_LABELS & labels):
                    continue
                cid = (target.get("clean_id") or "").strip().upper()
                cosine = (hit.get("scores") or {}).get("cosine", 0.0)
                if cid and (cid not in best_scores or cosine > best_scores[cid]):
                    best_scores[cid] = cosine
        sim_ranked = [cid for cid, _ in sorted(best_scores.items(), key=lambda x: x[1], reverse=True)]

    # --- Source 2: BM25 token search ---
    bm25_ranked: List[str] = []
    ts_file = ts_dir / "token_search.json"
    if ts_file.exists():
        ts = json.loads(ts_file.read_text(encoding="utf-8"))
        best_bm25: Dict[str, float] = {}
        for entry in ts.get("per_entity", []):
            for hit in entry.get("hits", []):
                cid = hit.get("clean_id", "").strip().upper()
                score = hit.get("score", 0.0)
                if cid and (cid not in best_bm25 or score > best_bm25[cid]):
                    best_bm25[cid] = score
        bm25_ranked = [cid for cid, _ in sorted(best_bm25.items(), key=lambda x: x[1], reverse=True)]
        regex_ids = ts.get("regex_matched_ids", [])
        if regex_ids:
            bm25_ranked = regex_ids + [cid for cid in bm25_ranked if cid not in set(regex_ids)]

    # --- Source 3: Sentence-level vector similarity ---
    sent_ranked: List[str] = []
    ss_file = ss_dir / "sentence_search.json"
    if ss_file.exists():
        ss = json.loads(ss_file.read_text(encoding="utf-8"))
        sent_ranked = ss.get("ranked_ids", [])

    # --- RRF reranking: use vector top-N as candidate pool, boost with other sources ---
    candidate_pool_size = 100
    candidate_ids = set(sim_ranked[:candidate_pool_size])

    # Filter BM25 and sentence lists to only candidates in the pool
    bm25_in_pool = [cid for cid in bm25_ranked if cid in candidate_ids]
    sent_in_pool = [cid for cid in sent_ranked if cid in candidate_ids]

    # Also allow BM25/sentence to promote items outside pool (new discoveries)
    # but cap at a reasonable number to avoid noise
    bm25_new = [cid for cid in bm25_ranked[:50] if cid not in candidate_ids]
    sent_new = [cid for cid in sent_ranked[:50] if cid not in candidate_ids]

    # Build the three ranked lists for RRF:
    # 1. Vector similarity (full pool ranking)
    # 2. BM25 (in-pool + top new)
    # 3. Sentence (in-pool + top new)
    rrf_sim = sim_ranked[:candidate_pool_size]
    rrf_bm25 = bm25_in_pool + bm25_new
    rrf_sent = sent_in_pool + sent_new

    fused = reciprocal_rank_fusion(rrf_sim, rrf_bm25, rrf_sent, k=k)

    # Track source contributions
    sim_set = set(rrf_sim)
    bm25_set = set(rrf_bm25)
    sent_set = set(rrf_sent)

    source_counts = {"all_three": 0, "two_sources": 0, "one_source": 0}
    for cid, _ in fused:
        n = sum([cid in sim_set, cid in bm25_set, cid in sent_set])
        if n >= 3:
            source_counts["all_three"] += 1
        elif n == 2:
            source_counts["two_sources"] += 1
        else:
            source_counts["one_source"] += 1

    results = {
        "method": "hybrid-rrf-rerank",
        "k": k,
        "candidate_pool_size": candidate_pool_size,
        "sources": {
            "entity_vector": len(rrf_sim),
            "bm25": len(rrf_bm25),
            "sentence_vector": len(rrf_sent),
        },
        "source_counts": source_counts,
        "n_fused": len(fused),
        "fused_ranked": [{"id": cid, "rrf_score": round(score, 6)} for cid, score in fused],
        "fused_ids": [cid for cid, _ in fused],
    }

    out_path = out_dir / "hybrid_rrf.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results
