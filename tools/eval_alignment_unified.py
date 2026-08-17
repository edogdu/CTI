"""Unified evaluation of all graph alignment retrieval methods on CTI-HAL.

Computes Recall@k, Precision@k, and MRR for:
  1. Entity-neighbors vector similarity
  2. BM25 full-text search
  3. Sentence vector similarity
  4. Sentence-level 3-source RRF
  5. Entity-level hybrid RRF

All methods evaluated on the same document set with the same ground truth.

Usage:
    python tools/eval_alignment_unified.py
    python tools/eval_alignment_unified.py --output results.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "src"))

EVAL_DIR = _repo / "experiments" / "ctihal-pipeline" / "eval"
ANN_DIR = _repo / "datasets" / "CTI-HAL" / "data"
MAPPINGS_CSV = _repo / "config" / "CTI_HAL_mappings.csv"
ATTACK_LABELS = {"UcoexMITREATTACK", "UcoexTACTICS", "UcoexSOFTWARE"}
KS = [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_mappings() -> Dict[str, Dict[str, str]]:
    mappings = {}
    with open(MAPPINGS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            doc = (row.get("Document") or "").strip()
            ident = (row.get("Identifier") or "").strip()
            group = (row.get("Group") or "").strip()
            if doc and ident:
                mappings[doc.lower().replace(".pdf", "")] = {"identifier": ident, "group": group}
    return mappings


def load_gt(ident: str, group: str) -> Dict[str, Set[str]]:
    techs, tacts, soft = set(), set(), set()
    p = ANN_DIR / group.lower() / f"{ident}.json"
    if not p.exists():
        return {"techniques": techs, "tactics": tacts, "software": soft}
    for ann in json.loads(p.read_text(encoding="utf-8")):
        t = ann.get("technique")
        if isinstance(t, str) and t.strip():
            techs.add(t.strip().upper())
        md = ann.get("metadata") or {}
        sub = md.get("sub_technique")
        if isinstance(sub, str) and sub.strip():
            techs.add(sub.strip().upper())
        for tid in md.get("tactic") or []:
            if isinstance(tid, str) and tid.strip():
                tacts.add(tid.strip().upper())
        for sid in md.get("tool") or []:
            if isinstance(sid, str) and sid.strip():
                soft.add(sid.strip().upper())
    return {"techniques": techs, "tactics": tacts, "software": soft}


def _normalize_stem(s: str) -> str:
    """Strip all non-alphanumeric chars for fuzzy matching."""
    import re
    return re.sub(r'[^a-z0-9]', '', s.lower())


def find_mapping(doc_id: str, mappings: dict):
    parts = doc_id.split("_", 1)
    pdf_stem = parts[1] if len(parts) > 1 else doc_id
    # Exact match first
    mapping = mappings.get(pdf_stem.lower())
    if mapping:
        return mapping
    # Normalized match (handles underscore vs space differences)
    norm_stem = _normalize_stem(pdf_stem)
    for stem, m in mappings.items():
        if _normalize_stem(stem) == norm_stem:
            return m
    # Substring fallback
    for stem, m in mappings.items():
        if pdf_stem.lower() in stem or stem in pdf_stem.lower():
            return m
    return None


def rollup(ids: Set[str]) -> Set[str]:
    expanded = set(ids)
    for i in ids:
        if "." in i and i.startswith("T"):
            expanded.add(i.split(".")[0])
    return expanded


# ---------------------------------------------------------------------------
# Per-method ranked list extraction
# ---------------------------------------------------------------------------

def _load_similarity_json(doc_dir: Path):
    """Load similarity_per_node.json, falling back to leaderboard if unavailable."""
    per_node = doc_dir / "similarity_per_node.json"
    try:
        return json.loads(per_node.read_text(encoding="utf-8")), "per_node"
    except (FileNotFoundError, OSError):
        pass
    leaderboard = doc_dir / "similarity_leaderboard.json"
    try:
        return json.loads(leaderboard.read_text(encoding="utf-8")), "leaderboard"
    except (FileNotFoundError, OSError):
        return None, None


def extract_entity_vec_ranked(doc_dir: Path) -> List[str]:
    """Entity-neighbors vector: per-entity top-k union, ranked by best cosine."""
    obj, fmt = _load_similarity_json(doc_dir)
    if obj is None:
        return []
    best: Dict[str, float] = {}
    if fmt == "leaderboard" and isinstance(obj, list):
        for entry in obj:
            target = entry.get("target") or {}
            labels = set(target.get("labels") or [])
            if not (ATTACK_LABELS & labels):
                continue
            cid = (target.get("clean_id") or "").strip().upper()
            cos = (entry.get("scores") or {}).get("cosine", 0.0)
            if cid and (cid not in best or cos > best[cid]):
                best[cid] = cos
    else:
        for entry in (obj.values() if isinstance(obj, dict) else obj):
            if not isinstance(entry, dict):
                continue
            for hit in entry.get("top_k") or []:
                target = hit.get("target") or {}
                labels = set(target.get("labels") or [])
                if not (ATTACK_LABELS & labels):
                    continue
                cid = (target.get("clean_id") or "").strip().upper()
                cos = (hit.get("scores") or {}).get("cosine", 0.0)
                if cid and (cid not in best or cos > best[cid]):
                    best[cid] = cos
    return [c for c, _ in sorted(best.items(), key=lambda x: x[1], reverse=True)]


def extract_entity_vec_pool(doc_dir: Path, per_entity_k: int = 10) -> Set[str]:
    """Entity-neighbors vector: per-entity top-k union (the pool approach used in main eval)."""
    obj, fmt = _load_similarity_json(doc_dir)
    if obj is None:
        return set()
    if fmt == "leaderboard" and isinstance(obj, list):
        # Leaderboard is already a global ranking — take all unique IDs
        pool = set()
        for entry in obj:
            target = entry.get("target") or {}
            labels = set(target.get("labels") or [])
            if not (ATTACK_LABELS & labels):
                continue
            cid = (target.get("clean_id") or "").strip().upper()
            if cid:
                pool.add(cid)
        return pool
    pool = set()
    for entry in (obj.values() if isinstance(obj, dict) else obj):
        if not isinstance(entry, dict):
            continue
        count = 0
        for hit in entry.get("top_k") or []:
            if count >= per_entity_k:
                break
            target = hit.get("target") or {}
            labels = set(target.get("labels") or [])
            if not (ATTACK_LABELS & labels):
                continue
            cid = (target.get("clean_id") or "").strip().upper()
            if cid:
                pool.add(cid)
                count += 1
    return pool


def extract_bm25_ranked(doc_dir: Path) -> List[str]:
    """BM25: ranked by Lucene score."""
    ts_file = doc_dir / "token_search.json"
    try:
        ts = json.loads(ts_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return []
    best: Dict[str, float] = {}
    for entry in ts.get("per_entity", []):
        for hit in entry.get("hits", []):
            cid = hit.get("clean_id", "").strip().upper()
            score = hit.get("score", 0.0)
            if cid and (cid not in best or score > best[cid]):
                best[cid] = score
    ranked = [c for c, _ in sorted(best.items(), key=lambda x: x[1], reverse=True)]
    # Prepend regex matches (exact ID matches, highest confidence)
    regex_ids = [cid.upper() for cid in ts.get("regex_matched_ids", [])]
    if regex_ids:
        ranked = regex_ids + [c for c in ranked if c not in set(regex_ids)]
    return ranked


def extract_sentence_vec_ranked(doc_dir: Path) -> List[str]:
    """Sentence vector: ranked by best cosine across sentences."""
    ss_file = doc_dir / "sentence_search.json"
    try:
        ss = json.loads(ss_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return []
    return [cid.upper() for cid in ss.get("ranked_ids", [])]


def extract_sentence_rrf_ranked(doc_dir: Path, variant: str = "max") -> List[str]:
    """Sentence-level RRF: ranked by max or sum score."""
    sr_file = doc_dir / "sentence_retrieval.json"
    try:
        sr = json.loads(sr_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return []
    key = f"ranked_ids_{variant}"
    return [cid.upper() for cid in sr.get(key, [])]


def extract_hybrid_rrf_ranked(doc_dir: Path) -> List[str]:
    """Entity-level hybrid RRF."""
    rrf_file = doc_dir / "hybrid_rrf.json"
    try:
        rrf = json.loads(rrf_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return []
    return [cid.upper() for cid in rrf.get("fused_ids", [])]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(ranked: List[str], gt_ids: Set[str], ks: List[int],
                    use_rollup: bool = False) -> Dict:
    """Compute R@k, P@k, and MRR for a ranked list against ground truth."""
    if use_rollup:
        gt_ids = rollup(gt_ids)
        ranked_rolled = []
        seen = set()
        for cid in ranked:
            expanded = rollup({cid})
            for e in expanded:
                if e not in seen:
                    ranked_rolled.append(e)
                    seen.add(e)
        ranked = ranked_rolled

    results = {}
    for k in ks:
        top = set(ranked[:k])
        matched = top & gt_ids
        r = len(matched) / len(gt_ids) if gt_ids else 0
        p = len(matched) / len(top) if top else 0
        results[k] = {"recall": r, "precision": p, "n_matched": len(matched), "n_gt": len(gt_ids)}

    # MRR: for each GT ID, find its rank in the list (1-indexed)
    rr_sum = 0.0
    n_found = 0
    for gt_id in gt_ids:
        try:
            rank = ranked.index(gt_id) + 1
            rr_sum += 1.0 / rank
            n_found += 1
        except ValueError:
            pass  # GT ID not in ranked list — contributes 0
    mrr = rr_sum / len(gt_ids) if gt_ids else 0
    results["mrr"] = mrr

    return results


def extract_bm25_pool(doc_dir: Path, per_entity_k: int = 10) -> Set[str]:
    """BM25: per-entity top-k union (pool)."""
    ts_file = doc_dir / "token_search.json"
    try:
        ts = json.loads(ts_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return set()
    pool = set()
    for entry in ts.get("per_entity", []):
        for i, hit in enumerate(entry.get("hits", [])):
            if i >= per_entity_k:
                break
            cid = (hit.get("clean_id") or "").strip().upper()
            if cid:
                pool.add(cid)
    return pool


def extract_sentence_vec_pool(doc_dir: Path, per_sentence_k: int = 10) -> Set[str]:
    """Sentence vector: per-sentence top-k union (pool)."""
    ss_file = doc_dir / "sentence_search.json"
    try:
        ss = json.loads(ss_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return set()
    pool = set()
    for sent_data in ss.get("per_sentence", []):
        hits = sent_data.get("hits") or sent_data.get("top_k") or []
        for i, hit in enumerate(hits):
            if i >= per_sentence_k:
                break
            cid = (hit.get("clean_id") or hit.get("id") or "").strip().upper()
            if cid:
                pool.add(cid)
    return pool


def extract_sentence_rrf_pool(doc_dir: Path, per_sentence_k: int = 10) -> Set[str]:
    """Sentence-level RRF: per-sentence top-k union (pool)."""
    sr_file = doc_dir / "sentence_retrieval.json"
    try:
        sr = json.loads(sr_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return set()
    pool = set()
    for sent_data in sr.get("per_sentence", []):
        hits = sent_data.get("fused_top5") or []
        for i, hit in enumerate(hits):
            if i >= per_sentence_k:
                break
            cid = (hit.get("clean_id") or hit.get("id") or "").strip().upper()
            if cid:
                pool.add(cid)
    return pool


def extract_hybrid_rrf_pool(doc_dir: Path, doc_level_k: int = 100) -> Set[str]:
    """Hybrid RRF: top-k from doc-level fused list (pool)."""
    rrf_file = doc_dir / "hybrid_rrf.json"
    try:
        rrf = json.loads(rrf_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return set()
    pool = set()
    fused = rrf.get("fused_ids", [])
    for i, cid in enumerate(fused):
        if i >= doc_level_k:
            break
        cid = cid.strip().upper()
        if cid:
            pool.add(cid)
    return pool


def compute_pool_recall(pool: Set[str], gt_ids: Set[str], use_rollup: bool = False) -> float:
    """Recall of a pool (unordered set) against GT."""
    if use_rollup:
        pool = rollup(pool)
        gt_ids = rollup(gt_ids)
    return len(pool & gt_ids) / len(gt_ids) if gt_ids else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified graph alignment evaluation")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    parser.add_argument("--eval-dir", default=None, help="Override experiment eval directory")
    args = parser.parse_args()

    global EVAL_DIR
    if args.eval_dir:
        EVAL_DIR = Path(args.eval_dir)

    mappings = load_mappings()

    # Collect per-doc results
    methods = {
        "Entity-neighbors vector": ("similarity", extract_entity_vec_ranked),
        "BM25 (entity-level)": ("token_search", extract_bm25_ranked),
        "Sentence vector": ("sentence_search", extract_sentence_vec_ranked),
        "Sentence RRF (max)": ("sentence_retrieval", lambda d: extract_sentence_rrf_ranked(d, "max")),
        "Hybrid RRF": ("hybrid", extract_hybrid_rrf_ranked),
    }

    # Find common doc set
    all_doc_dirs = {}
    for doc_dir in sorted((EVAL_DIR / "similarity").iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        mapping = find_mapping(doc_id, mappings)
        if not mapping:
            continue
        gt = load_gt(mapping["identifier"], mapping["group"])
        all_gt = gt["techniques"] | gt["tactics"] | gt["software"]
        if not all_gt:
            continue
        all_doc_dirs[doc_id] = {"gt": gt, "all_gt": all_gt, "group": mapping["group"].lower()}

    print(f"Evaluating {len(all_doc_dirs)} documents\n")

    # Compute metrics per method
    method_results = {}
    for method_name, (subdir, extract_fn) in methods.items():
        recalls = {k: [] for k in KS}
        precisions = {k: [] for k in KS}
        recalls_rollup = {k: [] for k in KS}
        mrrs = []
        mrrs_rollup = []
        cat_recalls = {cat: {k: [] for k in KS} for cat in ["techniques", "tactics", "software"]}
        cat_recalls_rollup = {cat: {k: [] for k in KS} for cat in ["techniques", "tactics", "software"]}
        n_docs = 0

        for doc_id, doc_info in all_doc_dirs.items():
            doc_dir = EVAL_DIR / subdir / doc_id
            ranked = extract_fn(doc_dir)
            if not ranked:
                continue

            n_docs += 1
            gt_ids = doc_info["all_gt"]

            # Raw metrics
            m = compute_metrics(ranked, gt_ids, KS, use_rollup=False)
            for k in KS:
                recalls[k].append(m[k]["recall"])
                precisions[k].append(m[k]["precision"])
            mrrs.append(m["mrr"])

            # Rollup metrics
            m_r = compute_metrics(ranked, gt_ids, KS, use_rollup=True)
            for k in KS:
                recalls_rollup[k].append(m_r[k]["recall"])
            mrrs_rollup.append(m_r["mrr"])

            # Per-category
            for cat in ["techniques", "tactics", "software"]:
                cat_gt = doc_info["gt"][cat]
                if not cat_gt:
                    continue
                mc = compute_metrics(ranked, cat_gt, KS, use_rollup=False)
                for k in KS:
                    cat_recalls[cat][k].append(mc[k]["recall"])
                mc_r = compute_metrics(ranked, cat_gt, KS, use_rollup=True)
                for k in KS:
                    cat_recalls_rollup[cat][k].append(mc_r[k]["recall"])

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        method_results[method_name] = {
            "n_docs": n_docs,
            "recall": {k: round(avg(recalls[k]), 4) for k in KS},
            "precision": {k: round(avg(precisions[k]), 4) for k in KS},
            "mrr": round(avg(mrrs), 4),
            "recall_rollup": {k: round(avg(recalls_rollup[k]), 4) for k in KS},
            "mrr_rollup": round(avg(mrrs_rollup), 4),
            "category_recall": {
                cat: {k: round(avg(cat_recalls[cat][k]), 4) for k in KS}
                for cat in ["techniques", "tactics", "software"]
            },
            "category_recall_rollup": {
                cat: {k: round(avg(cat_recalls_rollup[cat][k]), 4) for k in KS}
                for cat in ["techniques", "tactics", "software"]
            },
        }

    # Compute pool recall for ALL methods at k=5, 10, 20 (rollup only)
    POOL_KS = [5, 10, 20]
    pool_extractors = {
        "Entity-neighbors vector": ("similarity", lambda d, k: extract_entity_vec_pool(d, per_entity_k=k)),
        "BM25 (entity-level)": ("token_search", lambda d, k: extract_bm25_pool(d, per_entity_k=k)),
        "Sentence vector": ("sentence_search", lambda d, k: extract_sentence_vec_pool(d, per_sentence_k=k)),
        "Sentence RRF (max)": ("sentence_retrieval", lambda d, k: extract_sentence_rrf_pool(d, per_sentence_k=k)),
        "Hybrid RRF": ("hybrid", lambda d, k: extract_hybrid_rrf_pool(d, doc_level_k=k * 10)),
    }

    all_pool_results = {}
    for method_name, (subdir, extractor) in pool_extractors.items():
        pool_at_k = {}
        for pk in POOL_KS:
            recalls_rollup = []
            for doc_id, doc_info in all_doc_dirs.items():
                pool = extractor(EVAL_DIR / subdir / doc_id, pk)
                if pool:
                    recalls_rollup.append(compute_pool_recall(pool, doc_info["all_gt"], True))
            pool_at_k[pk] = recalls_rollup
        all_pool_results[method_name] = pool_at_k

    # Keep backward compat aliases
    pool_recalls = []
    pool_recalls_rollup = all_pool_results["Entity-neighbors vector"].get(10, [])

    # Per-group for entity-vec
    group_results = defaultdict(list)
    group_results_rollup = defaultdict(list)
    for doc_id, doc_info in all_doc_dirs.items():
        ranked = extract_entity_vec_ranked(EVAL_DIR / "similarity" / doc_id)
        if not ranked:
            continue
        m = compute_metrics(ranked, doc_info["all_gt"], [10], use_rollup=False)
        m_r = compute_metrics(ranked, doc_info["all_gt"], [10], use_rollup=True)
        group_results[doc_info["group"]].append(m[10]["recall"])
        group_results_rollup[doc_info["group"]].append(m_r[10]["recall"])

    # -----------------------------------------------------------------------
    # Print Tables
    # -----------------------------------------------------------------------

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    print("=" * 90)
    print("TABLE A: Retrieval Method Comparison")
    print("=" * 90)
    print(f"{'Method':<30s} {'R@1':>6s} {'R@3':>6s} {'R@5':>6s} {'R@10':>6s} {'R@10+':>6s} {'P@10':>6s} {'MRR':>6s} {'MRR+':>6s} {'Docs':>5s}")
    print("-" * 90)
    for name, mr in method_results.items():
        print(f"{name:<30s} "
              f"{mr['recall'][1]:6.3f} {mr['recall'][3]:6.3f} {mr['recall'][5]:6.3f} {mr['recall'][10]:6.3f} "
              f"{mr['recall_rollup'][10]:6.3f} {mr['precision'][10]:6.3f} "
              f"{mr['mrr']:6.3f} {mr['mrr_rollup']:6.3f} {mr['n_docs']:5d}")
    # Pool recall lines for all methods at k=5, 10, 20 (rollup only)
    print(f"\n{'--- Pool R+ (rollup) ---':<30s} {'@5':>6s} {'@10':>6s} {'@20':>6s} {'Docs':>5s}")
    print("-" * 55)
    for method_name, pool_at_k in all_pool_results.items():
        short_name = method_name.replace("(entity-level)", "").replace("(max)", "").strip()
        vals = []
        n = 0
        for pk in POOL_KS:
            r = pool_at_k.get(pk, [])
            vals.append(avg(r))
            n = max(n, len(r))
        print(f"{short_name + ' pool':<30s} {vals[0]:6.3f} {vals[1]:6.3f} {vals[2]:6.3f} {n:5d}")

    print()
    print("R@10+ = Recall@10 with sub-technique rollup")
    print("MRR+ = MRR with sub-technique rollup")

    print()
    print("=" * 80)
    print("TABLE B: Per-Category Recall@10 by Method")
    print("=" * 80)
    print(f"{'Method':<30s} {'Tech':>6s} {'Tech+':>6s} {'Tact':>6s} {'Tact+':>6s} {'Soft':>6s} {'Soft+':>6s}")
    print("-" * 80)
    for name, mr in method_results.items():
        cr = mr["category_recall"]
        crr = mr["category_recall_rollup"]
        print(f"{name:<30s} "
              f"{cr['techniques'][10]:6.3f} {crr['techniques'][10]:6.3f} "
              f"{cr['tactics'][10]:6.3f} {crr['tactics'][10]:6.3f} "
              f"{cr['software'][10]:6.3f} {crr['software'][10]:6.3f}")

    print()
    print("=" * 60)
    print("TABLE C: Per-Group Recall@10 (Entity-neighbors vector)")
    print("=" * 60)
    print(f"{'Group':<20s} {'Docs':>5s} {'R@10':>7s} {'R@10+':>7s}")
    print("-" * 60)
    for g in sorted(group_results.keys(), key=lambda x: -avg(group_results_rollup[x])):
        n = len(group_results[g])
        print(f"{g:<20s} {n:5d} {avg(group_results[g]):7.3f} {avg(group_results_rollup[g]):7.3f}")

    # Save
    if args.output:
        out = {
            "methods": method_results,
            "entity_vec_pool": {
                "recall": round(avg(pool_recalls_rollup), 4) if pool_recalls_rollup else 0,
                "recall_rollup": round(avg(pool_recalls_rollup), 4) if pool_recalls_rollup else 0,
            },
            "method_pools": {
                name: {
                    str(pk): round(avg(pool_at_k.get(pk, [])), 4)
                    for pk in POOL_KS
                }
                for name, pool_at_k in all_pool_results.items()
            },
            "per_group": {
                g: {"n": len(v), "recall": round(avg(v), 4), "recall_rollup": round(avg(group_results_rollup[g]), 4)}
                for g, v in group_results.items()
            },
        }
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
