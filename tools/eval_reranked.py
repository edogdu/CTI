"""Evaluate pool-based retrieval at document-level cutoffs.

For each method, aggregates per-entity/per-sentence top-10 results into a
document-level pool, then evaluates at document-level k = 10, 20, 50, 100.

Two ranking strategies:
  - max_score: rank by best single score per ATT&CK ID
  - freq_weighted: rank by max_score * (1 + log(count))

Usage:
    python tools/eval_reranked.py --eval-dir experiments/ctihal-pipeline/eval
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "src"))

ANN_DIR = _repo / "datasets" / "CTI-HAL" / "data"
MAPPINGS_CSV = _repo / "config" / "CTI_HAL_mappings.csv"

DOC_KS = [10, 20, 50, 100]
PER_ENTITY_K = 10


def load_mappings():
    mappings = {}
    with open(MAPPINGS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            doc = (row.get("Document") or "").strip()
            ident = (row.get("Identifier") or "").strip()
            group = (row.get("Group") or "").strip()
            if doc and ident:
                mappings[doc.lower().replace(".pdf", "")] = {
                    "identifier": ident,
                    "group": group,
                }
    return mappings


def load_gt(ident, group):
    techs, tacts, soft = set(), set(), set()
    p = ANN_DIR / group.lower() / f"{ident}.json"
    if not p.exists():
        return None
    for ann in json.loads(p.read_text(encoding="utf-8")):
        t = ann.get("technique")
        if isinstance(t, str) and t.strip():
            techs.add(t.strip().upper())
        md = ann.get("metadata") or {}
        for tid in md.get("tactic") or []:
            if isinstance(tid, str) and tid.strip():
                tacts.add(tid.strip().upper())
        sub = md.get("sub_technique")
        if isinstance(sub, str) and sub.strip():
            techs.add(sub.strip().upper())
        for sid in md.get("tool") or []:
            if isinstance(sid, str) and sid.strip():
                soft.add(sid.strip().upper())
    all_gt = techs | tacts | soft
    return all_gt if all_gt else None


def rollup(ids):
    expanded = set(ids)
    for i in ids:
        if "." in i and i.startswith("T"):
            expanded.add(i.split(".")[0])
    return expanded


def find_mapping(doc_id, mappings):
    parts = doc_id.split("_", 1)
    pdf_stem = parts[1] if len(parts) > 1 else doc_id
    mapping = mappings.get(pdf_stem.lower())
    if mapping:
        return mapping
    norm = re.sub(r'[^a-z0-9]', '', pdf_stem.lower())
    for stem, m in mappings.items():
        if re.sub(r'[^a-z0-9]', '', stem.lower()) == norm:
            return m
    return None


def rank_by_max(scores: Dict[str, float]) -> List[str]:
    """Rank by max single score."""
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]


def rank_by_freq(scores: Dict[str, float], counts: Dict[str, int]) -> List[str]:
    """Rank by max_score * (1 + log(count))."""
    combined = {
        cid: scores[cid] * (1 + math.log(max(counts.get(cid, 1), 1)))
        for cid in scores
    }
    return [cid for cid, _ in sorted(combined.items(), key=lambda x: -x[1])]


def extract_scores(doc_dir: Path, method: str) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Extract max_score and count per ATT&CK ID from per-entity/per-sentence results.

    Returns (max_scores, counts) where:
      max_scores[cid] = best single score for this ID
      counts[cid] = number of distinct entities/sentences that retrieved this ID
    """
    max_scores: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    if method == "entity_vec":
        pn_file = doc_dir / "similarity_per_node.json"
        try:
            pn = json.loads(pn_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError):
            return {}, {}
        for nid, entry in pn.items():
            seen = set()
            for i, hit in enumerate(entry.get("top_k") or []):
                if i >= PER_ENTITY_K:
                    break
                target = hit.get("target") or {}
                cid = (target.get("clean_id") or "").strip().upper()
                if not cid:
                    continue
                score = hit.get("scores", {}).get("cosine", 0)
                max_scores[cid] = max(max_scores.get(cid, 0), score)
                if cid not in seen:
                    counts[cid] = counts.get(cid, 0) + 1
                    seen.add(cid)

    elif method == "bm25":
        ts_file = doc_dir / "token_search.json"
        try:
            ts = json.loads(ts_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError):
            return {}, {}
        for entry in ts.get("per_entity", []):
            seen = set()
            for i, hit in enumerate(entry.get("hits", [])):
                if i >= PER_ENTITY_K:
                    break
                cid = (hit.get("clean_id") or "").strip().upper()
                score = hit.get("score", 0)
                if not cid:
                    continue
                max_scores[cid] = max(max_scores.get(cid, 0), score)
                if cid not in seen:
                    counts[cid] = counts.get(cid, 0) + 1
                    seen.add(cid)

    elif method == "sentence_vec":
        ss_file = doc_dir / "sentence_search.json"
        try:
            ss = json.loads(ss_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError):
            return {}, {}
        for sent_data in ss.get("per_sentence", []):
            seen = set()
            hits = sent_data.get("hits") or sent_data.get("top_k") or []
            for i, hit in enumerate(hits):
                if i >= PER_ENTITY_K:
                    break
                cid = (hit.get("clean_id") or hit.get("id") or "").strip().upper()
                score = hit.get("cosine", hit.get("score", 0))
                if not cid:
                    continue
                max_scores[cid] = max(max_scores.get(cid, 0), score)
                if cid not in seen:
                    counts[cid] = counts.get(cid, 0) + 1
                    seen.add(cid)

    elif method == "sentence_rrf":
        sr_file = doc_dir / "sentence_retrieval.json"
        try:
            sr = json.loads(sr_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError):
            return {}, {}
        for sent_data in sr.get("per_sentence", []):
            seen = set()
            rrf_scores = sent_data.get("rrf_scores_max") or sent_data.get("rrf_scores")
            if rrf_scores and isinstance(rrf_scores, dict):
                for cid, score in list(rrf_scores.items())[:PER_ENTITY_K]:
                    cid = cid.strip().upper()
                    if not cid:
                        continue
                    max_scores[cid] = max(max_scores.get(cid, 0), score)
                    if cid not in seen:
                        counts[cid] = counts.get(cid, 0) + 1
                        seen.add(cid)
            else:
                for i, hit in enumerate(sent_data.get("fused_top5") or []):
                    if i >= PER_ENTITY_K:
                        break
                    cid = (hit.get("clean_id") or hit.get("id") or "").strip().upper()
                    score = hit.get("rrf_score", hit.get("score", 0))
                    if not cid:
                        continue
                    max_scores[cid] = max(max_scores.get(cid, 0), score)
                    if cid not in seen:
                        counts[cid] = counts.get(cid, 0) + 1
                        seen.add(cid)

    elif method == "hybrid_rrf":
        rrf_file = doc_dir / "hybrid_rrf.json"
        try:
            rrf = json.loads(rrf_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError):
            return {}, {}
        # Hybrid is already doc-level; treat each entry as count=1
        for i, entry in enumerate(rrf.get("fused_ranked", [])):
            if isinstance(entry, dict):
                cid = (entry.get("clean_id") or entry.get("id") or "").strip().upper()
                score = entry.get("rrf_score", entry.get("score", 0))
            elif isinstance(entry, str):
                cid = entry.strip().upper()
                score = 1.0 / (i + 1)  # Use rank-based score
            else:
                continue
            if not cid:
                continue
            max_scores[cid] = max(max_scores.get(cid, 0), score)
            counts[cid] = counts.get(cid, 0) + 1

    return max_scores, counts


def compute_metrics_rollup(ranked: List[str], gt: Set[str], ks: List[int]):
    """Compute R@k, P@k with rollup, and MRR with rollup."""
    # Expand ranked with rollup (insert parent after child)
    ranked_expanded = []
    seen = set()
    for cid in ranked:
        if cid not in seen:
            ranked_expanded.append(cid)
            seen.add(cid)
        if "." in cid and cid.startswith("T"):
            parent = cid.split(".")[0]
            if parent not in seen:
                ranked_expanded.append(parent)
                seen.add(parent)

    gt_r = rollup(gt)

    results = {}
    for k in ks:
        top_k = set(ranked_expanded[:k])
        recall = len(top_k & gt_r) / len(gt_r) if gt_r else 0
        precision = len(top_k & gt_r) / k if k > 0 else 0
        results[k] = {"recall": recall, "precision": precision}

    # MRR
    rr_sum = 0
    for gt_id in gt_r:
        try:
            rank = ranked_expanded.index(gt_id) + 1
            rr_sum += 1.0 / rank
        except ValueError:
            pass
    results["mrr"] = rr_sum / len(gt_r) if gt_r else 0

    # Pool recall (entire ranked list)
    pool = set(ranked_expanded)
    results["pool"] = len(pool & gt_r) / len(gt_r) if gt_r else 0

    return results


def main():
    parser = argparse.ArgumentParser(description="Pool-based retrieval at document-level cutoffs")
    parser.add_argument("--eval-dir", required=True, help="Experiment eval directory")
    parser.add_argument("--output", default=None, help="Save results JSON")
    parser.add_argument("--per-doc-output", default=None,
                        help="Save per-document metrics + ranked IDs JSON (additive; --output is unchanged)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    mappings = load_mappings()

    methods = {
        "Entity-neighbors vector": ("similarity", "entity_vec"),
        "BM25 (entity-level)": ("token_search", "bm25"),
        "Sentence vector": ("sentence_search", "sentence_vec"),
        "Sentence RRF (max)": ("sentence_retrieval", "sentence_rrf"),
        "Hybrid RRF": ("hybrid", "hybrid_rrf"),
    }

    # Find all doc IDs from similarity dir
    sim_dir = eval_dir / "similarity"
    if not sim_dir.exists():
        print(f"Error: {sim_dir} not found")
        return
    doc_ids = sorted(d.name for d in sim_dir.iterdir() if d.is_dir())

    # Match to ground truth
    doc_gt = {}
    for doc_id in doc_ids:
        mapping = find_mapping(doc_id, mappings)
        if not mapping:
            continue
        gt = load_gt(mapping["identifier"], mapping["group"])
        if gt:
            doc_gt[doc_id] = gt

    print(f"Evaluating {len(doc_gt)} documents")
    print(f"Per-entity/sentence k={PER_ENTITY_K}, doc-level cutoffs: {DOC_KS}")

    # Header
    k_headers = " ".join(f"R@{k:>3d}" for k in DOC_KS)
    print(f"\n{'Method':<28s} {'Ranking':<12s} {k_headers}  {'MRR':>6s} {'Pool':>6s} {'Docs':>4s}")
    print("-" * 95)

    all_results = {}
    all_per_doc = {}

    for method_name, (subdir, method_key) in methods.items():
        # Collect per-doc results for both ranking strategies
        max_metrics = defaultdict(list)
        freq_metrics = defaultdict(list)
        max_mrrs = []
        freq_mrrs = []
        max_pools = []
        freq_pools = []
        n = 0
        per_doc = {"max_score": {}, "freq_weighted": {}}

        for doc_id, gt in doc_gt.items():
            doc_dir = eval_dir / subdir / doc_id
            scores, counts = extract_scores(doc_dir, method_key)
            if not scores:
                continue

            # Rank by max score
            ranked_max = rank_by_max(scores)
            m_max = compute_metrics_rollup(ranked_max, gt, DOC_KS)

            # Rank by freq-weighted
            ranked_freq = rank_by_freq(scores, counts)
            m_freq = compute_metrics_rollup(ranked_freq, gt, DOC_KS)

            for k in DOC_KS:
                max_metrics[k].append(m_max[k]["recall"])
                freq_metrics[k].append(m_freq[k]["recall"])
            max_mrrs.append(m_max["mrr"])
            freq_mrrs.append(m_freq["mrr"])
            max_pools.append(m_max["pool"])
            freq_pools.append(m_freq["pool"])
            n += 1

            if args.per_doc_output:
                per_doc["max_score"][doc_id] = {
                    "recall": {str(k): m_max[k]["recall"] for k in DOC_KS},
                    "mrr": m_max["mrr"], "pool": m_max["pool"],
                    "n_gt": len(gt), "ranked_ids": ranked_max,
                }
                per_doc["freq_weighted"][doc_id] = {
                    "recall": {str(k): m_freq[k]["recall"] for k in DOC_KS},
                    "mrr": m_freq["mrr"], "pool": m_freq["pool"],
                    "n_gt": len(gt), "ranked_ids": ranked_freq,
                }

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        # Print max-score ranking
        vals_max = " ".join(f"{avg(max_metrics[k]):6.3f}" for k in DOC_KS)
        print(f"{method_name:<28s} {'max_score':<12s} {vals_max}  {avg(max_mrrs):6.3f} {avg(max_pools):6.3f} {n:4d}")

        # Print freq-weighted ranking
        vals_freq = " ".join(f"{avg(freq_metrics[k]):6.3f}" for k in DOC_KS)
        print(f"{'':<28s} {'freq_wt':<12s} {vals_freq}  {avg(freq_mrrs):6.3f} {avg(freq_pools):6.3f} {n:4d}")

        all_results[method_name] = {
            "max_score": {
                "recall": {str(k): round(avg(max_metrics[k]), 4) for k in DOC_KS},
                "mrr": round(avg(max_mrrs), 4),
                "pool": round(avg(max_pools), 4),
                "n_docs": n,
            },
            "freq_weighted": {
                "recall": {str(k): round(avg(freq_metrics[k]), 4) for k in DOC_KS},
                "mrr": round(avg(freq_mrrs), 4),
                "pool": round(avg(freq_pools), 4),
                "n_docs": n,
            },
        }
        all_per_doc[method_name] = per_doc
        print()

    if args.per_doc_output:
        p = Path(args.per_doc_output)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "eval_dir": str(eval_dir),
            "doc_ks": DOC_KS,
            "ground_truth": {d: sorted(g) for d, g in doc_gt.items()},
            "methods": all_per_doc,
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved per-doc: {p}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
