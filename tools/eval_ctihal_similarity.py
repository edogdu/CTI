"""Evaluate CTI-HAL pipeline similarity results against ground truth annotations.

Reads similarity results from experiments/ctihal-pipeline/eval/similarity/
and ground truth from datasets/CTI-HAL/data/ via the mappings CSV.

Computes:
- Hit@k, Precision@k, Recall@k at k=1,3,5,10
- Per-category breakdown (techniques, tactics, software)
- Per-group and overall summaries
- Cosine score distributions for TP vs FP (threshold analysis)

Usage:
    python tools/eval_ctihal_similarity.py
    python tools/eval_ctihal_similarity.py --ks 1,3,5,10,20
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "src"))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXPERIMENT_DIR = _repo / "experiments" / "ctihal-pipeline"
SIMILARITY_DIR = EXPERIMENT_DIR / "eval" / "similarity"
ANALYSIS_DIR = EXPERIMENT_DIR / "eval" / "analysis"
ANNOTATIONS_DIR = _repo / "datasets" / "CTI-HAL" / "data"
MAPPINGS_CSV = _repo / "config" / "CTI_HAL_mappings.csv"

ATTACK_LABELS = {"UcoexMITREATTACK", "UcoexTACTICS", "UcoexSOFTWARE"}


def _normalize_stem(s: str) -> str:
    """Strip all non-alphanumeric chars for fuzzy doc_id matching."""
    return re.sub(r'[^a-z0-9]', '', s.lower())


def _find_mapping(doc_id: str, mappings: dict):
    """Find mapping for a doc_id, handling sanitized filenames."""
    parts = doc_id.split("_", 1)
    pdf_stem = parts[1] if len(parts) > 1 else doc_id
    # Exact
    m = mappings.get(pdf_stem.lower()) or mappings.get(doc_id.lower())
    if m:
        return m
    # Normalized (alphanumeric only)
    norm = _normalize_stem(pdf_stem)
    for stem, m in mappings.items():
        if _normalize_stem(stem) == norm:
            return m
    # Substring fallback
    for stem, m in mappings.items():
        if pdf_stem.lower() in stem or stem in pdf_stem.lower():
            return m
    return None

ATTACK_ID_PATTERNS = {
    "technique": re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE),
    "tactic": re.compile(r"\bTA0\d{3}\b", re.IGNORECASE),
    "software": re.compile(r"\bS\d{4}\b", re.IGNORECASE),
}

# Sub-technique regex: T1234.567 -> T1234
_SUB_TECHNIQUE_RE = re.compile(r"^(T\d{4})\.\d{3}$", re.IGNORECASE)


def rollup_ids(ids: Set[str]) -> Set[str]:
    """Expand a set of ATT&CK IDs with sub-technique rollup.

    For each sub-technique (T1234.567), also adds the parent technique (T1234).
    This gives credit when we find a specific sub-technique but the ground truth
    only lists the parent, or vice versa.

    Returns a new set containing both the original IDs and any rolled-up parents.
    """
    expanded = set(ids)
    for aid in ids:
        m = _SUB_TECHNIQUE_RE.match(aid)
        if m:
            expanded.add(m.group(1).upper())
    return expanded


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_mappings(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Map document filename (lowercased) -> {identifier, group}."""
    mapping = {}
    if not csv_path.exists():
        return mapping
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = (row.get("Document") or "").strip()
            ident = (row.get("Identifier") or "").strip()
            group = (row.get("Group") or "").strip()
            if doc and ident:
                stem = doc.lower().replace(".pdf", "")
                mapping[stem] = {"identifier": ident, "group": group}
    return mapping


def load_ground_truth(ann_dir: Path, identifier: str, group: str) -> Dict[str, Set[str]]:
    """Load ATT&CK IDs from annotation JSON(s) for a given report identifier."""
    techs, tacts, soft = set(), set(), set()

    # Find annotation file — directly in group folder
    candidates = []
    group_dir = ann_dir / group.lower()
    p = group_dir / f"{identifier}.json"
    if p.exists():
        candidates.append(p)

    for json_path in candidates:
        try:
            anns = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for ann in anns:
            t = ann.get("technique")
            if isinstance(t, str) and t.strip():
                techs.add(t.strip().upper())
            md = ann.get("metadata") or {}
            for tid in (md.get("tactic") or []):
                if isinstance(tid, str) and tid.strip():
                    tacts.add(tid.strip().upper())
            sub = md.get("sub_technique")
            if isinstance(sub, str) and sub.strip():
                techs.add(sub.strip().upper())
            for sid in (md.get("tool") or []):
                if isinstance(sid, str) and sid.strip():
                    soft.add(sid.strip().upper())

    return {"techniques": techs, "tactics": tacts, "software": soft}


# ---------------------------------------------------------------------------
# Similarity result parsing
# ---------------------------------------------------------------------------

def parse_similarity_results(sim_dir: Path, max_k: int = 20) -> List[Dict]:
    """Parse similarity_per_node.json into a list of query entries.

    Each entry has: source (name, type), top_k list of (clean_id, cosine, labels).
    """
    per_node = sim_dir / "similarity_per_node.json"
    try:
        obj = json.loads(per_node.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        # Fall back to leaderboard if per_node is missing/unreadable
        leaderboard = sim_dir / "similarity_leaderboard.json"
        try:
            obj = json.loads(leaderboard.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                # Leaderboard is a flat list, wrap into per-node format
                return [{"source_name": e.get("source", {}).get("name", ""),
                         "source_type": e.get("source", {}).get("type", ""),
                         "hits": [{"clean_id": (e.get("target", {}).get("clean_id") or "").strip().upper(),
                                   "cosine": (e.get("scores", {}).get("cosine", 0.0))}]}
                        for e in obj if isinstance(e, dict)]
        except (FileNotFoundError, OSError):
            return []
        return []
    entries = []

    def process(entry: Any):
        if not isinstance(entry, dict) or "top_k" not in entry:
            return
        source = entry.get("source", {})
        hits = []
        for h in (entry.get("top_k") or [])[:max_k]:
            if not isinstance(h, dict):
                continue
            target = h.get("target") or {}
            labels = set(target.get("labels") or [])
            if not (ATTACK_LABELS & labels):
                continue
            cid = (target.get("clean_id") or "").strip().upper()
            cosine = (h.get("scores") or {}).get("cosine", 0.0)
            if cid:
                hits.append({"clean_id": cid, "cosine": cosine, "labels": labels,
                             "name": target.get("name", ""), "description": target.get("description", "")})
        if hits:
            entries.append({"source": source, "hits": hits})

    if isinstance(obj, dict):
        for v in obj.values():
            process(v)
    elif isinstance(obj, list):
        for v in obj:
            process(v)

    return entries


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

@dataclass
class DocMetrics:
    doc_id: str
    group: str
    gt: Dict[str, List[str]] = field(default_factory=dict)
    n_queries: int = 0
    # Per-k metrics
    metrics_at_k: Dict[int, Dict[str, float]] = field(default_factory=dict)
    # Per-category per-k
    category_metrics_at_k: Dict[str, Dict[int, Dict[str, float]]] = field(default_factory=dict)
    # Rollup metrics (sub-technique -> parent)
    rollup_metrics_at_k: Dict[int, Dict[str, float]] = field(default_factory=dict)
    rollup_category_metrics_at_k: Dict[str, Dict[int, Dict[str, float]]] = field(default_factory=dict)
    # Cosine scores for threshold analysis
    tp_cosines: List[float] = field(default_factory=list)
    fp_cosines: List[float] = field(default_factory=list)


def compute_doc_metrics(
    doc_id: str,
    group: str,
    gt: Dict[str, Set[str]],
    entries: List[Dict],
    ks: Tuple[int, ...],
) -> DocMetrics:
    all_gt = gt["techniques"] | gt["tactics"] | gt["software"]
    max_k = max(ks)

    result = DocMetrics(
        doc_id=doc_id,
        group=group,
        gt={cat: sorted(ids) for cat, ids in gt.items()},
        n_queries=len(entries),
    )

    # Collect IDs at each k (union across queries)
    ids_at_k: Dict[int, Set[str]] = {k: set() for k in ks}
    # Per-query hit tracking
    query_hits_at_k: Dict[int, int] = {k: 0 for k in ks}
    tp_at_k: Dict[int, int] = {k: 0 for k in ks}
    fp_at_k: Dict[int, int] = {k: 0 for k in ks}

    for entry in entries:
        hits = entry["hits"][:max_k]
        for k in ks:
            cut = hits[:k]
            cut_ids = {h["clean_id"] for h in cut}
            ids_at_k[k].update(cut_ids)
            tp = len(cut_ids & all_gt)
            fp = len(cut_ids - all_gt)
            if tp > 0:
                query_hits_at_k[k] += 1
            tp_at_k[k] += tp
            fp_at_k[k] += fp

        # Cosine scores for threshold analysis (use all hits up to max_k)
        for h in hits:
            if h["clean_id"] in all_gt:
                result.tp_cosines.append(h["cosine"])
            else:
                result.fp_cosines.append(h["cosine"])

    # Compute overall metrics at each k
    n_q = len(entries) or 1
    for k in ks:
        covered = ids_at_k[k] & all_gt
        result.metrics_at_k[k] = {
            "hit_rate": query_hits_at_k[k] / n_q,
            "precision": tp_at_k[k] / (tp_at_k[k] + fp_at_k[k]) if (tp_at_k[k] + fp_at_k[k]) else 0.0,
            "recall": len(covered) / len(all_gt) if all_gt else 0.0,
            "n_matched": len(covered),
            "n_gt": len(all_gt),
            "matched_ids": sorted(covered),
            "missed_ids": sorted(all_gt - ids_at_k[k]),
        }

    # Per-category metrics at each k
    for cat, cat_ids in [("techniques", gt["techniques"]), ("tactics", gt["tactics"]), ("software", gt["software"])]:
        cat_metrics: Dict[int, Dict[str, float]] = {}
        for k in ks:
            covered = ids_at_k[k] & cat_ids
            cat_metrics[k] = {
                "recall": len(covered) / len(cat_ids) if cat_ids else 0.0,
                "n_matched": len(covered),
                "n_gt": len(cat_ids),
            }
        result.category_metrics_at_k[cat] = cat_metrics

    # --- Rollup metrics (sub-technique -> parent technique) ---
    # Expand both predicted and GT IDs, then recompute
    all_gt_rolled = rollup_ids(all_gt)
    result.rollup_metrics_at_k = {}
    result.rollup_category_metrics_at_k = {}
    for k in ks:
        predicted_rolled = rollup_ids(ids_at_k[k])
        covered = predicted_rolled & all_gt_rolled
        result.rollup_metrics_at_k[k] = {
            "recall": len(covered) / len(all_gt_rolled) if all_gt_rolled else 0.0,
            "n_matched": len(covered),
            "n_gt": len(all_gt_rolled),
            "matched_ids": sorted(covered),
        }

    # Per-category rollup
    for cat, cat_ids in [("techniques", gt["techniques"]), ("tactics", gt["tactics"]), ("software", gt["software"])]:
        cat_rolled = rollup_ids(cat_ids)
        cat_metrics_r: Dict[int, Dict[str, float]] = {}
        for k in ks:
            predicted_rolled = rollup_ids(ids_at_k[k])
            covered = predicted_rolled & cat_rolled
            cat_metrics_r[k] = {
                "recall": len(covered) / len(cat_rolled) if cat_rolled else 0.0,
                "n_matched": len(covered),
                "n_gt": len(cat_rolled),
            }
        result.rollup_category_metrics_at_k[cat] = cat_metrics_r

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate CTI-HAL similarity results")
    parser.add_argument("--ks", default="1,3,5,10", help="Comma-separated k values")
    parser.add_argument("--sim-dir", default=str(SIMILARITY_DIR))
    parser.add_argument("--output", default=str(ANALYSIS_DIR))
    args = parser.parse_args()

    ks = tuple(sorted(int(x) for x in args.ks.split(",")))
    sim_dir = Path(args.sim_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    mappings = load_mappings(MAPPINGS_CSV)
    print(f"Loaded {len(mappings)} document mappings")

    all_metrics: List[DocMetrics] = []
    all_tp_cosines: List[float] = []
    all_fp_cosines: List[float] = []

    for doc_dir in sorted(sim_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        # Parse group and pdf stem from doc_id (e.g., "apt29_AnalysisOfCyberattackOnUS")
        parts = doc_id.split("_", 1)
        group = parts[0] if len(parts) > 1 else ""
        pdf_stem = parts[1] if len(parts) > 1 else doc_id

        # Find ground truth via mappings CSV
        mapping = _find_mapping(doc_id, mappings)
        if not mapping:
            print(f"  {doc_id}: no mapping found, skipping")
            continue

        gt = load_ground_truth(ANNOTATIONS_DIR, mapping["identifier"], mapping["group"])
        all_gt = gt["techniques"] | gt["tactics"] | gt["software"]
        if not all_gt:
            print(f"  {doc_id}: no ground truth IDs found")
            continue

        entries = parse_similarity_results(doc_dir, max_k=max(ks))
        if not entries:
            print(f"  {doc_id}: no similarity results")
            continue

        metrics = compute_doc_metrics(doc_id, group, gt, entries, ks)
        all_metrics.append(metrics)
        all_tp_cosines.extend(metrics.tp_cosines)
        all_fp_cosines.extend(metrics.fp_cosines)

        # Print summary for this doc
        m = metrics.metrics_at_k.get(ks[-1], {})
        print(f"  {doc_id}: recall@{ks[-1]}={m.get('recall', 0):.2f} "
              f"({m.get('n_matched', 0)}/{m.get('n_gt', 0)}) "
              f"queries={metrics.n_queries}")

    if not all_metrics:
        print("No documents evaluated.")
        return

    # -----------------------------------------------------------------------
    # Per-group summary
    # -----------------------------------------------------------------------
    groups = sorted(set(m.group for m in all_metrics))
    group_summary = {}
    for g in groups:
        g_metrics = [m for m in all_metrics if m.group == g]
        group_summary[g] = {"n_docs": len(g_metrics)}
        for k in ks:
            recalls = [m.metrics_at_k[k]["recall"] for m in g_metrics]
            precisions = [m.metrics_at_k[k]["precision"] for m in g_metrics]
            hit_rates = [m.metrics_at_k[k]["hit_rate"] for m in g_metrics]
            n = len(recalls)
            group_summary[g][f"recall@{k}"] = round(sum(recalls) / n, 4) if n else 0
            group_summary[g][f"precision@{k}"] = round(sum(precisions) / n, 4) if n else 0
            group_summary[g][f"hit@{k}"] = round(sum(hit_rates) / n, 4) if n else 0

    # -----------------------------------------------------------------------
    # Overall summary
    # -----------------------------------------------------------------------
    overall = {"n_docs": len(all_metrics)}
    for k in ks:
        recalls = [m.metrics_at_k[k]["recall"] for m in all_metrics]
        precisions = [m.metrics_at_k[k]["precision"] for m in all_metrics]
        hit_rates = [m.metrics_at_k[k]["hit_rate"] for m in all_metrics]
        n = len(recalls)
        overall[f"recall@{k}"] = round(sum(recalls) / n, 4)
        overall[f"precision@{k}"] = round(sum(precisions) / n, 4)
        overall[f"hit@{k}"] = round(sum(hit_rates) / n, 4)

    # Per-category overall
    for cat in ["techniques", "tactics", "software"]:
        cat_data = {}
        for k in ks:
            cat_recalls = [m.category_metrics_at_k[cat][k]["recall"] for m in all_metrics
                          if m.category_metrics_at_k[cat][k]["n_gt"] > 0]
            cat_data[f"recall@{k}"] = round(sum(cat_recalls) / len(cat_recalls), 4) if cat_recalls else 0
        overall[f"{cat}"] = cat_data

    # -----------------------------------------------------------------------
    # Cosine threshold analysis
    # -----------------------------------------------------------------------
    import numpy as np
    cosine_analysis = {}
    if all_tp_cosines:
        tp_arr = np.array(all_tp_cosines)
        cosine_analysis["tp"] = {
            "count": len(tp_arr), "mean": round(float(tp_arr.mean()), 4),
            "median": round(float(np.median(tp_arr)), 4),
            "std": round(float(tp_arr.std()), 4),
            "min": round(float(tp_arr.min()), 4),
            "max": round(float(tp_arr.max()), 4),
        }
    if all_fp_cosines:
        fp_arr = np.array(all_fp_cosines)
        cosine_analysis["fp"] = {
            "count": len(fp_arr), "mean": round(float(fp_arr.mean()), 4),
            "median": round(float(np.median(fp_arr)), 4),
            "std": round(float(fp_arr.std()), 4),
            "min": round(float(fp_arr.min()), 4),
            "max": round(float(fp_arr.max()), 4),
        }
    # Optimal threshold (maximize F1 over a sweep)
    if all_tp_cosines and all_fp_cosines:
        all_scores = [(c, True) for c in all_tp_cosines] + [(c, False) for c in all_fp_cosines]
        best_f1, best_thresh = 0, 0
        for thresh in np.arange(0.1, 0.9, 0.01):
            tp = sum(1 for c, is_tp in all_scores if c >= thresh and is_tp)
            fp = sum(1 for c, is_tp in all_scores if c >= thresh and not is_tp)
            fn = sum(1 for c, is_tp in all_scores if c < thresh and is_tp)
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            if f1 > best_f1:
                best_f1, best_thresh = f1, float(thresh)
        cosine_analysis["optimal_threshold"] = round(best_thresh, 3)
        cosine_analysis["optimal_f1"] = round(best_f1, 4)

    # -----------------------------------------------------------------------
    # Token search evaluation
    # -----------------------------------------------------------------------
    token_dir = EXPERIMENT_DIR / "eval" / "token_search"
    token_results = {}
    if token_dir.exists():
        print(f"\nToken search results:")
        for doc_dir in sorted(token_dir.iterdir()):
            if not doc_dir.is_dir():
                continue
            ts_file = doc_dir / "token_search.json"
            if not ts_file.exists():
                continue
            doc_id = doc_dir.name
            ts = json.loads(ts_file.read_text(encoding="utf-8"))
            ts_ids = set(ts.get("combined_matched_ids", []) or ts.get("matched_ids", []))

            # Find GT for this doc
            mapping = _find_mapping(doc_id, mappings)
            if not mapping:
                continue

            gt = load_ground_truth(ANNOTATIONS_DIR, mapping["identifier"], mapping["group"])
            all_gt_ids = gt["techniques"] | gt["tactics"] | gt["software"]
            if not all_gt_ids:
                continue

            matched = ts_ids & all_gt_ids
            precision = len(matched) / len(ts_ids) if ts_ids else 0
            recall = len(matched) / len(all_gt_ids) if all_gt_ids else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

            # Per-category
            cat_recall = {}
            for cat, cat_ids in [("techniques", gt["techniques"]), ("tactics", gt["tactics"]), ("software", gt["software"])]:
                cat_recall[cat] = len(ts_ids & cat_ids) / len(cat_ids) if cat_ids else 0

            # BM25-only and regex-only breakdowns
            bm25_ids = set(ts.get("bm25_matched_ids", []))
            regex_ids = set(ts.get("regex_matched_ids", []))
            bm25_recall = len(bm25_ids & all_gt_ids) / len(all_gt_ids) if all_gt_ids else 0
            regex_recall = len(regex_ids & all_gt_ids) / len(all_gt_ids) if all_gt_ids else 0

            token_results[doc_id] = {
                "matched": sorted(matched),
                "missed": sorted(all_gt_ids - ts_ids),
                "spurious": sorted(ts_ids - all_gt_ids),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "n_matched": len(matched),
                "n_gt": len(all_gt_ids),
                "n_found": len(ts_ids),
                "bm25_recall": round(bm25_recall, 4),
                "regex_recall": round(regex_recall, 4),
                "category_recall": {c: round(v, 4) for c, v in cat_recall.items()},
            }
            print(f"  {doc_id}: recall={recall:.2f} ({len(matched)}/{len(all_gt_ids)}) "
                  f"precision={precision:.2f} f1={f1:.2f}")
            for cat in ["techniques", "tactics", "software"]:
                cat_ids = gt[cat]
                found = ts_ids & cat_ids
                print(f"    {cat:12s}: {len(found)}/{len(cat_ids)}")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    results = {
        "overall": overall,
        "per_group": group_summary,
        "cosine_analysis": cosine_analysis,
        "token_search": token_results,
        "per_document": {m.doc_id: {
            "group": m.group,
            "n_queries": m.n_queries,
            "gt": m.gt,
            "metrics_at_k": {str(k): v for k, v in m.metrics_at_k.items()},
            "category_metrics_at_k": {
                cat: {str(k): v for k, v in cat_ks.items()}
                for cat, cat_ks in m.category_metrics_at_k.items()
            },
        } for m in all_metrics},
    }

    out_path = output_dir / "similarity_eval.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"CTI-HAL Similarity Evaluation ({len(all_metrics)} documents)")
    print(f"{'='*70}")
    print(f"\nOverall (macro-averaged):")
    for k in ks:
        print(f"  k={k:2d}  hit={overall[f'hit@{k}']:.4f}  "
              f"precision={overall[f'precision@{k}']:.4f}  "
              f"recall={overall[f'recall@{k}']:.4f}")

    print(f"\nPer category recall:")
    for cat in ["techniques", "tactics", "software"]:
        vals = "  ".join(f"@{k}={overall[cat][f'recall@{k}']:.3f}" for k in ks)
        print(f"  {cat:12s}  {vals}")

    # Rollup summary
    print(f"\nWith sub-technique rollup (macro-averaged):")
    for k in ks:
        rollup_recalls = [m.rollup_metrics_at_k[k]["recall"] for m in all_metrics
                          if k in m.rollup_metrics_at_k]
        n = len(rollup_recalls) or 1
        print(f"  k={k:2d}  recall={sum(rollup_recalls)/n:.4f}")

    print(f"\nPer category recall (with rollup):")
    for cat in ["techniques", "tactics", "software"]:
        vals_parts = []
        for k in ks:
            cat_recalls = [m.rollup_category_metrics_at_k[cat][k]["recall"] for m in all_metrics
                          if cat in m.rollup_category_metrics_at_k and k in m.rollup_category_metrics_at_k[cat]
                          and m.rollup_category_metrics_at_k[cat][k]["n_gt"] > 0]
            avg = sum(cat_recalls) / len(cat_recalls) if cat_recalls else 0
            vals_parts.append(f"@{k}={avg:.3f}")
        print(f"  {cat:12s}  {'  '.join(vals_parts)}")

    if cosine_analysis:
        print(f"\nCosine score analysis:")
        if "tp" in cosine_analysis:
            tp = cosine_analysis["tp"]
            print(f"  TP: mean={tp['mean']:.4f} median={tp['median']:.4f} std={tp['std']:.4f} (n={tp['count']})")
        if "fp" in cosine_analysis:
            fp = cosine_analysis["fp"]
            print(f"  FP: mean={fp['mean']:.4f} median={fp['median']:.4f} std={fp['std']:.4f} (n={fp['count']})")
        if "optimal_threshold" in cosine_analysis:
            print(f"  Optimal threshold: {cosine_analysis['optimal_threshold']:.3f} (F1={cosine_analysis['optimal_f1']:.4f})")

    print(f"\nPer group:")
    for g in groups:
        gs = group_summary[g]
        r_vals = "  ".join(f"@{k}={gs.get(f'recall@{k}', 0):.3f}" for k in ks)
        print(f"  {g:15s} ({gs['n_docs']} docs)  recall: {r_vals}")

    if token_results:
        print(f"\nBM25 Search:")
        for doc_id, tr in token_results.items():
            print(f"  {doc_id}: recall={tr['recall']:.4f} precision={tr['precision']:.4f} f1={tr['f1']:.4f} "
                  f"({tr['n_matched']}/{tr['n_gt']})")

    # -----------------------------------------------------------------------
    # Hybrid RRF evaluation
    # -----------------------------------------------------------------------
    hybrid_dir = EXPERIMENT_DIR / "eval" / "hybrid"
    hybrid_results = {}
    if hybrid_dir.exists():
        for doc_dir in sorted(hybrid_dir.iterdir()):
            if not doc_dir.is_dir():
                continue
            rrf_file = doc_dir / "hybrid_rrf.json"
            if not rrf_file.exists():
                continue
            doc_id = doc_dir.name
            rrf = json.loads(rrf_file.read_text(encoding="utf-8"))
            rrf_ids = set(rrf.get("fused_ids", []))

            # Find GT
            mapping = _find_mapping(doc_id, mappings)
            if not mapping:
                continue

            gt = load_ground_truth(ANNOTATIONS_DIR, mapping["identifier"], mapping["group"])
            all_gt_ids = gt["techniques"] | gt["tactics"] | gt["software"]
            if not all_gt_ids:
                continue

            # Eval at different k cutoffs using the RRF ranking
            fused_ranked = [item["id"] for item in rrf.get("fused_ranked", [])]
            hybrid_at_k = {}
            for eval_k in [10, 20, 50, 100]:
                top_ids = set(fused_ranked[:eval_k])
                matched = top_ids & all_gt_ids
                p = len(matched) / len(top_ids) if top_ids else 0
                r = len(matched) / len(all_gt_ids) if all_gt_ids else 0
                f1_val = 2 * p * r / (p + r) if (p + r) else 0
                hybrid_at_k[eval_k] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1_val, 4),
                                        "n_matched": len(matched), "n_gt": len(all_gt_ids)}

            # Per-category at k=50
            cat_recall = {}
            top50 = set(fused_ranked[:50])
            for cat, cat_ids in [("techniques", gt["techniques"]), ("tactics", gt["tactics"]), ("software", gt["software"])]:
                cat_recall[cat] = round(len(top50 & cat_ids) / len(cat_ids), 4) if cat_ids else 0

            # All IDs (no cutoff)
            all_matched = rrf_ids & all_gt_ids
            all_p = len(all_matched) / len(rrf_ids) if rrf_ids else 0
            all_r = len(all_matched) / len(all_gt_ids) if all_gt_ids else 0
            all_f1 = 2 * all_p * all_r / (all_p + all_r) if (all_p + all_r) else 0

            # Source overlap stats — support both 2-source and 3-source formats
            sc = rrf.get("source_counts", {})
            hybrid_results[doc_id] = {
                "all": {"precision": round(all_p, 4), "recall": round(all_r, 4), "f1": round(all_f1, 4),
                        "n_matched": len(all_matched), "n_gt": len(all_gt_ids), "n_found": len(rrf_ids)},
                "at_k": hybrid_at_k,
                "category_recall_at_50": cat_recall,
                "sources": rrf.get("sources", {}),
                "all_three": sc.get("all_three", 0),
                "two_sources": sc.get("two_sources", 0),
                "one_source": sc.get("one_source", 0),
            }

    results["hybrid"] = hybrid_results

    if hybrid_results:
        print(f"\nHybrid RRF:")
        for doc_id, hr in hybrid_results.items():
            print(f"  {doc_id}:")
            for eval_k, m in sorted(hr["at_k"].items()):
                print(f"    @{eval_k:3d}: recall={m['recall']:.4f} precision={m['precision']:.4f} f1={m['f1']:.4f} ({m['n_matched']}/{m['n_gt']})")
            print(f"    categories@50: tech={hr['category_recall_at_50']['techniques']:.3f} "
                  f"tact={hr['category_recall_at_50']['tactics']:.3f} "
                  f"soft={hr['category_recall_at_50']['software']:.3f}")
            print(f"    3-source={hr['all_three']}, 2-source={hr['two_sources']}, 1-source={hr['one_source']}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
