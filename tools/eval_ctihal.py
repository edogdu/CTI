"""Evaluate CTI-HAL pipeline results against ground truth annotations.

Computes Hit@k metrics: for each ground truth ATT&CK ID (technique, tactic,
software) in a report, checks whether it appears in the top-k similarity
matches from the UCKG alignment stage.

Usage:
    python tools/eval_ctihal.py \
        --triples experiments/ctihal-pipeline/eval/triples/ \
        --similarity experiments/ctihal-pipeline/eval/similarity/ \
        --annotations datasets/CTI-HAL/data/ \
        --output experiments/ctihal-pipeline/eval/hit_at_k.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "src"))

# ATT&CK ID patterns
TECHNIQUE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
TACTIC_RE = re.compile(r"\bTA0\d{3}\b", re.IGNORECASE)
SOFTWARE_RE = re.compile(r"\bS\d{4}\b", re.IGNORECASE)

ATTACK_LABELS = {"UcoexMITREATTACK", "ATT&CKTechnique", "UcoexTACTICS", "ATT&CKTactic", "UcoexSOFTWARE", "ATT&CKSoftware"}


def load_ground_truth(annotations_dir: Path) -> Dict[str, Dict[str, Set[str]]]:
    """Load ground truth ATT&CK IDs per report from CTI-HAL annotation JSONs.

    Returns: {group_reportname: {"techniques": set, "tactics": set, "software": set}}
    """
    gt = {}
    for json_file in sorted(annotations_dir.rglob("*.json")):
        # Determine group and report name
        parts = json_file.relative_to(annotations_dir).parts
        if "annotator" in str(json_file):
            group = parts[0]  # e.g., apt29
        else:
            group = parts[0]
        report_name = json_file.stem

        key = f"{group}_{report_name}"
        if key in gt:
            # Merge annotations from multiple annotators
            pass
        else:
            gt[key] = {"techniques": set(), "tactics": set(), "software": set()}

        try:
            anns = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        for ann in anns:
            tech = ann.get("technique")
            if tech and isinstance(tech, str):
                gt[key]["techniques"].add(tech.upper())
            meta = ann.get("metadata", {})
            if not meta:
                continue
            for t in (meta.get("tactic") or []):
                if t:
                    gt[key]["tactics"].add(t.upper())
            sub = meta.get("sub_technique")
            if sub and isinstance(sub, str):
                gt[key]["techniques"].add(sub.upper())
            for s in (meta.get("tool") or []):
                if s:
                    gt[key]["software"].add(s.upper())

    return gt


def extract_ids_from_similarity(sim_dir: Path) -> Set[str]:
    """Extract ATT&CK IDs from similarity results (per_node or leaderboard)."""
    ids = set()
    per_node = sim_dir / "similarity_per_node.json"
    if not per_node.exists():
        return ids

    try:
        data = json.loads(per_node.read_text(encoding="utf-8"))
    except Exception:
        return ids

    def _collect(obj):
        if isinstance(obj, dict):
            # Check for clean_id on targets
            if "top_k" in obj:
                for hit in obj.get("top_k", []):
                    target = hit.get("target", {})
                    labels = set(target.get("labels", []))
                    if ATTACK_LABELS & labels:
                        cid = target.get("clean_id", "")
                        if cid:
                            ids.add(cid.strip().upper())
                        # Also check URI
                        uri = target.get("uri", "")
                        if uri:
                            for pat in [TECHNIQUE_RE, TACTIC_RE, SOFTWARE_RE]:
                                for m in pat.findall(uri):
                                    ids.add(m.upper())
            for v in obj.values():
                _collect(v)
        elif isinstance(obj, list):
            for v in obj:
                _collect(v)

    _collect(data)
    return ids


def compute_hit_at_k(gt_ids: Set[str], predicted_ids: Set[str]) -> Dict[str, float]:
    """Compute Hit@k style metrics (is any GT ID found in predictions?)."""
    if not gt_ids:
        return {"hit_rate": 0.0, "hits": 0, "total": 0}
    hits = gt_ids & predicted_ids
    return {
        "hit_rate": len(hits) / len(gt_ids),
        "hits": len(hits),
        "total": len(gt_ids),
        "found": sorted(hits),
        "missing": sorted(gt_ids - predicted_ids),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CTI-HAL pipeline against ground truth")
    parser.add_argument("--similarity", default="experiments/ctihal-pipeline/eval/similarity/",
                        help="Directory with per-document similarity results")
    parser.add_argument("--annotations", default="datasets/CTI-HAL/data/",
                        help="CTI-HAL annotation directory")
    parser.add_argument("--output", default="experiments/ctihal-pipeline/eval/hit_at_k.json",
                        help="Output JSON path")
    args = parser.parse_args()

    sim_dir = Path(args.similarity)
    ann_dir = Path(args.annotations)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth annotations...")
    gt = load_ground_truth(ann_dir)
    print(f"  {len(gt)} report annotations loaded")

    print("Evaluating similarity results...")
    per_doc = {}
    per_group = defaultdict(lambda: {"techniques": {"hits": 0, "total": 0},
                                      "tactics": {"hits": 0, "total": 0},
                                      "software": {"hits": 0, "total": 0}})

    for doc_dir in sorted(sim_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        predicted_ids = extract_ids_from_similarity(doc_dir)

        # Find matching ground truth (try various key formats)
        group = doc_id.split("_")[0] if "_" in doc_id else ""
        report = "_".join(doc_id.split("_")[1:]) if "_" in doc_id else doc_id

        # Try to match against GT keys
        gt_entry = None
        for gt_key, gt_val in gt.items():
            if gt_key.lower() == doc_id.lower() or gt_key.lower().startswith(doc_id.lower()):
                gt_entry = gt_val
                break
            # Try partial match on report name
            if report.lower() in gt_key.lower():
                gt_entry = gt_val
                break

        if gt_entry is None:
            per_doc[doc_id] = {"status": "no_ground_truth", "predicted_ids": sorted(predicted_ids)}
            continue

        all_gt = gt_entry["techniques"] | gt_entry["tactics"] | gt_entry["software"]
        metrics = compute_hit_at_k(all_gt, predicted_ids)

        tech_metrics = compute_hit_at_k(gt_entry["techniques"], predicted_ids)
        tactic_metrics = compute_hit_at_k(gt_entry["tactics"], predicted_ids)
        software_metrics = compute_hit_at_k(gt_entry["software"], predicted_ids)

        per_doc[doc_id] = {
            "overall": metrics,
            "techniques": tech_metrics,
            "tactics": tactic_metrics,
            "software": software_metrics,
            "predicted_ids": sorted(predicted_ids),
        }

        # Aggregate per group
        for cat in ["techniques", "tactics", "software"]:
            cat_m = {"techniques": tech_metrics, "tactics": tactic_metrics, "software": software_metrics}[cat]
            per_group[group][cat]["hits"] += cat_m.get("hits", 0)
            per_group[group][cat]["total"] += cat_m.get("total", 0)

    # Compute aggregates
    total_hits = sum(d.get("overall", {}).get("hits", 0) for d in per_doc.values() if isinstance(d.get("overall"), dict))
    total_gt = sum(d.get("overall", {}).get("total", 0) for d in per_doc.values() if isinstance(d.get("overall"), dict))

    results = {
        "overall": {
            "hit_rate": total_hits / total_gt if total_gt else 0,
            "hits": total_hits,
            "total": total_gt,
        },
        "per_group": {g: {cat: {**v, "hit_rate": v["hits"] / v["total"] if v["total"] else 0}
                          for cat, v in cats.items()}
                      for g, cats in per_group.items()},
        "per_document": per_doc,
    }

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary
    print(f"\nOverall Hit Rate: {results['overall']['hit_rate']:.4f} ({total_hits}/{total_gt})")
    print(f"\nPer Group:")
    for g, cats in sorted(results["per_group"].items()):
        tech = cats["techniques"]
        print(f"  {g:15s}  techniques: {tech['hits']}/{tech['total']} = {tech['hit_rate']:.2f}"
              f"  tactics: {cats['tactics']['hits']}/{cats['tactics']['total']}"
              f"  software: {cats['software']['hits']}/{cats['software']['total']}")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
