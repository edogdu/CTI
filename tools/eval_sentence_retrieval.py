"""Evaluate sentence-level retrieval for CTI-HAL TTP detection.

Usage:
    python tools/eval_sentence_retrieval.py --run          # compute + evaluate
    python tools/eval_sentence_retrieval.py --run --limit 1  # 1 doc only
    python tools/eval_sentence_retrieval.py                # evaluate existing results
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "src"))

DEFAULT_EVAL_DIR = _repo / "experiments" / "ctihal-pipeline" / "eval"
EVAL_DIR = DEFAULT_EVAL_DIR  # overridden by --eval-dir
SENT_DIR = EVAL_DIR / "sentence_retrieval"
ANN_DIR = _repo / "datasets" / "CTI-HAL" / "data"
MAPPINGS_CSV = _repo / "config" / "CTI_HAL_mappings.csv"


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
        return {"techniques": techs, "tactics": tacts, "software": soft}
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
    return {"techniques": techs, "tactics": tacts, "software": soft}


def rollup(ids):
    expanded = set(ids)
    for i in ids:
        if "." in i and i.startswith("T"):
            expanded.add(i.split(".")[0])
    return expanded


def find_mapping(doc_id, mappings):
    import re
    parts = doc_id.split("_", 1)
    pdf_stem = parts[1] if len(parts) > 1 else doc_id
    # Exact
    mapping = mappings.get(pdf_stem.lower())
    if mapping:
        return mapping
    # Normalized (alphanumeric only)
    norm = re.sub(r'[^a-z0-9]', '', pdf_stem.lower())
    for stem, m in mappings.items():
        if re.sub(r'[^a-z0-9]', '', stem.lower()) == norm:
            return m
    # Substring fallback
    for stem, m in mappings.items():
        if pdf_stem.lower() in stem or stem in pdf_stem.lower():
            return m
    return None


def run_retrieval(limit=None, database=None):
    """Compute sentence-level retrieval for documents that don't have results yet."""
    from neo4j import GraphDatabase
    from cti_analysis.config import load_config
    from cti_analysis.graph_alignment.sentence_retrieval.retriever import (
        run_sentence_retrieval,
    )

    cfg = load_config()
    driver_kwargs = {}
    if database:
        driver_kwargs["database"] = database
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
                                  **driver_kwargs)

    # Find all doc_ids in Neo4j
    with driver.session() as session:
        result = session.run("MATCH (d:CTIDocument) RETURN d.id AS id ORDER BY d.id")
        doc_ids = [rec["id"] for rec in result]

    if limit:
        doc_ids = doc_ids[:limit]

    print(f"Found {len(doc_ids)} documents in Neo4j")

    for i, doc_id in enumerate(doc_ids):
        out_dir = SENT_DIR / doc_id
        out_file = out_dir / "sentence_retrieval.json"
        if out_file.exists():
            print(f"  [{i+1}/{len(doc_ids)}] {doc_id}: already computed, skipping")
            continue

        print(f"  [{i+1}/{len(doc_ids)}] {doc_id}: computing...", end="", flush=True)
        try:
            r = run_sentence_retrieval(
                driver,
                doc_id=doc_id,
                output_dir=str(out_dir),
                model=cfg.embeddings.model,
            )
            print(f" {r.get('n_unique_attack_ids', 0)} IDs from "
                  f"{r.get('n_sentences', 0)} sentences "
                  f"({r.get('n_sentences_with_triples', 0)} with triples)")
        except Exception as e:
            print(f" ERROR: {e}")

    driver.close()


def evaluate():
    """Evaluate existing sentence-level retrieval results."""
    mappings = load_mappings()

    # Load results
    docs = {}
    for doc_dir in sorted(SENT_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue
        result_file = doc_dir / "sentence_retrieval.json"
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError):
            continue

        doc_id = doc_dir.name
        mapping = find_mapping(doc_id, mappings)
        if not mapping:
            continue
        gt = load_gt(mapping["identifier"], mapping["group"])
        all_gt = gt["techniques"] | gt["tactics"] | gt["software"]
        if not all_gt:
            continue

        docs[doc_id] = {
            "data": data,
            "gt": gt,
            "all_gt": all_gt,
            "group": mapping["group"].lower(),
        }

    if not docs:
        print("No sentence retrieval results found. Run with --run first.")
        return

    print(f"Loaded {len(docs)} documents with ground truth\n")

    # Per-sentence hit rate
    total_sents = 0
    sents_with_hits = 0
    for d in docs.values():
        for ps in d["data"].get("per_sentence", []):
            total_sents += 1
            fused_ids = {item["id"] for item in ps.get("fused_top5", [])}
            if fused_ids & d["all_gt"]:
                sents_with_hits += 1

    if total_sents:
        print(f"Per-sentence hit rate: {sents_with_hits}/{total_sents} "
              f"= {sents_with_hits/total_sents:.4f}")
    print()

    # Document-level recall@k
    ks = [10, 20, 50, 100]
    variants = [
        ("max-score", "ranked_ids_max"),
        ("sum-score", "ranked_ids_sum"),
    ]

    print(f"{'Variant':<15s}  {'@10':>7s}  {'@20':>7s}  {'@50':>7s}  {'@100':>7s}  "
          f"{'@10r':>7s}  {'@50r':>7s}  {'@100r':>7s}")
    print("-" * 95)

    for vname, vkey in variants:
        recalls = {k: [] for k in ks}
        recalls_rollup = {k: [] for k in ks}

        for d in docs.values():
            ranked = d["data"].get(vkey, [])
            for k in ks:
                top = set(ranked[:k])
                r = len(top & d["all_gt"]) / len(d["all_gt"])
                recalls[k].append(r)

                top_r = rollup(top)
                gt_r = rollup(d["all_gt"])
                rr = len(top_r & gt_r) / len(gt_r) if gt_r else 0
                recalls_rollup[k].append(rr)

        row = f"{vname:<15s}"
        for k in ks:
            avg = sum(recalls[k]) / len(recalls[k])
            row += f"  {avg:7.4f}"
        for k in [10, 50, 100]:
            avg = sum(recalls_rollup[k]) / len(recalls_rollup[k])
            row += f"  {avg:7.4f}"
        print(row)

    # Per-category recall@10 for max-score
    print("\nPer-category recall@10 (max-score):")
    for cat in ["techniques", "tactics", "software"]:
        cat_recalls = []
        for d in docs.values():
            cat_gt = d["gt"][cat]
            if not cat_gt:
                continue
            ranked = d["data"].get("ranked_ids_max", [])
            top = set(ranked[:10])
            cat_recalls.append(len(top & cat_gt) / len(cat_gt))
        if cat_recalls:
            avg = sum(cat_recalls) / len(cat_recalls)
            print(f"  {cat:12s}: {avg:.4f} ({len(cat_recalls)} docs)")

    # Per-category recall@10 with rollup
    print("\nPer-category recall@10 with rollup (max-score):")
    for cat in ["techniques", "tactics", "software"]:
        cat_recalls = []
        for d in docs.values():
            cat_gt = d["gt"][cat]
            if not cat_gt:
                continue
            ranked = d["data"].get("ranked_ids_max", [])
            top_r = rollup(set(ranked[:10]))
            gt_r = rollup(cat_gt)
            cat_recalls.append(len(top_r & gt_r) / len(gt_r) if gt_r else 0)
        if cat_recalls:
            avg = sum(cat_recalls) / len(cat_recalls)
            print(f"  {cat:12s}: {avg:.4f} ({len(cat_recalls)} docs)")

    # Per-group breakdown
    print("\nPer-group recall@10 (max-score, with rollup):")
    groups = {}
    for d in docs.values():
        g = d["group"]
        if g not in groups:
            groups[g] = []
        ranked = d["data"].get("ranked_ids_max", [])
        top_r = rollup(set(ranked[:10]))
        gt_r = rollup(d["all_gt"])
        groups[g].append(len(top_r & gt_r) / len(gt_r) if gt_r else 0)

    for g in sorted(groups):
        avg = sum(groups[g]) / len(groups[g])
        print(f"  {g:15s} ({len(groups[g]):2d} docs): {avg:.4f}")

    # Per-doc detail
    print("\nPer-document recall@10 (max-score):")
    for doc_id, d in sorted(docs.items()):
        ranked = d["data"].get("ranked_ids_max", [])
        top = set(ranked[:10])
        matched = top & d["all_gt"]
        r = len(matched) / len(d["all_gt"])
        n_sents = d["data"].get("n_sentences", 0)
        n_triples = d["data"].get("n_sentences_with_triples", 0)
        print(f"  {doc_id}: {r:.2f} ({len(matched)}/{len(d['all_gt'])}) "
              f"[{n_sents} sents, {n_triples} w/ triples]")

    # Save summary
    summary = {
        "n_documents": len(docs),
        "per_sentence_hit_rate": sents_with_hits / total_sents if total_sents else 0,
    }
    for vname, vkey in variants:
        for k in ks:
            recalls_list = []
            for d in docs.values():
                ranked = d["data"].get(vkey, [])
                top_r = rollup(set(ranked[:k]))
                gt_r = rollup(d["all_gt"])
                recalls_list.append(len(top_r & gt_r) / len(gt_r) if gt_r else 0)
            summary[f"{vname}_recall@{k}_rollup"] = round(
                sum(recalls_list) / len(recalls_list), 4
            ) if recalls_list else 0

    summary_path = SENT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Sentence-level retrieval evaluation")
    parser.add_argument("--run", action="store_true", help="Compute retrieval (needs Neo4j)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of docs")
    parser.add_argument("--force", action="store_true", help="Recompute even if results exist")
    parser.add_argument("--eval-dir", type=str, default=None, help="Override eval directory")
    parser.add_argument("--db", type=str, default=None, help="Neo4j database name")
    args = parser.parse_args()

    global EVAL_DIR, SENT_DIR
    if args.eval_dir:
        EVAL_DIR = Path(args.eval_dir)
        SENT_DIR = EVAL_DIR / "sentence_retrieval"

    if args.run:
        if args.force:
            import shutil
            if SENT_DIR.exists():
                shutil.rmtree(SENT_DIR)
        run_retrieval(limit=args.limit, database=args.db)

    evaluate()


if __name__ == "__main__":
    main()
