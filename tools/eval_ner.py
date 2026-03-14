"""NER evaluation: run pass-1 entity extraction on each test doc,
compare to gold DNRTI entities, report per-type and aggregate metrics.

Usage:
    python tools/eval_ner.py [--data PATH] [--model NAME] [--url URL]
                             [--timeout INT] [--output PATH] [--limit INT]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "src"))
sys.path.insert(0, str(_repo))

from cti_analysis.triple_extraction.extraction.extractor import _extract_entities_from_chunk
from tools.eval_shared import (
    aggregate_metrics, compute_ner_metrics, norm_name,
    parse_gold_entities, parse_gold_triples,
)

logging.basicConfig(level=logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="NER evaluation against gold DNRTI labels")
    parser.add_argument("--data", default="results/finetuning/data/test_dnrti.json")
    parser.add_argument("--model", default="gemma3-cti-4b")
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output", default=None,
                        help="JSON output path (default: results/eval/ner_<timestamp>.json)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N docs (for quick smoke tests)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    with open(data_path, encoding="utf-8") as f:
        docs = json.load(f)
    if args.limit:
        docs = docs[: args.limit]

    output_path = Path(args.output) if args.output else (
        Path("results/eval") / f"ner_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"NER Evaluation")
    print(f"  model  : {args.model}")
    print(f"  data   : {data_path} ({len(docs)} docs)")
    print(f"  output : {output_path}")
    print()

    per_doc_results = []
    per_type_agg: dict = {}

    for i, doc in enumerate(docs):
        text = doc.get("text", "")
        gold = parse_gold_entities(doc)

        if not text.strip() or not gold:
            continue

        # Pass 1: entity extraction
        pred_raw = _extract_entities_from_chunk(
            text,
            model=args.model,
            base_url=args.url,
            temperature=0.1,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        pred = [(d["name"], d["type"]) for d in pred_raw]

        metrics = compute_ner_metrics(gold, pred)

        # Accumulate per-type
        for etype, counts in metrics["per_type"].items():
            if etype not in per_type_agg:
                per_type_agg[etype] = {"tp": 0, "fp": 0, "fn": 0}
            for k in ("tp", "fp", "fn"):
                per_type_agg[etype][k] += counts[k]

        per_doc_results.append({
            "doc_idx": i,
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        })

        if (i + 1) % 50 == 0:
            done = i + 1
            print(f"  [{done}/{len(docs)}] running TP={sum(r['tp'] for r in per_doc_results)} "
                  f"FP={sum(r['fp'] for r in per_doc_results)} "
                  f"FN={sum(r['fn'] for r in per_doc_results)}")

    summary = aggregate_metrics(per_doc_results)

    # Per-type summary table
    per_type_summary = {}
    for etype, counts in per_type_agg.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_type_summary[etype] = {
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "gold": tp + fn,
        }

    result = {
        "task": "NER",
        "model": args.model,
        "data": str(data_path),
        "n_docs": len(docs),
        "n_evaluated": len(per_doc_results),
        "timestamp": datetime.now().isoformat(),
        "micro": {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in summary["micro"].items()},
        "macro": {k: round(v, 4) for k, v in summary["macro"].items()},
        "per_type": per_type_summary,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Print results
    m = result["micro"]
    print(f"\n{'='*60}")
    print(f"NER RESULTS  (model: {args.model})")
    print(f"{'='*60}")
    print(f"Docs evaluated : {result['n_evaluated']} / {result['n_docs']}")
    print(f"\nMICRO  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
    print(f"       TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    mm = result["macro"]
    print(f"MACRO  P={mm['precision']:.4f}  R={mm['recall']:.4f}  F1={mm['f1']:.4f}")
    print(f"\nPER-TYPE:")
    print(f"  {'Type':<10} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'Gold':>6}")
    print(f"  {'-'*55}")
    for etype, s in sorted(per_type_summary.items()):
        print(f"  {etype:<10} {s['precision']:>6.4f} {s['recall']:>6.4f} {s['f1']:>6.4f} "
              f"{s['tp']:>5} {s['fp']:>5} {s['fn']:>5} {s['gold']:>6}")
    print(f"\nSaved: {output_path}")

    return result


if __name__ == "__main__":
    main()
