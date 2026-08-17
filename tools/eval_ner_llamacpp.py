"""NER evaluation via llama.cpp backend.

Runs pass-1 entity extraction on each test doc, compares to gold DNRTI
entities using fuzzy matching (SequenceMatcher >= 0.70).

Usage:
    python tools/eval_ner_llamacpp.py \
        --data results/finetuning/data/test_dnrti.json \
        --experiment base-gemma2-9b-dedup \
        --model-label "gemma2-9b-base" \
        --url http://127.0.0.1:8080 \
        [--lora-id 0] \
        [--limit 10]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cti_analysis.llm_backend import LlamaCppBackend
from cti_analysis.ontology import ENTITY_EXTRACTION_PROMPT, TYPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

FUZZY_THRESHOLD = 0.70


def fuzzy_match(pred_name: str, gold_name: str) -> bool:
    pn = pred_name.lower().strip().strip('"').strip("'").rstrip(".")
    gn = gold_name.lower().strip()
    return pn == gn or SequenceMatcher(None, pn, gn).ratio() >= FUZZY_THRESHOLD


def main():
    parser = argparse.ArgumentParser(description="NER eval via llama.cpp")
    parser.add_argument("--data", required=True, help="Test data JSON (DNRTI format)")
    parser.add_argument("--experiment", required=True, help="Experiment name for output")
    parser.add_argument("--model-label", default="gemma2-9b", help="Model label for results JSON")
    parser.add_argument("--dataset-label", default="", help="Dataset description for results JSON")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="llama-server URL")
    parser.add_argument("--lora-id", type=int, default=None, help="LoRA adapter ID (None = no adapter)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N docs")
    parser.add_argument("--output", default=None, help="Output JSON path (default: experiments/<experiment>/eval/ner.json)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: {data_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(data_path, encoding="utf-8") as f:
        test_data = json.load(f)
    if args.limit:
        test_data = test_data[: args.limit]

    output_path = Path(args.output) if args.output else (
        Path("experiments") / args.experiment / "eval" / "ner.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = LlamaCppBackend(url=args.url)
    lora_id = args.lora_id

    logger.info("Running NER eval on %d test docs (lora_id=%s)", len(test_data), lora_id)

    type_tp, type_fp, type_fn = Counter(), Counter(), Counter()
    start = time.time()

    for i, doc in enumerate(test_data):
        text = doc.get("text", "").strip()
        if not text:
            continue

        # Parse gold entities
        gold_entities = []
        for ent in doc.get("entities", []):
            if isinstance(ent, list) and len(ent) >= 4:
                gold_entities.append((ent[2].lower().strip(), ent[3]))

        if not gold_entities:
            continue

        # Run NER extraction
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        response = backend.generate(prompt, lora_id=lora_id)

        # Parse predicted entities
        pred_entities = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line or line.upper() == "NONE":
                continue
            if line.startswith("entity_name") or line.startswith("---"):
                continue
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                name = parts[0].lower().strip()
                etype = parts[1].strip()
                if name and etype in TYPES:
                    pred_entities.append((name, etype))

        # Match predicted to gold (fuzzy name match + exact type match)
        gold_matched = set()
        pred_matched = set()
        for gi, (gn, gt) in enumerate(gold_entities):
            for pi, (pn, pt) in enumerate(pred_entities):
                if pi not in pred_matched and gt == pt and fuzzy_match(pn, gn):
                    gold_matched.add(gi)
                    pred_matched.add(pi)
                    type_tp[gt] += 1
                    break

        for gi, (gn, gt) in enumerate(gold_entities):
            if gi not in gold_matched:
                type_fn[gt] += 1

        for pi, (pn, pt) in enumerate(pred_entities):
            if pi not in pred_matched:
                type_fp[pt] += 1

        if (i + 1) % 50 == 0:
            logger.info(
                "Processed %d/%d  TP=%d FP=%d FN=%d",
                i + 1, len(test_data),
                sum(type_tp.values()), sum(type_fp.values()), sum(type_fn.values()),
            )

    elapsed = time.time() - start
    tp = sum(type_tp.values())
    fp = sum(type_fp.values())
    fn = sum(type_fn.values())
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0

    all_types = sorted(set(list(type_tp.keys()) + list(type_fp.keys()) + list(type_fn.keys())))
    per_type = {}
    mp, mr, mf, nt = 0, 0, 0, 0
    for t in all_types:
        ttp, tfp, tfn = type_tp[t], type_fp[t], type_fn[t]
        tp_ = ttp / (ttp + tfp) if ttp + tfp else 0
        tr_ = ttp / (ttp + tfn) if ttp + tfn else 0
        tf_ = 2 * tp_ * tr_ / (tp_ + tr_) if tp_ + tr_ else 0
        per_type[t] = {
            "precision": round(tp_, 4), "recall": round(tr_, 4), "f1": round(tf_, 4),
            "tp": ttp, "fp": tfp, "fn": tfn, "gold": ttp + tfn,
        }
        if ttp + tfn > 0:
            mp += tp_; mr += tr_; mf += tf_; nt += 1

    results = {
        "model": args.model_label,
        "experiment": args.experiment,
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "task": "NER",
        "test_docs": len(test_data),
        "dataset": args.dataset_label or str(data_path),
        "lora_id": lora_id,
        "micro": {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn},
        "macro": {"precision": round(mp / nt, 4) if nt else 0, "recall": round(mr / nt, 4) if nt else 0, "f1": round(mf / nt, 4) if nt else 0},
        "per_type": per_type,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nNER Results ({args.model_label}, {args.experiment}):")
    print(f"  Micro: P={p:.4f} R={r:.4f} F1={f1:.4f}")
    print(f"  Macro: P={mp/nt:.4f} R={mr/nt:.4f} F1={mf/nt:.4f}" if nt else "  Macro: N/A")
    print(f"  TP={tp} FP={fp} FN={fn}")
    print(f"  Time: {elapsed:.1f}s")
    print()
    for t in sorted(per_type.keys()):
        d = per_type[t]
        print(f"  {t:15s} P={d['precision']:.2f} R={d['recall']:.2f} F1={d['f1']:.2f} (gold={d['gold']})")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
