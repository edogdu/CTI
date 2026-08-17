"""RE evaluation (oracle NER) via llama.cpp backend.

Runs pass-2 relation extraction with gold entity hints on each test doc,
compares to gold DNRTI relations using fuzzy matching (SequenceMatcher >= 0.70).

Usage:
    python tools/eval_re_llamacpp.py \
        --data results/finetuning/data/test_dnrti.json \
        --experiment base-gemma2-9b-dedup \
        --model-label "gemma2-9b-base" \
        --url http://127.0.0.1:8080 \
        [--lora-id 1] \
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
from cti_analysis.ontology import RELATION_EXTRACTION_PROMPT, format_entity_hints

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

FUZZY_THRESHOLD = 0.70


def fuzzy_match(pred_name: str, gold_name: str) -> bool:
    pn = pred_name.lower().strip().strip('"').strip("'").rstrip(".")
    gn = gold_name.lower().strip()
    return pn == gn or SequenceMatcher(None, pn, gn).ratio() >= FUZZY_THRESHOLD


def normalize_pred(p: str) -> str:
    return p.lower().replace("-", "").replace("_", "").replace(" ", "")


def main():
    parser = argparse.ArgumentParser(description="RE eval (oracle NER) via llama.cpp")
    parser.add_argument("--data", required=True, help="Test data JSON (DNRTI format)")
    parser.add_argument("--experiment", required=True, help="Experiment name for output")
    parser.add_argument("--model-label", default="gemma2-9b", help="Model label for results JSON")
    parser.add_argument("--dataset-label", default="", help="Dataset description for results JSON")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="llama-server URL")
    parser.add_argument("--lora-id", type=int, default=None, help="LoRA adapter ID (None = no adapter)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N docs")
    parser.add_argument("--output", default=None, help="Output JSON path (default: experiments/<experiment>/eval/re.json)")
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
        Path("experiments") / args.experiment / "eval" / "re.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = LlamaCppBackend(url=args.url)
    lora_id = args.lora_id

    logger.info("Running RE eval on %d test docs (oracle NER, lora_id=%s)", len(test_data), lora_id)

    pred_tp, pred_fp, pred_fn = Counter(), Counter(), Counter()
    start = time.time()

    for i, doc in enumerate(test_data):
        text = doc.get("text", "").strip()
        if not text:
            continue

        # Parse gold entities
        gold_ents = []
        ent_map = {}
        for idx, ent in enumerate(doc.get("entities", [])):
            if isinstance(ent, list) and len(ent) >= 4:
                gold_ents.append({"name": ent[2], "type": ent[3]})
                ent_map[idx] = ent[2].lower().strip()

        # Parse gold triples (DNRTI: [predicate, head_idx=OBJECT, tail_idx=SUBJECT])
        gold_triples = set()
        for rel in doc.get("relations", []):
            if not isinstance(rel, list) or len(rel) < 3:
                continue
            pred = normalize_pred(rel[0])
            if pred in ["norelation", "no_relation", "none"]:
                continue
            subj = ent_map.get(rel[2])  # tail_idx = SUBJECT
            obj = ent_map.get(rel[1])   # head_idx = OBJECT
            if subj and obj and pred:
                gold_triples.add((subj, pred, obj))

        if not gold_triples:
            continue

        # Run RE extraction with gold entity hints
        hints = format_entity_hints(gold_ents)
        prompt = RELATION_EXTRACTION_PROMPT.format(entity_hints=hints, text=text)
        response = backend.generate(prompt, lora_id=lora_id)

        # Parse predicted triples
        pred_triples = set()
        for line in response.strip().split("\n"):
            line = line.strip()
            if "|" not in line or line.lower().startswith("none"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                pred_norm = normalize_pred(parts[2])
                if pred_norm and pred_norm not in ["norelation"]:
                    pred_triples.add((parts[0].lower().strip(), pred_norm, parts[3].lower().strip()))

        # Match predicted to gold (fuzzy name match + exact predicate match)
        gold_matched = set()
        pred_matched = set()
        for gi, gt in enumerate(gold_triples):
            for pi, pt in enumerate(pred_triples):
                if pi not in pred_matched and gt[1] == pt[1] and fuzzy_match(pt[0], gt[0]) and fuzzy_match(pt[2], gt[2]):
                    gold_matched.add(gi)
                    pred_matched.add(pi)
                    pred_tp[gt[1]] += 1
                    break

        gold_list = list(gold_triples)
        pred_list = list(pred_triples)
        for gi in range(len(gold_list)):
            if gi not in gold_matched:
                pred_fn[gold_list[gi][1]] += 1
        for pi in range(len(pred_list)):
            if pi not in pred_matched:
                pred_fp[pred_list[pi][1]] += 1

        if (i + 1) % 100 == 0:
            logger.info("Processed %d/%d", i + 1, len(test_data))

    elapsed = time.time() - start
    tp = sum(pred_tp.values())
    fp = sum(pred_fp.values())
    fn = sum(pred_fn.values())
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0

    all_preds = sorted(set(list(pred_tp.keys()) + list(pred_fp.keys()) + list(pred_fn.keys())))
    per_pred = {}
    mp, mr, mf, nt = 0, 0, 0, 0
    for t in all_preds:
        ttp, tfp, tfn = pred_tp[t], pred_fp[t], pred_fn[t]
        tp_ = ttp / (ttp + tfp) if ttp + tfp else 0
        tr_ = ttp / (ttp + tfn) if ttp + tfn else 0
        tf_ = 2 * tp_ * tr_ / (tp_ + tr_) if tp_ + tr_ else 0
        per_pred[t] = {
            "precision": round(tp_, 4), "recall": round(tr_, 4), "f1": round(tf_, 4),
            "tp": ttp, "fp": tfp, "fn": tfn, "gold": ttp + tfn,
        }
        if ttp + tfn > 0:
            mp += tp_; mr += tr_; mf += tf_; nt += 1

    results = {
        "model": args.model_label,
        "experiment": args.experiment,
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "task": "RE",
        "test_docs": len(test_data),
        "dataset": args.dataset_label or str(data_path),
        "lora_id": lora_id,
        "micro": {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn},
        "macro": {"precision": round(mp / nt, 4) if nt else 0, "recall": round(mr / nt, 4) if nt else 0, "f1": round(mf / nt, 4) if nt else 0},
        "per_pred": per_pred,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nRE Results ({args.model_label}, {args.experiment}, oracle NER):")
    print(f"  Micro: P={p:.4f} R={r:.4f} F1={f1:.4f}")
    print(f"  Macro: P={mp/nt:.4f} R={mr/nt:.4f} F1={mf/nt:.4f}" if nt else "  Macro: N/A")
    print(f"  TP={tp} FP={fp} FN={fn}")
    print(f"  Time: {elapsed:.1f}s")
    print()
    for t in sorted(per_pred.keys()):
        d = per_pred[t]
        print(f"  {t:25s} P={d['precision']:.2f} R={d['recall']:.2f} F1={d['f1']:.2f} (gold={d['gold']})")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
