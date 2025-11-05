# triple_extraction/eval_cli.py
from __future__ import annotations

import argparse, json, os, sys, time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

# --- Import the extractor module so we can tweak its config in-process
import triple_extraction.extraction_semantic_only_v1 as ext
from triple_extraction.extraction_semantic_only_v1 import extract_triples_from_text

# --- Notebook-style evaluation helpers you requested
from triple_extraction.evaluation.notebook_eval import (
    cti_hal_ground_truth,
    evaluate_sample_vector_similarity,
)

# --- Dataset loaders
try:
    from datasets.loaders import (
        load_cti_hal_samples,
        load_anno_ctr_samples,
        load_dnrti_samples,
    )
except Exception as e:
    print("ERROR importing datasets.loaders:", e)
    print("Run from the project root and ensure datasets/loaders.py exists.")
    sys.exit(1)


@dataclass
class Row:
    sample_id: str
    extracted_count: int
    gt_count: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    time: float
    hit: int


def _load_dataset(dataset: str, base: str, split: str, samples: int) -> List[Dict[str, Any]]:
    if dataset == "cti-hal":
        return load_cti_hal_samples(base, num_samples=samples)
    elif dataset == "anno-ctr":
        sp = split if split and split.lower() != "auto" else "test"
        data = load_anno_ctr_samples(base, num_samples=samples, split=sp)
        if not data and sp != "dev":
            data = load_anno_ctr_samples(base, num_samples=samples, split="dev")
        if not data and sp != "train":
            data = load_anno_ctr_samples(base, num_samples=samples, split="train")
        return data
    elif dataset == "dnrti":
        sp = split if split and split.lower() != "auto" else "test"
        data = load_dnrti_samples(base, num_samples=samples, split=sp)
        if not data and sp != "dev":
            data = load_dnrti_samples(base, num_samples=samples, split="dev")
        if not data and sp != "train":
            data = load_dnrti_samples(base, num_samples=samples, split="train")
        return data
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _force_extractor_config(model: str, ollama: str, use_llm: bool, prompts: int, consensus: int):
    """
    Make absolutely sure the extractor is LLM-on in THIS Python process.
    """
    # env (for consistency with any code that re-reads env)
    if ollama: os.environ["OLLAMA_BASE_URL"] = ollama
    if model:  os.environ["CTI_MODEL_NAME"]  = model
    os.environ["USE_LLM"] = "1" if use_llm else "0"

    # module-level config (the real switch)
    ext.config.model_name = model or ext.config.model_name
    ext.config.ollama_base_url = (ollama or ext.config.ollama_base_url).rstrip("/")
    ext.config.use_llm = bool(use_llm)
    ext.config.num_prompts = max(1, int(prompts))
    ext.config.consensus_m = max(1, int(consensus))


def _preflight_smoke():
    """
    Fire one tiny extraction so you can see if we’re actually calling the LLM.
    """
    sample = "APT29 used spearphishing to deploy Cobalt Strike."
    t0 = time.time()
    pred = extract_triples_from_text(sample)
    dt = time.time() - t0
    print(f"\n[SMOKE] USE_LLM={ext.config.use_llm} model={ext.config.model_name} base={ext.config.ollama_base_url}")
    print("[SMOKE] prompts =", ext.config.num_prompts, "consensus_m =", ext.config.consensus_m)
    print("[SMOKE] extracted:", pred, f"in {dt:.2f}s\n")
    return pred


def run_eval(dataset: str, base: str, split: str, samples: int, threshold: float, debug_first: int):
    data = _load_dataset(dataset, base, split, samples)
    if not data:
        print("No samples loaded (check --base path / dataset contents).")
        return {}, []

    print("\n" + "=" * 78)
    print(f" RUN EVALUATION: {dataset} ({len(data)} samples)")
    print("=" * 78)
    print(f"{'Sample':<24} {'Ext':<3} {'GT':<3} {'TP':<3} {'FP':<3} {'FN':<3}  P     R     F1    Time")
    print("-" * 78)

    rows: List[Row] = []
    agg = dict(tp=0, fp=0, fn=0, pred=0, gold=0, hits=0)

    for i, s in enumerate(data):
        sid = s.get("id", f"{dataset}_{i}")
        text = s.get("text", "") or ""

        gt = s.get("ground_truth_triples")
        if gt is None and dataset == "cti-hal":
            gt = cti_hal_ground_truth(s.get("annotations", []), s.get("threat_actor", ""))

        t0 = time.time()
        pred = extract_triples_from_text(text)
        dt = time.time() - t0

        if i < max(0, int(debug_first)):
            print(f"\n[DEBUG] {sid}")
            print("  pred:", pred[:8])
            print("  gt  :", (gt or [])[:8])

        metrics = evaluate_sample_vector_similarity(
            extracted=pred,
            ground=gt or [],
            threshold=threshold,
            debug=False
        )
        row = Row(
            sample_id=sid,
            extracted_count=len(pred),
            gt_count=len(gt or []),
            tp=metrics["tp"],
            fp=metrics["fp"],
            fn=metrics["fn"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            time=dt,
            hit=metrics.get("hit", 1 if metrics.get("tp", 0) > 0 else 0),
        )
        rows.append(row)

        agg["tp"] += row.tp; agg["fp"] += row.fp; agg["fn"] += row.fn
        agg["pred"] += row.extracted_count; agg["gold"] += row.gt_count; agg["hits"] += row.hit

        print(f"{sid[:24]:<24} {row.extracted_count:<3} {row.gt_count:<3} {row.tp:<3} {row.fp:<3} {row.fn:<3} "
              f"{row.precision:.2f}  {row.recall:.2f}  {row.f1:.2f}  {dt:>5.1f}s")

    # Macro (mean of per-sample)
    if rows:
        macro_p = sum(r.precision for r in rows)/len(rows)
        macro_r = sum(r.recall for r in rows)/len(rows)
        macro_f = sum(r.f1 for r in rows)/len(rows)
    else:
        macro_p = macro_r = macro_f = 0.0

    # Micro (global)
    micro_p = agg["tp"] / (agg["tp"] + agg["fp"]) if (agg["tp"] + agg["fp"]) > 0 else 0.0
    micro_r = agg["tp"] / (agg["tp"] + agg["fn"]) if (agg["tp"] + agg["fn"]) > 0 else 0.0
    micro_f = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    print("-" * 78)
    print("Macro:", {"precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f, 4)})
    print("Micro:", {"precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f, 4),
                     "tp": agg["tp"], "fp": agg["fp"], "fn": agg["fn"], "gold": agg["gold"], "pred": agg["pred"],
                     "hit_rate": (agg["hits"] / len(rows)) if rows else 0.0})

    summary = {
        "dataset": dataset,
        "samples": len(rows),
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f},
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f,
                  "tp": agg["tp"], "fp": agg["fp"], "fn": agg["fn"], "gold": agg["gold"], "pred": agg["pred"],
                  "hit_rate": (agg["hits"] / len(rows)) if rows else 0.0},
        "threshold": threshold,
        "llm": {"use_llm": ext.config.use_llm, "model": ext.config.model_name, "base": ext.config.ollama_base_url,
                "prompts": ext.config.num_prompts, "consensus_m": ext.config.consensus_m},
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cti-hal", "anno-ctr", "dnrti"], help="Dataset to evaluate")
    ap.add_argument("--base", required=True, help="Dataset base folder (e.g., data/ANNO-CTR)")
    ap.add_argument("--samples", type=int, default=10, help="Max number of samples")
    ap.add_argument("--split", default="auto", help="Split: test/dev/train/auto (where supported)")
    ap.add_argument("--threshold", type=float, default=0.55, help="Cosine threshold for triple matching")
    ap.add_argument("--out", default="output/eval_results.json", help="Write results JSON here")
    ap.add_argument("--debug-first", type=int, default=1, help="Debug-print first N samples")
    # Control the extractor explicitly:
    ap.add_argument("--model", default=os.environ.get("CTI_MODEL_NAME", "gemma2:9b"))
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    ap.add_argument("--use-llm", action="store_true", help="Force enable LLM in extractor")
    ap.add_argument("--no-llm", action="store_true", help="Force disable LLM in extractor")
    ap.add_argument("--prompts", type=int, default=int(os.environ.get("NUM_PROMPTS", "3")))
    ap.add_argument("--consensus", type=int, default=int(os.environ.get("CONSENSUS_M", "2")))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    use_llm = True
    if args.use_llm: use_llm = True
    if args.no_llm:  use_llm = False

    _force_extractor_config(
        model=args.model,
        ollama=args.ollama,
        use_llm=use_llm,
        prompts=args.prompts,
        consensus=args.consensus,
    )

    # Pre-flight: prove we’re actually calling the model
    _ = _preflight_smoke()

    summary, rows = run_eval(
        dataset=args.dataset,
        base=args.base,
        split=args.split,
        samples=args.samples,
        threshold=args.threshold,
        debug_first=args.debug_first
    )
    if not rows:
        print("No results to save.")
        return

    payload = {"summary": summary, "rows": [asdict(r) for r in rows]}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
