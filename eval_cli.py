from __future__ import annotations
import argparse, json, os, sys, time, re, tempfile, pathlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
import html


# ===== import your runner =====
try:
    from cti_analysis.triple_extraction.extraction import CyberTripleExtractor, ExtractionOptions
except Exception:
    from cti_analysis.triple_extraction.extraction import CyberTripleExtractor, ExtractionOptions  # fallback

from cti_analysis.triple_extraction.evaluation.notebook_eval import (
    cti_hal_ground_truth,
    evaluate_sample_vector_similarity,
)

try:
    from cti_analysis.triple_extraction.evaluation.loaders import (
        load_cti_hal_samples,
        load_anno_ctr_samples,
        load_dnrti_samples,
    )
except Exception as e:
    print("ERROR importing loaders:", e)
    sys.exit(1)

# ===== defaults for press-Run experience =====
DEFAULTS = {
    "dataset": "anno-ctr",   # cti-hal, anno-ctr, dnrti
    "base": "./data",          # root folder where dataset loaders expect data
    "samples": 8,
    "split": "auto",
    "threshold": 0.55,
    "out": "output/eval_results.json",
    "debug_first": 1,
    "normalize": True,
    # ExtractionOptions
    "run_sentence": True,
    "consensus_on_sentence": False,
    "run_semantic": False,
    "consensus_on_chunk": False,
    "combine_stages": False,
    "consensus_passes": 3,
    "consensus_m": 2,
}

# optional override file next to this script
LOCAL_CFG_PATH = "eval.local.json"

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

# -------- normalization helpers --------
_PRED_MAP = {"executes": "uses", "leverages": "uses", "employs": "uses"}
_TID_RX = re.compile(r"^t(\d{4,5})$", re.I)

def _norm_pred(p: str) -> str:
    p = (p or "").strip().lower()
    return _PRED_MAP.get(p, p)

def _norm_ent(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[\s\-_]+", " ", s)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return s

def _norm_tid(s: str) -> str:
    s = (s or "").strip().lower()
    m = _TID_RX.match(s)
    return f"t{m.group(1)}" if m else s

def _normalize_triple(t: Tuple[str, str, str]) -> Tuple[str, str, str]:
    sub, pred, obj = t
    return (_norm_ent(sub), _norm_pred(pred), _norm_tid(obj))

def _normalize_list(triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    if not triples:
        return []
    seen, out = set(), []
    for t in map(_normalize_triple, triples):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

# -------- dataset loader --------
def _load_dataset(dataset: str, base: str, split: str, samples: int) -> List[Dict[str, Any]]:
    if dataset == "cti-hal":
        return load_cti_hal_samples(base, num_samples=samples)
    elif dataset == "anno-ctr":
        sp = split if split and split.lower() != "auto" else "test"
        data = load_anno_ctr_samples(base, num_samples=samples, split=sp) or []
        if not data and sp != "dev":
            data = load_anno_ctr_samples(base, num_samples=samples, split="dev") or []
        if not data and sp != "train":
            data = load_anno_ctr_samples(base, num_samples=samples, split="train") or []
        return data
    elif dataset == "dnrti":
        sp = split if split and split.lower() != "auto" else "test"
        data = load_dnrti_samples(base, num_samples=samples, split=sp) or []
        if not data and sp != "dev":
            data = load_dnrti_samples(base, num_samples=samples, split="dev") or []
        if not data and sp != "train":
            data = load_dnrti_samples(base, num_samples=samples, split="train") or []
        return data
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

# -------- adapter to CyberTripleExtractor --------
def _dict_triple_to_tuple(t: Dict[str, Any]) -> Tuple[str, str, str] | None:
    try:
        sub = (t.get("subject") or {}).get("name") or (t.get("subject") or {}).get("value")
        obj = (t.get("object")  or {}).get("name") or (t.get("object")  or {}).get("value")
        pred = t.get("predicate")
        if isinstance(sub, str) and isinstance(obj, str) and isinstance(pred, str):
            return (sub, pred, obj)
    except Exception:
        pass
    return None

def _extract_with_runner(text: str,
                         extractor: CyberTripleExtractor,
                         options: ExtractionOptions) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    # Docling does not allow .txt; give it a tiny HTML document
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "sample.html"
        p.write_text(
            f"<html><head><meta charset='utf-8'></head>"
            f"<body><p>{html.escape(text)}</p></body></html>",
            encoding="utf-8"
        )
        extractor.file_path = str(p)
        chunk_results = extractor.run(options)

        dict_triples = getattr(extractor, "valid_triples", None)
        if dict_triples is None:
            dict_triples = []
            for _, _, _, final_triples in (chunk_results or []):
                dict_triples.extend(final_triples or [])

        for dt in dict_triples:
            tup = _dict_triple_to_tuple(dt) if isinstance(dt, dict) else None
            if tup:
                triples.append(tup)
    return triples


def _smoke(extractor: CyberTripleExtractor, options: ExtractionOptions):
    sample = "APT29 used spearphishing to deploy Cobalt Strike."
    t0 = time.time()
    pred = _extract_with_runner(sample, extractor, options)
    print(f"[SMOKE] extracted={pred} in {time.time()-t0:.2f}s")

# -------- core eval --------
def run_eval(dataset: str, base: str, split: str, samples: int,
             threshold: float, debug_first: int, normalize: bool,
             options: ExtractionOptions):
    data = _load_dataset(dataset, base, split, samples)
    if not data:
        print("No samples loaded. Check dataset path.")
        return {}, []
    extractor = CyberTripleExtractor(file_path="", document_id="EVAL_RUN", model_name="gemma2:9b")
    print("\n" + "=" * 78)
    print(f" RUN EVALUATION via CyberTripleExtractor: {dataset} ({len(data)} samples)")
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
        pred = _extract_with_runner(text, extractor, options)
        dt = time.time() - t0

        pred_in = _normalize_list(pred) if normalize else (pred or [])
        gt_in   = _normalize_list(gt or []) if normalize else (gt or [])

        if i < max(0, int(debug_first)):
            print(f"\n[DEBUG] {sid}")
            print("  pred:", pred_in[:6])
            print("  gt  :", gt_in[:6])

        metrics = evaluate_sample_vector_similarity(
            extracted=pred_in,
            ground=gt_in,
            threshold=threshold,
            debug=False
        )

        row = Row(
            sample_id=sid,
            extracted_count=len(pred_in),
            gt_count=len(gt_in),
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

    if rows:
        macro_p = sum(r.precision for r in rows)/len(rows)
        macro_r = sum(r.recall for r in rows)/len(rows)
        macro_f = sum(r.f1 for r in rows)/len(rows)
    else:
        macro_p = macro_r = macro_f = 0.0

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
        "runner": "CyberTripleExtractor",
        "threshold": threshold,
        "normalize": normalize,
    }
    return summary, rows

def _merge_defaults_with_cli() -> argparse.Namespace:
    defaults = DEFAULTS.copy()
    if os.path.exists(LOCAL_CFG_PATH):
        try:
            with open(LOCAL_CFG_PATH, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            defaults.update({k: overrides[k] for k in overrides if k in defaults})
            print(f"[INFO] Loaded overrides from {LOCAL_CFG_PATH}")
        except Exception as e:
            print(f"[WARN] Could not parse {LOCAL_CFG_PATH}: {e}")

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--dataset", choices=["cti-hal", "anno-ctr", "dnrti"])
    ap.add_argument("--base")
    ap.add_argument("--samples", type=int)
    ap.add_argument("--split")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--out")
    ap.add_argument("--debug-first", type=int)
    ap.add_argument("--normalize", type=lambda x: str(x).lower() not in {"0","false","no"})
    ap.add_argument("--run-sentence", type=lambda x: str(x).lower() not in {"0","false","no"})
    ap.add_argument("--consensus-on-sentence", type=lambda x: str(x).lower() not in {"0","false","no"})
    ap.add_argument("--run-semantic", type=lambda x: str(x).lower() not in {"0","false","no"})
    ap.add_argument("--consensus-on-chunk", type=lambda x: str(x).lower() not in {"0","false","no"})
    ap.add_argument("--combine-stages", type=lambda x: str(x).lower() not in {"0","false","no"})
    ap.add_argument("--consensus-passes", type=int)
    ap.add_argument("--consensus-m", type=int)
    # parse known, keep anything missing as None
    args, _ = ap.parse_known_args()
    for k, v in defaults.items():
        if getattr(args, k.replace("-", "_"), None) is None:
            setattr(args, k.replace("-", "_"), v)
    return args

def main():
    args = _merge_defaults_with_cli()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    options = ExtractionOptions(
        run_sentence=bool(args.run_sentence),
        consensus_on_sentence=bool(args.consensus_on_sentence),
        run_semantic=bool(args.run_semantic),
        consensus_on_chunk=bool(args.consensus_on_chunk),
        combine_stages=bool(args.combine_stages),
        consensus_passes=max(1, int(args.consensus_passes)),
        consensus_m=max(1, int(args.consensus_m)),
    )
    _smoke(CyberTripleExtractor(file_path="", document_id="EVAL_RUN", model_name="gemma2:9b"), options)
    summary, rows = run_eval(
        dataset=args.dataset,
        base=args.base,
        split=args.split,
        samples=args.samples,
        threshold=args.threshold,
        debug_first=args.debug_first,
        normalize=args.normalize,
        options=options,
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