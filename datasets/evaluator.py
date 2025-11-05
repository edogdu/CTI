# cti/datasets/eval.py
from __future__ import annotations
from typing import List, Tuple, Dict

Triple = Tuple[str, str, str]

def prf1(true_triples: List[Triple], pred_triples: List[Triple]) -> Dict[str, float]:
    T = set(true_triples)
    P = set(pred_triples)
    tp = len(T & P); fp = len(P - T); fn = len(T - P)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "gold": len(T), "pred": len(P)}

def evaluate_samples(samples: List[Dict], extract_fn) -> Dict:
    per_doc = []
    macro = {"precision":0,"recall":0,"f1":0}
    for s in samples:
        gold = [tuple(t) for t in s.get("ground_truth_triples", [])]
        pred = extract_fn(s["text"])
        res = prf1(gold, pred)
        per_doc.append({"id": s["id"], "dataset": s.get("dataset"), **res})
        macro["precision"] += res["precision"]; macro["recall"] += res["recall"]; macro["f1"] += res["f1"]
    n = max(1, len(per_doc))
    macro = {k: v/n for k,v in macro.items()}
    # micro over all triples
    all_gold = [tuple(t) for s in samples for t in s.get("ground_truth_triples", [])]
    all_pred = [p for s in samples for p in extract_fn(s["text"])]
    micro = prf1(all_gold, all_pred)
    return {"macro": macro, "micro": micro, "per_doc": per_doc}
