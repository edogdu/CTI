"""Shared matching and metric utilities for NER and RE evaluation."""
from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def norm_name(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = text.replace("the ", "").replace("a ", "")
    return " ".join(text.split())


def norm_pred(pred: str) -> str:
    """Lowercase + strip punctuation/spaces so 'hasAttackTime' == 'hasattacktime'."""
    return pred.lower().replace("-", "").replace("_", "").replace(" ", "").strip()


# ---------------------------------------------------------------------------
# Fuzzy entity matching
# ---------------------------------------------------------------------------

def _entity_match(a: str, b: str, threshold: float = 0.70) -> bool:
    a, b = norm_name(a), norm_name(b)
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 3 and short in long:
        return True
    return SequenceMatcher(None, a, b).ratio() >= threshold


# ---------------------------------------------------------------------------
# Gold data parsing
# ---------------------------------------------------------------------------

def parse_gold_entities(doc: Dict) -> List[Tuple[str, str]]:
    """Return list of (name, type) from DNRTI doc.
    Entity format: [start, end, name, type]
    """
    out = []
    for ent in doc.get("entities", []):
        if isinstance(ent, list) and len(ent) >= 4:
            name = str(ent[2]).strip()
            etype = str(ent[3]).strip()
            if name:
                out.append((name, etype))
    return out


def parse_gold_triples(doc: Dict) -> List[Tuple[str, str, str]]:
    """Return list of (subject_name, predicate, object_name) from DNRTI doc.
    Relation format: [predicate, head_idx, tail_idx]
      head_idx = object index, tail_idx = subject index
    """
    entities = doc.get("entities", [])
    entity_map = {}
    for i, ent in enumerate(entities):
        if isinstance(ent, list) and len(ent) >= 3:
            entity_map[i] = str(ent[2]).strip()

    triples = []
    for rel in doc.get("relations", []):
        if not isinstance(rel, list) or len(rel) < 3:
            continue
        pred = str(rel[0]).strip()
        if norm_pred(pred) in ("norelation", "norelation", "none"):
            continue
        head_idx, tail_idx = rel[1], rel[2]
        obj_name = entity_map.get(head_idx)
        subj_name = entity_map.get(tail_idx)
        if subj_name and obj_name:
            triples.append((subj_name, norm_pred(pred), obj_name))
    return triples


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_ner_metrics(
    gold: List[Tuple[str, str]],
    pred: List[Tuple[str, str]],
) -> Dict:
    """
    Compare predicted entities to gold.
    Matching: fuzzy name + exact type.
    Returns per-type counts and aggregate TP/FP/FN.
    """
    per_type: Dict[str, Dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    gold_by_type: Dict[str, List[str]] = defaultdict(list)
    for name, etype in gold:
        gold_by_type[etype].append(name)

    pred_by_type: Dict[str, List[str]] = defaultdict(list)
    for name, etype in pred:
        pred_by_type[etype].append(name)

    all_types = set(gold_by_type) | set(pred_by_type)
    for etype in all_types:
        g_names = list(gold_by_type[etype])
        p_names = list(pred_by_type[etype])
        matched_g: Set[int] = set()
        matched_p: Set[int] = set()
        for pi, pn in enumerate(p_names):
            for gi, gn in enumerate(g_names):
                if gi not in matched_g and _entity_match(pn, gn):
                    matched_g.add(gi)
                    matched_p.add(pi)
                    break
        tp = len(matched_g)
        per_type[etype]["tp"] += tp
        per_type[etype]["fp"] += len(p_names) - len(matched_p)
        per_type[etype]["fn"] += len(g_names) - tp

    total_tp = sum(v["tp"] for v in per_type.values())
    total_fp = sum(v["fp"] for v in per_type.values())
    total_fn = sum(v["fn"] for v in per_type.values())
    return {"per_type": dict(per_type), "tp": total_tp, "fp": total_fp, "fn": total_fn}


def compute_re_metrics(
    gold: List[Tuple[str, str, str]],
    pred: List[Tuple[str, str, str]],
) -> Dict:
    """
    Compare predicted triples to gold.
    Matching: fuzzy subject + fuzzy object + exact predicate.
    Returns per-predicate counts and aggregate TP/FP/FN.
    """
    per_pred: Dict[str, Dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    def _triple_match(g: Tuple, p: Tuple) -> bool:
        gs, gp, go = g
        ps, pp, po = p
        return gp == pp and _entity_match(gs, ps) and _entity_match(go, po)

    gold_by_pred: Dict[str, List] = defaultdict(list)
    for t in gold:
        gold_by_pred[t[1]].append(t)

    pred_by_pred: Dict[str, List] = defaultdict(list)
    for t in pred:
        pred_by_pred[t[1]].append(t)

    all_preds = set(gold_by_pred) | set(pred_by_pred)
    for p in all_preds:
        g_list = gold_by_pred[p]
        p_list = pred_by_pred[p]
        matched_g: Set[int] = set()
        matched_p: Set[int] = set()
        for pi, pt in enumerate(p_list):
            for gi, gt in enumerate(g_list):
                if gi not in matched_g and _triple_match(gt, pt):
                    matched_g.add(gi)
                    matched_p.add(pi)
                    break
        tp = len(matched_g)
        per_pred[p]["tp"] += tp
        per_pred[p]["fp"] += len(p_list) - len(matched_p)
        per_pred[p]["fn"] += len(g_list) - tp

    total_tp = sum(v["tp"] for v in per_pred.values())
    total_fp = sum(v["fp"] for v in per_pred.values())
    total_fn = sum(v["fn"] for v in per_pred.values())
    return {"per_pred": dict(per_pred), "tp": total_tp, "fp": total_fp, "fn": total_fn}


def aggregate_metrics(results: List[Dict]) -> Dict:
    """Aggregate per-doc metric dicts into micro + macro summary."""
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)

    def _f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0, p, r

    micro_f1, micro_p, micro_r = _f1(total_tp, total_fp, total_fn)

    doc_f1s, doc_ps, doc_rs = [], [], []
    for r in results:
        f, p, rec = _f1(r["tp"], r["fp"], r["fn"])
        doc_f1s.append(f)
        doc_ps.append(p)
        doc_rs.append(rec)

    macro_p = sum(doc_ps) / len(doc_ps) if doc_ps else 0.0
    macro_r = sum(doc_rs) / len(doc_rs) if doc_rs else 0.0
    macro_f1 = sum(doc_f1s) / len(doc_f1s) if doc_f1s else 0.0

    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1,
                  "tp": total_tp, "fp": total_fp, "fn": total_fn},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
    }
