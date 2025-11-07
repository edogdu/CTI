# core/normalization.py
from typing import Tuple, Dict
from core.data_models import Triple
from core.data_models import canonicalize_entity as _canon_entity_type
from core.data_models import normalize_predicate as _norm_pred

def normalize_name(s: str) -> str:
    s = (s or "").strip().lower()
    return " ".join(s.split())

def to_triple_tuple(t: Triple) -> Tuple[str, str, str]:
    # subject/predicate/object downcased; predicate normalized to your set
    return (normalize_name(t.subject), _norm_pred(t.predicate), normalize_name(t.object))

def to_unified_dict(t: Triple) -> Dict:
    return {
        "subject": {"name": normalize_name(t.subject), "type": _canon_entity_type(t.subject_type)},
        "predicate": _norm_pred(t.predicate),
        "object": {"name": normalize_name(t.object), "type": _canon_entity_type(t.object_type)},
        "confidence": t.confidence,
        "evidence": t.evidence,
        "metadata": t.metadata or {},
    }
