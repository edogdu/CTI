import re
import json
import logging

import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

from cti_analysis.models.triples import Entity, Triple, TripleBatch
from cti_analysis.llm_backend import LlamaCppBackend
from cti_analysis.ontology import TYPES, PREDS, SCHEMA, canon_type, canon_pred

repair_logger = logging.getLogger(__name__)

# Module-level backend (set by run_repair_ir from pipeline config)
_BACKEND: Optional[LlamaCppBackend] = None
_RE_LORA_ID: Optional[int] = None
LIM = 500


def _is_valid_name(x):
    """Check if x is a non-empty string within length limit."""
    return isinstance(x, str) and 0 < len(x.strip()) <= LIM


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================

def run_repair_ir(cfg, batches: List[TripleBatch], backend: Optional[LlamaCppBackend] = None) -> List[TripleBatch]:
    """Run repair logic on invalid triples from extraction.

    Reads batch.meta["invalid_triples"], runs det_fix -> validate -> Markov ->
    (optionally LLM) -> adds repaired triples back into the batch.
    """
    global _BACKEND, _RE_LORA_ID
    if backend is not None:
        _BACKEND = backend
    backend_cfg = getattr(cfg, "_backend_cfg", None)
    if _BACKEND is None and backend_cfg:
        _BACKEND = LlamaCppBackend(url=backend_cfg.url, timeout=backend_cfg.timeout)
    _RE_LORA_ID = backend_cfg.re_lora_id if backend_cfg else 1
    use_llm = getattr(cfg, "use_llm_repair", False)

    processed = []
    for batch in batches:
        invalids = (batch.meta or {}).get("invalid_triples", [])
        if not invalids:
            processed.append(batch)
            continue

        ctx_text = (batch.meta or {}).get("chunk_text", "")

        # Package invalids the way the repair loop expects
        entries = [{"triple": inv, "context": ctx_text} for inv in invalids]

        # Precompute Markov labels for this context
        try:
            sents, smooth_types = _smoothed_sentence_labels(ctx_text, entries)
        except Exception:
            sents, smooth_types = [], []

        repaired, bad = [], []
        for e in entries:
            t = e["triple"]

            # 1) Deterministic fix -> validate
            t1 = det_fix(t)
            ok, reasons = validate(t1)
            if ok:
                repaired.append(t1)
                continue

            # 2) Markov smoothing
            if sents:
                t2 = _apply_sentence_smoothing_to_triple(t1, ctx_text, sents, smooth_types)
                if t2 != t1:
                    ok2, reasons2 = validate(t2)
                    if ok2:
                        repaired.append(t2)
                        continue
                    t1, reasons = t2, reasons2

            # 3) Optional LLM repair
            if use_llm and should_call_llm(reasons):
                ptxt = prompt(t1, ctx_text)
                for c in ask_llm(ptxt):
                    c = det_fix(c)
                    ok3, _ = validate(c)
                    if ok3:
                        repaired.append(c)
                        t1 = None
                        break
                if t1 is None:
                    continue

            bad.append(t1)

        # Convert repaired dicts to Triple objects and merge with existing valids
        new_triples = list(batch.triples)
        for t in repaired:
            subj = t.get("subject", {})
            obj = t.get("object", {})
            new_triples.append(Triple(
                subject=Entity(
                    name=subj.get("name", "") if isinstance(subj, dict) else str(subj),
                    type=subj.get("type", "indicator") if isinstance(subj, dict) else "indicator",
                ),
                predicate=t.get("predicate", ""),
                object=Entity(
                    name=obj.get("name", "") if isinstance(obj, dict) else str(obj),
                    type=obj.get("type", "indicator") if isinstance(obj, dict) else "indicator",
                ),
                meta={"evidence": t.get("evidence"), "repair": "deterministic"},
            ))

        new_meta = (batch.meta or {}).copy()
        new_meta.pop("invalid_triples", None)
        new_meta["still_invalid"] = bad
        new_meta["repair_stats"] = {
            "input": len(invalids), "repaired": len(repaired), "still_bad": len(bad),
        }

        repair_logger.info(
            "[repair] %s: %d invalid -> %d repaired, %d still bad",
            batch.doc_id, len(invalids), len(repaired), len(bad),
        )
        processed.append(TripleBatch(
            doc_id=batch.doc_id, triples=new_triples,
            meta=new_meta, version=batch.version,
        ))

    return processed


# =============================================================================
# DETERMINISTIC FIX
# =============================================================================

def det_fix(t):
    """Normalize a triple dict: strip whitespace, canonicalize types and predicates."""
    t = dict(t) if isinstance(t, dict) else {"_raw": t}
    t.setdefault("subject", {})
    t.setdefault("object", {})
    for side in ("subject", "object"):
        name = t[side].get("name")
        if isinstance(name, str):
            name = name.strip()
            t[side]["name"] = name if name else "UNKNOWN"
        typ = t[side].get("type")
        if isinstance(typ, str):
            t[side]["type"] = canon_type(typ)
    if isinstance(t.get("predicate"), str):
        t["predicate"] = canon_pred(t["predicate"].strip())
    return t


# =============================================================================
# VALIDATION
# =============================================================================

def validate(t):
    """Validate a triple dict against the canonical ontology."""
    if not isinstance(t, dict):
        return False, ["not_object"]

    sub, obj, p = t.get("subject"), t.get("object"), t.get("predicate")
    errs = []

    # Structure checks
    if not isinstance(sub, dict):
        errs.append("subject_not_object")
    if not isinstance(obj, dict):
        errs.append("object_not_object")

    pp = canon_pred(p) if isinstance(p, str) else p
    st = canon_type(sub.get("type") if isinstance(sub, dict) else "")
    ot = canon_type(obj.get("type") if isinstance(obj, dict) else "")

    if not isinstance(pp, str) or pp not in PREDS:
        errs.append("predicate_invalid")
    if st not in TYPES:
        errs.append("subject.type_not_allowed")
    if ot not in TYPES:
        errs.append("object.type_not_allowed")

    # Domain/Range checks
    if isinstance(pp, str) and pp in SCHEMA:
        dom, rng = SCHEMA[pp]
        if st not in dom:
            errs.append("subject.type_domain")
        if ot not in rng:
            errs.append("object.type_range")

    # Name sanity checks
    for side in ("subject", "object"):
        ent = t.get(side) or {}
        nm = ent.get("name")
        if not _is_valid_name(nm):
            errs.append(f"{side}.name_bad")

    return (len(errs) == 0), errs


# =============================================================================
# LLM REPAIR GATING
# =============================================================================

# Reasons that are structural/schema and NOT fixable by paraphrasing
FATAL_FOR_LOOSE = {
    "predicate_invalid",
    "subject_not_object",
    "object_not_object",
}

# Reasons where a paraphrase/rename might help
FIXABLE_BY_LOOSE = {
    "subject.name_bad",
    "object.name_bad",
    "subject.type_not_allowed",
    "object.type_not_allowed",
    "subject.type_domain",
    "object.type_domain",
    "subject.type_range",
    "object.type_range",
}


def should_call_llm(reasons: list[str]) -> bool:
    """Return True only if there is at least one fixable reason and no fatal reasons."""
    if not reasons:
        return False
    if any(r in FATAL_FOR_LOOSE for r in reasons):
        return False
    return any(r in FIXABLE_BY_LOOSE for r in reasons)


# =============================================================================
# LLM REPAIR
# =============================================================================

def _extract_json_list_loose(text: str):
    """Parse the first/last JSON array from text; ignore code fences/chatter."""
    if not isinstance(text, str):
        return []
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
    left = t.find("[")
    right = t.rfind("]")
    if left == -1 or right == -1 or right <= left:
        return []
    try:
        return json.loads(t[left:right + 1])
    except Exception:
        return []


def prompt(t, ctx=""):
    """Build a repair prompt for the LLM."""
    return (
        "You are a CTI triple repair assistant. "
        "You may propose multiple alternative repaired triples that are valid under the STIX 2.1 ontology.\n"
        "\n"
        "Your task:\n"
        "- Repair or reinterpret the SUBJECT, PREDICATE, and OBJECT so each candidate triple is valid.\n"
        "- You may paraphrase entity names, adjust roles, or reinterpret relationships when supported by context.\n"
        "- You may produce multiple plausible mappings if the context allows more than one interpretation.\n"
        "- Expand ambiguous or incomplete names when the context implies a clearer entity.\n"
        "- You are allowed to correct ambiguous types by selecting the most contextually appropriate STIX 2.1 type.\n"
        "\n"
        "Strict constraints:\n"
        f"- Subject/object 'type' MUST come ONLY from: {sorted(TYPES)}\n"
        f"- Predicate MUST come ONLY from: {sorted(PREDS)}\n"
        "- Do NOT invent completely new entities not implied by the context.\n"
        "- Keep names realistic and aligned with the text.\n"
        "- Changes must remain grounded in the contextual meaning.\n"
        "\n"
        "Output rules:\n"
        "- Return ONLY a JSON array with **3 to 5 repaired triple candidates**.\n"
        "- Each element must follow the required schema.\n"
        "- No explanations, no commentary, no markdown.\n"
        "\n"
        "Required schema for each triple:\n"
        "{\n"
        '  "subject": {"name": "...", "type": "..."},\n'
        '  "predicate": "...",\n'
        '  "object": {"name": "...", "type": "..."},\n'
        '  "confidence": 0.0\n'
        "}\n"
        "\n"
        f"Context (truncated): {ctx[:800]}\n"
        f"Original triple:\n{json.dumps(t, ensure_ascii=False)}"
    )


def ask_llm(ptxt):
    """Call llama-server for repair; return list of triple dicts or []."""
    if _BACKEND is None:
        repair_logger.warning("No backend configured for LLM repair")
        return []
    try:
        txt = _BACKEND.generate(ptxt, lora_id=_RE_LORA_ID)
        arr = _extract_json_list_loose(txt)
        return arr if isinstance(arr, list) else []
    except Exception as e:
        repair_logger.warning("LLM call failed: %s", e)
        return []


# =============================================================================
# MARKOV SMOOTHING (repair-specific, operates on raw dicts)
# =============================================================================

def split_sentences(text: str):
    """Split text into sentences for Markov processing."""
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [p for p in parts if p]


class MarkovEntitySmoother:
    """HMM-based entity type smoother for the repair stage."""

    def __init__(self, states=None, alpha=0.1):
        self.states = states or sorted(TYPES) + ["other"]
        n = len(self.states)
        self.alpha = alpha
        self.P = np.full((n, n), alpha)
        np.fill_diagonal(self.P, 1.0)
        self.P /= self.P.sum(axis=1, keepdims=True)
        self.pi = np.full(n, 1.0 / n)

    def _sid(self, t):
        return self.states.index(t) if t in self.states else self.states.index("other")

    def viterbi(self, obs_types, conf=None):
        n = len(self.states)
        T = len(obs_types)
        conf = conf or [0.8] * T
        E = np.zeros((T, n))
        for t in range(T):
            pid = self._sid(obs_types[t])
            for s in range(n):
                E[t, s] = conf[t] if s == pid else (1 - conf[t]) / (n - 1)
        V = np.log(self.pi + 1e-9) + np.log(E[0] + 1e-9)
        B = np.zeros((T, n), dtype=int)
        for t in range(1, T):
            V_new = np.empty(n)
            for s in range(n):
                sc = V + np.log(self.P[:, s] + 1e-9) + np.log(E[t, s] + 1e-9)
                B[t, s] = int(np.argmax(sc))
                V_new[s] = np.max(sc)
            V = V_new
        path = [int(np.argmax(V))]
        for t in range(T - 1, 0, -1):
            path.append(B[t, path[-1]])
        path = path[::-1]
        return [self.states[p] for p in path]


def _build_sentence_majorities_for_context(triples, ctx_text: str):
    sents = split_sentences(ctx_text)
    if not sents:
        return [], [], []
    per_sent_types = [[] for _ in sents]
    for e in triples:
        t = e.get("triple") if isinstance(e, dict) else e
        if not isinstance(t, dict):
            continue
        sub = t.get("subject") or {}
        obj = t.get("object") or {}
        sname = sub.get("name") if isinstance(sub.get("name"), str) else ""
        oname = obj.get("name") if isinstance(obj.get("name"), str) else ""
        stype = canon_type(sub.get("type")) if isinstance(sub.get("type"), str) else None
        otype = canon_type(obj.get("type")) if isinstance(obj.get("type"), str) else None
        for i, sent in enumerate(sents):
            if sname and sname in sent and stype:
                per_sent_types[i].append(stype)
            if oname and oname in sent and otype:
                per_sent_types[i].append(otype)
    types, conf = [], []
    for lst in per_sent_types:
        if lst:
            top, cnt = Counter(lst).most_common(1)[0]
            types.append(top)
            conf.append(cnt / max(1, len(lst)))
        else:
            types.append("other")
            conf.append(0.5)
    return sents, types, conf


def _smoothed_sentence_labels(ctx_text: str, triples):
    sents, types, conf = _build_sentence_majorities_for_context(triples, ctx_text)
    if not sents:
        return [], []
    sm = MarkovEntitySmoother()
    smooth_types = sm.viterbi(types, conf)
    return sents, smooth_types


def _apply_sentence_smoothing_to_triple(triple, ctx_text: str, sents, smooth_types):
    if not sents or not smooth_types:
        return triple
    t = dict(triple)
    for side in ("subject", "object"):
        side_obj = dict(t.get(side) or {})
        name = side_obj.get("name") if isinstance(side_obj.get("name"), str) else ""
        cur_type = canon_type(side_obj.get("type")) if isinstance(side_obj.get("type"), str) else None
        if not name:
            continue
        idx = next((i for i, s in enumerate(sents) if name in s), None)
        if idx is None:
            continue
        candidate = smooth_types[idx]
        cand_canon = canon_type(candidate)
        if cur_type not in TYPES:
            if cand_canon in TYPES:
                side_obj["type"] = cand_canon
                t[side] = side_obj
    return t
