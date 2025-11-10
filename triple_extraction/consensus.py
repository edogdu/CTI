import re
from collections import defaultdict
from difflib import SequenceMatcher

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _key(t):
    s = t.get("subject", {}) or {}
    o = t.get("object", {}) or {}
    p = _norm(t.get("predicate") or "")
    return (_norm(s.get("name","")), _norm(s.get("type") or ""),
            p,
            _norm(o.get("name","")), _norm(o.get("type") or ""))

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def _quote_overlap(a: str, b: str) -> bool:
    aW = set(_norm(a).lower().split()); bW = set(_norm(b).lower().split())
    if not aW or not bW: return False
    return len(aW & bW) >= max(3, min(len(aW), len(bW)) // 2)

def consensus_filter(candidate_lists,
                     allowed_types, allowed_preds,
                     m=2, tau_name=0.90):
    """Keep only triples that reach quorum m across multiple prompt outputs."""
    # 1) whitelist + flatten with type guards
    all_triples = []
    for triples in (candidate_lists or []):
        for t in (triples or []):
            if not isinstance(t, dict):
                continue
            s = t.get("subject") or {}
            o = t.get("object") or {}
            p = _norm(t.get("predicate") or "")
            st = _norm(s.get("type") or ""); ot = _norm(o.get("type") or "")
            if st in allowed_types and ot in allowed_types and p in allowed_preds:
                all_triples.append(t)

    if not all_triples:
        return []

    # 2) exact buckets
    buckets = defaultdict(list)
    for t in all_triples:
        buckets[_key(t)].append(t)

    keys = list(buckets.keys())
    visited = set()
    accepted = []

    # 3) merge near-dupes within same (types + predicate)
    for i, ki in enumerate(keys):
        if ki in visited: continue
        group = list(buckets[ki])
        si,ti,pi,oi,ui = ki

        for j, kj in enumerate(keys):
            if j <= i or kj in visited: 
                continue
            sj,tj,pj,oj,uj = kj
            if pi != pj or ti != tj or ui != uj:
                continue
            if _sim(si, sj) >= tau_name and _sim(oi, oj) >= tau_name:
                qi = (group[0].get("evidence") or {}).get("quote","")
                qj = (buckets[kj][0].get("evidence") or {}).get("quote","")
                if _quote_overlap(qi, qj):
                    group += buckets[kj]
                    visited.add(kj)

        # 4) quorum
        if len(group) >= max(2, m):
            rep = max(group, key=lambda t: len(((t.get("evidence") or {}).get("quote") or "")))
            accepted.append(rep)

        visited.add(ki)

    return accepted
