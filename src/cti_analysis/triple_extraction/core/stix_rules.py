# core/stix_rules.py
import re, json
from collections import Counter
from typing import Tuple, List, Dict

# ==== Vocab pulled from your script ====
TYPES = {
    "threat-actor","intrusion-set","campaign","malware","tool","infrastructure","attack-pattern",
    "course-of-action","indicator","vulnerability","software","product","organization","identity",
    "sector","country","location","city","ipv4-addr","ipv6-addr","domain-name","url","file",
    "email-addr","observed-data","report","sighting","x-mitre-data-source","x-mitre-data-component",
    "observable","user-account","directory","autonomous-system","date","malware-analysis"
}
PREDS = {
    "uses","targets","attributed_to","exploits","affects","mitigates","indicates","detects","based_on",
    "derived_from","observed_in","communicates_with","hosts","delivers","drops","located_at",
    "uses_technique","subtechnique_of","revoked_by","duplicate_of","originates_from","impacts",
    "resolves_to","downloads_from","writes_to","reads_from","emails_to","observed_on",
    "analysis_of","characterizes","sighting_of","sighted_at","sighted_by","variant_of"
}
SCHEMA: Dict[str, Tuple[set, set]] = {
    "uses": ({"threat-actor","intrusion-set","campaign"}, {"malware","tool","infrastructure","attack-pattern"}),
    "targets": ({"threat-actor","intrusion-set","campaign","malware","attack-pattern"}, {"identity","organization","sector","location","country"}),
    "attributed_to": ({"campaign","intrusion-set"}, {"threat-actor"}),
    "exploits": ({"threat-actor","malware","tool"}, {"vulnerability"}),
    "uses_technique": ({"threat-actor","malware","tool","campaign"}, {"attack-pattern"}),
    "originates_from": ({"threat-actor","intrusion-set"}, {"location","country"}),
    "based_on": ({"indicator"}, {"file","url","domain-name","ipv4-addr","ipv6-addr","email-addr","observable","artifact"}),
    "derived_from": ({"indicator"}, {"report","observed-data","artifact","log"}),
    "detects": ({"indicator","tool","x-mitre-data-source"}, {"malware","campaign","intrusion-set","infrastructure","tool","vulnerability"}),
    "indicates": ({"indicator"}, {"malware","campaign","intrusion-set","attack-pattern","vulnerability","organization","sector","location"}),
    "resolves_to": ({"domain-name"}, {"ipv4-addr","ipv6-addr"}),
    "hosts": ({"infrastructure"}, {"domain-name","ipv4-addr","ipv6-addr","url","file"}),
    "delivers": ({"infrastructure","malware","tool"}, {"malware","tool","file","url"}),
    "drops": ({"malware","tool"}, {"malware","tool","file"}),
    "located_at": ({"infrastructure","organization","identity"}, {"location","country","city"}),
    "observed_on": ({"threat-actor","malware","tool","infrastructure","indicator","attack-pattern"}, {"date"}),
    "impacts": ({"threat-actor","malware","attack-pattern"}, {"identity","organization","sector"}),
}

TYPE_ALIASES = {
    "threatactor":"threat-actor","threat actor":"threat-actor",
    "intrusionset":"intrusion-set","intrusion set":"intrusion-set",
    "attackpattern":"attack-pattern","attack pattern":"attack-pattern",
    "courseofaction":"course-of-action","course of action":"course-of-action",
    "domain":"domain-name","domainname":"domain-name",
    "ipv4":"ipv4-addr","ipv4addr":"ipv4-addr","ipv4-address":"ipv4-addr",
    "ipv6":"ipv6-addr","ipv6addr":"ipv6-addr","ipv6-address":"ipv6-addr",
    "email":"email-addr","emailaddress":"email-addr","email addr":"email-addr",
    "useraccount":"user-account","user account":"user-account",
    "autonomoussystem":"autonomous-system","autonomous system":"autonomous-system",
    "observeddata":"observed-data","observed data":"observed-data",
    "malwareanalysis":"malware-analysis",
}
PRED_ALIASES = {
    "communicateswith":"communicates_with",
    "usestechnique":"uses_technique",
    "subtechniqueof":"subtechnique_of",
    "revokedby":"revoked_by",
    "duplicateof":"duplicate_of",
    "originatesfrom":"originates_from",
    "resolvesto":"resolves_to",
    "downloadsfrom":"downloads_from",
    "writes_to":"writes_to","reads_from":"reads_from","emails_to":"emails_to",
    "observedon":"observed_on","analysisof":"analysis_of",
    "sightingof":"sighting_of","sightedat":"sighted_at","sightedby":"sighted_by",
    "variantof":"variant_of",
}

def canon_type(t: str) -> str:
    if not isinstance(t, str): return t
    t2 = t.strip().lower().replace(" ", "").replace("_", "-")
    return TYPE_ALIASES.get(t2, t2)

def canon_pred(p: str) -> str:
    if not isinstance(p, str): return p
    s = p.strip().lower().replace(" ", "_")
    return PRED_ALIASES.get(s, s)

def _ne(x) -> bool:
    return isinstance(x, str) and 0 < len(x.strip()) <= 500

def det_fix(triple: dict) -> dict:
    """Deterministic cleanup: strip, canonicalize names and types, normalize predicate."""
    t = dict(triple) if isinstance(triple, dict) else {"_raw": triple}
    t.setdefault("subject", {}); t.setdefault("object", {})
    for side in ("subject","object"):
        name = t[side].get("name")
        if isinstance(name, str):
            t[side]["name"] = name.strip() or "UNKNOWN"
        typ = t[side].get("type")
        if isinstance(typ, str):
            t[side]["type"] = canon_type(typ)
    if isinstance(t.get("predicate"), str):
        t["predicate"] = canon_pred(t["predicate"].strip())
    return t

def validate(t: dict) -> Tuple[bool, List[str]]:
    """Schema checks, domain/range, and name sanity."""
    errs = []
    sub, obj, p = t.get("subject"), t.get("object"), t.get("predicate")
    if not isinstance(sub, dict): errs.append("subject_not_object")
    if not isinstance(obj, dict): errs.append("object_not_object")
    pp = canon_pred(p) if isinstance(p, str) else p
    st = canon_type(sub.get("type") if isinstance(sub, dict) else "")
    ot = canon_type(obj.get("type") if isinstance(obj, dict) else "")
    if not isinstance(pp, str) or pp not in PREDS: errs.append("predicate_invalid")
    if st not in TYPES: errs.append("subject.type_not_allowed")
    if ot not in TYPES: errs.append("object.type_not_allowed")
    if isinstance(pp, str) and pp in SCHEMA:
        dom, rng = SCHEMA[pp]
        if st not in dom: errs.append("subject.type_domain")
        if ot not in rng: errs.append("object.type_range")
    if not _ne(sub.get("name") if isinstance(sub, dict) else ""): errs.append("subject.name_bad")
    if not _ne(obj.get("name") if isinstance(obj, dict) else ""): errs.append("object.name_bad")
    return (len(errs) == 0), errs

# ===== Sentence-level Markov smoothing (entities) =====
def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [p for p in parts if p]

class MarkovEntitySmoother:
    def __init__(self, states=None, alpha=0.1):
        import numpy as np
        self.np = np
        self.states = states or [
            "malware","tool","attack-pattern","threat-actor","intrusion-set",
            "vulnerability","indicator","org","tactic","other"
        ]
        n = len(self.states); self.alpha = alpha
        self.P = np.full((n, n), alpha)
        np.fill_diagonal(self.P, 1.0)
        self.P /= self.P.sum(axis=1, keepdims=True)
        self.pi = np.full(n, 1.0 / n)

    def _sid(self, t):
        return self.states.index(t) if t in self.states else self.states.index("other")

    def viterbi(self, obs_types, conf=None):
        np = self.np
        n = len(self.states); T = len(obs_types)
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

def _build_sentence_majorities_for_context(triples: List[dict], ctx_text: str):
    """Infer a type label per sentence by majority vote of entities appearing in that sentence."""
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

def smoothed_sentence_labels(ctx_text: str, triples: List[dict]):
    sents, types, conf = _build_sentence_majorities_for_context(triples, ctx_text)
    if not sents:
        return [], []
    sm = MarkovEntitySmoother()
    smooth_types = sm.viterbi(types, conf)
    return sents, smooth_types

def apply_sentence_smoothing_to_triple(triple: dict, ctx_text: str, sents: List[str], smooth_types: List[str]):
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
        # only relax invalid or generic labels
        if (cur_type not in TYPES) or (cur_type in {"org","tactic","other"}):
            if cand_canon in TYPES:
                side_obj["type"] = cand_canon
                t[side] = side_obj
    return t
