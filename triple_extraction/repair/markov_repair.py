# repair/markov_repairer.py
from __future__ import annotations
from typing import List, Dict
import numpy as np
from collections import Counter

from core.pipeline import Repairer
from core.data_models import Triple, ExtractionResult

class MarkovRepairer(Repairer):
    """
    Deterministic normalization + light Markov smoothing over entity types.
    Works with core.data_models.Triple (strings, not nested dicts).
    """

    def __init__(self, config=None):
        super().__init__("markov_repairer", config)
        self.alpha = (config.ALPHA if hasattr(config, "ALPHA") else 0.1) if config else 0.1

        # States you had in your script
        self.states = [
            "malware","tool","attack-pattern","threat-actor","intrusion-set",
            "vulnerability","indicator","org","tactic","other"
        ]

        n = len(self.states)
        self.P = np.full((n, n), self.alpha)
        np.fill_diagonal(self.P, 1.0)
        self.P /= self.P.sum(axis=1, keepdims=True)

        # Aliases from your version
        self.TYPE_ALIASES = {
            "threatactor": "threat-actor", "threat actor": "threat-actor",
            "intrusionset": "intrusion-set", "intrusion set": "intrusion-set",
            "attackpattern": "attack-pattern", "attack pattern": "attack-pattern",
            "domain": "domain-name", "domainname": "domain-name",
            "ipv4": "ipv4-addr", "malware": "malware", "tool": "tool",
        }
        self.PRED_ALIASES = {
            "use":"uses","using":"uses","utilized":"uses","utilize":"uses",
            "target":"targets","targeting":"targets",
            "exploit":"exploits","exploiting":"exploits",
            "deliver":"delivers","delivering":"delivers",
            "drop":"drops","dropping":"drops"
        }

        # Allow list (tweak as needed)
        self.ALLOWED_PREDICATES = {
            "uses","targets","exploits","communicates-with","delivers","drops","compromises","executes"
        }

    # ----- Pipeline hook -----
    def process_triples(self, triples: List[Triple]) -> List[Triple]:
        if not triples:
            return triples

        # 1) deterministic normalization
        det = [self._deterministic_fix(t) for t in triples]

        # 2) Markov smoothing for entity types using each triple's evidence as context
        # We infer a "sentence label" per triple from local majority; then smooth.
        smoothed = self._smooth_types_by_context(det)

        return smoothed

    # ----- Deterministic normalization -----
    def _deterministic_fix(self, t: Triple) -> Triple:
        # names
        t.subject = (t.subject or "").strip() or "UNKNOWN"
        t.object  = (t.object or "").strip() or "UNKNOWN"

        # types
        t.subject_type = self._canon_type((t.subject_type or "").strip() or "other")
        t.object_type  = self._canon_type((t.object_type  or "").strip() or "other")

        # predicate
        p = (t.predicate or "").strip().lower().replace(" ", "_")
        p = self.PRED_ALIASES.get(p, p)
        # prefer hyphen for “communicates-with” style if you use that in eval;
        # keep simple verbs otherwise
        p = p.replace("_", "-") if p == "communicates_with" else p
        t.predicate = p

        # guardrail for empty predicate
        if not t.predicate:
            t.predicate = "uses"
        return t

    def _canon_type(self, typ: str) -> str:
        t2 = typ.strip().lower().replace(" ", "").replace("_", "-")
        return self.TYPE_ALIASES.get(t2, t2)

    # ----- Markov smoothing (lightweight) -----
    def _smooth_types_by_context(self, triples: List[Triple]) -> List[Triple]:
        # Group by evidence (your Triple has an `evidence` string)
        buckets: Dict[str, List[int]] = {}
        for i, t in enumerate(triples):
            key = (t.evidence or "").strip()
            buckets.setdefault(key, []).append(i)

        for ev, idxs in buckets.items():
            if len(idxs) == 1:
                continue  # nothing to smooth

            # majority vote per side
            subj_types = [triples[i].subject_type for i in idxs]
            obj_types  = [triples[i].object_type  for i in idxs]
            subj_major, subj_conf = self._majority(subj_types)
            obj_major,  obj_conf  = self._majority(obj_types)

            # Light “Viterbi-like” selection: only relabel if current is weak/generic
            for i in idxs:
                t = triples[i]
                if t.subject_type in {"other","org","tactic","unknown"} and subj_conf >= 0.6:
                    t.subject_type = subj_major
                if t.object_type in {"other","org","tactic","unknown"} and obj_conf >= 0.6:
                    t.object_type = obj_major

                # keep predicates within allowed vocab if possible
                if t.predicate not in self.ALLOWED_PREDICATES:
                    # pick the closest allowed by simple heuristics
                    t.predicate = self._nearest_allowed_pred(t.predicate)

        return triples

    def _majority(self, labels: List[str]) -> tuple[str, float]:
        c = Counter([ (x or "other") for x in labels ])
        lab, cnt = c.most_common(1)[0]
        return lab, cnt / max(1, len(labels))

    def _nearest_allowed_pred(self, p: str) -> str:
        if not self.ALLOWED_PREDICATES:
            return "uses"
        # exact alias first
        if p in self.PRED_ALIASES:
            p = self.PRED_ALIASES[p]
            if p in self.ALLOWED_PREDICATES:
                return p
        # fallback: prefix/substring match
        for cand in self.ALLOWED_PREDICATES:
            if p.startswith(cand) or cand.startswith(p):
                return cand
        # default
        return "uses"
