"""Skew Zero Forcing (SZF) post-processor for knowledge graphs.

Propagates confidence through the graph structure:
1. High-confidence entities "force" connected relations.
2. High-confidence relations "force" connected entities.
3. Iterate until convergence or max iterations.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple

from cti_analysis.ontology import SCHEMA
from .markov import SmoothableEntity


class TripleWithTypes:
    """Relation triple with confidence for SZF processing."""
    __slots__ = ("subject", "subject_type", "predicate",
                 "object", "object_type", "confidence", "evidence")

    def __init__(self, subject: str, subject_type: str, predicate: str,
                 object: str, object_type: str, confidence: float = 0.8,
                 evidence: str = ""):
        self.subject = subject
        self.subject_type = subject_type
        self.predicate = predicate
        self.object = object
        self.object_type = object_type
        self.confidence = confidence
        self.evidence = evidence

    def __hash__(self):
        return hash((self.subject.lower(), self.predicate, self.object.lower()))

    def __eq__(self, other):
        if not isinstance(other, TripleWithTypes):
            return False
        return (self.subject.lower() == other.subject.lower() and
                self.predicate == other.predicate and
                self.object.lower() == other.object.lower())


def _normalize_name(text: str) -> str:
    """Lightweight normalization for entity name matching."""
    return text.strip().lower()


def _is_compatible(subj_type: str, predicate: str, obj_type: str) -> bool:
    """Check if a triple's types are semantically compatible per SCHEMA."""
    if predicate in SCHEMA:
        valid_subj, valid_obj = SCHEMA[predicate]
        return subj_type in valid_subj and obj_type in valid_obj
    # No schema constraint → allow
    return True


class GraphPostProcessorSZF:
    """SZF confidence propagator.

    Args:
        entity_threshold: Min confidence to consider an entity "forced".
        relation_threshold: Min confidence to consider a relation "forced".
        max_iters: Maximum propagation iterations.
    """

    def __init__(self, entity_threshold: float = 0.80,
                 relation_threshold: float = 0.75,
                 max_iters: int = 50):
        self.entity_threshold = entity_threshold
        self.relation_threshold = relation_threshold
        self.max_iters = max_iters

    def run(self, entities: List[SmoothableEntity],
            triples: List[TripleWithTypes]) -> Tuple[List[SmoothableEntity], List[TripleWithTypes]]:
        """Run SZF propagation on entities and triples."""
        if not entities or not triples:
            return entities, triples

        E = {i: e for i, e in enumerate(entities)}
        R = {j: t for j, t in enumerate(triples)}

        forced_entities = {i for i, e in E.items()
                          if e.confidence >= self.entity_threshold}
        forced_relations = {j for j, t in R.items()
                           if t.confidence >= self.relation_threshold
                           and _is_compatible(t.subject_type, t.predicate, t.object_type)}

        # Build adjacency
        name_to_eids: dict[str, set[int]] = defaultdict(set)
        for i, e in E.items():
            name_to_eids[_normalize_name(e.text)].add(i)

        ent_to_rels: dict[int, set[int]] = defaultdict(set)
        rel_to_ents: dict[int, set[int]] = defaultdict(set)

        for j, t in R.items():
            for eid in name_to_eids.get(_normalize_name(t.subject), ()):
                ent_to_rels[eid].add(j)
                rel_to_ents[j].add(eid)
            for eid in name_to_eids.get(_normalize_name(t.object), ()):
                ent_to_rels[eid].add(j)
                rel_to_ents[j].add(eid)

        # Propagation loop
        changed = True
        iteration = 0
        while changed and iteration < self.max_iters:
            changed = False
            iteration += 1

            for j, t in R.items():
                if j not in forced_relations:
                    if (any(eid in forced_entities for eid in rel_to_ents[j])
                            and _is_compatible(t.subject_type, t.predicate, t.object_type)):
                        forced_relations.add(j)
                        changed = True

            for i in E:
                if i not in forced_entities:
                    if any(rid in forced_relations for rid in ent_to_rels[i]):
                        forced_entities.add(i)
                        changed = True

        # Update confidences
        updated_entities = [
            SmoothableEntity(e.text, e.type, max(e.confidence, 0.9) if i in forced_entities else e.confidence)
            for i, e in E.items()
        ]
        updated_triples = [
            TripleWithTypes(t.subject, t.subject_type, t.predicate,
                            t.object, t.object_type,
                            max(t.confidence, 0.9) if j in forced_relations else t.confidence,
                            t.evidence)
            for j, t in R.items()
        ]

        return updated_entities, updated_triples


def apply_szf(entities: List[SmoothableEntity], triples: List[TripleWithTypes],
              entity_thresh: Optional[float] = None,
              relation_thresh: Optional[float] = None) -> Tuple[List[SmoothableEntity], List[TripleWithTypes]]:
    """Convenience wrapper for SZF post-processing."""
    processor = GraphPostProcessorSZF(
        entity_threshold=entity_thresh or 0.80,
        relation_threshold=relation_thresh or 0.75,
    )
    return processor.run(entities, triples)
