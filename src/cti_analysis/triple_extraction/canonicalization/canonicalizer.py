from __future__ import annotations

import re
from typing import List, Dict, Any, Tuple, Optional

from cti_analysis.models.triples import Entity, Triple, TripleBatch
from cti_analysis.ontology import ENTITY_ALIAS_MAP, canon_pred
from cti_analysis.algorithms.markov import SmoothableEntity, apply_markov_smoothing
from cti_analysis.algorithms.szf import TripleWithTypes, apply_szf


# =============================================================================
# ENTITY / PREDICATE NORMALIZATION
# =============================================================================

def normalize_entity(text: str) -> str:
    """Normalize entity text for comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'^(the\s+|a\s+)', '', text)
    text = re.sub(r'\s+', ' ', text)
    return ENTITY_ALIAS_MAP.get(text, text)


def normalize_predicate(pred: str) -> str:
    """Normalize predicate using the canonical ontology mapping."""
    if not pred:
        return "associatedWith"
    return canon_pred(pred)


# =============================================================================
# IR TYPE CONVERSION HELPERS
# =============================================================================

def _extract_entity_info(entity: Any) -> Tuple[str, Optional[str]]:
    """Extract name and type from entity (Entity, dict, or str)."""
    if isinstance(entity, Entity):
        return (entity.name, entity.type)
    elif isinstance(entity, dict):
        name = entity.get("name", "") or entity.get("text", "")
        entity_type = entity.get("type", "indicator")
        return (str(name), entity_type)
    elif isinstance(entity, str):
        return (entity, "indicator")
    else:
        return (str(entity), "indicator")


def _triple_to_typed(t: Triple) -> TripleWithTypes:
    """Convert IR Triple to TripleWithTypes for processing."""
    subj_name, subj_type = _extract_entity_info(t.subject)
    obj_name, obj_type = _extract_entity_info(t.object)

    confidence = t.meta.get("confidence", 0.8) if isinstance(t.meta, dict) else 0.8
    evidence = t.meta.get("evidence", "") if isinstance(t.meta, dict) else ""
    if isinstance(evidence, dict):
        evidence = evidence.get("quote", "")

    return TripleWithTypes(
        subject=normalize_entity(subj_name),
        subject_type=subj_type,
        predicate=normalize_predicate(t.predicate),
        object=normalize_entity(obj_name),
        object_type=obj_type,
        confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.8,
        evidence=str(evidence),
    )


def _typed_to_triple(t: TripleWithTypes, original: Triple) -> Triple:
    """Convert TripleWithTypes back to IR Triple."""
    meta = original.meta.copy() if isinstance(original.meta, dict) else {}
    meta["confidence"] = t.confidence
    if t.evidence:
        meta["evidence"] = t.evidence

    return Triple(
        subject=Entity(name=t.subject, type=t.subject_type),
        predicate=t.predicate,
        object=Entity(name=t.object, type=t.object_type),
        meta=meta,
    )


# =============================================================================
# CANONICALIZATION STAGE
# =============================================================================

def run_canonicalization_ir(cfg, batches: List[TripleBatch]) -> List[TripleBatch]:
    """Apply canonicalization (Markov smoothing and/or SZF) to triple batches.

    Configuration options (from cfg.canonicalization):
    - enable_markov: bool (default: False)
    - markov_alpha: float (default: 0.1)
    - enable_szf: bool (default: False)
    - szf_entity_threshold: float (default: 0.80)
    - szf_relation_threshold: float (default: 0.75)
    """
    if not getattr(cfg, "enabled", True):
        return batches

    enable_markov = getattr(cfg, "enable_markov", False)
    markov_alpha = getattr(cfg, "markov_alpha", 0.1)
    enable_szf = getattr(cfg, "enable_szf", False)
    szf_entity_thresh = getattr(cfg, "szf_entity_threshold", 0.80)
    szf_relation_thresh = getattr(cfg, "szf_relation_threshold", 0.75)

    print(f"[canonicalizer] Processing {len(batches)} batches (markov={enable_markov}, szf={enable_szf})")
    processed_batches = []

    for batch in batches:
        if not batch.triples:
            processed_batches.append(batch)
            continue

        # Convert IR triples to typed triples for processing
        typed_triples = [_triple_to_typed(t) for t in batch.triples]

        # Extract entities from triples
        entities_dict = {}
        for t in typed_triples:
            subj_key = normalize_entity(t.subject)
            obj_key = normalize_entity(t.object)

            if subj_key not in entities_dict:
                entities_dict[subj_key] = SmoothableEntity(
                    text=t.subject, type=t.subject_type, confidence=t.confidence
                )
            if obj_key not in entities_dict:
                entities_dict[obj_key] = SmoothableEntity(
                    text=t.object, type=t.object_type, confidence=t.confidence
                )

        entities = list(entities_dict.values())

        # Get text from batch meta for Markov smoothing
        text = ""
        if isinstance(batch.meta, dict):
            text = batch.meta.get("chunk_text", "") or batch.meta.get("text", "")

        # Apply Markov smoothing if enabled
        if enable_markov and entities:
            if not text:
                print(f"[canonicalizer] Warning: No text for batch {batch.doc_id}, skipping Markov")
            else:
                entities = apply_markov_smoothing(entities, text, alpha=markov_alpha)

            # Update entity types in triples
            entity_type_map = {normalize_entity(e.text): e.type for e in entities}
            for t in typed_triples:
                subj_norm = normalize_entity(t.subject)
                obj_norm = normalize_entity(t.object)
                if subj_norm in entity_type_map:
                    t.subject_type = entity_type_map[subj_norm]
                if obj_norm in entity_type_map:
                    t.object_type = entity_type_map[obj_norm]

        # Apply SZF if enabled
        if enable_szf and entities and typed_triples:
            entities, typed_triples = apply_szf(
                entities, typed_triples,
                entity_thresh=szf_entity_thresh,
                relation_thresh=szf_relation_thresh,
            )

        # Convert back to IR triples
        processed_triples = []
        for i, typed_t in enumerate(typed_triples):
            original_t = batch.triples[i]
            processed_triples.append(_typed_to_triple(typed_t, original_t))

        processed_batches.append(TripleBatch(
            doc_id=batch.doc_id,
            triples=processed_triples,
            meta=batch.meta.copy() if isinstance(batch.meta, dict) else {},
            version=batch.version,
        ))

    return processed_batches
