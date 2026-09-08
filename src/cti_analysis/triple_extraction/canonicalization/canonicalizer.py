from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple, Optional

from cti_analysis.models.triples import Entity, Triple, TripleBatch
from cti_analysis.ontology import canon_pred
from cti_analysis.algorithms.markov import MarkovEntitySmoother


# =============================================================================
# ENTITY / PREDICATE NORMALIZATION
# =============================================================================

def normalize_entity(text: str) -> str:
    """Normalize entity text: lowercase, strip articles/punctuation, collapse whitespace."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'^(the\s+|a\s+)', '', text)
    # Strip leading/trailing punctuation (keep internal: hyphens, dots in CVEs)
    text = re.sub(r'^[^\w]+', '', text)
    text = re.sub(r'[^\w]+$', '', text)
    # Strip wrapping parens/quotes
    if len(text) > 2 and text[0] in "('\"" and text[-1] in ")'\"":
        text = text[1:-1]
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_predicate(pred: str) -> str:
    """Normalize predicate using the canonical ontology mapping."""
    if not pred:
        return "associatedWith"
    return canon_pred(pred)


# =============================================================================
# NEAR-DUPLICATE ENTITY MERGING
# =============================================================================

def _merge_similar_entities(name_counts: Dict[str, int],
                            threshold: float = 0.85) -> Dict[str, str]:
    """Merge near-duplicate entity names using stripped comparison + fuzzy matching.

    Returns dict mapping original name -> canonical name.
    """
    # Build stripped-form equivalence classes
    stripped_to_names: Dict[str, list] = defaultdict(list)
    for name in name_counts:
        stripped = re.sub(r'[^a-z0-9]', '', name)
        stripped_to_names[stripped].append(name)

    merge_map: Dict[str, str] = {}
    for stripped, names in stripped_to_names.items():
        if len(names) == 1:
            merge_map[names[0]] = names[0]
            continue
        canonical = max(names, key=lambda n: name_counts[n])
        for name in names:
            merge_map[name] = canonical

    # Fuzzy match remaining singletons
    singletons = [n for n, canon in merge_map.items() if canon == n]
    singletons.sort(key=len)

    for i, name_a in enumerate(singletons):
        if merge_map[name_a] != name_a:
            continue
        len_a = len(name_a)
        for j in range(i + 1, len(singletons)):
            name_b = singletons[j]
            if merge_map[name_b] != name_b:
                continue
            len_b = len(name_b)
            if abs(len_a - len_b) > max(len_a, len_b) * (1 - threshold):
                continue
            if len_a < 4 or len_b < 4:
                continue
            ratio = SequenceMatcher(None, name_a, name_b).ratio()
            if ratio >= threshold:
                if name_counts[name_a] >= name_counts[name_b]:
                    merge_map[name_b] = name_a
                else:
                    merge_map[name_a] = name_b

    return merge_map


# =============================================================================
# PER-ENTITY VITERBI TYPE RESOLUTION
# =============================================================================

def _resolve_entity_types(entity_observations: Dict[str, List[str]],
                          alpha: float = 0.1) -> Dict[str, str]:
    """Document-level entity type resolution via per-entity Viterbi smoothing.

    For each entity with inconsistent types across the document, runs Viterbi
    to find the most consistent type. Entities with a single consistent type
    or single occurrence are left unchanged.
    """
    smoother = MarkovEntitySmoother(alpha=alpha)
    resolved = {}

    for name, type_seq in entity_observations.items():
        unique_types = set(type_seq)
        if len(unique_types) == 1 or len(type_seq) == 1:
            resolved[name] = type_seq[0]
        else:
            smoothed = smoother.viterbi(type_seq)
            resolved[name] = Counter(smoothed).most_common(1)[0][0]

    return resolved


# =============================================================================
# CANONICALIZATION STAGE
# =============================================================================

def run_canonicalization_ir(cfg, batches: List[TripleBatch]) -> List[TripleBatch]:
    """Apply canonicalization to triple batches for a single document.

    Pipeline (operates on ALL batches for a document at once):
    1. Normalize entity names (lowercase, strip articles/punctuation)
    2. Normalize predicates (canonical ontology mapping)
    3. Merge near-duplicate entity names (stripped comparison + fuzzy matching)
    4. Per-entity Viterbi type resolution (HMM smooths inconsistent types)
    5. Deduplicate identical triples

    Configuration options (from cfg.canonicalization):
    - enabled: bool (default: True)
    - merge_threshold: float (default: 0.85)
    - markov_alpha: float (default: 0.1)
    """
    if not getattr(cfg, "enabled", True):
        return batches

    merge_threshold = getattr(cfg, "merge_threshold", 0.85)
    markov_alpha = getattr(cfg, "markov_alpha", 0.1)

    # Flatten all triples across batches for document-level processing
    # Keep track of which batch each triple came from
    all_triples = []  # (batch_idx, triple_idx, Triple)
    for bi, batch in enumerate(batches):
        for ti, t in enumerate(batch.triples):
            all_triples.append((bi, ti, t))

    if not all_triples:
        return batches

    # Step 1: Normalize names and predicates
    normalized = []  # (batch_idx, subj_name, subj_type, pred, obj_name, obj_type, original_triple)
    for bi, ti, t in all_triples:
        subj_name = t.subject.name if isinstance(t.subject, Entity) else str(t.subject)
        subj_type = t.subject.type if isinstance(t.subject, Entity) else "indicator"
        obj_name = t.object.name if isinstance(t.object, Entity) else str(t.object)
        obj_type = t.object.type if isinstance(t.object, Entity) else "indicator"

        normalized.append({
            "batch_idx": bi,
            "subj_name": normalize_entity(subj_name),
            "subj_type": subj_type,
            "predicate": normalize_predicate(t.predicate),
            "obj_name": normalize_entity(obj_name),
            "obj_type": obj_type,
            "original": t,
        })

    # Step 2: Count entity names for merging
    name_counts: Dict[str, int] = defaultdict(int)
    for n in normalized:
        if n["subj_name"]:
            name_counts[n["subj_name"]] += 1
        if n["obj_name"]:
            name_counts[n["obj_name"]] += 1

    # Step 3: Merge near-duplicate entities
    merge_map = _merge_similar_entities(name_counts, threshold=merge_threshold)
    n_merged = sum(1 for k, v in merge_map.items() if k != v)

    for n in normalized:
        if n["subj_name"] in merge_map:
            n["subj_name"] = merge_map[n["subj_name"]]
        if n["obj_name"] in merge_map:
            n["obj_name"] = merge_map[n["obj_name"]]

    # Step 4: Per-entity Viterbi type resolution
    entity_observations: Dict[str, List[str]] = defaultdict(list)
    for n in normalized:
        if n["subj_name"]:
            entity_observations[n["subj_name"]].append(n["subj_type"])
        if n["obj_name"]:
            entity_observations[n["obj_name"]].append(n["obj_type"])

    type_map = _resolve_entity_types(entity_observations, alpha=markov_alpha)

    for n in normalized:
        if n["subj_name"] in type_map:
            n["subj_type"] = type_map[n["subj_name"]]
        if n["obj_name"] in type_map:
            n["obj_type"] = type_map[n["obj_name"]]

    # Step 5: Deduplicate and rebuild batches
    seen = set()
    batch_triples: Dict[int, List[Triple]] = defaultdict(list)

    for n in normalized:
        key = (n["subj_name"], n["predicate"], n["obj_name"])
        if key in seen:
            continue
        seen.add(key)

        meta = n["original"].meta.copy() if isinstance(n["original"].meta, dict) else {}
        triple = Triple(
            subject=Entity(name=n["subj_name"], type=n["subj_type"]),
            predicate=n["predicate"],
            object=Entity(name=n["obj_name"], type=n["obj_type"]),
            meta=meta,
        )
        batch_triples[n["batch_idx"]].append(triple)

    # Rebuild batches preserving original structure
    processed_batches = []
    for bi, batch in enumerate(batches):
        triples = batch_triples.get(bi, [])
        processed_batches.append(TripleBatch(
            doc_id=batch.doc_id,
            triples=triples,
            meta=batch.meta.copy() if isinstance(batch.meta, dict) else {},
            version=batch.version,
        ))

    total_before = sum(len(b.triples) for b in batches)
    total_after = sum(len(b.triples) for b in processed_batches)
    print(f"[canonicalizer] {total_before} triples, {n_merged} entities merged, "
          f"{len(type_map)} types resolved, {total_before - total_after} deduped, "
          f"{total_after} output")

    return processed_batches
