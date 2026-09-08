"""Enrich triple JSONs with source sentences from Neo4j, then canonicalize in-place.

Reads triple JSONs from an experiment folder, looks up the source sentence for each
triple from the corresponding Neo4j database, adds "sourceSentence" to each triple,
then applies name/predicate normalization and optional per-entity Viterbi type resolution.

Usage:
    python tools/enrich_and_canonicalize.py --source-db neo4j --experiment-dir experiments/ctihal-markov --enable-type-resolution
    python tools/enrich_and_canonicalize.py --source-db chunked --experiment-dir experiments/ctihal-chunked-markov --enable-type-resolution
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "src"))

from cti_analysis.config import load_config
from cti_analysis.ontology import canon_pred
from cti_analysis.algorithms.markov import MarkovEntitySmoother


def normalize_entity_name(text: str) -> str:
    """Normalize entity text: lowercase, strip articles/punctuation, collapse whitespace."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'^(the\s+|a\s+)', '', text)
    # Strip leading/trailing punctuation (but keep internal: hyphens, dots in CVEs, etc.)
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


def fetch_sentence_map(session, doc_id: str) -> dict[tuple, str]:
    """Build (subj_lower, pred_lower, obj_lower) -> sentence_text mapping from Neo4j."""
    result = session.run(
        """
        MATCH (d:CTIDocument {id: $doc_id})-[:CONTAINS]->(sent:CTISentence)
        MATCH (sent)-[:MENTIONS]->(subj:CTIEntity)-[r]->(obj:CTIEntity)<-[:MENTIONS]-(sent)
        RETURN subj.name AS subj, type(r) AS pred, obj.name AS obj, sent.text AS text
        """,
        doc_id=doc_id,
    )
    mapping = {}
    for rec in result:
        key = (rec["subj"].lower().strip(), rec["pred"].lower().strip(), rec["obj"].lower().strip())
        # Keep the first (or longest) sentence if multiple
        if key not in mapping or len(rec["text"] or "") > len(mapping[key]):
            mapping[key] = rec["text"] or ""
    return mapping


def enrich_triples(triples: list[dict], sent_map: dict[tuple, str]) -> tuple[int, int]:
    """Add sourceSentence to each triple. Returns (matched, total)."""
    matched = 0
    for t in triples:
        subj = t["subject"]["name"] if isinstance(t["subject"], dict) else str(t["subject"])
        obj = t["object"]["name"] if isinstance(t["object"], dict) else str(t["object"])
        pred = t.get("predicate", "")
        key = (subj.lower().strip(), pred.lower().strip(), obj.lower().strip())
        if key in sent_map:
            t["sourceSentence"] = sent_map[key]
            matched += 1
        else:
            t["sourceSentence"] = ""
    return matched, len(triples)


def merge_similar_entities(triples: list[dict], threshold: float = 0.85) -> dict[str, str]:
    """Merge near-duplicate entity names using normalized string comparison.

    Builds equivalence classes of entity names that are identical after
    stripping non-alphanumeric characters (catches "cozy bear" / "cozybear",
    "apt-29" / "apt29", etc.). For each class, picks the most frequent
    surface form as canonical.

    For names that don't match exactly after stripping, falls back to
    SequenceMatcher ratio with the given threshold.

    Args:
        triples: List of triple dicts (must already be name-normalized).
        threshold: Minimum similarity ratio for fuzzy merging (0.0-1.0).

    Returns:
        dict mapping original entity name -> canonical entity name
    """
    from difflib import SequenceMatcher

    # Count occurrences of each entity name
    name_counts: dict[str, int] = defaultdict(int)
    for t in triples:
        for role in ("subject", "object"):
            e = t[role]
            if isinstance(e, dict):
                name = e.get("name", "")
                if name:
                    name_counts[name] += 1

    # Build stripped-form equivalence classes
    # "cozy bear" and "cozybear" both strip to "cozybear"
    stripped_to_names: dict[str, list[str]] = defaultdict(list)
    for name in name_counts:
        stripped = re.sub(r'[^a-z0-9]', '', name)
        stripped_to_names[stripped].append(name)

    # For each equivalence class, pick the most frequent surface form
    merge_map: dict[str, str] = {}
    for stripped, names in stripped_to_names.items():
        if len(names) == 1:
            merge_map[names[0]] = names[0]
            continue
        # Pick the most common surface form as canonical
        canonical = max(names, key=lambda n: name_counts[n])
        for name in names:
            merge_map[name] = canonical

    # Second pass: fuzzy match remaining singletons against each other
    # Only compare names of similar length to avoid O(n^2) blowup
    singletons = [n for n, canon in merge_map.items() if canon == n]
    singletons.sort(key=len)

    for i, name_a in enumerate(singletons):
        if merge_map[name_a] != name_a:
            continue  # Already merged
        len_a = len(name_a)
        for j in range(i + 1, len(singletons)):
            name_b = singletons[j]
            if merge_map[name_b] != name_b:
                continue
            len_b = len(name_b)
            # Skip if length difference too large (can't meet threshold)
            if abs(len_a - len_b) > max(len_a, len_b) * (1 - threshold):
                continue
            # Skip very short names (too many false positives)
            if len_a < 4 or len_b < 4:
                continue
            ratio = SequenceMatcher(None, name_a, name_b).ratio()
            if ratio >= threshold:
                # Merge less frequent into more frequent
                if name_counts[name_a] >= name_counts[name_b]:
                    merge_map[name_b] = name_a
                else:
                    merge_map[name_a] = name_b

    return merge_map


def resolve_entity_types(triples: list[dict], alpha: float = 0.1) -> dict[str, str]:
    """Document-level entity type resolution via per-entity Viterbi smoothing.

    For each unique entity name, collects all type assignments in document order
    and runs Viterbi to find the most consistent type sequence. The HMM transition
    matrix penalizes type changes, so isolated disagreements get smoothed out by
    surrounding context.

    For entities with only one type across the document, Viterbi is skipped
    (already consistent). For entities appearing only once, the original type
    is kept.

    Args:
        triples: List of triple dicts (must already be name-normalized).
        alpha: HMM transition smoothing (lower = stricter, fewer type changes).

    Returns:
        dict mapping normalized entity name -> resolved type
    """
    # Collect all (name -> ordered list of type observations) in document order
    entity_observations: dict[str, list[str]] = defaultdict(list)

    for t in triples:
        for role in ("subject", "object"):
            e = t[role]
            if isinstance(e, dict):
                name = e.get("name", "")
                etype = e.get("type", "indicator")
            else:
                name = str(e)
                etype = "indicator"
            if name:
                entity_observations[name].append(etype)

    smoother = MarkovEntitySmoother(alpha=alpha)
    resolved = {}

    for name, type_seq in entity_observations.items():
        unique_types = set(type_seq)

        if len(unique_types) == 1:
            # Already consistent — no smoothing needed
            resolved[name] = type_seq[0]
        elif len(type_seq) == 1:
            # Single appearance — keep as-is
            resolved[name] = type_seq[0]
        else:
            # Multiple appearances with type disagreement — run Viterbi
            smoothed = smoother.viterbi(type_seq)
            # Pick the majority of the smoothed sequence
            from collections import Counter
            resolved[name] = Counter(smoothed).most_common(1)[0][0]

    return resolved


def canonicalize_triples(triples: list[dict], enable_type_resolution: bool = False,
                         merge_threshold: float = 0.85) -> list[dict]:
    """Canonicalize triples: normalize, merge similar entities, resolve types, deduplicate.

    Pipeline:
    1. Normalize entity names (lowercase, strip articles/whitespace)
    2. Merge near-duplicate entity names (stripped comparison + fuzzy matching)
    3. Resolve entity types via per-entity Viterbi (if enabled)
    4. Deduplicate identical triples
    """
    if not triples:
        return triples

    # Step 1: Normalize names and predicates
    for t in triples:
        if isinstance(t["subject"], dict):
            t["subject"]["name"] = normalize_entity_name(t["subject"]["name"])
        if isinstance(t["object"], dict):
            t["object"]["name"] = normalize_entity_name(t["object"]["name"])
        t["predicate"] = normalize_predicate(t["predicate"])

    # Step 2: Merge similar entity names
    merge_map = merge_similar_entities(triples, threshold=merge_threshold)
    n_merged = sum(1 for k, v in merge_map.items() if k != v)
    for t in triples:
        for role in ("subject", "object"):
            e = t[role]
            if isinstance(e, dict):
                name = e.get("name", "")
                if name in merge_map:
                    e["name"] = merge_map[name]

    # Step 3: Document-level entity type resolution via per-entity Viterbi
    if enable_type_resolution:
        type_map = resolve_entity_types(triples)
        for t in triples:
            for role in ("subject", "object"):
                e = t[role]
                if isinstance(e, dict):
                    name = e.get("name", "")
                    if name in type_map:
                        e["type"] = type_map[name]

    # Step 4: Deduplicate (same subject+predicate+object after normalization+merging)
    seen = set()
    deduped = []
    for t in triples:
        subj_name = t["subject"]["name"] if isinstance(t["subject"], dict) else str(t["subject"])
        obj_name = t["object"]["name"] if isinstance(t["object"], dict) else str(t["object"])
        key = (subj_name, t["predicate"], obj_name)
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    return deduped


def main():
    parser = argparse.ArgumentParser(description="Enrich triple JSONs with sentences, then canonicalize")
    parser.add_argument("--source-db", required=True, help="Source Neo4j database for sentence lookup")
    parser.add_argument("--experiment-dir", required=True, help="Experiment directory with eval/triples/")
    parser.add_argument("--enable-type-resolution", action="store_true",
                        help="Enable document-level entity type resolution (majority vote)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    args = parser.parse_args()

    from neo4j import GraphDatabase
    cfg = load_config()
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))

    triples_dir = Path(args.experiment_dir) / "eval" / "triples"
    if not triples_dir.exists():
        print(f"Error: {triples_dir} not found")
        return

    json_files = sorted(triples_dir.glob("*.json"))
    print(f"Processing {len(json_files)} documents from '{args.source_db}' -> {triples_dir}")
    print(f"  Type resolution: {args.enable_type_resolution}")

    total_triples_before = 0
    total_triples_after = 0
    total_matched = 0
    total_total = 0

    for i, json_file in enumerate(json_files):
        doc_id = json_file.stem
        data = json.loads(json_file.read_text(encoding="utf-8"))
        triples = data.get("triples", [])
        if not triples:
            continue

        n_before = len(triples)
        total_triples_before += n_before

        # Enrich with source sentences
        with driver.session(database=args.source_db) as s:
            sent_map = fetch_sentence_map(s, doc_id)

        matched, total = enrich_triples(triples, sent_map)
        total_matched += matched
        total_total += total

        # Canonicalize
        canon_triples = canonicalize_triples(triples,
                                              enable_type_resolution=args.enable_type_resolution)
        n_after = len(canon_triples)
        total_triples_after += n_after

        # Update JSON
        data["triples"] = canon_triples
        data["n_triples"] = n_after

        if not args.dry_run:
            json_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"  [{i+1}/{len(json_files)}] {doc_id}: {n_before} -> {n_after} triples "
              f"({matched}/{total} sentences matched)")

    driver.close()

    print(f"\nTotal: {total_triples_before} -> {total_triples_after} triples "
          f"({total_triples_before - total_triples_after} deduped)")
    print(f"Sentence coverage: {total_matched}/{total_total} "
          f"({total_matched/total_total*100:.1f}%)")

    # Run graph quality metrics
    if not args.dry_run:
        print("\nComputing graph quality metrics...")
        from subprocess import run as subprocess_run
        metrics_dir = Path(args.experiment_dir) / "eval" / "graph_metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        subprocess_run([
            sys.executable, str(_repo / "tools" / "eval_graph_quality.py"),
            "--from-json", str(triples_dir),
            "--output", str(metrics_dir / "quality.json"),
        ])


if __name__ == "__main__":
    main()
