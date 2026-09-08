"""Evaluate retrieval at the annotation level: for each CTI-HAL annotated sentence,
did we retrieve the correct ATT&CK ID in that sentence's top-k results?

This is a fine-grained evaluation that measures per-sentence retrieval accuracy
rather than document-level recall.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "src"))

ANN_DIR = _repo / "datasets" / "CTI-HAL" / "data"
MAPPINGS_CSV = _repo / "config" / "CTI_HAL_mappings.csv"
SENT_RETRIEVAL_DIR = _repo / "experiments" / "ctihal-pipeline" / "eval" / "sentence_retrieval"


def load_mappings():
    mappings = {}
    with open(MAPPINGS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            doc = (row.get("Document") or "").strip()
            ident = (row.get("Identifier") or "").strip()
            group = (row.get("Group") or "").strip()
            if doc and ident:
                mappings[doc.lower().replace(".pdf", "")] = {
                    "identifier": ident,
                    "group": group,
                }
    return mappings


def load_annotations(ident, group):
    """Load per-sentence annotations from CTI-HAL."""
    p = ANN_DIR / group.lower() / f"{ident}.json"
    if not p.exists():
        return []
    anns = json.loads(p.read_text(encoding="utf-8"))
    results = []
    for a in anns:
        ctx = (a.get("context") or "").strip()
        if not ctx:
            continue
        ids = set()
        tech = a.get("technique")
        if isinstance(tech, str) and tech.strip():
            ids.add(tech.strip().upper())
        md = a.get("metadata") or {}
        sub = md.get("sub_technique")
        if isinstance(sub, str) and sub.strip():
            ids.add(sub.strip().upper())
        for tid in md.get("tactic") or []:
            if isinstance(tid, str) and tid.strip():
                ids.add(tid.strip().upper())
        for sid in md.get("tool") or []:
            if isinstance(sid, str) and sid.strip():
                ids.add(sid.strip().upper())
        if ids:
            results.append({"context": ctx, "ids": ids})
    return results


def match_annotation_to_sentence(ann_context, sentences, threshold=0.5):
    """Find the best matching CTISentence for an annotation context."""
    best_ratio = 0
    best_sent = None
    ctx_lower = ann_context.lower()[:200]

    for s in sentences:
        # Fast substring check
        if ann_context[:50].lower() in s["text"].lower():
            return s, 1.0
        ratio = SequenceMatcher(None, ctx_lower, s["text"].lower()[:200]).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_sent = s
    if best_ratio >= threshold:
        return best_sent, best_ratio
    return None, best_ratio


def rollup(ids):
    expanded = set(ids)
    for i in ids:
        if "." in i and i.startswith("T"):
            expanded.add(i.split(".")[0])
    return expanded


def main():
    from neo4j import GraphDatabase
    from cti_analysis.config import load_config

    cfg = load_config()
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    mappings = load_mappings()

    # Get all doc_ids
    with driver.session() as s:
        r = s.run("MATCH (d:CTIDocument) RETURN d.id AS id ORDER BY d.id")
        doc_ids = [rec["id"] for rec in r]

    total_anns = 0
    total_matched_sents = 0
    hits_at_k = defaultdict(int)  # k -> count of annotations where GT found in top-k
    hits_at_k_rollup = defaultdict(int)
    cat_hits = defaultdict(lambda: defaultdict(int))  # category -> k -> hits
    cat_total = defaultdict(int)  # category -> total
    per_doc_results = {}

    for doc_id in doc_ids:
        import re as _re
        parts = doc_id.split("_", 1)
        pdf_stem = parts[1] if len(parts) > 1 else doc_id
        mapping = mappings.get(pdf_stem.lower())
        if not mapping:
            norm = _re.sub(r'[^a-z0-9]', '', pdf_stem.lower())
            for stem, m in mappings.items():
                if _re.sub(r'[^a-z0-9]', '', stem.lower()) == norm:
                    mapping = m
                    break
        if not mapping:
            continue

        annotations = load_annotations(mapping["identifier"], mapping["group"])
        if not annotations:
            continue

        # Load CTISentences from Neo4j
        with driver.session() as s:
            r = s.run(
                """
                MATCH (d:CTIDocument {id: $doc_id})-[:CONTAINS]->(s:CTISentence)
                RETURN s.id AS id, s.text AS text
                ORDER BY s.sent_idx
                """,
                doc_id=doc_id,
            )
            sentences = [{"id": rec["id"], "text": rec["text"] or ""} for rec in r]

        if not sentences:
            continue

        # Load sentence retrieval results
        sr_file = SENT_RETRIEVAL_DIR / doc_id / "sentence_retrieval.json"
        sr_per_sent = {}
        try:
            sr = json.loads(sr_file.read_text(encoding="utf-8"))
            for ps in sr.get("per_sentence", []):
                sr_per_sent[ps["sent_id"]] = ps
        except (FileNotFoundError, OSError):
            continue

        doc_hits = 0
        doc_total = 0
        doc_matched = 0

        for ann in annotations:
            matched_sent, ratio = match_annotation_to_sentence(ann["context"], sentences)
            if not matched_sent:
                continue

            total_matched_sents += 1
            doc_matched += 1
            doc_total += 1
            total_anns += 1

            # Get this sentence's retrieval results
            ps = sr_per_sent.get(matched_sent["id"], {})
            fused = ps.get("fused_top5", [])
            fused_ids = {item["id"] for item in fused}

            # Also check full fused list if available
            # The top5 might not be enough — check if any GT ID appears
            ann_ids = ann["ids"]
            found = bool(fused_ids & ann_ids)

            # Categorize
            for aid in ann_ids:
                if aid.startswith("TA"):
                    cat_total["tactics"] += 1
                    if found:
                        cat_hits["tactics"][5] += 1
                elif aid.startswith("S"):
                    cat_total["software"] += 1
                    if found:
                        cat_hits["software"][5] += 1
                else:
                    cat_total["techniques"] += 1
                    if found:
                        cat_hits["techniques"][5] += 1

            if found:
                hits_at_k[5] += 1
                doc_hits += 1

            # Rollup check
            fused_rolled = rollup(fused_ids)
            ann_rolled = rollup(ann_ids)
            if fused_rolled & ann_rolled:
                hits_at_k_rollup[5] += 1

        if doc_total > 0:
            per_doc_results[doc_id] = {
                "n_annotations": len(annotations),
                "n_matched": doc_matched,
                "hits": doc_hits,
                "recall": round(doc_hits / doc_matched, 4) if doc_matched else 0,
            }

    driver.close()

    # Report
    print(f"Sentence-Level Annotation Evaluation")
    print(f"=" * 60)
    print(f"Documents evaluated: {len(per_doc_results)}")
    print(f"Total annotations: {total_anns}")
    print(f"Annotations matched to sentences: {total_matched_sents}")
    print()

    if total_anns > 0:
        r5 = hits_at_k.get(5, 0) / total_anns
        r5_rollup = hits_at_k_rollup.get(5, 0) / total_anns
        print(f"Per-annotation hit rate (top-5 fused):")
        print(f"  Raw:    {hits_at_k.get(5, 0)}/{total_anns} = {r5:.4f}")
        print(f"  Rollup: {hits_at_k_rollup.get(5, 0)}/{total_anns} = {r5_rollup:.4f}")
        print()

        print(f"Per category:")
        for cat in ["techniques", "tactics", "software"]:
            t = cat_total.get(cat, 0)
            h = cat_hits.get(cat, {}).get(5, 0)
            print(f"  {cat:12s}: {h}/{t} = {h/t:.4f}" if t else f"  {cat:12s}: n/a")

    print()
    print(f"Per-document breakdown:")
    for doc_id, dr in sorted(per_doc_results.items()):
        print(f"  {doc_id}: {dr['hits']}/{dr['n_matched']} = {dr['recall']:.2f} "
              f"({dr['n_annotations']} annotations)")


if __name__ == "__main__":
    main()
