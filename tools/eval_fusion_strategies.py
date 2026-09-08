"""Evaluate different fusion strategies for CTI-HAL similarity results.

Reports both per-entity and per-document recall to understand
which strategy best retrieves ATT&CK IDs from extracted entities.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

_repo = Path(__file__).parent.parent
EVAL_DIR = _repo / "experiments" / "ctihal-pipeline" / "eval"
ANN_DIR = _repo / "datasets" / "CTI-HAL" / "data"
MAPPINGS_CSV = _repo / "config" / "CTI_HAL_mappings.csv"
ATTACK_LABELS = {"UcoexMITREATTACK", "UcoexTACTICS", "UcoexSOFTWARE"}


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


def load_gt(ident, group):
    ids = set()
    p = ANN_DIR / group.lower() / f"{ident}.json"
    if not p.exists():
        return ids
    for ann in json.loads(p.read_text(encoding="utf-8")):
        t = ann.get("technique")
        if isinstance(t, str) and t.strip():
            ids.add(t.strip().upper())
        md = ann.get("metadata") or {}
        for tid in md.get("tactic") or []:
            if isinstance(tid, str) and tid.strip():
                ids.add(tid.strip().upper())
        sub = md.get("sub_technique")
        if isinstance(sub, str) and sub.strip():
            ids.add(sub.strip().upper())
        for sid in md.get("tool") or []:
            if isinstance(sid, str) and sid.strip():
                ids.add(sid.strip().upper())
    return ids


def rollup(ids):
    expanded = set(ids)
    for i in ids:
        if "." in i and i.startswith("T"):
            expanded.add(i.split(".")[0])
    return expanded


def find_mapping(doc_id, mappings):
    import re
    parts = doc_id.split("_", 1)
    pdf_stem = parts[1] if len(parts) > 1 else doc_id
    # Exact
    mapping = mappings.get(pdf_stem.lower())
    if mapping:
        return mapping
    # Normalized (alphanumeric only)
    norm = re.sub(r'[^a-z0-9]', '', pdf_stem.lower())
    for stem, m in mappings.items():
        if re.sub(r'[^a-z0-9]', '', stem.lower()) == norm:
            return m
    # Substring fallback
    for stem, m in mappings.items():
        if pdf_stem.lower() in stem or stem in pdf_stem.lower():
            return m
    return None


def load_all_docs(mappings):
    """Load all sources per document."""
    docs = {}
    for doc_dir in sorted((EVAL_DIR / "similarity").iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        mapping = find_mapping(doc_id, mappings)
        if not mapping:
            continue
        gt = load_gt(mapping["identifier"], mapping["group"])
        if not gt:
            continue

        # Vector similarity: per-entity top-k and document-level aggregation
        try:
            obj = json.loads(
                (doc_dir / "similarity_per_node.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, OSError):
            continue

        # Per-entity: each entity's ranked matches
        entity_matches = []
        # Document-level: best score per ATT&CK ID across all entities
        doc_best = {}

        for entry in obj.values() if isinstance(obj, dict) else obj:
            if not isinstance(entry, dict):
                continue
            src = entry.get("source", {})
            entity_ranked = []
            for hit in entry.get("top_k") or []:
                target = hit.get("target") or {}
                labels = set(target.get("labels") or [])
                if not (ATTACK_LABELS & labels):
                    continue
                cid = (target.get("clean_id") or "").strip().upper()
                cos = (hit.get("scores") or {}).get("cosine", 0.0)
                if cid:
                    entity_ranked.append((cid, cos))
                    if cid not in doc_best or cos > doc_best[cid]:
                        doc_best[cid] = cos
            entity_matches.append(
                {
                    "name": src.get("name", ""),
                    "type": src.get("type", ""),
                    "ranked": [c for c, _ in entity_ranked],
                }
            )

        sim_ranked = [
            c
            for c, _ in sorted(doc_best.items(), key=lambda x: x[1], reverse=True)
        ]

        # BM25
        bm25_ranked = []
        try:
            ts = json.loads(
                (EVAL_DIR / "token_search" / doc_id / "token_search.json").read_text(
                    encoding="utf-8"
                )
            )
            bb = {}
            for e in ts.get("per_entity", []):
                for h in e.get("hits", []):
                    c = h.get("clean_id", "").strip().upper()
                    s = h.get("score", 0.0)
                    if c and (c not in bb or s > bb[c]):
                        bb[c] = s
            bm25_ranked = [
                c for c, _ in sorted(bb.items(), key=lambda x: x[1], reverse=True)
            ]
        except (FileNotFoundError, OSError):
            pass

        # Sentence
        sent_ranked = []
        try:
            ss = json.loads(
                (
                    EVAL_DIR / "sentence_search" / doc_id / "sentence_search.json"
                ).read_text(encoding="utf-8")
            )
            sent_ranked = ss.get("ranked_ids", [])
        except (FileNotFoundError, OSError):
            pass

        docs[doc_id] = {
            "gt": gt,
            "sim": sim_ranked,
            "bm25": bm25_ranked,
            "sent": sent_ranked,
            "entity_matches": entity_matches,
        }
    return docs


# ---- Fusion strategies ----


def strategy_vector_only(d):
    return d["sim"]


def strategy_bm25_boost(d):
    """Promote vector items that BM25 also found."""
    sim = list(d["sim"])
    bset = set(d["bm25"])
    boosted = [c for c in sim if c in bset]
    rest = [c for c in sim if c not in bset]
    return boosted + rest


def strategy_any_boost(d):
    """Promote vector items that any other source also found."""
    sim = list(d["sim"])
    other = set(d["bm25"]) | set(d["sent"])
    boosted = [c for c in sim if c in other]
    rest = [c for c in sim if c not in other]
    return boosted + rest


def strategy_wrrf(d, w_sim=5.0, w_bm25=1.0, w_sent=1.0, k=60):
    """Weighted RRF across all sources."""
    scores = {}
    for rank, c in enumerate(d["sim"]):
        scores[c] = scores.get(c, 0) + w_sim / (k + rank + 1)
    for rank, c in enumerate(d["bm25"]):
        scores[c] = scores.get(c, 0) + w_bm25 / (k + rank + 1)
    for rank, c in enumerate(d["sent"]):
        scores[c] = scores.get(c, 0) + w_sent / (k + rank + 1)
    return [c for c, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def eval_document_level(docs, ranked_fn, k=10, use_rollup=False):
    """Per-document recall: take top-k from the fused ranked list,
    what fraction of GT IDs are found?"""
    recalls = []
    for d in docs.values():
        ranked = ranked_fn(d)
        top = set(ranked[:k])
        gt = rollup(d["gt"]) if use_rollup else d["gt"]
        if use_rollup:
            top = rollup(top)
        recalls.append(len(top & gt) / len(gt) if gt else 0)
    return sum(recalls) / len(recalls) if recalls else 0


def eval_document_pool(docs, k_per_entity=10, use_rollup=False):
    """Main eval approach: each entity gets k matches, union across all
    entities gives the candidate pool. Reports recall of that pool."""
    recalls = []
    for d in docs.values():
        pool = set()
        for em in d["entity_matches"]:
            pool.update(em["ranked"][:k_per_entity])
        gt = rollup(d["gt"]) if use_rollup else d["gt"]
        if use_rollup:
            pool = rollup(pool)
        recalls.append(len(pool & gt) / len(gt) if gt else 0)
    return sum(recalls) / len(recalls) if recalls else 0


def eval_entity_level(docs, k=10):
    """Per-entity recall: for each entity query, how often does any GT ID
    appear in its top-k matches?"""
    hits, total = 0, 0
    for d in docs.values():
        for em in d["entity_matches"]:
            top = set(em["ranked"][:k])
            if top & d["gt"]:
                hits += 1
            total += 1
    return hits / total if total else 0


def main():
    mappings = load_mappings()
    docs = load_all_docs(mappings)
    print(f"Loaded {len(docs)} documents\n")

    strategies = {
        "Vector only": strategy_vector_only,
        "Vector+BM25 boost": strategy_bm25_boost,
        "Vector+any boost": strategy_any_boost,
        "Weighted RRF 3:1:1": lambda d: strategy_wrrf(d, 3, 1, 1),
        "Weighted RRF 5:1:1": lambda d: strategy_wrrf(d, 5, 1, 1),
        "Weighted RRF 5:2:1": lambda d: strategy_wrrf(d, 5, 2, 1),
        "Weighted RRF 10:1:1": lambda d: strategy_wrrf(d, 10, 1, 1),
    }

    # Entity-level recall (vector only)
    ent_recall = eval_entity_level(docs, k=10)
    n_ents = sum(len(d["entity_matches"]) for d in docs.values())
    print(f"Entity-level recall @10 (any GT in entity top-k): {ent_recall:.4f}")
    print(f"  ({n_ents} entity queries across {len(docs)} docs)")
    print()

    # Pool-based recall (main eval approach: per-entity top-k, union)
    pool_raw = eval_document_pool(docs, k_per_entity=10, use_rollup=False)
    pool_rollup = eval_document_pool(docs, k_per_entity=10, use_rollup=True)
    print(f"Vector pool recall (per-entity top-10, union): {pool_raw:.4f} (rollup: {pool_rollup:.4f})")
    print()

    # Document-level comparison: all IDs ranked globally, take top-k
    print("Document-level recall (global ranked list, top-k):")
    print(f"{'Strategy':<30s}  {'@10':>6s}  {'@20':>6s}  {'@50':>6s}  {'@100':>6s}  {'@10r':>6s}  {'@50r':>6s}  {'@100r':>6s}")
    print("-" * 100)
    for name, fn in strategies.items():
        r10 = eval_document_level(docs, fn, 10)
        r20 = eval_document_level(docs, fn, 20)
        r50 = eval_document_level(docs, fn, 50)
        r100 = eval_document_level(docs, fn, 100)
        r10r = eval_document_level(docs, fn, 10, use_rollup=True)
        r50r = eval_document_level(docs, fn, 50, use_rollup=True)
        r100r = eval_document_level(docs, fn, 100, use_rollup=True)
        print(f"{name:<30s}  {r10:6.4f}  {r20:6.4f}  {r50:6.4f}  {r100:6.4f}  {r10r:6.4f}  {r50r:6.4f}  {r100r:6.4f}")


if __name__ == "__main__":
    main()
