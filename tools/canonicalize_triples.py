"""Canonicalize triples from a source Neo4j database and insert into a target database.

Rebuilds triples from Neo4j (preserving sentence provenance), runs canonicalization
(name normalization + optional Markov smoothing + optional SZF), saves to experiment
folder, and inserts into the target Neo4j database.

Usage:
    python tools/canonicalize_triples.py --source-db neo4j --target-db canonicalized --experiment-dir experiments/ctihal-canonicalized
    python tools/canonicalize_triples.py --source-db chunked --target-db chunkedcanon --experiment-dir experiments/ctihal-chunked-canonicalized
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "src"))

from cti_analysis.models.triples import Entity, Triple, TripleBatch
from cti_analysis.triple_extraction.canonicalization.canonicalizer import run_canonicalization_ir
from cti_analysis.graph_alignment.graph_insertion.inserter import (
    store_batches_in_neo4j, embed_graph_nodes,
)
from cti_analysis.graph_alignment.sentence_retrieval.retriever import embed_sentences
from cti_analysis.config import load_config


def fetch_doc_triples_with_sentences(session, doc_id: str):
    """Fetch all triples for a document, grouped by source sentence."""
    result = session.run(
        """
        MATCH (d:CTIDocument {id: $doc_id})-[:CONTAINS]->(sent:CTISentence)
        MATCH (sent)-[:MENTIONS]->(subj:CTIEntity)-[r]->(obj:CTIEntity)<-[:MENTIONS]-(sent)
        RETURN sent.id AS sent_id, sent.text AS sent_text,
               sent.page_no AS page_no, sent.sent_idx AS sent_idx,
               subj.name AS subj_name, subj.type AS subj_type,
               type(r) AS predicate,
               obj.name AS obj_name, obj.type AS obj_type,
               d.file_name AS file_name
        """,
        doc_id=doc_id,
    )

    # Group by sentence
    by_sentence = defaultdict(lambda: {"text": "", "page_no": 0, "sent_idx": 0,
                                        "file_name": "", "triples": []})
    for rec in result:
        sid = rec["sent_id"]
        by_sentence[sid]["text"] = rec["sent_text"] or ""
        by_sentence[sid]["page_no"] = rec["page_no"] or 0
        by_sentence[sid]["sent_idx"] = rec["sent_idx"] or 0
        by_sentence[sid]["file_name"] = rec["file_name"] or ""
        by_sentence[sid]["triples"].append({
            "subj_name": rec["subj_name"],
            "subj_type": rec["subj_type"],
            "predicate": rec["predicate"],
            "obj_name": rec["obj_name"],
            "obj_type": rec["obj_type"],
        })

    return by_sentence


def build_batches(doc_id: str, by_sentence: dict) -> list[TripleBatch]:
    """Convert sentence-grouped triples to TripleBatch objects."""
    batches = []
    for sent_id, data in by_sentence.items():
        triples = []
        seen = set()
        for t in data["triples"]:
            key = (t["subj_name"], t["predicate"], t["obj_name"])
            if key in seen:
                continue
            seen.add(key)
            triples.append(Triple(
                subject=Entity(name=t["subj_name"], type=t["subj_type"]),
                predicate=t["predicate"],
                object=Entity(name=t["obj_name"], type=t["obj_type"]),
                meta={"evidence": data["text"]},
            ))
        if triples:
            batches.append(TripleBatch(
                doc_id=doc_id,
                triples=triples,
                meta={
                    "chunk_ids": [sent_id],
                    "chunk_text": data["text"],
                    "page_no": data["page_no"],
                    "sent_idx": data["sent_idx"],
                    "file_name": data["file_name"],
                    "pdf_name": data["file_name"],
                },
            ))
    return batches


def batches_to_json(doc_id: str, batches: list[TripleBatch], group: str, file_name: str) -> dict:
    """Convert canonicalized batches to the standard triple JSON format."""
    triples_data = []
    for b in batches:
        for t in b.triples:
            triples_data.append({
                "subject": {"name": t.subject.name, "type": t.subject.type},
                "predicate": t.predicate,
                "object": {"name": t.object.name, "type": t.object.type},
            })
    return {
        "doc_id": doc_id,
        "group": group,
        "pdf_name": file_name,
        "n_chunks": len(batches),
        "n_triples": len(triples_data),
        "triples": triples_data,
    }


def main():
    parser = argparse.ArgumentParser(description="Canonicalize triples from Neo4j")
    parser.add_argument("--source-db", required=True, help="Source Neo4j database name")
    parser.add_argument("--target-db", required=True, help="Target Neo4j database name")
    parser.add_argument("--experiment-dir", required=True, help="Target experiment directory")
    parser.add_argument("--enable-markov", action="store_true", help="Enable Markov smoothing")
    parser.add_argument("--enable-szf", action="store_true", help="Enable SZF propagation")
    args = parser.parse_args()

    cfg = load_config()
    # Override canonicalization settings
    cfg.canonicalization.enabled = True
    cfg.canonicalization.enable_markov = args.enable_markov
    cfg.canonicalization.enable_szf = args.enable_szf

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))

    exp_dir = Path(args.experiment_dir)
    triples_dir = exp_dir / "eval" / "triples"
    triples_dir.mkdir(parents=True, exist_ok=True)

    # Get all doc_ids from source database
    with driver.session(database=args.source_db) as s:
        r = s.run("MATCH (d:CTIDocument) RETURN d.id AS id ORDER BY d.id")
        doc_ids = [rec["id"] for rec in r]

    print(f"Processing {len(doc_ids)} documents from '{args.source_db}' -> '{args.target_db}'")
    print(f"  Markov: {args.enable_markov}, SZF: {args.enable_szf}")

    # Clear target database CTI nodes (keep UCKG)
    with driver.session(database=args.target_db) as s:
        r1 = s.run("MATCH (n:CTISentence) DETACH DELETE n RETURN count(n) AS c")
        r2 = s.run("MATCH (n:CTIEntity) DETACH DELETE n RETURN count(n) AS c")
        r3 = s.run("MATCH (n:CTIDocument) DETACH DELETE n RETURN count(n) AS c")
        print(f"  Cleared target db: {r1.single()['c']} sentences, "
              f"{r2.single()['c']} entities, {r3.single()['c']} documents")

    total_triples_before = 0
    total_triples_after = 0

    for i, doc_id in enumerate(doc_ids):
        # Fetch from source
        with driver.session(database=args.source_db) as s:
            by_sentence = fetch_doc_triples_with_sentences(s, doc_id)

        if not by_sentence:
            print(f"  [{i+1}/{len(doc_ids)}] {doc_id}: no triples, skipping")
            continue

        # Build batches
        batches = build_batches(doc_id, by_sentence)
        n_before = sum(len(b.triples) for b in batches)
        total_triples_before += n_before

        # Canonicalize
        canon_batches = run_canonicalization_ir(cfg.canonicalization, batches)
        n_after = sum(len(b.triples) for b in canon_batches)
        total_triples_after += n_after

        # Extract group from doc_id
        group = doc_id.split("_", 1)[0]
        file_name = by_sentence[next(iter(by_sentence))]["file_name"]

        # Save JSON
        doc_json = batches_to_json(doc_id, canon_batches, group, file_name)
        out_path = triples_dir / f"{doc_id}.json"
        out_path.write_text(json.dumps(doc_json, indent=2, ensure_ascii=False), encoding="utf-8")

        # Insert into target Neo4j
        target_driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
                                              database=args.target_db)
        store_batches_in_neo4j(canon_batches, target_driver, doc_id=doc_id)
        target_driver.close()

        print(f"  [{i+1}/{len(doc_ids)}] {doc_id}: {n_before} -> {n_after} triples, "
              f"{len(by_sentence)} sentences")

    print(f"\nTotal: {total_triples_before} -> {total_triples_after} triples")

    # Embed entity nodes in target database
    print("Embedding entity nodes...")
    target_driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
                                          database=args.target_db)
    n_emb = embed_graph_nodes(target_driver, model=cfg.embeddings.model, strategy=cfg.embeddings.strategy)
    print(f"  {n_emb} entity nodes embedded")

    # Embed sentence nodes
    print("Embedding sentence nodes...")
    for doc_id in doc_ids:
        embed_sentences(target_driver, doc_id=doc_id, model=cfg.embeddings.model)
    print("  Done")

    target_driver.close()

    # Run graph quality metrics
    print("Computing graph quality metrics...")
    from subprocess import run as subprocess_run
    metrics_dir = exp_dir / "eval" / "graph_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    subprocess_run([
        sys.executable, str(_repo / "tools" / "eval_graph_quality.py"),
        "--from-json", str(triples_dir),
        "--output", str(metrics_dir / "quality.json"),
    ])

    print(f"\nDone. Results in {exp_dir}")


if __name__ == "__main__":
    main()
