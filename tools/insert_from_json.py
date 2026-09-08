"""Insert canonicalized triple JSONs into a Neo4j database.

Reads triple JSONs from an experiment folder, converts to TripleBatch objects
(grouping by sourceSentence), inserts into Neo4j, then embeds entity and
sentence nodes.

Usage:
    python tools/insert_from_json.py --target-db markov --experiment-dir experiments/ctihal-markov
    python tools/insert_from_json.py --target-db chunkedmarkov --experiment-dir experiments/ctihal-chunked-markov
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
from cti_analysis.graph_alignment.graph_insertion.inserter import (
    store_batches_in_neo4j, embed_graph_nodes,
)
from cti_analysis.graph_alignment.sentence_retrieval.retriever import embed_sentences
from cti_analysis.config import load_config


def json_to_batches(doc_id: str, data: dict) -> list[TripleBatch]:
    """Convert a triple JSON file to TripleBatch objects grouped by source sentence."""
    triples = data.get("triples", [])
    file_name = data.get("pdf_name", "")

    # Group triples by source sentence
    by_sentence: dict[str, list[Triple]] = defaultdict(list)
    for t in triples:
        sent = t.get("sourceSentence", "")
        subj = t["subject"] if isinstance(t["subject"], dict) else {"name": str(t["subject"]), "type": "indicator"}
        obj = t["object"] if isinstance(t["object"], dict) else {"name": str(t["object"]), "type": "indicator"}

        by_sentence[sent].append(Triple(
            subject=Entity(name=subj["name"], type=subj.get("type", "indicator")),
            predicate=t.get("predicate", "associatedWith"),
            object=Entity(name=obj["name"], type=obj.get("type", "indicator")),
            meta={"evidence": sent},
        ))

    batches = []
    for i, (sent_text, sent_triples) in enumerate(by_sentence.items()):
        sent_id = f"{doc_id}_sent_{i}"
        batches.append(TripleBatch(
            doc_id=doc_id,
            triples=sent_triples,
            meta={
                "chunk_ids": [sent_id],
                "chunk_text": sent_text,
                "pdf_name": file_name,
                "file_name": file_name,
                "page_no": 0,
                "sent_idx": i,
            },
        ))

    return batches


def main():
    parser = argparse.ArgumentParser(description="Insert triple JSONs into Neo4j")
    parser.add_argument("--target-db", required=True, help="Target Neo4j database name")
    parser.add_argument("--experiment-dir", required=True, help="Experiment directory with eval/triples/")
    args = parser.parse_args()

    from neo4j import GraphDatabase
    cfg = load_config()

    triples_dir = Path(args.experiment_dir) / "eval" / "triples"
    json_files = sorted(triples_dir.glob("*.json"))
    print(f"Inserting {len(json_files)} documents into '{args.target_db}'")

    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))

    # Clear existing CTI nodes (keep UCKG)
    with driver.session(database=args.target_db) as s:
        r1 = s.run("MATCH (n:CTISentence) DETACH DELETE n RETURN count(n) AS c")
        r2 = s.run("MATCH (n:CTIEntity) DETACH DELETE n RETURN count(n) AS c")
        r3 = s.run("MATCH (n:CTIDocument) DETACH DELETE n RETURN count(n) AS c")
        print(f"  Cleared: {r1.single()['c']} sentences, "
              f"{r2.single()['c']} entities, {r3.single()['c']} documents")

    total_triples = 0
    for i, json_file in enumerate(json_files):
        doc_id = json_file.stem
        data = json.loads(json_file.read_text(encoding="utf-8"))

        batches = json_to_batches(doc_id, data)
        n_triples = sum(len(b.triples) for b in batches)
        total_triples += n_triples

        target_driver = GraphDatabase.driver(
            cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
            database=args.target_db,
        )
        store_batches_in_neo4j(batches, target_driver, doc_id=doc_id)
        target_driver.close()

        if (i + 1) % 10 == 0 or i == len(json_files) - 1:
            print(f"  [{i+1}/{len(json_files)}] {total_triples} triples inserted")

    # Embed entity nodes
    print("Embedding entity nodes...")
    target_driver = GraphDatabase.driver(
        cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
        database=args.target_db,
    )
    n_emb = embed_graph_nodes(target_driver, model=cfg.embeddings.model,
                               strategy=cfg.embeddings.strategy)
    print(f"  {n_emb} entity nodes embedded")

    # Embed sentence nodes
    print("Embedding sentence nodes...")
    with target_driver.session() as s:
        r = s.run("MATCH (d:CTIDocument) RETURN d.id AS id ORDER BY d.id")
        doc_ids = [rec["id"] for rec in r]

    for doc_id in doc_ids:
        embed_sentences(target_driver, doc_id=doc_id, model=cfg.embeddings.model)
    print("  Done")

    target_driver.close()
    driver.close()
    print(f"\nTotal: {total_triples} triples in {len(json_files)} documents")


if __name__ == "__main__":
    main()
