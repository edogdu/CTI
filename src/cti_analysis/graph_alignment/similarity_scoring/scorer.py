import os
import json
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None
from pathlib import Path

from cti_analysis.models.triples import TripleBatch
from cti_analysis.models.scores import SimilarityScore


def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def _get_any_vector_dim(session, index_att) -> int:
    """Infer vector dimension from any node that has an embedding."""
    rec = session.run(
        f"""
        MATCH (n)
        WHERE n.{index_att} IS NOT NULL
        RETURN size(n.{index_att}) AS dim
        LIMIT 1
        """
    ).single()
    if not rec or rec["dim"] is None:
        raise RuntimeError("No embeddings found in the graph to infer vector dimension.")
    return int(rec["dim"])


def clear_similarity_target_label(session, index_label: str) -> None:
    """Remove the indexing label from any node that currently has it."""
    session.run(
        f"""
        MATCH (n:{index_label})
        REMOVE n:{index_label}
        """
    )


def ensure_similarity_target_label(session, index_property: str, index_label: str, allowed_labels: List[str]) -> None:
    """Tag only ATT&CK-related target nodes (by label) that have an embedding with the dedicated index label."""
    if not allowed_labels:
        raise ValueError("allowed_labels list cannot be empty.")

    # Build a Cypher clause like: n:UcoexMITREATTACK OR n:UcoexTACTICS OR n:UcoexSOFTWARE
    label_clause = " OR ".join([f"n:{lbl}" for lbl in allowed_labels])

    query = f"""
    MATCH (n)
    WHERE n.{index_property} IS NOT NULL
      AND ({label_clause})
      AND NOT n:{index_label}
    SET n:{index_label}
    RETURN count(n) AS labeled
    """
    print(query)
    result = session.run(query)
    count = result.single()["labeled"]
    print(f"Labeled {count} target nodes with :{index_label}")


def ensure_vector_index(session, dim: int, index_name: str, index_label: str, index_property: str, sim_func: str) -> None:
    """Create the vector index if it does not exist."""
    # Neo4j does not allow parameters inside schema options reliably, so inline the dimension.
    session.run(
         f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{index_label}) ON (n.{index_property})
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {dim},
            `vector.similarity_function`: '{sim_func}'
          }}
        }}
        """
    )


def fetch_sources(session, doc_id: str, index_property: str) -> List[Dict[str, Any]]:
    res = session.run(
        f"""
        MATCH (d:CTIDocument {{id: $doc_id}})-[:MENTIONS]->(n:CTIEntity)
        WHERE n.{index_property} IS NOT NULL
        RETURN elementId(n) AS uid,
               n.name AS name,
               n.type AS type,
               n.contexts AS contexts,
               labels(n) AS labels,
               n.{index_property} AS embedding
        """,
        doc_id=doc_id,
    )
    return [dict(r) for r in res]


def topk_from_db(session, embedding: List[float], src: Dict[str, Any], k: int, doc_id: str, index_name: str, index_property: str, allowed_labels: List[str]) -> List[Dict[str, Any]]:
    query = f"""
    CALL db.index.vector.queryNodes('{index_name}', $kplus, $embedding)
    YIELD node, score
    WITH node, score
    WHERE elementId(node) <> $src_uid
    RETURN elementId(node) AS uid,
           node.uri AS uri,
           node.ucoexNAME AS name,
           node.ucoexDESCRIPTION AS description,
           labels(node) AS labels,
           node.{index_property} AS embedding,
           score AS cosine
    ORDER BY cosine DESC
    LIMIT $k
    """
    kplus = max(k + 1, 20 * k)
    return [dict(r) for r in session.run(query, kplus=kplus, embedding=embedding, src_uid=src.get("uid"), doc_id=doc_id, k=k, allowed_labels=allowed_labels)]

def run_similarity(driver, doc_id, output_dir, embed_cfg: Dict) -> None:
    # embed_cfg mirrors cfg.embedding from config.yaml
    k = int(embed_cfg.similarity.top_k)
    dim = int(embed_cfg.dimensions)
    index_name = str(embed_cfg.similarity.index.name)
    index_label = str(embed_cfg.similarity.index.label)
    index_property = str(embed_cfg.property)
    sim_func = str(embed_cfg.similarity.index.similarity_function)
    
    allowed_labels = list(getattr(embed_cfg.similarity, 'allowed_labels', [
        "UcoexMITREATTACK", "UcoexTACTICS", "UcoexSOFTWARE"
    ]))
    
    ensure_output_dir(output_dir)

    with driver.session() as session:
        # Make sure labels & index are present
        # Start clean: ensure no stale target labels linger from prior runs
        clear_similarity_target_label(session, index_label)
        ensure_similarity_target_label(session, index_property, index_label, allowed_labels)
        ensure_vector_index(session, dim, index_name, index_label, index_property, sim_func)

        sources = fetch_sources(session, doc_id, index_property)

        if not sources:
            print("No CTIEntity sources with embeddings found.")
            return

        by_node: Dict[str, Any] = {}
        leaderboard: List[Dict[str, Any]] = []

        for s in sources:
            s_id = s["uid"] if s.get("uid") is not None else ""
            s_vec = s["embedding"]

            candidates = topk_from_db(session, s_vec, s, k, doc_id, index_name, index_property, allowed_labels)

            # Compute extra metrics locally for just these K
            s_np = np.asarray(s_vec, dtype=np.float32)
            s_norm = float(np.linalg.norm(s_np)) or 1.0

            top_entries = []
            for c in candidates:
                t_np = np.asarray(c["embedding"], dtype=np.float32)
                dot = float(np.dot(s_np, t_np))
                euc = float(np.linalg.norm(s_np - t_np))
                # cosine from DB is already provided according to SIM_FUNC; if SIM_FUNC != cosine,
                # the field still stores the DB's score but we recompute cosine here for consistency.
                # We'll keep the DB score under key 'db_score' and provide an explicit 'cosine'.
                db_score = float(c["cosine"]) if "cosine" in c else None
                # recompute cosine explicitly
                t_norm = float(np.linalg.norm(t_np)) or 1.0
                cos = float(dot / (s_norm * t_norm))

                uri_val = c.get("uri")
                clean_id = None
                if uri_val:
                    if "#" in uri_val:
                        clean_id = uri_val.split("#")[-1]
                    else:
                        clean_id = uri_val.rstrip("/").split("/")[-1]

                entry = {
                    "source": {
                        "uid": s_id,
                        "name": s.get("name"),
                        "type": s.get("type"),
                        "contexts": s.get("contexts"),
                        "labels": s.get("labels", []),
                    },
                    "target": {
                        "uid": str(c["uid"]),
                        "name": c.get("name"),
                        "description": c.get("description"),
                        "uri": uri_val,
                        "clean_id": clean_id,
                        "labels": c.get("labels", []),
                    },
                    "scores": {
                        "cosine": cos,
                        "dot": dot,
                        "euclidean": euc,
                        "db_score": db_score,
                    },
                }
                top_entries.append(entry)
                leaderboard.append(entry)

            by_node[str(s_id)] = {
                "source": {
                    "uid": s_id,
                    "name": s.get("name"),
                    "type": s.get("type"),
                    "contexts": s.get("contexts"),
                    "labels": s.get("labels", []),
                },
                "top_k": top_entries,
            }
        # End clean: remove target labels after similarity is computed
        clear_similarity_target_label(session, index_label)

    # Sort leaderboard by cosine desc, then dot desc (consistent with NumPy post-metrics)
    leaderboard_sorted = sorted(
        leaderboard,
        key=lambda x: (x["scores"]["cosine"], x["scores"]["dot"], x["scores"]["euclidean"]),
        reverse=True,
    )

    by_node_path = Path(output_dir) / "similarity_per_node.json"
    leaderboard_path = Path(output_dir) / "similarity_leaderboard.json"

    with open(by_node_path, "w", encoding="utf-8") as f:
        json.dump(by_node, f, ensure_ascii=False, indent=2)
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard_sorted, f, ensure_ascii=False, indent=2)

    print(f"Saved per-node results to: {by_node_path}")
    print(f"Saved global leaderboard to: {leaderboard_path}")


def run_from_manifest(driver, manifest_path: str, embed_cfg: Dict, doc_filter: Optional[str] = None) -> None:
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    runs = manifest.get("runs", [])
    if not runs:
        print(f"No runs defined in manifest {manifest_file}")
        return

    found = False
    for run in runs:
        doc_id = run.get("document_id")
        if doc_filter and doc_id != doc_filter:
            continue
        similarity_dir = run.get("similarity_dir")
        if not doc_id or not similarity_dir:
            print("Skipping manifest entry missing document_id or similarity_dir.")
            continue
        output_dir = Path(similarity_dir)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        print(f"Running similarity for document {doc_id} -> {output_dir}")
        run_similarity(driver, doc_id=doc_id, output_dir=output_dir, embed_cfg=embed_cfg)
        found = True

    if doc_filter and not found:
        print(f"Document id {doc_filter} not found in manifest {manifest_file}")


def run_scoring(cfg, batches: List[TripleBatch]) -> List[SimilarityScore]:
    """
    IR-friendly scorer stub; returns empty list until integrated with graph access.
    """
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run similarity scoring for CTI documents.")
    parser.add_argument("--manifest", default="output/CTI-HAL/manifest.json", help="Path to manifest.json")
    parser.add_argument("--doc-id", default=None, help="Optional document id to filter runs")
    args = parser.parse_args()

    from config.config import load_config
    cfg = load_config()

    driver = GraphDatabase.driver(os.getenv(cfg.neo4j.uri, "bolt://localhost:7687"),
                                  auth=(os.getenv(cfg.neo4j.user, "neo4j"), os.getenv(cfg.neo4j.password, "abcd90909090")))
    try:
        embed_cfg = cfg.embedding
        run_from_manifest(driver, args.manifest, embed_cfg, doc_filter=args.doc_id)
    finally:
        driver.close()
