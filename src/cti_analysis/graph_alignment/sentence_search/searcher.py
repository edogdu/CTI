"""Sentence-level vector similarity search against UCKG ATT&CK nodes.

Embeds each CTISentence text from a document and queries the UCKG vector
index to find matching ATT&CK techniques, tactics, and software. This
captures concepts that appear in the raw text but were never extracted
as named entities (e.g., "downloading additional payloads" -> T1105).

Results are a ranked list of ATT&CK IDs per sentence, which can be
fused with entity-level similarity and BM25 via RRF.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

logger = logging.getLogger(__name__)

ATTACK_LABELS = {"UcoexMITREATTACK", "UcoexTACTICS", "UcoexSOFTWARE"}


def _clean_id_from_uri(uri: str) -> str:
    if not uri:
        return ""
    if "#" in uri:
        return uri.split("#")[-1].strip().upper()
    return uri.rstrip("/").split("/")[-1].strip().upper()


def run_sentence_search(
    driver,
    doc_id: str,
    output_dir: str,
    model: str = "nomic-ai/nomic-embed-text-v1",
    top_k: int = 5,
    index_name: str = "attack_vec",
) -> Dict[str, Any]:
    """Run sentence-level vector similarity against UCKG.

    For each CTISentence linked to the document:
    1. Embed the sentence text with the same model used for UCKG
    2. Query the UCKG vector index for top-k matches
    3. Collect ATT&CK IDs with their cosine scores

    Args:
        driver: Neo4j driver
        doc_id: Document ID to fetch sentences for
        output_dir: Where to save results
        model: Embedding model (must match UCKG embeddings)
        top_k: Matches per sentence
        index_name: Neo4j vector index name

    Returns:
        Dict with ranked ATT&CK IDs and per-sentence matches
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Fetch sentences
    with driver.session() as session:
        result = session.run(
            """
            MATCH (d:CTIDocument {id: $doc_id})-[:CONTAINS]->(s:CTISentence)
            RETURN s.id AS id, s.text AS text
            """,
            doc_id=doc_id,
        )
        sentences = [dict(r) for r in result]

    if not sentences:
        # Fallback: if no CTISentence nodes, use entity contexts
        with driver.session() as session:
            result = session.run(
                """
                MATCH (d:CTIDocument {id: $doc_id})-[:MENTIONS]->(e:CTIEntity)
                WHERE e.contexts IS NOT NULL
                UNWIND e.contexts AS ctx
                WITH DISTINCT ctx AS text WHERE text IS NOT NULL AND text <> 'None' AND text <> ''
                RETURN 'ctx' AS id, text
                """,
                doc_id=doc_id,
            )
            sentences = [dict(r) for r in result]

    if not sentences:
        logger.warning("No sentences found for doc %s", doc_id)
        results = {"doc_id": doc_id, "method": "sentence-vector", "n_sentences": 0,
                   "ranked_ids": [], "per_sentence": []}
        (out_path / "sentence_search.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        return results

    # Ensure UCKG target nodes are labeled for the vector index
    with driver.session() as session:
        session.run("""
            MATCH (n)
            WHERE n.embedding IS NOT NULL
              AND (n:UcoexMITREATTACK OR n:UcoexTACTICS OR n:UcoexSOFTWARE)
              AND NOT n:SimilarityTarget
            SET n:SimilarityTarget
        """)

    # Embed sentences
    from cti_analysis.graph_alignment.graph_insertion.inserter import LocalEmbedder
    embedder = LocalEmbedder(model=model)
    texts = [s["text"] for s in sentences]
    print(f"  [sentence-search] Embedding {len(texts)} sentences...", flush=True)
    embeddings = embedder.embed(texts)

    # Query UCKG vector index for each sentence
    all_hits: Dict[str, float] = {}  # clean_id -> best cosine score
    per_sentence: List[Dict] = []

    with driver.session() as session:
        for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
            if not emb:
                continue

            result = session.run(
                f"""
                CALL db.index.vector.queryNodes('{index_name}', $k, $embedding)
                YIELD node, score
                RETURN node.ucoexNAME AS name,
                       node.uri AS uri,
                       labels(node) AS labels,
                       score AS cosine
                ORDER BY cosine DESC
                """,
                k=top_k,
                embedding=emb,
            )

            sent_hits = []
            for rec in result:
                labels = set(rec["labels"] or [])
                if not (ATTACK_LABELS & labels):
                    continue
                clean_id = _clean_id_from_uri(rec["uri"] or "")
                cosine = float(rec["cosine"])
                if clean_id:
                    sent_hits.append({
                        "clean_id": clean_id,
                        "name": rec["name"],
                        "cosine": cosine,
                    })
                    # Keep best score per ID
                    if clean_id not in all_hits or cosine > all_hits[clean_id]:
                        all_hits[clean_id] = cosine

            if sent_hits:
                per_sentence.append({
                    "sent_id": sent["id"],
                    "text": sent["text"][:200],
                    "hits": sent_hits,
                })

    # Build ranked list by best cosine score
    ranked = sorted(all_hits.items(), key=lambda x: x[1], reverse=True)

    results = {
        "doc_id": doc_id,
        "method": "sentence-vector",
        "n_sentences": len(sentences),
        "n_matched_ids": len(ranked),
        "ranked_ids": [cid for cid, _ in ranked],
        "ranked_with_scores": [{"id": cid, "cosine": round(score, 4)} for cid, score in ranked],
        "per_sentence": per_sentence,
    }

    result_path = out_path / "sentence_search.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [sentence-search] {len(ranked)} ATT&CK IDs matched from {len(per_sentence)} sentences", flush=True)
    return results
