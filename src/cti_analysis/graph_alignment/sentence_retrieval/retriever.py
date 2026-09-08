"""Sentence-level retrieval for CTI-HAL TTP detection.

For each sentence in a document, produces three retrieval signals:
1. Triple-context vector: embed assembled triples from that sentence, query UCKG
2. Raw sentence vector: embed sentence text, query UCKG
3. Sentence BM25: query UCKG Lucene index with sentence text

Per-sentence RRF fuses the three signals. Document-level aggregation
(max-score and sum-score) produces the final ranking.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ATTACK_LABELS = {"UcoexMITREATTACK", "UcoexTACTICS", "UcoexSOFTWARE"}
# Only consider sentences mentioning TTP-relevant entity types
TTP_ENTITY_TYPES = {"MAL", "TOOL", "ACT", "APT", "VULID", "VULNAME"}


def fetch_sentences(driver, doc_id: str) -> List[Dict[str, str]]:
    """Fetch all CTISentence nodes for a document, ordered by sent_idx."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (d:CTIDocument {id: $doc_id})-[:CONTAINS]->(s:CTISentence)
            RETURN s.id AS id, s.text AS text
            ORDER BY s.sent_idx
            """,
            doc_id=doc_id,
        )
        return [{"id": rec["id"], "text": rec["text"] or ""} for rec in result]


def fetch_ttp_sentences(driver, doc_id: str) -> List[Dict[str, str]]:
    """Fetch sentences that mention at least one TTP-relevant entity type."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (d:CTIDocument {id: $doc_id})-[:CONTAINS]->(s:CTISentence)
            MATCH (s)-[:MENTIONS]->(e:CTIEntity)
            WHERE e.type IN $types
            WITH s, collect(DISTINCT e.type) AS entity_types
            RETURN s.id AS id, s.text AS text, entity_types
            ORDER BY s.sent_idx
            """,
            doc_id=doc_id,
            types=list(TTP_ENTITY_TYPES),
        )
        return [{"id": rec["id"], "text": rec["text"] or "",
                 "entity_types": rec["entity_types"]} for rec in result]


def fetch_sentence_triples(session, sent_id: str) -> List[Dict[str, str]]:
    """Fetch triples where both subject and object are mentioned by the sentence
    and at least one is a TTP-relevant type."""
    result = session.run(
        """
        MATCH (s:CTISentence {id: $sid})-[:MENTIONS]->(subj:CTIEntity)
        MATCH (subj)-[r]->(obj:CTIEntity)
        WHERE (s)-[:MENTIONS]->(obj)
          AND (subj.type IN $types OR obj.type IN $types)
        RETURN subj.name AS subj_name, subj.type AS subj_type,
               type(r) AS predicate,
               obj.name AS obj_name, obj.type AS obj_type
        """,
        sid=sent_id,
        types=list(TTP_ENTITY_TYPES),
    )
    return [dict(rec) for rec in result]


def build_triple_text(triples: List[Dict[str, str]]) -> str:
    """Convert triples into natural-language text for embedding.

    Example: "APT29 (APT) uses Cobalt Strike (TOOL), targets US Government (IDTY)"
    """
    from cti_analysis.graph_alignment.graph_insertion.inserter import _split_camel

    parts = []
    seen = set()
    for t in triples:
        key = (t["subj_name"], t["predicate"], t["obj_name"])
        if key in seen:
            continue
        seen.add(key)
        pred = _split_camel(t["predicate"])
        parts.append(
            f"{t['subj_name']} ({t['subj_type']}) {pred} "
            f"{t['obj_name']} ({t['obj_type']})"
        )
    return ", ".join(parts)


def embed_sentences(driver, doc_id: str, model: str = "nomic-ai/nomic-embed-text-v1") -> int:
    """Embed all CTISentence nodes for a document and store on the node.

    Skips sentences that already have an embedding. Returns count of newly embedded.
    """
    from cti_analysis.graph_alignment.graph_insertion.inserter import LocalEmbedder

    with driver.session() as session:
        result = session.run(
            """
            MATCH (d:CTIDocument {id: $doc_id})-[:CONTAINS]->(s:CTISentence)
            WHERE s.embedding IS NULL
            RETURN s.id AS id, s.text AS text
            ORDER BY s.sent_idx
            """,
            doc_id=doc_id,
        )
        to_embed = [{"id": rec["id"], "text": rec["text"] or ""} for rec in result]

    if not to_embed:
        return 0

    texts = [s["text"] for s in to_embed]
    embedder = LocalEmbedder(model=model)
    embeddings = embedder.embed(texts)

    with driver.session() as session:
        for sent, emb in zip(to_embed, embeddings):
            if emb:
                session.run(
                    "MATCH (s:CTISentence {id: $sid}) SET s.embedding = $embedding",
                    sid=sent["id"],
                    embedding=emb,
                )

    return len(to_embed)


def _vector_query(session, embedding: list, index_name: str, top_k: int) -> List[Dict]:
    """Query UCKG vector index with an embedding, return ranked ATT&CK matches."""
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
        embedding=embedding,
    )
    hits = []
    for rec in result:
        labels = set(rec["labels"] or [])
        if not (ATTACK_LABELS & labels):
            continue
        from cti_analysis.graph_alignment.token_search.searcher import _clean_id_from_uri
        clean_id = _clean_id_from_uri(rec["uri"] or "")
        if clean_id:
            hits.append({"id": clean_id, "cosine": float(rec["cosine"])})
    return hits


def run_sentence_retrieval(
    driver,
    doc_id: str,
    output_dir: str,
    model: str = "nomic-ai/nomic-embed-text-v1",
    top_k: int = 10,
    index_name: str = "attack_vec",
    rrf_k: int = 60,
) -> Dict[str, Any]:
    """Run sentence-level retrieval for a document.

    For each sentence: triple-context vector + raw sentence vector + BM25.
    Fuse per-sentence with RRF, aggregate to document level.
    """
    from cti_analysis.graph_alignment.graph_insertion.inserter import LocalEmbedder
    from cti_analysis.graph_alignment.token_search.searcher import (
        _bm25_query,
        ensure_fulltext_index,
    )
    from cti_analysis.graph_alignment.rrf import reciprocal_rank_fusion

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fetch only sentences with TTP-relevant entities
    sentences = fetch_ttp_sentences(driver, doc_id)
    if not sentences:
        logger.warning("No TTP sentences found for %s", doc_id)
        return {"doc_id": doc_id, "n_sentences": 0}

    # Fetch triples for each sentence
    sentence_triples = {}
    with driver.session() as session:
        for sent in sentences:
            triples = fetch_sentence_triples(session, sent["id"])
            if triples:
                sentence_triples[sent["id"]] = triples

    # Load or compute sentence embeddings
    # Check which sentences already have embeddings stored on nodes
    sent_emb_map = {}  # sent_idx -> embedding
    sents_needing_embed = []
    with driver.session() as session:
        for i, sent in enumerate(sentences):
            result = session.run(
                "MATCH (s:CTISentence {id: $sid}) RETURN s.embedding AS emb",
                sid=sent["id"],
            )
            rec = result.single()
            if rec and rec["emb"]:
                sent_emb_map[i] = rec["emb"]
            else:
                sents_needing_embed.append((i, sent))

    embedder = LocalEmbedder(model=model)

    # Embed any sentences missing stored embeddings
    if sents_needing_embed:
        texts = [s["text"] for _, s in sents_needing_embed]
        print(f"  [sent-retrieval] Embedding {len(texts)} new sentences...", flush=True)
        new_embs = embedder.embed(texts)
        with driver.session() as session:
            for (idx, sent), emb in zip(sents_needing_embed, new_embs):
                if emb:
                    sent_emb_map[idx] = emb
                    session.run(
                        "MATCH (s:CTISentence {id: $sid}) SET s.embedding = $embedding",
                        sid=sent["id"], embedding=emb,
                    )
    else:
        print(f"  [sent-retrieval] All {len(sentences)} sentence embeddings loaded from Neo4j", flush=True)

    # Build and embed triple-context texts (always recomputed — lightweight and config-dependent)
    triple_texts = []  # (sent_idx, text)
    for i, sent in enumerate(sentences):
        triples = sentence_triples.get(sent["id"], [])
        if triples:
            triple_texts.append((i, build_triple_text(triples)))

    triple_emb_map = {}  # sent_idx -> embedding
    if triple_texts:
        triple_raw_texts = [t for _, t in triple_texts]
        print(f"  [sent-retrieval] Embedding {len(triple_raw_texts)} triple-texts...", flush=True)
        triple_embeddings = embedder.embed(triple_raw_texts)
        for (idx, _), emb in zip(triple_texts, triple_embeddings):
            if emb:
                triple_emb_map[idx] = emb

    # Ensure fulltext index exists for BM25
    ensure_fulltext_index(driver)

    # Per-sentence retrieval + fusion
    per_sentence_results = []
    doc_scores_max: Dict[str, float] = {}  # id -> max RRF score
    doc_scores_sum: Dict[str, float] = {}  # id -> sum RRF score
    n_with_triples = 0

    with driver.session() as session:
        for i, sent in enumerate(sentences):
            # Source 1: triple-context vector
            triple_vec_ranked = []
            if i in triple_emb_map:
                n_with_triples += 1
                hits = _vector_query(session, triple_emb_map[i], index_name, top_k)
                triple_vec_ranked = [h["id"] for h in hits]

            # Source 2: raw sentence vector
            sent_vec_ranked = []
            if i in sent_emb_map:
                hits = _vector_query(session, sent_emb_map[i], index_name, top_k)
                sent_vec_ranked = [h["id"] for h in hits]

            # Source 3: BM25
            bm25_ranked = []
            if sent["text"].strip():
                bm25_hits = _bm25_query(session, sent["text"], top_k=top_k, min_score=1.5)
                bm25_ranked = [h["clean_id"] for h in bm25_hits if h.get("clean_id")]

            # Filter empty lists for RRF
            lists = [l for l in [triple_vec_ranked, sent_vec_ranked, bm25_ranked] if l]
            if not lists:
                continue

            # Per-sentence RRF fusion
            fused = reciprocal_rank_fusion(*lists, k=rrf_k)

            # Aggregate to document level
            for attack_id, rrf_score in fused:
                if attack_id not in doc_scores_max or rrf_score > doc_scores_max[attack_id]:
                    doc_scores_max[attack_id] = rrf_score
                doc_scores_sum[attack_id] = doc_scores_sum.get(attack_id, 0.0) + rrf_score

            # Store per-sentence detail
            triple_text = ""
            triples = sentence_triples.get(sent["id"], [])
            if triples:
                triple_text = build_triple_text(triples)

            per_sentence_results.append({
                "sent_id": sent["id"],
                "text": sent["text"][:300],
                "n_triples": len(sentence_triples.get(sent["id"], [])),
                "triple_text": triple_text[:300],
                "n_sources": len(lists),
                "fused_top5": [{"id": cid, "rrf_score": round(s, 6)} for cid, s in fused[:5]],
            })

    # Build ranked lists
    ranked_max = [cid for cid, _ in sorted(doc_scores_max.items(), key=lambda x: x[1], reverse=True)]
    ranked_sum = [cid for cid, _ in sorted(doc_scores_sum.items(), key=lambda x: x[1], reverse=True)]

    results = {
        "doc_id": doc_id,
        "method": "sentence-retrieval-rrf",
        "n_sentences": len(sentences),
        "n_sentences_with_triples": n_with_triples,
        "n_unique_attack_ids": len(doc_scores_max),
        "ranked_ids_max": ranked_max,
        "ranked_ids_sum": ranked_sum,
        "ranked_with_scores_max": [
            {"id": cid, "rrf_score": round(doc_scores_max[cid], 6)} for cid in ranked_max
        ],
        "ranked_with_scores_sum": [
            {"id": cid, "rrf_score": round(doc_scores_sum[cid], 6)} for cid in ranked_sum
        ],
        "per_sentence": per_sentence_results,
    }

    out_path = out_dir / "sentence_retrieval.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved sentence retrieval for %s: %d IDs from %d sentences",
                doc_id, len(ranked_max), len(sentences))
    return results
