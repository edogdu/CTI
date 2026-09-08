"""Token-based and BM25 TTP detection against UCKG ATT&CK nodes.

Two methods:
1. **Token search**: Regex scan + exact/substring/fuzzy string matching
2. **BM25 search**: Neo4j full-text index (Lucene BM25) over UCKG node names
   and descriptions, queried with CTIEntity names and CTISentence texts

Both run as pipeline stages after graph insertion, producing per-document
match results for comparison against embedding similarity.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Set

# ATT&CK ID patterns
TECHNIQUE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
TACTIC_RE = re.compile(r"\bTA0\d{3}\b", re.IGNORECASE)
SOFTWARE_RE = re.compile(r"\bS\d{4}\b", re.IGNORECASE)

ATTACK_LABELS = ("UcoexMITREATTACK", "UcoexTACTICS", "UcoexSOFTWARE")
FULLTEXT_INDEX_NAME = "attackFulltext"


def _split_camel_pred(s: str) -> str:
    """Split camelCase, UPPER_SNAKE, or ALLCAPS predicate into lowercase words.
    e.g. 'usedBy' -> 'used by', 'AFFILIATED_WITH' -> 'affiliated with'
    """
    s = s.replace("_", " ")
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return re.sub(r'\s+', ' ', s).strip().lower()

# CTIEntity types worth searching against UCKG (TTP-relevant)
BM25_ENTITY_TYPES = {"MAL", "TOOL", "ACT", "APT", "VULID", "VULNAME"}


# =========================================================================
# Neo4j full-text index setup
# =========================================================================

def ensure_fulltext_index(driver) -> None:
    """Create a full-text index on UCKG ATT&CK node names and descriptions."""
    with driver.session() as session:
        session.run(
            f"""
            CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS
            FOR (n:UcoexMITREATTACK|UcoexTACTICS|UcoexSOFTWARE)
            ON EACH [n.ucoexNAME, n.ucoexDESCRIPTION]
            """
        )


# =========================================================================
# BM25 search via Neo4j full-text index
# =========================================================================

def _clean_id_from_uri(uri: str) -> str:
    if not uri:
        return ""
    if "#" in uri:
        return uri.split("#")[-1].strip().upper()
    return uri.rstrip("/").split("/")[-1].strip().upper()


def _sanitize_query(text: str) -> str:
    """Escape special Lucene characters and clean up query text."""
    # Remove characters that break Lucene query parser
    special = r'[+\-&|!(){}[\]^"~*?:\\/]'
    cleaned = re.sub(special, " ", text)
    # Remove Lucene reserved words that break the parser
    cleaned = re.sub(r'\b(AND|OR|NOT|TO)\b', ' ', cleaned)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _bm25_query(session, query_text: str, top_k: int = 10, min_score: float = 2.0) -> List[Dict[str, Any]]:
    """Query the full-text index with BM25 scoring."""
    sanitized = _sanitize_query(query_text)
    if not sanitized or len(sanitized) < 3:
        return []
    # Truncate to avoid Lucene TooManyClauses (max 1024 terms)
    words = sanitized.split()
    if len(words) > 200:
        sanitized = " ".join(words[:200])

    result = session.run(
        f"""
        CALL db.index.fulltext.queryNodes("{FULLTEXT_INDEX_NAME}", $q)
        YIELD node, score
        WHERE score > $min_score
        RETURN node.ucoexNAME AS name,
               node.uri AS uri,
               labels(node) AS labels,
               score
        ORDER BY score DESC
        LIMIT $k
        """,
        q=sanitized,
        k=top_k,
        min_score=min_score,
    )
    hits = []
    for rec in result:
        clean_id = _clean_id_from_uri(rec["uri"] or "")
        if clean_id:
            hits.append({
                "name": rec["name"],
                "clean_id": clean_id,
                "labels": rec["labels"],
                "score": rec["score"],
            })
    return hits


def run_bm25_search(
    driver,
    doc_id: str,
    output_dir: str,
    top_k: int = 3,
    min_score: float = 3.0,
) -> Dict[str, Any]:
    """Run BM25 full-text search for a document's entities and sentences.

    For each CTIEntity name and CTISentence text linked to the document,
    queries the UCKG full-text index and collects top-k ATT&CK matches.

    Returns dict with matched IDs per entity/sentence and writes to output_dir.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ensure_fulltext_index(driver)

    # Fetch document's entities with their graph neighbors
    with driver.session() as session:
        ent_result = session.run(
            """
            MATCH (d:CTIDocument {id: $doc_id})-[:MENTIONS]->(e:CTIEntity)
            RETURN DISTINCT e.id AS nid, e.name AS name, e.type AS type
            """,
            doc_id=doc_id,
        )
        entities = [dict(r) for r in ent_result]

        # For each entity, fetch its neighbors to build richer query text
        for ent in entities:
            nid = ent.get("nid", "")
            if not nid:
                continue
            nb_result = session.run(
                """
                MATCH (e:CTIEntity {id: $nid})-[r]->(other:CTIEntity)
                RETURN type(r) AS rel, other.name AS name, other.type AS type
                UNION
                MATCH (other:CTIEntity)-[r]->(e:CTIEntity {id: $nid})
                RETURN type(r) AS rel, other.name AS name, other.type AS type
                """,
                nid=nid,
            )
            neighbors = [dict(r) for r in nb_result]
            ent["neighbors"] = neighbors

        sent_result = session.run(
            """
            MATCH (d:CTIDocument {id: $doc_id})-[:CONTAINS]->(s:CTISentence)
            RETURN s.id AS id, s.text AS text
            """,
            doc_id=doc_id,
        )
        sentences = [dict(r) for r in sent_result]

    matched_ids: Set[str] = set()
    per_entity: List[Dict[str, Any]] = []
    per_sentence: List[Dict[str, Any]] = []

    with driver.session() as session:
        # Query with each entity name + neighbor context (only TTP-relevant types)
        for ent in entities:
            name = ent.get("name", "")
            etype = ent.get("type", "")
            if not name or len(name) < 3:
                continue
            if etype and etype not in BM25_ENTITY_TYPES:
                continue

            # Build entity-neighbors query text with camelCase splitting
            query_parts = [f"{name} ({etype})"]
            for nb in ent.get("neighbors", []):
                rel = _split_camel_pred(nb.get("rel", ""))
                nb_name = nb.get("name", "")
                nb_type = nb.get("type", "")
                if nb_name:
                    query_parts.append(f"{rel} {nb_name} ({nb_type})")
            query_text = ", ".join(query_parts)

            hits = _bm25_query(session, query_text, top_k=top_k, min_score=min_score)
            hit_ids = {h["clean_id"] for h in hits}
            matched_ids.update(hit_ids)
            if hits:
                per_entity.append({
                    "query": query_text,
                    "entity_name": name,
                    "type": etype,
                    "hits": hits,
                })

        # Sentence-level BM25 is too noisy (long text matches broadly).
        # Sentences are better handled by embedding similarity.
        # Only scan sentences for regex ATT&CK IDs (below).

    # Also do regex scan for ATT&CK IDs in entity names and sentences
    regex_ids: Set[str] = set()
    for ent in entities:
        name = ent.get("name", "")
        for m in TECHNIQUE_RE.findall(name):
            regex_ids.add(m.upper())
        for m in TACTIC_RE.findall(name):
            regex_ids.add(m.upper())
        for m in SOFTWARE_RE.findall(name):
            regex_ids.add(m.upper())
    for sent in sentences:
        text = sent.get("text", "")
        for m in TECHNIQUE_RE.findall(text):
            regex_ids.add(m.upper())
        for m in TACTIC_RE.findall(text):
            regex_ids.add(m.upper())
        for m in SOFTWARE_RE.findall(text):
            regex_ids.add(m.upper())

    all_ids = matched_ids | regex_ids

    results = {
        "doc_id": doc_id,
        "method": "bm25+regex",
        "n_entities": len(entities),
        "n_sentences": len(sentences),
        "bm25_matched_ids": sorted(matched_ids),
        "regex_matched_ids": sorted(regex_ids),
        "combined_matched_ids": sorted(all_ids),
        "n_bm25": len(matched_ids),
        "n_regex": len(regex_ids),
        "n_combined": len(all_ids),
        "per_entity": per_entity,
        "per_sentence": per_sentence,
    }

    result_path = out_path / "token_search.json"
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results
