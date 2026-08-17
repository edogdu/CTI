try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None  # neo4j optional; only needed for graph insertion stage
import requests
from typing import List, Any, Dict
import json
import numpy as np
import hashlib
import re

from cti_analysis.models.triples import TripleBatch
from cti_analysis.models.graph import NodeIR, EdgeIR, GraphInsertBatch

class LocalEmbedder:
    """Embedding using sentence-transformers (no external server needed)."""
    _model = None

    def __init__(self, model: str = "all-MiniLM-L6-v2", **kwargs):
        self.model_name = model

    def _get_model(self):
        if LocalEmbedder._model is None:
            from sentence_transformers import SentenceTransformer
            LocalEmbedder._model = SentenceTransformer(self.model_name, trust_remote_code=True)
        return LocalEmbedder._model

    def embed(self, texts: List[str]) -> List[Any]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [embeddings[i].tolist() for i in range(len(texts))]



def make_det_id(name: str, ntype: str, doc_id: str) -> str:
    key = f"{(ntype or '').strip().lower()}|{(name or '').strip().lower()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def _doc_id_from_metrics(metrics: dict) -> tuple[str, str]:
    file_name = None
    if isinstance(metrics, dict):
        file_name = metrics.get("file_name")
        content_hash = metrics.get("doc_sha256_id")
    else:
        content_hash = None
    if content_hash:
        return content_hash, file_name or "unknown"
    key = (file_name or "unknown").strip().lower()
    return hashlib.sha256(key.encode("utf-8")).hexdigest(), file_name or "unknown"


def run_insertion(cfg, batches: List[TripleBatch]) -> GraphInsertBatch:
    """
    Convert TripleBatch objects into a GraphInsertBatch (nodes/edges).
    Does not write to the database; call store_in_neo4j for legacy flow.
    """
    nodes: Dict[str, NodeIR] = {}
    edges: Dict[str, EdgeIR] = {}

    for batch in batches:
        for t in batch.triples:
            subj_id = hashlib.sha256(f"{t.subject}|subj".encode("utf-8")).hexdigest()
            obj_id = hashlib.sha256(f"{t.object}|obj".encode("utf-8")).hexdigest()
            nodes.setdefault(subj_id, NodeIR(id=subj_id, labels=["CTIEntity"], properties={"name": t.subject}))
            nodes.setdefault(obj_id, NodeIR(id=obj_id, labels=["CTIEntity"], properties={"name": t.object}))

            rel_id = hashlib.sha256(f"{subj_id}|{t.predicate}|{obj_id}".encode("utf-8")).hexdigest()
            edges.setdefault(
                rel_id,
                EdgeIR(
                    id=rel_id,
                    start_id=subj_id,
                    end_id=obj_id,
                    type=t.predicate.upper() or "RELATED_TO",
                    properties=t.meta,
                ),
            )

    return GraphInsertBatch(nodes=list(nodes.values()), edges=list(edges.values()))

def store_in_neo4j(chunk_obj, driver):
    chunk_metrics = chunk_obj.get("metrics", chunk_obj)
    chunk_data = chunk_obj.get("data", chunk_obj)

    doc_id, file_name = _doc_id_from_metrics(chunk_metrics)

    with driver.session() as session:
        # Ensure document node exists once
        session.execute_write(create_document, doc_id, file_name)

        for d in chunk_data:
            triples = d.get('triple', [])
            context = d.get('context', '')
            if isinstance(triples, list):
                for triple in triples:
                    if (
                        isinstance(triple, dict) and
                        all(k in triple for k in ['subject', 'predicate', 'object'])
                    ):
                        subj = triple['subject']
                        obj = triple['object']

                        if subj is None or obj is None:
                            print("Skipping triple due to null subject or object.")
                            continue

                        subj_name = subj['name'] if isinstance(subj, dict) and 'name' in subj else subj
                        subj_type = subj.get('type', '') if isinstance(subj, dict) else ''
                        obj_name = obj['name'] if isinstance(obj, dict) and 'name' in obj else obj
                        obj_type = obj.get('type', '') if isinstance(obj, dict) else ''

                        if not subj_name or not obj_name:
                            print("Skipping triple due to empty subject or object name.")
                            continue

                        print(f"Adding subject node: name='{subj_name}', class='{subj_type}', context='{context}'")
                        print(f"Adding object node: name='{obj_name}', class='{obj_type}', context='{context}'")
                        print(f"Adding relationship: {subj_name} -[{triple['predicate']}]-> {obj_name}")
                        subj_id = make_det_id(subj_name, subj_type, doc_id)
                        obj_id = make_det_id(obj_name, obj_type, doc_id)
                        session.execute_write(
                            create_triple,
                            subj_id, subj_name, subj_type,
                            triple['predicate'],
                            obj_id, obj_name, obj_type,
                            context,
                            doc_id,
                            file_name,
                        )
            elif (
                isinstance(triples, dict) and
                all(k in triples for k in ['subject', 'predicate', 'object'])
            ):
                subj = triples['subject']
                obj = triples['object']

                if subj is None or obj is None:
                    print("Skipping triple due to null subject or object.")
                    continue

                subj_name = subj['name'] if isinstance(subj, dict) and 'name' in subj else subj
                subj_type = subj.get('type', '') if isinstance(subj, dict) else ''
                obj_name = obj['name'] if isinstance(obj, dict) and 'name' in obj else obj
                obj_type = obj.get('type', '') if isinstance(obj, dict) else ''

                if not subj_name or not obj_name:
                    print("Skipping triple due to empty subject or object name.")
                    continue

                print(f"Adding subject node: name='{subj_name}', type='{subj_type}', context='{context}'")
                print(f"Adding object node: name='{obj_name}', type='{obj_type}', context='{context}'")
                print(f"Adding relationship: {subj_name} -[{triples['predicate']}]-> {obj_name}")
                subj_id = make_det_id(subj_name, subj_type, doc_id)
                obj_id = make_det_id(obj_name, obj_type, doc_id)
                session.execute_write(
                    create_triple,
                    subj_id, subj_name, subj_type,
                    triples['predicate'],
                    obj_id, obj_name, obj_type,
                    context,
                    doc_id,
                    file_name,
                )

def create_document(tx, doc_id: str, file_name: str):
    tx.run(
        """
        MERGE (d:CTIDocument {id: $doc_id})
        ON CREATE SET d.file_name = $file_name
        """,
        doc_id=doc_id,
        file_name=file_name,
    )

def create_triple(tx, subject_id, subject, sub_type, predicate, object_id, object_, obj_type, context, doc_id, file_name):
    # Build a safe relationship type directly from the predicate (no APOC, no property-only rels)
    # Uppercase and replace any non-alphanumeric chars with underscores; ensure not empty
    rel_type = (predicate or "RELATED_TO").upper()
    rel_type = re.sub(r"[^A-Z0-9]", "_", rel_type)
    if not rel_type:
        rel_type = "RELATED_TO"

    query = f"""
        MERGE (s:CTIEntity {{id: $subject_id}})
        ON CREATE SET s.name = $subject,
                      s.type = $sub_type,
                      s.contexts = [$context]
        ON MATCH  SET s.type = coalesce(s.type, $sub_type),
                      s.name = coalesce(s.name, $subject),
                      s.contexts = CASE
                          WHEN s.contexts IS NULL THEN [$context]
                          ELSE s.contexts + CASE WHEN $context IN s.contexts THEN [] ELSE [$context] END
                      END
        MERGE (o:CTIEntity {{id: $object_id}})
        ON CREATE SET o.name = $object,
                      o.type = $obj_type,
                      o.contexts = [$context]
        ON MATCH  SET o.type = coalesce(o.type, $obj_type),
                      o.name = coalesce(o.name, $object),
                      o.contexts = CASE
                          WHEN o.contexts IS NULL THEN [$context]
                          ELSE o.contexts + CASE WHEN $context IN o.contexts THEN [] ELSE [$context] END
                      END
        MERGE (s)-[r:{rel_type} {{type: $predicate}}]->(o)

        // Ensure CTIDocument exists and mention both entities
        MERGE (d:CTIDocument {{id: $doc_id}})
        ON CREATE SET d.file_name = $file_name
        MERGE (d)-[:MENTIONS]->(s)
        MERGE (d)-[:MENTIONS]->(o)
    """
    tx.run(
        query,
        subject_id=subject_id,
        subject=subject,
        sub_type=sub_type,
        predicate=predicate,
        object_id=object_id,
        object=object_,
        obj_type=obj_type,
        context=context,
        doc_id=doc_id,
        file_name=file_name,
    )

def embed_cti_entities_from_chunk(chunk_obj,
                                  driver,
                                  model,
                                  ollama_url: str = "http://localhost:11434/api/embeddings"):
    
    chunk_metrics = chunk_obj.get("metrics", chunk_obj)
    chunk_data = chunk_obj.get("data", chunk_obj)

    doc_id, file_name = _doc_id_from_metrics(chunk_metrics)

    # Aggregate contexts per (name, type)
    node_dict = {}
    for t in chunk_data:
        triples = t.get('triple', [])
        context = t.get('context', '')
        if isinstance(triples, list):
            it = triples
        elif isinstance(triples, dict) and all(k in triples for k in ['subject','predicate','object']):
            it = [triples]
        else:
            it = []
        for triple in it:
            if not (isinstance(triple, dict) and all(k in triple for k in ['subject','predicate','object'])):
                continue
            for node in (triple['subject'], triple['object']):
                if node is None:
                    continue
                name = node['name'] if isinstance(node, dict) and 'name' in node else node
                ntype = node.get('type', '') if isinstance(node, dict) else ''
                if not name:
                    continue
                key = (name, ntype)
                if key not in node_dict:
                    node_dict[key] = set()
                if context:
                    node_dict[key].add(context)

    # Persist aggregated contexts into Neo4j (idempotent add)
    with driver.session() as session:
        for (name, ntype), contexts in node_dict.items():
            nid = make_det_id(name, ntype, doc_id)
            session.run(
                """
                MATCH (n:CTIEntity {id:$id})
                SET n.contexts = coalesce(n.contexts, []) + [x IN $contexts WHERE NOT x IN coalesce(n.contexts, [])]
                """,
                id=nid,
                contexts=sorted(contexts),
            )

    # Build texts for embedding
    node_texts = []
    node_refs = []
    for (name, ntype), contexts in node_dict.items():
        context_text = "\n".join(sorted(contexts))
        text = f"name: {name}\ntype: {ntype}\ncontexts:\n{context_text}".strip()
        node_texts.append(text)
        node_refs.append({'name': name, 'type': ntype, 'contexts': context_text})

    # Compute embeddings
    embedder = LocalEmbedder(model=model)
    embeddings = embedder.embed(node_texts) if node_texts else []

    # Write embeddings back
    with driver.session() as session:
        for idx, node in enumerate(node_refs):
            name = node['name']
            ntype = node['type']
            nid = make_det_id(name, ntype, doc_id)
            # Normalize embedding before writing
            embedding = np.array(embeddings[idx], dtype=float)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (embedding / norm).tolist()
            else:
                embedding = embedding.tolist()
            session.run(
                """
                MATCH (n:CTIEntity {id: $id})
                SET n.embedding = $embedding, n:Vectorized
                """,
                id=nid,
                embedding=embedding,
            )

    # Return a simple summary for callers if they want it
    return [{
        'name': ref['name'],
        'type': ref['type'],
        'embedding_len': len(embeddings[i]) if i < len(embeddings) else 0
    } for i, ref in enumerate(node_refs)]

def _create_sentence(tx, sent_id: str, doc_id: str, text: str, page_no: int, sent_idx: int):
    tx.run(
        """
        MERGE (s:CTISentence {id: $sent_id})
        ON CREATE SET s.text = $text, s.page_no = $page_no, s.sent_idx = $sent_idx
        WITH s
        MATCH (d:CTIDocument {id: $doc_id})
        MERGE (d)-[:CONTAINS]->(s)
        """,
        sent_id=sent_id, doc_id=doc_id, text=text,
        page_no=page_no, sent_idx=sent_idx,
    )


def _link_entity_to_sentence(tx, entity_id: str, sent_id: str):
    tx.run(
        """
        MATCH (e:CTIEntity {id: $entity_id})
        MATCH (s:CTISentence {id: $sent_id})
        MERGE (s)-[:MENTIONS]->(e)
        """,
        entity_id=entity_id, sent_id=sent_id,
    )


def store_batches_in_neo4j(batches: List[TripleBatch], driver, doc_id: str = None) -> int:
    """Insert TripleBatch objects into Neo4j as CTIEntity nodes and relationships.

    Creates the following graph structure:
        (CTIDocument) -[:CONTAINS]-> (CTISentence) -[:MENTIONS]-> (CTIEntity)
        (CTIDocument) -[:MENTIONS]-> (CTIEntity)

    Document-level MENTIONS edges enable document-scoped similarity queries.
    Sentence-level structure enables per-sentence entity provenance.

    Args:
        doc_id: Override document ID for all batches. If None, uses batch.doc_id.
                Use this to group per-sentence batches under one CTIDocument.

    Returns the number of triples inserted.
    """
    count = 0
    with driver.session() as session:
        for batch in batches:
            effective_doc_id = doc_id or batch.doc_id or "unknown"
            file_name = ""
            chunk_text = ""
            page_no = 0
            sent_idx = 0
            # Use chunk_id as sentence ID (unique per sentence), not doc_id
            chunk_ids = batch.meta.get("chunk_ids", []) if isinstance(batch.meta, dict) else []
            sent_id = chunk_ids[0] if chunk_ids else (batch.doc_id or effective_doc_id)

            if isinstance(batch.meta, dict):
                file_name = batch.meta.get("pdf_name", "") or batch.meta.get("file_name", effective_doc_id)
                chunk_text = batch.meta.get("chunk_text", "") or ""
                page_no = batch.meta.get("page_no", 0)
                sent_idx = batch.meta.get("sent_idx", 0)

            session.execute_write(create_document, effective_doc_id, file_name)

            # Create sentence node linked to document
            if chunk_text:
                session.execute_write(
                    _create_sentence, sent_id, effective_doc_id,
                    chunk_text, page_no, sent_idx,
                )

            for t in batch.triples:
                subj_name = t.subject.name if hasattr(t.subject, "name") else str(t.subject)
                subj_type = t.subject.type if hasattr(t.subject, "type") else ""
                obj_name = t.object.name if hasattr(t.object, "name") else str(t.object)
                obj_type = t.object.type if hasattr(t.object, "type") else ""
                context = chunk_text
                if not context and isinstance(t.meta, dict):
                    context = t.meta.get("evidence", "") or ""

                subj_id = make_det_id(subj_name, subj_type, effective_doc_id)
                obj_id = make_det_id(obj_name, obj_type, effective_doc_id)
                session.execute_write(
                    create_triple,
                    subj_id, subj_name, subj_type,
                    t.predicate,
                    obj_id, obj_name, obj_type,
                    context, effective_doc_id, file_name,
                )

                # Link entities to their source sentence
                if chunk_text:
                    session.execute_write(_link_entity_to_sentence, subj_id, sent_id)
                    session.execute_write(_link_entity_to_sentence, obj_id, sent_id)

                count += 1
    return count


# CTIEntity types worth embedding for TTP similarity search
SIMILARITY_ENTITY_TYPES = {"MAL", "TOOL", "ACT", "APT", "VULID", "VULNAME"}


def _get_unembedded_nodes(driver) -> List[Dict]:
    """Fetch CTIEntity nodes that lack embeddings (TTP-relevant types only)."""
    type_list = list(SIMILARITY_ENTITY_TYPES)
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n:CTIEntity)
            WHERE n.embedding IS NULL
              AND n.type IN $types
            RETURN elementId(n) AS uid, n.id AS nid, n.name AS name, n.type AS type,
                   n.contexts AS contexts
            """,
            types=type_list,
        )
        return [dict(r) for r in result]


def _write_embeddings(driver, nodes: List[Dict], embeddings: List) -> None:
    """Persist embedding vectors back to Neo4j nodes."""
    with driver.session() as session:
        for i, n in enumerate(nodes):
            if i < len(embeddings):
                emb = np.array(embeddings[i], dtype=float)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = (emb / norm).tolist()
                else:
                    emb = emb.tolist()
                session.run(
                    "MATCH (n:CTIEntity {id: $nid}) SET n.embedding = $embedding, n:Vectorized",
                    nid=n["nid"], embedding=emb,
                )


def _split_camel(s: str) -> str:
    """Split camelCase, PascalCase, UPPER_SNAKE, or ALLCAPS into lowercase words.

    Examples: 'usedBy' -> 'used by', 'affiliatedWith' -> 'affiliated with',
              'hasAttackLocation' -> 'has attack location',
              'TARGETED_BY' -> 'targeted by', 'USES' -> 'uses'
    """
    import re
    # Replace underscores with spaces
    s = s.replace("_", " ")
    # Insert space before each uppercase letter that follows a lowercase
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()


# Map DNRTI abbreviations to natural-language labels for embedding clarity
TYPE_LABELS = {
    "APT": "threat actor",
    "MAL": "malware",
    "TOOL": "tool",
    "ACT": "attack activity",
    "IDTY": "identity",
    "LOC": "location",
    "TIME": "time",
    "FILE": "file",
    "SECTEAM": "security team",
    "OS": "operating system",
    "VULID": "vulnerability ID",
    "VULNAME": "vulnerability",
    "HASH": "hash",
    "DOM": "domain",
    "ENCR": "encryption",
    "IP": "IP address",
    "URL": "URL",
    "PROT": "protocol",
    "EMAIL": "email",
}


def _readable_type(t: str) -> str:
    """Convert DNRTI type abbreviation to a readable label for embeddings."""
    return TYPE_LABELS.get(t.upper(), t.lower()) if t else ""


def _build_texts_entity_context(nodes: List[Dict], driver=None) -> List[str]:
    """Strategy: entity-context. Embed name + type + source sentence contexts."""
    texts = []
    for n in nodes:
        ctx = ""
        if n.get("contexts"):
            ctx_list = n["contexts"] if isinstance(n["contexts"], list) else [str(n["contexts"])]
            ctx = "\n".join(c for c in ctx_list[:5] if c and c != "None")
        text = f"name: {n['name']}\ntype: {n.get('type', '')}"
        if ctx:
            text += f"\ncontexts:\n{ctx}"
        texts.append(text.strip())
    return texts


def _build_texts_entity_neighbors(nodes: List[Dict], driver=None) -> List[str]:
    """Strategy: entity-neighbors. Embed name + type + all connected triples from the graph.

    For each entity, fetches all relationships to/from other CTIEntity nodes
    and builds a text like:
        APT29 (APT) uses Mimikatz (TOOL), targets US Government (IDTY), ...
    """
    if driver is None:
        return _build_texts_entity_context(nodes, driver)

    texts = []
    with driver.session() as session:
        for n in nodes:
            nid = n["nid"]
            # Outgoing relationships
            out_result = session.run(
                """
                MATCH (n:CTIEntity {id: $nid})-[r]->(other:CTIEntity)
                RETURN type(r) AS rel, other.name AS name, other.type AS type
                """,
                nid=nid,
            )
            out_rels = [dict(r) for r in out_result]

            # Incoming relationships
            in_result = session.run(
                """
                MATCH (n:CTIEntity {id: $nid})<-[r]-(other:CTIEntity)
                RETURN type(r) AS rel, other.name AS name, other.type AS type
                """,
                nid=nid,
            )
            in_rels = [dict(r) for r in in_result]

            parts = [f"{n['name']} ({n.get('type', '')})"]
            for r in out_rels:
                parts.append(f"{_split_camel(r['rel'])} {r['name']} ({r.get('type', '')})")
            for r in in_rels:
                parts.append(f"{_split_camel(r['rel'])} by {r['name']} ({r.get('type', '')})")

            texts.append(", ".join(parts))
    return texts


def _build_texts_triple_concat(nodes: List[Dict], driver=None) -> List[str]:
    """Strategy: triple-concat. For each entity, concatenate all triples it
    appears in as full subject-predicate-object sentences."""
    if driver is None:
        return _build_texts_entity_context(nodes, driver)

    texts = []
    with driver.session() as session:
        for n in nodes:
            nid = n["nid"]
            result = session.run(
                """
                MATCH (n:CTIEntity {id: $nid})-[r]->(other:CTIEntity)
                RETURN n.name AS subj, type(r) AS rel, other.name AS obj
                UNION
                MATCH (other:CTIEntity)-[r]->(n:CTIEntity {id: $nid})
                RETURN other.name AS subj, type(r) AS rel, n.name AS obj
                """,
                nid=nid,
            )
            triples = [f"{r['subj']} {_split_camel(r['rel'])} {r['obj']}" for r in result]
            if triples:
                texts.append(". ".join(triples))
            else:
                texts.append(f"{n['name']} ({n.get('type', '')})")
    return texts


def _build_texts_sentence_direct(nodes: List[Dict], driver=None) -> List[str]:
    """Strategy: sentence-direct. Use the raw source sentence as the embedding text,
    bypassing entity name/type entirely."""
    if driver is None:
        return _build_texts_entity_context(nodes, driver)

    texts = []
    with driver.session() as session:
        for n in nodes:
            nid = n["nid"]
            # Find source sentences via CTISentence -> MENTIONS -> CTIEntity
            result = session.run(
                """
                MATCH (s:CTISentence)-[:MENTIONS]->(n:CTIEntity {id: $nid})
                RETURN s.text AS text
                LIMIT 3
                """,
                nid=nid,
            )
            sentences = [r["text"] for r in result if r.get("text")]
            if sentences:
                texts.append(" ".join(sentences))
            else:
                # Fallback to contexts
                ctx = n.get("contexts") or []
                if isinstance(ctx, list):
                    ctx = [c for c in ctx if c and c != "None"]
                texts.append(" ".join(ctx) if ctx else n["name"])
    return texts


EMBEDDING_STRATEGIES = {
    "entity-context": _build_texts_entity_context,
    "entity-neighbors": _build_texts_entity_neighbors,
    "triple-concat": _build_texts_triple_concat,
    "sentence-direct": _build_texts_sentence_direct,
}


def embed_graph_nodes(
    driver,
    model: str = "nomic-ai/nomic-embed-text-v1",
    strategy: str = "entity-context",
) -> int:
    """Generate and persist embeddings for all CTIEntity nodes that lack them.

    Args:
        driver: Neo4j driver
        model: sentence-transformers model name
        strategy: one of entity-context, entity-neighbors, triple-concat, sentence-direct

    Returns the number of nodes embedded.
    """
    nodes = _get_unembedded_nodes(driver)
    if not nodes:
        return 0

    build_fn = EMBEDDING_STRATEGIES.get(strategy, _build_texts_entity_context)
    print(f"  [embed] Strategy: {strategy}, {len(nodes)} nodes", flush=True)
    texts = build_fn(nodes, driver)

    embedder = LocalEmbedder(model=model)
    embeddings = embedder.embed(texts)

    _write_embeddings(driver, nodes, embeddings)
    return len(nodes)


if __name__ == "__main__":
    import os
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "abcd90909090")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with open("./output/CTI-HAL/apt29/AnalysisOfCyberattackOnUS/chunk_data_AnalysisOfCyberattackOnUS_gemma2_9b.json", "r", encoding="utf-8") as f:
        chunk_json = json.load(f)

    print("Storing triples in Neo4j...")
    store_in_neo4j(chunk_json, driver)
    print("Triples stored. Embedding CTIEntity nodes...")
    n = embed_graph_nodes(driver)
    print(f"Embedded {n} CTIEntity nodes.")
    driver.close()
