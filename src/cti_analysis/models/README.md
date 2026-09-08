# CTI Analysis Models

This directory contains the **Intermediate Representation (IR)** dataclasses used throughout the CTI Analysis pipeline. These lightweight, in-memory data structures ensure type safety and consistent data flow between pipeline stages.

## Design Philosophy

- **Immutable by Default**: IR objects are dataclasses that represent data at a specific point in the pipeline.
- **Metadata Preservation**: Each IR type includes a `meta` dictionary that accumulates information as data flows through stages.
- **JSON Serializable**: All IR types can be converted to dictionaries via `asdict()` for JSON/JSONL output.
- **Stage Enrichment**: Each stage adds to the IR rather than creating new types, preserving lineage and context.

## Document Models

### `RawDocument`

The initial representation of a document loaded from disk.

```python
@dataclass
class RawDocument:
    doc_id: str                    # Unique document identifier
    source_path: str               # Path to the original file
    text: str                      # Raw document text
    meta: Dict[str, Any]           # Dataset-specific metadata
```

**Usage**: Created by `load_raw_documents()` in `pipeline.py` when reading from datasets (DNRTI JSON, CTI-HAL PDFs, etc.).

**Example metadata**:
- DNRTI: `entities`, `relations`, `ent_labels`, `expected_entities`, `expected_relations`
- CTI-HAL: `path`, `source_file`

### `NormalizedDocument`

Cleaned and normalized document representation.

```python
@dataclass
class NormalizedDocument:
    doc_id: str                    # Inherited from RawDocument
    text: str                      # Normalized/cleaned text
    meta: Dict[str, Any]           # Inherited + normalization metadata
```

**Usage**: Produced by `run_normalization()` in `data_normalization/normalize.py`. Inherits metadata from `RawDocument` and may add normalization-specific fields (e.g., `prealigned_triples` for DNRTI).

### `Chunk`

A segment of a normalized document, typically a sentence or semantic unit.

```python
@dataclass
class Chunk:
    doc_id: str                    # Parent document ID
    chunk_id: str                  # Unique chunk identifier (e.g., "{doc_id}_chunk0")
    text: str                      # Chunk text
    meta: Dict[str, Any]           # Inherited from NormalizedDocument + chunk-specific fields
```

**Usage**: Produced by `run_chunking()` in `triple_extraction/semantic_chunking/chunker.py`, or created directly when chunking is disabled.

**Helper Function**: `chunk_from_text(doc, chunk_id, text, meta_override=None)` creates a `Chunk` inheriting metadata from a `NormalizedDocument`.

## Triple Models

### `Triple`

A subject-predicate-object relationship extracted from text.

```python
@dataclass
class Triple:
    subject: str                   # Subject entity name
    predicate: str                 # Relationship predicate
    object: str                    # Object entity name
    meta: Dict[str, Any]           # Evidence, confidence, source chunk, etc.
```

**Usage**: Created by `run_extraction()` in `triple_extraction/extraction/extractor.py` via LLM extraction.

**Note**: The current implementation uses strings for `subject` and `object`, but the extractor may create dictionaries with `name` and `type` fields. Future refactoring may align the model definition with actual usage.

**Example metadata**:
- `evidence`: Quote or context from source text
- `confidence`: Extraction confidence score
- `chunk_id`: Source chunk identifier

### `TripleBatch`

A collection of triples extracted from one or more chunks of a document.

```python
@dataclass
class TripleBatch:
    doc_id: str                    # Parent document ID
    triples: List[Triple]          # Extracted triples
    meta: Dict[str, Any]           # Batch metadata (chunk_ids, doc lineage, etc.)
    version: Optional[str]          # Pipeline version or model version
```

**Usage**: Produced by `run_extraction()` and passed through repair, canonicalization, and graph alignment stages.

**Helper Function**: `triple_batch_for_doc(doc_id, triples, chunk_ids=None, extra_meta=None, version=None)` creates a `TripleBatch` with proper metadata structure.

**Example metadata**:
- `chunk_ids`: List of source chunk IDs
- `doc_id`: Document identifier (also stored as top-level field)
- `expected_entities`: Ground truth entities (for evaluation)
- `expected_relations`: Ground truth relations (for evaluation)

## Graph Models

### `NodeIR`

A graph node representation for Neo4j insertion.

```python
@dataclass
class NodeIR:
    id: str                        # Unique node identifier
    labels: List[str]              # Neo4j labels (e.g., ["Entity", "Malware"])
    properties: Dict[str, Any]     # Node properties (name, type, etc.)
```

**Usage**: Created by `run_insertion()` in `graph_alignment/graph_insertion/inserter.py` when converting triples to graph nodes.

### `EdgeIR`

A graph edge/relationship representation for Neo4j insertion.

```python
@dataclass
class EdgeIR:
    id: str                        # Unique edge identifier
    start_id: str                  # Source node ID
    end_id: str                    # Target node ID
    type: str                      # Relationship type (e.g., "USES", "TARGETS")
    properties: Dict[str, Any]     # Edge properties (confidence, evidence, etc.)
```

**Usage**: Created by `run_insertion()` when converting triples to graph edges.

### `GraphInsertBatch`

A batch of nodes and edges ready for Neo4j insertion.

```python
@dataclass
class GraphInsertBatch:
    nodes: List[NodeIR]            # Nodes to insert
    edges: List[EdgeIR]            # Edges to insert
```

**Usage**: Produced by `run_insertion()` and written to `results/graph_alignment/graph_insertion/`.

## Score Models

### `SimilarityScore`

A similarity score between a query entity and a target entity in the graph.

```python
@dataclass
class SimilarityScore:
    query_id: str                  # Query entity identifier
    target_id: str                 # Target entity identifier in graph
    score: float                   # Similarity score (0.0 to 1.0)
    meta: Dict[str, Any]           # Scoring metadata (method, embeddings, etc.)
```

**Usage**: Produced by `run_scoring()` in `graph_alignment/similarity_scoring/scorer.py`.

### `RerankResult`

Reranked similarity scores for a query entity.

```python
@dataclass
class RerankResult:
    query_id: str                  # Query entity identifier
    ranked_targets: List[SimilarityScore]  # Sorted list of similarity scores
    meta: Dict[str, Any]           # Reranking metadata (method, parameters, etc.)
```

**Usage**: Produced by `run_reranking()` in `graph_alignment/reranking/reranker.py`.

## Data Flow Example

```
1. RawDocument (from DNRTI JSON)
   ├── doc_id: "dnrti_0"
   ├── text: "APT29 uses Cobalt Strike..."
   └── meta: {entities: [...], relations: [...]}

2. NormalizedDocument
   ├── doc_id: "dnrti_0"
   ├── text: "APT29 uses Cobalt Strike..."
   └── meta: {entities: [...], relations: [...], prealigned_triples: [...]}

3. Chunk
   ├── doc_id: "dnrti_0"
   ├── chunk_id: "dnrti_0_chunk0"
   ├── text: "APT29 uses Cobalt Strike..."
   └── meta: {entities: [...], relations: [...], expected_entities: [...], ...}

4. TripleBatch
   ├── doc_id: "dnrti_0"
   ├── triples: [
   │     Triple(subject="APT29", predicate="uses", object="Cobalt Strike", ...),
   │     ...
   │   ]
   └── meta: {chunk_ids: ["dnrti_0_chunk0"], expected_entities: [...], ...}

5. GraphInsertBatch
   ├── nodes: [
   │     NodeIR(id="apt29", labels=["Entity", "ThreatActor"], ...),
   │     NodeIR(id="cobalt_strike", labels=["Entity", "Tool"], ...)
   │   ]
   └── edges: [
         EdgeIR(start_id="apt29", end_id="cobalt_strike", type="USES", ...)
       ]
```

## Serialization

All IR types can be serialized to JSON using Python's `dataclasses.asdict()`:

```python
from dataclasses import asdict
import json

chunk = Chunk(doc_id="doc1", chunk_id="chunk0", text="...", meta={})
json_str = json.dumps(asdict(chunk), indent=2)
```

The pipeline automatically saves IR objects to JSON/JSONL files in `results/` after each stage using `utils.io.save_stage_outputs()`.

## Extending IR Types

When adding new fields to IR types:

1. **Add to dataclass**: Use `field(default_factory=dict)` for optional dictionaries.
2. **Update helpers**: Modify helper functions (e.g., `chunk_from_text()`, `triple_batch_for_doc()`) to handle new fields.
3. **Preserve metadata**: Ensure `meta` dictionaries are copied and enriched, not replaced.
4. **Update serialization**: Verify `asdict()` works correctly with new fields.

## Type Safety

The pipeline uses Python type hints for static analysis. When working with IR types:

- Import from `cti_analysis.models` (e.g., `from cti_analysis.models.documents import Chunk`)
- Use type hints in function signatures (e.g., `def process(chunks: List[Chunk]) -> List[TripleBatch]`)
- Leverage IDE autocomplete and type checking tools (mypy, pylance)

