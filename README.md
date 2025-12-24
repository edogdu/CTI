# CTI Analysis

We are developing a framework for converting unstructured Cyber Threat Intelligence (CTI) reports to document-scope knowledge graphs and aligning them to a unified cybersecurity ontology, allowing large-scale analysis of the CTI landscape.

Our framework utilizes LLMs for extracting entities and relationships from CTI sentences. It then inserts resulting triples to graph storage (Neo4j), generates embeddings over observed sentence contexts for semantic alignment, and computes similarity scores against a source graph. We utilize the Unified Cybersecurity Knowledge Graph for this task as it contains a large dataset scope, such as MITRE ATT&CK, CAPEC, and CVE/CVSS.

## Architecture Overview

The CTI Analysis pipeline has been refactored into a **modular, stage-based architecture** that processes documents through a series of configurable stages. Each stage operates on standardized **Intermediate Representation (IR)** dataclasses, ensuring type safety and clear data flow.

### Pipeline Flow

```
Raw Documents → Normalization → Chunking → Extraction → Repair → Canonicalization → Graph Insertion → Similarity Scoring → Reranking
```

1. **Data Normalization** (`data_normalization/`): Converts raw documents (PDFs, JSON, text files) into `NormalizedDocument` objects with cleaned text and metadata.

2. **Semantic Chunking** (`triple_extraction/semantic_chunking/`): Splits normalized documents into `Chunk` objects for processing. Can be disabled to process entire documents as single chunks.

3. **Triple Extraction** (`triple_extraction/extraction/`): Uses LLMs (Ollama) to extract subject-predicate-object triples from chunks, producing `TripleBatch` objects. Supports parallel processing for performance.

4. **Triple Repair** (`triple_extraction/triple_repair/`): Deduplicates and validates extracted triples.

5. **Canonicalization** (`triple_extraction/canonicalization/`): Maps entity names to canonical forms and ontology identifiers.

6. **Graph Insertion** (`graph_alignment/graph_insertion/`): Converts `TripleBatch` objects into `GraphInsertBatch` (nodes and edges) and inserts them into Neo4j.

7. **Similarity Scoring** (`graph_alignment/similarity_scoring/`): Computes similarity scores between extracted entities and existing graph nodes.

8. **Reranking** (`graph_alignment/reranking/`): Applies reranking logic over similarity scores to produce final alignment results.

### Key Design Principles

- **IR-Based Data Flow**: All stages use shared dataclasses (`RawDocument`, `NormalizedDocument`, `Chunk`, `Triple`, `TripleBatch`, etc.) defined in `src/cti_analysis/models/`. See `src/cti_analysis/models/README.md` for detailed documentation.

- **Configuration-Driven**: Pipeline behavior is controlled via YAML configs:
  - `config/config.yaml`: Global settings (paths, models, database connections)
  - `config/pipeline.yaml`: Per-stage toggles (enable/disable stages)

- **Stage Independence**: Each stage can be enabled or disabled independently, allowing flexible pipeline configurations.

- **Progress Persistence**: Intermediate results are automatically saved to `results/` after each stage, enabling inspection and debugging.

## Prerequisites

- **Python 3.9+**
- **Ollama**: Required for LLM-based triple extraction. Install and ensure the model specified in `config.yaml` (default: `gemma2:9b`) is available.
- **Neo4j**: Required for graph storage and alignment. Follow the directions in the UCKG repo to download and run the UCKG: https://github.com/edogdu/UCKG

## Installation

1. Clone the repository:
```bash
git clone https://github.com/edogdu/CTI.git ./CTI
cd CTI
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in editable mode:
```bash
pip install -e .
```

## Configuration

### `config/config.yaml`

Configure global settings:
- **paths**: Output and dataset directories
- **dataset**: Dataset name and file path (e.g., `dnrti` or `cti-hal`)
- **models**: LLM model names for extraction and embeddings
- **ollama**: Ollama base URL
- **neo4j**: Neo4j connection details

### `config/pipeline.yaml`

Enable or disable pipeline stages:
```yaml
modules:
  semantic_chunking: false
  repair: false
  canonicalization: false
  graph_insertion: true
  similarity: true
  reranking: true
```

## Usage

### Running the Pipeline

**Option 1: Direct execution**
```bash
python main.py
```

**Option 2: Module execution**
```bash
python -m cti_analysis.pipeline
```

**Option 3: Using the script**
```bash
python scripts/run_pipeline.py
```

### Running Tests

Run the DNRTI integration test:
```bash
pytest tests/test_pipeline_dnrti.py -s
```

The `-s` flag shows print statements during test execution.

## Project Layout

```
CTI-Analysis/
├── README.md                    # This file
├── pyproject.toml              # Package configuration
├── requirements.txt            # Python dependencies
├── main.py                     # Thin entrypoint → calls cti_analysis.pipeline
├── config/
│   ├── config.yaml             # Global configuration
│   └── pipeline.yaml           # Stage toggles
├── datasets/                    # Dataset files (DNRTI, CTI-HAL, etc.)
├── src/
│   └── cti_analysis/
│       ├── __init__.py
│       ├── pipeline.py         # Main orchestrator
│       ├── config.py            # Config loading and typed dataclasses
│       ├── models/              # IR dataclasses (see models/README.md)
│       │   ├── documents.py     # RawDocument, NormalizedDocument, Chunk
│       │   ├── triples.py       # Triple, TripleBatch
│       │   ├── graph.py         # NodeIR, EdgeIR, GraphInsertBatch
│       │   └── scores.py        # SimilarityScore, RerankResult
│       ├── data_normalization/
│       │   └── normalize.py     # run_normalization()
│       ├── triple_extraction/
│       │   ├── semantic_chunking/
│       │   │   └── chunker.py   # run_chunking()
│       │   ├── extraction/
│       │   │   └── extractor.py  # run_extraction()
│       │   ├── triple_repair/
│       │   │   └── repair.py     # run_repair_ir()
│       │   └── canonicalization/
│       │       └── canonicalizer.py  # run_canonicalization_ir()
│       ├── graph_alignment/
│       │   ├── graph_insertion/
│       │   │   └── inserter.py   # run_insertion()
│       │   ├── similarity_scoring/
│       │   │   └── scorer.py     # run_scoring()
│       │   └── reranking/
│       │       └── reranker.py   # run_reranking()
│       ├── analysis/            # Metrics and reports
│       └── utils/               # Utility modules (io, db, dataset, id)
├── results/                     # Runtime artifacts (auto-generated)
│   ├── triple_extraction/
│   ├── graph_alignment/
│   └── analysis/
├── scripts/                     # Helper CLIs
│   └── run_pipeline.py
└── tests/                       # Test suite
    └── test_pipeline_dnrti.py
```

## Datasets

### CTI-HAL

The CTI-HAL dataset contains 81 CTI PDF documents annotated by human Cybersecurity analysts. Annotations label MITRE ATT&CK entities including Techniques, Subtechniques, Tactics, and Softwares, providing ground truth for evaluation.

**Source**: https://github.com/dessertlab/CTI-HAL

### DNRTI

The DNRTI dataset is a JSON file containing sentences with pre-annotated entities and relationships in STIX 2.1 format. Each entry includes:
- `text`: The sentence text
- `entities`: List of entity annotations
- `relations`: List of relationship annotations
- `ent_labels`: Entity type labels

## Goal

Our target is a fully automated pipeline that handles ingestion of CTI material in various formats, structures it into a document-level knowledge graph, and aligns that graph with known Cybersecurity entities in a managed "source" graph (UCKG).

<br>
![Image visualizing the workflow of CTI-Align](/resources/images/CTI-Align.png "CTI-Align")
<br>