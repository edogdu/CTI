from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Any

from cti_analysis.config import load_config
from cti_analysis.models.documents import RawDocument, NormalizedDocument, Chunk, chunk_from_text
from cti_analysis.models.triples import TripleBatch
from cti_analysis.models.scores import SimilarityScore, RerankResult
from cti_analysis.models.graph import GraphInsertBatch

from cti_analysis.data_normalization.normalize import run_normalization
from cti_analysis.triple_extraction.sentencizer import sentencize_document
from cti_analysis.triple_extraction.semantic_chunking.chunker import run_chunking
from cti_analysis.triple_extraction.extraction.extractor import run_extraction
from cti_analysis.triple_extraction.triple_repair.repair import run_repair_ir
from cti_analysis.triple_extraction.canonicalization.canonicalizer import run_canonicalization_ir
from cti_analysis.graph_alignment.graph_insertion.inserter import run_insertion
from cti_analysis.graph_alignment.similarity_scoring.scorer import run_scoring
from cti_analysis.graph_alignment.reranking.reranker import run_reranking
from cti_analysis.utils.io import save_stage_outputs, save_run_manifest

__all__ = ["run_pipeline", "main"]


# -------------------------------
# Helpers
# -------------------------------
def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


_docling_converter = None

def _get_docling_converter():
    """Lazy-init a shared DocumentConverter (avoids re-loading weights per PDF)."""
    global _docling_converter
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter
        _docling_converter = DocumentConverter()
    return _docling_converter


def _convert_pdf(pdf_path: Path):
    """Convert a PDF and return the docling document object with page structure."""
    converter = _get_docling_converter()
    result = converter.convert(str(pdf_path))
    return result.document


def load_raw_documents(cfg) -> List[RawDocument]:
    """
    Load documents based on dataset name.
    - dnrti: read the DNRTI JSON file into RawDocuments (one per entry)
    - cti-hal: extract text from PDF reports under datasets/CTI-HAL/reports/
    - default: read all .txt under data/raw
    """
    dataset = getattr(cfg, "dataset_name", "").lower()
    datasets_dir = Path(getattr(cfg, "datasets_dir", "datasets"))
    dataset_file = getattr(cfg, "dataset_file", None)

    if dataset == "cti-hal":
        reports_dir = datasets_dir / "CTI-HAL" / "reports"
        if not reports_dir.exists():
            log(f"[Load] CTI-HAL reports dir missing: {reports_dir}")
            return []
        pdfs = sorted(reports_dir.rglob("*.pdf"))
        log(f"[Load] Found {len(pdfs)} PDFs in {reports_dir}")
        docs: List[RawDocument] = []
        for pi, pdf_path in enumerate(pdfs):
            group = pdf_path.relative_to(reports_dir).parts[0]
            doc_id = f"{group}_{pdf_path.stem}"
            log(f"[Load] [{pi+1}/{len(pdfs)}] Converting {group}/{pdf_path.name}...")
            try:
                docling_doc = _convert_pdf(pdf_path)
            except Exception as e:
                log(f"[Load] Failed to convert {pdf_path.name}: {e}")
                continue
            # Store full markdown as text, but pass docling doc object for
            # page-level sentencization in normalize.py
            text = docling_doc.export_to_markdown()
            if not text.strip():
                log(f"[Load] Empty text from {pdf_path.name}, skipping")
                continue
            docs.append(RawDocument(
                doc_id=doc_id,
                source_path=str(pdf_path),
                text=text,
                meta={"group": group, "pdf_name": pdf_path.name,
                      "docling_doc": docling_doc},
            ))
        log(f"[Load] loaded {len(docs)} CTI-HAL docs from {reports_dir}")
        return docs

    dnrti_file = Path(dataset_file) if dataset_file else datasets_dir / "DNRTI" / "dnrti_aug_stix2_je_subset.json"

    if dataset == "dnrti" or dnrti_file.exists():
        if not dnrti_file.exists():
            log(f"[Load] DNRTI file missing: {dnrti_file}")
            return []
        import json

        data = json.loads(dnrti_file.read_text(encoding="utf-8"))
        docs: List[RawDocument] = []
        for i, entry in enumerate(data):
            text = entry.get("text", "")
            entities = entry.get("entities") or []
            relations = entry.get("relations") or []
            ent_labels = sorted({e[3] for e in entities if isinstance(e, list) and len(e) >= 4})
            docs.append(
                RawDocument(
                    doc_id=f"dnrti_{i}",
                    source_path=str(dnrti_file),
                    text=text,
                    meta={
                        "entities": entities,
                        "relations": relations,
                        "ent_labels": ent_labels,
                        "expected_entities": entities,
                        "expected_relations": relations,
                        "index": i,
                    },
                )
            )
        log(f"[Load] loaded {len(docs)} DNRTI docs from {dnrti_file}")
        return docs

    raw_dir = Path("data/raw")
    docs: List[RawDocument] = []
    if not raw_dir.exists():
        log(f"[Load] raw dir missing: {raw_dir}")
        return docs
    for path in raw_dir.rglob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        doc_id = path.stem
        docs.append(RawDocument(doc_id=doc_id, source_path=str(path), text=text, meta={"path": str(path)}))
    log(f"[Load] loaded {len(docs)} raw docs from {raw_dir}")
    return docs


# -------------------------------
# Per-unit processing
# -------------------------------
def _process_unit(
    cfg,
    doc: NormalizedDocument,
    unit_idx: int,
    total_units: int,
) -> tuple[List[Chunk], List[TripleBatch]]:
    """Process a single unit (sentence or document) through all triple extraction stages.

    Returns (chunks, triple_batches) for this unit.
    """
    pfx = f"[{unit_idx + 1}/{total_units} {doc.doc_id}]"

    # 1) Sentencize: split document into sentence-level chunks
    has_docling = isinstance(doc.meta, dict) and "docling_doc" in doc.meta
    if has_docling:
        log(f"{pfx} Sentencizing...")
        chunks: List[Chunk] = sentencize_document(doc)
        log(f"{pfx} {len(chunks)} sentences after filtering")
    else:
        chunks = [chunk_from_text(doc, f"{doc.doc_id}_chunk0", doc.text)]

    if not chunks:
        log(f"{pfx} No relevant sentences, skipping")
        return [], []

    # 2) Semantic chunking (optional): merge sentence chunks into groups
    if getattr(cfg.semantic_chunking, "enabled", False):
        log(f"{pfx} Semantic chunking ({len(chunks)} sentences -> merging)...")
        chunks = run_chunking(cfg.semantic_chunking, [doc], sentence_chunks=chunks)
        log(f"{pfx} {len(chunks)} chunks after merging")

    # 3) Extract triples
    log(f"{pfx} Extracting triples from {len(chunks)} chunks...")
    cfg.extraction._backend_cfg = cfg.backend
    triple_batches: List[TripleBatch] = run_extraction(cfg.extraction, chunks)
    n_triples = sum(len(b.triples) for b in triple_batches)
    log(f"{pfx} Extracted {n_triples} triples")

    # 4) Repair
    if getattr(cfg.repair, "enabled", True):
        log(f"{pfx} Repairing...")
        triple_batches = run_repair_ir(cfg.repair, triple_batches)

    # 5) Canonicalize
    if getattr(cfg.canonicalization, "enabled", True):
        log(f"{pfx} Canonicalizing...")
        triple_batches = run_canonicalization_ir(cfg.canonicalization, triple_batches)

    n_triples = sum(len(b.triples) for b in triple_batches)
    log(f"{pfx} Done: {len(chunks)} chunks, {n_triples} triples")

    return chunks, triple_batches


# -------------------------------
# Orchestration
# -------------------------------
def run_pipeline() -> int:
    cfg = load_config()

    # Generate run_id for this pipeline execution (shared across all stages)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log(f"[Pipeline] Starting run_id={run_id}")
    save_run_manifest(run_id, cfg)

    # 1) Load raw docs
    raw_docs = load_raw_documents(cfg)

    # 2) Normalize
    normalized_docs: List[NormalizedDocument] = run_normalization(cfg, raw_docs)

    # 3) Process each document through extraction stages
    all_chunks: List[Chunk] = []
    all_triple_batches: List[TripleBatch] = []

    for i, doc in enumerate(normalized_docs):
        chunks, batches = _process_unit(cfg, doc, i, len(normalized_docs))
        all_chunks.extend(chunks)
        all_triple_batches.extend(batches)

    # 4) Save extraction stage outputs
    save_stage_outputs("triple_extraction/semantic_chunking", all_chunks, run_id=run_id)
    save_stage_outputs("triple_extraction/extraction", all_triple_batches, run_id=run_id)

    total_triples = sum(len(b.triples) for b in all_triple_batches)
    log(f"[Pipeline] Extraction stages complete: {len(all_triple_batches)} batches, {total_triples} triples")

    # 5) Graph alignment stages (always operate across all units)
    graph_batch: GraphInsertBatch | None = None
    if getattr(cfg.graph_insertion, "enabled", True):
        graph_batch = run_insertion(cfg.graph_insertion, all_triple_batches)
        save_stage_outputs("graph_alignment/graph_insertion", graph_batch, run_id=run_id)

        # Write to Neo4j if configured
        if cfg.neo4j and graph_batch:
            from cti_analysis.graph_alignment.graph_insertion.inserter import (
                store_batches_in_neo4j, embed_graph_nodes,
            )
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
                log("[Pipeline] Inserting triples into Neo4j...")
                store_batches_in_neo4j(all_triple_batches, driver)
                log("[Pipeline] Embedding CTIEntity nodes...")
                embed_graph_nodes(driver, model=cfg.embeddings.model,
                                  strategy=cfg.embeddings.strategy)
                driver.close()
                log("[Pipeline] Neo4j insertion + embedding complete")
            except ImportError:
                log("[Pipeline] neo4j package not installed, skipping graph write")
            except Exception as e:
                log(f"[Pipeline] Neo4j error: {e}")

    scores: List[SimilarityScore] = []
    if getattr(cfg.similarity_scoring, "enabled", True) and cfg.neo4j:
        from cti_analysis.graph_alignment.similarity_scoring.scorer import run_similarity
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
            # Run similarity for each unique document
            doc_ids = list({b.doc_id for b in all_triple_batches})
            for doc_id in doc_ids:
                sim_output_dir = Path(cfg.output_dir) / "graph_alignment" / "similarity_scoring" / run_id / doc_id
                log(f"[Pipeline] Running similarity search for {doc_id}...")
                run_similarity(driver, doc_id=doc_id, output_dir=str(sim_output_dir), embed_cfg=cfg)
            driver.close()
        except ImportError:
            log("[Pipeline] neo4j package not installed, skipping similarity")
        except Exception as e:
            log(f"[Pipeline] Similarity scoring error: {e}")

    if getattr(cfg.reranking, "enabled", True) and scores:
        reranked: List[RerankResult] = run_reranking(cfg.reranking, scores)
        save_stage_outputs("graph_alignment/reranking", reranked, run_id=run_id)

    log("[Pipeline] completed")
    return 0


def main(argv: List[str] | None = None) -> int:
    return run_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())
