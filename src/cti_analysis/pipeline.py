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
from cti_analysis.triple_extraction.semantic_chunking.chunker import run_chunking
from cti_analysis.triple_extraction.extraction.extractor import run_extraction
from cti_analysis.triple_extraction.triple_repair.repair import run_repair_ir
from cti_analysis.triple_extraction.canonicalization.canonicalizer import run_canonicalization_ir
from cti_analysis.graph_alignment.graph_insertion.inserter import run_insertion
from cti_analysis.graph_alignment.similarity_scoring.scorer import run_scoring
from cti_analysis.graph_alignment.reranking.reranker import run_reranking
from cti_analysis.utils.io import save_stage_outputs

__all__ = ["run_pipeline", "main"]


# -------------------------------
# Helpers
# -------------------------------
def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def load_raw_documents(cfg) -> List[RawDocument]:
    """
    Load documents based on dataset name.
    - dnrti: read the DNRTI JSON file into RawDocuments (one per entry)
    - default: read all .txt under data/raw
    """
    dataset = getattr(cfg, "dataset_name", "").lower()
    datasets_dir = Path(getattr(cfg, "datasets_dir", "datasets"))
    dataset_file = getattr(cfg, "dataset_file", None)
    dnrti_file = Path(dataset_file) if dataset_file else datasets_dir / "DNRTI" / "dnrti_aug_stix2_je.json"

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
# Orchestration
# -------------------------------
def run_pipeline() -> int:
    cfg = load_config()

    # 1) Load raw docs
    raw_docs = load_raw_documents(cfg)

    # 2) Normalize
    normalized_docs: List[NormalizedDocument] = run_normalization(cfg, raw_docs)

    # 3) Chunking
    if getattr(cfg.semantic_chunking, "enabled", True):
        chunks: List[Chunk] = run_chunking(cfg.semantic_chunking, normalized_docs)
    else:
        # Fallback: one chunk per document when chunking is disabled
        chunks = [chunk_from_text(doc, f"{doc.doc_id}_chunk0", doc.text) for doc in normalized_docs]
    save_stage_outputs("triple_extraction/semantic_chunking", chunks)

    # 4) Extraction
    triple_batches: List[TripleBatch] = run_extraction(cfg.extraction, chunks)  # type: ignore
    log(f"[Stage] Extraction done: batches={len(triple_batches)}")
    save_stage_outputs("triple_extraction/extraction", triple_batches)

    # 5) Repair
    if getattr(cfg.repair, "enabled", True):
        triple_batches = run_repair_ir(cfg.repair, triple_batches)
        save_stage_outputs("triple_extraction/triple_repair", triple_batches)

    # 6) Canonicalization
    if getattr(cfg.canonicalization, "enabled", True):
        triple_batches = run_canonicalization_ir(cfg.canonicalization, triple_batches)
        save_stage_outputs("triple_extraction/canonicalization", triple_batches)

    # 7) Graph insertion
    graph_batch: GraphInsertBatch | None = None
    if getattr(cfg.graph_insertion, "enabled", True):
        graph_batch = run_insertion(cfg.graph_insertion, triple_batches)
        save_stage_outputs("graph_alignment/graph_insertion", graph_batch)

    # 8) Similarity scoring
    scores: List[SimilarityScore] = []
    if getattr(cfg.similarity_scoring, "enabled", True):
        scores = run_scoring(cfg.similarity_scoring, triple_batches)
        save_stage_outputs("graph_alignment/similarity_scoring", scores)

    # 9) Reranking
    if getattr(cfg.reranking, "enabled", True) and scores:
        reranked: List[RerankResult] = run_reranking(cfg.reranking, scores)
        save_stage_outputs("graph_alignment/reranking", reranked)

    log("[Pipeline] completed")
    return 0


def main(argv: List[str] | None = None) -> int:
    return run_pipeline()


if __name__ == "__main__":
    raise SystemExit(main())

