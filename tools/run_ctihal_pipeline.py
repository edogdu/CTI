"""Run the full CTI pipeline on CTI-HAL PDF reports.

Usage:
    python tools/run_ctihal_pipeline.py [--limit N] [--group GROUP]

Automatically starts llama-server with the configured model and LoRA adapters.
Requires Neo4j running on localhost:7687 with UCKG loaded (unless --skip-graph).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "src"))
sys.path.insert(0, str(_repo))

from cti_analysis.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run CTI pipeline on CTI-HAL reports")
    parser.add_argument("--pdf", default=None, help="Process a single PDF file (any path)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N reports")
    parser.add_argument("--group", default=None, help="Only process this threat actor group")
    parser.add_argument("--skip-graph", action="store_true", help="Skip Neo4j graph stages")
    parser.add_argument("--no-server", action="store_true", help="Don't start llama-server (assume already running)")
    parser.add_argument("--experiment", default="raw", help="Experiment name (selects Neo4j database and output folder)")
    parser.add_argument("--start-from", type=int, default=0, help="Skip first N PDFs (resume from position)")
    args = parser.parse_args()

    cfg = load_config()
    cfg.dataset_name = "cti-hal"
    cfg.dataset_file = None

    if args.skip_graph:
        cfg.graph_insertion.enabled = False
        cfg.similarity_scoring.enabled = False

    # Override pipeline stage toggles based on experiment
    if args.experiment == "raw":
        cfg.semantic_chunking.enabled = False
        cfg.canonicalization.enabled = False
    elif args.experiment == "chunked":
        cfg.semantic_chunking.enabled = True
        cfg.canonicalization.enabled = False
    elif args.experiment == "canonicalized":
        cfg.semantic_chunking.enabled = False
        cfg.canonicalization.enabled = True
    elif args.experiment == "chunked-canonicalized":
        cfg.semantic_chunking.enabled = True
        cfg.canonicalization.enabled = True
    elif args.experiment == "demo":
        cfg.semantic_chunking.enabled = True
        cfg.canonicalization.enabled = True

    from cti_analysis.llm_backend import LlamaServer
    from cti_analysis.pipeline import _convert_pdf, _get_docling_converter, _process_unit, log
    from cti_analysis.models.documents import RawDocument, NormalizedDocument
    from cti_analysis.data_normalization.normalize import run_normalization
    from cti_analysis.graph_alignment.graph_insertion.inserter import (
        store_batches_in_neo4j, embed_graph_nodes,
    )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    # Map experiment names to folder names and Neo4j databases
    exp_config = {
        "raw":                    {"folder": "ctihal-pipeline",              "db": "neo4j"},
        "chunked":                {"folder": "ctihal-chunked",              "db": "chunked"},
        "canonicalized":          {"folder": "ctihal-canonicalized",        "db": "canonicalized"},
        "chunked-canonicalized":  {"folder": "ctihal-chunked-canonicalized","db": "chunkedcanon"},
    }
    ec = exp_config.get(args.experiment, {"folder": f"ctihal-{args.experiment}", "db": args.experiment})
    exp_folder = ec["folder"]
    exp_db = ec["db"]
    experiment_dir = _repo / "experiments" / exp_folder
    eval_dir = experiment_dir / "eval"
    triples_dir = eval_dir / "triples"
    similarity_dir = eval_dir / "similarity"
    triples_dir.mkdir(parents=True, exist_ok=True)
    similarity_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to file + stdout
    log_path = eval_dir / f"run_{run_id}.log"
    _log_file = open(log_path, "w", encoding="utf-8")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()
                except UnicodeEncodeError:
                    s.write(data.encode("ascii", errors="replace").decode("ascii"))
                    s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()
        def isatty(self):
            return False

    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)

    log(f"[CTI-HAL] Log: {log_path}")
    log(f"[CTI-HAL] Experiment '{args.experiment}': db={exp_db}, "
        f"chunking={cfg.semantic_chunking.enabled}, canonicalization={cfg.canonicalization.enabled}")
    log(f"[CTI-HAL] Run ID: {run_id}")

    # Find and filter PDFs BEFORE converting
    datasets_dir = Path(cfg.datasets_dir)
    reports_dir = datasets_dir / "CTI-HAL" / "reports"

    if args.pdf:
        # Single-PDF mode: accept any path
        pdf_path = Path(args.pdf).resolve()
        if not pdf_path.exists():
            log(f"[CTI-HAL] ERROR: PDF not found: {pdf_path}")
            sys.exit(1)
        pdfs = [pdf_path]
        # Use parent folder as group, or "demo" if not under reports_dir
        try:
            _group = pdf_path.relative_to(reports_dir).parts[0]
        except ValueError:
            _group = "demo"
        log(f"[CTI-HAL] Single-PDF mode: {pdf_path.name} (group={_group})")
    else:
        pdfs = sorted(reports_dir.rglob("*.pdf"))
        _group = None

        if args.group:
            pdfs = [p for p in pdfs if p.relative_to(reports_dir).parts[0] == args.group]
        if args.start_from:
            pdfs = pdfs[args.start_from:]
        if args.limit:
            pdfs = pdfs[:args.limit]

    log(f"[CTI-HAL] {len(pdfs)} PDFs to process")

    # Clear previous CTI data from Neo4j (each experiment has its own database)
    # SAFETY: never auto-clear the default 'neo4j' database (contains raw experiment + UCKG)
    # SAFETY: skip cleanup when resuming (--start-from)
    if cfg.graph_insertion.enabled and cfg.neo4j and exp_db != "neo4j" and not args.start_from:
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
                                          database=exp_db)
            with driver.session() as s:
                r1 = s.run("MATCH (n:CTISentence) DETACH DELETE n RETURN count(n) AS c")
                r2 = s.run("MATCH (n:CTIEntity) DETACH DELETE n RETURN count(n) AS c")
                r3 = s.run("MATCH (n:CTIDocument) DETACH DELETE n RETURN count(n) AS c")
                log(f"[CTI-HAL] Cleared Neo4j db '{exp_db}': {r1.single()['c']} sentences, "
                    f"{r2.single()['c']} entities, {r3.single()['c']} documents")
            driver.close()
        except Exception as e:
            log(f"[CTI-HAL] Neo4j clear error: {e}")
    elif exp_db == "neo4j":
        log("[CTI-HAL] Skipping Neo4j cleanup (default database protected)")

    # Start llama-server
    server = None
    if not args.no_server:
        server = LlamaServer.from_config(cfg.backend)
        log("[CTI-HAL] Starting llama-server...")
        server.start()
        log(f"[CTI-HAL] llama-server running at {server.url}")

    # Process one PDF at a time
    all_results = []
    for pi, pdf_path in enumerate(pdfs):
        import re as _re
        if args.pdf:
            group = _group
        else:
            group = pdf_path.relative_to(reports_dir).parts[0]
        # Sanitize doc_id: replace special chars with underscore for safe file paths
        safe_stem = _re.sub(r'[^\w\-.]', '_', pdf_path.stem)
        safe_stem = _re.sub(r'_+', '_', safe_stem).strip('_')
        doc_id = f"{group}_{safe_stem}"
        log(f"\n[CTI-HAL] === [{pi+1}/{len(pdfs)}] {group}/{pdf_path.name} ===")

        # 1) Convert PDF
        log(f"[CTI-HAL] Converting PDF...")
        try:
            docling_doc = _convert_pdf(pdf_path)
        except Exception as e:
            log(f"[CTI-HAL] Failed to convert: {e}")
            continue

        text = docling_doc.export_to_markdown()
        if not text.strip():
            log(f"[CTI-HAL] Empty text, skipping")
            continue

        # 2) Create RawDocument and normalize
        raw_doc = RawDocument(
            doc_id=doc_id,
            source_path=str(pdf_path),
            text=text,
            meta={"group": group, "pdf_name": pdf_path.name, "docling_doc": docling_doc},
        )
        normalized = run_normalization(cfg, [raw_doc])
        if not normalized:
            log(f"[CTI-HAL] No normalized documents, skipping")
            continue

        # 3) Process through pipeline (sentencize → extract → repair → canonicalize)
        doc = normalized[0]
        try:
            chunks, batches = _process_unit(cfg, doc, pi, len(pdfs))
        except Exception as e:
            log(f"[CTI-HAL] Error: {e}")
            import traceback
            traceback.print_exc()
            continue

        # 4) Save per-document triples
        n_triples = sum(len(b.triples) for b in batches)
        triples_data = []
        for b in batches:
            for t in b.triples:
                triples_data.append({
                    "subject": {"name": t.subject.name, "type": t.subject.type} if hasattr(t.subject, "name") else str(t.subject),
                    "predicate": t.predicate,
                    "object": {"name": t.object.name, "type": t.object.type} if hasattr(t.object, "name") else str(t.object),
                })

        doc_result = {
            "doc_id": doc_id,
            "group": group,
            "pdf_name": pdf_path.name,
            "n_chunks": len(chunks),
            "n_triples": n_triples,
            "triples": triples_data,
        }
        all_results.append(doc_result)

        out_path = triples_dir / f"{doc_id}.json"
        out_path.write_text(json.dumps(doc_result, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"[CTI-HAL] Saved {n_triples} triples → {out_path.name}")

        # 5) Graph insertion + similarity (per document)
        if cfg.graph_insertion.enabled and cfg.neo4j:
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
                                              database=exp_db)

                n_inserted = store_batches_in_neo4j(batches, driver, doc_id=doc_id)
                log(f"[CTI-HAL] {n_inserted} triples → Neo4j")

                n_embedded = embed_graph_nodes(
                    driver, model=cfg.embeddings.model,
                    strategy=cfg.embeddings.strategy,
                )
                log(f"[CTI-HAL] {n_embedded} entity nodes embedded")

                from cti_analysis.graph_alignment.sentence_retrieval.retriever import embed_sentences
                n_sent_emb = embed_sentences(driver, doc_id=doc_id, model=cfg.embeddings.model)
                log(f"[CTI-HAL] {n_sent_emb} sentence nodes embedded")

                if cfg.similarity_scoring.enabled:
                    from cti_analysis.graph_alignment.similarity_scoring.scorer import run_similarity
                    doc_sim_dir = similarity_dir / doc_id
                    run_similarity(driver, doc_id=doc_id, output_dir=str(doc_sim_dir), embed_cfg=cfg)
                    log(f"[CTI-HAL] Similarity -> {doc_sim_dir.name}/")

                # BM25 + regex token search against UCKG
                from cti_analysis.graph_alignment.token_search.searcher import run_bm25_search
                token_dir = eval_dir / "token_search" / doc_id
                ts_results = run_bm25_search(driver, doc_id=doc_id, output_dir=str(token_dir))
                log(f"[CTI-HAL] BM25 search: {ts_results['n_bm25']} IDs (bm25) + "
                    f"{ts_results['n_regex']} (regex) = {ts_results['n_combined']} combined")

                # Sentence-level vector search
                from cti_analysis.graph_alignment.sentence_search.searcher import run_sentence_search
                sent_dir = eval_dir / "sentence_search" / doc_id
                ss_results = run_sentence_search(
                    driver, doc_id=doc_id, output_dir=str(sent_dir),
                    model=cfg.embeddings.model,
                )
                log(f"[CTI-HAL] Sentence search: {ss_results['n_matched_ids']} IDs "
                    f"from {ss_results['n_sentences']} sentences")

                # Hybrid RRF fusion (entity vector + BM25 + sentence vector)
                from cti_analysis.graph_alignment.rrf import fuse_all_sources
                hybrid_dir = eval_dir / "hybrid" / doc_id
                rrf_results = fuse_all_sources(
                    similarity_dir=str(similarity_dir / doc_id),
                    token_search_dir=str(token_dir),
                    sentence_search_dir=str(sent_dir),
                    output_dir=str(hybrid_dir),
                )
                sc = rrf_results.get("source_counts", {})
                log(f"[CTI-HAL] Hybrid RRF: {rrf_results['n_fused']} fused IDs "
                    f"(3-source: {sc.get('all_three',0)}, "
                    f"2-source: {sc.get('two_sources',0)}, "
                    f"1-source: {sc.get('one_source',0)})")

                driver.close()
            except Exception as e:
                log(f"[CTI-HAL] Graph error: {e}")

        # Memory cleanup after each document
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Restart llama-server every 15 docs to prevent memory leak
        if server and (pi + 1) % 15 == 0 and (pi + 1) < len(pdfs):
            log(f"[CTI-HAL] Restarting llama-server (memory management, doc {pi+1}/{len(pdfs)})...")
            server.stop()
            import time as _time
            _time.sleep(3)
            server.start()
            log(f"[CTI-HAL] llama-server restarted")

    # Summary
    summary = {
        "run_id": run_id,
        "n_documents": len(all_results),
        "total_triples": sum(r["n_triples"] for r in all_results),
        "per_group": {},
    }
    for r in all_results:
        g = r["group"]
        if g not in summary["per_group"]:
            summary["per_group"][g] = {"n_docs": 0, "n_triples": 0}
        summary["per_group"][g]["n_docs"] += 1
        summary["per_group"][g]["n_triples"] += r["n_triples"]

    summary_path = eval_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n[CTI-HAL] Done: {len(all_results)} docs, {summary['total_triples']} total triples")
    log(f"[CTI-HAL] Summary: {summary_path}")

    if server is not None:
        server.stop()
        log("[CTI-HAL] llama-server stopped")

    _log_file.close()


if __name__ == "__main__":
    main()
