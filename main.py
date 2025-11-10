from pathlib import Path
import os
import yaml
import time
import traceback
import sys
from util.db import (
    get_driver,
    ensure_constraints,
    clear_cti_graph as db_clear_cti_graph,
    document_exists as db_document_exists,
)

# Extra utility imports
from util.dataset import ensure_repo, enumerate_cti_hal
from util.id import compute_document_id
from util.io import (
    save_json as io_save_json,
    load_json as io_load_json,
    load_progress,
    save_progress,
    append_manifest_entry,
)

from config.config import load_config

def _clean_state(workdir: Path):
    # remove progress file
    progress_path = workdir / "progress.json"
    try:
        if progress_path.exists():
            progress_path.unlink()
            log(f"[Clean] Removed progress file: {progress_path}")
    except Exception as e:
        log(f"[Clean] Warning: could not remove progress file: {e}")

    # rotate manifest if present
    man_path = workdir / "manifest.json"
    try:
        if man_path.exists():
            ts = time.strftime("%Y%m%d-%H%M%S")
            backup = man_path.with_name(f"manifest.{ts}.bak.json")
            man_path.rename(backup)
            log(f"[Clean] Rotated manifest to: {backup}")
    except Exception as e:
        log(f"[Clean] Warning: could not rotate manifest: {e}")

def stage_extraction(input_path: Path, document_id: str, model: str, ollama_base_url: str, out_dir: Path) -> Path:
    from triple_extraction import extraction as extraction  # local module

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem.replace(" ", "_")
    chunk_json = out_dir / f"chunk_data_{stem}_{model.replace(':','_')}.json"

    log(f"[Extraction] model={model}, file={input_path}")
    extractor = extraction.CyberTripleExtractor(str(input_path), document_id=document_id, model_name=model, ollama_base_url=ollama_base_url)
    
    raw_chunk_results = extractor.run()
    extractor.build_dict(raw_chunk_results)

    if hasattr(extractor, "save_to_json"):
        extractor.save_to_json(str(chunk_json))  # preferred path if available
    else:
        # fallback: try common attribute name; otherwise just dump raw
        chunk_data = getattr(extractor, "chunk_data", raw_chunk_results)
        io_save_json(chunk_data, chunk_json)

    log(f"[Extraction] wrote {chunk_json}")
    return chunk_json

def stage_insertion(chunk_json_path: Path, driver) -> None:
    import graph_alignment.insertion as insertion  # local module

    chunk_obj = io_load_json(chunk_json_path) 
    
    log("[Insertion] inserting triples into Neo4j...")
    insertion.store_in_neo4j(
        chunk_obj,
        driver
    )
    log("[Insertion] complete")

def stage_embed_cti_entities(chunk_json_path: Path, driver, model: str, ollama_base_url: str) -> None:
    import graph_alignment.insertion as insertion  # local module

    chunk_obj = io_load_json(chunk_json_path)
    log("[Embedding] embedding CTIEntity nodes...")
    insertion.embed_cti_entities_from_chunk(
        chunk_obj,
        driver,
        model=model,
        ollama_url=f"{ollama_base_url.rstrip('/')}/api/embeddings",
    )
    log("[Embedding] CTIEntity nodes embedded")

# Similarity 
def stage_similarity(driver, document_id: str, sim_output_dir: Path, embed_cfg) -> None:
    # set per-run output dir for similarity_scoring
    os.environ["SIM_OUTPUT_DIR"] = str(sim_output_dir)
    from graph_alignment.similarity_scoring import run_similarity
    log(f"[Similarity] running vector top-k scoring via Neo4j index... -> {sim_output_dir}")
    run_similarity(driver, document_id, sim_output_dir, embed_cfg)
    log("[Similarity] results saved under similarity_scoring outputs/")

def process_single_pdf(pdf: Path, group: str, ann_L: Path, ann_S: Path, cfg: dict, driver) -> dict:
    out_dir = Path(cfg.paths.output_dir) / cfg.dataset.name / group / pdf.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    started_ts = time.strftime("%Y-%m-%d %H:%M:%S")

    document_id = compute_document_id(pdf)
    document_props = {
        "id": document_id,
        "path": str(pdf),
        "group": group,
        "name": pdf.stem,
        "created_ts": started_ts,
    }

    # Extract -> chunk json in out_dir
    chunk_json = stage_extraction(pdf, document_id, cfg.models.extraction_model, cfg.ollama.base_url, out_dir)

    # Insert into Neo4j
    stage_insertion(chunk_json, driver)

    # Embed CTIEntity nodes after insertion
    stage_embed_cti_entities(chunk_json, driver, cfg.models.embedding_model, cfg.ollama.base_url)

    # Similarity: per-PDF results go to out_dir/vec_results
    sim_dir = out_dir / cfg.embedding.similarity.dir
    stage_similarity(driver, document_id, sim_dir, cfg.embedding)

    return {
        "pdf": str(pdf),
        "group": group,
        "chunk_json": str(chunk_json),
        "similarity_dir": str(sim_dir),
        "document_id": document_id,
        "document_props": document_props,
        "annotations": {
            "annotator_L": str(ann_L),
            "annotator_S": str(ann_S),
        },
        "started": started_ts,
    }

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def main() -> int:
    try:
        cfg = load_config()

        WORKDIR = Path(cfg.paths.output_dir).resolve()
        DATASETS_DIR = Path(cfg.paths.datasets_dir).resolve()
        REPO_DIR = DATASETS_DIR / cfg.dataset.name
        WORKDIR_REPO = WORKDIR / cfg.dataset.name

        log(f"[Config] Workdir: {WORKDIR}")
        log(f"[Config] Datasets dir: {DATASETS_DIR}")
        log(f"[Config] CTI-HAL repo: {REPO_DIR}")
        log(f"[Config] Models: extract={cfg.models.extraction_model}, embed={cfg.models.embedding_model}")
        log(f"[Config] Ollama: {cfg.ollama.base_url}")
        log(f"[Config] Neo4j URI: {cfg.neo4j.uri} user: {cfg.neo4j.user}")

        driver = get_driver(cfg.neo4j.uri, cfg.neo4j.user, cfg.neo4j.password)
        try:
            if "--clean" in sys.argv:
                log("[Clean] Starting clean run: resetting progress and manifest...")
                _clean_state(WORKDIR_REPO)
            if "--fresh-graph" in sys.argv:
                log("[Fresh] Clearing :CTIEntity and :CTIDocument nodes per --fresh-graph…")
                db_clear_cti_graph(driver)

            ensure_constraints(driver)

            ensure_repo(cfg.dataset.repo, REPO_DIR)

            PROGRESS_PATH = WORKDIR / cfg.dataset.name / "progress.json"
            progress = load_progress(PROGRESS_PATH)
            completed_set = set(progress.get("completed", []))
            failed_entries = progress.get("failed", [])
            man_path = WORKDIR / cfg.dataset.name / "manifest.json"

            count = 0

            for pdf, group, ann_L, ann_S in enumerate_cti_hal(REPO_DIR):
                if not pdf.exists():
                    continue
                rel_pdf = str(pdf.relative_to(REPO_DIR))  # resume key
                if rel_pdf in completed_set:
                    log(f"[Skip] Already completed: {rel_pdf}")
                    continue

                # Skip if CTI-Document already exists in the graph
                doc_id = compute_document_id(pdf)
                if db_document_exists(driver, doc_id):
                    log(f"[Skip] CTIDocument already exists (id={doc_id}): {rel_pdf}")
                    continue

                log(f"[Pipeline] Processing PDF: {rel_pdf}")
                try:
                    entry = process_single_pdf(pdf, group, ann_L, ann_S, cfg, driver)
                    count += 1

                    # mark completed and persist progress after EACH file
                    completed_set.add(rel_pdf)
                    progress["completed"] = sorted(list(completed_set))
                    save_progress(progress, PROGRESS_PATH)
                    append_manifest_entry(man_path, REPO_DIR, entry)

                except KeyboardInterrupt:
                    log("[Stop] Interrupted by user. Saving progress...")
                    progress["completed"] = sorted(list(completed_set))
                    save_progress(progress, PROGRESS_PATH)
                    break
                except Exception as e:
                    log(f"[Error] Failed on {rel_pdf}: {e}")
                    traceback.print_exc()
                    failed_entries.append({"pdf": rel_pdf, "error": str(e)})
                    progress["failed"] = failed_entries
                    save_progress(progress, PROGRESS_PATH)
                    # continue to next file

            log(f"Completed {count} new PDFs in this session. Manifest: {man_path}")
            log(f"Progress file: {PROGRESS_PATH}")
            return 0
        finally:
            driver.close()

    except Exception as e:
        log("Pipeline failed: " + str(e))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Optional flag: `python main.py --clean` to reset progress & rotate manifest
    raise SystemExit(main())