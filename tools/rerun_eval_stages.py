"""Re-run missing alignment evaluation stages for a given experiment config.

Iterates over all docs in the similarity/ folder (which is complete for all configs),
checks which downstream stages are missing, and runs them.

Usage:
    python tools/rerun_eval_stages.py --db neo4j --eval-dir experiments/ctihal-pipeline/eval
    python tools/rerun_eval_stages.py --db chunked --eval-dir experiments/ctihal-chunked/eval
    python tools/rerun_eval_stages.py --db canonicalized --eval-dir experiments/ctihal-markov/eval
    python tools/rerun_eval_stages.py --db chunkedcanon --eval-dir experiments/ctihal-chunked-markov/eval
"""
import argparse
import sys
from pathlib import Path

_repo = Path(__file__).parent.parent
sys.path.insert(0, str(_repo / "src"))

from cti_analysis.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Neo4j database name")
    parser.add_argument("--eval-dir", required=True, help="Experiment eval directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    cfg = load_config()
    eval_dir = Path(args.eval_dir)
    similarity_dir = eval_dir / "similarity"

    if not similarity_dir.exists():
        print(f"Error: {similarity_dir} not found")
        return

    # Get all doc IDs from similarity (the complete set)
    doc_ids = sorted(d.name for d in similarity_dir.iterdir() if d.is_dir())
    print(f"Database: {args.db}, Eval dir: {eval_dir}")
    print(f"Documents: {len(doc_ids)}")

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
                                  database=args.db)

    # Check which docs need which stages
    missing_token = []
    missing_sent = []
    missing_hybrid = []

    for doc_id in doc_ids:
        token_dir = eval_dir / "token_search" / doc_id
        sent_dir = eval_dir / "sentence_search" / doc_id
        hybrid_dir = eval_dir / "hybrid" / doc_id

        has_token = (token_dir / "token_search.json").exists()
        has_sent = (sent_dir / "sentence_search.json").exists()
        has_hybrid = (hybrid_dir / "hybrid_rrf.json").exists()

        if not has_token:
            missing_token.append(doc_id)
        if not has_sent:
            missing_sent.append(doc_id)
        if not has_hybrid:
            missing_hybrid.append(doc_id)

    print(f"\nMissing stages:")
    print(f"  token_search:    {len(missing_token)} docs")
    print(f"  sentence_search: {len(missing_sent)} docs")
    print(f"  hybrid:          {len(missing_hybrid)} docs")

    if args.dry_run:
        if missing_token:
            print(f"\nWould run token_search for: {missing_token[:5]}...")
        if missing_sent:
            print(f"Would run sentence_search for: {missing_sent[:5]}...")
        if missing_hybrid:
            print(f"Would run hybrid for: {missing_hybrid[:5]}...")
        return

    # Run missing token_search
    if missing_token:
        from cti_analysis.graph_alignment.token_search.searcher import run_bm25_search
        print(f"\n--- Running token_search for {len(missing_token)} docs ---")
        for i, doc_id in enumerate(missing_token):
            token_dir = eval_dir / "token_search" / doc_id
            try:
                with driver.session() as s:
                    r = s.run("MATCH (d:CTIDocument {id: $did}) RETURN d.id", did=doc_id).single()
                    if not r:
                        print(f"  [{i+1}/{len(missing_token)}] {doc_id}: NOT FOUND in db, skipping")
                        continue
                results = run_bm25_search(driver, doc_id=doc_id, output_dir=str(token_dir))
                print(f"  [{i+1}/{len(missing_token)}] {doc_id}: {results['n_combined']} IDs")
            except Exception as e:
                print(f"  [{i+1}/{len(missing_token)}] {doc_id}: ERROR {e}")

    # Run missing sentence_search
    if missing_sent:
        from cti_analysis.graph_alignment.sentence_search.searcher import run_sentence_search
        print(f"\n--- Running sentence_search for {len(missing_sent)} docs ---")
        for i, doc_id in enumerate(missing_sent):
            sent_dir = eval_dir / "sentence_search" / doc_id
            try:
                results = run_sentence_search(
                    driver, doc_id=doc_id, output_dir=str(sent_dir),
                    model=cfg.embeddings.model,
                )
                print(f"  [{i+1}/{len(missing_sent)}] {doc_id}: {results['n_matched_ids']} IDs")
            except Exception as e:
                print(f"  [{i+1}/{len(missing_sent)}] {doc_id}: ERROR {e}")

    # Run missing hybrid (needs token_search + sentence_search + similarity)
    if missing_hybrid:
        from cti_analysis.graph_alignment.rrf import fuse_all_sources
        print(f"\n--- Running hybrid RRF for {len(missing_hybrid)} docs ---")
        for i, doc_id in enumerate(missing_hybrid):
            sim_dir = eval_dir / "similarity" / doc_id
            token_dir = eval_dir / "token_search" / doc_id
            sent_dir = eval_dir / "sentence_search" / doc_id
            hybrid_dir = eval_dir / "hybrid" / doc_id

            # Check prerequisites
            if not (token_dir / "token_search.json").exists():
                print(f"  [{i+1}/{len(missing_hybrid)}] {doc_id}: missing token_search, skipping")
                continue
            if not (sent_dir / "sentence_search.json").exists():
                print(f"  [{i+1}/{len(missing_hybrid)}] {doc_id}: missing sentence_search, skipping")
                continue

            try:
                results = fuse_all_sources(
                    similarity_dir=str(sim_dir),
                    token_search_dir=str(token_dir),
                    sentence_search_dir=str(sent_dir),
                    output_dir=str(hybrid_dir),
                )
                sc = results.get("source_counts", {})
                print(f"  [{i+1}/{len(missing_hybrid)}] {doc_id}: {results['n_fused']} fused IDs")
            except Exception as e:
                print(f"  [{i+1}/{len(missing_hybrid)}] {doc_id}: ERROR {e}")

    driver.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
