#!/usr/bin/env python3
"""
Tumeteor Secondary Benchmark Evaluation
=========================================
Evaluates the CTI-to-ATT&CK reranker on the tüMETEOR / Security-TTP-Mapping
dataset as an independent generalization test.

Two evaluations:
  1. Single-stage model (checkpoints/best/) — PURE GENERALIZATION
     This model was trained only on CTI-HAL data and never saw tumeteor.
     Performance here shows whether the model generalizes to unseen data
     from a different annotation source.

  2. Two-stage model (checkpoints/best_two_stage/) — RETENTION TEST
     This model was pre-trained on tumeteor then fine-tuned on CTI-HAL.
     Performance here shows whether CTI-HAL fine-tuning caused
     catastrophic forgetting of tumeteor patterns.

Usage:
  python tumeteor_eval.py

  Expects these files:
    data/reranker_pairs_enriched_tumeteor.jsonl
    checkpoints/best/                    (single-stage, 87.67% on CTI-HAL)
    checkpoints/best_two_stage/          (two-stage, 91.1% on CTI-HAL)

Author: Shane Waldrop — Angelo State University / ARL Grant W911NF-24-2-0180
"""

import json
import random
import numpy as np
import sys
import time
import csv
import argparse
from collections import defaultdict
from pathlib import Path

RANDOM_SEED = 42


def load_and_group_data(filepath):
    """Load enriched JSONL and group by query_norm.
    Same structure as CTI-HAL enriched file."""
    query_data = defaultdict(lambda: {
        'query_raw': None,
        'actor': None,
        'candidates': [],
        'positive_count': 0,
        'negative_count': 0,
    })
    total_rows = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            query_norm = row.get('query_norm', '')
            query_raw = row.get('query_raw', '')
            candidate_text = row.get('candidate_text', '')
            candidate_id = row.get('candidate_id', '')
            candidate_norm = row.get('candidate_norm', '')
            label = row.get('label', 0)
            actor = row.get('actor', 'unknown')
            if not query_norm or not query_raw:
                continue
            if query_data[query_norm]['query_raw'] is None:
                query_data[query_norm]['query_raw'] = query_raw
                query_data[query_norm]['actor'] = actor if actor else 'unknown'
            query_data[query_norm]['candidates'].append({
                'text': candidate_text,
                'label': label,
                'id': candidate_norm if candidate_norm else candidate_id,
            })
            if label == 1:
                query_data[query_norm]['positive_count'] += 1
            else:
                query_data[query_norm]['negative_count'] += 1
    return dict(query_data), total_rows


def evaluate_model_on_data(model, query_data, label="Model"):
    """Evaluate a CrossEncoder model on all queries in query_data.
    Returns overall metrics and per-actor breakdown."""

    results_by_actor = defaultdict(list)
    all_results = []
    skipped = 0

    query_norms = [qn for qn, qd in query_data.items()
                   if qd['positive_count'] > 0 and len(qd['candidates']) > 0]

    print(f"\n  Evaluating {len(query_norms)} queries...")
    start_time = time.time()

    for i, qn in enumerate(query_norms):
        qd = query_data[qn]
        query_raw = qd['query_raw']
        actor = qd['actor']

        texts = [[query_raw, c['text']] for c in qd['candidates']]
        true_labels = [c['label'] for c in qd['candidates']]

        if not texts or sum(true_labels) == 0:
            skipped += 1
            continue

        # Score all candidates
        scores = model.predict(texts, show_progress_bar=False)

        # Rank by score descending
        ranked_indices = np.argsort(scores)[::-1]
        ranked_labels = [true_labels[j] for j in ranked_indices]

        p_at_1 = ranked_labels[0] if ranked_labels else 0
        hit_at_3 = 1 if 1 in ranked_labels[:3] else 0
        hit_at_5 = 1 if 1 in ranked_labels[:5] else 0

        result = {'p@1': p_at_1, 'hit@3': hit_at_3, 'hit@5': hit_at_5}
        all_results.append(result)
        results_by_actor[actor].append(result)

        # Progress every 100 queries
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(query_norms) - i - 1) / rate
            print(f"    [{i+1}/{len(query_norms)}] "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - start_time

    if not all_results:
        print(f"  [WARN] No valid queries to evaluate!")
        return None

    # Compute metrics
    overall_p1 = np.mean([r['p@1'] for r in all_results])
    overall_h3 = np.mean([r['hit@3'] for r in all_results])
    overall_h5 = np.mean([r['hit@5'] for r in all_results])
    n_correct = sum(r['p@1'] for r in all_results)
    n_total = len(all_results)

    print(f"\n  {label} RESULTS:")
    print(f"    P@1:     {overall_p1:.4f}  ({n_correct}/{n_total})")
    print(f"    Hit@3:   {overall_h3:.4f}")
    print(f"    Hit@5:   {overall_h5:.4f}")
    print(f"    Time:    {elapsed:.1f}s ({elapsed/n_total*1000:.0f}ms per query)")
    if skipped:
        print(f"    Skipped: {skipped} queries (no positives or no candidates)")

    # Per-actor breakdown
    actor_metrics = {}
    if len(results_by_actor) > 1:
        print(f"\n    Per-actor P@1:")
        print(f"    {'Actor':<20} | {'P@1':>8} | {'Correct':>8} | {'Total':>6}")
        print(f"    {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
        for actor in sorted(results_by_actor):
            results = results_by_actor[actor]
            c = sum(r['p@1'] for r in results)
            t = len(results)
            p1 = c / t if t > 0 else 0
            actor_metrics[actor] = {'p_at_1': p1, 'correct': int(c), 'total': t}
            print(f"    {actor:<20} | {p1:>8.4f} | {int(c):>8} | {t:>6}")
    else:
        # If there's only one actor group or no actor info
        for actor, results in results_by_actor.items():
            c = sum(r['p@1'] for r in results)
            t = len(results)
            actor_metrics[actor] = {'p_at_1': c/t if t > 0 else 0,
                                    'correct': int(c), 'total': t}

    return {
        'p_at_1': round(overall_p1, 4),
        'hit_at_3': round(overall_h3, 4),
        'hit_at_5': round(overall_h5, 4),
        'n_correct': int(n_correct),
        'n_total': n_total,
        'eval_time_s': round(elapsed, 1),
        'actor_metrics': actor_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tumeteor secondary benchmark evaluation")
    parser.add_argument('--data', default='data/reranker_pairs_enriched_tumeteor.jsonl',
                        help="Path to tumeteor enriched JSONL")
    parser.add_argument('--single-stage', default='checkpoints/best',
                        help="Path to single-stage model checkpoint")
    parser.add_argument('--two-stage', default='checkpoints/best_two_stage',
                        help="Path to two-stage model checkpoint")
    parser.add_argument('--skip-single', action='store_true',
                        help="Skip single-stage evaluation")
    parser.add_argument('--skip-two', action='store_true',
                        help="Skip two-stage evaluation")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  TUMETEOR SECONDARY BENCHMARK EVALUATION")
    print("  Independent Generalization Test")
    print("=" * 70)

    # Check data file
    if not Path(args.data).exists():
        sys.exit(f"[FAIL] Tumeteor data not found: {args.data}\n"
                 f"  Expected at: data/reranker_pairs_enriched_tumeteor.jsonl")

    # Load data
    print(f"\n  Loading tumeteor data from: {args.data}")
    query_data, total_rows = load_and_group_data(args.data)
    n_with_pos = sum(1 for qd in query_data.values() if qd['positive_count'] > 0)
    print(f"  Loaded {total_rows:,} rows, {len(query_data):,} unique queries")
    print(f"  Queries with gold labels: {n_with_pos}")

    # Show actor distribution
    actor_counts = defaultdict(int)
    for qd in query_data.values():
        actor_counts[qd['actor']] += 1
    print(f"\n  Actor/source distribution:")
    for actor in sorted(actor_counts):
        print(f"    {actor}: {actor_counts[actor]}")

    # Show sample of the data
    sample_qn = next(iter(query_data))
    sample = query_data[sample_qn]
    print(f"\n  Sample query: \"{sample['query_raw'][:80]}...\"")
    print(f"    Candidates: {len(sample['candidates'])}")
    print(f"    Positives:  {sample['positive_count']}")
    if sample['candidates']:
        print(f"    First candidate: {sample['candidates'][0]['text'][:60]}")

    # Import CrossEncoder
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        sys.exit("[FAIL] sentence-transformers not installed.\n"
                 "  pip install sentence-transformers")

    results = {}

    # ── Evaluation 1: Single-stage model (pure generalization) ──
    if not args.skip_single:
        if Path(args.single_stage).exists():
            print(f"\n  {'='*60}")
            print(f"  EVALUATION 1: SINGLE-STAGE MODEL (pure generalization)")
            print(f"  Model: {args.single_stage}")
            print(f"  This model NEVER saw tumeteor data during training.")
            print(f"  {'='*60}")

            model = CrossEncoder(args.single_stage)
            results['single_stage'] = evaluate_model_on_data(
                model, query_data, label="Single-stage (87.67% on CTI-HAL)")
            del model  # Free memory
        else:
            print(f"\n  [SKIP] Single-stage model not found at: {args.single_stage}")

    # ── Evaluation 2: Two-stage model (retention test) ──
    if not args.skip_two:
        if Path(args.two_stage).exists():
            print(f"\n  {'='*60}")
            print(f"  EVALUATION 2: TWO-STAGE MODEL (retention after fine-tuning)")
            print(f"  Model: {args.two_stage}")
            print(f"  This model was pre-trained on tumeteor, then fine-tuned on CTI-HAL.")
            print(f"  {'='*60}")

            model = CrossEncoder(args.two_stage)
            results['two_stage'] = evaluate_model_on_data(
                model, query_data, label="Two-stage (91.1% on CTI-HAL)")
            del model
        else:
            print(f"\n  [SKIP] Two-stage model not found at: {args.two_stage}")

    # ── Comparison summary ──
    if results:
        print(f"\n  {'='*60}")
        print(f"  COMPARISON SUMMARY")
        print(f"  {'='*60}")

        print(f"\n  {'Model':<35} | {'CTI-HAL P@1':>12} | {'Tumeteor P@1':>13}")
        print(f"  {'-'*35}-+-{'-'*12}-+-{'-'*13}")

        if 'single_stage' in results:
            r = results['single_stage']
            print(f"  {'Single-stage (generalization)':<35} | {'87.67%':>12} | "
                  f"{r['p_at_1']:>12.2%}")

        if 'two_stage' in results:
            r = results['two_stage']
            print(f"  {'Two-stage (retention)':<35} | {'91.10%':>12} | "
                  f"{r['p_at_1']:>12.2%}")

        # Paper-ready sentences
        print(f"\n  FOR THE PAPER:")
        if 'single_stage' in results:
            r = results['single_stage']
            print(f"  \"To assess cross-benchmark generalization, we evaluated the")
            print(f"   single-stage model on the Security-TTP-Mapping benchmark")
            print(f"   ({r['n_total']} queries) without any adaptation. The model")
            print(f"   achieved {r['p_at_1']:.1%} P@1, demonstrating [strong/moderate]")
            print(f"   transfer to an independent test set with different annotation")
            print(f"   conventions.\"")

        if 'two_stage' in results:
            r = results['two_stage']
            print(f"\n  \"The two-stage model, which included Security-TTP-Mapping")
            print(f"   data in its pre-training curriculum, achieved {r['p_at_1']:.1%}")
            print(f"   P@1 on this benchmark, confirming that CTI-HAL fine-tuning")
            print(f"   did not cause catastrophic forgetting of the broader")
            print(f"   threat intelligence patterns.\"")

        # Save results
        out_dir = Path("tumeteor_eval_results")
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_path = out_dir / "tumeteor_eval_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to: {summary_path}")

    print(f"\n  [DONE] Tumeteor evaluation complete.")


if __name__ == '__main__':
    main()
