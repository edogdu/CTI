#!/usr/bin/env python3
"""
eval_diagnostic.py — Detailed model evaluation with miss analysis
==================================================================
Loads a saved model and evaluates against CTI-HAL test split.
Reports P@1, Hit@3, Hit@5 per actor, plus detailed miss analysis
showing what the model predicted vs. the correct answer.

Usage:
    python eval_diagnostic.py
    python eval_diagnostic.py --model checkpoints/best_two_stage
    python eval_diagnostic.py --data data/reranker_pairs_enriched.jsonl

Branch: dataset-expansion
Author: Shane Waldrop
"""

import json
import random
import numpy as np
import argparse
import csv
import os
from collections import defaultdict

RANDOM_SEED = 42


def load_data(filepath):
    """Load and organize data by query (same logic as finetune_production.py)."""
    query_data = defaultdict(lambda: {
        'query_raw': None,
        'actor': None,
        'candidates': [],
        'positive_count': 0,
        'negative_count': 0
    })

    total = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line.strip())
            total += 1

            query_norm = row.get('query_norm', '').strip()
            query_raw = row.get('query_raw', '').strip()
            candidate_text = row.get('candidate_text', '').strip()
            candidate_id = row.get('candidate_id', '').strip()
            label = int(row.get('label', 0))
            actor = row.get('actor', 'unknown').strip()

            if not query_norm:
                continue

            # Skip external data (tumeteor, zenodo)
            if actor.startswith('external_'):
                continue

            query_data[query_norm]['query_raw'] = query_raw
            query_data[query_norm]['actor'] = actor
            query_data[query_norm]['candidates'].append({
                'text': candidate_text,
                'id': candidate_id,
                'label': label
            })

            if label == 1:
                query_data[query_norm]['positive_count'] += 1
            else:
                query_data[query_norm]['negative_count'] += 1

    print(f"Loaded {total:,} rows, {len(query_data)} unique CTI-HAL queries")
    return dict(query_data)


def create_test_split(query_data, train_ratio=0.8, val_ratio=0.1):
    """Recreate the exact same split as finetune_production.py."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    queries_by_actor = defaultdict(list)
    for query_norm, data in query_data.items():
        if data['positive_count'] > 0:
            queries_by_actor[data['actor']].append(query_norm)

    train_queries = []
    val_queries = []
    test_queries = []

    for actor, queries in queries_by_actor.items():
        random.shuffle(queries)
        n_queries = len(queries)
        n_train = int(n_queries * train_ratio)
        n_val = int(n_queries * val_ratio)

        train_queries.extend(queries[:n_train])
        val_queries.extend(queries[n_train:n_train + n_val])
        test_queries.extend(queries[n_train + n_val:])

    random.shuffle(train_queries)
    random.shuffle(val_queries)
    random.shuffle(test_queries)

    # Count per actor in test
    actor_counts = defaultdict(int)
    for q in test_queries:
        actor_counts[query_data[q]['actor']] += 1

    print(f"\nTest split: {len(test_queries)} queries")
    for actor, count in sorted(actor_counts.items()):
        print(f"  {actor}: {count}")

    return test_queries


def evaluate(model, test_queries, query_data):
    """
    Full evaluation with P@1, Hit@3, Hit@5 per actor,
    plus detailed miss analysis.
    """
    from sentence_transformers import CrossEncoder

    results_by_actor = defaultdict(list)
    all_misses = []

    for i, query_norm in enumerate(test_queries):
        if query_norm not in query_data:
            continue

        data = query_data[query_norm]
        query_raw = data['query_raw']
        actor = data['actor']

        texts = [[query_raw, c['text']] for c in data['candidates']]
        true_labels = [c['label'] for c in data['candidates']]
        candidate_ids = [c['id'] for c in data['candidates']]
        candidate_texts = [c['text'] for c in data['candidates']]

        if not texts or sum(true_labels) == 0:
            continue

        scores = model.predict(texts, show_progress_bar=False)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_labels = [true_labels[idx] for idx in ranked_indices]
        ranked_ids = [candidate_ids[idx] for idx in ranked_indices]
        ranked_texts = [candidate_texts[idx] for idx in ranked_indices]
        ranked_scores = [float(scores[idx]) for idx in ranked_indices]

        p_at_1 = ranked_labels[0]
        hit_at_3 = 1 if 1 in ranked_labels[:3] else 0
        hit_at_5 = 1 if 1 in ranked_labels[:5] else 0

        # Find gold position
        gold_rank = None
        for r, lab in enumerate(ranked_labels):
            if lab == 1:
                gold_rank = r + 1  # 1-indexed
                break

        # Find gold ID
        gold_ids = [candidate_ids[j] for j, lab in enumerate(true_labels) if lab == 1]

        results_by_actor[actor].append({
            'query_norm': query_norm,
            'query_raw': query_raw,
            'p@1': p_at_1,
            'hit@3': hit_at_3,
            'hit@5': hit_at_5,
            'gold_rank': gold_rank,
            'gold_ids': gold_ids,
            'predicted_id': ranked_ids[0],
            'predicted_text': ranked_texts[0],
            'predicted_score': ranked_scores[0],
            'top3_ids': ranked_ids[:3],
            'top3_scores': ranked_scores[:3],
            'top5_ids': ranked_ids[:5],
            'top5_scores': ranked_scores[:5],
        })

        if p_at_1 == 0:
            all_misses.append({
                'actor': actor,
                'query_raw': query_raw[:120],
                'gold_ids': gold_ids,
                'predicted_id': ranked_ids[0],
                'predicted_text': ranked_texts[0],
                'gold_rank': gold_rank,
                'top3_ids': ranked_ids[:3],
                'top3_scores': [f"{s:.4f}" for s in ranked_scores[:3]],
                'top5_ids': ranked_ids[:5],
            })

    return results_by_actor, all_misses


def print_results(results_by_actor, all_misses, save_dir):
    """Print formatted results and miss analysis."""

    print("\n" + "=" * 80)
    print("  FULL EVALUATION RESULTS")
    print("=" * 80)

    # Per-actor table
    print(f"\n{'Actor':<15} | {'P@1':>7} | {'Hit@3':>7} | {'Hit@5':>7} | {'N':>5}")
    print("-" * 55)

    all_p1, all_h3, all_h5, all_n = [], [], [], 0
    rows_for_csv = []

    for actor in sorted(results_by_actor.keys()):
        results = results_by_actor[actor]
        n = len(results)
        p1 = np.mean([r['p@1'] for r in results])
        h3 = np.mean([r['hit@3'] for r in results])
        h5 = np.mean([r['hit@5'] for r in results])

        print(f"{actor:<15} | {p1:>7.4f} | {h3:>7.4f} | {h5:>7.4f} | {n:>5}")

        all_p1.extend([r['p@1'] for r in results])
        all_h3.extend([r['hit@3'] for r in results])
        all_h5.extend([r['hit@5'] for r in results])
        all_n += n

        rows_for_csv.append({
            'actor': actor, 'p@1': f"{p1:.4f}",
            'hit@3': f"{h3:.4f}", 'hit@5': f"{h5:.4f}",
            'n_queries': n
        })

    print("-" * 55)
    overall_p1 = np.mean(all_p1)
    overall_h3 = np.mean(all_h3)
    overall_h5 = np.mean(all_h5)
    print(f"{'OVERALL':<15} | {overall_p1:>7.4f} | {overall_h3:>7.4f} | {overall_h5:>7.4f} | {all_n:>5}")

    # Save CSV
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "eval_full_metrics.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['actor', 'p@1', 'hit@3', 'hit@5', 'n_queries'])
        writer.writeheader()
        writer.writerows(rows_for_csv)
    print(f"\n  Saved metrics to {csv_path}")

    # Miss analysis
    print("\n" + "=" * 80)
    print(f"  MISS ANALYSIS ({len(all_misses)} misses)")
    print("=" * 80)

    # Categorize misses
    parent_child = 0
    sibling = 0
    unrelated = 0
    in_top3 = 0
    in_top5 = 0

    for miss in all_misses:
        gold_id = miss['gold_ids'][0] if miss['gold_ids'] else ""
        pred_id = miss['predicted_id']

        # Check if parent/child confusion
        if '.' in gold_id and pred_id == gold_id.split('.')[0]:
            parent_child += 1
        elif '.' in pred_id and gold_id == pred_id.split('.')[0]:
            parent_child += 1
        elif '.' in gold_id and '.' in pred_id and gold_id.split('.')[0] == pred_id.split('.')[0]:
            sibling += 1
        else:
            unrelated += 1

        if miss['gold_rank'] and miss['gold_rank'] <= 3:
            in_top3 += 1
        if miss['gold_rank'] and miss['gold_rank'] <= 5:
            in_top5 += 1

    print(f"\n  Error Type Breakdown:")
    print(f"    Parent/child confusion: {parent_child} ({parent_child/max(len(all_misses),1)*100:.0f}%)")
    print(f"    Sibling confusion:      {sibling} ({sibling/max(len(all_misses),1)*100:.0f}%)")
    print(f"    Unrelated:              {unrelated} ({unrelated/max(len(all_misses),1)*100:.0f}%)")

    print(f"\n  Recovery Analysis:")
    print(f"    Gold in top 3: {in_top3}/{len(all_misses)} ({in_top3/max(len(all_misses),1)*100:.0f}%)")
    print(f"    Gold in top 5: {in_top5}/{len(all_misses)} ({in_top5/max(len(all_misses),1)*100:.0f}%)")

    print(f"\n  {'#':<3} {'Actor':<12} {'Gold':<14} {'Predicted':<14} {'Gold@':<6} {'Query (truncated)'}")
    print("  " + "-" * 95)

    for i, miss in enumerate(sorted(all_misses, key=lambda x: x['actor']), 1):
        gold = miss['gold_ids'][0] if miss['gold_ids'] else "?"
        pred = miss['predicted_id']
        rank = str(miss['gold_rank']) if miss['gold_rank'] else ">N"
        query = miss['query_raw'][:60]
        print(f"  {i:<3} {miss['actor']:<12} {gold:<14} {pred:<14} {rank:<6} {query}")

    # Save miss details
    miss_path = os.path.join(save_dir, "eval_misses.csv")
    with open(miss_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'actor', 'gold_id', 'predicted_id', 'predicted_text',
            'gold_rank', 'top3_ids', 'top3_scores', 'query'
        ])
        writer.writeheader()
        for miss in sorted(all_misses, key=lambda x: x['actor']):
            writer.writerow({
                'actor': miss['actor'],
                'gold_id': ';'.join(miss['gold_ids']),
                'predicted_id': miss['predicted_id'],
                'predicted_text': miss['predicted_text'],
                'gold_rank': miss['gold_rank'],
                'top3_ids': ';'.join(miss['top3_ids']),
                'top3_scores': ';'.join(miss['top3_scores']),
                'query': miss['query_raw'],
            })
    print(f"\n  Saved miss details to {miss_path}")

    # Hard negative recommendation
    print("\n" + "=" * 80)
    print("  HARD NEGATIVE RECOMMENDATION")
    print("=" * 80)
    pct_structural = (parent_child + sibling) / max(len(all_misses), 1) * 100
    if pct_structural >= 30:
        print(f"\n  ✅ {pct_structural:.0f}% of misses are parent/child/sibling confusion.")
        print(f"     Hard negative mining would LIKELY help significantly.")
        print(f"     These {parent_child + sibling} errors are exactly what hard negatives target.")
    elif pct_structural >= 15:
        print(f"\n  ⚠️  {pct_structural:.0f}% of misses are structural confusion.")
        print(f"     Hard negatives would help some, but other improvements needed too.")
    else:
        print(f"\n  ❌ Only {pct_structural:.0f}% structural confusion.")
        print(f"     Hard negatives alone won't move the needle much.")
        print(f"     Consider: more diverse training data or different base model.")


def main():
    parser = argparse.ArgumentParser(description="Diagnostic model evaluation")
    parser.add_argument("--model", default="checkpoints/best_two_stage",
                        help="Path to saved model directory")
    parser.add_argument("--data", default="data/reranker_pairs_enriched.jsonl",
                        help="Path to CTI-HAL JSONL data")
    parser.add_argument("--save-dir", default="eval_results",
                        help="Where to save results")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  DIAGNOSTIC EVALUATION")
    print(f"  Model: {args.model}")
    print(f"  Data:  {args.data}")
    print("=" * 80)

    # Load data (CTI-HAL only)
    print("\n[1/3] Loading data...")
    query_data = load_data(args.data)

    # Recreate test split
    print("\n[2/3] Creating test split (same seed as training)...")
    test_queries = create_test_split(query_data)

    # Load model and evaluate
    print(f"\n[3/3] Loading model and evaluating ({len(test_queries)} test queries)...")
    print("  (This may take a few minutes on CPU)\n")

    from sentence_transformers import CrossEncoder
    model = CrossEncoder(args.model, max_length=512)

    results_by_actor, all_misses = evaluate(model, test_queries, query_data)
    print_results(results_by_actor, all_misses, args.save_dir)


if __name__ == "__main__":
    main()
