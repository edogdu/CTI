#!/usr/bin/env python3
"""
Hard Negative Mining for CTI-to-ATT&CK Reranker
=================================================
Uses the current best model's own mistakes to create targeted training
signal. For each training query, we identify candidates that the model
scores highly despite being incorrect — these "hard negatives" represent
the model's blind spots (e.g., method-vs-purpose confusions like
ranking T1047/WMI above T1082/System Discovery).

The mined hard negatives are injected back into the training data as
additional rows, ensuring they survive random negative sampling during
the next training round.

Phase 1 (this script): Mine hard negatives on CPU (~3 min)
Phase 2 (Colab):       Retrain two-stage model on augmented data

Usage:
  python hard_negative_mining.py

  Expects:
    data/reranker_pairs_enriched_v2.jsonl
    checkpoints/best_two_stage_v2/

  Produces:
    data/reranker_pairs_enriched_v2_hardneg.jsonl  (augmented training data)
    hard_neg_results/mining_report.json             (analysis of mined negatives)

Author: Shane Waldrop — Angelo State University / ARL Grant W911NF-24-2-0180
"""

import json
import random
import re
import sys
import time
import numpy as np
from collections import defaultdict
from pathlib import Path

RANDOM_SEED = 42
ATTACK_ID_RE = re.compile(r'(TA\d{4}|T\d{4}(?:\.\d{3})?|S\d{4})')


def load_and_group_data(filepath):
    """Load enriched JSONL and group by query_norm, preserving raw rows."""
    query_data = defaultdict(lambda: {
        'query_raw': None,
        'actor': None,
        'candidates': [],
        'positive_count': 0,
    })
    # Also store all raw rows for reconstruction
    all_rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            all_rows.append(row)
            qn = row.get('query_norm', '')
            qr = row.get('query_raw', '')
            ct = row.get('candidate_text', '')
            ci = row.get('candidate_id', '')
            cn = row.get('candidate_norm', '')
            label = row.get('label', 0)
            actor = row.get('actor', 'unknown')
            if not qn or not qr:
                continue
            if query_data[qn]['query_raw'] is None:
                query_data[qn]['query_raw'] = qr
                query_data[qn]['actor'] = actor if actor else 'unknown'
            query_data[qn]['candidates'].append({
                'text': ct,
                'label': label,
                'id': cn if cn else ci,
                'candidate_id': ci,
                'candidate_norm': cn,
                'candidate_text': ct,
                'row_index': len(all_rows) - 1,  # track which raw row this is
            })
            if label == 1:
                query_data[qn]['positive_count'] += 1
    return dict(query_data), all_rows


def create_splits(query_data):
    """Create train/val/test splits matching finetune_production.py."""
    random.seed(RANDOM_SEED)
    queries_by_actor = defaultdict(list)
    for qn, data in query_data.items():
        actor = data['actor']
        if actor.startswith('external_'):
            continue
        if data['positive_count'] > 0:
            queries_by_actor[actor].append(qn)

    train_queries = []
    val_queries = []
    test_queries = []
    for actor, queries in queries_by_actor.items():
        random.shuffle(queries)
        n = len(queries)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train_queries.extend(queries[:n_train])
        val_queries.extend(queries[n_train:n_train + n_val])
        test_queries.extend(queries[n_train + n_val:])

    random.shuffle(train_queries)
    random.shuffle(val_queries)
    random.shuffle(test_queries)
    return train_queries, val_queries, test_queries


def extract_id(raw_id):
    """Extract clean ATT&CK ID."""
    match = ATTACK_ID_RE.match(raw_id)
    return match.group(1) if match else raw_id


def main():
    print("\n" + "=" * 70)
    print("  HARD NEGATIVE MINING")
    print("  Finding the model's confident mistakes in training data")
    print("=" * 70)

    data_path = "data/reranker_pairs_enriched_v2.jsonl"
    model_path = "checkpoints/best_two_stage_v2"
    output_path = "data/reranker_pairs_enriched_v2_hardneg.jsonl"

    if not Path(data_path).exists():
        sys.exit(f"[FAIL] Data not found: {data_path}")
    if not Path(model_path).exists():
        sys.exit(f"[FAIL] Model not found: {model_path}")

    # ── Load data ──
    print(f"\n  Loading data from: {data_path}")
    query_data, all_rows = load_and_group_data(data_path)
    train_queries, val_queries, test_queries = create_splits(query_data)
    print(f"  Total queries: {len(query_data)}")
    print(f"  Train: {len(train_queries)}  Val: {len(val_queries)}  Test: {len(test_queries)}")

    # ── Load model ──
    print(f"\n  Loading model from: {model_path}")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_path)
    print(f"  Model loaded.")

    # ── Mine hard negatives from training queries ──
    print(f"\n  Mining hard negatives from {len(train_queries)} training queries...")
    start = time.time()

    hard_negatives = []  # List of (query_norm, candidate_info, model_score, rank)
    train_errors = 0
    train_total = 0

    # Track statistics
    all_hard_neg_gaps = []
    all_hard_neg_types = defaultdict(int)

    for i, qn in enumerate(train_queries):
        qd = query_data[qn]
        if qd['positive_count'] == 0 or not qd['candidates']:
            continue

        query_raw = qd['query_raw']
        actor = qd['actor']

        # Score all candidates
        texts = [[query_raw, c['text']] for c in qd['candidates']]
        scores = model.predict(texts, show_progress_bar=False)

        # Build scored candidate list
        scored_candidates = []
        for j, c in enumerate(qd['candidates']):
            scored_candidates.append({
                'idx': j,
                'id': extract_id(c['id']),
                'score': float(scores[j]),
                'label': c['label'],
                'candidate_id': c['candidate_id'],
                'candidate_norm': c['candidate_norm'],
                'candidate_text': c['candidate_text'],
                'row_index': c['row_index'],
            })

        # Sort by score descending
        ranked = sorted(scored_candidates, key=lambda x: x['score'], reverse=True)
        train_total += 1

        # Check if top-1 is wrong (training error)
        if ranked[0]['label'] != 1:
            train_errors += 1

        # Find ALL hard negatives: wrong candidates that score above the
        # lowest-scoring gold candidate (i.e., they "outcompete" at least
        # one correct answer)
        gold_scores = [c['score'] for c in scored_candidates if c['label'] == 1]
        if not gold_scores:
            continue
        min_gold_score = min(gold_scores)
        max_gold_score = max(gold_scores)

        # A hard negative is a wrong candidate that:
        #   1. Scores higher than the minimum gold score, OR
        #   2. Is within 2.0 score points of the maximum gold score
        # This catches the "almost right" predictions that cause errors
        threshold = max(min_gold_score, max_gold_score - 2.0)

        for c in scored_candidates:
            if c['label'] == 0 and c['score'] >= threshold:
                gap = c['score'] - max_gold_score
                hard_negatives.append({
                    'query_norm': qn,
                    'query_raw': query_raw,
                    'actor': actor,
                    'candidate': c,
                    'score_gap': gap,
                    'max_gold_score': max_gold_score,
                })
                all_hard_neg_gaps.append(gap)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            print(f"    [{i+1}/{len(train_queries)}] "
                  f"{elapsed:.0f}s elapsed, "
                  f"{len(hard_negatives)} hard negatives found so far")

    elapsed = time.time() - start
    print(f"\n  Mining complete in {elapsed:.0f}s")
    print(f"  Training accuracy (current model): {(train_total-train_errors)/train_total:.2%} "
          f"({train_total-train_errors}/{train_total})")
    print(f"  Hard negatives found: {len(hard_negatives)}")

    if not hard_negatives:
        print("  [WARN] No hard negatives found! Model may be overfitting training data.")
        sys.exit(0)

    # ── Analyze hard negatives ──
    print(f"\n  {'='*60}")
    print(f"  HARD NEGATIVE ANALYSIS")
    print(f"  {'='*60}")

    # Score gap distribution
    gaps = np.array(all_hard_neg_gaps)
    print(f"\n  Score gap distribution (negative score - best gold score):")
    print(f"    Mean:   {np.mean(gaps):+.4f}")
    print(f"    Median: {np.median(gaps):+.4f}")
    print(f"    Min:    {np.min(gaps):+.4f}")
    print(f"    Max:    {np.max(gaps):+.4f}")

    # How many actually outscore all gold candidates?
    outscoring = sum(1 for g in all_hard_neg_gaps if g > 0)
    print(f"\n  Hard negatives that OUTSCORE best gold: {outscoring}/{len(hard_negatives)} "
          f"({outscoring/len(hard_negatives)*100:.1f}%)")

    # Per-actor distribution
    actor_counts = defaultdict(int)
    for hn in hard_negatives:
        actor_counts[hn['actor']] += 1
    print(f"\n  Per-actor hard negative counts:")
    for actor in sorted(actor_counts):
        print(f"    {actor:<14}: {actor_counts[actor]}")

    # Most commonly confused technique pairs
    confusion_pairs = defaultdict(int)
    for hn in hard_negatives:
        qn = hn['query_norm']
        qd = query_data[qn]
        gold_ids = set(extract_id(c['id']) for c in qd['candidates'] if c['label'] == 1)
        pred_id = hn['candidate']['id']
        for gid in gold_ids:
            if gid.startswith('T') or gid.startswith('S'):  # Skip tactics for clarity
                pair = f"{pred_id} confused with {gid}"
                confusion_pairs[pair] += 1

    print(f"\n  Top 15 most confused technique pairs:")
    for pair, count in sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"    {count:>3}x  {pair}")

    # ── Create augmented training data ──
    print(f"\n  {'='*60}")
    print(f"  CREATING AUGMENTED TRAINING DATA")
    print(f"  {'='*60}")

    # Strategy: duplicate each hard negative row 3x in the output file.
    # This ensures they survive random negative sampling during training.
    # The original data is preserved unchanged; we just add extra copies
    # of the hardest negatives.

    DUPLICATION_FACTOR = 3  # Each hard negative appears 3 additional times

    # Collect the row indices of hard negatives
    hard_neg_row_indices = set()
    hard_neg_rows = []
    for hn in hard_negatives:
        row_idx = hn['candidate']['row_index']
        if row_idx not in hard_neg_row_indices:
            hard_neg_row_indices.add(row_idx)
            # Get the original row and mark it as a hard negative
            row = all_rows[row_idx].copy()
            row['hard_negative'] = True
            row['hard_neg_score_gap'] = hn['score_gap']
            hard_neg_rows.append(row)

    print(f"  Unique hard negative rows: {len(hard_neg_rows)}")
    print(f"  Duplication factor: {DUPLICATION_FACTOR}x")
    print(f"  Additional rows added: {len(hard_neg_rows) * DUPLICATION_FACTOR}")

    # Write augmented file: original data + duplicated hard negatives
    with open(output_path, 'w', encoding='utf-8') as f:
        # First, write all original rows unchanged
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

        # Then append duplicated hard negatives
        for _ in range(DUPLICATION_FACTOR):
            for row in hard_neg_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    orig_size = Path(data_path).stat().st_size
    new_size = Path(output_path).stat().st_size
    print(f"\n  Original file: {orig_size/1024/1024:.1f} MB ({len(all_rows):,} rows)")
    print(f"  Augmented file: {new_size/1024/1024:.1f} MB "
          f"({len(all_rows) + len(hard_neg_rows)*DUPLICATION_FACTOR:,} rows)")

    # ── Save mining report ──
    out_dir = Path("hard_neg_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'mining_model': model_path,
        'train_queries': len(train_queries),
        'train_accuracy': round((train_total - train_errors) / train_total, 4),
        'hard_negatives_found': len(hard_negatives),
        'unique_hard_neg_rows': len(hard_neg_rows),
        'duplication_factor': DUPLICATION_FACTOR,
        'outscoring_count': outscoring,
        'score_gap_stats': {
            'mean': round(float(np.mean(gaps)), 4),
            'median': round(float(np.median(gaps)), 4),
            'min': round(float(np.min(gaps)), 4),
            'max': round(float(np.max(gaps)), 4),
        },
        'per_actor': dict(actor_counts),
        'top_confusion_pairs': [
            {'pair': p, 'count': c}
            for p, c in sorted(confusion_pairs.items(),
                              key=lambda x: x[1], reverse=True)[:30]
        ],
    }

    with open(out_dir / "mining_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  Mining report saved to: {out_dir / 'mining_report.json'}")

    print(f"\n  NEXT STEPS:")
    print(f"  1. Upload augmented file to Google Drive:")
    print(f"     {output_path}")
    print(f"  2. Run two-stage training on Colab with the augmented data")
    print(f"  3. Compare P@1 against the current 94.52% baseline")

    print(f"\n  [DONE] Hard negative mining complete.")


if __name__ == '__main__':
    main()
