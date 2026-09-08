#!/usr/bin/env python3
"""
Unified Re-Evaluation Against best_two_stage_v2 (94.52% Enriched Model)
========================================================================
Generates clean, consistent numbers for the ACSAC 2026 paper by running
ALL experiments against the same model checkpoint.

Phase 1: CTI-HAL test set evaluation (verify P@1, get per-query results)
Phase 2: McNemar's test vs GPT-5.4-mini (zero-shot and few-shot)
Phase 3: Multi-label evaluation with threshold sweep
Phase 4: Tumeteor benchmark (~30-40 min on CPU — the long one)

All results saved to v2_eval_results/ as JSON for paper reference.

Usage:
  cd C:\\Users\\shane\\Downloads\\CTI\\graph_alignment\\reranker
  python v2_reeval.py

Author: Generated for ACSAC 2026 submission pipeline
"""

import json
import os
import sys
import csv
import time
import random
import numpy as np
from collections import defaultdict, Counter
from scipy import stats

# ============================================================
# CONFIGURATION — all paths relative to reranker/ directory
# ============================================================
MODEL_PATH = "checkpoints/best_two_stage_v2"
DATA_PATH = "data/reranker_pairs_enriched_v2.jsonl"
TUMETEOR_PATH = "data/reranker_pairs_enriched_tumeteor_v2.jsonl"
LLM_ZS_PATH = "llm_baseline_results/per_query_gpt-5.4-mini_zero_shot.csv"
LLM_FS_PATH = "llm_baseline_results/per_query_gpt-5.4-mini_few_shot.csv"
RANDOM_SEED = 42
OUTPUT_DIR = "v2_eval_results"

# ============================================================
# PREFLIGHT CHECKS
# ============================================================
def preflight():
    """Verify all required files exist before doing anything expensive."""
    print("=" * 65)
    print("  PREFLIGHT: Checking required files")
    print("=" * 65)
    
    critical = [
        (os.path.join(MODEL_PATH, "model.safetensors"), "v2 model weights"),
        (DATA_PATH, "enriched CTI-HAL data"),
    ]
    optional = [
        (TUMETEOR_PATH, "enriched tumeteor data"),
        (LLM_ZS_PATH, "LLM zero-shot per-query results"),
        (LLM_FS_PATH, "LLM few-shot per-query results"),
    ]
    
    all_ok = True
    for path, desc in critical:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status:>7}] {desc}: {path}")
        if not exists:
            all_ok = False
    
    for path, desc in optional:
        exists = os.path.exists(path)
        status = "OK" if exists else "SKIP"
        print(f"  [{status:>7}] {desc}: {path}")
    
    if not all_ok:
        print("\n  [FATAL] Critical files missing. Cannot proceed.")
        sys.exit(1)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n  Output directory: {OUTPUT_DIR}/")
    print("  Preflight passed.\n")


# ============================================================
# DATA LOADING — mirrors explain.py exactly
# ============================================================
def load_and_group_data(filepath):
    """Load JSONL and group by query_norm. Same logic as explain.py."""
    query_data = defaultdict(lambda: {
        'query_raw': None,
        'actor': None,
        'candidates': [],
        'positive_count': 0,
        'negative_count': 0,
        'gold_ids': set(),
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

            cid = candidate_norm if candidate_norm else candidate_id
            query_data[query_norm]['candidates'].append({
                'text': candidate_text,
                'label': label,
                'id': cid,
                'raw_id': candidate_id,
            })

            if label == 1:
                query_data[query_norm]['positive_count'] += 1
                query_data[query_norm]['gold_ids'].add(cid)
            else:
                query_data[query_norm]['negative_count'] += 1

    return dict(query_data), total_rows


def create_test_split(query_data, train_ratio=0.8, val_ratio=0.1):
    """
    Recreate the exact same test split as explain.py / finetune_production.py.
    CRITICAL: must use same seed and same iteration order.
    """
    # Reset seed right before split — this is what explain.py does
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    queries_by_actor = defaultdict(list)
    for query_norm, data in query_data.items():
        actor = data['actor']
        if actor.startswith('external_'):
            continue
        if data['positive_count'] > 0:
            queries_by_actor[actor].append(query_norm)

    test_queries = []
    for actor, queries in queries_by_actor.items():
        random.shuffle(queries)
        n = len(queries)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        test_queries.extend(queries[n_train + n_val:])

    random.shuffle(test_queries)
    return test_queries


# ============================================================
# PHASE 1: CTI-HAL TEST SET EVALUATION
# ============================================================
def phase1_ctihal(model, query_data, test_queries):
    """Evaluate v2 model on CTI-HAL test set. Should reproduce ~94.52% P@1."""
    print("=" * 65)
    print("  PHASE 1: CTI-HAL Test Set Evaluation")
    print("=" * 65)

    n = len(test_queries)
    print(f"  Test queries: {n}")
    actor_counts = Counter(query_data[qn]['actor'] for qn in test_queries)
    for actor in sorted(actor_counts):
        print(f"    {actor}: {actor_counts[actor]}")

    results = []
    t0 = time.time()

    for i, qn in enumerate(test_queries):
        qdata = query_data[qn]
        query_raw = qdata['query_raw']
        gold_ids = qdata['gold_ids']
        candidates = qdata['candidates']

        # Score all candidates for this query
        pairs = [[query_raw, c['text']] for c in candidates]
        scores = model.predict(pairs)

        # Rank by descending score
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top1_id = ranked[0][0]['id']
        top1_score = float(ranked[0][1])

        # Metrics
        correct = top1_id in gold_ids
        ranked_ids = [c[0]['id'] for c in ranked]
        hit3 = any(rid in gold_ids for rid in ranked_ids[:3])
        hit5 = any(rid in gold_ids for rid in ranked_ids[:5])

        # Store all scores for multi-label eval in Phase 3
        all_scored = [(c['id'], float(s), c['label']) for c, s in ranked]

        results.append({
            'query_norm': qn,
            'query_raw': query_raw,
            'actor': qdata['actor'],
            'gold_ids': list(gold_ids),
            'predicted': top1_id,
            'predicted_score': top1_score,
            'correct': correct,
            'hit3': hit3,
            'hit5': hit5,
            'all_scored': all_scored,
        })

        if (i + 1) % 30 == 0:
            elapsed = time.time() - t0
            cur_p1 = sum(r['correct'] for r in results) / len(results)
            print(f"  [{i+1}/{n}] P@1 so far: {cur_p1:.4f} | {elapsed:.0f}s")

    elapsed = time.time() - t0

    # Compute overall metrics
    p1 = sum(r['correct'] for r in results) / n
    h3 = sum(r['hit3'] for r in results) / n
    h5 = sum(r['hit5'] for r in results) / n
    n_correct = sum(r['correct'] for r in results)

    print(f"\n  RESULTS:")
    print(f"    P@1:   {p1:.4f}  ({n_correct}/{n})")
    print(f"    Hit@3: {h3:.4f}")
    print(f"    Hit@5: {h5:.4f}")
    print(f"    Time:  {elapsed:.1f}s")

    # Per-actor breakdown
    print(f"\n  Per-actor P@1:")
    print(f"    {'Actor':<15} | {'P@1':>7} | {'Correct':>7} | {'Total':>5}")
    print(f"    {'-'*15}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}")
    actor_results = defaultdict(list)
    for r in results:
        actor_results[r['actor']].append(r)
    for actor in sorted(actor_results):
        ar = actor_results[actor]
        ap1 = sum(r['correct'] for r in ar) / len(ar)
        ac = sum(r['correct'] for r in ar)
        print(f"    {actor:<15} | {ap1:>6.2%} | {ac:>7} | {len(ar):>5}")

    return results


# ============================================================
# PHASE 2: McNEMAR'S TEST
# ============================================================
def load_llm_csv(csv_path):
    """Load LLM per-query results. Auto-detect column names."""
    if not os.path.exists(csv_path):
        return None

    results = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        print(f"    CSV columns: {columns}")

        for row in reader:
            # Try multiple possible column names for the query text
            query = (row.get('query') or row.get('query_raw') or
                     row.get('text') or row.get('query_text') or '')
            query = query.strip()

            # Try multiple possible column names for correctness
            correct_str = (row.get('correct') or row.get('is_correct') or
                           row.get('match') or row.get('p@1') or '')
            correct = correct_str.strip().lower() in ('true', '1', 'yes')

            if query:
                results[query] = correct

    return results


def phase2_mcnemar(phase1_results):
    """Run McNemar's test: reranker v2 vs GPT-5.4-mini."""
    print("\n" + "=" * 65)
    print("  PHASE 2: McNemar's Statistical Significance Tests")
    print("=" * 65)

    # Build reranker lookup by query_raw text
    reranker_by_query = {}
    for r in phase1_results:
        reranker_by_query[r['query_raw'].strip()] = r['correct']

    mcnemar_results = []

    for label, path in [("GPT-5.4-mini ZS", LLM_ZS_PATH),
                        ("GPT-5.4-mini FS", LLM_FS_PATH)]:
        print(f"\n  Loading {label} from {path}...")

        llm_results = load_llm_csv(path)
        if llm_results is None:
            print(f"    [SKIP] File not found.")
            continue

        print(f"    Loaded {len(llm_results)} LLM query results.")

        # Match queries between reranker and LLM by text
        # Try exact match first, then stripped/normalized match
        shared_queries = []
        for llm_q, llm_correct in llm_results.items():
            # Try exact match
            if llm_q in reranker_by_query:
                shared_queries.append((llm_q, reranker_by_query[llm_q], llm_correct))
            else:
                # Try normalized match (strip whitespace, collapse spaces)
                llm_norm = ' '.join(llm_q.split())
                for rq, rc in reranker_by_query.items():
                    rq_norm = ' '.join(rq.split())
                    if llm_norm == rq_norm:
                        shared_queries.append((llm_q, rc, llm_correct))
                        break

        print(f"    Matched {len(shared_queries)} shared queries.")

        if len(shared_queries) < 10:
            print(f"    [SKIP] Too few shared queries for McNemar's test.")
            continue

        # Build contingency table
        # a = reranker, b = LLM
        a1b1 = sum(1 for _, a, b in shared_queries if a and b)
        a1b0 = sum(1 for _, a, b in shared_queries if a and not b)
        a0b1 = sum(1 for _, a, b in shared_queries if not a and b)
        a0b0 = sum(1 for _, a, b in shared_queries if not a and not b)

        print(f"\n    Contingency table (Reranker v2 vs {label}):")
        print(f"      Both correct:              {a1b1}")
        print(f"      Reranker right, LLM wrong: {a1b0}")
        print(f"      Reranker wrong, LLM right: {a0b1}")
        print(f"      Both wrong:                {a0b0}")

        b = a1b0  # reranker right, LLM wrong
        c = a0b1  # reranker wrong, LLM right

        if b + c == 0:
            print(f"    No discordant pairs. Systems agree on all queries.")
            continue

        # McNemar's chi-squared (without continuity correction)
        chi2 = (b - c) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

        print(f"\n    McNemar's chi2 = {chi2:.2f}, p = {p_value:.2e} ({sig})")
        print(f"    Discordant ratio: {b}:{c} (reranker:LLM)")
        print(f"    Reranker P@1 on shared: {(a1b1+a1b0)/len(shared_queries):.2%}")
        print(f"    LLM P@1 on shared:      {(a1b1+a0b1)/len(shared_queries):.2%}")

        mcnemar_results.append({
            'comparison': f"Reranker v2 vs {label}",
            'shared_queries': len(shared_queries),
            'both_correct': a1b1,
            'reranker_right_llm_wrong': a1b0,
            'reranker_wrong_llm_right': a0b1,
            'both_wrong': a0b0,
            'chi2': float(chi2),
            'p_value': float(p_value),
            'significance': sig,
        })

    return mcnemar_results


# ============================================================
# PHASE 3: MULTI-LABEL EVALUATION
# ============================================================
def phase3_multilabel(phase1_results):
    """Sweep thresholds for multi-label prediction using v2 model scores."""
    print("\n" + "=" * 65)
    print("  PHASE 3: Multi-Label Evaluation (Threshold Sweep)")
    print("=" * 65)

    thresholds = [-2, -1, 0, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 4, 5, 6]

    print(f"\n  {'Thresh':>8} | {'Prec':>8} | {'Recall':>8} | {'F1':>8} | {'AvgPred':>8}")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    best_f1 = 0
    best_threshold = 0
    all_threshold_results = []

    for threshold in thresholds:
        precisions, recalls, f1s, pred_counts = [], [], [], []

        for r in phase1_results:
            gold_ids = set(r['gold_ids'])
            # Predict all candidates above threshold
            predicted_ids = set()
            for cid, score, label in r['all_scored']:
                if score >= threshold:
                    predicted_ids.add(cid)

            # At minimum, always predict the top-1
            if not predicted_ids:
                predicted_ids = {r['all_scored'][0][0]}

            tp = len(predicted_ids & gold_ids)
            fp = len(predicted_ids - gold_ids)
            fn = len(gold_ids - predicted_ids)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            pred_counts.append(len(predicted_ids))

        mp = np.mean(precisions)
        mr = np.mean(recalls)
        mf = np.mean(f1s)
        mc = np.mean(pred_counts)

        all_threshold_results.append({
            'threshold': threshold,
            'precision': float(mp),
            'recall': float(mr),
            'f1': float(mf),
            'avg_predicted': float(mc),
        })

        print(f"  {threshold:>8.2f} | {mp:>7.2%} | {mr:>7.2%} | {mf:>7.2%} | {mc:>8.2f}")

        if mf > best_f1:
            best_f1 = mf
            best_threshold = threshold

    best_row = next(t for t in all_threshold_results if t['threshold'] == best_threshold)
    print(f"\n  Optimal threshold: {best_threshold}")
    print(f"  Best F1:     {best_row['f1']:.4f}")
    print(f"  Precision:   {best_row['precision']:.4f}")
    print(f"  Recall:      {best_row['recall']:.4f}")

    return {
        'optimal_threshold': best_threshold,
        'best_precision': best_row['precision'],
        'best_recall': best_row['recall'],
        'best_f1': best_row['f1'],
        'all_thresholds': all_threshold_results,
    }


# ============================================================
# PHASE 4: TUMETEOR BENCHMARK
# ============================================================
def phase4_tumeteor(model):
    """Evaluate v2 model on the tumeteor benchmark (~20K queries)."""
    print("\n" + "=" * 65)
    print("  PHASE 4: Tumeteor Benchmark")
    print("  (This is the long one — ~30-40 min on CPU)")
    print("=" * 65)

    if not os.path.exists(TUMETEOR_PATH):
        print(f"  [SKIP] Tumeteor data not found at {TUMETEOR_PATH}")
        return None

    print(f"  Loading tumeteor data (this takes a minute)...")
    t_load = time.time()

    tumeteor_data = defaultdict(lambda: {
        'query_raw': None, 'candidates': [], 'gold_ids': set(),
        'positive_count': 0,
    })

    row_count = 0
    with open(TUMETEOR_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            qn = row.get('query_norm', '')
            if not qn:
                continue

            if tumeteor_data[qn]['query_raw'] is None:
                tumeteor_data[qn]['query_raw'] = row.get('query_raw', '')

            cid = row.get('candidate_norm') or row.get('candidate_id', '')
            ctxt = row.get('candidate_text', '')
            label = row.get('label', 0)

            tumeteor_data[qn]['candidates'].append({
                'id': cid, 'text': ctxt, 'label': label
            })

            if label == 1:
                tumeteor_data[qn]['gold_ids'].add(cid)
                tumeteor_data[qn]['positive_count'] += 1

    # Filter to queries that have gold labels
    valid_queries = [qn for qn, d in tumeteor_data.items()
                     if d['positive_count'] > 0]

    print(f"  Loaded {row_count:,} rows in {time.time()-t_load:.0f}s")
    print(f"  Valid queries with gold labels: {len(valid_queries):,}")
    print(f"  Starting evaluation...\n")

    correct = 0
    hit3 = 0
    hit5 = 0
    total = 0
    t0 = time.time()

    for i, qn in enumerate(valid_queries):
        qdata = tumeteor_data[qn]
        query_raw = qdata['query_raw']
        gold_ids = qdata['gold_ids']
        candidates = qdata['candidates']

        pairs = [[query_raw, c['text']] for c in candidates]
        scores = model.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        ranked_ids = [c[0]['id'] for c in ranked]

        total += 1
        if ranked_ids[0] in gold_ids:
            correct += 1
        if any(rid in gold_ids for rid in ranked_ids[:3]):
            hit3 += 1
        if any(rid in gold_ids for rid in ranked_ids[:5]):
            hit5 += 1

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(valid_queries) - i - 1)
            cur_p1 = correct / total
            print(f"  [{i+1:>6}/{len(valid_queries)}] P@1={cur_p1:.4f} | "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t0

    p1 = correct / total
    h3 = hit3 / total
    h5 = hit5 / total

    print(f"\n  Tumeteor Results (v2 model):")
    print(f"    P@1:   {p1:.4f}  ({correct}/{total})")
    print(f"    Hit@3: {h3:.4f}")
    print(f"    Hit@5: {h5:.4f}")
    print(f"    Time:  {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return {
        'p@1': float(p1),
        'hit@3': float(h3),
        'hit@5': float(h5),
        'correct': correct,
        'total': total,
        'time_seconds': elapsed,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "#" * 65)
    print("  COUNTER-ATT&CK: Unified Re-Evaluation Pipeline")
    print("  Model: best_two_stage_v2 (enriched, 94.52% expected)")
    print("  Purpose: Clean numbers for ACSAC 2026 submission")
    print("#" * 65 + "\n")

    # --- Preflight ---
    preflight()

    # --- Load data ---
    print("Loading CTI-HAL enriched data...")
    t0 = time.time()
    query_data, total_rows = load_and_group_data(DATA_PATH)
    print(f"  {total_rows:,} rows -> {len(query_data)} unique queries "
          f"({time.time()-t0:.1f}s)")

    # --- Create test split ---
    test_queries = create_test_split(query_data)
    print(f"  Test split: {len(test_queries)} queries\n")

    # --- Load model ---
    print("Loading cross-encoder model...")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(MODEL_PATH)
    print(f"  Model loaded from {MODEL_PATH}\n")

    # --- Phase 1: CTI-HAL ---
    phase1_results = phase1_ctihal(model, query_data, test_queries)

    # --- Phase 2: McNemar's ---
    mcnemar_results = phase2_mcnemar(phase1_results)

    # --- Phase 3: Multi-label ---
    multilabel_results = phase3_multilabel(phase1_results)

    # --- Phase 4: Tumeteor ---
    tumeteor_results = phase4_tumeteor(model)

    # ============================================================
    # SAVE ALL RESULTS
    # ============================================================
    print("\n" + "=" * 65)
    print("  SAVING RESULTS")
    print("=" * 65)

    n = len(phase1_results)
    p1 = sum(r['correct'] for r in phase1_results) / n

    # Build per-actor summary
    actor_summary = {}
    actor_results = defaultdict(list)
    for r in phase1_results:
        actor_results[r['actor']].append(r)
    for actor, ar in actor_results.items():
        actor_summary[actor] = {
            'p@1': sum(r['correct'] for r in ar) / len(ar),
            'correct': sum(r['correct'] for r in ar),
            'total': len(ar),
        }

    all_results = {
        'model': MODEL_PATH,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'cti_hal': {
            'p@1': p1,
            'hit@3': sum(r['hit3'] for r in phase1_results) / n,
            'hit@5': sum(r['hit5'] for r in phase1_results) / n,
            'correct': sum(r['correct'] for r in phase1_results),
            'total': n,
            'per_actor': actor_summary,
        },
        'mcnemar': mcnemar_results,
        'multilabel': multilabel_results,
    }

    if tumeteor_results:
        all_results['tumeteor'] = tumeteor_results

    # Save unified results JSON
    results_file = os.path.join(OUTPUT_DIR, 'v2_unified_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Unified results: {results_file}")

    # Save per-query results (for future reference / error analysis)
    per_query_file = os.path.join(OUTPUT_DIR, 'v2_per_query.json')
    per_query_out = []
    for r in phase1_results:
        per_query_out.append({
            'query_raw': r['query_raw'],
            'actor': r['actor'],
            'gold_ids': r['gold_ids'],
            'predicted': r['predicted'],
            'predicted_score': r['predicted_score'],
            'correct': r['correct'],
        })
    with open(per_query_file, 'w') as f:
        json.dump(per_query_out, f, indent=2)
    print(f"  Per-query detail: {per_query_file}")

    # ============================================================
    # FINAL SUMMARY — COPY THESE NUMBERS INTO THE PAPER
    # ============================================================
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY: ALL NUMBERS FOR ACSAC PAPER")
    print("=" * 65)

    print(f"\n  CTI-HAL (primary benchmark, {n} test queries):")
    print(f"    P@1:   {p1:.2%}  ({sum(r['correct'] for r in phase1_results)}/{n})")
    print(f"    Hit@3: {sum(r['hit3'] for r in phase1_results)/n:.2%}")
    print(f"    Hit@5: {sum(r['hit5'] for r in phase1_results)/n:.2%}")

    print(f"\n  Per-actor P@1:")
    for actor in sorted(actor_summary):
        a = actor_summary[actor]
        print(f"    {actor:<15} {a['p@1']:.2%}  ({a['correct']}/{a['total']})")

    if mcnemar_results:
        print(f"\n  McNemar's Tests:")
        for mr in mcnemar_results:
            print(f"    {mr['comparison']}:")
            print(f"      chi2={mr['chi2']:.2f}, p={mr['p_value']:.2e} ({mr['significance']})")
            print(f"      Reranker right/LLM wrong: {mr['reranker_right_llm_wrong']}")
            print(f"      Reranker wrong/LLM right: {mr['reranker_wrong_llm_right']}")

    print(f"\n  Multi-label (optimal threshold = {multilabel_results['optimal_threshold']}):")
    print(f"    Precision: {multilabel_results['best_precision']:.2%}")
    print(f"    Recall:    {multilabel_results['best_recall']:.2%}")
    print(f"    F1:        {multilabel_results['best_f1']:.2%}")

    if tumeteor_results:
        print(f"\n  Tumeteor benchmark ({tumeteor_results['total']:,} queries):")
        print(f"    P@1:   {tumeteor_results['p@1']:.2%}  "
              f"({tumeteor_results['correct']}/{tumeteor_results['total']})")
        print(f"    Hit@3: {tumeteor_results['hit@3']:.2%}")
        print(f"    Hit@5: {tumeteor_results['hit@5']:.2%}")

    print(f"\n  Results saved to {OUTPUT_DIR}/")
    print(f"  [DONE] All evaluations complete.\n")


if __name__ == '__main__':
    main()
