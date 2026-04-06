#!/usr/bin/env python3
"""
Error Analysis for Enriched Two-Stage Model (94.52% P@1)
=========================================================
Identifies the 8 remaining errors, categorizes them by type,
and provides actionable insights for targeted improvements.

Error categories:
  - TACTIC_VS_TECHNIQUE: Model predicts a tactic (TA####) when gold is
    a technique (T####), or vice versa
  - SIBLING_TECHNIQUE: Model predicts a closely related technique
    (same parent or same tactic)
  - WRONG_GRANULARITY: Model predicts parent technique when gold is
    sub-technique, or vice versa (e.g., T1059 vs T1059.001)
  - UNRELATED: Model predicts a completely different technique

Usage:
  python error_analysis_v2.py

  Expects:
    data/reranker_pairs_enriched_v2.jsonl
    checkpoints/best_two_stage_v2/

Author: Shane Waldrop — Angelo State University / ARL Grant W911NF-24-2-0180
"""

import json
import random
import re
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

RANDOM_SEED = 42
ATTACK_ID_RE = re.compile(r'(TA\d{4}|T\d{4}(?:\.\d{3})?|S\d{4})')


def load_and_group_data(filepath):
    """Load enriched JSONL and group by query_norm."""
    query_data = defaultdict(lambda: {
        'query_raw': None,
        'actor': None,
        'candidates': [],
        'positive_count': 0,
    })
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
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
            })
            if label == 1:
                query_data[qn]['positive_count'] += 1
    return dict(query_data)


def create_test_split(query_data):
    """Recreate exact test split (same as finetune_production.py)."""
    random.seed(RANDOM_SEED)
    queries_by_actor = defaultdict(list)
    for qn, data in query_data.items():
        actor = data['actor']
        if actor.startswith('external_'):
            continue
        if data['positive_count'] > 0:
            queries_by_actor[actor].append(qn)
    test_queries = []
    for actor, queries in queries_by_actor.items():
        random.shuffle(queries)
        n = len(queries)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        test_queries.extend(queries[n_train + n_val:])
    random.shuffle(test_queries)
    return test_queries


def extract_id(raw_id):
    """Extract clean ATT&CK ID from candidate_id field."""
    match = ATTACK_ID_RE.match(raw_id)
    return match.group(1) if match else raw_id


def classify_error(predicted_id, gold_ids):
    """Classify the type of error based on predicted vs gold IDs."""
    pred = predicted_id

    # Check if it's a tactic vs technique confusion
    pred_is_tactic = pred.startswith('TA')
    gold_has_tactic = any(g.startswith('TA') for g in gold_ids)
    gold_has_technique = any(g.startswith('T1') for g in gold_ids)
    gold_has_software = any(g.startswith('S') for g in gold_ids)

    if pred_is_tactic and not gold_has_tactic:
        return "TACTIC_VS_TECHNIQUE", "Predicted tactic, gold is technique/software"
    if not pred_is_tactic and gold_has_tactic and not gold_has_technique:
        return "TACTIC_VS_TECHNIQUE", "Predicted technique, gold is tactic"

    # Check for parent/child granularity confusion
    for g in gold_ids:
        if pred.startswith('T') and g.startswith('T'):
            # Check if one is parent of the other
            if '.' in pred and not '.' in g:
                if pred.split('.')[0] == g:
                    return "WRONG_GRANULARITY", f"Predicted sub-technique {pred}, gold is parent {g}"
            if '.' in g and not '.' in pred:
                if g.split('.')[0] == pred:
                    return "WRONG_GRANULARITY", f"Predicted parent {pred}, gold is sub-technique {g}"

    # Check for sibling techniques (same parent)
    for g in gold_ids:
        if '.' in pred and '.' in g:
            if pred.split('.')[0] == g.split('.')[0]:
                return "SIBLING_TECHNIQUE", f"Same parent: {pred} vs {g}"

    # Check for techniques in the same tactic (would need ATT&CK data)
    # For now, mark as potentially related if they share numeric prefix
    for g in gold_ids:
        if pred.startswith('T') and g.startswith('T'):
            # Check if technique numbers are close (heuristic)
            try:
                pred_num = int(pred.replace('T', '').split('.')[0])
                gold_num = int(g.replace('T', '').split('.')[0])
                if abs(pred_num - gold_num) <= 5:
                    return "NEARBY_TECHNIQUE", f"Close technique numbers: {pred} vs {g}"
            except ValueError:
                pass

    return "UNRELATED", f"Predicted {pred}, gold includes {', '.join(gold_ids)}"


def main():
    print("\n" + "=" * 70)
    print("  ERROR ANALYSIS — ENRICHED TWO-STAGE MODEL (94.52% P@1)")
    print("  Identifying and categorizing the 8 remaining misses")
    print("=" * 70)

    data_path = "data/reranker_pairs_enriched_v2.jsonl"
    model_path = "checkpoints/best_two_stage_v2"

    if not Path(data_path).exists():
        sys.exit(f"[FAIL] Data not found: {data_path}")
    if not Path(model_path).exists():
        sys.exit(f"[FAIL] Model not found: {model_path}")

    # Load data and create test split
    print(f"\n  Loading data from: {data_path}")
    query_data = load_and_group_data(data_path)
    test_queries = create_test_split(query_data)
    print(f"  Test queries: {len(test_queries)}")

    # Load model
    print(f"  Loading model from: {model_path}")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_path)
    print(f"  Model loaded.\n")

    # Evaluate and collect errors
    correct = 0
    total = 0
    errors = []
    all_results = []

    for qn in test_queries:
        qd = query_data[qn]
        if qd['positive_count'] == 0 or not qd['candidates']:
            continue

        query_raw = qd['query_raw']
        actor = qd['actor']
        texts = [[query_raw, c['text']] for c in qd['candidates']]
        scores = model.predict(texts, show_progress_bar=False)

        # Get gold labels and predictions
        gold_ids = set()
        candidate_info = []
        for j, c in enumerate(qd['candidates']):
            cid = extract_id(c['id'])
            candidate_info.append({
                'id': cid,
                'score': float(scores[j]),
                'label': c['label'],
                'text_preview': c['text'][:80],
            })
            if c['label'] == 1:
                gold_ids.add(cid)

        # Sort by score descending
        ranked = sorted(candidate_info, key=lambda x: x['score'], reverse=True)
        top_pred = ranked[0]
        total += 1

        if top_pred['label'] == 1:
            correct += 1
        else:
            # This is a miss — collect detailed info
            # Find the highest-scoring gold candidate
            gold_ranked = [c for c in ranked if c['label'] == 1]
            best_gold = gold_ranked[0] if gold_ranked else None
            gold_rank = next((i+1 for i, c in enumerate(ranked) if c['label'] == 1), -1)

            error_type, error_detail = classify_error(top_pred['id'], gold_ids)

            errors.append({
                'query': query_raw,
                'actor': actor,
                'predicted_id': top_pred['id'],
                'predicted_score': top_pred['score'],
                'predicted_text': top_pred['text_preview'],
                'gold_ids': sorted(gold_ids),
                'best_gold_id': best_gold['id'] if best_gold else '?',
                'best_gold_score': best_gold['score'] if best_gold else 0,
                'best_gold_text': best_gold['text_preview'] if best_gold else '?',
                'gold_rank': gold_rank,
                'score_gap': top_pred['score'] - (best_gold['score'] if best_gold else 0),
                'error_type': error_type,
                'error_detail': error_detail,
                'top_5': [(c['id'], round(c['score'], 4), c['label']) for c in ranked[:5]],
            })

    p1 = correct / total if total > 0 else 0
    print(f"  P@1: {p1:.4f} ({correct}/{total})")
    print(f"  Errors: {len(errors)}")

    # ── Detailed error report ──
    print(f"\n  {'='*70}")
    print(f"  DETAILED ERROR ANALYSIS")
    print(f"  {'='*70}")

    for i, err in enumerate(errors):
        print(f"\n  ── ERROR {i+1}/{len(errors)} ──")
        print(f"  Actor:          {err['actor']}")
        print(f"  Error type:     {err['error_type']}")
        print(f"  Detail:         {err['error_detail']}")
        print(f"  Query:          {err['query'][:100]}")
        if len(err['query']) > 100:
            print(f"                  ...{err['query'][100:200]}")
        print(f"  ")
        print(f"  PREDICTED (WRONG):")
        print(f"    ID:    {err['predicted_id']}")
        print(f"    Score: {err['predicted_score']:.4f}")
        print(f"    Text:  {err['predicted_text']}")
        print(f"  ")
        print(f"  BEST GOLD (CORRECT):")
        print(f"    ID:    {err['best_gold_id']}")
        print(f"    Score: {err['best_gold_score']:.4f}")
        print(f"    Text:  {err['best_gold_text']}")
        print(f"    Rank:  {err['gold_rank']} (of {len(query_data[test_queries[0]]['candidates'])} candidates)")
        print(f"  ")
        print(f"  Score gap:      {err['score_gap']:.4f} (predicted - gold)")
        print(f"  All gold IDs:   {', '.join(err['gold_ids'])}")
        print(f"  ")
        print(f"  Top 5 predictions:")
        for rank, (cid, score, label) in enumerate(err['top_5']):
            marker = " ✓ GOLD" if label == 1 else ""
            print(f"    {rank+1}. {cid:<15} score={score:>7.4f}{marker}")

    # ── Summary by error type ──
    print(f"\n  {'='*70}")
    print(f"  ERROR TYPE SUMMARY")
    print(f"  {'='*70}")

    type_counts = defaultdict(int)
    type_actors = defaultdict(list)
    type_gaps = defaultdict(list)
    for err in errors:
        type_counts[err['error_type']] += 1
        type_actors[err['error_type']].append(err['actor'])
        type_gaps[err['error_type']].append(err['score_gap'])

    print(f"\n  {'Error Type':<22} | {'Count':>5} | {'Avg Gap':>8} | Actors")
    print(f"  {'-'*22}-+-{'-'*5}-+-{'-'*8}-+-{'-'*30}")
    for etype in sorted(type_counts, key=type_counts.get, reverse=True):
        avg_gap = np.mean(type_gaps[etype])
        actors = ', '.join(sorted(set(type_actors[etype])))
        print(f"  {etype:<22} | {type_counts[etype]:>5} | {avg_gap:>+8.4f} | {actors}")

    # ── Per-actor error breakdown ──
    print(f"\n  PER-ACTOR ERROR COUNTS:")
    actor_errors = defaultdict(int)
    actor_totals = defaultdict(int)
    for qn in test_queries:
        qd = query_data[qn]
        if qd['positive_count'] > 0 and qd['candidates']:
            actor_totals[qd['actor']] += 1
    for err in errors:
        actor_errors[err['actor']] += 1

    for actor in sorted(actor_totals):
        errs = actor_errors.get(actor, 0)
        tot = actor_totals[actor]
        p1_actor = (tot - errs) / tot if tot > 0 else 0
        print(f"    {actor:<14}: {errs} errors / {tot} queries  (P@1 = {p1_actor:.2%})")

    # ── Actionable insights ──
    print(f"\n  {'='*70}")
    print(f"  ACTIONABLE INSIGHTS FOR IMPROVEMENT")
    print(f"  {'='*70}")

    tactic_errors = type_counts.get('TACTIC_VS_TECHNIQUE', 0)
    granularity_errors = type_counts.get('WRONG_GRANULARITY', 0)
    sibling_errors = type_counts.get('SIBLING_TECHNIQUE', 0)
    nearby_errors = type_counts.get('NEARBY_TECHNIQUE', 0)

    if tactic_errors + granularity_errors > 0:
        print(f"\n  HIERARCHICAL POST-PROCESSING would fix {tactic_errors + granularity_errors} "
              f"error(s) ({tactic_errors} tactic/technique + {granularity_errors} granularity)")
        print(f"  → Add post-processing that enforces ATT&CK parent-child relationships")

    if sibling_errors + nearby_errors > 0:
        print(f"\n  HARD NEGATIVE MINING would target {sibling_errors + nearby_errors} "
              f"error(s) ({sibling_errors} sibling + {nearby_errors} nearby)")
        print(f"  → Add confusing technique pairs as explicit hard negatives in training")

    close_calls = sum(1 for err in errors if abs(err['score_gap']) < 1.0)
    if close_calls > 0:
        print(f"\n  {close_calls} error(s) are CLOSE CALLS (score gap < 1.0)")
        print(f"  → These are borderline predictions that small improvements could fix")

    gold_in_top3 = sum(1 for err in errors if err['gold_rank'] <= 3)
    print(f"\n  Gold answer is in top 3 for {gold_in_top3}/{len(errors)} errors")
    if gold_in_top3 == len(errors):
        print(f"  → All errors are near-misses — the model has the right answer nearby")

    # Save results
    out_dir = Path("error_analysis_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "errors.json", 'w', encoding='utf-8') as f:
        json.dump(errors, f, indent=2, default=str)
    print(f"\n  Full error details saved to: error_analysis_v2/errors.json")
    print(f"\n  [DONE] Error analysis complete.")


if __name__ == '__main__':
    main()
