#!/usr/bin/env python3
"""
verify_v2_regeneration.py
=========================
Verifies that the regenerated v2 explainability outputs (eval_results_v2/) match
the documented v2_per_query.json predictions for the 146 CTI-HAL test queries.

This is the Step C verification gate. If predicted_ids match for all 146 queries,
Step C is complete and we proceed to Step C' by running recategorize_v1.py with
--token-csv pointing at eval_results_v2/token_importance.csv.

Usage (run from the reranker directory after running run_explain_v2.bat):
    python verify_v2_regeneration.py
    python verify_v2_regeneration.py --strict        (fail on ANY mismatch)
    python verify_v2_regeneration.py --output-dir eval_results_v2 \
                                      --reference v2_per_query.json

The verification produces:
    1. Match counts (N matched / N total) by query_raw
    2. Per-query predicted_id agreement
    3. Per-query correctness agreement
    4. Per-actor accuracy comparison (regen vs documented)
    5. Score-drift histogram (informational; cross-platform drift expected)
    6. CSV file integrity (token_importance.csv structural checks)
    7. Provenance summary (if provenance.json exists)

Exit codes:
    0  = PASS (all 146 predicted_ids match, all per-actor accuracies match)
    1  = PARTIAL (per-actor accuracy matches but some individual predictions differ)
    2  = FAIL (per-actor accuracy differs from documented v2 numbers)
    3  = ERROR (missing files, parse failure, structural integrity violation)
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


# Documented v2 per-actor accuracy (memory entries + verified from v2_per_query.json)
EXPECTED_PER_ACTOR = {
    'apt29':        (39, 41),   # 95.12%
    'carbanak':     (27, 27),   # 100.00%
    'fin6':         (19, 21),   # 90.48%
    'fin7':         (24, 24),   # 100.00%
    'oilrig':       (14, 17),   # 82.35%
    'sandworm':     (8, 9),     # 88.89%
    'wizardspider': (7, 7),     # 100.00%
}
EXPECTED_TOTAL_CORRECT = 138
EXPECTED_TOTAL = 146


def load_explanations(path):
    """Load explanations.json produced by explain.py."""
    with open(path, 'r', encoding='utf-8') as f:
        explanations = json.load(f)
    if not isinstance(explanations, list):
        raise ValueError(f"Expected list, got {type(explanations)}")
    return explanations


def load_reference(path):
    """Load v2_per_query.json reference."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_token_csv(path):
    """Load token_importance.csv as list of dict rows."""
    rows = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def join_by_query(explanations, reference):
    """
    Join explanations and reference by query_raw (exact string match).
    Returns (matched, regen_only, ref_only) lists.
    """
    regen_by_q = {e['query']: e for e in explanations}
    ref_by_q   = {r['query_raw']: r for r in reference}

    matched_q  = set(regen_by_q) & set(ref_by_q)
    regen_only = set(regen_by_q) - set(ref_by_q)
    ref_only   = set(ref_by_q)   - set(regen_by_q)

    pairs = []
    for q in matched_q:
        pairs.append((regen_by_q[q], ref_by_q[q]))

    return pairs, sorted(regen_only), sorted(ref_only)


def histogram_buckets(values, edges=(0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Build a simple histogram of absolute values."""
    buckets = []
    for lo, hi in zip(edges, edges[1:]):
        n = sum(1 for v in values if lo <= abs(v) < hi)
        buckets.append((lo, hi, n))
    n_above = sum(1 for v in values if abs(v) >= edges[-1])
    buckets.append((edges[-1], float('inf'), n_above))
    return buckets


def verify(args):
    output_dir = Path(args.output_dir)
    explanations_path = output_dir / 'explanations.json'
    csv_path          = output_dir / 'token_importance.csv'
    provenance_path   = output_dir / 'provenance.json'
    reference_path    = Path(args.reference)

    # --- File existence checks ---
    print('=' * 72)
    print('  STEP C VERIFICATION — v2 regeneration vs v2_per_query.json')
    print('=' * 72)
    print()
    print(f'  Output directory:  {output_dir}')
    print(f'  Reference file:    {reference_path}')
    print(f'  Strict mode:       {args.strict}')
    print()

    missing = []
    if not explanations_path.exists():
        missing.append(f'  - {explanations_path}')
    if not csv_path.exists():
        missing.append(f'  - {csv_path}')
    if not reference_path.exists():
        missing.append(f'  - {reference_path}')
    if missing:
        print('ERROR: required files not found:')
        for m in missing:
            print(m)
        return 3

    # --- Load all three ---
    print('[1/6] Loading inputs...')
    explanations = load_explanations(explanations_path)
    reference    = load_reference(reference_path)
    csv_rows     = load_token_csv(csv_path)
    print(f'      explanations.json: {len(explanations)} entries')
    print(f'      reference:         {len(reference)} entries')
    print(f'      token_importance.csv: {len(csv_rows)} rows')
    print()

    if len(reference) != EXPECTED_TOTAL:
        print(f'  ERROR: reference has {len(reference)} entries, expected {EXPECTED_TOTAL}')
        return 3

    # --- Structural integrity of CSV ---
    print('[2/6] CSV structural integrity check...')
    expected_csv_cols = {
        'query_idx', 'query_preview', 'actor', 'correct',
        'predicted_id', 'gold_id', 'confidence', 'margin',
        'token_position', 'token', 'importance_raw', 'importance_norm', 'impact',
    }
    if csv_rows:
        actual_cols = set(csv_rows[0].keys())
        if actual_cols != expected_csv_cols:
            print(f'      ERROR: CSV columns mismatch')
            print(f'        expected: {sorted(expected_csv_cols)}')
            print(f'        got:      {sorted(actual_cols)}')
            print(f'        missing:  {sorted(expected_csv_cols - actual_cols)}')
            print(f'        extra:    {sorted(actual_cols - expected_csv_cols)}')
            return 3

    # Check unique query_idx count
    unique_idxs = {row['query_idx'] for row in csv_rows}
    if len(unique_idxs) != len(explanations):
        print(f'      WARNING: CSV has {len(unique_idxs)} unique query_idx vs '
              f'{len(explanations)} explanations')

    # Check no NaN / blank importance values
    nan_count = sum(1 for row in csv_rows
                    if row['importance_raw'] in ('', 'nan', 'NaN', 'null'))
    if nan_count > 0:
        print(f'      WARNING: {nan_count} CSV rows have blank/NaN importance_raw')
    else:
        print(f'      CSV columns OK; all {len(csv_rows)} rows have numeric importance.')
    print()

    # --- Join and verify --- 
    print('[3/6] Joining explanations to reference by query_raw...')
    pairs, regen_only, ref_only = join_by_query(explanations, reference)
    print(f'      matched:    {len(pairs)}')
    print(f'      regen-only: {len(regen_only)}')
    print(f'      ref-only:   {len(ref_only)}')

    if regen_only:
        print(f'\n      Regen-only queries (in our output but not in reference):')
        for q in regen_only[:5]:
            print(f'        - {q[:70]!r}')
        if len(regen_only) > 5:
            print(f'        ... and {len(regen_only) - 5} more')

    if ref_only:
        print(f'\n      Reference-only queries (in v2_per_query.json but not regenerated):')
        for q in ref_only[:5]:
            print(f'        - {q[:70]!r}')
        if len(ref_only) > 5:
            print(f'        ... and {len(ref_only) - 5} more')
    print()

    if len(pairs) != EXPECTED_TOTAL:
        print(f'  ERROR: matched {len(pairs)} queries, expected {EXPECTED_TOTAL}.')
        print(f'         Cannot proceed with verification.')
        return 3

    # --- Predicted-ID agreement ---
    print('[4/6] Per-query predicted_id agreement check...')
    pred_match = 0
    pred_mismatch = []
    correct_match = 0
    score_drifts = []
    for regen, ref in pairs:
        regen_pred = regen['prediction']['attack_id']
        ref_pred   = ref['predicted']
        if regen_pred == ref_pred:
            pred_match += 1
        else:
            pred_mismatch.append({
                'query': regen['query'][:70],
                'actor': regen.get('actor') or ref.get('actor'),
                'regen_pred': regen_pred,
                'ref_pred':   ref_pred,
                'regen_correct': regen['prediction']['correct'],
                'ref_correct':   ref['correct'],
                'regen_score':   regen['prediction']['score'],
                'ref_score':     ref['predicted_score'],
                'regen_margin':  regen['margin'],
            })

        if bool(regen['prediction']['correct']) == bool(ref['correct']):
            correct_match += 1

        # Score drift (|regen_score - ref_score|)
        score_drifts.append(regen['prediction']['score'] - ref['predicted_score'])

    print(f'      predicted_id match:  {pred_match} / {len(pairs)}'
          f'  ({100.0 * pred_match / len(pairs):.2f}%)')
    print(f'      correctness match:   {correct_match} / {len(pairs)}'
          f'  ({100.0 * correct_match / len(pairs):.2f}%)')

    if pred_mismatch:
        print(f'\n      Predicted_id mismatches ({len(pred_mismatch)}):')
        print(f'      {"Query":<50}  {"Actor":<14}  {"Regen":<14}  {"Ref":<14}  '
              f'{"R_Cor":<6} {"M":<6}  Δscore   margin')
        for m in pred_mismatch:
            dscore = m['regen_score'] - m['ref_score']
            print(f'      {m["query"]:<50}  {m["actor"]:<14}  '
                  f'{m["regen_pred"]:<14}  {m["ref_pred"]:<14}  '
                  f'{str(m["regen_correct"]):<6} {str(m["ref_correct"]):<6}  '
                  f'{dscore:+7.3f}  {m["regen_margin"]:+.3f}')
    print()

    # --- Per-actor accuracy ---
    print('[5/6] Per-actor accuracy verification...')
    regen_actor_stats = defaultdict(lambda: [0, 0])  # [correct, total]
    for regen, ref in pairs:
        actor = regen.get('actor') or ref.get('actor')
        regen_actor_stats[actor][1] += 1
        if regen['prediction']['correct']:
            regen_actor_stats[actor][0] += 1

    print(f'      {"Actor":<14}  {"Regenerated":<22}  {"Documented":<22}  Status')
    print(f'      {"-"*14}  {"-"*22}  {"-"*22}  ------')
    actor_status_all_ok = True
    for actor in sorted(EXPECTED_PER_ACTOR.keys()):
        exp_c, exp_t = EXPECTED_PER_ACTOR[actor]
        got_c, got_t = regen_actor_stats.get(actor, [0, 0])
        match = (exp_c, exp_t) == (got_c, got_t)
        if not match:
            actor_status_all_ok = False
        status = 'MATCH' if match else '*** MISMATCH ***'
        regen_str = f'{got_c}/{got_t} ({100.0*got_c/got_t:.2f}%)' if got_t else 'N/A'
        exp_str   = f'{exp_c}/{exp_t} ({100.0*exp_c/exp_t:.2f}%)'
        print(f'      {actor:<14}  {regen_str:<22}  {exp_str:<22}  {status}')

    total_c = sum(r[0] for r in regen_actor_stats.values())
    total_t = sum(r[1] for r in regen_actor_stats.values())
    total_match = (total_c, total_t) == (EXPECTED_TOTAL_CORRECT, EXPECTED_TOTAL)
    total_status = 'MATCH' if total_match else '*** MISMATCH ***'
    regen_total = f'{total_c}/{total_t} ({100.0*total_c/total_t:.2f}%)' if total_t else 'N/A'
    exp_total   = f'{EXPECTED_TOTAL_CORRECT}/{EXPECTED_TOTAL} ({100.0*EXPECTED_TOTAL_CORRECT/EXPECTED_TOTAL:.2f}%)'
    print(f'      {"TOTAL":<14}  {regen_total:<22}  {exp_total:<22}  {total_status}')
    print()

    # --- Score-drift histogram ---
    print('[6/6] Score-drift histogram (regen - reference, raw logits)...')
    print(f'      Note: cross-platform drift between training-time GPU TF32 and')
    print(f'            inference-time CPU FP32 is expected and not a failure.')
    print(f'            What matters is the predicted_id match above.')
    print()
    if score_drifts:
        print(f'      |drift|        count')
        print(f'      ------------  -----')
        buckets = histogram_buckets([abs(d) for d in score_drifts])
        for lo, hi, n in buckets:
            if hi == float('inf'):
                label = f'  ≥ {lo:>7.0e}'
            else:
                label = f'  {lo:>7.0e} - {hi:.0e}'
            bar = '#' * min(n, 60)
            print(f'      {label:<14} {n:>4}  {bar}')
        max_drift = max(abs(d) for d in score_drifts)
        mean_drift = sum(score_drifts) / len(score_drifts)
        mean_abs = sum(abs(d) for d in score_drifts) / len(score_drifts)
        print()
        print(f'      max |drift|:  {max_drift:.6f}')
        print(f'      mean drift:   {mean_drift:+.6f}  (signed)')
        print(f'      mean |drift|: {mean_abs:.6f}')
    print()

    # --- Provenance summary ---
    if provenance_path.exists():
        print('Provenance summary (from provenance.json):')
        try:
            with open(provenance_path) as f:
                prov = json.load(f)
            keys_to_show = [
                'started_at_utc', 'finished_at_utc', 'wall_seconds',
                'torch_version', 'sentence_transformers_version',
                'platform', 'processor', 'seed',
                'torch_num_threads', 'torch_num_interop_threads',
                'torch_deterministic_algorithms',
                'data_sha256', 'hash_canary',
            ]
            for k in keys_to_show:
                v = prov.get(k, '<missing>')
                if isinstance(v, str) and len(v) > 70:
                    v = v[:67] + '...'
                print(f'  {k:<35} = {v}')
            env = prov.get('env_filtered', {})
            for ek in ('PYTHONHASHSEED', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                       'MKL_CBWR', 'KMP_DETERMINISTIC_REDUCTION'):
                print(f'  env[{ek!r}] = {env.get(ek, "<not set>")}')
        except Exception as e:
            print(f'  (failed to read provenance.json: {e})')
        print()

    # --- Verdict ---
    print('=' * 72)
    if pred_match == EXPECTED_TOTAL and actor_status_all_ok and total_match:
        print('  VERDICT: PASS')
        print(f'           All {EXPECTED_TOTAL}/{EXPECTED_TOTAL} predicted_ids match v2_per_query.json.')
        print(f'           Per-actor accuracy reproduces documented v2 numbers exactly.')
        print(f'           Step C is COMPLETE. Proceed to Step C\' by running:')
        print(f'             python recategorize_v1.py \\')
        print(f'                    --token-csv eval_results_v2/token_importance.csv \\')
        print(f'                    --output-dir eval_results_v2/categorization')
        print('=' * 72)
        return 0
    elif actor_status_all_ok and total_match:
        print('  VERDICT: PARTIAL')
        print(f'           Per-actor accuracy matches documented v2 numbers, but')
        print(f'           {len(pred_mismatch)} individual predicted_ids differ from v2_per_query.json.')
        print(f'           This can happen if cross-platform FP drift flipped near-tied')
        print(f'           predictions. Review the mismatches above; if all flips involve')
        print(f'           queries with margin < 0.01, this is acceptable for the paper.')
        print('=' * 72)
        return 1 if not args.strict else 2
    else:
        print('  VERDICT: FAIL')
        print(f'           Per-actor accuracy DOES NOT match documented v2 numbers.')
        print(f'           This indicates the regeneration is producing different')
        print(f'           predictions than the canonical v2_reeval.py run.')
        print(f'           Possible causes:')
        print(f'             - wrong checkpoint loaded (verify model_file_hashes in provenance)')
        print(f'             - wrong data file (verify data_sha256 in provenance)')
        print(f'             - test split mismatch (check seed=42 propagation)')
        print(f'             - silent activation_fn change (sigmoid vs identity)')
        print(f'           Halt and investigate before running Step C\'.')
        print('=' * 72)
        return 2


def main():
    parser = argparse.ArgumentParser(
        description='Verify v2 regeneration matches v2_per_query.json for ACSAC 2026 Step C')
    parser.add_argument('--output-dir', default='eval_results_v2',
                        help='Directory containing regenerated outputs (default: eval_results_v2)')
    parser.add_argument('--reference', default='v2_per_query.json',
                        help='Reference file with documented v2 predictions (default: v2_per_query.json)')
    parser.add_argument('--strict', action='store_true',
                        help='Fail (exit 2) on any individual prediction mismatch, even if per-actor accuracy matches')
    args = parser.parse_args()

    return verify(args)


if __name__ == '__main__':
    sys.exit(main())
