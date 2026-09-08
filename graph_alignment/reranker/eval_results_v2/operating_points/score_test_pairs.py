"""
ACSAC 2026 Step E — Stage 1: Score Test Pairs

Extract per-pair cross-encoder scores on the canonical 146-query CTI-HAL test
set. Produces `per_pair_scores.csv` which Stage 2 (`mcc_fpr_analysis.py`) reads
to compute MCC, F1, AUROC, AUPRC, operating-point analysis, and calibration
metrics with query-clustered bootstrap confidence intervals.

Why this script exists:

`v2_reeval.py` already scores every test-set pair during evaluation (see line
194: `scores = model.predict(pairs)`) and even constructs the per-candidate
score list (line 208: `all_scored = [(c['id'], float(s), c['label']) for c, s
in ranked]`), but its JSON-write step (lines 663-675) drops `all_scored`
before serialization, keeping only the top-1 prediction in `v2_per_query.json`.
Step E needs the full per-pair scores to compute pair-level binary metrics, so
this script regenerates them and persists everything.

Layer of analysis (per Shane's Option 1 decision):

This script scores the cross-encoder layer only. Hierarchical post-processing
(which adds the 0.69-point bump from 94.52% to 95.21% P@1 system-level) is a
downstream symbolic operation that does not affect per-pair scores, so it is
intentionally NOT applied here. Step E's MCC, F1, AUROC, AUPRC, and operating-
point metrics characterize the cross-encoder's intrinsic pair-relevance
discrimination ability; the 95.21% full-system result lives elsewhere in the
paper, in the main results section that describes hierarchical post-processing
as a contribution.

Output schema (12 columns):

    query_norm           Canonical (lower-cased, stripped) query identifier
    query_raw            Original query text as it appears in the corpus
    actor                APT actor for the query (one of 7)
    candidate_id         ATT&CK identifier ('T1234', 'TA0005', 'S0042', etc.)
    candidate_text       Enriched candidate text (truncated to 200 chars for CSV)
    score_raw_logit      Cross-encoder output (raw, NOT sigmoid-activated)
    score_sigmoid        Sigmoid-activated score in [0, 1]
    label_gold           1 if candidate is in this query's gold set, else 0
    rank_within_query    1-indexed rank (1 = highest score for this query)
    n_golds_for_query    Total gold candidates for this query
    n_candidates_for_query  Total candidate set size for this query
    candidate_text_hash  CRC32 of the full candidate text (for audit)

Encoding lesson from Step D:

Every file open() in this script uses explicit `encoding='utf-8'`, including
the CSV writer. Step D's `counterfactual_probe.py` lacked this on some writes
and produced cp1252-encoded output on Windows that broke downstream pandas
readers. The fix is uniform UTF-8 across all I/O.

Determinism contract:

Seeds: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`. The
`PYTHONHASHSEED=42` environment variable should be set by the caller (we
verify this is set and warn if not). The model itself is deterministic given
fixed weights — `model.predict()` calls `torch.no_grad()` internally and uses
deterministic CPU operations. The test split is reconstructed via
`v2_reeval.create_test_split` which seeds before shuffling, so the 146-query
identity is bit-identical to every prior v2 evaluation.

Expected runtime on CPU:

146 queries × ~20 candidates = ~2,920 pair scorings. With batch size 64 on a
typical modern CPU, expect 3-8 minutes total. No GPU required.

Usage:

    python eval_results_v2/operating_points/score_test_pairs.py \\
        --data-path data/reranker_pairs_enriched_v2.jsonl \\
        --model-path checkpoints/best_two_stage_v2 \\
        --output-dir eval_results_v2/operating_points
"""
import argparse
import csv
import json
import os
import random
import sys
import time
import zlib
from collections import Counter

import numpy as np


# =============================================================================
# Deterministic seeds — set BEFORE any model load
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# v2_reeval.py import helper
# =============================================================================
def _setup_v2_imports():
    """Locate v2_reeval.py and import its load_and_group_data + create_test_split.
    
    The script may be invoked from several different working directories
    (the user's reranker root, the operating_points subdirectory, etc.), so
    we search a few candidate locations. This mirrors the path-resolution
    strategy from Step D's `counterfactual_probe.py`.
    
    Returns:
        Tuple of (load_and_group_data fn, create_test_split fn, source_path).
        Exits the process with a clear error if v2_reeval.py cannot be found.
    """
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')),
        os.getcwd(),
        os.path.abspath(os.path.join(os.getcwd(), '..')),
        '/mnt/project',  # sandbox fallback
    ]
    for root in candidates:
        cand = os.path.join(root, 'v2_reeval.py')
        if os.path.isfile(cand):
            if root not in sys.path:
                sys.path.insert(0, root)
            import v2_reeval
            return (
                v2_reeval.load_and_group_data,
                v2_reeval.create_test_split,
                root,
            )
    print("ERROR: Could not locate v2_reeval.py in any candidate path:")
    for c in candidates:
        print(f"  - {c}")
    print("\nThe script needs v2_reeval.py for its canonical load_and_group_data")
    print("and create_test_split functions, which guarantee the 146-query test")
    print("split is bit-identical to every other v2 evaluation. Please run from")
    print("a directory where v2_reeval.py is in the parent tree.")
    sys.exit(1)


# =============================================================================
# Path resolution (file-or-directory, robust to working directory)
# =============================================================================
def _resolve_input_path(path, search_roots):
    """Resolve an input file or directory path with fallback search.
    
    The model checkpoint is a directory, the JSONL is a file. This resolver
    handles both via os.path.exists rather than os.path.isfile.
    """
    if os.path.exists(path):
        return os.path.abspath(path)
    basename = os.path.basename(path)
    for root in search_roots:
        if not root:
            continue
        cand = os.path.join(root, basename)
        if os.path.exists(cand):
            return os.path.abspath(cand)
        cand2 = os.path.join(root, path)
        if os.path.exists(cand2):
            return os.path.abspath(cand2)
    return path


# =============================================================================
# Sigmoid (numerically stable)
# =============================================================================
def _sigmoid(x):
    """Numerically stable sigmoid for both scalars and ndarrays."""
    x = np.asarray(x, dtype=np.float64)
    # For x >= 0: 1 / (1 + exp(-x))
    # For x <  0: exp(x) / (1 + exp(x))
    # This avoids overflow for very negative x.
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# =============================================================================
# Candidate type inference (T, TA, S, etc.) for audit
# =============================================================================
def _infer_candidate_type(candidate_id):
    """Map an ATT&CK identifier to its object type.
    
    'T1234' -> 'T' (technique)
    'T1234.001' -> 'T_sub' (sub-technique)
    'TA0005' -> 'TA' (tactic)
    'S0042' -> 'S' (software)
    'G0023' -> 'G' (group)
    'M1234' -> 'M' (mitigation)
    Everything else -> 'unknown'
    """
    if not candidate_id:
        return 'unknown'
    s = candidate_id.upper().strip()
    if s.startswith('TA'):
        return 'TA'
    if s.startswith('T') and '.' in s:
        return 'T_sub'
    if s.startswith('T'):
        return 'T'
    if s.startswith('S'):
        return 'S'
    if s.startswith('G'):
        return 'G'
    if s.startswith('M'):
        return 'M'
    return 'unknown'


# =============================================================================
# Main scoring loop
# =============================================================================
def score_test_set(model, query_data, test_queries, batch_size=64, verbose=True):
    """Score every (query, candidate) pair in the test set.
    
    Returns a list of dicts, one per pair, with all the audit columns we
    need for Stage 2 analysis.
    
    The function calls model.predict() once per query rather than once globally
    because we need to rank candidates within each query (to compute
    rank_within_query). Per-query batching is cheap for small candidate sets.
    """
    rows = []
    t0 = time.time()
    
    for i, qn in enumerate(test_queries):
        qdata = query_data[qn]
        query_raw = qdata['query_raw']
        candidates = qdata['candidates']
        gold_ids = set(qdata['gold_ids'])
        n_candidates = len(candidates)
        n_golds = sum(1 for c in candidates if c['label'] == 1)
        
        # Score all candidates for this query in one batched call
        pairs = [[query_raw, c['text']] for c in candidates]
        scores_raw = model.predict(pairs, batch_size=batch_size)
        scores_raw = np.asarray(scores_raw, dtype=np.float64)
        scores_sig = _sigmoid(scores_raw)
        
        # Compute ranks (1 = highest scoring). argsort returns ascending order;
        # we want descending, so flip. Then ranks[i] = position of candidate i
        # when sorted by descending score (1-indexed).
        order = np.argsort(-scores_raw, kind='stable')
        rank_within_query = np.empty(n_candidates, dtype=np.int32)
        rank_within_query[order] = np.arange(1, n_candidates + 1)
        
        for k, c in enumerate(candidates):
            # CRC32 hash of full candidate text as an audit field.
            # CRC32 is fast and deterministic across Python versions
            # (unlike Python's hash() which is salted unless PYTHONHASHSEED set).
            text_hash = format(
                zlib.crc32(c['text'].encode('utf-8')) & 0xFFFFFFFF, '08x'
            )
            rows.append({
                'query_norm': qn,
                'query_raw': query_raw,
                'actor': qdata['actor'],
                'candidate_id': c['id'],
                'candidate_type': _infer_candidate_type(c['id']),
                'candidate_text': c['text'][:200],  # truncated for CSV readability
                'candidate_text_hash': text_hash,
                'score_raw_logit': float(scores_raw[k]),
                'score_sigmoid': float(scores_sig[k]),
                'label_gold': int(c['label']),
                'rank_within_query': int(rank_within_query[k]),
                'n_golds_for_query': n_golds,
                'n_candidates_for_query': n_candidates,
            })
        
        if verbose and (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(test_queries) - i - 1) / rate
            print(f"    [{i+1}/{len(test_queries)}] elapsed: {elapsed:.0f}s  "
                  f"rate: {rate:.1f}/s  ETA: {eta:.0f}s")
    
    elapsed = time.time() - t0
    if verbose:
        print(f"  Total scoring time: {elapsed:.0f}s "
              f"({elapsed/len(test_queries):.2f}s per query, "
              f"{len(rows)} total pairs scored)")
    
    return rows


# =============================================================================
# CSV writer (UTF-8 explicit)
# =============================================================================
CSV_FIELDNAMES = [
    'query_norm', 'query_raw', 'actor',
    'candidate_id', 'candidate_type', 'candidate_text', 'candidate_text_hash',
    'score_raw_logit', 'score_sigmoid',
    'label_gold', 'rank_within_query',
    'n_golds_for_query', 'n_candidates_for_query',
]


def write_per_pair_csv(rows, output_path):
    """Write per-pair scores to CSV with explicit UTF-8 encoding.
    
    Step D lesson learned: never use the default `open()` for text I/O on
    Windows — it picks up cp1252 from the system locale and silently produces
    encoding-mismatched output. Always pass `encoding='utf-8'` explicitly.
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  Wrote {len(rows)} pair-rows to {output_path}")


# =============================================================================
# Sanity-check the test split before scoring
# =============================================================================
def report_test_split_summary(query_data, test_queries):
    """Print a one-screen summary of the test split for visual sanity check.
    
    This prints the actor distribution, candidate-set size statistics, and
    gold-density statistics so the user can confirm at a glance that the
    test split matches every prior v2 evaluation (146 queries, 7 actors,
    median 20 candidates, mean 2.87 golds per query).
    """
    print(f"  Test queries: {len(test_queries)}")
    
    # Actor distribution
    actor_counts = Counter()
    for qn in test_queries:
        actor_counts[query_data[qn]['actor']] += 1
    print(f"  Actor distribution:")
    for actor, n in sorted(actor_counts.items(), key=lambda x: -x[1]):
        print(f"    {actor}: {n}")
    
    # Candidate-set size and gold-density statistics
    n_cands = [len(query_data[qn]['candidates']) for qn in test_queries]
    n_golds = [query_data[qn]['positive_count'] for qn in test_queries]
    print(f"  Candidates per query: "
          f"min={min(n_cands)}, max={max(n_cands)}, "
          f"median={int(np.median(n_cands))}, mean={np.mean(n_cands):.1f}")
    print(f"  Golds per query:      "
          f"min={min(n_golds)}, max={max(n_golds)}, "
          f"median={int(np.median(n_golds))}, mean={np.mean(n_golds):.2f}")
    
    # Total pair count
    total_pairs = sum(n_cands)
    total_positives = sum(n_golds)
    print(f"  Total pairs to score: {total_pairs}")
    print(f"  Total positive pairs: {total_positives}")
    print(f"  Total negative pairs: {total_pairs - total_positives}")
    print(f"  Pair-level imbalance: "
          f"{(total_pairs - total_positives)/total_positives:.2f}:1 (neg:pos)")


# =============================================================================
# Provenance recorder
# =============================================================================
def write_provenance(args, search_roots, v2_source, n_test, n_pairs, output_dir):
    """Record run metadata for reproducibility."""
    provenance = {
        'script': 'score_test_pairs.py',
        'step': 'E (operating points and binary classification)',
        'stage': '1 of 2 (scoring extraction)',
        'random_seed': SEED,
        'pythonhashseed_env': os.environ.get('PYTHONHASHSEED', '(not set)'),
        'args': {
            'data_path': args.data_path,
            'model_path': args.model_path,
            'output_dir': args.output_dir,
            'batch_size': args.batch_size,
        },
        'resolved_paths': {
            'data_path': _resolve_input_path(args.data_path, search_roots),
            'model_path': _resolve_input_path(args.model_path, search_roots),
            'v2_reeval_source': v2_source,
        },
        'test_set_stats': {
            'n_test_queries': n_test,
            'n_pairs_scored': n_pairs,
        },
        'note_on_p_at_1': (
            'This script scores the cross-encoder layer only. P@1 derived '
            'from these scores is the standalone cross-encoder number (94.52% '
            'expected). Hierarchical post-processing (which adds 0.69 P@1 '
            'points to reach the system-level 95.21%) is intentionally NOT '
            'applied here; it does not affect per-pair scores and is reported '
            'in the main results section of the paper, not in Step E.'
        ),
    }
    pp_path = os.path.join(output_dir, 'provenance_step_e_stage1.json')
    with open(pp_path, 'w', encoding='utf-8') as f:
        json.dump(provenance, f, indent=2)
    print(f"  Provenance: {pp_path}")


# =============================================================================
# Self-test (no model required)
# =============================================================================
def _self_test():
    """Self-test the encoding-safe utilities (sigmoid, type inference, hash)."""
    print("Running self-tests...")
    
    # Sigmoid at well-known values
    assert abs(_sigmoid(np.array([0.0]))[0] - 0.5) < 1e-12, "sigmoid(0) != 0.5"
    assert abs(_sigmoid(np.array([100.0]))[0] - 1.0) < 1e-10, "sigmoid(100) != 1.0"
    assert abs(_sigmoid(np.array([-100.0]))[0]) < 1e-10, "sigmoid(-100) != 0.0"
    # Symmetry: sigmoid(-x) = 1 - sigmoid(x)
    s_pos = _sigmoid(np.array([2.5]))[0]
    s_neg = _sigmoid(np.array([-2.5]))[0]
    assert abs(s_pos + s_neg - 1.0) < 1e-12, "sigmoid symmetry failed"
    # Numerical stability at extreme values
    assert not np.isnan(_sigmoid(np.array([1e9]))[0]), "sigmoid overflowed at 1e9"
    assert not np.isnan(_sigmoid(np.array([-1e9]))[0]), "sigmoid overflowed at -1e9"
    print("  sigmoid: OK")
    
    # Type inference
    cases = {
        'T1059': 'T',
        'T1059.001': 'T_sub',
        'TA0005': 'TA',
        'S0042': 'S',
        'G0007': 'G',
        'M1234': 'M',
        '': 'unknown',
        'X9999': 'unknown',
    }
    for cid, expected in cases.items():
        got = _infer_candidate_type(cid)
        assert got == expected, f"type({cid!r}): expected {expected}, got {got}"
    print("  candidate type inference: OK")
    
    # CRC32 deterministic
    h1 = format(zlib.crc32('hello'.encode('utf-8')) & 0xFFFFFFFF, '08x')
    h2 = format(zlib.crc32('hello'.encode('utf-8')) & 0xFFFFFFFF, '08x')
    assert h1 == h2, "CRC32 not deterministic"
    assert h1 == '3610a686', f"CRC32 of 'hello' expected 3610a686, got {h1}"
    print("  CRC32 hash: OK")
    
    print("Self-tests passed.\n")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='ACSAC Step E Stage 1: Extract per-pair scores from v2 cross-encoder.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data-path', default='data/reranker_pairs_enriched_v2.jsonl',
                        help='Path to enriched JSONL (default: %(default)s)')
    parser.add_argument('--model-path', default='checkpoints/best_two_stage_v2',
                        help='Path to v2 cross-encoder checkpoint (default: %(default)s)')
    parser.add_argument('--output-dir', default='eval_results_v2/operating_points',
                        help='Output directory (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Cross-encoder batch size (default: %(default)s)')
    parser.add_argument('--self-test', action='store_true',
                        help='Run self-tests on utility functions and exit.')
    args = parser.parse_args()
    
    if args.self_test:
        _self_test()
        return 0
    
    print('='*75)
    print('ACSAC 2026 Step E — Stage 1: Score Test Pairs')
    print('='*75)
    print(f"  data:       {args.data_path}")
    print(f"  model:      {args.model_path}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  seed:       {SEED}")
    print(f"  PYTHONHASHSEED env: {os.environ.get('PYTHONHASHSEED', '(not set)')}")
    if os.environ.get('PYTHONHASHSEED', '') != '42':
        print("  WARNING: PYTHONHASHSEED is not 42; cross-platform determinism "
              "may not hold exactly, though the model itself is deterministic.")
    print()
    
    # ==================================================
    # [1/5] Locate v2_reeval and resolve paths
    # ==================================================
    print("[1/5] Locating v2_reeval.py and resolving input paths...")
    load_and_group_data, create_test_split, v2_source = _setup_v2_imports()
    print(f"  v2_reeval source: {v2_source}")
    
    search_roots = [
        v2_source,
        os.path.dirname(os.path.abspath(args.data_path)),
        os.getcwd(),
        os.path.abspath(os.path.join(os.getcwd(), '..')),
    ]
    args.data_path = _resolve_input_path(args.data_path, search_roots)
    args.model_path = _resolve_input_path(args.model_path, search_roots)
    print(f"  data:  {args.data_path}")
    print(f"  model: {args.model_path}")
    if not os.path.exists(args.data_path):
        print(f"ERROR: data file not found: {args.data_path}")
        return 1
    if not os.path.exists(args.model_path):
        print(f"ERROR: model directory not found: {args.model_path}")
        return 1
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ==================================================
    # [2/5] Load data and construct test split
    # ==================================================
    print()
    print("[2/5] Loading enriched JSONL and constructing test split...")
    query_data, total_rows = load_and_group_data(args.data_path)
    print(f"  Total rows in JSONL: {total_rows}")
    print(f"  Unique queries: {len(query_data)}")
    test_queries = create_test_split(query_data)
    print()
    report_test_split_summary(query_data, test_queries)
    
    # ==================================================
    # [3/5] Load v2 cross-encoder
    # ==================================================
    print()
    print("[3/5] Loading v2 cross-encoder model...")
    # Set torch seed before model load for any random initialization
    try:
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
    except ImportError:
        print("  WARNING: torch not importable; that's odd for a "
              "sentence-transformers model. Proceeding anyway.")
    
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(args.model_path)
    print(f"  Model loaded from {args.model_path}")
    
    # ==================================================
    # [4/5] Score every test pair
    # ==================================================
    print()
    print("[4/5] Scoring every (query, candidate) pair...")
    rows = score_test_set(model, query_data, test_queries,
                           batch_size=args.batch_size)
    
    # ==================================================
    # [5/5] Write outputs (CSV + provenance)
    # ==================================================
    print()
    print("[5/5] Writing per-pair CSV and provenance...")
    out_csv = os.path.join(args.output_dir, 'per_pair_scores.csv')
    write_per_pair_csv(rows, out_csv)
    write_provenance(args, search_roots, v2_source,
                     n_test=len(test_queries), n_pairs=len(rows),
                     output_dir=args.output_dir)
    
    # Quick sanity check on results: report observed P@1
    p_at_1 = sum(
        1 for qn in test_queries
        if any(r['rank_within_query'] == 1 and r['label_gold'] == 1
               for r in rows if r['query_norm'] == qn)
    ) / len(test_queries)
    print()
    print(f"  Sanity check — observed P@1 at top-1 prediction: {p_at_1:.4f}")
    print(f"  Expected (cross-encoder only, no hierarchical post-processing): 0.9452")
    if abs(p_at_1 - 0.9452) > 0.01:
        print(f"  WARNING: observed P@1 differs from expected by more than 1%.")
        print(f"  This may indicate a model or data mismatch. Investigate before "
              f"running Stage 2.")
    else:
        print(f"  Matches expected cross-encoder P@1. Proceed to Stage 2.")
    
    print()
    print(f"Step E Stage 1 COMPLETE. Outputs in: {args.output_dir}")
    print(f"Next: run mcc_fpr_analysis.py on {out_csv}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
