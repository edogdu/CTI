"""
counterfactual_probe.py — Main pipeline for ACSAC 2026 Step D.

This script:

    1. Loads the v2 reranker test split using v2_reeval's canonical
       load_and_group_data() and create_test_split() functions, so the
       146-query test set is bit-identical to Step C / Step C'.
    2. Stratifies the 146 queries into Stratum A (n=21, actor-bearing)
       and Stratum B (n=125, actor-free) using the post-filter
       safe-alias-token mapping derived from vocabulary_v14.json and
       the v14 STIX bundle.
    3. Builds the four substitute pools and the actor-technique-prior
       map via cf_substitution.
    4. Loads the v2 cross-encoder model from
       checkpoints/best_two_stage_v2 and the v2 tokenizer for
       subword-token-delta computation.
    5. Runs the five interventions on Stratum A and the two insertion
       sub-conditions on Stratum B, scoring every (query, intervention,
       substitute) triple against the candidate set.
    6. Computes the per-triple metrics: top-1 flip, gold-rank change,
       gold-score delta, gold-margin delta (= margin_perturbed -
       margin_original), top-1 score delta, top-5 Jaccard, top-5 RBO at
       p=0.8, top-5 RBO sensitivity at p=0.75/0.85/0.9, top-5 Kendall
       tau-b, correctness flags, subword token delta, surface-form
       relaxation level, technique-prior Jaccard.
    7. Writes outputs:
         - substitute_pool.json
         - actor_technique_priors.json
         - counterfactual_manifest.jsonl  (one line per triple)
         - counterfactual_results.csv     (flat Minimal Pairs format)
         - counterfactual_aggregates.json (by-actor and by-stratum)
         - counterfactual_summary.md      (paper-ready findings)
         - provenance_step_d.json         (run metadata, hashes, seeds)

The script is designed to be run on Shane's Windows machine where the
v2 model checkpoint and v2 enriched data are available locally. Paths
are configurable via command-line --output-dir; the default Python
working-directory expectation matches the v2_reeval.py convention
(reranker/ as cwd).

Usage on Shane's Windows machine:
    cd C:\\Users\\shane\\Downloads\\CTI\\graph_alignment\\reranker\\
       graph_alignment\\reranker
    set PYTHONHASHSEED=42
    set CUBLAS_WORKSPACE_CONFIG=:4096:8
    python counterfactual_probe.py --output-dir eval_results_v2/counterfactual

Reference design across all eleven research documents:
    - PRIMARY inferential test: paired Δ-margin difference-in-differences
      with paired permutation + BCa-bootstrap CI clustered by query
      (per ChatGPT this turn + Claude operational answers + Yang et al.
      2026 §3.3).
    - CO-PRIMARY operational summary: top-1 flip rate with Wilson CI;
      mid-p McNemar where b+c >= 5.
    - SECONDARY rank stability: RBO@5 with persistence p=0.8 (per the
      principled 1/(1-p)=5 derivation from Webber-Moffat-Zobel 2010);
      sensitivity at p=0.75/0.85/0.9; Jaccard@5 descriptive.
    - APPENDIX robustness: linear mixed-effects model on Δ-margin via
      statsmodels.MixedLM (per Gemini PDF + Perplexity Inferential Layer).

Author: Shane Waldrop, ACSAC 2026 paper, Step D (May 2026).
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict, Counter
from datetime import datetime, timezone

import numpy as np

# Project-wide seeds (also used by cf_substitution)
PYTHONHASHSEED = 42
MODEL_SEED = 42
SUBSTITUTE_SAMPLING_SEED = 42
PERMUTATION_SEED = 42
BOOTSTRAP_SEED = 42
INSERTION_TEMPLATE_SEED = 42

os.environ.setdefault('PYTHONHASHSEED', str(PYTHONHASHSEED))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def _resolve_input_path(path, search_roots):
    """Resolve an input file or directory path with a fallback search.
    
    The input data files for this project (enterprise-attack-v14.json,
    reranker_pairs_enriched_v2.jsonl, vocabulary_v14.json) live in
    slightly different locations across deployments — sometimes in a
    `data/` subfolder, sometimes directly in the reranker directory,
    sometimes in eval_results_v2/categorization/. The model checkpoint
    is a *directory* (best_two_stage_v2/), not a file. Rather than
    failing immediately if the user-supplied path doesn't exist, we
    accept either files or directories and try the given path first,
    then look in a short list of plausible search roots.
    
    Args:
        path: the path the user supplied (relative or absolute, file or dir)
        search_roots: list of directories to search if `path` doesn't
            resolve directly. Order matters; first hit wins.
    
    Returns:
        An absolute path that exists on disk, or the original `path`
        unchanged if no fallback was found (the caller will raise the
        appropriate error downstream).
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
        # Also try with the full relative path appended (e.g. data/foo.json)
        cand2 = os.path.join(root, path)
        if os.path.exists(cand2):
            return os.path.abspath(cand2)
    return path  # let the downstream open() / from_pretrained() raise


# ============================================================================
# v2_reeval imports — these provide the canonical test split and data loader
# ============================================================================
# The path setup mirrors recategorize_v2.py (Step C'): we add the project
# directory to sys.path and import from v2_reeval.
def _setup_v2_imports():
    """Make v2_reeval's load_and_group_data and create_test_split importable.
    
    Searches several plausible locations for v2_reeval.py so the script
    works across different repo layouts:
      - Shane's local Windows layout: script at <repo>/.../reranker/
        eval_results_v2/counterfactual/  -> v2_reeval.py is 2 levels up.
      - Doubly-nested legacy layout (graph_alignment/reranker/
        graph_alignment/reranker/): -> v2_reeval.py is 3 levels up.
      - Claude sandbox: /mnt/project/v2_reeval.py.
      - User's current working directory (in case they cd'd into
        the reranker folder before invoking the script).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    # Walk up 0, 1, 2, 3, 4 parent directories from the script's location
    cur = here
    for _ in range(5):
        candidates.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break  # filesystem root reached; stop walking up
        cur = parent
    # Also try the user's current working directory and the sandbox path
    candidates.append(os.getcwd())
    candidates.append('/mnt/project')
    
    for c in candidates:
        if os.path.isfile(os.path.join(c, 'v2_reeval.py')):
            sys.path.insert(0, c)
            return c
    raise RuntimeError(
        "Cannot find v2_reeval.py. Searched these locations: " +
        str(candidates) + ". Either run the script from the reranker "
        "directory containing v2_reeval.py, or move the script tree so "
        "v2_reeval.py is reachable by walking up at most 4 parent "
        "directories from this script.")


# ============================================================================
# Substitute infrastructure imports
# ============================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf_substitution import (
    classify_surface_form, recast_to_match,
    build_pool_actor_primary, build_pool_proper_noun, build_pool_malware,
    build_actor_technique_priors, jaccard_similarity, find_actor_spans,
    select_substitutes, select_generic_phrase, select_soft_deletion,
    apply_substitute, apply_insertion,
    GENERIC_PHRASE_POOL, TEST_ACTOR_CANONICAL,
    SUBSTITUTE_SAMPLING_SEED,
)


# ============================================================================
# Stratification: post-filter safe-alias-token mapping
# ============================================================================
def build_safe_alias_tokens(stix_bundle_path, vocab_path, test_actors):
    """
    For each test actor, build the set of alias tokens that:
      1. Appear in the actor's STIX intrusion-set primary or aliases,
         after normalisation (lower-case, comma/period/paren stripped).
      2. Are at least 3 characters (filter out '6', '23', etc.).
      3. Do NOT appear in any other test actor's alias set (no shared
         tokens like 'iron', 'bear', 'spider', 'gold').
      4. Are present in vocabulary_v14.json with a SAFE category
         (actor_primary, actor_alias, ambiguous_actor_software,
         ambiguous_actor_software_alias, or ambiguous_actor_campaign).
    
    This mirrors the empirical sanity-check Stratum-A construction
    verified during the planning phase: 21 actor-bearing queries.
    """
    SAFE_CATS = {'actor_primary', 'actor_alias',
                 'ambiguous_actor_software', 'ambiguous_actor_software_alias',
                 'ambiguous_actor_campaign'}
    
    with open(stix_bundle_path) as f:
        bundle = json.load(f)
    with open(vocab_path) as f:
        vocab = json.load(f)
    
    raw_alias_tokens = {}
    for ta, info in test_actors.items():
        g_id = info['stix_g_id']
        for obj in bundle['objects']:
            if obj.get('type') != 'intrusion-set':
                continue
            if obj.get('revoked') or obj.get('x_mitre_deprecated'):
                continue
            if not any(r.get('external_id') == g_id
                       for r in obj.get('external_references', [])):
                continue
            tokens = set()
            for a in [obj.get('name', '')] + obj.get('aliases', []):
                norm = a.lower().replace(',', '').replace('.', '').replace('(', '').replace(')', '')
                for t in norm.split():
                    if len(t) >= 3:
                        tokens.add(t)
            raw_alias_tokens[ta] = tokens
            break
    
    # Find tokens shared across test actors
    all_tokens_flat = []
    for tok_set in raw_alias_tokens.values():
        all_tokens_flat.extend(tok_set)
    shared = {t for t, c in Counter(all_tokens_flat).items() if c > 1}
    
    # Filter: actor-specific, vocab-safe
    safe = {}
    for ta, tokens in raw_alias_tokens.items():
        filtered = set()
        for t in tokens:
            if t in shared:
                continue
            if t not in vocab['tokens']:
                continue
            if vocab['tokens'][t]['category'] not in SAFE_CATS:
                continue
            filtered.add(t)
        safe[ta] = filtered
    
    return safe, shared


def stratify_test_queries(query_data, test_queries, safe_alias_tokens):
    """
    Partition the 146 test queries into Stratum A (actor-bearing) and
    Stratum B (actor-free) based on whether each query contains any of
    its actor's safe-alias tokens.
    
    Returns:
        stratum_a:  list of dicts {qn, actor, query_raw, gold_ids,
                                    safe_alias_match, candidates}
        stratum_b:  list of dicts (same structure, no safe_alias_match)
    """
    a_records = []
    b_records = []
    for qn in test_queries:
        d = query_data[qn]
        actor = d['actor']
        qr = d['query_raw']
        qr_tokens = set(qr.lower()
                        .replace(',', ' ').replace('.', ' ')
                        .replace('(', ' ').replace(')', ' ').split())
        match = qr_tokens & safe_alias_tokens.get(actor, set())
        rec = {
            'qn': qn,
            'actor': actor,
            'query_raw': qr,
            'gold_ids': sorted(d['gold_ids']),
            'candidates': d['candidates'],
        }
        if match:
            rec['safe_alias_match'] = sorted(match)
            a_records.append(rec)
        else:
            b_records.append(rec)
    return a_records, b_records


# ============================================================================
# Subword-token-delta via the v2 tokenizer
# ============================================================================
def compute_subword_token_delta(tokenizer, original_text, perturbed_text):
    """
    Compute the difference in WordPiece subword tokens between
    perturbed and original text. Returns int (perturbed_count -
    original_count). For a clean swap this is usually 0 to +/-2.
    
    The cross-encoder's positional encoding shifts when this is
    nonzero, which is the central confound the subword-length match
    addresses. We record this as a covariate per all eleven documents'
    consensus.
    """
    if tokenizer is None:
        return -999  # sentinel: tokenizer not available
    orig_ids = tokenizer.encode(original_text, add_special_tokens=False)
    pert_ids = tokenizer.encode(perturbed_text, add_special_tokens=False)
    return len(pert_ids) - len(orig_ids)


# ============================================================================
# RBO (Rank-Biased Overlap) computation
# ============================================================================
def rbo_at_k(list1, list2, k, p):
    """
    Compute Rank-Biased Overlap up to depth k with persistence p.
    
    RBO formula (Webber, Moffat, Zobel 2010): for two indefinite-length
    rankings, RBO weighs agreement at depth d by p^(d-1) and sums these
    weighted agreements, then normalises by the sum of the weights up
    to depth k.
    
    This is the "extrapolation-aware" RBO: we compute up to depth k and
    treat unseen tail as agreement-neutral, which is the standard
    truncated form used when k is fixed.
    
    Per the user's locked-in stack: primary value is p=0.8 for k=5
    (1/(1-0.8) = 5 expected user inspection depth). Sensitivity values
    p=0.75, 0.85, 0.9 reported in appendix.
    
    Returns: float in [0.0, 1.0]
    
    Verified against the WMZ paper definition and implementations in
    the rbstar / pyterrier libraries for the truncated case.
    """
    if not list1 or not list2:
        return 0.0
    k = min(k, len(list1), len(list2))
    if k <= 0:
        return 0.0
    
    # At each depth d, the agreement A_d is |intersection of prefixes| / d.
    # The RBO sum is (1-p) * sum_{d=1..k} p^(d-1) * A_d, but for the
    # truncated form with finite k we normalise by (1-p^k) so the result
    # lies in [0, 1]: see Urbano (2020) and PyTerrier docs.
    set1 = []
    set2 = []
    overlap_set = set()
    weighted_sum = 0.0
    for d in range(1, k + 1):
        if list1[d-1] not in set2 and list1[d-1] not in overlap_set:
            set1.append(list1[d-1])
        else:
            overlap_set.add(list1[d-1])
        if list2[d-1] not in set1 and list2[d-1] not in overlap_set:
            set2.append(list2[d-1])
        else:
            overlap_set.add(list2[d-1])
        # Recompute overlap at depth d
        prefix1 = set(list1[:d])
        prefix2 = set(list2[:d])
        agreement = len(prefix1 & prefix2) / d
        weighted_sum += (p ** (d - 1)) * agreement
    
    # Normalise: total weight in p-geometric series up to depth k
    norm = (1 - p**k) / (1 - p) if p != 1.0 else float(k)
    return weighted_sum / norm


def kendall_tau_b_top_k(list1, list2, k, universe=None):
    """
    Compute Kendall tau-b on the top-k of two rankings.
    
    Build a common universe = (top-k of list1) U (top-k of list2),
    then assign each item its rank in list1 (or +inf if not in top-k)
    and similarly for list2. Compute tau-b on these paired rank
    vectors.
    
    Returns: float in [-1.0, 1.0], or NaN if universe is empty.
    """
    if not list1 or not list2:
        return float('nan')
    top1 = list1[:k]
    top2 = list2[:k]
    universe_items = list(set(top1) | set(top2))
    if len(universe_items) < 2:
        return float('nan')
    # Assign ranks (1-indexed); items outside top-k get rank k+1 (tie)
    def rank_in(lst, item, max_rank):
        try:
            return lst.index(item) + 1
        except ValueError:
            return max_rank + 1
    r1 = [rank_in(top1, x, k) for x in universe_items]
    r2 = [rank_in(top2, x, k) for x in universe_items]
    from scipy.stats import kendalltau
    res = kendalltau(r1, r2, variant='b')
    if hasattr(res, 'statistic'):
        return float(res.statistic) if not np.isnan(res.statistic) else float('nan')
    # older scipy
    return float(res[0]) if not np.isnan(res[0]) else float('nan')


# ============================================================================
# Per-query metric computation
# ============================================================================
def score_query(model, query_text, candidates):
    """
    Score one query against its candidate set using the v2 model.
    
    Returns: dict with
        ranked:        list of (cid, score, label) tuples sorted by score desc
        top1:          (cid, score, label) of rank 1
        all_scores:    np.array of scores in candidate order
        cand_ids:      list of cids in input order
    """
    pairs = [[query_text, c['text']] for c in candidates]
    scores = model.predict(pairs)
    cand_with_scores = [
        (c['id'], float(s), c['label'])
        for c, s in zip(candidates, scores)
    ]
    ranked = sorted(cand_with_scores, key=lambda x: x[1], reverse=True)
    return {
        'ranked': ranked,
        'top1': ranked[0],
        'all_scores': np.array([s for _, s, _ in cand_with_scores]),
        'cand_ids': [c['id'] for c in candidates],
    }


def compute_metrics(orig_score_result, pert_score_result, gold_ids):
    """
    Given the original and perturbed scoring results plus the gold-id
    set, compute all per-triple metrics.
    
    Returns a dict with:
        top1_orig_id, top1_pert_id, top1_flip
        gold_rank_orig, gold_rank_pert, gold_rank_change
        gold_score_orig, gold_score_pert, gold_score_delta
        top1_score_delta, gold_margin_orig, gold_margin_pert,
        gold_margin_delta
        top5_jaccard, top5_kendall_tau_b
        top5_rbo_p075, top5_rbo_p08, top5_rbo_p085, top5_rbo_p09
        correct_orig, correct_pert
    """
    gold_set = set(gold_ids)
    
    # Top-1 flip
    top1_orig = orig_score_result['top1'][0]
    top1_pert = pert_score_result['top1'][0]
    top1_flip = (top1_orig != top1_pert)
    
    # Correctness
    correct_orig = top1_orig in gold_set
    correct_pert = top1_pert in gold_set
    
    # Gold-rank: lowest rank position (1-indexed) where any gold id appears
    def gold_rank(ranked):
        for i, (cid, _, _) in enumerate(ranked):
            if cid in gold_set:
                return i + 1
        return len(ranked) + 1  # not found
    gold_rank_orig = gold_rank(orig_score_result['ranked'])
    gold_rank_pert = gold_rank(pert_score_result['ranked'])
    
    # Gold score: highest score among gold-id candidates
    def gold_score(ranked):
        best = float('-inf')
        for cid, score, _ in ranked:
            if cid in gold_set:
                best = max(best, score)
        return best if best != float('-inf') else float('nan')
    gold_score_orig = gold_score(orig_score_result['ranked'])
    gold_score_pert = gold_score(pert_score_result['ranked'])
    gold_score_delta = gold_score_pert - gold_score_orig
    
    # Gold margin: score(gold) - max(score(non-gold))
    # This is the ChatGPT-recommended primary continuous outcome (May 2026)
    def gold_margin(ranked):
        best_gold = float('-inf')
        best_non_gold = float('-inf')
        for cid, score, _ in ranked:
            if cid in gold_set:
                best_gold = max(best_gold, score)
            else:
                best_non_gold = max(best_non_gold, score)
        if best_gold == float('-inf') or best_non_gold == float('-inf'):
            return float('nan')
        return best_gold - best_non_gold
    gold_margin_orig = gold_margin(orig_score_result['ranked'])
    gold_margin_pert = gold_margin(pert_score_result['ranked'])
    gold_margin_delta = gold_margin_pert - gold_margin_orig
    
    # Top-1 score delta: how much did the original top-1 candidate's score change?
    orig_top1_score = orig_score_result['top1'][1]
    # Find this candidate in the perturbed ranked list
    pert_top1_orig_score = float('nan')
    for cid, s, _ in pert_score_result['ranked']:
        if cid == top1_orig:
            pert_top1_orig_score = s
            break
    top1_score_delta = pert_top1_orig_score - orig_top1_score
    
    # Top-5 list-overlap metrics
    top5_orig_ids = [r[0] for r in orig_score_result['ranked'][:5]]
    top5_pert_ids = [r[0] for r in pert_score_result['ranked'][:5]]
    top5_jaccard = (len(set(top5_orig_ids) & set(top5_pert_ids)) /
                    len(set(top5_orig_ids) | set(top5_pert_ids))
                    if top5_orig_ids or top5_pert_ids else 0.0)
    top5_kendall = kendall_tau_b_top_k(top5_orig_ids, top5_pert_ids, k=5)
    rbo_p075 = rbo_at_k(top5_orig_ids, top5_pert_ids, k=5, p=0.75)
    rbo_p08  = rbo_at_k(top5_orig_ids, top5_pert_ids, k=5, p=0.80)
    rbo_p085 = rbo_at_k(top5_orig_ids, top5_pert_ids, k=5, p=0.85)
    rbo_p09  = rbo_at_k(top5_orig_ids, top5_pert_ids, k=5, p=0.90)
    
    return {
        'top1_orig_id': top1_orig,
        'top1_pert_id': top1_pert,
        'top1_flip': bool(top1_flip),
        'gold_rank_orig': int(gold_rank_orig),
        'gold_rank_pert': int(gold_rank_pert),
        'gold_rank_change': int(gold_rank_pert - gold_rank_orig),
        'gold_score_orig': float(gold_score_orig),
        'gold_score_pert': float(gold_score_pert),
        'gold_score_delta': float(gold_score_delta),
        'top1_score_delta': float(top1_score_delta),
        'gold_margin_orig': float(gold_margin_orig),
        'gold_margin_pert': float(gold_margin_pert),
        'gold_margin_delta': float(gold_margin_delta),
        'top5_jaccard': float(top5_jaccard),
        'top5_kendall_tau_b': float(top5_kendall) if not np.isnan(top5_kendall) else None,
        'top5_rbo_p075': float(rbo_p075),
        'top5_rbo_p08':  float(rbo_p08),
        'top5_rbo_p085': float(rbo_p085),
        'top5_rbo_p09':  float(rbo_p09),
        'correct_orig': bool(correct_orig),
        'correct_pert': bool(correct_pert),
        'top5_orig_ids': top5_orig_ids,
        'top5_pert_ids': top5_pert_ids,
    }


# ============================================================================
# Main pipeline
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='ACSAC Step D counterfactual probe')
    parser.add_argument('--data-path',
                        default='data/reranker_pairs_enriched_v2.jsonl',
                        help='Path to v2 enriched JSONL.')
    parser.add_argument('--model-path',
                        default='checkpoints/best_two_stage_v2',
                        help='Path to v2 cross-encoder checkpoint.')
    parser.add_argument('--stix-path',
                        default='data/enterprise-attack-v14.json',
                        help='Path to v14 STIX bundle.')
    parser.add_argument('--vocab-path',
                        default='eval_results_v2/categorization/vocabulary_v14.json',
                        help='Path to vocabulary_v14.json from Step A.')
    parser.add_argument('--output-dir',
                        default='eval_results_v2/counterfactual',
                        help='Output directory for Step D artifacts.')
    parser.add_argument('--n-substitutes', type=int, default=5,
                        help='Number of substitutes per query for swap/'
                             'malware/proper-noun pools.')
    parser.add_argument('--skip-scoring', action='store_true',
                        help='Build manifest only; skip model scoring (for testing).')
    parser.add_argument('--limit-queries', type=int, default=None,
                        help='Limit total scoring loops to first N queries '
                             '(for smoke testing).')
    args = parser.parse_args()
    
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'figures'), exist_ok=True)
    
    # Set determinism
    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    
    # Resolve input paths against likely roots before the script needs them.
    # We try the user-supplied path first, then the script's parent
    # directories, then the cwd. This makes the script forgiving of
    # whether the STIX bundle lives in `data/` or in the reranker root,
    # and similarly for vocabulary_v14.json.
    here = os.path.dirname(os.path.abspath(__file__))
    search_roots = [here]
    cur = here
    for _ in range(4):
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        search_roots.append(parent)
        cur = parent
    search_roots.append(os.getcwd())
    
    args.data_path  = _resolve_input_path(args.data_path,  search_roots)
    args.stix_path  = _resolve_input_path(args.stix_path,  search_roots)
    args.vocab_path = _resolve_input_path(args.vocab_path, search_roots)
    args.model_path = _resolve_input_path(args.model_path, search_roots)
    
    print("=" * 75)
    print("ACSAC 2026 Step D: Counterfactual Probe")
    print("=" * 75)
    print(f"  data:          {args.data_path}")
    print(f"  model:         {args.model_path}")
    print(f"  stix:          {args.stix_path}")
    print(f"  vocab:         {args.vocab_path}")
    print(f"  output_dir:    {out_dir}")
    print(f"  n_substitutes: {args.n_substitutes}")
    print(f"  seeds: model={MODEL_SEED}, sub={SUBSTITUTE_SAMPLING_SEED}, "
          f"perm={PERMUTATION_SEED}, boot={BOOTSTRAP_SEED}, "
          f"insert={INSERTION_TEMPLATE_SEED}")
    print()
    
    # -------- Step 1: Import v2_reeval and load test split --------
    project_root = _setup_v2_imports()
    print(f"  v2_reeval source: {project_root}")
    from v2_reeval import load_and_group_data, create_test_split
    
    print("\n[1/7] Loading v2 enriched data and creating test split...")
    query_data, total_rows = load_and_group_data(args.data_path)
    test_queries = create_test_split(query_data)
    print(f"  Total rows: {total_rows}; queries: {len(query_data)}; "
          f"test set: {len(test_queries)}")
    assert len(test_queries) == 146, f"Test set size mismatch: {len(test_queries)} != 146"
    
    # -------- Step 2: Build safe-alias-token map and stratify --------
    print("\n[2/7] Stratifying queries into Stratum A (actor-bearing) "
          "and Stratum B (actor-free)...")
    safe_alias, shared_tokens = build_safe_alias_tokens(
        args.stix_path, args.vocab_path, TEST_ACTOR_CANONICAL)
    for ta, toks in safe_alias.items():
        print(f"    {ta}: {len(toks)} safe alias tokens")
    print(f"    Shared-across-actors tokens excluded: {sorted(shared_tokens)}")
    
    stratum_a, stratum_b = stratify_test_queries(query_data, test_queries, safe_alias)
    print(f"  Stratum A: {len(stratum_a)}; Stratum B: {len(stratum_b)}")
    print(f"  A breakdown: {Counter(r['actor'] for r in stratum_a)}")
    print(f"  B breakdown: {Counter(r['actor'] for r in stratum_b)}")
    assert len(stratum_a) == 21, f"Stratum A size mismatch: {len(stratum_a)} != 21"
    assert len(stratum_b) == 125, f"Stratum B size mismatch: {len(stratum_b)} != 125"
    
    # -------- Step 3: Build substitute pools and technique-prior map --------
    print("\n[3/7] Building substitute pools and actor-technique-prior map...")
    with open(args.vocab_path) as f:
        vocab = json.load(f)
    test_norms_to_exclude = {'apt29','carbanak','fin6','fin7','oilrig','sandworm','wizard'}
    collision_tokens = set(vocab.get('curated_token_collisions_used', []))
    
    pool_a = build_pool_actor_primary(args.vocab_path,
                                       test_norms_to_exclude, collision_tokens)
    pool_b = build_pool_proper_noun(args.vocab_path)
    pool_d = build_pool_malware(args.vocab_path)
    print(f"  Pool A (actor swap): {len(pool_a)}")
    print(f"  Pool B (proper-noun placebo): {len(pool_b)}")
    print(f"  Pool D (malware placebo): {len(pool_d)}")
    
    actor_techs = build_actor_technique_priors(args.stix_path)
    print(f"  Built actor-technique priors for {len(actor_techs)} actors")
    
    # Save substitute_pool.json and actor_technique_priors.json
    with open(os.path.join(out_dir, 'substitute_pool.json'), 'w') as f:
        json.dump({
            'pool_a_actor_primary': pool_a,
            'pool_b_proper_noun':   pool_b,
            'pool_c_generic_phrase': GENERIC_PHRASE_POOL,
            'pool_d_malware':       pool_d,
            'excluded_test_actor_norms': sorted(test_norms_to_exclude),
            'excluded_collision_tokens': sorted(collision_tokens),
            'shared_across_actors_tokens': sorted(shared_tokens),
            'safe_alias_tokens_per_actor': {k: sorted(v)
                                              for k, v in safe_alias.items()},
        }, f, indent=2)
    
    with open(os.path.join(out_dir, 'actor_technique_priors.json'), 'w') as f:
        json.dump({
            'actor_to_techniques': actor_techs,
            'pairwise_jaccard_test_actors': {
                f"{a}__vs__{b}": jaccard_similarity(
                    actor_techs.get(TEST_ACTOR_CANONICAL[a]['stix_g_id'], []),
                    actor_techs.get(TEST_ACTOR_CANONICAL[b]['stix_g_id'], []))
                for a in TEST_ACTOR_CANONICAL for b in TEST_ACTOR_CANONICAL if a < b
            },
        }, f, indent=2)
    
    # -------- Step 4: Pre-build the manifest of all (query, intervention,
    # substitute) triples WITHOUT scoring (so we can audit before model load)
    # --------
    print("\n[4/7] Building manifest of all perturbation triples...")
    manifest = build_manifest(stratum_a, stratum_b, pool_a, pool_b, pool_d,
                               actor_techs, args.n_substitutes)
    print(f"  Total perturbed triples: {sum(1 for m in manifest if not m['is_baseline'])}")
    print(f"  Baseline (unperturbed) entries: {sum(1 for m in manifest if m['is_baseline'])}")
    print(f"  Total scoring loops needed: {len(manifest)}")
    
    # Save the pre-scoring manifest
    with open(os.path.join(out_dir, 'counterfactual_manifest.jsonl'), 'w') as f:
        for m in manifest:
            f.write(json.dumps(m) + '\n')
    
    # -------- Step 5: Model loading and scoring --------
    if args.skip_scoring:
        print("\n[5/7] --skip-scoring set; halting before model load.")
        return
    
    print("\n[5/7] Loading v2 cross-encoder model + tokenizer...")
    try:
        from sentence_transformers import CrossEncoder
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"  FATAL: cannot import sentence_transformers/transformers: {e}")
        return
    
    model = CrossEncoder(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"  Model loaded from {args.model_path}")
    
    # -------- Step 6: Score everything --------
    print("\n[6/7] Scoring every (query, intervention, substitute) triple...")
    if args.limit_queries:
        manifest = manifest[:args.limit_queries]
        print(f"  --limit-queries {args.limit_queries}: limiting manifest")
    
    results = []
    cache_orig = {}  # qn -> score_query result on the unperturbed query
    
    t0 = time.time()
    for i, m in enumerate(manifest):
        # Build the perturbed query text
        if m['intervention'] == 'baseline':
            query_text = m['query_raw']
        elif m['intervention'].startswith('insertion_'):
            query_text = m['perturbed_query']
        else:
            # swap, proper_noun, generic_phrase, malware, soft_deletion
            query_text = m['perturbed_query']
        
        # Score
        cands = m['candidates_for_scoring']  # list of dicts
        result = score_query(model, query_text, cands)
        
        if m['is_baseline']:
            # Cache the result for later perturbed-vs-baseline comparisons.
            cache_orig[m['qn']] = result
            # CRITICAL: still populate the metric columns for baseline
            # rows so cf_figures.py can read correct_orig/correct_pert
            # from baselines when building the McNemar contingency table.
            # Comparing the result to itself gives all-zero deltas, no
            # flip, full RBO/Jaccard/Kendall agreement, and correctness
            # flags that reflect whether the unperturbed top-1 is in
            # the gold set. This is the natural "self-comparison"
            # interpretation of a baseline row.
            baseline_metrics = compute_metrics(result, result, m['gold_ids'])
            baseline_metrics['subword_token_delta'] = 0
            results.append({**m, **baseline_metrics})
            continue
        
        # Compute metrics against the cached original
        orig_result = cache_orig.get(m['qn'])
        if orig_result is None:
            # Score the original first
            orig_text = m['query_raw']
            orig_result = score_query(model, orig_text, cands)
            cache_orig[m['qn']] = orig_result
        
        metrics = compute_metrics(orig_result, result, m['gold_ids'])
        
        # Compute subword token delta
        if 'perturbed_query' in m:
            tok_delta = compute_subword_token_delta(tokenizer, m['query_raw'],
                                                     m['perturbed_query'])
            metrics['subword_token_delta'] = tok_delta
        else:
            metrics['subword_token_delta'] = -999
        
        results.append({**m, **metrics})
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{len(manifest)}] elapsed: {elapsed:.0f}s")
    
    elapsed = time.time() - t0
    print(f"  Total scoring time: {elapsed:.0f}s "
          f"({elapsed / max(len(results), 1):.2f}s per loop)")
    
    # -------- Step 7: Write outputs --------
    print("\n[7/7] Writing CSV results, aggregates, summary, and provenance...")
    write_results_csv(results, os.path.join(out_dir, 'counterfactual_results.csv'))
    
    aggregates = compute_aggregates(results)
    with open(os.path.join(out_dir, 'counterfactual_aggregates.json'), 'w') as f:
        json.dump(aggregates, f, indent=2)
    
    summary = build_summary_md(aggregates, len(stratum_a), len(stratum_b))
    with open(os.path.join(out_dir, 'counterfactual_summary.md'), 'w') as f:
        f.write(summary)
    
    provenance = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'seeds': {
            'PYTHONHASHSEED': PYTHONHASHSEED,
            'MODEL_SEED': MODEL_SEED,
            'SUBSTITUTE_SAMPLING_SEED': SUBSTITUTE_SAMPLING_SEED,
            'PERMUTATION_SEED': PERMUTATION_SEED,
            'BOOTSTRAP_SEED': BOOTSTRAP_SEED,
            'INSERTION_TEMPLATE_SEED': INSERTION_TEMPLATE_SEED,
        },
        'paths': {
            'data': args.data_path,
            'model': args.model_path,
            'stix': args.stix_path,
            'vocab': args.vocab_path,
            'output_dir': out_dir,
        },
        'counts': {
            'test_queries': len(test_queries),
            'stratum_a': len(stratum_a),
            'stratum_b': len(stratum_b),
            'pool_a_size': len(pool_a),
            'pool_b_size': len(pool_b),
            'pool_d_size': len(pool_d),
            'manifest_entries': len(manifest),
            'scored_entries': len(results),
        },
        'runtime_seconds': elapsed,
    }
    with open(os.path.join(out_dir, 'provenance_step_d.json'), 'w') as f:
        json.dump(provenance, f, indent=2)
    
    print(f"\nStep D pipeline COMPLETE. Outputs in: {out_dir}")


# ============================================================================
# Manifest builder
# ============================================================================
def build_manifest(stratum_a, stratum_b, pool_a, pool_b, pool_d,
                   actor_techs, n_substitutes):
    """
    Build the complete pre-scoring manifest of every (query, intervention,
    substitute) triple. The manifest is the artifact that lets a reviewer
    audit exactly what perturbations were generated, BEFORE any scoring,
    per Document 12's reproducibility schema.
    
    Returns: list of dicts. Each dict has at minimum:
        qn, actor, query_raw, gold_ids, candidates_for_scoring,
        is_baseline (bool), intervention (str), and for non-baseline:
        substitute (dict), perturbed_query (str)
    """
    insertion_rng = np.random.default_rng(INSERTION_TEMPLATE_SEED)
    manifest = []
    
    # Add baselines first (one per query, both strata)
    for rec in stratum_a + stratum_b:
        manifest.append({
            'manifest_id': f"baseline__{rec['qn'][:32]}",
            'qn': rec['qn'],
            'stratum': 'A' if rec in stratum_a else 'B',
            'actor': rec['actor'],
            'query_raw': rec['query_raw'],
            'perturbed_query': rec['query_raw'],
            'gold_ids': rec['gold_ids'],
            'candidates_for_scoring': rec['candidates'],
            'is_baseline': True,
            'intervention': 'baseline',
            'substitute': None,
        })
    
    # ---- Stratum A interventions ----
    for rec in stratum_a:
        # Identify spans (we use the FIRST span only for substitution; if
        # the query has multiple alias-token mentions we record all spans
        # but perturb only one for clean intervention semantics)
        safe_set = set(rec['safe_alias_match'])
        spans = find_actor_spans(rec['query_raw'], safe_set)
        if not spans:
            continue
        primary_span = spans[0]
        original_surface = primary_span['surface']
        actor = rec['actor']
        actor_g_id = TEST_ACTOR_CANONICAL[actor]['stix_g_id']
        
        # Build the actor's family-alias norm set (to exclude from swap)
        family_aliases_norm = set()
        for tok in safe_set:
            family_aliases_norm.add(tok)
        # Also add the test-actor's own normalized name
        family_aliases_norm.add(actor)
        # And the well-known FIN family / APT29 alias cluster expansion
        # is already captured via safe_set.
        
        # ---- C1: actor swap (primary) ----
        c1_subs = select_substitutes(
            original_surface=original_surface,
            original_actor_g_id=actor_g_id,
            gold_technique_set=rec['gold_ids'],
            pool=pool_a,
            actor_techniques=actor_techs,
            n_substitutes=n_substitutes,
            exclude_norms=family_aliases_norm,
            seed=SUBSTITUTE_SAMPLING_SEED,
        )
        for j, s in enumerate(c1_subs):
            pq = apply_substitute(rec['query_raw'], primary_span, s['recast'])
            manifest.append({
                'manifest_id': f"swap__{rec['qn'][:24]}__{j}",
                'qn': rec['qn'],
                'stratum': 'A',
                'actor': actor,
                'query_raw': rec['query_raw'],
                'perturbed_query': pq,
                'gold_ids': rec['gold_ids'],
                'candidates_for_scoring': rec['candidates'],
                'is_baseline': False,
                'intervention': 'swap',
                'substitute': s,
                'span_used': primary_span,
                'original_surface': original_surface,
            })
        
        # ---- C2: proper-noun placebo ----
        c2_subs = select_substitutes(
            original_surface=original_surface,
            original_actor_g_id=actor_g_id,
            gold_technique_set=rec['gold_ids'],
            pool=pool_b,
            actor_techniques=actor_techs,
            n_substitutes=n_substitutes,
            exclude_norms=set(),
            seed=SUBSTITUTE_SAMPLING_SEED + 1,
        )
        for j, s in enumerate(c2_subs):
            pq = apply_substitute(rec['query_raw'], primary_span, s['recast'])
            manifest.append({
                'manifest_id': f"propernoun__{rec['qn'][:24]}__{j}",
                'qn': rec['qn'],
                'stratum': 'A',
                'actor': actor,
                'query_raw': rec['query_raw'],
                'perturbed_query': pq,
                'gold_ids': rec['gold_ids'],
                'candidates_for_scoring': rec['candidates'],
                'is_baseline': False,
                'intervention': 'proper_noun_placebo',
                'substitute': s,
                'span_used': primary_span,
                'original_surface': original_surface,
            })
        
        # ---- C3: generic-phrase placebo ----
        # Pass the 4-char window before the span so select_generic_phrase
        # can detect "The "/"the " article-precedence and use article-less
        # substitute forms to avoid "The the adversary" duplicate articles.
        preceding_text = (rec['query_raw'][max(0, primary_span['start']-4):primary_span['start']]
                          if primary_span['start'] > 0 else '')
        c3_subs = select_generic_phrase(primary_span['start'], n_substitutes=3,
                                         preceding_text=preceding_text)
        for j, s in enumerate(c3_subs):
            pq = apply_substitute(rec['query_raw'], primary_span, s['recast'])
            manifest.append({
                'manifest_id': f"generic__{rec['qn'][:24]}__{j}",
                'qn': rec['qn'],
                'stratum': 'A',
                'actor': actor,
                'query_raw': rec['query_raw'],
                'perturbed_query': pq,
                'gold_ids': rec['gold_ids'],
                'candidates_for_scoring': rec['candidates'],
                'is_baseline': False,
                'intervention': 'generic_phrase_placebo',
                'substitute': s,
                'span_used': primary_span,
                'original_surface': original_surface,
            })
        
        # ---- C4: malware placebo (per user's confirmation) ----
        c4_subs = select_substitutes(
            original_surface=original_surface,
            original_actor_g_id=actor_g_id,
            gold_technique_set=rec['gold_ids'],
            pool=pool_d,
            actor_techniques=actor_techs,
            n_substitutes=n_substitutes,
            exclude_norms=set(),
            seed=SUBSTITUTE_SAMPLING_SEED + 2,
        )
        for j, s in enumerate(c4_subs):
            pq = apply_substitute(rec['query_raw'], primary_span, s['recast'])
            manifest.append({
                'manifest_id': f"malware__{rec['qn'][:24]}__{j}",
                'qn': rec['qn'],
                'stratum': 'A',
                'actor': actor,
                'query_raw': rec['query_raw'],
                'perturbed_query': pq,
                'gold_ids': rec['gold_ids'],
                'candidates_for_scoring': rec['candidates'],
                'is_baseline': False,
                'intervention': 'malware_placebo',
                'substitute': s,
                'span_used': primary_span,
                'original_surface': original_surface,
            })
        
        # ---- C5: soft deletion ----
        c5_subs = select_soft_deletion(primary_span['start'], original_surface)
        for j, s in enumerate(c5_subs):
            pq = apply_substitute(rec['query_raw'], primary_span, s['recast'])
            manifest.append({
                'manifest_id': f"deletion__{rec['qn'][:24]}__{j}",
                'qn': rec['qn'],
                'stratum': 'A',
                'actor': actor,
                'query_raw': rec['query_raw'],
                'perturbed_query': pq,
                'gold_ids': rec['gold_ids'],
                'candidates_for_scoring': rec['candidates'],
                'is_baseline': False,
                'intervention': 'soft_deletion',
                'substitute': s,
                'span_used': primary_span,
                'original_surface': original_surface,
            })
    
    # ---- Stratum B insertion (user's Option A: prepend "[ACTOR] was observed ") ----
    for rec in stratum_b:
        actor = rec['actor']
        actor_canonical = TEST_ACTOR_CANONICAL[actor]['canonical_name']
        actor_g_id = TEST_ACTOR_CANONICAL[actor]['stix_g_id']
        
        # ---- C6a: own-actor insertion ----
        pq, prefix = apply_insertion(rec['query_raw'], actor_canonical)
        manifest.append({
            'manifest_id': f"insert_own__{rec['qn'][:24]}",
            'qn': rec['qn'],
            'stratum': 'B',
            'actor': actor,
            'query_raw': rec['query_raw'],
            'perturbed_query': pq,
            'gold_ids': rec['gold_ids'],
            'candidates_for_scoring': rec['candidates'],
            'is_baseline': False,
            'intervention': 'insertion_own_actor',
            'substitute': {
                'norm': actor,
                'primary': actor_canonical,
                'recast': actor_canonical,
                'bucket': 'insertion_template',
                'fallback_level': 0,
                'tech_jaccard': 1.0,  # by definition - own actor
                'subword_token_delta': -1,
            },
            'span_used': None,
            'original_surface': None,
            'insertion_prefix': prefix,
        })
        
        # ---- C6b: different-actor insertion ----
        c6b_subs = select_substitutes(
            original_surface=actor_canonical,
            original_actor_g_id=actor_g_id,
            gold_technique_set=rec['gold_ids'],
            pool=pool_a,
            actor_techniques=actor_techs,
            n_substitutes=5,
            exclude_norms={'apt29','carbanak','fin6','fin7','oilrig',
                            'sandworm','wizard'} | {actor},
            seed=SUBSTITUTE_SAMPLING_SEED + 3,
        )
        for j, s in enumerate(c6b_subs):
            pq, prefix = apply_insertion(rec['query_raw'], s['recast'])
            manifest.append({
                'manifest_id': f"insert_diff__{rec['qn'][:24]}__{j}",
                'qn': rec['qn'],
                'stratum': 'B',
                'actor': actor,
                'query_raw': rec['query_raw'],
                'perturbed_query': pq,
                'gold_ids': rec['gold_ids'],
                'candidates_for_scoring': rec['candidates'],
                'is_baseline': False,
                'intervention': 'insertion_different_actor',
                'substitute': s,
                'span_used': None,
                'original_surface': None,
                'insertion_prefix': prefix,
            })
    
    return manifest


# ============================================================================
# CSV writer
# ============================================================================
def write_results_csv(results, output_path):
    """
    Write the flat Minimal Pairs CSV per Perplexity Comprehensive
    Synthesis recommendation.
    """
    fieldnames = [
        'manifest_id', 'qn', 'stratum', 'actor', 'intervention',
        'is_baseline', 'original_surface', 'substitute_norm',
        'substitute_primary', 'substitute_recast', 'substitute_bucket',
        'fallback_level', 'tech_jaccard', 'subword_token_delta',
        'gold_ids',
        'query_raw', 'perturbed_query',
        'top1_orig_id', 'top1_pert_id', 'top1_flip',
        'gold_rank_orig', 'gold_rank_pert', 'gold_rank_change',
        'gold_score_orig', 'gold_score_pert', 'gold_score_delta',
        'top1_score_delta', 'gold_margin_orig', 'gold_margin_pert',
        'gold_margin_delta',
        'top5_jaccard', 'top5_kendall_tau_b',
        'top5_rbo_p075', 'top5_rbo_p08', 'top5_rbo_p085', 'top5_rbo_p09',
        'correct_orig', 'correct_pert',
    ]
    with open(output_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in results:
            row = {}
            row['manifest_id'] = r.get('manifest_id', '')
            row['qn'] = r.get('qn', '')
            row['stratum'] = r.get('stratum', '')
            row['actor'] = r.get('actor', '')
            row['intervention'] = r.get('intervention', '')
            row['is_baseline'] = r.get('is_baseline', '')
            row['original_surface'] = r.get('original_surface') or ''
            sub = r.get('substitute') or {}
            row['substitute_norm'] = sub.get('norm', '') if sub else ''
            row['substitute_primary'] = sub.get('primary', '') if sub else ''
            row['substitute_recast'] = sub.get('recast', '') if sub else ''
            row['substitute_bucket'] = sub.get('bucket', '') if sub else ''
            row['fallback_level'] = sub.get('fallback_level', '') if sub else ''
            row['tech_jaccard'] = sub.get('tech_jaccard', '') if sub else ''
            row['gold_ids'] = ';'.join(r.get('gold_ids', []))
            # Include the original and perturbed query strings so the
            # case-study figure (cf_figures.py fig4) can render the
            # actual text, and so reviewers can audit any row visually.
            row['query_raw'] = r.get('query_raw', '') or ''
            row['perturbed_query'] = r.get('perturbed_query', '') or ''
            for k in ['top1_orig_id', 'top1_pert_id', 'top1_flip',
                      'gold_rank_orig', 'gold_rank_pert', 'gold_rank_change',
                      'gold_score_orig', 'gold_score_pert', 'gold_score_delta',
                      'top1_score_delta', 'gold_margin_orig',
                      'gold_margin_pert', 'gold_margin_delta',
                      'top5_jaccard', 'top5_kendall_tau_b',
                      'top5_rbo_p075', 'top5_rbo_p08', 'top5_rbo_p085',
                      'top5_rbo_p09', 'correct_orig', 'correct_pert',
                      'subword_token_delta']:
                row[k] = r.get(k, '')
            w.writerow(row)


# ============================================================================
# Aggregates and summary
# ============================================================================
def compute_aggregates(results):
    """
    Compute by-actor and by-stratum aggregates with Wilson CIs and
    paired statistics. The bulk of the inferential layer (paired
    permutation, BCa bootstrap, mid-p McNemar, mixed-effects) lives in
    cf_figures.py and runs as a post-processing step on the CSV; the
    aggregates here are descriptive summaries for sanity-checking.
    """
    # Group results by intervention
    by_intervention = defaultdict(list)
    for r in results:
        if r.get('is_baseline'):
            continue
        by_intervention[r['intervention']].append(r)
    
    aggs = {}
    for intervention, rows in by_intervention.items():
        flips = [r['top1_flip'] for r in rows if 'top1_flip' in r]
        margin_deltas = [r.get('gold_margin_delta', float('nan')) for r in rows]
        margin_deltas = [d for d in margin_deltas if not (isinstance(d, float) and np.isnan(d))]
        rank_changes = [r.get('gold_rank_change', 0) for r in rows]
        score_deltas = [r.get('gold_score_delta', float('nan')) for r in rows]
        score_deltas = [d for d in score_deltas if not (isinstance(d, float) and np.isnan(d))]
        
        n = len(rows)
        n_flip = sum(flips) if flips else 0
        flip_rate = n_flip / n if n else 0.0
        
        # Wilson 95% CI on flip rate
        from scipy.stats import beta as beta_dist
        if n > 0:
            # Wilson score interval
            from scipy.stats import norm as norm_dist
            z = norm_dist.ppf(0.975)
            p = flip_rate
            denom = 1 + z*z/n
            center = (p + z*z/(2*n)) / denom
            half = (z * (p*(1-p)/n + z*z/(4*n*n))**0.5) / denom
            wilson_lo = max(0, center - half)
            wilson_hi = min(1, center + half)
        else:
            wilson_lo = wilson_hi = 0.0
        
        aggs[intervention] = {
            'n_triples': n,
            'flip_rate': flip_rate,
            'wilson_95ci_lo': float(wilson_lo),
            'wilson_95ci_hi': float(wilson_hi),
            'mean_margin_delta': float(np.mean(margin_deltas)) if margin_deltas else float('nan'),
            'std_margin_delta':  float(np.std(margin_deltas, ddof=1)) if len(margin_deltas) > 1 else float('nan'),
            'mean_score_delta':  float(np.mean(score_deltas)) if score_deltas else float('nan'),
            'mean_rank_change':  float(np.mean(rank_changes)) if rank_changes else 0.0,
        }
    
    # Per-actor breakdown for swap intervention only (focal claim)
    swap_by_actor = defaultdict(list)
    for r in results:
        if r.get('intervention') == 'swap':
            swap_by_actor[r['actor']].append(r)
    aggs['swap_per_actor'] = {}
    for a, rows in swap_by_actor.items():
        flips = [r['top1_flip'] for r in rows]
        margin_deltas = [r.get('gold_margin_delta', float('nan')) for r in rows]
        margin_deltas = [d for d in margin_deltas if not np.isnan(d)]
        aggs['swap_per_actor'][a] = {
            'n_triples': len(rows),
            'flip_rate': sum(flips) / len(rows) if rows else 0.0,
            'mean_margin_delta': float(np.mean(margin_deltas)) if margin_deltas else float('nan'),
        }
    
    return aggs


def build_summary_md(aggregates, n_a, n_b):
    """Write the paper-ready summary markdown."""
    lines = []
    lines.append("# Step D — Counterfactual Probe Summary\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    lines.append(f"Stratum A (actor-bearing): {n_a} queries; "
                  f"Stratum B (actor-free): {n_b} queries.\n\n")
    lines.append("## Aggregate flip rates by intervention\n")
    lines.append("| Intervention | N triples | Flip rate | Wilson 95% CI | "
                  "Mean Δ-margin | Mean Δ-rank |\n")
    lines.append("|---|---:|---:|---|---:|---:|\n")
    for k, v in aggregates.items():
        if k == 'swap_per_actor':
            continue
        wilson = f"[{v['wilson_95ci_lo']:.3f}, {v['wilson_95ci_hi']:.3f}]"
        lines.append(f"| {k} | {v['n_triples']} | "
                      f"{v['flip_rate']:.3f} | {wilson} | "
                      f"{v['mean_margin_delta']:.3f} | "
                      f"{v['mean_rank_change']:.3f} |\n")
    lines.append("\n## Swap flip rate per actor (Stratum A)\n")
    lines.append("| Actor | N triples | Flip rate | Mean Δ-margin |\n")
    lines.append("|---|---:|---:|---:|\n")
    for a, v in aggregates.get('swap_per_actor', {}).items():
        lines.append(f"| {a} | {v['n_triples']} | "
                      f"{v['flip_rate']:.3f} | {v['mean_margin_delta']:.3f} |\n")
    lines.append("\n## Note on inferential statistics\n")
    lines.append("Paired permutation tests, BCa bootstrap CIs, mid-p\n")
    lines.append("McNemar, RBO-stability, and the appendix mixed-effects\n")
    lines.append("model are computed by cf_figures.py as a separate\n")
    lines.append("post-processing pass. See counterfactual_results.csv\n")
    lines.append("for per-triple data and provenance_step_d.json for\n")
    lines.append("seed and runtime metadata.\n")
    return ''.join(lines)


if __name__ == '__main__':
    main()
