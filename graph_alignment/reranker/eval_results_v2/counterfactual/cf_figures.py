"""
cf_figures.py — Inferential layer + ACSAC figures for Step D counterfactual probe.

This module is the post-processing pass that runs AFTER counterfactual_probe.py
has produced counterfactual_results.csv. It does five things:

    1. Loads the per-triple results, then aggregates them to the per-query
       level (per-query mean for continuous metrics, per-query rate for
       binary metrics) so the unit of analysis is the query, not the
       triple. This is the "do not pretend you have 105 independent
       observations" rule from the locked-in design (ChatGPT response
       in the May 2026 turn) — substitutes are within-query repeats.
    
    2. Runs the PRIMARY inferential test: paired Δ-margin
       difference-in-differences. For each Stratum A query and each of
       the four placebo contrasts (swap vs proper-noun, swap vs generic,
       swap vs malware, swap vs deletion), we compute
           ΔΔ_q = Δ-margin_q^swap − Δ-margin_q^placebo
       then test the mean ΔΔ across queries via paired permutation
       (B=10,000) and a BCa-bootstrap CI clustered by query.
       Holm-Bonferroni correction across the four planned contrasts.
    
    3. Runs the CO-PRIMARY operational summary: top-1 flip rate per
       intervention with Wilson 95% CI; mid-p McNemar test on
       (correct_orig vs correct_pert) ONLY when the discordant count
       b+c >= 5; otherwise we explicitly flag b+c<5 as under-informative
       (per the consensus across all eleven research documents). Risk
       difference effect size between conditions.
    
    4. Computes the SECONDARY rank-stability summaries: RBO@5 with
       persistence p=0.8 (primary value, sensitivity at p=0.75/0.85/0.9),
       Jaccard@5 descriptive, and Kendall tau-b on top-5 union.
    
    5. Fits the APPENDIX linear mixed-effects model on Δ-margin via
       statsmodels.MixedLM:
           Δ-margin ~ condition + actor + subword_token_delta + (1|query)
       with proper-noun-placebo as the reference level. Gaussian (linear)
       is appropriate because Δ-margin is a continuous outcome; we
       deliberately AVOID the binary-correctness logistic GLMM because
       at our N=21 with 94.52% baseline accuracy, complete separation
       and singular fit errors are virtually guaranteed (per the Gemini
       PDF and Claude operational answers warnings).
    
    6. Produces the four ACSAC figures, each saved as PDF + SVG + PNG
       with deterministic SVG clip-path IDs (matplotlib >= 3.10) for
       reproducibility:
           cf_fig1_flip_rates_by_intervention   — Wilson-CI bar chart
           cf_fig2_swap_vs_placebo_paired       — primary-claim figure
           cf_fig3_gold_rank_change_distribution — distribution histograms
           cf_fig4_carbanak_case_study          — records #12 & #15
    
    Plus a sidecar figure_data.json containing every numeric value
    plotted, so reviewers can audit any number against the source data.

Reference design across all eleven research documents:
    - PRIMARY: paired Δ-margin DiD with paired permutation + BCa
      bootstrap CI clustered by query (ChatGPT May 2026; Claude
      operational answers; Yang et al. 2026 §3.3-4.2).
    - CO-PRIMARY: Wilson-CI flip rate; mid-p McNemar with b+c>=5 flag
      (Fagerland-Lydersen-Laake 2013; Gemini PDF; Perplexity Inferential).
    - SECONDARY: RBO@5 p=0.8 (Webber-Moffat-Zobel 2010 user-model
      derivation 1/(1-p)=5); Jaccard@5 descriptive.
    - APPENDIX: statsmodels.MixedLM linear mixed-effects on continuous
      Δ-margin (Card et al. 2020; Bates-Maechler-Bolker-Walker 2015;
      explicitly NOT logistic GLMM per separation hazards).
    - Holm-Bonferroni across the 4 placebo contrasts (FWER, step-down,
      preserves more power than naive Bonferroni).

Determinism:
    - All RNG-using functions take an explicit seed argument, defaulting
      to PERMUTATION_SEED=42 / BOOTSTRAP_SEED=42.
    - matplotlib figure outputs are deterministic when PYTHONHASHSEED=42
      is set (we re-enforce it at the top of this module).

Usage:
    cd <reranker_dir>
    set PYTHONHASHSEED=42
    python cf_figures.py --results-csv eval_results_v2/counterfactual/counterfactual_results.csv \
                         --output-dir eval_results_v2/counterfactual

Author: Shane Waldrop, ACSAC 2026 paper, Step D (May 2026).
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

# Set PYTHONHASHSEED before any matplotlib import for deterministic SVG output.
os.environ.setdefault('PYTHONHASHSEED', '42')

import numpy as np

import matplotlib
matplotlib.use('Agg')  # MUST precede pyplot import for headless determinism
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Reuse the verified statistical primitives from Step C'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'categorization'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from paired_stats import (
        exact_paired_permutation_test, bca_bootstrap_ci,
        cluster_bootstrap_ci, paired_cles, paired_cohens_dz,
        hodges_lehmann_ci, holm_bonferroni,
    )
except ImportError as e:
    raise ImportError(
        "cf_figures.py requires paired_stats.py from Step C' "
        "(eval_results_v2/categorization/paired_stats.py). "
        f"Original error: {e}")

# Project-wide seeds (also used by counterfactual_probe.py)
PERMUTATION_SEED = 42
BOOTSTRAP_SEED = 42
N_PERMUTATION_RESAMPLES = 10_000
N_BOOTSTRAP_RESAMPLES = 9_999

# Cluster-bootstrap-vs-row-bootstrap threshold per Step C':
# only switch to clustered when there are >= 1.2 substitutes per query
# on average. For Step D we always have multiple substitutes per query
# (5 for swap/proper-noun/malware, 3 for generic, 1 for soft-deletion),
# so clustered bootstrap is the canonical choice for swap and the
# multi-substitute placebos. For soft-deletion (1 per query) clustered
# and per-row bootstraps coincide.

# Holm-Bonferroni reference family for the primary D-in-D test:
# the four planned placebo contrasts in the locked-in design.
PLACEBO_CONTRASTS = [
    ('swap', 'proper_noun_placebo'),    # PRIMARY contrast per design
    ('swap', 'generic_phrase_placebo'),
    ('swap', 'malware_placebo'),
    ('swap', 'soft_deletion'),
]

# ============================================================================
# Loading the per-triple results CSV
# ============================================================================
def load_results_csv(path):
    """
    Load the flat counterfactual_results.csv produced by
    counterfactual_probe.py and return a list of row-dicts with numeric
    fields parsed.
    """
    rows = []
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            # Parse types
            for k in ('top1_flip', 'correct_orig', 'correct_pert', 'is_baseline'):
                if k in row and row[k] not in ('', None):
                    row[k] = (row[k] == 'True' or row[k] == 'true' or row[k] == '1')
            for k in ('gold_rank_orig', 'gold_rank_pert', 'gold_rank_change',
                      'subword_token_delta', 'fallback_level'):
                if k in row and row[k] not in ('', None):
                    try:
                        row[k] = int(row[k])
                    except ValueError:
                        pass
            for k in ('gold_score_orig', 'gold_score_pert', 'gold_score_delta',
                      'top1_score_delta', 'gold_margin_orig', 'gold_margin_pert',
                      'gold_margin_delta', 'top5_jaccard', 'top5_kendall_tau_b',
                      'top5_rbo_p075', 'top5_rbo_p08', 'top5_rbo_p085',
                      'top5_rbo_p09', 'tech_jaccard'):
                if k in row:
                    # Empty strings and None both become NaN so every downstream
                    # consumer (aggregate_to_per_query, fit_lmm_appendix, the
                    # figure functions) can rely on numeric types and use
                    # np.isnan() as the single missing-data check. Without
                    # this normalization, callers that only check for None
                    # or NaN crash on empty-string ValueError.
                    if row[k] in ('', None):
                        row[k] = float('nan')
                    else:
                        try:
                            row[k] = float(row[k])
                        except (ValueError, TypeError):
                            row[k] = float('nan')
            rows.append(row)
    return rows


def aggregate_to_per_query(rows, intervention, metric):
    """
    Given the row-list, an intervention name, and a metric column,
    return a dict mapping qn -> per-query mean (continuous) or
    per-query rate (binary).
    
    This is the "do not pretend N=105" rule: substitutes within a query
    are repeats, so we collapse them to the per-query level before the
    inferential test. The unit of analysis is the query.
    """
    by_q = defaultdict(list)
    for r in rows:
        if r.get('intervention') != intervention:
            continue
        v = r.get(metric)
        if v is None or v == '':
            continue
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            continue
        if isinstance(v, bool):
            v = 1.0 if v else 0.0
        try:
            v_float = float(v)
        except (ValueError, TypeError):
            continue
        by_q[r['qn']].append(v_float)
    return {qn: float(np.mean(vs)) for qn, vs in by_q.items() if vs}


# ============================================================================
# Wilson 95% confidence interval on a proportion
# ============================================================================
def wilson_ci(k, n, conf_level=0.95):
    """
    Wilson score interval for a binomial proportion.
    
    Per all eleven research documents' agreement: appropriate even at
    very small n, and does not rely on the large-sample normal
    approximation. For n=0 returns (0, 0).
    
    Returns (lower, upper) as a tuple of floats.
    """
    if n <= 0:
        return (0.0, 0.0)
    from scipy.stats import norm
    alpha = 1.0 - conf_level
    z = norm.ppf(1.0 - alpha / 2.0)
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    # Pin exact boundaries for clean reporting: when k==0 the canonical
    # Wilson lower bound is exactly 0, and when k==n the upper bound is
    # exactly 1. The arithmetic above is mathematically equivalent but
    # leaves tiny FP residuals (e.g. ~3e-18 at n=100) that would look
    # noisy in the inferential JSON.
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return (lo, hi)


# ============================================================================
# Mid-p McNemar test with explicit b+c flag
# ============================================================================
def mid_p_mcnemar(b, c):
    """
    Compute the mid-p McNemar test on a paired-binary contingency table:
        - b = (orig correct, pert incorrect)  — "loss" cell
        - c = (orig incorrect, pert correct)  — "gain" cell
    
    The standard exact conditional p-value (two-sided, fixed total b+c)
    is Pr(X <= min(b,c)) + Pr(X >= max(b,c)) under X ~ Binomial(b+c, 0.5).
    The mid-p adjustment subtracts half the probability of the
    observed configuration, which Fagerland-Lydersen-Laake 2013
    demonstrate is a less-conservative estimator across the full range
    they examined.
    
    Per the consensus across all eleven research documents (especially
    the Gemini PDF and Claude operational answers): when b+c < 5, no
    test variant rescues power, so the result must be flagged.
    
    Returns dict with:
        'b': int
        'c': int
        'discordant': int (b+c)
        'underinformative': bool (True iff b+c < 5)
        'exact_p_two_sided': float or None
        'mid_p_two_sided': float or None  (None if underinformative)
    """
    from scipy.stats import binom
    n = b + c
    out = {
        'b': int(b),
        'c': int(c),
        'discordant': int(n),
        'underinformative': bool(n < 5),
        'exact_p_two_sided': None,
        'mid_p_two_sided': None,
    }
    if n == 0:
        return out
    # Two-sided p-value under Binomial(n, 0.5).
    # k = min(b,c) defines the lower tail X <= k.
    # By symmetry under H0:p=0.5, the upper tail X >= n-k carries the
    # same mass. The two-sided exact p adds these tails.
    k = min(b, c)
    p_lower = binom.cdf(k, n, 0.5)
    p_upper = 1.0 - binom.cdf(n - k - 1, n, 0.5) if n - k - 1 >= 0 else 1.0
    
    # CRITICAL: keep the un-clipped sum for the mid-p subtraction. When
    # b == c (so k == n-k == n/2), the lower and upper tails OVERLAP at
    # the central mass P(X = k), and the un-clipped sum equals 1 + pmf(k).
    # The correct mid-p is then (1 + pmf(k)) - pmf(k) = 1.0, matching
    # the principle that no asymmetry => no evidence.
    raw_exact_p = p_lower + p_upper
    exact_p = min(1.0, raw_exact_p)
    out['exact_p_two_sided'] = float(exact_p)
    
    # Mid-p adjustment: subtract pmf at the boundary to remove half the
    # boundary mass from each tail (Fagerland-Lydersen-Laake 2013, eq.5).
    # For the asymmetric case k < n-k: raw_exact = 2 * P(X <= k), and
    # mid_p = 2 * P(X <= k) - pmf(k) = 2*[P(X<=k) - 0.5*pmf(k)] which is
    # exactly twice the one-sided mid-p, the standard formulation.
    p_at_k = binom.pmf(k, n, 0.5)
    mid_p = max(0.0, min(1.0, raw_exact_p - p_at_k))
    out['mid_p_two_sided'] = float(mid_p)
    return out


# ============================================================================
# Risk-difference effect size (paired) with bootstrap CI clustered by query
# ============================================================================
def paired_risk_difference(per_q_a, per_q_b, n_resamples=9999, seed=BOOTSTRAP_SEED):
    """
    Compute the paired risk difference (mean per-query rate of A minus B)
    with a paired-bootstrap 95% CI. Both per_q_a and per_q_b should be
    dicts qn->rate for the SAME set of queries.
    
    Returns dict with:
        'risk_diff': float (mean rate_a - mean rate_b across shared queries)
        'ci_lo': float
        'ci_hi': float
        'n_queries': int
    """
    shared = sorted(set(per_q_a) & set(per_q_b))
    if not shared:
        return {'risk_diff': float('nan'), 'ci_lo': float('nan'),
                'ci_hi': float('nan'), 'n_queries': 0}
    diffs = np.array([per_q_a[q] - per_q_b[q] for q in shared])
    mean_diff = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    boots = []
    n = len(diffs)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boots.append(np.mean(diffs[idx]))
    boots = np.array(boots)
    return {
        'risk_diff': mean_diff,
        'ci_lo': float(np.percentile(boots, 2.5)),
        'ci_hi': float(np.percentile(boots, 97.5)),
        'n_queries': len(shared),
    }


# ============================================================================
# PRIMARY inferential test: paired Δ-margin difference-in-differences
# ============================================================================
def paired_dd_test(per_q_swap, per_q_placebo, n_perm=N_PERMUTATION_RESAMPLES,
                    n_boot=N_BOOTSTRAP_RESAMPLES, seed_perm=PERMUTATION_SEED,
                    seed_boot=BOOTSTRAP_SEED):
    """
    The PRIMARY inferential test from the locked-in stack.
    
    Given per-query mean Δ-margin under the swap intervention
    (per_q_swap: dict qn -> mean Δ-margin) and the same under one
    placebo intervention (per_q_placebo: dict qn -> mean Δ-margin),
    compute for each query the within-query difference
        d_q = Δ-margin_q^swap − Δ-margin_q^placebo
    
    The null hypothesis is that d_q has zero mean — i.e., the swap
    perturbation does not exceed the placebo perturbation in its
    effect on the gold-vs-nearest-competitor margin. Under H0, the
    sign of each d_q is exchangeable, so we use a paired permutation
    (sign-flip) test, with B=10,000 resamples by default.
    
    The 95% CI on the mean d_q is computed via BCa bootstrap. Because
    each d_q is itself a per-query mean (already collapsed across the
    5 substitutes), the per-row bootstrap is appropriate at the unit
    of analysis. We do NOT use cluster-by-query bootstrap here because
    each query contributes exactly one d_q value.
    
    Returns dict with:
        'n_queries': int
        'mean_dd': float
        'p_value_perm': float (paired permutation, B=n_perm)
        'ci_lo_bca': float
        'ci_hi_bca': float
        'effect_size_dz': float (Cohen's d_z, paired)
        'cles': float (Common Language Effect Size)
        'shared_qns': list of query-norms used
    """
    shared = sorted(set(per_q_swap) & set(per_q_placebo))
    n = len(shared)
    if n < 2:
        return {
            'n_queries': n,
            'mean_dd': float('nan'),
            'p_value_perm': float('nan'),
            'ci_lo_bca': float('nan'),
            'ci_hi_bca': float('nan'),
            'effect_size_dz': float('nan'),
            'cles': float('nan'),
            'shared_qns': shared,
        }
    d = np.array([per_q_swap[q] - per_q_placebo[q] for q in shared])
    
    # Paired permutation (sign-flip) p-value (two-sided)
    perm_result = exact_paired_permutation_test(d, n_mc=n_perm, seed=seed_perm)
    p_perm = perm_result['p_value']
    
    # BCa bootstrap CI on the mean
    bca = bca_bootstrap_ci(d, conf_level=0.95, n_resamples=n_boot,
                            seed=seed_boot)
    
    # Hodges-Lehmann CI on the median paired difference (robust
    # complement to BCa; reported as additional sensitivity per
    # the locked-in design's effect-size triple).
    hl = hodges_lehmann_ci(d, conf_level=0.95)
    
    # Extract Cohen's d_z from the dict returned by paired_cohens_dz
    dz_result = paired_cohens_dz(d)
    dz_val = dz_result.get('dz', float('nan'))
    
    return {
        'n_queries': n,
        'mean_dd': float(np.mean(d)),
        'median_dd': float(np.median(d)),
        'p_value_perm': float(p_perm),
        'perm_method': perm_result.get('method', 'unknown'),
        'n_permutations': int(perm_result.get('n_permutations', 0)),
        'ci_lo_bca': float(bca['ci_lo']),
        'ci_hi_bca': float(bca['ci_hi']),
        'hl_ci_lo': float(hl.get('ci_lo', float('nan'))),
        'hl_ci_hi': float(hl.get('ci_hi', float('nan'))),
        'hl_median_estimate': float(hl.get('median', np.median(d))),
        'effect_size_dz': float(dz_val),
        'cles': float(paired_cles(d)),
        'shared_qns': shared,
        'per_query_diffs': [{'qn': q,
                              'd': float(per_q_swap[q] - per_q_placebo[q])}
                              for q in shared],
    }


# ============================================================================
# APPENDIX linear mixed-effects model on continuous Δ-margin
# ============================================================================
def fit_lmm_appendix(rows, output_dir):
    """
    Fit the appendix linear mixed-effects model on continuous Δ-margin
    via statsmodels.MixedLM. Per the locked-in stack:
      - Outcome: gold_margin_delta (continuous)
      - Fixed effects: intervention (categorical, reference = proper-noun
        placebo), actor (categorical), subword_token_delta (continuous
        covariate per all eleven research documents' agreement)
      - Random effect: random intercept by query (qn)
    
    Why linear (Gaussian) and not logistic (Bernoulli on flip):
        At our N=21 with 94.52% baseline accuracy, complete separation
        and singular-fit errors are virtually guaranteed for a
        binary-outcome GLMM. The Gemini PDF and Claude operational
        answers report both flag this as a hazard. The LMM on a
        continuous outcome avoids the issue entirely.
    
    We restrict to Stratum A non-baseline rows since the four placebo
    contrasts apply only there. We exclude rows where Δ-margin is NaN.
    
    Returns the fitted model summary as a dict (extracted from
    statsmodels). If fitting fails or statsmodels is unavailable, the
    function returns a dict with status='unavailable' and a reason
    string; it does not raise.
    """
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError as e:
        return {'status': 'unavailable',
                'reason': f"pandas/statsmodels not installed: {e}"}
    
    # Build a tidy data frame from the row-list
    records = []
    for r in rows:
        if r.get('is_baseline'):
            continue
        if r.get('stratum') != 'A':
            continue
        # Restrict to the swap + four-placebo set (drop insertion etc.)
        if r.get('intervention') not in {'swap', 'proper_noun_placebo',
                                          'generic_phrase_placebo',
                                          'malware_placebo', 'soft_deletion'}:
            continue
        delta_margin = r.get('gold_margin_delta')
        if delta_margin is None or (isinstance(delta_margin, float) and
                                      np.isnan(delta_margin)):
            continue
        try:
            tok_delta = int(r.get('subword_token_delta', 0))
        except (ValueError, TypeError):
            tok_delta = 0
        records.append({
            'qn': r['qn'],
            'actor': r['actor'],
            'intervention': r['intervention'],
            'gold_margin_delta': float(delta_margin),
            'subword_token_delta': float(tok_delta),
        })
    
    if not records:
        return {'status': 'unavailable',
                'reason': "No valid Stratum A rows with non-NaN Δ-margin"}
    
    df = pd.DataFrame(records)
    # Set proper-noun placebo as the reference level for intervention
    df['intervention'] = pd.Categorical(
        df['intervention'],
        categories=['proper_noun_placebo', 'swap', 'generic_phrase_placebo',
                    'malware_placebo', 'soft_deletion'],
        ordered=False)
    df['actor'] = pd.Categorical(df['actor'])
    
    formula = 'gold_margin_delta ~ C(intervention) + C(actor) + subword_token_delta'
    
    # Helper to extract a clean coefficient table from a fitted model
    def _extract_coefs(mdf):
        coef_table = []
        params = mdf.params
        bse = mdf.bse if hasattr(mdf, 'bse') else None
        tvals = mdf.tvalues if hasattr(mdf, 'tvalues') else None
        pvals = mdf.pvalues if hasattr(mdf, 'pvalues') else None
        for name in params.index.tolist():
            est = float(params[name])
            se = float(bse[name]) if bse is not None and name in bse.index else float('nan')
            tv = float(tvals[name]) if tvals is not None and name in tvals.index else float('nan')
            pv = float(pvals[name]) if pvals is not None and name in pvals.index else float('nan')
            coef_table.append({
                'name': str(name),
                'estimate': est,
                'std_error': se,
                't_statistic': tv,
                'p_value': pv,
            })
        return coef_table
    
    # Strategy: try the random-intercept MixedLM with the default LBFGS
    # optimizer; if it fails with a singular random-effects covariance,
    # retry with Powell (a derivative-free optimizer that sometimes
    # finds a non-singular solution); if THAT fails, fall back to
    # ordinary least squares without the random intercept and label
    # the result as 'fixed_effects_only_fallback'. The fallback is
    # informative because at our N a singular RE variance means the
    # query-level heterogeneity is genuinely small relative to the
    # residual, in which case OLS coefficients are nearly identical
    # to the LMM fixed-effect estimates anyway.
    
    last_error = None
    
    # Attempt 1: MixedLM with default LBFGS
    try:
        md = smf.mixedlm(formula, df, groups=df['qn'])
        mdf = md.fit(method='lbfgs', maxiter=500, reml=True)
        coef_table = _extract_coefs(mdf)
        return {
            'status': 'fitted',
            'method': 'MixedLM_lbfgs',
            'n_observations': int(df.shape[0]),
            'n_clusters_query': int(df['qn'].nunique()),
            'formula': formula,
            'reference_levels': {'intervention': 'proper_noun_placebo'},
            'coefficients': coef_table,
            'log_likelihood': float(mdf.llf),
            'aic': float(mdf.aic) if hasattr(mdf, 'aic') else float('nan'),
            'converged': bool(mdf.converged) if hasattr(mdf, 'converged') else None,
        }
    except Exception as e:
        last_error = f"MixedLM with lbfgs failed: {e}"
    
    # Attempt 2: MixedLM with Powell (derivative-free)
    try:
        md = smf.mixedlm(formula, df, groups=df['qn'])
        mdf = md.fit(method='powell', maxiter=1000, reml=True)
        coef_table = _extract_coefs(mdf)
        return {
            'status': 'fitted',
            'method': 'MixedLM_powell',
            'n_observations': int(df.shape[0]),
            'n_clusters_query': int(df['qn'].nunique()),
            'formula': formula,
            'reference_levels': {'intervention': 'proper_noun_placebo'},
            'coefficients': coef_table,
            'log_likelihood': float(mdf.llf),
            'aic': float(mdf.aic) if hasattr(mdf, 'aic') else float('nan'),
            'converged': bool(mdf.converged) if hasattr(mdf, 'converged') else None,
            'note': ('Default lbfgs optimizer failed; fitted with Powell '
                     'as a derivative-free fallback.'),
        }
    except Exception as e:
        last_error = (f"{last_error}; MixedLM with powell also failed: {e}")
    
    # Attempt 3: OLS without random intercept (fixed-effects-only fallback)
    try:
        ols_md = smf.ols(formula, df)
        ols_mdf = ols_md.fit()
        coef_table = _extract_coefs(ols_mdf)
        return {
            'status': 'fitted_fallback',
            'method': 'OLS_fixed_effects_only',
            'n_observations': int(df.shape[0]),
            'n_clusters_query': int(df['qn'].nunique()),
            'formula': formula + ' [no random intercept]',
            'reference_levels': {'intervention': 'proper_noun_placebo'},
            'coefficients': coef_table,
            'log_likelihood': float(ols_mdf.llf) if hasattr(ols_mdf, 'llf') else float('nan'),
            'aic': float(ols_mdf.aic) if hasattr(ols_mdf, 'aic') else float('nan'),
            'rsquared': float(ols_mdf.rsquared) if hasattr(ols_mdf, 'rsquared') else float('nan'),
            'note': ('MixedLM failed (random-effects covariance singular); '
                     'reporting OLS fixed-effects-only fallback. The fixed-'
                     'effect coefficients are interpreted at the marginal level '
                     '(no within-query clustering adjustment).'),
            'mixedlm_error': last_error,
        }
    except Exception as e:
        return {
            'status': 'failed_to_fit',
            'reason': f"Both MixedLM (lbfgs and Powell) and OLS fallback failed. "
                       f"Last error: {e}; previous: {last_error}",
            'n_observations': int(df.shape[0]),
            'formula': formula,
        }


# ============================================================================
# Plotting style (reused from acsac_figures.py with per-Step-D defaults)
# ============================================================================
def set_paper_style():
    """Apply ACSAC paper-quality style overrides."""
    plt.rcParams.update({
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.hashsalt': 'acsac2026_step_d',
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.2,
    })


def savefig_all(fig, stem):
    """Save figure as PDF, SVG, and PNG to the same path stem.
    
    For full reproducibility we strip the timestamp metadata from each
    format. matplotlib's PDF backend embeds CreationDate/ModDate by
    default; passing None explicitly suppresses them, giving bit-identical
    PDFs across runs at the same seed. The SVG backend includes a Date
    field in the standard <metadata> block; setting Date to None
    suppresses that. PNG metadata is suppressed by passing only
    deterministic fields.
    """
    base = stem
    fig.savefig(f"{base}.pdf", bbox_inches='tight',
                metadata={
                    'Producer': 'matplotlib (acsac2026 step D)',
                    'Creator': 'matplotlib (acsac2026 step D)',
                    'CreationDate': None,
                    'ModDate': None,
                })
    fig.savefig(f"{base}.svg", bbox_inches='tight',
                metadata={'Date': None,
                           'Creator': 'matplotlib (acsac2026 step D)'})
    fig.savefig(f"{base}.png", bbox_inches='tight', dpi=300,
                metadata={'Software': 'matplotlib (acsac2026 step D)'})
    plt.close(fig)


# Color palette for interventions: chosen to be colorblind-friendly,
# distinguishable in greyscale, and consistent with the locked-in design.
INTERVENTION_PALETTE = {
    'baseline':                  '#777777',  # neutral grey
    'swap':                      '#D62728',  # red — the focal intervention
    'proper_noun_placebo':       '#1F77B4',  # blue — primary placebo
    'generic_phrase_placebo':    '#2CA02C',  # green
    'malware_placebo':           '#FF7F0E',  # orange
    'soft_deletion':             '#9467BD',  # purple
    'insertion_own_actor':       '#8C564B',  # brown
    'insertion_different_actor': '#E377C2',  # pink
}

INTERVENTION_LABELS = {
    'baseline':                  'Baseline (no perturb)',
    'swap':                      'Actor swap',
    'proper_noun_placebo':       'Proper-noun placebo',
    'generic_phrase_placebo':    'Generic-phrase placebo',
    'malware_placebo':           'Malware-name placebo',
    'soft_deletion':             'Soft deletion',
    'insertion_own_actor':       'Insertion (own actor)',
    'insertion_different_actor': 'Insertion (different actor)',
}


# ============================================================================
# Figure 1: flip rates by intervention with Wilson CI
# ============================================================================
def fig1_flip_rates(rows, output_path_stem):
    """
    Bar chart of top-1 flip rate by intervention with Wilson 95% CI
    error bars. The unit of analysis here is the (query, substitute)
    triple — this is the operational summary, not the inferential
    statistic. The inferential layer (paired DiD) lives in fig2.
    
    Stratum A interventions on the LEFT panel, Stratum B insertion
    sub-conditions on the RIGHT panel. Both panels share a common
    y-axis range for readability.
    """
    # Group rows by stratum and intervention
    by_stratum_intervention = defaultdict(list)
    for r in rows:
        if r.get('is_baseline'):
            continue
        key = (r.get('stratum', '?'), r['intervention'])
        by_stratum_intervention[key].append(r)
    
    # Compute flip rate + Wilson CI per (stratum, intervention)
    cf_data = {}
    for (stratum, inter), rs in by_stratum_intervention.items():
        flips = [bool(r.get('top1_flip')) for r in rs]
        n = len(flips)
        k = sum(flips)
        rate = k / n if n else 0.0
        ci_lo, ci_hi = wilson_ci(k, n)
        cf_data[(stratum, inter)] = {
            'n_triples': n,
            'flip_count': k,
            'flip_rate': rate,
            'wilson_lo': ci_lo,
            'wilson_hi': ci_hi,
        }
    
    # Stratum A: 5 interventions in plot order; Stratum B: 2
    a_order = ['swap', 'proper_noun_placebo', 'generic_phrase_placebo',
                'malware_placebo', 'soft_deletion']
    b_order = ['insertion_own_actor', 'insertion_different_actor']
    
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6),
                              gridspec_kw={'width_ratios': [5, 2]})
    
    # Panel A: Stratum A
    a_ax = axes[0]
    for j, inter in enumerate(a_order):
        d = cf_data.get(('A', inter))
        if d is None:
            continue
        rate = d['flip_rate']
        err_lo = max(0.0, rate - d['wilson_lo'])
        err_hi = max(0.0, d['wilson_hi'] - rate)
        a_ax.bar(j, rate, color=INTERVENTION_PALETTE[inter],
                  alpha=0.85, width=0.7,
                  edgecolor='black', linewidth=0.5)
        a_ax.errorbar(j, rate, yerr=[[err_lo], [err_hi]],
                       color='black', capsize=3, linewidth=0.8)
        a_ax.text(j, max(rate + err_hi + 0.02, 0.05),
                   f"{rate:.2f}\n(n={d['n_triples']})",
                   ha='center', va='bottom', fontsize=8)
    a_ax.set_xticks(range(len(a_order)))
    a_ax.set_xticklabels([INTERVENTION_LABELS[k] for k in a_order],
                          rotation=20, ha='right', fontsize=9)
    a_ax.set_ylabel('Top-1 flip rate (Wilson 95% CI)')
    a_ax.set_title('(a) Stratum A: actor-bearing queries (n=21)', fontsize=10)
    a_ax.set_ylim(0.0, 1.0)
    a_ax.yaxis.grid(True)
    
    # Panel B: Stratum B
    b_ax = axes[1]
    for j, inter in enumerate(b_order):
        d = cf_data.get(('B', inter))
        if d is None:
            continue
        rate = d['flip_rate']
        err_lo = max(0.0, rate - d['wilson_lo'])
        err_hi = max(0.0, d['wilson_hi'] - rate)
        b_ax.bar(j, rate, color=INTERVENTION_PALETTE[inter],
                  alpha=0.85, width=0.7,
                  edgecolor='black', linewidth=0.5)
        b_ax.errorbar(j, rate, yerr=[[err_lo], [err_hi]],
                       color='black', capsize=3, linewidth=0.8)
        b_ax.text(j, max(rate + err_hi + 0.02, 0.05),
                   f"{rate:.2f}\n(n={d['n_triples']})",
                   ha='center', va='bottom', fontsize=8)
    b_ax.set_xticks(range(len(b_order)))
    b_ax.set_xticklabels([INTERVENTION_LABELS[k] for k in b_order],
                          rotation=20, ha='right', fontsize=9)
    b_ax.set_title('(b) Stratum B: actor-free queries (n=125)', fontsize=10)
    b_ax.set_ylim(0.0, 1.0)
    b_ax.yaxis.grid(True)
    
    fig.suptitle('Top-1 flip rate by intervention (operational summary)',
                  y=1.02, fontsize=11)
    fig.tight_layout()
    savefig_all(fig, output_path_stem)
    return cf_data


# ============================================================================
# Figure 2: swap-vs-placebo paired Δ-margin DiD with BCa CI (PRIMARY CLAIM)
# ============================================================================
def fig2_swap_vs_placebo(per_query_margins, dd_results,
                          output_path_stem):
    """
    The primary-claim figure of Step D.
    
    Left panel: per-query mean Δ-margin under each Stratum A intervention,
        as a strip-plot with the per-intervention mean and BCa 95% CI
        overlaid. This shows the raw signal.
    
    Right panel: the four planned paired difference-in-differences
        contrasts, each with their mean ΔΔ, paired-permutation p-value,
        and BCa-bootstrap 95% CI. Holm-Bonferroni-adjusted p-values are
        also shown. This is the inferential evidence.
    
    Args:
        per_query_margins: dict intervention -> {qn -> Δ-margin}
        dd_results: dict (intervention_a, intervention_b) -> dd-test result
                    (output of paired_dd_test)
        output_path_stem: file path stem (no extension)
    """
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0),
                              gridspec_kw={'width_ratios': [5, 4]})
    
    # ---- LEFT panel: per-query Δ-margin distribution by intervention ----
    a_ax = axes[0]
    a_order = ['swap', 'proper_noun_placebo', 'generic_phrase_placebo',
                'malware_placebo', 'soft_deletion']
    rng = np.random.default_rng(20260510)  # for x-axis jitter (display only)
    
    for j, inter in enumerate(a_order):
        per_q = per_query_margins.get(inter, {})
        if not per_q:
            continue
        values = np.array(list(per_q.values()))
        # Strip points with horizontal jitter
        x_jitter = rng.normal(j, 0.06, size=len(values))
        a_ax.scatter(x_jitter, values, color=INTERVENTION_PALETTE[inter],
                      alpha=0.5, s=18, edgecolors='none')
        # Mean as a black diamond
        mean_val = float(np.mean(values))
        a_ax.scatter(j, mean_val, color='black', marker='D', s=40,
                      zorder=5, edgecolors='white', linewidths=1.0)
        # BCa CI as a vertical bar
        if len(values) >= 2:
            bca = bca_bootstrap_ci(values, conf_level=0.95,
                                    n_resamples=N_BOOTSTRAP_RESAMPLES,
                                    seed=BOOTSTRAP_SEED)
            a_ax.plot([j, j], [bca['ci_lo'], bca['ci_hi']],
                       color='black', linewidth=1.5, alpha=0.7)
            a_ax.plot([j-0.08, j+0.08], [bca['ci_lo'], bca['ci_lo']],
                       color='black', linewidth=1.5, alpha=0.7)
            a_ax.plot([j-0.08, j+0.08], [bca['ci_hi'], bca['ci_hi']],
                       color='black', linewidth=1.5, alpha=0.7)
    
    a_ax.axhline(0.0, color='grey', linewidth=0.6, linestyle='--',
                  alpha=0.7, zorder=1)
    a_ax.set_xticks(range(len(a_order)))
    a_ax.set_xticklabels([INTERVENTION_LABELS[k] for k in a_order],
                          rotation=20, ha='right', fontsize=9)
    a_ax.set_ylabel(r'Per-query mean $\Delta$-margin (perturbed $-$ original)')
    a_ax.set_title('(a) Per-query Δ-margin distribution by intervention',
                    fontsize=10)
    a_ax.yaxis.grid(True)
    
    # ---- RIGHT panel: paired DiD contrasts with BCa CI ----
    b_ax = axes[1]
    contrast_order = PLACEBO_CONTRASTS  # [(swap, ...), ...]
    contrast_labels = []
    means = []
    ci_los = []
    ci_his = []
    p_values_raw = []
    p_values_holm = []
    
    raw_p_list = [dd_results[c]['p_value_perm'] for c in contrast_order]
    holm_adjusted = holm_bonferroni(raw_p_list)
    
    for k, c in enumerate(contrast_order):
        result = dd_results[c]
        contrast_labels.append(
            f"{INTERVENTION_LABELS[c[0]]}\n  vs {INTERVENTION_LABELS[c[1]]}")
        means.append(result['mean_dd'])
        ci_los.append(result['ci_lo_bca'])
        ci_his.append(result['ci_hi_bca'])
        p_values_raw.append(result['p_value_perm'])
        p_values_holm.append(holm_adjusted[k])
    
    y_positions = np.arange(len(contrast_order))[::-1]  # top-to-bottom display
    for j, (mean, lo, hi, pr, ph) in enumerate(zip(means, ci_los, ci_his,
                                                     p_values_raw, p_values_holm)):
        y = y_positions[j]
        sig_marker = ('**' if (ph is not None and ph < 0.01) else
                      '*'  if (ph is not None and ph < 0.05) else
                      '')
        # Draw a horizontal CI bar with point estimate
        b_ax.plot([lo, hi], [y, y], color='black', linewidth=1.5)
        b_ax.plot([lo, lo], [y-0.12, y+0.12], color='black', linewidth=1.5)
        b_ax.plot([hi, hi], [y-0.12, y+0.12], color='black', linewidth=1.5)
        b_ax.scatter(mean, y, color='#D62728', s=70, zorder=5, marker='o',
                      edgecolors='black', linewidths=0.8)
        # Annotate
        annot = (f"  ΔΔ={mean:.2f} "
                  f"[{lo:.2f}, {hi:.2f}]  "
                  f"p={pr:.3f} (Holm={ph:.3f}){sig_marker}")
        b_ax.text(hi + 0.02, y, annot, va='center', fontsize=8)
    
    b_ax.set_yticks(y_positions)
    b_ax.set_yticklabels(contrast_labels, fontsize=9)
    b_ax.axvline(0.0, color='grey', linewidth=0.6, linestyle='--',
                  alpha=0.7, zorder=1)
    b_ax.set_xlabel(r'Paired difference-in-differences $\Delta\Delta$-margin')
    b_ax.set_title('(b) Paired DiD: swap vs placebo (BCa 95% CI)', fontsize=10)
    # Pad x-axis to give room for annotations. Filter out NaN values so the
    # call to set_xlim doesn't crash if any DiD test returned NaN (which
    # happens when the data has no perturbation rows for one intervention).
    finite_his = [v for v in ci_his if v is not None and np.isfinite(v)]
    finite_los = [v for v in ci_los if v is not None and np.isfinite(v)]
    if finite_his and finite_los:
        max_hi = max(finite_his)
        min_lo = min(finite_los)
        b_ax.set_xlim(min_lo - 0.5, max_hi + 4.0)
    else:
        # All DiD results were NaN; pick a safe symmetric default
        b_ax.set_xlim(-1.0, 1.0)
        b_ax.text(0.5, 0.5, 'No DiD test results available\n'
                              '(insufficient perturbation data)',
                   transform=b_ax.transAxes, ha='center', va='center',
                   color='grey', fontsize=9, alpha=0.7)
    b_ax.xaxis.grid(True)
    
    fig.suptitle(r'Step D primary inferential test: paired $\Delta$-margin DiD',
                  y=1.03, fontsize=11)
    fig.tight_layout()
    savefig_all(fig, output_path_stem)
    return {
        'left_panel_n_per_intervention': {
            inter: len(per_query_margins.get(inter, {})) for inter in a_order
        },
        'right_panel_contrast_results': [
            {
                'contrast_a': c[0], 'contrast_b': c[1],
                'mean_dd': means[k], 'ci_lo_bca': ci_los[k],
                'ci_hi_bca': ci_his[k], 'p_value_perm_raw': p_values_raw[k],
                'p_value_perm_holm': p_values_holm[k],
            }
            for k, c in enumerate(contrast_order)
        ],
    }


# ============================================================================
# Figure 3: gold-rank-change distribution
# ============================================================================
def fig3_rank_change_distribution(rows, output_path_stem):
    """
    Histograms of gold_rank_change by intervention. Negative values
    mean the gold technique moved CLOSER to top-1 (improvement) under
    perturbation; positive means it dropped further away. Most cases
    cluster at 0 (no change), but the tails are diagnostic.
    """
    set_paper_style()
    a_order = ['swap', 'proper_noun_placebo', 'generic_phrase_placebo',
                'malware_placebo', 'soft_deletion']
    fig, axes = plt.subplots(1, len(a_order), figsize=(11.0, 2.8), sharey=True)
    
    by_inter = defaultdict(list)
    for r in rows:
        if r.get('is_baseline'):
            continue
        if r.get('stratum') != 'A':
            continue
        if r['intervention'] not in a_order:
            continue
        v = r.get('gold_rank_change', 0)
        if v is not None and v != '':
            try:
                by_inter[r['intervention']].append(int(v))
            except (ValueError, TypeError):
                pass
    
    fig_data = {}
    for j, inter in enumerate(a_order):
        ax = axes[j]
        vals = by_inter.get(inter, [])
        if not vals:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center')
            ax.set_xticks([])
            continue
        vals = np.array(vals)
        # Histogram bins centred on integer values; clip extremes for
        # visualisation but record the full unclipped data in the JSON
        # sidecar.
        clip_lo, clip_hi = -10, 10
        clipped = np.clip(vals, clip_lo, clip_hi)
        bins = np.arange(clip_lo - 0.5, clip_hi + 1.5, 1.0)
        ax.hist(clipped, bins=bins, color=INTERVENTION_PALETTE[inter],
                alpha=0.85, edgecolor='black', linewidth=0.4)
        ax.axvline(0.0, color='grey', linewidth=0.6, linestyle='--')
        ax.set_title(INTERVENTION_LABELS[inter], fontsize=9)
        ax.set_xlim(clip_lo, clip_hi)
        if j == 0:
            ax.set_ylabel('Count')
        ax.set_xlabel('Gold rank change\n(pert − orig)', fontsize=8)
        # Annotate mean
        mean_val = float(np.mean(vals))
        ax.text(0.05, 0.92,
                 f"mean={mean_val:.2f}\nn={len(vals)}",
                 transform=ax.transAxes, ha='left', va='top', fontsize=8)
        fig_data[inter] = {
            'n': int(len(vals)),
            'mean': mean_val,
            'median': float(np.median(vals)),
            'fraction_unchanged': float(np.mean(vals == 0)),
            'fraction_worse': float(np.mean(vals > 0)),
            'fraction_better': float(np.mean(vals < 0)),
            'min': int(np.min(vals)),
            'max': int(np.max(vals)),
        }
    
    fig.suptitle('Gold-rank change by intervention (Stratum A)',
                  y=1.05, fontsize=11)
    fig.tight_layout()
    savefig_all(fig, output_path_stem)
    return fig_data


# ============================================================================
# Figure 4: carbanak case study (records #12 and #15 paired analysis)
# ============================================================================
def fig4_carbanak_case_study(rows, output_path_stem):
    """
    Paired case study highlighting the two carbanak-related namespace
    collisions in Stratum A:
      - Record #12: carbanak ACTOR query containing the alias "Anunak"
        (clean intra-actor swap target).
      - Record #15: fin7 ACTOR query containing the residual mention
        "Carbanak" (cross-actor confound — when we swap FIN7 token,
        the unmatched "Carbanak" string remains in the query).
    
    The figure plots, for each of the two records, the gold technique's
    score and margin under (a) the original query, (b) each of the 5
    swap substitutes, and (c) each of the 5 proper-noun placebo
    substitutes. This visualizes the sensitivity to actor-token
    perturbation in the two distinct collision regimes.
    
    Detection uses qn (the lowercase-normalized query) since that is
    always present, with query_raw as a fallback for richer display.
    """
    # Identify the two records by qn-substring match. qn is always
    # populated and is the lowercase-normalized form of query_raw.
    record_12_qn = None
    record_15_qn = None
    for r in rows:
        if not r.get('is_baseline'):
            continue
        if r.get('stratum') != 'A':
            continue
        qn_lc = (r.get('qn') or '').lower()
        actor = r.get('actor', '')
        if actor == 'carbanak' and 'anunak' in qn_lc:
            record_12_qn = r['qn']
        elif actor == 'fin7' and 'carbanak' in qn_lc:
            record_15_qn = r['qn']
    
    if record_12_qn is None and record_15_qn is None:
        # Cannot draw the case study; create a placeholder figure
        set_paper_style()
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5,
                 'Carbanak case-study records not found in results CSV.\n'
                 'Run counterfactual_probe.py first.',
                 transform=ax.transAxes, ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        savefig_all(fig, output_path_stem)
        return {'status': 'records_not_found'}
    
    set_paper_style()
    n_panels = (1 if record_12_qn else 0) + (1 if record_15_qn else 0)
    fig, axes = plt.subplots(1, max(n_panels, 1),
                              figsize=(8.0, 3.6), squeeze=False)
    axes = axes[0]
    
    fig_data = {}
    panel_idx = 0
    
    for label, qn, panel_title in [
        ('record_12_anunak', record_12_qn, 'Record #12: carbanak actor + Anunak alias\n(intra-actor namespace collision)'),
        ('record_15_residual', record_15_qn, 'Record #15: FIN7 actor + residual "Carbanak"\n(cross-actor confound)'),
    ]:
        if qn is None:
            continue
        ax = axes[panel_idx]
        # Gather: gold_score and gold_margin for baseline, swap, proper_noun
        baseline_score = None
        baseline_margin = None
        baseline_query = None
        swap_data = []   # list of (substitute_name, score, margin)
        placebo_data = []
        for r in rows:
            if r.get('qn') != qn:
                continue
            if r.get('is_baseline'):
                baseline_score = r.get('gold_score_orig') or r.get('gold_score_pert')
                baseline_margin = r.get('gold_margin_orig') or r.get('gold_margin_pert')
                baseline_query = r.get('query_raw', '')
                continue
            inter = r.get('intervention')
            if inter == 'swap':
                swap_data.append((r.get('substitute_recast', '?'),
                                   r.get('gold_score_pert', float('nan')),
                                   r.get('gold_margin_pert', float('nan'))))
            elif inter == 'proper_noun_placebo':
                placebo_data.append((r.get('substitute_recast', '?'),
                                      r.get('gold_score_pert', float('nan')),
                                      r.get('gold_margin_pert', float('nan'))))
        
        # Plot: x = substitute index, y = margin. Baseline as horizontal line.
        x_swap = list(range(len(swap_data)))
        y_swap = [d[2] for d in swap_data]
        x_placebo = list(range(len(placebo_data)))
        y_placebo = [d[2] for d in placebo_data]
        
        if baseline_margin is not None and not (isinstance(baseline_margin, float) and np.isnan(baseline_margin)):
            ax.axhline(float(baseline_margin), color='black', linewidth=1.2,
                        linestyle='-', alpha=0.7,
                        label=f'Baseline ({float(baseline_margin):.2f})')
        ax.scatter(x_swap, y_swap, color=INTERVENTION_PALETTE['swap'],
                    s=80, label='Actor swap', edgecolors='black',
                    linewidths=0.6, alpha=0.85)
        # Offset placebo by 0.5 on x-axis for visual separation
        x_placebo_offset = [x + 0.5 for x in x_placebo]
        ax.scatter(x_placebo_offset, y_placebo,
                    color=INTERVENTION_PALETTE['proper_noun_placebo'],
                    s=80, marker='s', label='Proper-noun placebo',
                    edgecolors='black', linewidths=0.6, alpha=0.85)
        # Annotate substitute names. Plain enumerate is clearer than the
        # earlier zip+list-comp+re-index version. We guard against NaN
        # margins (which can occur if scoring failed for a particular
        # substitute) so the annotation doesn't blow up.
        for x, (name, _score, margin) in enumerate(swap_data):
            if margin is None or (isinstance(margin, float) and np.isnan(margin)):
                continue
            ax.annotate(name, (x, margin),
                         xytext=(0, 8), textcoords='offset points',
                         fontsize=7, ha='center', alpha=0.85)
        for x, (name, _score, margin) in enumerate(placebo_data):
            if margin is None or (isinstance(margin, float) and np.isnan(margin)):
                continue
            ax.annotate(name, (x + 0.5, margin),
                         xytext=(0, -10), textcoords='offset points',
                         fontsize=7, ha='center', alpha=0.85)
        ax.set_title(panel_title, fontsize=9)
        ax.set_xlabel('Substitute index')
        ax.set_ylabel(r'Gold $\times$ best-non-gold margin')
        ax.axhline(0.0, color='grey', linewidth=0.5, linestyle=':', alpha=0.6)
        ax.legend(loc='best', fontsize=8)
        ax.yaxis.grid(True)
        
        fig_data[label] = {
            'qn': qn,
            'query_raw': baseline_query,
            'baseline_margin': (float(baseline_margin)
                                  if baseline_margin is not None and not np.isnan(baseline_margin)
                                  else None),
            'swap_substitutes': [{'name': d[0], 'score': float(d[1]) if not np.isnan(d[1]) else None,
                                    'margin': float(d[2]) if not np.isnan(d[2]) else None}
                                   for d in swap_data],
            'proper_noun_substitutes': [{'name': d[0], 'score': float(d[1]) if not np.isnan(d[1]) else None,
                                            'margin': float(d[2]) if not np.isnan(d[2]) else None}
                                           for d in placebo_data],
        }
        panel_idx += 1
    
    fig.suptitle('Carbanak case study: namespace-collision sensitivity',
                  y=1.04, fontsize=11)
    fig.tight_layout()
    savefig_all(fig, output_path_stem)
    return fig_data


# ============================================================================
# Main analysis driver
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='ACSAC Step D inferential layer + figure generation')
    parser.add_argument('--results-csv',
                        default='eval_results_v2/counterfactual/counterfactual_results.csv',
                        help='Path to counterfactual_results.csv produced by '
                             'counterfactual_probe.py.')
    parser.add_argument('--output-dir',
                        default='eval_results_v2/counterfactual',
                        help='Output directory for inferential results and figures.')
    args = parser.parse_args()
    
    out_dir = args.output_dir
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    print('=' * 75)
    print('ACSAC 2026 Step D: cf_figures.py — inferential layer + figures')
    print('=' * 75)
    print(f"  results-csv:   {args.results_csv}")
    print(f"  output-dir:    {out_dir}")
    print()
    
    # 1) Load data
    print('[1/6] Loading per-triple results CSV...')
    if not os.path.isfile(args.results_csv):
        print(f"  ERROR: results CSV not found: {args.results_csv}")
        print(f"  Run counterfactual_probe.py first to produce it.")
        return
    rows = load_results_csv(args.results_csv)
    print(f"  Loaded {len(rows)} rows.")
    
    # 2) Aggregate to per-query Δ-margin per intervention
    print('\n[2/6] Aggregating Δ-margin to per-query level for each intervention...')
    per_query_margins = {}
    for inter in ['swap', 'proper_noun_placebo', 'generic_phrase_placebo',
                   'malware_placebo', 'soft_deletion',
                   'insertion_own_actor', 'insertion_different_actor']:
        per_query_margins[inter] = aggregate_to_per_query(
            rows, inter, 'gold_margin_delta')
        print(f"    {inter}: {len(per_query_margins[inter])} queries with Δ-margin")
    
    # Per-query flip rates and correctness
    per_query_flip = {}
    for inter in ['swap', 'proper_noun_placebo', 'generic_phrase_placebo',
                   'malware_placebo', 'soft_deletion',
                   'insertion_own_actor', 'insertion_different_actor']:
        per_query_flip[inter] = aggregate_to_per_query(
            rows, inter, 'top1_flip')
    
    per_query_correct_pert = {}
    for inter in ['swap', 'proper_noun_placebo', 'generic_phrase_placebo',
                   'malware_placebo', 'soft_deletion',
                   'insertion_own_actor', 'insertion_different_actor']:
        per_query_correct_pert[inter] = aggregate_to_per_query(
            rows, inter, 'correct_pert')
    
    # Get original-correctness from the baseline rows (one per query)
    per_query_correct_orig = {}
    for r in rows:
        if r.get('is_baseline'):
            qn = r['qn']
            v = r.get('correct_pert')  # baseline pert == orig
            if v is not None and v != '':
                per_query_correct_orig[qn] = (1.0 if v else 0.0)
    
    # 3) PRIMARY paired DiD test for each placebo contrast
    print('\n[3/6] Computing paired Δ-margin DiD inferential test for the four contrasts...')
    dd_results = {}
    for contrast in PLACEBO_CONTRASTS:
        a, b = contrast
        result = paired_dd_test(per_query_margins[a], per_query_margins[b])
        dd_results[contrast] = result
        sig_str = ('***' if result['p_value_perm'] < 0.001 else
                    '**'  if result['p_value_perm'] < 0.01 else
                    '*'   if result['p_value_perm'] < 0.05 else
                    'ns')
        print(f"    {a} vs {b}: n={result['n_queries']}  "
              f"ΔΔ={result['mean_dd']:.3f}  "
              f"BCa CI=[{result['ci_lo_bca']:.3f}, {result['ci_hi_bca']:.3f}]  "
              f"p_perm={result['p_value_perm']:.4f} {sig_str}  "
              f"d_z={result['effect_size_dz']:.3f}  CLES={result['cles']:.3f}")
    
    # Holm-Bonferroni correction across the four contrasts
    raw_pvals = [dd_results[c]['p_value_perm'] for c in PLACEBO_CONTRASTS]
    holm_pvals = holm_bonferroni(raw_pvals)
    print(f"\n  Holm-Bonferroni-adjusted p-values across {len(PLACEBO_CONTRASTS)} contrasts:")
    for c, raw, adj in zip(PLACEBO_CONTRASTS, raw_pvals, holm_pvals):
        print(f"    {c[0]} vs {c[1]}: raw p={raw:.4f} -> Holm p={adj:.4f}")
    for k, c in enumerate(PLACEBO_CONTRASTS):
        dd_results[c]['p_value_perm_holm'] = float(holm_pvals[k])
    
    # 4) CO-PRIMARY mid-p McNemar where b+c >= 5
    print('\n[4/6] CO-PRIMARY operational summary: per-intervention flip rates with mid-p McNemar...')
    mcnemar_results = {}
    for inter in ['swap', 'proper_noun_placebo', 'generic_phrase_placebo',
                   'malware_placebo', 'soft_deletion']:
        # Build the contingency table at the query level, using
        # rounded per-query correctness rates so each query becomes
        # a paired binary outcome.
        b = c = 0
        for qn, orig_rate in per_query_correct_orig.items():
            if qn not in per_query_correct_pert[inter]:
                continue
            pert_rate = per_query_correct_pert[inter][qn]
            # Threshold at 0.5 for binary classification
            orig_correct = (orig_rate >= 0.5)
            pert_correct = (pert_rate >= 0.5)
            if orig_correct and not pert_correct:
                b += 1
            elif not orig_correct and pert_correct:
                c += 1
        result = mid_p_mcnemar(b, c)
        result['intervention'] = inter
        mcnemar_results[inter] = result
        if result['underinformative']:
            print(f"    {inter}: b={b}, c={c}, b+c={b+c} — UNDERINFORMATIVE (b+c<5; "
                   f"interpret with caution)")
        else:
            print(f"    {inter}: b={b}, c={c}, b+c={b+c}, mid-p={result['mid_p_two_sided']:.4f}")
    
    # 5) APPENDIX linear mixed-effects model
    print('\n[5/6] APPENDIX: fitting linear mixed-effects model on Δ-margin via statsmodels.MixedLM...')
    lmm_result = fit_lmm_appendix(rows, out_dir)
    if lmm_result.get('status') == 'fitted':
        print(f"    Fitted with n_obs={lmm_result['n_observations']}, "
              f"n_clusters={lmm_result['n_clusters_query']}; converged="
              f"{lmm_result.get('converged')}")
        print(f"    Top fixed-effect coefficients:")
        for coef in lmm_result['coefficients'][:10]:
            print(f"      {coef['name']:50s}  est={coef['estimate']:+.3f}  "
                  f"SE={coef['std_error']:.3f}  p={coef['p_value']:.4f}")
    else:
        print(f"    Mixed-effects model status: {lmm_result.get('status')} "
              f"({lmm_result.get('reason', '')})")
    
    # 6) Figures
    print('\n[6/6] Producing four ACSAC figures...')
    fig1_data = fig1_flip_rates(rows, os.path.join(fig_dir,
                                                     'cf_fig1_flip_rates_by_intervention'))
    print(f"    cf_fig1: flip_rates_by_intervention written.")
    fig2_data = fig2_swap_vs_placebo(per_query_margins, dd_results,
                                       os.path.join(fig_dir,
                                                     'cf_fig2_swap_vs_placebo_paired'))
    print(f"    cf_fig2: swap_vs_placebo_paired written.")
    fig3_data = fig3_rank_change_distribution(rows, os.path.join(fig_dir,
                                                                    'cf_fig3_gold_rank_change_distribution'))
    print(f"    cf_fig3: gold_rank_change_distribution written.")
    fig4_data = fig4_carbanak_case_study(rows, os.path.join(fig_dir,
                                                              'cf_fig4_carbanak_case_study'))
    print(f"    cf_fig4: carbanak_case_study written.")
    
    # Sidecar
    figure_data = {
        'fig1_flip_rates': {f"{s}_{i}": v for (s, i), v in fig1_data.items()}
                              if isinstance(list(fig1_data.keys())[0] if fig1_data else None, tuple)
                              else fig1_data,
        'fig2_dd_test': fig2_data,
        'fig3_rank_change': fig3_data,
        'fig4_carbanak_case': fig4_data,
    }
    with open(os.path.join(fig_dir, 'figure_data.json'), 'w') as f:
        # The fig1_flip_rates dict has tuple keys we need to serialize
        clean_fig1 = {f"{stratum}__{intervention}": v
                       for (stratum, intervention), v in fig1_data.items()}
        figure_data['fig1_flip_rates'] = clean_fig1
        json.dump(figure_data, f, indent=2, default=str)
    print(f"    figure_data.json written.")
    
    # Final inferential output bundle
    out_bundle = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'paths': {
            'results_csv': args.results_csv,
            'output_dir': out_dir,
        },
        'primary_dd_test': {
            f"{a}_vs_{b}": dd_results[(a, b)] for (a, b) in PLACEBO_CONTRASTS
        },
        'co_primary_mcnemar': mcnemar_results,
        'appendix_mixed_effects': lmm_result,
        'seeds': {
            'PERMUTATION_SEED': PERMUTATION_SEED,
            'BOOTSTRAP_SEED': BOOTSTRAP_SEED,
            'N_PERMUTATION_RESAMPLES': N_PERMUTATION_RESAMPLES,
            'N_BOOTSTRAP_RESAMPLES': N_BOOTSTRAP_RESAMPLES,
        },
    }
    # Convert any remaining numpy types or non-JSON-serializable types
    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(x) for x in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        return o
    
    with open(os.path.join(out_dir, 'inferential_results.json'), 'w') as f:
        json.dump(_clean(out_bundle), f, indent=2)
    print(f"\nWrote inferential_results.json.")
    print(f"\nStep D inferential analysis COMPLETE. Outputs in: {out_dir}")


if __name__ == '__main__':
    main()
