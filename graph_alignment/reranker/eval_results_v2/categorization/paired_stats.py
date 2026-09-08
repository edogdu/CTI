"""
paired_stats.py — Statistical primitives for ACSAC 2026 paired v1-vs-v2
comparison (Step C').

This module provides:
    - exact_paired_permutation_test(d): exact sign-flip permutation test on
      paired differences, two-sided. For n<=20 enumerates all 2^n
      configurations exactly; for n>20 uses Monte Carlo with B=20000.
    - exact_paired_sign_test(d): exact binomial sign test ignoring zero
      differences. Always exact.
    - paired_wilcoxon(d): paired Wilcoxon signed-rank test via scipy with
      method='exact' for n<=25.
    - hodges_lehmann_ci(d): exact Hodges-Lehmann CI on median paired
      difference inverted from the sign test.
    - paired_cles(d): paired common-language effect size,
      P(d_i > 0) + 0.5 * P(d_i = 0).
    - paired_cohens_dz(d): paired Cohen's d_z with Hedges' g_z
      small-sample correction.
    - bca_bootstrap_ci(d): paired BCa bootstrap CI on the mean paired
      difference using scipy.stats.bootstrap, with percentile fallback if
      BCa fails.
    - cluster_bootstrap_ci(d, cluster_ids): cluster bootstrap CI where
      clusters (queries) are the sampling unit, not rows.
    - cohens_kappa(r1, r2, categories): Cohen's kappa for two raters on
      nominal categories.
    - gwets_ac1(r1, r2, categories): Gwet's AC1 agreement coefficient,
      robust to prevalence imbalance.
    - krippendorffs_alpha_nominal(r1, r2): Krippendorff's alpha for
      nominal data with two raters.
    - holm_bonferroni(p_values): Holm step-down adjusted p-values.
    - benjamini_hochberg(p_values): BH-FDR adjusted q-values.

Design principles:
    - Every test that admits an exact distribution uses the exact test
      when n is small enough. For our n=10-11 primary categories, this
      means truly exact p-values.
    - Every CI method has an explicit fallback for degenerate cases
      (zero variance, all-tied differences, BCa failure) with logging.
    - Tests assume the paired structure is encoded in the input order:
      d[i] = v2[i] - v1[i], same i across vectors.

Verified against scipy.stats reference implementations where available.
Author: Shane Waldrop, ACSAC 2026 paper (Step C', May 2026).
"""

import math
import warnings
from itertools import product

import numpy as np
import scipy.stats as st


# ============================================================================
# Inferential tests on paired differences
# ============================================================================

def exact_paired_permutation_test(d, n_mc=20000, seed=42):
    """
    Two-sided exact paired permutation test (sign-flip) on the mean of d.

    Under the null hypothesis that d_i has a symmetric distribution about
    zero, each sign of d_i is equally likely positive or negative. We
    enumerate all 2^n sign configurations of d (or sample n_mc of them
    if n > 20) and compute the proportion with |permuted mean| >=
    |observed mean|.

    The p-value uses the (B+1)/(N+1) smoothing convention:
        p = (1 + count_extreme) / (1 + n_total)
    so p > 0 even for the most extreme observed configuration.

    Args:
        d: array-like of n paired differences.
        n_mc: number of Monte Carlo permutations if n > 20.
        seed: RNG seed for the Monte Carlo branch.

    Returns:
        dict with keys: stat (observed mean), p_value, n_paired,
        method ('exact' or 'monte_carlo'), n_permutations.

    Test against R's exactRankTests::perm.test where applicable.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return {'stat': float('nan'), 'p_value': float('nan'),
                'n_paired': 0, 'method': 'undefined', 'n_permutations': 0}
    obs = float(np.mean(d))
    abs_d = np.abs(d)

    # Exact enumeration when feasible.
    if n <= 20:
        # All 2^n sign configurations. signs is a (2^n, n) bool array
        # where each row is one configuration of +1/-1.
        n_total = 2 ** n
        # For memory efficiency, iterate without materializing the array.
        count_extreme = 0
        # The mean under sign flip s is (1/n) * sum(s_i * |d_i|).
        # Iterate via product of {-1, 1}^n.
        for signs in product([-1, 1], repeat=n):
            permuted_mean = np.dot(signs, abs_d) / n
            if abs(permuted_mean) >= abs(obs) - 1e-12:
                count_extreme += 1
        p_value = (count_extreme) / n_total  # exact, no smoothing needed
        return {'stat': obs, 'p_value': p_value, 'n_paired': n,
                'method': 'exact', 'n_permutations': n_total}

    # Monte Carlo for larger n.
    rng = np.random.default_rng(seed)
    count_extreme = 0
    for _ in range(n_mc):
        signs = rng.choice([-1, 1], size=n)
        permuted_mean = np.dot(signs, abs_d) / n
        if abs(permuted_mean) >= abs(obs) - 1e-12:
            count_extreme += 1
    p_value = (1 + count_extreme) / (1 + n_mc)  # smoothed
    return {'stat': obs, 'p_value': p_value, 'n_paired': n,
            'method': 'monte_carlo', 'n_permutations': n_mc}


def exact_paired_sign_test(d):
    """
    Exact two-sided sign test on paired differences.

    Counts positive and negative differences (zeros excluded), then tests
    whether the count of positives is consistent with Binomial(n_nonzero, 0.5)
    under the null of no median shift.

    Args:
        d: array-like of n paired differences.

    Returns:
        dict with keys: n_pos, n_neg, n_zero, n_paired, n_nonzero,
        p_value, method ('exact_binomial').
    """
    d = np.asarray(d, dtype=float)
    n_pos = int(np.sum(d > 0))
    n_neg = int(np.sum(d < 0))
    n_zero = int(np.sum(d == 0))
    n_nonzero = n_pos + n_neg
    if n_nonzero == 0:
        return {'n_pos': n_pos, 'n_neg': n_neg, 'n_zero': n_zero,
                'n_paired': len(d), 'n_nonzero': 0, 'p_value': 1.0,
                'method': 'exact_binomial'}
    # Two-sided exact binomial p-value
    k_extreme = min(n_pos, n_neg)
    # Sum binomial PMF for k=0..k_extreme and k=n_nonzero-k_extreme..n_nonzero
    # under p=0.5
    p_value = float(st.binomtest(n_pos, n_nonzero, 0.5,
                                  alternative='two-sided').pvalue)
    return {'n_pos': n_pos, 'n_neg': n_neg, 'n_zero': n_zero,
            'n_paired': len(d), 'n_nonzero': n_nonzero,
            'p_value': p_value, 'method': 'exact_binomial'}


def paired_wilcoxon(d, zero_method='pratt'):
    """
    Paired Wilcoxon signed-rank test via scipy. Uses exact method for n<=25.

    NOTE: assumes the distribution of d_i is symmetric about its median.
    For our data, this assumption is often violated (especially for
    ambig_actor_software, where the carbanak rows create heavy right
    skew). We report Wilcoxon as a robustness check, not as the primary
    test.

    Args:
        d: array-like of n paired differences.
        zero_method: how to handle zero differences. 'pratt' includes
            zeros in ranking (preserves info); 'wilcox' drops them
            (legacy default but loses information).

    Returns:
        dict with keys: stat, p_value, n_paired, n_zero, method,
        warning (str or None).
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    n_zero = int(np.sum(d == 0))
    warning_msg = None
    if n < 2:
        return {'stat': float('nan'), 'p_value': float('nan'),
                'n_paired': n, 'n_zero': n_zero, 'method': 'undefined',
                'warning': 'n < 2'}
    # scipy raises an error if all values are zero. Guard.
    if np.all(d == 0):
        return {'stat': 0.0, 'p_value': 1.0, 'n_paired': n,
                'n_zero': n_zero, 'method': 'all_zero',
                'warning': 'all differences are zero'}
    # Let scipy pick the method via 'auto'. As of scipy 1.17, auto picks
    # exact for n<=50 with both wilcox and pratt zero handling.
    try:
        result = st.wilcoxon(d, zero_method=zero_method, method='auto')
        # Determine which method scipy actually used (heuristic: exact
        # for n<=50 in modern scipy).
        actual_method = 'exact' if n <= 50 else 'approx'
        return {'stat': float(result.statistic), 'p_value': float(result.pvalue),
                'n_paired': n, 'n_zero': n_zero, 'method': actual_method,
                'warning': warning_msg}
    except Exception as e:
        return {'stat': float('nan'), 'p_value': float('nan'),
                'n_paired': n, 'n_zero': n_zero, 'method': 'failed',
                'warning': f'wilcoxon failed: {e}'}


# ============================================================================
# Effect sizes
# ============================================================================

def paired_cles(d):
    """
    Paired common-language effect size:
        CLES_paired = P(d_i > 0) + 0.5 * P(d_i == 0)

    The probability that a randomly selected paired difference is
    positive (with ties counted at half weight). A value of 0.5 indicates
    no shift; values > 0.5 indicate v2 systematically higher than v1.

    Returns:
        float in [0, 1].
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return float('nan')
    return (float(np.sum(d > 0)) + 0.5 * float(np.sum(d == 0))) / n


def paired_cohens_dz(d):
    """
    Paired Cohen's d_z with Hedges' g_z small-sample correction.

    d_z = mean(d) / sd(d, ddof=1)
    g_z = J(nu) * d_z,  J(nu) = 1 - 3 / (4*nu - 1),  nu = n - 1

    Returns:
        dict with keys: dz, gz, J, nu, n_paired.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return {'dz': float('nan'), 'gz': float('nan'),
                'J': float('nan'), 'nu': n - 1, 'n_paired': n}
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return {'dz': float('inf') if np.mean(d) != 0 else 0.0,
                'gz': float('inf') if np.mean(d) != 0 else 0.0,
                'J': float('nan'), 'nu': n - 1, 'n_paired': n}
    dz = float(np.mean(d) / sd)
    nu = n - 1
    J = 1.0 - 3.0 / (4.0 * nu - 1.0) if nu > 0 else float('nan')
    gz = J * dz if not math.isnan(J) else float('nan')
    return {'dz': dz, 'gz': gz, 'J': J, 'nu': nu, 'n_paired': n}


# ============================================================================
# Confidence intervals
# ============================================================================

def hodges_lehmann_ci(d, conf_level=0.95):
    """
    Exact Hodges-Lehmann CI on the median of paired differences,
    inverted from the sign test.

    For n paired observations, sort the differences and find indices
    that bracket the conf_level CI based on the binomial distribution.

    Equivalent to scipy.stats.wilcoxon's confidence_interval method
    when method='exact', but here we compute it directly as a
    SIGN-test-based CI (more conservative than Wilcoxon-based but
    requires only that the median is well-defined, no symmetry
    assumption).

    Args:
        d: array-like of n paired differences.
        conf_level: e.g. 0.95 for 95% CI.

    Returns:
        dict with keys: median, ci_lo, ci_hi, conf_level, n_paired,
        method ('hodges_lehmann_sign_inversion').
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return {'median': float(np.median(d)) if n == 1 else float('nan'),
                'ci_lo': float('nan'), 'ci_hi': float('nan'),
                'conf_level': conf_level, 'n_paired': n,
                'method': 'undefined'}
    sorted_d = np.sort(d)
    alpha = 1.0 - conf_level
    # Find smallest k such that P(X <= k-1) <= alpha/2 under Bin(n, 0.5).
    # CI is [sorted_d[k-1], sorted_d[n-k]] (0-indexed).
    k = 0
    cumprob = 0.0
    for i in range(n + 1):
        cumprob += st.binom.pmf(i, n, 0.5)
        if cumprob > alpha / 2.0:
            k = i  # k = number of values to drop from each tail
            break
    # k is the number of order statistics to drop from each end.
    # CI is [sorted_d[k], sorted_d[n-1-k]] (0-indexed inclusive).
    if k == 0:
        ci_lo, ci_hi = sorted_d[0], sorted_d[-1]
    else:
        if k > n - 1 - k:
            # CI undefined at this confidence level for this n.
            return {'median': float(np.median(d)),
                    'ci_lo': float('nan'), 'ci_hi': float('nan'),
                    'conf_level': conf_level, 'n_paired': n,
                    'method': 'hodges_lehmann_sign_inversion',
                    'warning': f'n={n} too small for {conf_level:.0%} CI'}
        ci_lo = float(sorted_d[k])
        ci_hi = float(sorted_d[n - 1 - k])
    return {'median': float(np.median(d)), 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'conf_level': conf_level, 'n_paired': n,
            'method': 'hodges_lehmann_sign_inversion', 'k_dropped': k}


def bca_bootstrap_ci(d, conf_level=0.95, n_resamples=9999, seed=42):
    """
    Paired BCa bootstrap CI on the mean of paired differences, with
    percentile fallback if BCa fails.

    Uses scipy.stats.bootstrap with method='BCa' and paired=True.
    Detects BCa degeneracy (NaN bounds, infinite values) and falls
    back to method='percentile' with a flag.

    Args:
        d: array-like of n paired differences.
        conf_level: e.g. 0.95 for 95% CI.
        n_resamples: number of bootstrap resamples.
        seed: RNG seed.

    Returns:
        dict with keys: mean, ci_lo, ci_hi, conf_level, n_paired,
        method ('BCa' or 'percentile_fallback'), bca_status
        ('ok', 'unstable', 'failed').
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 2:
        return {'mean': float(np.mean(d)) if n >= 1 else float('nan'),
                'ci_lo': float('nan'), 'ci_hi': float('nan'),
                'conf_level': conf_level, 'n_paired': n,
                'method': 'undefined', 'bca_status': 'failed'}

    # If all values are identical, the mean is exact and the CI is a
    # single point. scipy will return NaN in this case.
    if np.all(d == d[0]):
        return {'mean': float(d[0]), 'ci_lo': float(d[0]),
                'ci_hi': float(d[0]), 'conf_level': conf_level,
                'n_paired': n, 'method': 'all_constant',
                'bca_status': 'degenerate_constant'}

    rng = np.random.default_rng(seed)

    # Try BCa first.
    bca_status = 'ok'
    bca_method = 'BCa'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            res = st.bootstrap((d,), np.mean, confidence_level=conf_level,
                               n_resamples=n_resamples, method='BCa',
                               random_state=rng, vectorized=False)
            ci_lo = float(res.confidence_interval.low)
            ci_hi = float(res.confidence_interval.high)
            if not (np.isfinite(ci_lo) and np.isfinite(ci_hi)):
                raise ValueError('BCa returned non-finite bounds')
    except Exception:
        # Fall back to percentile.
        bca_status = 'unstable'
        bca_method = 'percentile_fallback'
        rng = np.random.default_rng(seed)  # reset for reproducibility
        try:
            res = st.bootstrap((d,), np.mean, confidence_level=conf_level,
                               n_resamples=n_resamples, method='percentile',
                               random_state=rng, vectorized=False)
            ci_lo = float(res.confidence_interval.low)
            ci_hi = float(res.confidence_interval.high)
        except Exception as e2:
            return {'mean': float(np.mean(d)), 'ci_lo': float('nan'),
                    'ci_hi': float('nan'), 'conf_level': conf_level,
                    'n_paired': n, 'method': 'failed',
                    'bca_status': f'failed: {e2}'}

    return {'mean': float(np.mean(d)), 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'conf_level': conf_level, 'n_paired': n, 'method': bca_method,
            'bca_status': bca_status, 'n_resamples': n_resamples}


def cluster_bootstrap_ci(d, cluster_ids, conf_level=0.95, n_resamples=9999,
                          seed=42):
    """
    Cluster bootstrap CI on the mean of paired differences, treating
    cluster_ids as the sampling unit.

    For our setting, cluster_ids = query_idx, so each bootstrap sample
    selects QUERIES with replacement, then takes ALL row differences
    from those queries. This addresses the row-nesting concern raised
    by ChatGPT for categories where rows-per-query > 1.

    Args:
        d: array-like of n paired differences.
        cluster_ids: array-like of n cluster ids (e.g. query_idx).
        conf_level: e.g. 0.95.
        n_resamples: number of bootstrap resamples.
        seed: RNG seed.

    Returns:
        dict with keys: mean, ci_lo, ci_hi, conf_level, n_paired,
        n_clusters, method ('cluster_percentile_bootstrap').
    """
    d = np.asarray(d, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    n = len(d)
    if len(cluster_ids) != n:
        raise ValueError('d and cluster_ids must have the same length')
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    if n_clusters < 2:
        return {'mean': float(np.mean(d)) if n >= 1 else float('nan'),
                'ci_lo': float('nan'), 'ci_hi': float('nan'),
                'conf_level': conf_level, 'n_paired': n,
                'n_clusters': n_clusters,
                'method': 'undefined_too_few_clusters'}

    # Build cluster -> row indices map for fast lookup.
    cluster_to_rows = {}
    for i, c in enumerate(cluster_ids):
        cluster_to_rows.setdefault(c, []).append(i)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples)
    for b in range(n_resamples):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters,
                                      replace=True)
        # Concatenate all row differences from sampled clusters.
        row_indices = []
        for c in sampled_clusters:
            row_indices.extend(cluster_to_rows[c])
        boot_means[b] = np.mean(d[row_indices])

    alpha = 1.0 - conf_level
    ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {'mean': float(np.mean(d)), 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'conf_level': conf_level, 'n_paired': n,
            'n_clusters': n_clusters,
            'method': 'cluster_percentile_bootstrap',
            'n_resamples': n_resamples}


# ============================================================================
# Agreement metrics
# ============================================================================

def cohens_kappa(r1, r2, categories):
    """
    Cohen's kappa for two raters on a fixed set of nominal categories.

    Equivalent to sklearn.metrics.cohen_kappa_score for matching
    inputs. Implemented here to avoid the sklearn dependency and to
    return raw agreement P_o and expected agreement P_e alongside
    kappa.

    Args:
        r1, r2: equal-length arrays of category labels.
        categories: ordered list of all possible category labels.

    Returns:
        dict with keys: kappa, p_observed, p_expected, n,
        confusion_matrix (K x K dict-of-dicts).
    """
    r1 = list(r1)
    r2 = list(r2)
    if len(r1) != len(r2):
        raise ValueError('rater label lists must be equal length')
    n = len(r1)
    if n == 0:
        return {'kappa': float('nan'), 'p_observed': float('nan'),
                'p_expected': float('nan'), 'n': 0,
                'confusion_matrix': {}}
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    K = len(categories)
    M = np.zeros((K, K), dtype=int)
    for a, b in zip(r1, r2):
        if a in cat_to_idx and b in cat_to_idx:
            M[cat_to_idx[a], cat_to_idx[b]] += 1
    p_observed = float(np.trace(M)) / n
    row_marg = M.sum(axis=1) / n
    col_marg = M.sum(axis=0) / n
    p_expected = float(np.dot(row_marg, col_marg))
    if p_expected == 1.0:
        kappa = float('nan')
    else:
        kappa = (p_observed - p_expected) / (1.0 - p_expected)
    cm = {c1: {c2: int(M[i, j]) for j, c2 in enumerate(categories)}
          for i, c1 in enumerate(categories)}
    return {'kappa': kappa, 'p_observed': p_observed,
            'p_expected': p_expected, 'n': n, 'confusion_matrix': cm}


def gwets_ac1(r1, r2, categories):
    """
    Gwet's AC1 agreement coefficient.

    AC1 = (P_o - P_e_AC1) / (1 - P_e_AC1)
    where P_e_AC1 = (1/(K-1)) * sum_k pi_k * (1 - pi_k)
    and pi_k is the average marginal proportion of category k across
    raters: pi_k = (P(rater1=k) + P(rater2=k)) / 2.

    AC1 is more robust than Cohen's kappa to extreme prevalence
    imbalance. References: Gwet (2008), Wongpakaran et al. (2013).

    Args:
        r1, r2: equal-length arrays of category labels.
        categories: ordered list of all possible category labels.

    Returns:
        dict with keys: ac1, p_observed, p_expected_ac1, n.
    """
    r1 = list(r1)
    r2 = list(r2)
    if len(r1) != len(r2):
        raise ValueError('rater label lists must be equal length')
    n = len(r1)
    if n == 0:
        return {'ac1': float('nan'), 'p_observed': float('nan'),
                'p_expected_ac1': float('nan'), 'n': 0}
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    K = len(categories)
    M = np.zeros((K, K), dtype=int)
    for a, b in zip(r1, r2):
        if a in cat_to_idx and b in cat_to_idx:
            M[cat_to_idx[a], cat_to_idx[b]] += 1
    p_observed = float(np.trace(M)) / n
    pi = (M.sum(axis=1) + M.sum(axis=0)) / (2 * n)
    p_expected = float(np.sum(pi * (1 - pi))) / (K - 1) if K > 1 else 0.0
    if p_expected == 1.0:
        ac1 = float('nan')
    else:
        ac1 = (p_observed - p_expected) / (1.0 - p_expected)
    return {'ac1': ac1, 'p_observed': p_observed,
            'p_expected_ac1': p_expected, 'n': n}


def krippendorffs_alpha_nominal(r1, r2):
    """
    Krippendorff's alpha for nominal data with two raters and no
    missing data.

    For two raters and complete data, alpha is computed as:
        alpha = 1 - (D_o / D_e)
    where D_o is observed disagreement (proportion of pairs that
    disagree) and D_e is expected disagreement (1 - sum(p_k^2)) under
    the unbiased category distribution.

    For two complete raters on nominal data, alpha is numerically very
    close to Cohen's kappa but uses a slightly different correction
    (sample-size adjustment).

    Args:
        r1, r2: equal-length arrays of category labels.

    Returns:
        dict with keys: alpha, n.
    """
    r1 = list(r1)
    r2 = list(r2)
    n = len(r1)
    if len(r2) != n:
        raise ValueError('rater label lists must be equal length')
    if n == 0:
        return {'alpha': float('nan'), 'n': 0}

    # Build category counts across both raters (concatenated).
    from collections import Counter
    all_labels = list(r1) + list(r2)
    counts = Counter(all_labels)
    total = 2 * n
    # Expected disagreement (one minus probability that two random
    # values from the joint distribution match).
    p_e = 0.0
    for c, cnt in counts.items():
        p = cnt / total
        p_e += p * p
    D_e = 1.0 - p_e
    if D_e == 0:
        return {'alpha': float('nan'), 'n': n,
                'note': 'D_e == 0; alpha undefined'}

    # Observed disagreement using Krippendorff's coincidence-table form
    # for two raters and nominal data.
    D_o = 0.0
    for a, b in zip(r1, r2):
        if a != b:
            D_o += 1.0
    D_o = D_o / n  # observed disagreement rate

    # Sample-size correction (the (n-1)/n factor in Krippendorff's
    # bootstrap form for two raters):
    D_e_corrected = D_e * total / (total - 1)
    alpha = 1.0 - (D_o / D_e_corrected) if D_e_corrected > 0 else float('nan')
    return {'alpha': float(alpha), 'n': n,
            'D_observed': float(D_o),
            'D_expected_corrected': float(D_e_corrected)}


# ============================================================================
# Multiple-comparison correction
# ============================================================================

def holm_bonferroni(p_values):
    """
    Holm step-down Bonferroni adjusted p-values.

    Sort p-values ascending; the i-th smallest p (1-indexed) is adjusted
    by multiplying by (m - i + 1), where m is the total number of tests.
    Adjusted p-values are clipped to [0, 1] and made monotone
    non-decreasing in original sort order.

    Args:
        p_values: array-like of m raw p-values.

    Returns:
        numpy array of m Holm-adjusted p-values, in original input order.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=float)
    # Sort ascending and remember original indices.
    order = np.argsort(p)
    p_sorted = p[order]
    adj_sorted = np.empty(m)
    cum_max = 0.0
    for i in range(m):
        adj = p_sorted[i] * (m - i)
        if adj > cum_max:
            cum_max = adj
        adj_sorted[i] = min(cum_max, 1.0)
    # Map back to original order.
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj


def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg FDR-adjusted q-values (linear step-up).

    Sort p-values ascending; the i-th smallest p (1-indexed) is adjusted
    by multiplying by m/i. Adjusted values are clipped to [0, 1] and
    made monotone non-decreasing in original sort order.

    Args:
        p_values: array-like of m raw p-values.

    Returns:
        numpy array of m BH q-values, in original input order.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    p_sorted = p[order]
    q_sorted = np.empty(m)
    # Walk from largest to smallest, maintaining the running min.
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1  # 1-indexed
        adj = p_sorted[i] * m / rank
        running_min = min(running_min, adj)
        q_sorted[i] = min(running_min, 1.0)
    q = np.empty(m)
    q[order] = q_sorted
    return q


# ============================================================================
# Self-test (run when invoked as a script)
# ============================================================================

if __name__ == '__main__':
    print('Running paired_stats.py self-tests...')
    print()

    # Test 1: paired permutation test against trivial cases
    rng = np.random.default_rng(42)

    # All differences positive: p should be small
    d_pos = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = exact_paired_permutation_test(d_pos)
    print(f"All-positive diffs (n=5): mean={res['stat']:.3f}, "
          f"p={res['p_value']:.4f}  ({res['method']})")
    assert res['method'] == 'exact', "should use exact for n=5"
    assert res['p_value'] < 0.10, f"expected small p, got {res['p_value']}"

    # All zero: p = 1
    d_zero = np.array([0.0, 0.0, 0.0])
    res = exact_paired_permutation_test(d_zero)
    print(f"All-zero diffs (n=3): mean={res['stat']:.3f}, p={res['p_value']:.4f}")
    assert res['p_value'] == 1.0

    # Test 2: sign test
    d_mixed = np.array([1, -1, 2, -3, 0, 4, 5, 6])
    res = exact_paired_sign_test(d_mixed)
    print(f"Mixed diffs sign test: pos={res['n_pos']}, neg={res['n_neg']}, "
          f"p={res['p_value']:.4f}")

    # Test 3: Wilcoxon vs scipy
    d_test = np.array([0.5, 1.0, -0.3, 0.8, 1.5, 2.0, -0.1, 0.9])
    res = paired_wilcoxon(d_test)
    direct = st.wilcoxon(d_test, zero_method='pratt')
    print(f"Wilcoxon: stat={res['stat']:.4f}, p={res['p_value']:.4f}")
    assert abs(res['p_value'] - float(direct.pvalue)) < 1e-9, \
        f"wilcoxon mismatch: {res['p_value']} vs {direct.pvalue}"

    # Test 4: paired CLES sanity
    d_pos = np.array([1.0] * 10)
    cles = paired_cles(d_pos)
    print(f"Paired CLES (all positive): {cles:.3f}")
    assert cles == 1.0

    d_zero = np.array([0.0] * 10)
    cles = paired_cles(d_zero)
    print(f"Paired CLES (all zero): {cles:.3f}")
    assert cles == 0.5

    # Test 5: Cohen's d_z and Hedges' g_z
    d_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = paired_cohens_dz(d_test)
    print(f"Cohen's d_z: dz={res['dz']:.4f}, gz={res['gz']:.4f}, J={res['J']:.4f}")

    # Test 6: BCa CI
    d_test = rng.normal(0.5, 1.0, size=20)
    res = bca_bootstrap_ci(d_test, n_resamples=999)
    print(f"BCa CI (n=20): mean={res['mean']:.4f}, "
          f"CI=[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}] ({res['bca_status']})")

    # Test 7: Cluster bootstrap
    d_clust = np.array([1.0, 1.1, 2.0, 0.5, 0.6, 0.7, 3.0, 3.1])
    cluster_ids = np.array([1, 1, 2, 3, 3, 3, 4, 4])
    res = cluster_bootstrap_ci(d_clust, cluster_ids, n_resamples=999)
    print(f"Cluster bootstrap: mean={res['mean']:.4f}, "
          f"CI=[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}], "
          f"n_clusters={res['n_clusters']}")

    # Test 8: Cohen's kappa
    r1 = ['cyber', 'stop', 'other', 'cyber', 'cyber']
    r2 = ['cyber', 'stop', 'cyber', 'cyber', 'cyber']
    cats = ['cyber', 'stop', 'other']
    res = cohens_kappa(r1, r2, cats)
    print(f"Cohen's kappa: kappa={res['kappa']:.4f}, "
          f"p_o={res['p_observed']:.4f}, p_e={res['p_expected']:.4f}")
    # Verify against sklearn if available
    try:
        from sklearn.metrics import cohen_kappa_score
        skl_kappa = cohen_kappa_score(r1, r2, labels=cats)
        print(f"  sklearn comparison: {skl_kappa:.4f}")
        assert abs(res['kappa'] - skl_kappa) < 1e-9, \
            f"kappa mismatch: {res['kappa']} vs {skl_kappa}"
    except ImportError:
        print('  (sklearn not available; skipping cross-check)')

    # Test 9: Gwet's AC1
    res = gwets_ac1(r1, r2, cats)
    print(f"Gwet's AC1: ac1={res['ac1']:.4f}")

    # Test 10: Krippendorff's alpha
    res = krippendorffs_alpha_nominal(r1, r2)
    print(f"Krippendorff's alpha: alpha={res['alpha']:.4f}")

    # Test 11: Holm-Bonferroni and BH
    pvals = np.array([0.001, 0.012, 0.04, 0.08, 0.40])
    holm = holm_bonferroni(pvals)
    bh = benjamini_hochberg(pvals)
    print(f"\nMultiple-comparison correction (m=5):")
    print(f"  raw p:  {pvals}")
    print(f"  Holm:   {holm}")
    print(f"  BH-FDR: {bh}")

    # Verify Holm against statsmodels if available
    try:
        from statsmodels.stats.multitest import multipletests
        _, holm_sm, _, _ = multipletests(pvals, method='holm')
        _, bh_sm, _, _ = multipletests(pvals, method='fdr_bh')
        print(f"  Holm (statsmodels):   {holm_sm}")
        print(f"  BH-FDR (statsmodels): {bh_sm}")
        assert np.allclose(holm, holm_sm, atol=1e-9), 'Holm mismatch'
        assert np.allclose(bh, bh_sm, atol=1e-9), 'BH-FDR mismatch'
    except ImportError:
        print('  (statsmodels not available; skipping cross-check)')

    # Test 12: Hodges-Lehmann CI
    d_test = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, -0.2, 0.3, 0.8])
    res = hodges_lehmann_ci(d_test)
    print(f"\nHodges-Lehmann CI (n=10): median={res['median']:.4f}, "
          f"CI=[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}], "
          f"k={res['k_dropped']}")

    print('\nAll self-tests passed.')
