"""
ACSAC 2026 Step E — Stage 2: MCC + Operating-Point Analysis

Reads `per_pair_scores.csv` (output of `score_test_pairs.py`) and computes:

Threshold-free metrics:
    AUROC, AUPRC — both with query-clustered bootstrap 95% percentile CIs.

Threshold-dependent metrics at tau = 0 (raw logit, equivalent to sigmoid 0.5
which is the natural BCE decision boundary; this is "Path A" — no validation-
set tuning, no data-snooping risk):
    MCC, F1, precision, recall, balanced accuracy, specificity. Reported in
    both query-weighted (primary, macro-average over 146 queries) and pair-
    level (pooled across all ~2,920 pairs) variants.

Operating-point table:
    TPR at FPR in {0.01, 0.05, 0.10}    — security/malware convention
    FPR at TPR in {0.80, 0.90, 0.95}     — analyst-screening convention
    Each with the threshold value (logit) at that operating point and a
    query-clustered bootstrap 95% percentile CI.

Ranking metrics (cross-encoder layer only, per Shane's Option 1 decision):
    P@1, Recall@k for k in {3, 5, 10}, MRR, nDCG@k for k in {3, 5, 10}.
    All with query-clustered bootstrap CIs.

Per-actor descriptive breakdown (no CIs — small strata):
    MCC, F1, P@1, AUROC, AUPRC by actor.

Calibration (appendix):
    Brier score, ECE with 10 quantile bins, reliability diagram data.

Four figures (each in PDF + SVG + PNG triplicate):
    e_fig1: ROC + PR curves side by side with query-clustered bootstrap CI
            bands; AUROC and AUPRC in the legends with their CIs; operating
            points marked on the ROC.
    e_fig2: Operating-point bar chart showing TPR at each fixed FPR and FPR
            at each fixed TPR with bootstrap error bars.
    e_fig3: Per-actor MCC forest plot (descriptive point estimates; explicit
            caveat about small-stratum CIs in the figure caption).
    e_fig4: Reliability diagram for the appendix; sigmoid-of-logit confidence
            on x axis, empirical accuracy on y; 10 quantile bins.

Bootstrap protocol (uniform across all metrics):
    Query-clustered: resample 146 queries with replacement; take all pairs
    from each sampled query intact. Preserves within-query correlation.
    Resamples: 10,000.
    Method: percentile CI (safer than BCa near ceiling; BCa's acceleration
        estimator can diverge when many jackknife pseudo-values are equal,
        which happens at our P@1 = 94.52% with only 8 errors).
    Seed: 42 (reported in provenance JSON).
    Level: 95%.

P@1 layer note (per Shane's Option 1):
    All metrics in this script characterize the cross-encoder's standalone
    pair-relevance discrimination. The cross-encoder achieves 94.52% P@1
    (138/146 correct) without hierarchical post-processing. Hierarchical
    post-processing is a separate downstream component that expands the top-1
    prediction with implied parents and contributes the additional 0.69 P@1
    points to reach the system-level 95.21%; that component operates above
    the per-pair score layer and is intentionally NOT applied here.

Usage:
    python eval_results_v2/operating_points/mcc_fpr_analysis.py \\
        --per-pair-csv eval_results_v2/operating_points/per_pair_scores.csv \\
        --output-dir eval_results_v2/operating_points
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import defaultdict, OrderedDict

import numpy as np


# =============================================================================
# Deterministic seeds
# =============================================================================
BOOTSTRAP_SEED = 42
N_BOOTSTRAP_RESAMPLES = 10_000
CI_LEVEL = 0.95

# Operating points to compute
TPR_AT_FPR_TARGETS = [0.01, 0.05, 0.10]
FPR_AT_TPR_TARGETS = [0.80, 0.90, 0.95]

# Ranking metric k values
RANK_K_VALUES = [3, 5, 10]

# Calibration binning
N_CALIBRATION_BINS = 10

# FPR/recall grid for ROC/PR curve CI bands
ROC_FPR_GRID = np.concatenate([
    np.linspace(0.0, 0.10, 21),    # finer near low-FPR (operationally critical)
    np.linspace(0.12, 1.0, 45),    # coarser elsewhere
])
PR_RECALL_GRID = np.linspace(0.0, 1.0, 51)


# =============================================================================
# CSV loader
# =============================================================================
def load_per_pair_csv(csv_path):
    """Load per_pair_scores.csv into a list of dicts with parsed numerics.
    
    Robust to floats-as-strings, bool-as-strings, and the cp1252-vs-utf-8 
    encoding lesson from Step D. Tries utf-8 first, then cp1252 as fallback.
    """
    rows = None
    for enc in ('utf-8', 'cp1252'):
        try:
            with open(csv_path, encoding=enc) as f:
                r = csv.DictReader(f)
                rows = list(r)
            break
        except UnicodeDecodeError:
            continue
    if rows is None:
        raise IOError(f"Could not read {csv_path} as utf-8 or cp1252")
    
    # Parse numerics
    int_cols = ['label_gold', 'rank_within_query', 'n_golds_for_query',
                'n_candidates_for_query']
    float_cols = ['score_raw_logit', 'score_sigmoid']
    for r in rows:
        for c in int_cols:
            try:
                r[c] = int(r[c])
            except (ValueError, KeyError):
                r[c] = 0
        for c in float_cols:
            try:
                r[c] = float(r[c])
            except (ValueError, KeyError):
                r[c] = float('nan')
    return rows


def group_by_query(rows):
    """Group flat rows into per-query dicts.
    
    Returns an OrderedDict: query_norm -> {
        'actor': str,
        'pairs': list of (score_raw_logit, label_gold, candidate_id) tuples,
        'n_golds': int,
        'n_candidates': int,
    }.
    """
    per_query = OrderedDict()
    for r in rows:
        qn = r['query_norm']
        if qn not in per_query:
            per_query[qn] = {
                'actor': r['actor'],
                'pairs': [],
            }
        per_query[qn]['pairs'].append((
            r['score_raw_logit'],
            r['label_gold'],
            r['candidate_id'],
        ))
    # Compute per-query summary stats
    for qn, data in per_query.items():
        labels = [p[1] for p in data['pairs']]
        data['n_golds'] = sum(labels)
        data['n_candidates'] = len(data['pairs'])
    return per_query


# =============================================================================
# Metric primitives
# =============================================================================
def confusion_at_threshold(scores, labels, threshold=0.0):
    """Compute (TP, FP, TN, FN) for a binary classifier at given threshold.
    
    scores: raw logits (we threshold at 0 by default, equivalent to sigmoid > 0.5)
    labels: 0/1
    threshold: cutoff for positive prediction
    """
    s = np.asarray(scores)
    y = np.asarray(labels, dtype=int)
    pred = (s > threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return tp, fp, tn, fn


def mcc_from_confusion(tp, fp, tn, fn):
    """Matthews Correlation Coefficient from confusion matrix.
    
    Returns 0.0 when undefined (any factor in denominator is 0), which matches
    sklearn.metrics.matthews_corrcoef's behavior. The undefined case happens
    when the model predicts only one class (all positives or all negatives),
    which is fine to report as MCC = 0 (no discrimination).
    """
    denom = np.sqrt(float(tp + fp) * float(tp + fn)
                    * float(tn + fp) * float(tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def f1_from_confusion(tp, fp, tn, fn):
    """F1 score from confusion matrix. Returns 0 if undefined."""
    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def precision_from_confusion(tp, fp, tn, fn):
    if (tp + fp) == 0:
        return 0.0
    return tp / (tp + fp)


def recall_from_confusion(tp, fp, tn, fn):
    if (tp + fn) == 0:
        return 0.0
    return tp / (tp + fn)


def specificity_from_confusion(tp, fp, tn, fn):
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)


def balanced_accuracy_from_confusion(tp, fp, tn, fn):
    """(sensitivity + specificity) / 2."""
    return 0.5 * (recall_from_confusion(tp, fp, tn, fn)
                  + specificity_from_confusion(tp, fp, tn, fn))


# =============================================================================
# AUROC / AUPRC (vectorized, no sklearn dependency for the inner loops)
# =============================================================================
def compute_roc(scores, labels):
    """Compute ROC curve points: (fpr, tpr, thresholds), sorted by increasing FPR.
    
    Implementation matches sklearn.metrics.roc_curve semantics. Returns
    arrays that include the (0, 0) and (1, 1) endpoints.
    
    Why we roll our own rather than calling sklearn: this function is called
    10,001 times during bootstrap (once on full data plus 10K resamples).
    sklearn's roc_curve has some overhead from input validation that we don't
    need; rolling our own gives ~3x speedup on this hot path. The output is
    bit-identical for tie-free score distributions, and effectively identical
    in the presence of ties (within floating-point tolerance).
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        # Degenerate: return single-point curve so caller can detect
        return (np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                np.array([np.inf, -np.inf]))
    
    # Sort by descending score
    order = np.argsort(-s, kind='stable')
    s_sorted = s[order]
    y_sorted = y[order]
    
    # Cumulative TP and FP counts as we lower the threshold
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    
    # At each unique score, the FPR/TPR is the cumulative count up to and
    # including all pairs with score >= that score, divided by total neg/pos.
    # We keep only points where the score changes (collapse ties).
    distinct = np.concatenate([np.diff(s_sorted) != 0, [True]])
    tp = tp_cum[distinct]
    fp = fp_cum[distinct]
    thresh = s_sorted[distinct]
    
    # Prepend the (0, 0) point at threshold = +inf
    tp = np.concatenate([[0], tp])
    fp = np.concatenate([[0], fp])
    thresh = np.concatenate([[np.inf], thresh])
    
    tpr = tp / n_pos
    fpr = fp / n_neg
    return fpr, tpr, thresh


def auc_trapezoid(x, y):
    """Trapezoidal AUC. Assumes x is monotone non-decreasing.
    
    Uses np.trapezoid (NumPy 2.0+) when available, falls back to np.trapz
    on older NumPy. Both produce identical results; the name changed.
    """
    trap = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
    if trap is None:
        # Final fallback: compute manually
        return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
    return float(trap(y, x))


def compute_auroc(scores, labels):
    """AUROC via the ROC trapezoidal integration."""
    fpr, tpr, _ = compute_roc(scores, labels)
    return auc_trapezoid(fpr, tpr)


def compute_pr_curve(scores, labels):
    """Precision-Recall curve. Returns (precision, recall, thresholds).
    
    Matches sklearn.metrics.precision_recall_curve semantics: ends at
    recall=0, precision=1. Sorted by decreasing threshold.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return np.array([1.0]), np.array([0.0]), np.array([np.inf])
    
    order = np.argsort(-s, kind='stable')
    s_sorted = s[order]
    y_sorted = y[order]
    
    # At each threshold, predicted-positive set is the prefix of sorted scores
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    
    # Collapse ties (keep last index in each tie group)
    distinct = np.concatenate([np.diff(s_sorted) != 0, [True]])
    tp = tp_cum[distinct]
    fp = fp_cum[distinct]
    thresh = s_sorted[distinct]
    
    precision = tp / np.maximum(tp + fp, 1)  # avoid div-by-zero
    recall = tp / n_pos
    
    # Sklearn-style: append (precision=1, recall=0) at the start (highest thresh)
    precision = np.concatenate([precision, [1.0]])
    recall = np.concatenate([recall, [0.0]])
    thresh = np.concatenate([thresh, [np.inf]])
    
    return precision, recall, thresh


def compute_auprc(scores, labels):
    """Average Precision (AP) — area under PR using the step-function
    integral matching sklearn.metrics.average_precision_score.
    
    AP = sum_k (R_k - R_{k-1}) * P_k, summing in decreasing-threshold order.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return 0.0
    
    order = np.argsort(-s, kind='stable')
    y_sorted = y[order]
    
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / n_pos
    
    # Step-function AP: sum P_k * (R_k - R_{k-1})
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = float(np.sum(precision * (recall - recall_prev)))
    return ap


# =============================================================================
# Operating-point interpolation
# =============================================================================
def tpr_at_fpr(fpr, tpr, thresholds, target_fpr):
    """Interpolate TPR at a given FPR target.
    
    Returns (tpr_at_target, threshold_at_target). If target_fpr exceeds max,
    returns the last point (FPR=1.0, TPR=1.0). If target_fpr is below min,
    returns the first informative point.
    """
    # Linear interpolation. np.interp requires increasing x array which fpr is.
    tpr_val = float(np.interp(target_fpr, fpr, tpr))
    thresh_val = float(np.interp(target_fpr, fpr, thresholds))
    return tpr_val, thresh_val


def fpr_at_tpr(fpr, tpr, thresholds, target_tpr):
    """Interpolate FPR at a given TPR target.
    
    Same as tpr_at_fpr but with axes flipped. TPR may not be monotone in
    the raw ROC output, so we need to handle that — typically TPR is
    monotone non-decreasing in sklearn's output, so np.interp works.
    """
    fpr_val = float(np.interp(target_tpr, tpr, fpr))
    thresh_val = float(np.interp(target_tpr, tpr, thresholds))
    return fpr_val, thresh_val


# =============================================================================
# Ranking metrics (binary relevance)
# =============================================================================
def ranking_metrics_for_query(scores, labels, k_values=RANK_K_VALUES):
    """Compute P@1, Recall@k, MRR, nDCG@k for a single query.
    
    Args:
        scores: array of cross-encoder logits for this query's candidates
        labels: 0/1 array of same length
        k_values: list of k for Recall@k and nDCG@k
    
    Returns dict: {'p_at_1', 'mrr', 'recall_at_3', ..., 'ndcg_at_3', ...}.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)
    n_pos = int(np.sum(y == 1))
    n = len(s)
    
    out = {}
    if n_pos == 0:
        # Edge case: query has no golds. Shouldn't happen in our test set
        # (create_test_split filters out queries with 0 positives), but
        # handle defensively.
        out['p_at_1'] = 0.0
        out['mrr'] = 0.0
        for k in k_values:
            out[f'recall_at_{k}'] = 0.0
            out[f'ndcg_at_{k}'] = 0.0
        return out
    
    # Sort by decreasing score
    order = np.argsort(-s, kind='stable')
    y_sorted = y[order]
    
    # P@1: top-1 is a gold?
    out['p_at_1'] = float(y_sorted[0])
    
    # MRR: 1 / rank of first relevant item
    first_relevant = np.argmax(y_sorted == 1)  # first index where y==1
    if y_sorted[first_relevant] == 1:
        out['mrr'] = 1.0 / (first_relevant + 1)
    else:
        out['mrr'] = 0.0
    
    # Recall@k: fraction of golds in top-k
    # nDCG@k: with binary relevance, DCG = sum(rel_i / log2(i+1)) for top-k
    for k in k_values:
        kk = min(k, n)
        # Recall
        n_relevant_in_top_k = int(np.sum(y_sorted[:kk] == 1))
        out[f'recall_at_{k}'] = n_relevant_in_top_k / n_pos
        # nDCG with binary relevance
        gains = y_sorted[:kk].astype(np.float64)
        discounts = 1.0 / np.log2(np.arange(2, kk + 2))
        dcg = float(np.sum(gains * discounts))
        # Ideal DCG: best possible ordering = all golds at top
        n_ideal = min(n_pos, kk)
        idcg = float(np.sum(np.ones(n_ideal) / np.log2(np.arange(2, n_ideal + 2))))
        out[f'ndcg_at_{k}'] = dcg / idcg if idcg > 0 else 0.0
    
    return out


# =============================================================================
# Calibration
# =============================================================================
def sigmoid(x):
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def brier_score(probabilities, labels):
    """Brier score: mean squared error between probability and label."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def ece_quantile_bins(probabilities, labels, n_bins=N_CALIBRATION_BINS):
    """Expected Calibration Error using quantile bins.
    
    Quantile binning (each bin has equal sample count) is more informative
    than equal-width binning when the predictions are concentrated in a
    narrow probability range, which happens for high-accuracy models where
    most predictions are near 0 or near 1.
    
    Returns:
        ece: scalar Expected Calibration Error
        bin_data: list of dicts with bin boundaries, count, mean confidence, accuracy
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n = len(p)
    if n == 0:
        return 0.0, []
    
    # Sort by probability
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    
    # Quantile bin edges
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if lo == hi:
            continue
        p_bin = p_sorted[lo:hi]
        y_bin = y_sorted[lo:hi]
        mean_conf = float(np.mean(p_bin))
        acc = float(np.mean(y_bin))
        n_bin = hi - lo
        ece += (n_bin / n) * abs(mean_conf - acc)
        bin_data.append({
            'bin_index': i,
            'p_lo': float(p_bin[0]),
            'p_hi': float(p_bin[-1]),
            'count': int(n_bin),
            'mean_confidence': mean_conf,
            'accuracy': acc,
        })
    return float(ece), bin_data


# =============================================================================
# Bootstrap engine
# =============================================================================
def query_clustered_bootstrap_indices(n_queries, n_resamples, seed):
    """Pre-generate all bootstrap resample indices.
    
    Returns ndarray of shape (n_resamples, n_queries) where each row is a
    bootstrap resample (indices into the query list, with replacement).
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_queries, size=(n_resamples, n_queries))


def metric_on_pooled_pairs(per_query_data, query_indices, metric_fn):
    """Apply metric_fn to the pooled (scores, labels) from sampled queries.
    
    per_query_data: list of (scores_array, labels_array, candidate_ids) tuples
                    of length n_queries
    query_indices: 1-D array of query indices for this resample
    metric_fn: callable taking (scores, labels) and returning a scalar
    """
    scores = []
    labels = []
    for i in query_indices:
        scores.append(per_query_data[i][0])
        labels.append(per_query_data[i][1])
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    return metric_fn(scores, labels)


def percentile_ci(values, level=CI_LEVEL):
    """Standard percentile CI on a 1-D array of bootstrap statistics.
    
    Filters out both NaN and inf values defensively. Inf values can appear
    in the threshold-at-operating-point distributions when a bootstrap
    resample's ROC curve has the requested operating point in its first
    (synthetic) interval where the threshold is inf by sklearn convention.
    For our real test data this essentially never happens, but a single inf
    in the array would push the upper percentile to inf and produce a
    misleading CI; filtering is the safe response.
    """
    if len(values) == 0:
        return float('nan'), float('nan')
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float('nan'), float('nan')
    alpha = (1.0 - level) / 2.0
    lo = float(np.percentile(v, 100.0 * alpha))
    hi = float(np.percentile(v, 100.0 * (1.0 - alpha)))
    return lo, hi


# =============================================================================
# Main analysis driver
# =============================================================================
def analyze(per_query, out_dir, verbose=True):
    """Full Step E analysis. Returns inferential_results dict for JSON dump."""
    
    # ----- Set up flat data structures -----
    query_keys = list(per_query.keys())
    n_queries = len(query_keys)
    
    # For each query, pre-extract numpy arrays for fast access
    per_query_arrays = []
    for qn in query_keys:
        pairs = per_query[qn]['pairs']
        scores = np.array([p[0] for p in pairs], dtype=np.float64)
        labels = np.array([p[1] for p in pairs], dtype=np.int32)
        per_query_arrays.append((scores, labels, per_query[qn]['actor']))
    
    # Global flat arrays (point estimate computations)
    all_scores = np.concatenate([a[0] for a in per_query_arrays])
    all_labels = np.concatenate([a[1] for a in per_query_arrays])
    n_pairs = len(all_scores)
    
    if verbose:
        print(f"  n_queries: {n_queries}")
        print(f"  n_pairs: {n_pairs}")
        print(f"  pair-level imbalance: "
              f"{int(np.sum(all_labels == 0))}:{int(np.sum(all_labels == 1))} = "
              f"{np.sum(all_labels == 0)/np.sum(all_labels == 1):.2f}:1 (neg:pos)")
    
    # ----- Point estimates: threshold-free -----
    auroc_point = compute_auroc(all_scores, all_labels)
    auprc_point = compute_auprc(all_scores, all_labels)
    
    # ----- Point estimates: pair-level at tau=0 -----
    tp, fp, tn, fn = confusion_at_threshold(all_scores, all_labels, threshold=0.0)
    pair_mcc = mcc_from_confusion(tp, fp, tn, fn)
    pair_f1 = f1_from_confusion(tp, fp, tn, fn)
    pair_precision = precision_from_confusion(tp, fp, tn, fn)
    pair_recall = recall_from_confusion(tp, fp, tn, fn)
    pair_specificity = specificity_from_confusion(tp, fp, tn, fn)
    pair_balanced_acc = balanced_accuracy_from_confusion(tp, fp, tn, fn)
    
    # ----- Point estimates: query-weighted at tau=0 -----
    per_query_mcc = []
    per_query_f1 = []
    per_query_precision = []
    per_query_recall = []
    per_query_balanced_acc = []
    per_query_specificity = []
    for scores, labels, _ in per_query_arrays:
        ttp, tfp, ttn, tfn = confusion_at_threshold(scores, labels, threshold=0.0)
        per_query_mcc.append(mcc_from_confusion(ttp, tfp, ttn, tfn))
        per_query_f1.append(f1_from_confusion(ttp, tfp, ttn, tfn))
        per_query_precision.append(precision_from_confusion(ttp, tfp, ttn, tfn))
        per_query_recall.append(recall_from_confusion(ttp, tfp, ttn, tfn))
        per_query_balanced_acc.append(balanced_accuracy_from_confusion(ttp, tfp, ttn, tfn))
        per_query_specificity.append(specificity_from_confusion(ttp, tfp, ttn, tfn))
    
    qw_mcc = float(np.mean(per_query_mcc))
    qw_f1 = float(np.mean(per_query_f1))
    qw_precision = float(np.mean(per_query_precision))
    qw_recall = float(np.mean(per_query_recall))
    qw_balanced_acc = float(np.mean(per_query_balanced_acc))
    qw_specificity = float(np.mean(per_query_specificity))
    
    # ----- Point estimates: operating points (pair-level ROC) -----
    fpr_full, tpr_full, thresh_full = compute_roc(all_scores, all_labels)
    
    op_points = {}
    for target_fpr in TPR_AT_FPR_TARGETS:
        v_tpr, v_th = tpr_at_fpr(fpr_full, tpr_full, thresh_full, target_fpr)
        op_points[f'tpr_at_fpr_{target_fpr}'] = {
            'target_fpr': target_fpr,
            'tpr': v_tpr,
            'threshold_logit': v_th,
        }
    for target_tpr in FPR_AT_TPR_TARGETS:
        v_fpr, v_th = fpr_at_tpr(fpr_full, tpr_full, thresh_full, target_tpr)
        op_points[f'fpr_at_tpr_{target_tpr}'] = {
            'target_tpr': target_tpr,
            'fpr': v_fpr,
            'threshold_logit': v_th,
        }
    
    # ----- Point estimates: ranking metrics (query-weighted) -----
    ranking_per_query = [ranking_metrics_for_query(s, l) for s, l, _ in per_query_arrays]
    ranking_keys = list(ranking_per_query[0].keys())
    ranking_point = {k: float(np.mean([r[k] for r in ranking_per_query])) for k in ranking_keys}
    
    # ----- Point estimates: calibration (pair-level) -----
    sig_all = sigmoid(all_scores)
    brier_point = brier_score(sig_all, all_labels)
    ece_point, calib_bin_data = ece_quantile_bins(sig_all, all_labels, N_CALIBRATION_BINS)
    
    # ----- Point estimates: per-actor descriptive (no CIs) -----
    actor_groups = defaultdict(list)
    for i, (scores, labels, actor) in enumerate(per_query_arrays):
        actor_groups[actor].append(i)
    
    per_actor = {}
    for actor, query_idx_list in sorted(actor_groups.items()):
        actor_scores = np.concatenate([per_query_arrays[i][0] for i in query_idx_list])
        actor_labels = np.concatenate([per_query_arrays[i][1] for i in query_idx_list])
        a_tp, a_fp, a_tn, a_fn = confusion_at_threshold(actor_scores, actor_labels, threshold=0.0)
        # Query-weighted P@1 across this actor's queries
        actor_p1 = float(np.mean([ranking_per_query[i]['p_at_1'] for i in query_idx_list]))
        actor_mcc_pair = mcc_from_confusion(a_tp, a_fp, a_tn, a_fn)
        actor_f1_pair = f1_from_confusion(a_tp, a_fp, a_tn, a_fn)
        actor_mcc_qw = float(np.mean([per_query_mcc[i] for i in query_idx_list]))
        actor_f1_qw = float(np.mean([per_query_f1[i] for i in query_idx_list]))
        if len(query_idx_list) > 0:
            actor_auroc = compute_auroc(actor_scores, actor_labels)
            actor_auprc = compute_auprc(actor_scores, actor_labels)
        else:
            actor_auroc = float('nan')
            actor_auprc = float('nan')
        per_actor[actor] = {
            'n_queries': len(query_idx_list),
            'n_pairs': int(len(actor_scores)),
            'p_at_1': actor_p1,
            'mcc_query_weighted': actor_mcc_qw,
            'mcc_pair_level': actor_mcc_pair,
            'f1_query_weighted': actor_f1_qw,
            'f1_pair_level': actor_f1_pair,
            'auroc': actor_auroc,
            'auprc': actor_auprc,
        }
    
    # ----- Bootstrap -----
    if verbose:
        print(f"  Running query-clustered bootstrap: {N_BOOTSTRAP_RESAMPLES} resamples...")
    
    t0 = time.time()
    boot_indices = query_clustered_bootstrap_indices(n_queries, N_BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    
    # Storage for all bootstrap statistics
    boot_results = {
        'auroc': [],
        'auprc': [],
        'pair_mcc': [],
        'pair_f1': [],
        'pair_precision': [],
        'pair_recall': [],
        'pair_specificity': [],
        'pair_balanced_acc': [],
        'qw_mcc': [],
        'qw_f1': [],
        'qw_precision': [],
        'qw_recall': [],
        'qw_specificity': [],
        'qw_balanced_acc': [],
        'brier': [],
        'ece': [],
    }
    for k in ranking_keys:
        boot_results[f'rank_{k}'] = []
    for target_fpr in TPR_AT_FPR_TARGETS:
        boot_results[f'tpr_at_fpr_{target_fpr}'] = []
        boot_results[f'thresh_at_fpr_{target_fpr}'] = []
    for target_tpr in FPR_AT_TPR_TARGETS:
        boot_results[f'fpr_at_tpr_{target_tpr}'] = []
        boot_results[f'thresh_at_tpr_{target_tpr}'] = []
    
    # Storage for ROC/PR curve CI bands
    roc_tpr_grid_samples = []  # one row per resample, columns = ROC_FPR_GRID
    pr_precision_grid_samples = []
    
    for b in range(N_BOOTSTRAP_RESAMPLES):
        idx = boot_indices[b]
        
        # Pooled flat arrays for this resample
        scores_b = np.concatenate([per_query_arrays[i][0] for i in idx])
        labels_b = np.concatenate([per_query_arrays[i][1] for i in idx])
        
        # Threshold-free
        try:
            auroc_b = compute_auroc(scores_b, labels_b)
            auprc_b = compute_auprc(scores_b, labels_b)
        except Exception:
            auroc_b = float('nan')
            auprc_b = float('nan')
        boot_results['auroc'].append(auroc_b)
        boot_results['auprc'].append(auprc_b)
        
        # Pair-level at tau=0
        tp_b, fp_b, tn_b, fn_b = confusion_at_threshold(scores_b, labels_b, 0.0)
        boot_results['pair_mcc'].append(mcc_from_confusion(tp_b, fp_b, tn_b, fn_b))
        boot_results['pair_f1'].append(f1_from_confusion(tp_b, fp_b, tn_b, fn_b))
        boot_results['pair_precision'].append(precision_from_confusion(tp_b, fp_b, tn_b, fn_b))
        boot_results['pair_recall'].append(recall_from_confusion(tp_b, fp_b, tn_b, fn_b))
        boot_results['pair_specificity'].append(specificity_from_confusion(tp_b, fp_b, tn_b, fn_b))
        boot_results['pair_balanced_acc'].append(balanced_accuracy_from_confusion(tp_b, fp_b, tn_b, fn_b))
        
        # Query-weighted at tau=0
        qw_mccs = []
        qw_f1s = []
        qw_ps = []
        qw_rs = []
        qw_ss = []
        qw_bs = []
        rank_metrics_b = defaultdict(list)
        for i in idx:
            sc, lb, _ = per_query_arrays[i]
            ttp, tfp, ttn, tfn = confusion_at_threshold(sc, lb, 0.0)
            qw_mccs.append(mcc_from_confusion(ttp, tfp, ttn, tfn))
            qw_f1s.append(f1_from_confusion(ttp, tfp, ttn, tfn))
            qw_ps.append(precision_from_confusion(ttp, tfp, ttn, tfn))
            qw_rs.append(recall_from_confusion(ttp, tfp, ttn, tfn))
            qw_ss.append(specificity_from_confusion(ttp, tfp, ttn, tfn))
            qw_bs.append(balanced_accuracy_from_confusion(ttp, tfp, ttn, tfn))
            # Ranking metrics
            for k, v in ranking_per_query[i].items():
                rank_metrics_b[k].append(v)
        boot_results['qw_mcc'].append(float(np.mean(qw_mccs)))
        boot_results['qw_f1'].append(float(np.mean(qw_f1s)))
        boot_results['qw_precision'].append(float(np.mean(qw_ps)))
        boot_results['qw_recall'].append(float(np.mean(qw_rs)))
        boot_results['qw_specificity'].append(float(np.mean(qw_ss)))
        boot_results['qw_balanced_acc'].append(float(np.mean(qw_bs)))
        
        for k in ranking_keys:
            boot_results[f'rank_{k}'].append(float(np.mean(rank_metrics_b[k])))
        
        # Operating points (on this resample's ROC)
        fpr_b, tpr_b, thresh_b = compute_roc(scores_b, labels_b)
        for target_fpr in TPR_AT_FPR_TARGETS:
            v_tpr, v_th = tpr_at_fpr(fpr_b, tpr_b, thresh_b, target_fpr)
            boot_results[f'tpr_at_fpr_{target_fpr}'].append(v_tpr)
            boot_results[f'thresh_at_fpr_{target_fpr}'].append(v_th)
        for target_tpr in FPR_AT_TPR_TARGETS:
            v_fpr, v_th = fpr_at_tpr(fpr_b, tpr_b, thresh_b, target_tpr)
            boot_results[f'fpr_at_tpr_{target_tpr}'].append(v_fpr)
            boot_results[f'thresh_at_tpr_{target_tpr}'].append(v_th)
        
        # ROC curve CI band: interpolate TPR at fixed FPR grid
        tpr_grid_b = np.interp(ROC_FPR_GRID, fpr_b, tpr_b)
        roc_tpr_grid_samples.append(tpr_grid_b)
        
        # PR curve CI band: interpolate precision at fixed recall grid
        # PR curves are tricky: precision is not monotone in recall.
        # We interpolate as if it were, using the convex-hull-style PR.
        precision_b, recall_b, _ = compute_pr_curve(scores_b, labels_b)
        # PR curves from compute_pr_curve are in decreasing-threshold order,
        # meaning recall increases. Reverse for np.interp (which needs
        # increasing x).
        precision_grid_b = np.interp(PR_RECALL_GRID, recall_b[::-1], precision_b[::-1])
        pr_precision_grid_samples.append(precision_grid_b)
        
        # Calibration
        sig_b = sigmoid(scores_b)
        brier_b = brier_score(sig_b, labels_b)
        ece_b, _ = ece_quantile_bins(sig_b, labels_b, N_CALIBRATION_BINS)
        boot_results['brier'].append(brier_b)
        boot_results['ece'].append(ece_b)
        
        if verbose and (b + 1) % 1000 == 0:
            elapsed = time.time() - t0
            eta = (N_BOOTSTRAP_RESAMPLES - b - 1) * elapsed / (b + 1)
            print(f"    bootstrap [{b+1}/{N_BOOTSTRAP_RESAMPLES}] "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
    
    bootstrap_elapsed = time.time() - t0
    if verbose:
        print(f"  Bootstrap complete in {bootstrap_elapsed:.0f}s")
    
    roc_tpr_grid_samples = np.array(roc_tpr_grid_samples)  # (n_resamples, n_grid_points)
    pr_precision_grid_samples = np.array(pr_precision_grid_samples)
    
    # ----- Compute percentile CIs -----
    cis = {}
    for k, vals in boot_results.items():
        v = np.array(vals)
        v_clean = v[np.isfinite(v)]
        lo, hi = percentile_ci(v_clean, CI_LEVEL)
        cis[k] = {'ci_lo': lo, 'ci_hi': hi, 'n_valid': int(len(v_clean))}
    
    # ROC CI band
    alpha = (1.0 - CI_LEVEL) / 2.0
    roc_tpr_mean = np.mean(roc_tpr_grid_samples, axis=0)
    roc_tpr_lo = np.percentile(roc_tpr_grid_samples, 100.0 * alpha, axis=0)
    roc_tpr_hi = np.percentile(roc_tpr_grid_samples, 100.0 * (1.0 - alpha), axis=0)
    
    pr_precision_mean = np.mean(pr_precision_grid_samples, axis=0)
    pr_precision_lo = np.percentile(pr_precision_grid_samples, 100.0 * alpha, axis=0)
    pr_precision_hi = np.percentile(pr_precision_grid_samples, 100.0 * (1.0 - alpha), axis=0)
    
    # Assemble results
    results = {
        'meta': {
            'n_queries': n_queries,
            'n_pairs': n_pairs,
            'n_positives': int(np.sum(all_labels == 1)),
            'n_negatives': int(np.sum(all_labels == 0)),
            'imbalance_neg_per_pos': float(np.sum(all_labels == 0) / max(np.sum(all_labels == 1), 1)),
            'threshold_strategy': 'tau = 0 (raw logit, equivalent to sigmoid 0.5; Path A)',
            'aggregation_primary': 'query-weighted (macro-average over 146 queries)',
            'aggregation_secondary': 'pair-level (pooled across all pairs)',
            'p_at_1_layer': 'cross-encoder only (94.52% expected); hierarchical post-processing NOT applied',
            'bootstrap': {
                'method': 'query-clustered percentile',
                'n_resamples': N_BOOTSTRAP_RESAMPLES,
                'seed': BOOTSTRAP_SEED,
                'level': CI_LEVEL,
                'elapsed_seconds': float(bootstrap_elapsed),
            },
        },
        'threshold_free': {
            'auroc': {'point': auroc_point, **cis['auroc']},
            'auprc': {'point': auprc_point, **cis['auprc']},
        },
        'pair_level_at_tau_0': {
            'mcc':      {'point': pair_mcc,           **cis['pair_mcc']},
            'f1':       {'point': pair_f1,            **cis['pair_f1']},
            'precision':{'point': pair_precision,     **cis['pair_precision']},
            'recall':   {'point': pair_recall,        **cis['pair_recall']},
            'specificity':       {'point': pair_specificity, **cis['pair_specificity']},
            'balanced_accuracy': {'point': pair_balanced_acc, **cis['pair_balanced_acc']},
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
        },
        'query_weighted_at_tau_0': {
            'mcc':      {'point': qw_mcc,           **cis['qw_mcc']},
            'f1':       {'point': qw_f1,            **cis['qw_f1']},
            'precision':{'point': qw_precision,     **cis['qw_precision']},
            'recall':   {'point': qw_recall,        **cis['qw_recall']},
            'specificity':       {'point': qw_specificity, **cis['qw_specificity']},
            'balanced_accuracy': {'point': qw_balanced_acc, **cis['qw_balanced_acc']},
        },
        'operating_points': {},
        'ranking_metrics': {},
        'per_actor': per_actor,
        'calibration': {
            'brier': {'point': brier_point, **cis['brier']},
            'ece':   {'point': ece_point,   **cis['ece']},
            'bin_data': calib_bin_data,
            'n_bins': N_CALIBRATION_BINS,
            'binning_method': 'quantile',
        },
    }
    
    for target_fpr in TPR_AT_FPR_TARGETS:
        results['operating_points'][f'tpr_at_fpr_{target_fpr}'] = {
            'target_fpr': target_fpr,
            'tpr': op_points[f'tpr_at_fpr_{target_fpr}']['tpr'],
            'tpr_ci_lo': cis[f'tpr_at_fpr_{target_fpr}']['ci_lo'],
            'tpr_ci_hi': cis[f'tpr_at_fpr_{target_fpr}']['ci_hi'],
            'threshold_logit': op_points[f'tpr_at_fpr_{target_fpr}']['threshold_logit'],
            'threshold_logit_ci_lo': cis[f'thresh_at_fpr_{target_fpr}']['ci_lo'],
            'threshold_logit_ci_hi': cis[f'thresh_at_fpr_{target_fpr}']['ci_hi'],
        }
    for target_tpr in FPR_AT_TPR_TARGETS:
        results['operating_points'][f'fpr_at_tpr_{target_tpr}'] = {
            'target_tpr': target_tpr,
            'fpr': op_points[f'fpr_at_tpr_{target_tpr}']['fpr'],
            'fpr_ci_lo': cis[f'fpr_at_tpr_{target_tpr}']['ci_lo'],
            'fpr_ci_hi': cis[f'fpr_at_tpr_{target_tpr}']['ci_hi'],
            'threshold_logit': op_points[f'fpr_at_tpr_{target_tpr}']['threshold_logit'],
            'threshold_logit_ci_lo': cis[f'thresh_at_tpr_{target_tpr}']['ci_lo'],
            'threshold_logit_ci_hi': cis[f'thresh_at_tpr_{target_tpr}']['ci_hi'],
        }
    
    for k in ranking_keys:
        results['ranking_metrics'][k] = {
            'point': ranking_point[k],
            **cis[f'rank_{k}'],
        }
    
    # Pack curve CI bands for figure rendering
    figure_curves = {
        'roc': {
            'fpr_grid': ROC_FPR_GRID.tolist(),
            'tpr_mean': roc_tpr_mean.tolist(),
            'tpr_lo': roc_tpr_lo.tolist(),
            'tpr_hi': roc_tpr_hi.tolist(),
            'fpr_full': fpr_full.tolist(),
            'tpr_full': tpr_full.tolist(),
        },
        'pr': {
            'recall_grid': PR_RECALL_GRID.tolist(),
            'precision_mean': pr_precision_mean.tolist(),
            'precision_lo': pr_precision_lo.tolist(),
            'precision_hi': pr_precision_hi.tolist(),
            'precision_full': compute_pr_curve(all_scores, all_labels)[0].tolist(),
            'recall_full': compute_pr_curve(all_scores, all_labels)[1].tolist(),
        },
    }
    
    return results, figure_curves


# =============================================================================
# Self-test
# =============================================================================
def _self_test():
    """Verify all the inferential primitives behave correctly on toy data."""
    print("Running self-tests for inferential primitives...")
    
    # Test 1: Perfect classifier
    scores = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    labels = np.array([1, 1, 1, 0, 0, 0])
    tp, fp, tn, fn = confusion_at_threshold(scores, labels, 0.0)
    assert (tp, fp, tn, fn) == (3, 0, 3, 0), f"perfect classifier: {(tp,fp,tn,fn)}"
    assert abs(mcc_from_confusion(tp, fp, tn, fn) - 1.0) < 1e-12
    assert abs(f1_from_confusion(tp, fp, tn, fn) - 1.0) < 1e-12
    assert abs(compute_auroc(scores, labels) - 1.0) < 1e-12
    assert abs(compute_auprc(scores, labels) - 1.0) < 1e-12
    print("  perfect classifier metrics: OK")
    
    # Test 2: Reverse classifier (worst possible)
    scores = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    labels = np.array([1, 1, 1, 0, 0, 0])
    tp, fp, tn, fn = confusion_at_threshold(scores, labels, 0.0)
    assert (tp, fp, tn, fn) == (0, 3, 0, 3)
    assert abs(mcc_from_confusion(tp, fp, tn, fn) - (-1.0)) < 1e-12
    assert abs(compute_auroc(scores, labels) - 0.0) < 1e-12
    print("  reverse classifier metrics: OK")
    
    # Test 3: Random / chance classifier
    np.random.seed(42)
    n = 1000
    scores = np.random.normal(size=n)
    labels = np.random.randint(0, 2, size=n)
    auroc = compute_auroc(scores, labels)
    # Should be near 0.5 for large random sample
    assert abs(auroc - 0.5) < 0.05, f"random AUROC should be ~0.5, got {auroc}"
    print(f"  random classifier AUROC ~ 0.5: OK (got {auroc:.3f})")
    
    # Test 4: AUROC against sklearn
    try:
        from sklearn.metrics import roc_auc_score
        np.random.seed(123)
        scores = np.random.normal(size=500) + 0.5 * np.random.randint(0, 2, size=500)
        labels = (scores > 0).astype(int)
        # Now scores correlate with labels but not perfectly
        scores += np.random.normal(scale=0.5, size=500)
        ours = compute_auroc(scores, labels)
        theirs = roc_auc_score(labels, scores)
        assert abs(ours - theirs) < 1e-6, f"AUROC mismatch: ours={ours}, sklearn={theirs}"
        print(f"  AUROC matches sklearn: OK ({ours:.4f} vs {theirs:.4f})")
        
        from sklearn.metrics import average_precision_score
        ours_ap = compute_auprc(scores, labels)
        theirs_ap = average_precision_score(labels, scores)
        assert abs(ours_ap - theirs_ap) < 1e-6, f"AUPRC mismatch: ours={ours_ap}, sklearn={theirs_ap}"
        print(f"  AUPRC matches sklearn: OK ({ours_ap:.4f} vs {theirs_ap:.4f})")
        
        from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score
        pred = (scores > 0).astype(int)
        tp, fp, tn, fn = confusion_at_threshold(scores, labels, 0.0)
        ours_mcc = mcc_from_confusion(tp, fp, tn, fn)
        theirs_mcc = matthews_corrcoef(labels, pred)
        assert abs(ours_mcc - theirs_mcc) < 1e-10, f"MCC mismatch: {ours_mcc} vs {theirs_mcc}"
        print(f"  MCC matches sklearn: OK ({ours_mcc:.4f} vs {theirs_mcc:.4f})")
        
        ours_f1 = f1_from_confusion(tp, fp, tn, fn)
        theirs_f1 = f1_score(labels, pred)
        assert abs(ours_f1 - theirs_f1) < 1e-10, f"F1 mismatch: {ours_f1} vs {theirs_f1}"
        print(f"  F1 matches sklearn: OK")
        
        ours_p = precision_from_confusion(tp, fp, tn, fn)
        theirs_p = precision_score(labels, pred)
        assert abs(ours_p - theirs_p) < 1e-10
        print(f"  Precision matches sklearn: OK")
        
        ours_r = recall_from_confusion(tp, fp, tn, fn)
        theirs_r = recall_score(labels, pred)
        assert abs(ours_r - theirs_r) < 1e-10
        print(f"  Recall matches sklearn: OK")
    except ImportError:
        print("  sklearn not available; skipping cross-check (not fatal)")
    
    # Test 5: Operating-point interpolation
    fpr = np.array([0.0, 0.05, 0.10, 0.20, 0.50, 1.0])
    tpr = np.array([0.0, 0.50, 0.70, 0.85, 0.95, 1.0])
    thresh = np.array([np.inf, 3.0, 2.0, 1.0, 0.5, -np.inf])
    v_tpr, _ = tpr_at_fpr(fpr, tpr, thresh, 0.05)
    assert abs(v_tpr - 0.50) < 1e-10
    v_tpr, _ = tpr_at_fpr(fpr, tpr, thresh, 0.075)
    assert abs(v_tpr - 0.60) < 1e-10  # halfway between 0.50 and 0.70
    v_fpr, _ = fpr_at_tpr(fpr, tpr, thresh, 0.85)
    assert abs(v_fpr - 0.20) < 1e-10
    print("  operating-point interpolation: OK")
    
    # Test 6: Ranking metrics
    # Query with 3 candidates, golds at positions 1 and 3 (after ranking)
    scores = np.array([3.0, 2.0, 1.0])
    labels = np.array([1, 0, 1])  # gold at rank 1 and rank 3
    rm = ranking_metrics_for_query(scores, labels, k_values=[1, 2, 3])
    assert rm['p_at_1'] == 1.0
    assert rm['mrr'] == 1.0
    assert rm['recall_at_1'] == 0.5  # 1 of 2 golds in top-1
    assert rm['recall_at_2'] == 0.5  # 1 of 2 golds in top-2
    assert rm['recall_at_3'] == 1.0  # 2 of 2 golds in top-3
    print(f"  ranking metrics on toy query: OK (P@1=1.0, R@3=1.0)")
    
    # Test 7: Bootstrap reproducibility
    idx1 = query_clustered_bootstrap_indices(10, 100, 42)
    idx2 = query_clustered_bootstrap_indices(10, 100, 42)
    assert np.array_equal(idx1, idx2), "bootstrap not reproducible with same seed"
    idx3 = query_clustered_bootstrap_indices(10, 100, 99)
    assert not np.array_equal(idx1, idx3), "different seeds should give different samples"
    print("  bootstrap reproducibility: OK")
    
    # Test 8: Sigmoid
    assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-12
    print("  sigmoid: OK")
    
    # Test 9: ECE on perfect calibration
    # If predictions are uniformly random in [0,1] and labels are Bernoulli(p),
    # ECE should be ~0 when bin accuracy = bin mean confidence
    np.random.seed(42)
    n = 10000
    p = np.random.uniform(0, 1, n)
    y = (np.random.uniform(0, 1, n) < p).astype(int)  # perfectly calibrated
    ece, bins = ece_quantile_bins(p, y, 10)
    assert ece < 0.05, f"Perfect calibration ECE should be small, got {ece}"
    print(f"  ECE on perfectly-calibrated data: OK ({ece:.4f})")
    
    print("Self-tests passed.\n")


# =============================================================================
# Main entry point
# =============================================================================
def main():
    global N_BOOTSTRAP_RESAMPLES
    parser = argparse.ArgumentParser(
        description='ACSAC Step E Stage 2: MCC + operating-point analysis with bootstrap CIs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--per-pair-csv', default='eval_results_v2/operating_points/per_pair_scores.csv',
                        help='Path to per_pair_scores.csv (default: %(default)s)')
    parser.add_argument('--output-dir', default='eval_results_v2/operating_points',
                        help='Output directory (default: %(default)s)')
    parser.add_argument('--n-resamples', type=int, default=N_BOOTSTRAP_RESAMPLES,
                        help=f'Bootstrap resample count (default: %(default)s)')
    parser.add_argument('--no-figures', action='store_true',
                        help='Skip figure generation (faster for testing)')
    parser.add_argument('--self-test', action='store_true',
                        help='Run self-tests on inferential primitives and exit.')
    args = parser.parse_args()
    N_BOOTSTRAP_RESAMPLES = args.n_resamples
    
    if args.self_test:
        _self_test()
        return 0
    
    print('='*75)
    print('ACSAC 2026 Step E — Stage 2: MCC + Operating-Point Analysis')
    print('='*75)
    print(f"  per-pair csv: {args.per_pair_csv}")
    print(f"  output_dir:   {args.output_dir}")
    print(f"  bootstrap:    {N_BOOTSTRAP_RESAMPLES} resamples, seed={BOOTSTRAP_SEED}")
    print()
    
    if not os.path.isfile(args.per_pair_csv):
        print(f"ERROR: per-pair CSV not found at {args.per_pair_csv}")
        print(f"Run score_test_pairs.py first.")
        return 1
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # [1/5] Load CSV
    print("[1/5] Loading per-pair scores CSV...")
    rows = load_per_pair_csv(args.per_pair_csv)
    print(f"  Loaded {len(rows)} pair-rows.")
    per_query = group_by_query(rows)
    print(f"  Grouped into {len(per_query)} unique queries.")
    
    # [2/5] Analyze
    print()
    print("[2/5] Computing all metrics with query-clustered bootstrap CIs...")
    results, figure_curves = analyze(per_query, args.output_dir, verbose=True)
    
    # [3/5] Write inferential JSON
    print()
    print("[3/5] Writing inferential_results.json...")
    out_json = os.path.join(args.output_dir, 'inferential_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  Wrote {out_json}")
    
    # [4/5] Figures
    if not args.no_figures:
        print()
        print("[4/5] Generating figures...")
        from acsac_e_figures import (
            fig1_roc_pr, fig2_operating_points, fig3_per_actor_mcc,
            fig4_calibration_reliability,
        )
        figdir = os.path.join(args.output_dir, 'figures')
        os.makedirs(figdir, exist_ok=True)
        figure_data = {}
        figure_data['fig1'] = fig1_roc_pr(figure_curves, results, os.path.join(figdir, 'e_fig1_roc_pr_curves'))
        figure_data['fig2'] = fig2_operating_points(results, os.path.join(figdir, 'e_fig2_operating_points'))
        figure_data['fig3'] = fig3_per_actor_mcc(results, os.path.join(figdir, 'e_fig3_per_actor_mcc'))
        figure_data['fig4'] = fig4_calibration_reliability(results, os.path.join(figdir, 'e_fig4_calibration_reliability'))
        with open(os.path.join(figdir, 'figure_data.json'), 'w', encoding='utf-8') as f:
            json.dump(figure_data, f, indent=2, default=float)
        print(f"  Figures written to {figdir}")
    else:
        print()
        print("[4/5] Skipping figures (--no-figures).")
    
    # [5/5] Provenance + summary
    print()
    print("[5/5] Writing provenance and summary...")
    provenance = {
        'script': 'mcc_fpr_analysis.py',
        'step': 'E',
        'stage': '2 of 2 (analysis + figures)',
        'bootstrap_seed': BOOTSTRAP_SEED,
        'bootstrap_n_resamples': N_BOOTSTRAP_RESAMPLES,
        'ci_level': CI_LEVEL,
        'tpr_at_fpr_targets': TPR_AT_FPR_TARGETS,
        'fpr_at_tpr_targets': FPR_AT_TPR_TARGETS,
        'p_at_1_layer': 'cross-encoder only (94.52%); not including hierarchical post-processing',
    }
    with open(os.path.join(args.output_dir, 'provenance_step_e_stage2.json'),
              'w', encoding='utf-8') as f:
        json.dump(provenance, f, indent=2)
    
    write_summary_md(results, args.output_dir)
    
    print()
    print(f"Step E Stage 2 COMPLETE. Outputs in: {args.output_dir}")
    return 0


def write_summary_md(results, output_dir):
    """Write a paper-ready markdown summary of the Step E results."""
    meta = results['meta']
    tf = results['threshold_free']
    pl = results['pair_level_at_tau_0']
    qw = results['query_weighted_at_tau_0']
    rm = results['ranking_metrics']
    op = results['operating_points']
    cal = results['calibration']
    
    md = []
    md.append("# Step E — MCC + Operating-Point Analysis (Cross-Encoder Layer)\n")
    md.append(f"**Layer**: cross-encoder only (94.52% P@1, system-level with hierarchical post-processing is 95.21%).\n")
    md.append(f"**Test set**: {meta['n_queries']} queries, {meta['n_pairs']} (query, candidate) pairs.\n")
    md.append(f"**Imbalance**: {meta['imbalance_neg_per_pos']:.2f}:1 (negatives:positives) at the pair level.\n")
    md.append(f"**Threshold strategy**: {meta['threshold_strategy']}.\n")
    md.append(f"**Bootstrap**: {meta['bootstrap']['method']}, n={meta['bootstrap']['n_resamples']}, seed={meta['bootstrap']['seed']}, {meta['bootstrap']['level']*100:.0f}% CI.\n")
    md.append("\n## Threshold-Free Metrics\n")
    md.append(f"- AUROC = {tf['auroc']['point']:.4f}  [95% CI: {tf['auroc']['ci_lo']:.4f}, {tf['auroc']['ci_hi']:.4f}]\n")
    md.append(f"- AUPRC = {tf['auprc']['point']:.4f}  [95% CI: {tf['auprc']['ci_lo']:.4f}, {tf['auprc']['ci_hi']:.4f}]\n")
    md.append("\nNote: per Krzyzinski et al. (NeurIPS 2024), AUPRC is not categorically superior to AUROC under class imbalance. Both are reported for completeness.\n")
    
    md.append("\n## Query-Weighted Metrics at tau = 0 (PRIMARY)\n")
    md.append("Macro-average over 146 queries.\n")
    md.append(f"- MCC                = {qw['mcc']['point']:.4f}  [{qw['mcc']['ci_lo']:.4f}, {qw['mcc']['ci_hi']:.4f}]\n")
    md.append(f"- F1                 = {qw['f1']['point']:.4f}  [{qw['f1']['ci_lo']:.4f}, {qw['f1']['ci_hi']:.4f}]\n")
    md.append(f"- Precision          = {qw['precision']['point']:.4f}  [{qw['precision']['ci_lo']:.4f}, {qw['precision']['ci_hi']:.4f}]\n")
    md.append(f"- Recall             = {qw['recall']['point']:.4f}  [{qw['recall']['ci_lo']:.4f}, {qw['recall']['ci_hi']:.4f}]\n")
    md.append(f"- Specificity        = {qw['specificity']['point']:.4f}  [{qw['specificity']['ci_lo']:.4f}, {qw['specificity']['ci_hi']:.4f}]\n")
    md.append(f"- Balanced accuracy  = {qw['balanced_accuracy']['point']:.4f}  [{qw['balanced_accuracy']['ci_lo']:.4f}, {qw['balanced_accuracy']['ci_hi']:.4f}]\n")
    
    md.append("\n## Pair-Level Metrics at tau = 0 (sensitivity)\n")
    md.append("Pooled across all pairs (queries with more golds contribute more positive pairs).\n")
    md.append(f"- MCC                = {pl['mcc']['point']:.4f}  [{pl['mcc']['ci_lo']:.4f}, {pl['mcc']['ci_hi']:.4f}]\n")
    md.append(f"- F1                 = {pl['f1']['point']:.4f}  [{pl['f1']['ci_lo']:.4f}, {pl['f1']['ci_hi']:.4f}]\n")
    md.append(f"- Precision          = {pl['precision']['point']:.4f}  [{pl['precision']['ci_lo']:.4f}, {pl['precision']['ci_hi']:.4f}]\n")
    md.append(f"- Recall             = {pl['recall']['point']:.4f}  [{pl['recall']['ci_lo']:.4f}, {pl['recall']['ci_hi']:.4f}]\n")
    cm = pl['confusion_matrix']
    md.append(f"- Confusion matrix (pair-level): TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}\n")
    
    md.append("\n## Operating-Point Table\n")
    md.append("| Operating Point | Value | 95% CI | Threshold (logit) |\n")
    md.append("|---|---|---|---|\n")
    for tfpr in TPR_AT_FPR_TARGETS:
        d = op[f'tpr_at_fpr_{tfpr}']
        md.append(f"| TPR @ FPR={tfpr:.2f} | {d['tpr']:.4f} | [{d['tpr_ci_lo']:.4f}, {d['tpr_ci_hi']:.4f}] | {d['threshold_logit']:.3f} |\n")
    for ttpr in FPR_AT_TPR_TARGETS:
        d = op[f'fpr_at_tpr_{ttpr}']
        md.append(f"| FPR @ TPR={ttpr:.2f} | {d['fpr']:.4f} | [{d['fpr_ci_lo']:.4f}, {d['fpr_ci_hi']:.4f}] | {d['threshold_logit']:.3f} |\n")
    
    md.append("\n## Ranking Metrics (cross-encoder layer)\n")
    md.append("Macro-average over 146 queries. These do NOT include hierarchical post-processing (the system-level P@1 of 95.21% comes from adding that downstream component).\n")
    md.append(f"- P@1   = {rm['p_at_1']['point']:.4f}  [{rm['p_at_1']['ci_lo']:.4f}, {rm['p_at_1']['ci_hi']:.4f}]\n")
    md.append(f"- MRR   = {rm['mrr']['point']:.4f}  [{rm['mrr']['ci_lo']:.4f}, {rm['mrr']['ci_hi']:.4f}]\n")
    for k in RANK_K_VALUES:
        md.append(f"- R@{k}  = {rm[f'recall_at_{k}']['point']:.4f}  [{rm[f'recall_at_{k}']['ci_lo']:.4f}, {rm[f'recall_at_{k}']['ci_hi']:.4f}]\n")
    for k in RANK_K_VALUES:
        md.append(f"- nDCG@{k} = {rm[f'ndcg_at_{k}']['point']:.4f}  [{rm[f'ndcg_at_{k}']['ci_lo']:.4f}, {rm[f'ndcg_at_{k}']['ci_hi']:.4f}]\n")
    
    md.append("\n## Per-Actor Descriptive Breakdown (no CIs; small strata)\n")
    md.append("| Actor | n_queries | P@1 | MCC (qw) | F1 (qw) | AUROC | AUPRC |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for actor, d in results['per_actor'].items():
        md.append(f"| {actor} | {d['n_queries']} | {d['p_at_1']:.3f} | {d['mcc_query_weighted']:.3f} | {d['f1_query_weighted']:.3f} | {d['auroc']:.3f} | {d['auprc']:.3f} |\n")
    
    md.append("\n## Calibration (appendix)\n")
    md.append(f"- Brier score = {cal['brier']['point']:.4f}  [{cal['brier']['ci_lo']:.4f}, {cal['brier']['ci_hi']:.4f}]\n")
    md.append(f"- ECE ({cal['n_bins']} {cal['binning_method']} bins) = {cal['ece']['point']:.4f}  [{cal['ece']['ci_lo']:.4f}, {cal['ece']['ci_hi']:.4f}]\n")
    md.append("\nNote: calibration computed on raw sigmoid-of-logit scores. The cross-encoder was not trained with calibration as an objective; Platt scaling on held-out data could improve these metrics.\n")
    
    out_md = os.path.join(output_dir, 'step_e_summary.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(''.join(md))
    print(f"  Wrote {out_md}")


if __name__ == '__main__':
    sys.exit(main())
