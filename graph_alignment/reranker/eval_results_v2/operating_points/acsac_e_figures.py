"""
ACSAC 2026 Step E — Figure Generators

Four figures, each rendered as PDF + SVG + PNG for the paper.

Figure 1: ROC and PR curves side by side with query-clustered bootstrap CI
          bands. AUROC and AUPRC point estimates plus 95% CIs in legend.
          Operating points marked on the ROC.

Figure 2: Operating-point bar chart. Left panel: TPR at fixed FPR ∈ {0.01,
          0.05, 0.10} with bootstrap 95% CI error bars. Right panel: FPR at
          fixed TPR ∈ {0.80, 0.90, 0.95} with CI error bars.

Figure 3: Per-actor MCC forest plot. Point estimates only — caption explicitly
          notes that per-actor CIs are not interpreted inferentially due to
          small per-stratum sample sizes.

Figure 4: Reliability diagram (appendix). Sigmoid-of-logit confidence on
          x-axis (0 to 1), empirical accuracy on y-axis. 10 quantile bins.
          The diagonal y = x represents perfect calibration.

Determinism: SVG output is rendered with hashsalt=42 for byte-identical
output across runs with same data; PDF and PNG are also deterministic at
the matplotlib level.

Visual conventions match Step C' and Step D figures:
- Title-case axis labels
- Footnote-size legend
- 'whitesmoke' grid, alpha 0.3
- High-resolution output (150 DPI for PNG)
"""
import os
import json
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Match the Step C' / Step D style conventions
plt.rcParams['svg.hashsalt'] = '42'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.color'] = 'whitesmoke'

# Color palette (consistent across all figures)
COLOR_PRIMARY = '#1f77b4'      # blue
COLOR_SECONDARY = '#ff7f0e'    # orange
COLOR_TERTIARY = '#2ca02c'     # green
COLOR_OPERATING = '#d62728'    # red (operating points)
COLOR_DIAGONAL = '#888888'     # gray (reference diagonal / no-skill)
COLOR_CI_BAND = '#1f77b4'      # same as primary, with low alpha

# Output formats (matches Step D pattern)
OUTPUT_FORMATS = ['pdf', 'svg', 'png']


def _save_all_formats(fig, stem):
    """Save figure to PDF, SVG, and PNG at consistent DPI."""
    for fmt in OUTPUT_FORMATS:
        fig.savefig(f'{stem}.{fmt}', dpi=150, bbox_inches='tight',
                    format=fmt)


# =============================================================================
# Figure 1: ROC + PR with bootstrap CI bands
# =============================================================================
def fig1_roc_pr(figure_curves, results, output_stem):
    """Two-panel ROC and PR curves with bootstrap CI bands.
    
    Args:
        figure_curves: dict from analyze() containing roc/pr full curves and
            interpolated CI band data.
        results: full inferential_results dict (for AUROC/AUPRC point + CI
            values in the legend).
        output_stem: file path without extension; we write .pdf, .svg, .png.
    
    Returns:
        Dict of plotted-data for figure_data.json provenance.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # ----- ROC panel -----
    roc = figure_curves['roc']
    fpr_grid = np.array(roc['fpr_grid'])
    tpr_mean = np.array(roc['tpr_mean'])
    tpr_lo = np.array(roc['tpr_lo'])
    tpr_hi = np.array(roc['tpr_hi'])
    fpr_full = np.array(roc['fpr_full'])
    tpr_full = np.array(roc['tpr_full'])
    
    # Main ROC curve (from full data, no bootstrap)
    ax_roc.plot(fpr_full, tpr_full, color=COLOR_PRIMARY, linewidth=2.0,
                label=None, zorder=3)
    # Bootstrap CI band (shaded)
    ax_roc.fill_between(fpr_grid, tpr_lo, tpr_hi, alpha=0.20, color=COLOR_CI_BAND,
                         label='95% query-clustered bootstrap CI', zorder=2)
    # Diagonal (random classifier reference)
    ax_roc.plot([0, 1], [0, 1], color=COLOR_DIAGONAL, linestyle='--',
                linewidth=1.0, alpha=0.6, label='random classifier', zorder=1)
    
    # Operating points (TPR at fixed FPR)
    op_x = []
    op_y = []
    op_labels = []
    for tfpr, color in zip([0.01, 0.05, 0.10],
                            [COLOR_OPERATING, COLOR_OPERATING, COLOR_OPERATING]):
        # Interpolate from full ROC at exact target FPR for marker placement
        tpr_at_target = float(np.interp(tfpr, fpr_full, tpr_full))
        op_x.append(tfpr)
        op_y.append(tpr_at_target)
        op_labels.append(f'FPR={tfpr}')
    ax_roc.scatter(op_x, op_y, s=80, color=COLOR_OPERATING, marker='o',
                    zorder=5, edgecolor='black', linewidth=0.8,
                    label='operating points')
    for x, y, lab in zip(op_x, op_y, op_labels):
        ax_roc.annotate(lab, (x, y), textcoords='offset points',
                         xytext=(10, -8), fontsize=8)
    
    # AUROC in legend
    auroc = results['threshold_free']['auroc']
    auroc_text = (f"AUROC = {auroc['point']:.4f} "
                  f"[{auroc['ci_lo']:.4f}, {auroc['ci_hi']:.4f}]")
    ax_roc.set_xlim(-0.01, 1.01)
    ax_roc.set_ylim(-0.01, 1.01)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title(f'(a) ROC curve\n{auroc_text}')
    ax_roc.legend(loc='lower right', fontsize=9)
    
    # ----- PR panel -----
    pr = figure_curves['pr']
    recall_grid = np.array(pr['recall_grid'])
    prec_mean = np.array(pr['precision_mean'])
    prec_lo = np.array(pr['precision_lo'])
    prec_hi = np.array(pr['precision_hi'])
    precision_full = np.array(pr['precision_full'])
    recall_full = np.array(pr['recall_full'])
    
    # Sort PR for clean plotting (compute_pr_curve returns reverse order)
    sort_idx = np.argsort(recall_full)
    ax_pr.plot(recall_full[sort_idx], precision_full[sort_idx],
                color=COLOR_SECONDARY, linewidth=2.0, zorder=3)
    ax_pr.fill_between(recall_grid, prec_lo, prec_hi, alpha=0.20,
                       color=COLOR_SECONDARY,
                       label='95% query-clustered bootstrap CI', zorder=2)
    
    # No-skill reference (positive prevalence)
    n_pos = results['meta']['n_positives']
    n_total = results['meta']['n_pairs']
    no_skill = n_pos / n_total
    ax_pr.axhline(no_skill, color=COLOR_DIAGONAL, linestyle='--', linewidth=1.0,
                   alpha=0.6, label=f'no-skill ({no_skill:.3f})', zorder=1)
    
    auprc = results['threshold_free']['auprc']
    auprc_text = (f"AUPRC = {auprc['point']:.4f} "
                  f"[{auprc['ci_lo']:.4f}, {auprc['ci_hi']:.4f}]")
    ax_pr.set_xlim(-0.01, 1.01)
    ax_pr.set_ylim(-0.01, 1.01)
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title(f'(b) Precision-Recall curve\n{auprc_text}')
    ax_pr.legend(loc='lower left', fontsize=9)
    
    fig.suptitle('Cross-encoder pair-relevance discrimination on '
                 'the CTI-HAL test set (n=146 queries)', fontsize=11, y=1.02)
    fig.tight_layout()
    _save_all_formats(fig, output_stem)
    plt.close(fig)
    
    return {
        'roc_point_estimates': {
            'fpr_full': fpr_full.tolist(),
            'tpr_full': tpr_full.tolist(),
        },
        'roc_ci_band': {
            'fpr_grid': fpr_grid.tolist(),
            'tpr_mean': tpr_mean.tolist(),
            'tpr_ci_lo': tpr_lo.tolist(),
            'tpr_ci_hi': tpr_hi.tolist(),
        },
        'auroc': auroc,
        'pr_point_estimates': {
            'precision_full': precision_full.tolist(),
            'recall_full': recall_full.tolist(),
        },
        'pr_ci_band': {
            'recall_grid': recall_grid.tolist(),
            'precision_mean': prec_mean.tolist(),
            'precision_ci_lo': prec_lo.tolist(),
            'precision_ci_hi': prec_hi.tolist(),
        },
        'auprc': auprc,
        'operating_points_on_roc': [
            {'fpr': x, 'tpr': y, 'label': l} for x, y, l in zip(op_x, op_y, op_labels)
        ],
    }


# =============================================================================
# Figure 2: Operating points
# =============================================================================
def fig2_operating_points(results, output_stem):
    """Two-panel bar chart: TPR@FPR on left, FPR@TPR on right.
    
    Each bar shows point estimate; error bars show bootstrap 95% CI.
    Annotation strategy: value goes immediately above the error-bar tip in
    bold, and the [CI] τ=value annotation goes one line below that in small
    font. Y-axis is sized so the annotations have room without stacking.
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5))
    op = results['operating_points']
    
    # ----- TPR @ FPR panel -----
    targets_fpr = [0.01, 0.05, 0.10]
    tpr_vals = []
    tpr_los = []
    tpr_his = []
    threshs = []
    for t in targets_fpr:
        d = op[f'tpr_at_fpr_{t}']
        tpr_vals.append(d['tpr'])
        tpr_los.append(d['tpr_ci_lo'])
        tpr_his.append(d['tpr_ci_hi'])
        threshs.append(d['threshold_logit'])
    
    x_pos = np.arange(len(targets_fpr))
    labels = [f'FPR={t}' for t in targets_fpr]
    err_lo = [max(v - l, 0) for v, l in zip(tpr_vals, tpr_los)]
    err_hi = [max(h - v, 0) for v, h in zip(tpr_vals, tpr_his)]
    bars = ax_l.bar(x_pos, tpr_vals, yerr=[err_lo, err_hi],
                     color=COLOR_PRIMARY, alpha=0.85, capsize=8,
                     edgecolor='black', linewidth=0.7)
    
    # Size the y-axis to give room for annotations above the bars.
    # When values are all near 1.0, we lift the top of the axis to 1.18.
    max_top = max(h for h in tpr_his)
    ax_l.set_ylim(0, max(1.18, max_top + 0.15))
    
    # Annotate: value above error-bar tip, CI+threshold just below as small caption
    for i, (v, lo, hi, th) in enumerate(zip(tpr_vals, tpr_los, tpr_his, threshs)):
        bar_top = hi  # top of error bar
        ax_l.text(i, bar_top + 0.06, f'{v:.3f}',
                   ha='center', fontsize=10, fontweight='bold')
        ax_l.text(i, bar_top + 0.025,
                   f'[{lo:.3f}, {hi:.3f}]  τ={th:.2f}',
                   ha='center', fontsize=7.5, color='#444444')
    
    ax_l.set_xticks(x_pos)
    ax_l.set_xticklabels(labels)
    ax_l.set_ylabel('True Positive Rate')
    ax_l.set_xlabel('Operating-point FPR')
    ax_l.set_title('(a) TPR at fixed low FPR\n(security/malware convention)')
    
    # ----- FPR @ TPR panel -----
    targets_tpr = [0.80, 0.90, 0.95]
    fpr_vals = []
    fpr_los = []
    fpr_his = []
    threshs2 = []
    for t in targets_tpr:
        d = op[f'fpr_at_tpr_{t}']
        fpr_vals.append(d['fpr'])
        fpr_los.append(d['fpr_ci_lo'])
        fpr_his.append(d['fpr_ci_hi'])
        threshs2.append(d['threshold_logit'])
    
    x_pos = np.arange(len(targets_tpr))
    labels = [f'TPR={t}' for t in targets_tpr]
    err_lo = [max(v - l, 0) for v, l in zip(fpr_vals, fpr_los)]
    err_hi = [max(h - v, 0) for v, h in zip(fpr_vals, fpr_his)]
    bars = ax_r.bar(x_pos, fpr_vals, yerr=[err_lo, err_hi],
                     color=COLOR_SECONDARY, alpha=0.85, capsize=8,
                     edgecolor='black', linewidth=0.7)
    
    # Auto-scale y axis based on data. Need room above error bar tip for the
    # value and CI/threshold annotation. Floor of 0.10 so very-low values
    # still have visible annotation room.
    max_top = max(fpr_his)
    y_top = max(0.10, max_top * 1.6 + 0.01)
    ax_r.set_ylim(0, y_top)
    
    for i, (v, lo, hi, th) in enumerate(zip(fpr_vals, fpr_los, fpr_his, threshs2)):
        bar_top = hi
        annot_y_value = bar_top + 0.04 * y_top
        annot_y_caption = bar_top + 0.015 * y_top
        ax_r.text(i, annot_y_value, f'{v:.3f}',
                   ha='center', fontsize=10, fontweight='bold')
        ax_r.text(i, annot_y_caption,
                   f'[{lo:.3f}, {hi:.3f}]  τ={th:.2f}',
                   ha='center', fontsize=7.5, color='#444444')
    
    ax_r.set_xticks(x_pos)
    ax_r.set_xticklabels(labels)
    ax_r.set_ylabel('False Positive Rate')
    ax_r.set_xlabel('Operating-point TPR')
    ax_r.set_title('(b) FPR at fixed high TPR\n(analyst-screening convention)')
    
    fig.suptitle('Operating-point analysis with query-clustered bootstrap 95% CIs',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _save_all_formats(fig, output_stem)
    plt.close(fig)
    
    return {
        'tpr_at_fpr': [
            {'target_fpr': t, 'tpr': v, 'ci_lo': lo, 'ci_hi': hi, 'threshold_logit': th}
            for t, v, lo, hi, th in zip(targets_fpr, tpr_vals, tpr_los, tpr_his, threshs)
        ],
        'fpr_at_tpr': [
            {'target_tpr': t, 'fpr': v, 'ci_lo': lo, 'ci_hi': hi, 'threshold_logit': th}
            for t, v, lo, hi, th in zip(targets_tpr, fpr_vals, fpr_los, fpr_his, threshs2)
        ],
    }


# =============================================================================
# Figure 3: Per-actor MCC forest plot
# =============================================================================
def fig3_per_actor_mcc(results, output_stem):
    """Per-actor MCC forest plot. No CIs (small strata).
    
    Shows query-weighted MCC and pair-level MCC side by side as a clustered
    horizontal bar chart, with the actor's n_queries annotated.
    """
    per_actor = results['per_actor']
    actors = sorted(per_actor.keys(), key=lambda a: -per_actor[a]['n_queries'])
    
    mcc_qw = [per_actor[a]['mcc_query_weighted'] for a in actors]
    mcc_pl = [per_actor[a]['mcc_pair_level'] for a in actors]
    n_qs = [per_actor[a]['n_queries'] for a in actors]
    
    y = np.arange(len(actors))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y - width / 2, mcc_qw, width, label='query-weighted',
            color=COLOR_PRIMARY, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.barh(y + width / 2, mcc_pl, width, label='pair-level',
            color=COLOR_SECONDARY, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Reference line at MCC = 1.0
    ax.axvline(1.0, color=COLOR_DIAGONAL, linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axvline(0.0, color='black', linewidth=0.5)
    
    # Annotate values
    for i, (v_qw, v_pl, n_q) in enumerate(zip(mcc_qw, mcc_pl, n_qs)):
        ax.text(v_qw + 0.01, i - width / 2, f'{v_qw:.3f}', va='center', fontsize=8)
        ax.text(v_pl + 0.01, i + width / 2, f'{v_pl:.3f}', va='center', fontsize=8)
    
    labels_with_n = [f'{a}\n(n={n})' for a, n in zip(actors, n_qs)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels_with_n)
    ax.set_xlim(-0.05, 1.1)
    ax.set_xlabel('Matthews Correlation Coefficient')
    ax.set_title('Per-actor MCC (descriptive only; per-stratum CIs not reported\n'
                 'due to small sample sizes — point estimates only)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    _save_all_formats(fig, output_stem)
    plt.close(fig)
    
    return {
        'actors': actors,
        'mcc_query_weighted': mcc_qw,
        'mcc_pair_level': mcc_pl,
        'n_queries_per_actor': n_qs,
    }


# =============================================================================
# Figure 4: Calibration reliability diagram
# =============================================================================
def fig4_calibration_reliability(results, output_stem):
    """Reliability diagram: mean confidence vs empirical accuracy per bin.
    
    Diagonal y = x represents perfect calibration. Points above the diagonal
    indicate under-confidence (model is more accurate than it claims); points
    below indicate over-confidence (model is less accurate than it claims).
    """
    cal = results['calibration']
    bins = cal['bin_data']
    if not bins:
        # Fallback: empty plot with a note
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.text(0.5, 0.5, 'No calibration data available',
                ha='center', va='center', fontsize=14)
        _save_all_formats(fig, output_stem)
        plt.close(fig)
        return {'note': 'no calibration data'}
    
    confidences = [b['mean_confidence'] for b in bins]
    accuracies = [b['accuracy'] for b in bins]
    counts = [b['count'] for b in bins]
    
    fig, (ax_rd, ax_hist) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # ----- Reliability diagram (left) -----
    # Perfect calibration diagonal
    ax_rd.plot([0, 1], [0, 1], color=COLOR_DIAGONAL, linestyle='--',
                linewidth=1.5, label='perfect calibration', zorder=1)
    
    # Plot actual calibration: mean confidence vs accuracy per bin
    # Size points by bin count
    sizes = [max(40, 200 * c / max(counts)) for c in counts]
    ax_rd.scatter(confidences, accuracies, s=sizes, color=COLOR_PRIMARY,
                  alpha=0.75, edgecolor='black', linewidth=0.8,
                  zorder=3, label='calibration bins (size ∝ count)')
    
    # Connect bins with a line to show the calibration trajectory
    ax_rd.plot(confidences, accuracies, color=COLOR_PRIMARY, alpha=0.5,
               linewidth=1.0, zorder=2)
    
    ax_rd.set_xlim(-0.02, 1.02)
    ax_rd.set_ylim(-0.02, 1.02)
    ax_rd.set_xlabel('Mean predicted confidence (sigmoid of logit)')
    ax_rd.set_ylabel('Empirical accuracy')
    ax_rd.set_title(f'(a) Reliability diagram ({cal["n_bins"]} quantile bins)\n'
                    f'ECE = {cal["ece"]["point"]:.4f}, '
                    f'Brier = {cal["brier"]["point"]:.4f}')
    ax_rd.legend(loc='upper left', fontsize=9)
    
    # ----- Histogram (right) -----
    bin_indices = list(range(len(bins)))
    ax_hist.bar(bin_indices, counts, color=COLOR_SECONDARY, alpha=0.75,
                 edgecolor='black', linewidth=0.7)
    ax_hist.set_xticks(bin_indices)
    ax_hist.set_xticklabels([f'B{i+1}' for i in bin_indices], fontsize=8)
    ax_hist.set_xlabel(f'Quantile bin (1 = lowest confidence, '
                       f'{cal["n_bins"]} = highest)')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('(b) Bin counts (quantile binning, equal sample size)')
    
    fig.suptitle('Cross-encoder probability calibration (appendix)',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _save_all_formats(fig, output_stem)
    plt.close(fig)
    
    return {
        'bin_count': len(bins),
        'mean_confidence_per_bin': confidences,
        'accuracy_per_bin': accuracies,
        'count_per_bin': counts,
        'ece': cal['ece'],
        'brier': cal['brier'],
    }
