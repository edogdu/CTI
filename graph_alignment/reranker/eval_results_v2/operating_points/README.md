# Step E — MCC + Operating-Point Analysis

ACSAC 2026 binary-classification analysis of the v2 cross-encoder on the
146-query CTI-HAL test set. Stage 1 extracts per-pair scores from the
cross-encoder; Stage 2 computes MCC, F1, AUROC, AUPRC, operating-point
analysis, ranking metrics, and calibration metrics, all with query-clustered
bootstrap 95% percentile confidence intervals.

## What layer of the system this analyzes

Every metric in this folder characterizes the cross-encoder's standalone
pair-relevance discrimination ability. The cross-encoder achieves 94.52% P@1
(138 correct out of 146) on its own. Hierarchical post-processing is a
separate downstream component that takes the cross-encoder's top-1 prediction
and adds its implied parent technique and parent tactic; it fixes one
T1027 → TA0005 error on the test set and gets the system to 95.21% P@1. That
component does not affect per-pair scores, ROC curves, MCC, F1, or any other
metric reported here, so Step E intentionally analyzes the cross-encoder
without it. The 95.21% number belongs in the main results section of the
paper, alongside the description of hierarchical post-processing as a
contribution. The 94.52% number belongs in Step E's binary-classification
section.

## Design decisions locked in for this build

The threshold strategy is Path A: every threshold-dependent metric is reported
at τ = 0 on the raw logit, which is equivalent to a sigmoid probability of
0.5 and corresponds to the natural decision boundary for a BCE-trained
classifier. This requires no validation-set tuning and is impervious to any
data-snooping objection.

Aggregation is query-weighted for the primary headline binary-classification
table (MCC, F1, precision, recall, balanced accuracy) — each of the 146
queries contributes equally to the macro-average regardless of how many gold
candidates it has. Pair-level aggregation is reported as a sensitivity check.
The two values can differ substantially when per-query gold counts are
heterogeneous, which is the case in our test set (mean 2.87 positives per
query, max 12).

The operating-point table covers both reviewer conventions: TPR at FPR ∈
{0.01, 0.05, 0.10} for the security/malware reviewer (the LiRA convention),
and FPR at TPR ∈ {0.80, 0.90, 0.95} for the analyst-screening reviewer.

Per-actor breakdown is descriptive only — point estimates of MCC, F1, P@1,
AUROC, and AUPRC by actor, with no confidence intervals reported because
wizardspider has only a handful of queries in the test set and per-stratum
bootstrap CIs would be misleading.

Calibration analysis is in the appendix — Brier score, ECE with 10 quantile
bins, reliability diagram. Quantile binning is used rather than equal-width
because at near-ceiling accuracy most predictions are concentrated near
sigmoid(logit) = 0 or sigmoid(logit) = 1, leaving equal-width bins mostly
empty in the middle.

## Files in this folder

The two scripts that produce everything are `score_test_pairs.py` (Stage 1)
and `mcc_fpr_analysis.py` (Stage 2). The figure-generation logic lives in
`acsac_e_figures.py`, which Stage 2 imports automatically. Stage 1 reads
`reranker_pairs_enriched_v2.jsonl` and the v2 model checkpoint and produces
`per_pair_scores.csv`. Stage 2 reads that CSV and produces
`inferential_results.json`, `step_e_summary.md`,
`provenance_step_e_stage2.json`, and four figures under `figures/`.

## How to run on Shane's machine

Both scripts assume they are invoked from the reranker root, which on Shane's
laptop is `C:\Users\shane\Downloads\CTI\graph_alignment\reranker`. Open
cmd.exe, navigate there, and confirm the canonical environment variables and
seeds before running.

The first run is Stage 1, the scoring extraction. This loads the v2
cross-encoder, scores every (query, candidate) pair in the test set, and
writes `per_pair_scores.csv`. Expected runtime is three to eight minutes on
the i7 CPU:

```
set PYTHONHASHSEED=42
python eval_results_v2\operating_points\score_test_pairs.py
```

At the end Stage 1 prints a sanity check that reports the observed P@1 at
top-1 prediction; this should be 0.9452 (matching the cross-encoder-only
number we expect from v2_unified_results.json). If it deviates by more than
1% the script prints a warning, which means either the model checkpoint
changed or the test split is not reproducible — investigate before running
Stage 2.

The second run is Stage 2, the analysis. This is pure CSV-and-numpy work
with no model in memory, and completes in roughly thirty seconds to two
minutes depending on whether figures are generated:

```
python eval_results_v2\operating_points\mcc_fpr_analysis.py
```

This produces `inferential_results.json` with every number for every metric
along with bootstrap CIs, `step_e_summary.md` with a paper-ready markdown
summary, and four PDF + SVG + PNG figures under `figures/`. The bootstrap
seed is hard-coded to 42, so running Stage 2 twice with the same input CSV
produces bit-identical inferential output (verified during build).

To run the self-tests on either script without doing the full analysis:

```
python eval_results_v2\operating_points\score_test_pairs.py --self-test
python eval_results_v2\operating_points\mcc_fpr_analysis.py --self-test
```

Both self-test suites verify the inferential primitives against sklearn's
reference implementations to within numerical precision and exercise the
boundary cases (zero-variance per-query MCC, all-positive or all-negative
labels, tied scores, perfect and worst-case classifiers, sigmoid extremes,
bootstrap reproducibility under fixed seed).

## What's in inferential_results.json

The JSON has a flat top-level structure with sections `meta`,
`threshold_free`, `pair_level_at_tau_0`, `query_weighted_at_tau_0`,
`operating_points`, `ranking_metrics`, `per_actor`, and `calibration`. Every
metric section follows the same shape: `point` is the value computed on the
full test set, `ci_lo` and `ci_hi` are the 2.5th and 97.5th percentiles of
the 10,000-resample query-clustered bootstrap distribution, `n_valid` is the
count of resamples that produced a finite metric value (inf and NaN values
are filtered before percentile computation). The `meta.bootstrap` subsection
records the seed and the wall-clock duration of the bootstrap loop.

The `operating_points` section reports both `value` and `threshold_logit` for
each operating point. The threshold is the cross-encoder logit at which the
target operating point is achieved on the global pair-level ROC. Threshold
CIs are also reported.

## The four figures

Figure 1 is the ROC + PR pair, each with a shaded query-clustered bootstrap
95% CI band. The legend reports AUROC and AUPRC with their CIs. Operating
points at FPR = {0.01, 0.05, 0.10} are marked on the ROC curve.

Figure 2 is the operating-point bar chart. The left panel shows TPR at fixed
low FPR (security/malware convention); the right panel shows FPR at fixed
high TPR (analyst-screening convention). Both have bootstrap CI error bars
and the threshold value annotated below each value.

Figure 3 is the per-actor MCC forest plot, descriptive only, showing both
query-weighted and pair-level MCC by actor with sample size in the label.
The caption explicitly notes that per-stratum CIs are not reported.

Figure 4 is the calibration appendix figure, showing the reliability diagram
on the left (with the diagonal y = x for perfect calibration) and the bin
count histogram on the right. The diagram title includes the ECE and Brier
score.

## Provenance and reproducibility

Both Stage 1 and Stage 2 write provenance JSON files:
`provenance_step_e_stage1.json` records the resolved data path, model path,
PYTHONHASHSEED environment variable, and the test split statistics;
`provenance_step_e_stage2.json` records the bootstrap seed, n_resamples,
operating-point targets, and the explicit note that this analysis is at the
cross-encoder layer (94.52% P@1) without hierarchical post-processing.

Bit-identical reproducibility of Stage 2 inferential output across two runs
on the same input CSV was verified during build using a deep-diff comparison.
The only field that differs between runs is `meta.bootstrap.elapsed_seconds`
which is wall-clock and intentionally not deterministic.

## Notes on the metric primitives

The AUROC, AUPRC, MCC, F1, precision, and recall computations were verified
against sklearn 1.8.0 to within 1e-6 across multiple seeds, sample sizes, and
positive-class rates. The ROC and PR curve computations also match sklearn's
output bit-identically except in degenerate-label cases (all-positive or
all-negative labels) where this code returns a sentinel curve and sklearn
raises. Bootstrap percentile CIs filter both NaN and inf values defensively
to handle the rare case where a resampled ROC has its requested operating
point in the synthetic-origin interval where the threshold is inf by
sklearn convention.

The per-query MCC returns 0.0 when undefined (a query where the model
predicts all-positive or all-negative for the whole candidate set), matching
sklearn's behavior. For our test data with mean 2.87 positives per query and
roughly 17 negatives per query, the undefined case is uncommon. The macro-
average over queries treats these 0.0 values as actual contributions, which
is the more conservative choice — an alternative would be to drop undefined
queries from the average, which is a different statistic with different
interpretation.
