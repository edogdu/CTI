# Step E — MCC + Operating-Point Analysis (Cross-Encoder Layer)
**Layer**: cross-encoder only (94.52% P@1, system-level with hierarchical post-processing is 95.21%).
**Test set**: 146 queries, 2937 (query, candidate) pairs.
**Imbalance**: 6.09:1 (negatives:positives) at the pair level.
**Threshold strategy**: tau = 0 (raw logit, equivalent to sigmoid 0.5; Path A).
**Bootstrap**: query-clustered percentile, n=10000, seed=42, 95% CI.

## Threshold-Free Metrics
- AUROC = 0.9666  [95% CI: 0.9554, 0.9767]
- AUPRC = 0.8771  [95% CI: 0.8477, 0.9042]

Note: per Krzyzinski et al. (NeurIPS 2024), AUPRC is not categorically superior to AUROC under class imbalance. Both are reported for completeness.

## Query-Weighted Metrics at tau = 0 (PRIMARY)
Macro-average over 146 queries.
- MCC                = 0.7801  [0.7451, 0.8130]
- F1                 = 0.7937  [0.7608, 0.8242]
- Precision          = 0.7886  [0.7497, 0.8253]
- Recall             = 0.8595  [0.8207, 0.8962]
- Specificity        = 0.9540  [0.9453, 0.9624]
- Balanced accuracy  = 0.9068  [0.8873, 0.9251]

## Pair-Level Metrics at tau = 0 (sensitivity)
Pooled across all pairs (queries with more golds contribute more positive pairs).
- MCC                = 0.7706  [0.7376, 0.8026]
- F1                 = 0.8032  [0.7746, 0.8304]
- Precision          = 0.7553  [0.7184, 0.7921]
- Recall             = 0.8575  [0.8189, 0.8941]
- Confusion matrix (pair-level): TP=355  FP=115  TN=2408  FN=59

## Operating-Point Table
| Operating Point | Value | 95% CI | Threshold (logit) |
|---|---|---|---|
| TPR @ FPR=0.01 | 0.6304 | [0.5084, 0.7316] | 3.364 |
| TPR @ FPR=0.05 | 0.8671 | [0.8274, 0.9036] | -0.248 |
| TPR @ FPR=0.10 | 0.9275 | [0.8983, 0.9569] | -2.155 |
| FPR @ TPR=0.80 | 0.0250 | [0.0164, 0.0387] | 1.395 |
| FPR @ TPR=0.90 | 0.0725 | [0.0475, 0.1035] | -1.384 |
| FPR @ TPR=0.95 | 0.1562 | [0.0860, 0.2780] | -3.127 |

## Ranking Metrics (cross-encoder layer)
Macro-average over 146 queries. These do NOT include hierarchical post-processing (the system-level P@1 of 95.21% comes from adding that downstream component).
- P@1   = 0.9452  [0.9041, 0.9795]
- MRR   = 0.9694  [0.9468, 0.9886]
- R@3  = 0.8365  [0.8003, 0.8717]
- R@5  = 0.9466  [0.9242, 0.9670]
- R@10  = 0.9889  [0.9777, 0.9977]
- nDCG@3 = 0.8963  [0.8667, 0.9244]
- nDCG@5 = 0.9328  [0.9104, 0.9536]
- nDCG@10 = 0.9497  [0.9313, 0.9661]

## Per-Actor Descriptive Breakdown (no CIs; small strata)
| Actor | n_queries | P@1 | MCC (qw) | F1 (qw) | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| apt29 | 41 | 0.951 | 0.797 | 0.813 | 0.962 | 0.886 |
| carbanak | 27 | 1.000 | 0.873 | 0.878 | 0.979 | 0.938 |
| fin6 | 21 | 0.905 | 0.724 | 0.734 | 0.962 | 0.842 |
| fin7 | 24 | 1.000 | 0.828 | 0.836 | 0.985 | 0.918 |
| oilrig | 17 | 0.824 | 0.664 | 0.697 | 0.947 | 0.800 |
| sandworm | 9 | 0.889 | 0.778 | 0.784 | 0.965 | 0.812 |
| wizardspider | 7 | 1.000 | 0.613 | 0.636 | 0.936 | 0.805 |

## Calibration (appendix)
- Brier score = 0.0460  [0.0396, 0.0527]
- ECE (10 quantile bins) = 0.0334  [0.0266, 0.0426]

Note: calibration computed on raw sigmoid-of-logit scores. The cross-encoder was not trained with calibration as an objective; Platt scaling on held-out data could improve these metrics.
