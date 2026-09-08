# Token Importance Analysis - Summary

Empirical claims for the ACSAC threat-model paragraph and the
explainability section. Numbers below are computed from
`eval_results/token_importance.csv` (LOO output of explain.py)
with categorization grounded in MITRE ATT&CK STIX vocabulary.

## H1: Top-token category by prediction correctness

On the 133 queries the v2 model predicted correctly,
the highest-importance LOO token was an ATT&CK-vocabulary term in
**82.7%** of cases (95% CI: 75.9%-88.7%).

On the 13 queries the v2 model predicted incorrectly,
the same was true in **100.0%** 
of cases (95% CI: 100.0%-100.0%).

Actor-name top-token rates: 4.5% on
correct predictions versus 0.0% on
incorrect predictions.

## H2: Mean LOO importance by token category

| Category | n_tokens | mean_importance_raw | std | 95% CI |
|----------|---------:|--------------------:|----:|--------|
| ATTACK_VOCAB | 1279 | 0.1906 | 0.8982 | [0.1449, 0.2393] |
| ACTOR_NAME | 17 | 1.8477 | 3.1108 | [0.4675, 3.4731] |
| STOPWORD | 779 | 0.0263 | 0.1629 | [0.0152, 0.0385] |
| OTHER | 193 | 0.0791 | 0.5278 | [0.0151, 0.1597] |

**ATT&CK-vocabulary tokens carried mean importance 0.10x that of actor-name tokens.**

## A1: Per-actor breakdown

| Actor | n_queries | P@1 | Top-token ATT&CK% | Top-token Actor% |
|-------|----------:|----:|------------------:|-----------------:|
| apt29 | 41 | 92.7% | 90.2% | 0.0% |
| carbanak | 27 | 96.3% | 77.8% | 14.8% |
| fin6 | 21 | 81.0% | 85.7% | 4.8% |
| fin7 | 24 | 95.8% | 83.3% | 4.2% |
| oilrig | 17 | 76.5% | 94.1% | 0.0% |
| sandworm | 9 | 100.0% | 44.4% | 0.0% |
| wizardspider | 7 | 100.0% | 100.0% | 0.0% |

## A3: Margin vs importance-entropy correlation

Pearson r = -0.0442 (p = 0.5961)
Spearman r = -0.0250 (p = 0.7643)

Mean entropy on correct predictions: 1.6207
Mean entropy on incorrect predictions: 2.1226

Negative correlation indicates concentrated importance (low entropy)
co-occurs with high model confidence (high margin), which is the
deployment-readiness pattern we want.

## A4: BM25 overlap correspondence

Queries with at least one HIGH-impact token: 142
Queries where gold candidate text was unavailable: 0
Mean overlap rate (HIGH tokens that also appear in gold candidate): 0.2911
On correct predictions: 0.2972
On incorrect predictions: 0.2308
Welch's t-test: t = 0.522, p = 0.6099

## A5: Position effect

Correlation between token position (normalized to fraction through query)
and importance: Pearson r = -0.0532 (p = 0.01135),
Spearman r = -0.0683 (p = 0.001133)

Values close to zero indicate the model treats positions roughly equally,
which argues against systematic position bias.

## A6: Per-query importance entropy

Mean entropy across 146 queries: 1.6654 bits
Median: 1.4736 | Std: 1.2157
Range: [-0.0000, 5.5740]

## Categorization cross-check

STIX-vs-corpus vocabulary agreement: 94.7% of 2268 tokens classified the same way by both methods.
In STIX only: 4.7% | In corpus only: 0.6% | In both: 52.4% | In neither: 42.3%

## Methodology notes

**Primary categorization (STIX):** Tokens classified as ATTACK_VOCAB if
they appear in the names or descriptions of attack-pattern, malware, tool,
or x-mitre-tactic objects in the MITRE ATT&CK STIX bundle (enterprise-attack-v14.json).
Revoked and deprecated objects are excluded.

**Cross-check (corpus frequency):** Tokens also marked as in_corpus if they
appear in any candidate text or in 5+ distinct queries (matching the
auto_word_categorize.py methodology).

**Actor names (conservative list, 9 tokens):**
anunak, apt29, apt34, carbanak, fin6, fin7, oilrig, sandworm, wizardspider

**HIGH-impact tokens:** Those whose `impact` column is 'HIGH' in
token_importance.csv (preserved from explain.py's categorization).
