# Step D — Counterfactual Probing of the v2 Cross-Encoder

This directory contains the full counterfactual-probing pipeline for the
ACSAC 2026 paper. It implements the locked-in design from the May 2026
five-point user confirmation: Wizard Spider flagged n/a (no Stratum A
queries), Carbanak handled as a paired case study (records #12 and #15),
primary analysis on all 21 Stratum A queries, CamelCase fallback Option A
(substitute kept in title-case), and the full inferential stack with
paired Δ-margin DiD as the primary endpoint.


## Folder setup (read this first)

This module expects to live in a subfolder called `counterfactual` under
your existing `eval_results_v2` directory, as a sibling of the
`categorization` subfolder produced by Step C'. The reason is that
`cf_figures.py` imports `paired_stats.py` using the relative path
`../categorization`, so the layout must be:

    eval_results_v2\
        categorization\          (already populated by Step C')
            paired_stats.py
            acsac_figures.py
            vocabulary_v14.json
            ...
        counterfactual\          (this folder — create it)
            cf_substitution.py
            counterfactual_probe.py
            cf_figures.py
            README.md

If you put these files directly inside `eval_results_v2\` without the
`counterfactual` subfolder, the import will fail at runtime with
`ModuleNotFoundError: No module named 'paired_stats'`, because the
relative path will resolve to a location that does not exist. Creating
the subfolder is therefore not optional.

If you do not already have a `categorization` subfolder containing
`paired_stats.py`, you need to populate it from Step C' before running
the inferential layer.


## File layout

The three Python modules below are run in order. Each one is independent
and can be re-executed without touching the others, provided its inputs
are still on disk.

    cf_substitution.py         — Substitution and pool-construction logic.
                                  Library code, not directly invoked.
                                  Has its own self-test suite (12 tests).

    counterfactual_probe.py    — Main pipeline driver. Loads the v2
                                  reranker pairs, builds the manifest of
                                  1,295 perturbation triples, scores them
                                  through the v2 cross-encoder, computes
                                  per-triple metrics, and writes the flat
                                  CSV.

    cf_figures.py              — Inferential layer + figure generation.
                                  Operates on the CSV produced above. No
                                  model loading; pure post-processing.

    README.md                  — This file.

After a full run, the directory will also contain:

    counterfactual_manifest.jsonl   — All 1,295 manifest entries with
                                      qn, intervention, substitute,
                                      perturbed_query, etc.
    counterfactual_results.csv      — Flat per-triple metrics (35 cols),
                                      including query_raw and
                                      perturbed_query for auditability.
    counterfactual_summary.md       — Auto-generated descriptive summary
                                      (paper-ready Markdown).
    inferential_results.json        — Primary DiD test, mid-p McNemar,
                                      LMM appendix coefficients, seeds.
    figures/
        cf_fig1_flip_rates_by_intervention.{pdf,svg,png}
        cf_fig2_swap_vs_placebo_paired.{pdf,svg,png}
        cf_fig3_gold_rank_change_distribution.{pdf,svg,png}
        cf_fig4_carbanak_case_study.{pdf,svg,png}
        figure_data.json            — Every numeric value plotted, so
                                      reviewers can audit any number
                                      against the source data.


## Running the pipeline (Windows cmd.exe)

The pipeline must be run from the inner reranker directory because
`counterfactual_probe.py` imports modules from `v2_reeval.py` for
`load_and_group_data()` and `create_test_split()` to guarantee
bit-identical test splits with the rest of the v2 evaluation work.

Step 1 — set deterministic environment variables (every shell session):

    set PYTHONHASHSEED=42
    set CUBLAS_WORKSPACE_CONFIG=:4096:8

Step 2 — verify the substitution layer is healthy. This runs the full
12-test self-test suite and exits non-zero on any failure. Should print
"All cf_substitution self-tests PASSED" at the end:

    cd <repo>\graph_alignment\reranker\graph_alignment\reranker\
    python eval_results_v2\counterfactual\cf_substitution.py

Step 3 — run the probe. With a Colab T4 or local GPU this takes roughly
15 to 20 minutes. The `--limit-queries` flag is useful for an initial
smoke run; remove it for the full corpus:

    python eval_results_v2\counterfactual\counterfactual_probe.py ^
        --data-path reranker_pairs_enriched_v2.jsonl ^
        --stix-path enterprise-attack-v14.json ^
        --vocab-path eval_results_v2\categorization\vocabulary_v14.json ^
        --output-dir eval_results_v2\counterfactual\

(In cmd.exe, `^` is the line-continuation character. PowerShell uses
the backtick instead, but the user's environment is cmd.exe.)

Step 4 — run the inferential layer and figure generation. This takes
under one minute on CPU only; no GPU needed:

    python eval_results_v2\counterfactual\cf_figures.py ^
        --results-csv eval_results_v2\counterfactual\counterfactual_results.csv ^
        --output-dir eval_results_v2\counterfactual\

Step 5 — visually audit the four figures. Open the PDFs and verify they
look correct. The `figure_data.json` sidecar contains the underlying
numbers.


## What the inferential layer does

Per the locked-in stack, the analysis proceeds in three layers:

PRIMARY — paired Δ-margin difference-in-differences. For each Stratum A
query and each of the four placebo contrasts (swap vs proper-noun, swap
vs generic-phrase, swap vs malware, swap vs soft-deletion), we compute
within-query `d_q = Δmargin_q^swap − Δmargin_q^placebo`, then test the
mean d_q across queries with a paired permutation test (B = 10,000) and
a BCa bootstrap 95% CI. Holm-Bonferroni correction is applied across
the four contrasts. This is reported as ΔΔ in Figure 2 and the JSON.

CO-PRIMARY — top-1 flip rate per intervention with Wilson 95% CI
(Figure 1), plus mid-p McNemar test on (correct_orig vs correct_pert).
When the discordant cell count `b + c < 5`, the test is explicitly
flagged as UNDERINFORMATIVE rather than reported as "n.s." — this is
the consensus across all eleven research documents. At our N = 21 with
≈ 95% baseline accuracy, several conditions will trigger this flag,
which is exactly why the continuous Δ-margin DiD is the primary
endpoint and not the binary flip.

APPENDIX — linear mixed-effects model on continuous Δ-margin via
`statsmodels.MixedLM`, with the formula
`gold_margin_delta ~ C(intervention) + C(actor) + subword_token_delta`,
random intercept by query, and proper-noun-placebo as the reference
level for intervention. The fit has a three-stage fallback: lbfgs → Powell
(derivative-free) → OLS without random intercept. The first two
sometimes fail with "Random effects covariance is singular" when query-
level variance is small relative to residual variance (a likely regime
at N = 21 with ceiling effects in accuracy). The OLS fallback gives
fixed-effect coefficients that are nearly identical to the LMM fixed
effects in that regime, so reviewers always see useful numbers.


## Determinism guarantees

Every random operation in the pipeline takes an explicit seed:

    SUBSTITUTE_SAMPLING_SEED = 42   (substitute selection)
    PERMUTATION_SEED         = 42   (paired permutation test)
    BOOTSTRAP_SEED           = 42   (BCa bootstrap CI)

Plus the environment variable `PYTHONHASHSEED = 42` enforces deterministic
SVG clip-path IDs in matplotlib ≥ 3.10. The cross-platform hash canary
(verified earlier) is `2944337262402990785`. Two separate runs on
identical inputs produce byte-identical CSV, JSON, PDF, and SVG outputs.


## Manifest counts (sanity check)

A successful run will print these counts at the start of step 4:

    Total scoring loops needed: 1295
        baseline:                146   (21 Stratum A + 125 Stratum B)
        swap:                    105   (21 × 5 substitutes)
        proper_noun_placebo:     105   (21 × 5 substitutes)
        generic_phrase_placebo:   63   (21 × 3 substitutes)
        malware_placebo:         105   (21 × 5 substitutes)
        soft_deletion:            21   (21 × 1)
        insertion_own_actor:     125   (Stratum B × 1 own-actor token)
        insertion_diff_actor:    625   (Stratum B × 5 different-actor tokens)

Stratum A breakdown by actor: apt29 = 7, carbanak = 5, fin6 = 4,
sandworm = 2, fin7 = 2, oilrig = 1. Wizard Spider has zero Stratum A
queries and is therefore flagged n/a in the analysis.


## Reproducing the smoke test (no model needed)

The probe accepts a `--skip-scoring` flag that halts after the manifest
is built. This is useful for debugging substitution logic without
loading the cross-encoder:

    python eval_results_v2\counterfactual\counterfactual_probe.py ^
        --data-path reranker_pairs_enriched_v2.jsonl ^
        --stix-path enterprise-attack-v14.json ^
        --vocab-path eval_results_v2\categorization\vocabulary_v14.json ^
        --output-dir eval_results_v2\counterfactual\ ^
        --skip-scoring

The smoke test produces only `counterfactual_manifest.jsonl`. The CSV
and figures cannot be produced without scoring.


## Citations

The methodology is grounded in a four-document portfolio (May 2026):

  - Yang, Levy, Goldberg, Wallace 2026 — "Compared to What?"
    arXiv:2605.01048. Establishes the placebo-as-control framework
    that anchors the four placebo contrasts.

  - Webber, Moffat, Zobel 2010 — Original RBO derivation. RBO@5 with
    p = 0.8 corresponds to the user-model `1 / (1 - p) = 5`
    interpretation (top-5 most influential).

  - Fagerland, Lydersen, Laake 2013 — Mid-p McNemar test. Establishes
    the b + c < 5 threshold below which no test variant is informative.

  - CausaLM (Feder, Oved, Shalit, Reichart 2021, Comp Ling 47(2)) —
    Counterfactual representation framework. NOTE: this is NOT
    "Wang et al."; the previous misattribution has been corrected.

The four ACSAC figures are formatted to match the conference's
double-blind submission style: PDF + SVG + PNG outputs with embedded
metadata, matplotlib ≥ 3.10 deterministic SVG clip-path IDs, and no
identifying information in the figure files themselves.
