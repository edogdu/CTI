# AnnoCTR Dev Sweep Kit — technique-score fixes (Fixes 1–3)

## What this is
One command that tests all combinations of the three cheap fixes on the DEV
split (573 queries) with your real checkpoint, and prints a comparison table.
The model is never modified. The test split is never touched (the script
refuses it).

Configs: base | cw | mf | ep | cw_mf | cw_ep | mf_ep | cw_mf_ep
  cw = context window   -> neighboring sentences added to each question
                           (capped 300 words, mention-sentence kept whole)
  mf = mention first    -> annotated phrase(s) prepended: "phrase — sentence"
  ep = enriched cards   -> technique cards get a real procedure example
                           appended ("Example: Indrik Spider used ...");
                           517/625 techniques have one; cards actually get
                           SHORTER (600-char desc + example vs 1000-char desc)

Controlled design: all 8 configs score the SAME candidate pools per query
(frozen from baseline) — only texts differ, so P@1 deltas are attributable
to the fixes. The "BM25R@20" column separately shows what each variant's
retrieval recall would be with rebuilt pools (system view).

## Files
  annoctr_to_reranker.py   builder v2 (adds the three flags + pool freezing;
                           no-flag output verified identical to the v1 test set)
  annoctr_eval.py          unchanged runner (imported by the sweep)
  annoctr_dev_sweep.py     the orchestrator (build + score + table)
  README_SWEEP.md          this file

## One-time install (if missing)
  pip install rank_bm25

## Run (from C:\Users\shane\Downloads\CTI\graph_alignment\reranker)
  python annoctr_eval\annoctr_dev_sweep.py --annoctr-dir <ANNOCTR>\AnnoCTR --stix enterprise-attack-v14.json --checkpoint checkpoints\best_two_stage_v2
where <ANNOCTR> is your clone of github.com/boschresearch/anno-ctr-lrec-coling-2024
(git clone it next to the CTI folder if you don't have it).

Runtime: ~3 min of builds + ~25-35 min of CPU scoring. Walk away.

## Send back
  annoctr_eval\sweep_dev\sweep_results_dev.json
  annoctr_eval\sweep_dev\sweep_table_dev.txt
(or just paste the printed table)

## Reading the table
- "tech P@1 d(base)" is the target number: the technique-category delta.
- Positive deltas on tech P@1 with stable software/tactic numbers = a fix
  that helps where we need it without breaking what already works.
- Known behaviors, so nothing surprises you:
  * mf duplicates long technique mentions (AnnoCTR technique mentions can be
    30-word spans) — the table decides if that helps or hurts.
  * cw raises pair length; 1.2% of pairs exceed the model's 512-token limit
    and get tail-truncated (vs 0.3% at base) — same truncation regime the
    model always ran under.
  * BM25R@20 is computed fresh per variant (real data even in --self-test).
- Decision rule agreed: pick the winning combo on dev; the test split gets
  exactly one certification run of the locked config at the very end.
