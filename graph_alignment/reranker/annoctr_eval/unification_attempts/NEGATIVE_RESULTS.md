# Checkpoint Unification Attempts — Negative Results Record (2026-08-12)

Goal: a single checkpoint holding CTI-HAL test >= 136/146 (93.2%) while
staying within 2 points of annoctr_stage3_v1 on AnnoCTR. Two pre-declared
attempts were made; both failed the CTI-HAL test bar. Per the declared
protocol, best_two_stage_v2 remains the canonical pipeline checkpoint,
annoctr_stage3_v1 remains the AnnoCTR-adapted operating point, and this
model family receives no further test-set evaluations.

## Attempt 1 — interpolation sweep + rehearsal fine-tune (rehearsal_results.json)
- Weight interpolation v2<->s3, alpha 0.0-1.0 (0.05 refined): no alpha met
  both validation gates; best min-margin at alpha=0.95.
- Rehearsal from descendant, 25% replay (winner on validation): AnnoCTR dev
  81.2 overall / 73.8 technique; CTI-HAL val 86.7 (vs v2 85.9). Passed all
  validation gates. CTI-HAL TEST: 129/146 = 88.4 -> FAIL (bar 136).
- 45% replay: validation-equivalent to 25%; never tested (look conserved).
- Uniform soup(4): AnnoCTR dev 77.7, CTI-HAL val 88.2; below AnnoCTR gate.

## Attempt 2 — full joint curriculum retrain (joint_results.json)
- Faithful reproduction of best_two_stage_v2's curriculum (stage 1:
  tumeteor+zenodo 351,515 rows, 3 ep @ 2e-5; stage 2: 2 ep @ 1e-5) with one
  change: stage-2 data = CTI-HAL-train 10,727 + AnnoCTR-train 8,456.
- Readouts: AnnoCTR dev 80.1 overall / 74.4 technique; CTI-HAL val 87.4.
- CTI-HAL TEST (the declared final look): 134/146 = 91.8 -> FAIL (bar 136).
- AnnoCTR test: never evaluated (gate failed first).

## Interpretation
Every cleanly-trained candidate exceeded v2 on CTI-HAL *validation*
(86.7-88.2 vs 85.9) while trailing it on the 146-query *test* slice by
2-9 queries. This pattern is consistent with best_two_stage_v2 carrying a
small test-slice-specific advantage accumulated through lineage selection
(best -> best_two_stage -> best_two_stage_v2 were each selected partly on
those 146 queries) plus single-run training variance. The certified
two-operating-points framing in the paper stands; candidate weights are
archived offline and are not part of this repository.

Protocol: all tuning on AnnoCTR dev + CTI-HAL val only; one pre-declared
test look per attempt; verdicts final.
