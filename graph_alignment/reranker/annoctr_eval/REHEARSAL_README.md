# Rehearsal Kit — Single Unified Checkpoint (best_three_stage_v1 candidate)

One Colab session that runs the research-report plan end-to-end:
Stage A interpolation sweep (parent <-> descendant, alpha 0-1) +
Stage B rehearsal fine-tunes from the descendant (25% and 45% CTI-HAL
replay, interleaved) + Stage C soup/blend-backs and winner selection.
Everything is tuned on AnnoCTR dev + CTI-HAL val ONLY. No test set is
ever scored; CTI-HAL test keys are computed solely to prove the replay
data cannot leak into them.

Declared gates (relative, computed in-run):
  G1 AnnoCTR dev overall  >= descendant - 2.0 pts
  G2 AnnoCTR dev technique >= descendant - 2.0 pts
  G3 CTI-HAL val           >= parent    - 1.5 pts
Winner = gate-passer maximizing min(margin_old, margin_new).
No gate-passer -> nothing is zipped; the pre-declared fallback is the
two-checkpoint framing already drafted in v18.

## A. Your machine (~1 min)
The kit's 3 files go into:
  C:\Users\shane\Downloads\CTI\graph_alignment\reranker\annoctr_eval\
Then:
  cd C:\Users\shane\Downloads\CTI\graph_alignment\reranker
  python annoctr_eval\build_rehearsal_bundle.py
It verifies the certified descendant checkpoint byte-size and writes:
  annoctr_eval\rehearsal_colab_bundle.zip   (~90 MB)

## B. Colab (~35-45 min on T4)
New notebook -> Runtime -> T4 GPU. Upload rehearsal_colab_bundle.zip
via the Files pane (left sidebar; ~90 MB, give it a minute). One cell:

  !unzip -o rehearsal_colab_bundle.zip -d /content/reh
  !pip -q install sentence-transformers rank_bm25
  !python /content/reh/rehearsal_unify_colab.py

What you'll see, in order: repo clone + parent checkpoint pull; the
CTI-HAL reproduction asserts (must print 1111/135/146 and 10,727);
baselines for parent and descendant on both validation sets; the
alpha=0/alpha=1 endpoint checks; ~13 interpolation evals; two training
runs with progress bars; soup/blend evals; the candidate table and
either "WINNER: ..." or the no-gate-passer message.

## C. Bring back
1. /content/rehearsal_results.json  -> put it in the project files
   (and a copy at annoctr_eval\stage3\rehearsal_results.json).
2. /content/best_three_stage_v1_CANDIDATE.zip (only exists if a winner
   passed) -> download it but DO NOT extract yet. I read the results
   json first and confirm the winner; then it goes to
   checkpoints\best_three_stage_v1_CANDIDATE\ and we schedule the
   one-time test certification.
