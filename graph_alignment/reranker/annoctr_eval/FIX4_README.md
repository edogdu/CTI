# Fix 4 Kit — Stage-3 adaptation on AnnoCTR train

## What this does
Trains a NEW checkpoint (annoctr_stage3_v1) on AnnoCTR's 67 training
documents (fully disjoint from dev and test — proven at build time), using
the locked mf query format and the exact stage-2 recipe at the curriculum's
next learning rate (5e-6, 1 epoch, seed 42). best_two_stage_v2 is never
modified. The Colab run also produces, automatically:
  - AnnoCTR dev (mf) eval: base vs adapted, per category
  - CTI-HAL forgetting probe: base (sanity: must print 138/146) vs adapted

## Part A — on your machine (~1 min)
File annoctr_train_pairs.py goes into:
  C:\Users\shane\Downloads\CTI\graph_alignment\reranker\annoctr_eval\
Then:
  cd C:\Users\shane\Downloads\CTI\graph_alignment\reranker
  python annoctr_eval\annoctr_train_pairs.py --annoctr-dir C:\Users\shane\Downloads\annoctr\AnnoCTR --stix enterprise-attack-v14.json
It prints the disjointness proof and writes ONE upload file:
  annoctr_eval\stage3\stage3_colab_bundle.zip   (~4.4 MB)

## Part B — on Colab (~15-20 min)
1. colab.research.google.com -> New notebook -> Runtime -> Change runtime
   type -> T4 GPU.
2. Files pane (left sidebar) -> upload stage3_colab_bundle.zip to /content.
3. Paste and run this one cell:

   !unzip -o stage3_colab_bundle.zip -d /content/stage3
   !pip -q install sentence-transformers rank_bm25
   !python /content/stage3/stage3_annoctr_colab.py

4. When it finishes, download from the Files pane:
     /content/stage3_results.json            (paste or upload to me)
     /content/annoctr_stage3_v1_best.zip     (~90 MB — the adapted checkpoint)

## Part C — back on your machine
Extract the checkpoint zip CONTENTS into a NEW folder:
  C:\Users\shane\Downloads\CTI\graph_alignment\reranker\checkpoints\annoctr_stage3_v1\
(so that model.safetensors sits directly inside annoctr_stage3_v1\)
Keep stage3_results.json anywhere convenient and send it to me.

## What we read from the results
- annoctr_dev_adapted vs annoctr_dev_base: did Fix 4 stack on mf?
  (base dev-mf reference from the sweep: overall 0.6021 / technique 0.4353)
- ctihal_base must be 138/146 (harness sanity); ctihal_adapted tells us the
  forgetting cost, which becomes a paper table either way.
- Then, and only then, ONE test-split certification run of the winner.
