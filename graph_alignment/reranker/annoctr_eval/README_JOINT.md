# Option 2 — Full Joint Retrain (final pre-declared attempt)

Reproduces best_two_stage_v2's own curriculum with ONE change: stage 2
trains on CTI-HAL-train + AnnoCTR-train combined (19,183 pairs).
  Stage 1: ms-marco base -> tumeteor+zenodo (351,515 rows, the exact
           historical files, from the repo clone), 3 epochs @ 2e-5
  Stage 2: stage-1 best -> joint set, 2 epochs @ 1e-5, union validator
Ends with THE ONE pre-declared test look: CTI-HAL test (146) scored once,
verdict vs bar >=136. PASS -> local AnnoCTR test certification
(bar: overall P@1 >= 0.729). FAIL either -> permanent fallback
(two-checkpoint framing); the candidate is archived, never canonical.
This is the last test look for this model family.

## A. Your machine (~1 min)
Kit's 3 files go into
  C:\Users\shane\Downloads\CTI\graph_alignment\reranker\annoctr_eval\
then:
  cd C:\Users\shane\Downloads\CTI\graph_alignment\reranker
  python annoctr_eval\build_joint_bundle.py
-> annoctr_eval\joint_colab_bundle.zip (~5 MB)

## B. Colab (~50-60 min on T4 — keep the tab alive; stage 1 is the long part)
New or reused notebook -> Runtime -> T4 GPU. Upload joint_colab_bundle.zip.
One cell:

  !unzip -o joint_colab_bundle.zip -d /content/joint
  !pip -q install sentence-transformers rank_bm25
  !python /content/joint/joint_retrain_colab.py

Milestones to watch: "stage-1 sources verified ... 351515 rows",
"CTI-HAL split 1111/135/146 and 10,727 pairs reproduced", stage-1
progress bar (~30-35 min), stage-2 bar (~8 min), the tuning readouts,
then the boxed verdict line: "CTI-HAL test: N/146 ... PASS/FAIL".

## C. Bring back
  /content/joint_results.json          -> project files
  /content/joint_candidate.zip         -> download; extract ONLY on PASS,
                                          and only after I read the json.
On PASS, the local AnnoCTR certification command comes next (uses the
existing cert files; ~25 min CPU).
