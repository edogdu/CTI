# Data Licensing Notice — AnnoCTR-Derived Files

Files in this directory whose contents include text drawn from the AnnoCTR
corpus (per-query result CSVs, crosswalk reports, and any regenerated
`annoctr_*.jsonl` evaluation files) are derivative works of:

  Lukas Lange, Marc Mueller, Ghazaleh Haratinezhad Torbati, Dragan
  Milchevski, Patrick Grau, Subhash Chandra Pujari, Annemarie Friedrich.
  "AnnoCTR: A Dataset for Detecting and Linking Entities, Tactics, and
  Techniques in Cyber Threat Reports." LREC-COLING 2024, pp. 1147-1160.
  https://github.com/boschresearch/anno-ctr-lrec-coling-2024

The AnnoCTR corpus is licensed under Creative Commons Attribution-ShareAlike
4.0 International (CC-BY-SA 4.0). Accordingly, the AnnoCTR-derived DATA files
in this directory are likewise distributed under CC-BY-SA 4.0 with the
attribution above. This notice applies to those data files only; it does not
apply to the evaluation and training scripts in this directory, which are
original works of this repository's authors.

The large derived evaluation files (`annoctr_queries_*.jsonl`,
`annoctr_pairs_*.jsonl`, `annoctr_stage3_*.jsonl`) are intentionally not
committed; they regenerate bit-identically (deterministic, hash-pinned in
`build_provenance*.json` / `stage3_provenance.json`) via:

  python annoctr_to_reranker.py --annoctr-dir <AnnoCTR-clone>/AnnoCTR --stix ../enterprise-attack-v14.json --splits test dev
  python annoctr_train_pairs.py  --annoctr-dir <AnnoCTR-clone>/AnnoCTR --stix ../enterprise-attack-v14.json

MITRE ATT&CK content is used per the MITRE ATT&CK Terms of Use.
