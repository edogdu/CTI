# CTI-to-ATT&CK Neural Reranker

**95.21% Precision@1** on 146 test queries across 7 APT actors using a fine-tuned MiniLM cross-encoder with enriched ATT&CK descriptions and hierarchical post-processing. Winner of **Best Research Paper** at IEEE ICAIC 2026.

## Overview

This system maps unstructured cyber threat intelligence (CTI) passages to the MITRE ATT&CK framework by reformulating the problem as semantic retrieval rather than multi-label classification. A fine-tuned cross-encoder scores query-candidate pairs to determine whether a CTI passage describes the behavior captured by a given ATT&CK technique, tactic, or software entity.

The retrieval formulation provides two critical advantages over classification approaches like MITRE's TRAM. First, it handles the full 625+ ATT&CK vocabulary without retraining — new techniques are added by simply including their descriptions in the candidate pool. Second, it achieves 2.4× higher accuracy than classification (94.52% vs 39.04%) with 4.8× fewer parameters (22.7M vs 110M), because it learns one generalizable skill (semantic matching) rather than hundreds of per-technique decision boundaries.

The system runs on a laptop CPU at 55ms per query with no cloud dependency, making it suitable for air-gapped deployment in classified environments. It integrates as the final refinement stage of the project's CTI knowledge graph pipeline.

## Performance

### Current Best Results (checkpoints/best_two_stage_v2)

| Metric | Value |
|--------|-------|
| P@1 (without post-processing) | 94.52% (138/146) |
| P@1 (with hierarchical post-processing) | 95.21% (139/146) |
| Hit@3 | 99.32% |
| Hit@5 | 100.00% |
| Multi-label F1 (with post-processing) | 80.43% |
| Multi-label Recall (with post-processing) | 81.57% |
| Inference Latency | 55ms per query (CPU) |
| Model Parameters | 22.7M |
| Training Time | 16.4 min (T4 GPU) |

### Per-Actor Performance

| Actor | P@1 | Queries |
|-------|-----|---------|
| APT29 | 95.12% | 41 |
| Carbanak | 100.00% | 27 |
| FIN6 | 90.48% | 21 |
| FIN7 | 100.00% | 24 |
| OilRig | 82.35% | 17 |
| Sandworm | 88.89% | 9 |
| WizardSpider | 100.00% | 7 |

### Comparison Against Baselines

| System | P@1 | Params | Latency | Notes |
|--------|-----|--------|---------|-------|
| BM25 keyword baseline | 75.34% | 0 | 5ms | Lexical matching only |
| MiniLM off-the-shelf | 57.53% | 22.7M | 50ms | No fine-tuning |
| GPT-5.4-mini (zero-shot) | 45.89% | >>1B | 717ms | Cloud-dependent |
| GPT-5.4-mini (few-shot) | 48.59% | >>1B | 916ms | Cloud-dependent |
| SciBERT classifier (TRAM-style) | 39.04% | 110M | — | Classification approach, 124 classes |
| SecureBERT cross-encoder | 90.41% | 124.6M | 290ms | Cybersecurity domain pre-training |
| MiniLM single-stage | 87.67% | 22.7M | 55ms | Published at ICAIC 2026 |
| MiniLM two-stage (short labels) | 91.10% | 22.7M | 55ms | + tumeteor curriculum |
| **MiniLM two-stage (enriched)** | **94.52%** | **22.7M** | **55ms** | **+ ATT&CK descriptions** |
| **+ hierarchical post-processing** | **95.21%** | **22.7M** | **55ms** | **+ ATT&CK taxonomy rules** |

### Cross-Benchmark Validation

| Benchmark | P@1 | Queries | Source |
|-----------|-----|---------|--------|
| CTI-HAL (primary evaluation) | 94.52% | 146 | Expert-annotated CTI reports |
| Tumeteor (generalization, two-stage) | 93.85% | 20,604 | Derived MITRE procedures |
| Tumeteor (zero-shot, single-stage) | 80.95% | 20,604 | Pure generalization |
| 5-fold Cross-Validation (enriched) | 90.7% ± 2.0% | 1,392 | Stratified by actor |
| MITRE Procedures (zero-shot, all 625 techniques) | 53.80% | 500 | MITRE ATT&CK STIX data |

Note: The MITRE Procedures benchmark is the hardest evaluation setting — every query is scored against all 625 techniques with no candidate pre-filtering and no training on procedure examples. The 53.80% P@1 (336× better than random chance at 0.16%) with 77.40% Hit@3 demonstrates genuine zero-shot generalization to MITRE-authored data completely independent from the training sources.

## Architecture

The system uses a two-stage retrieve-then-rerank pipeline.

**Stage 1 — Candidate Retrieval.** Upstream components (Neo4j vector similarity search using nomic-embed embeddings) retrieve the top-K candidate ATT&CK entities for each CTI passage. The reranker receives these candidates for fine-grained semantic scoring.

**Stage 2 — Cross-Encoder Reranking.** Each query-candidate pair is concatenated and processed through a MiniLM cross-encoder (6 transformer layers, 384 hidden dimensions, 22.7M parameters). The model outputs a relevance score; candidates are ranked by score. The base model is `cross-encoder/ms-marco-MiniLM-L-6-v2`, pre-trained on the MS MARCO passage retrieval dataset and fine-tuned on CTI data.

**Stage 3 — Hierarchical Post-Processing.** ATT&CK's taxonomy (tactic → technique → sub-technique) is applied at inference time. When the model predicts a technique, parent techniques and associated tactics are automatically inferred via dictionary lookup. This encodes domain knowledge without retraining.

### Training Pipeline

The model uses two-stage curriculum training.

**Pre-training (Stage 1):** 1 epoch on 47,139 examples from the tumeteor/Security-TTP-Mapping dataset (20,604 queries). This exposes the model to a broad range of ATT&CK technique descriptions before domain-specific fine-tuning.

**Fine-tuning (Stage 2):** 2 epochs on 10,727 examples from CTI-HAL (1,111 training queries across 7 APT actors from 69 real threat reports). Negative sampling ratio is 2.5:1.

**Candidate Text Enrichment:** The key improvement over the published ICAIC results. Short technique labels like "T1059.001 — PowerShell" (max 69 characters) are replaced with full ATT&CK descriptions parsed from MITRE STIX data (`enterprise-attack-v14.json`), providing 200-800 characters of rich semantic content per candidate. This single change pushed P@1 from 91.10% to 94.52%.

### Key Finding: Task Pre-Training > Domain Pre-Training

A controlled comparison swapping the base model from MiniLM (22.7M params, MS MARCO pre-training) to SecureBERT (124.6M params, cybersecurity text pre-training) showed that MiniLM wins decisively: 94.52% vs 90.41%. MS MARCO's query-document matching pre-training transfers better to the retrieval task than cybersecurity domain knowledge, validating the information retrieval framing of CTI-to-ATT&CK mapping.

## Quick Start

### Prerequisites

Python 3.8+ with the following packages:

```bash
pip install torch sentence-transformers numpy scikit-learn
pip install rank_bm25  # For BM25 baseline evaluation
```

### Loading the Model

```python
from sentence_transformers import CrossEncoder
import numpy as np

# Load the best model (94.52% P@1)
model = CrossEncoder('./checkpoints/best_two_stage_v2')

# Example: Score a CTI passage against ATT&CK technique descriptions
query = "The attacker used PowerShell to download additional tools and establish persistence"

candidates = [
    "T1059.001 — PowerShell: Adversaries may abuse PowerShell commands and scripts for execution...",
    "T1105 — Ingress Tool Transfer: Adversaries may transfer tools from an external system...",
    "T1547 — Boot or Logon Autostart Execution: Adversaries may configure system settings...",
]

pairs = [[query, c] for c in candidates]
scores = model.predict(pairs)

best_idx = np.argmax(scores)
print(f"Best match: {candidates[best_idx][:60]}...")
print(f"Score: {scores[best_idx]:.4f}")
```

### Reproducing the Enrichment

To enrich candidate text with full ATT&CK descriptions:

1. Download `enterprise-attack-v14.json` from the [MITRE ATT&CK STIX repository](https://github.com/mitre-attack/attack-stix-data).
2. Place it in this directory.
3. Run: `python enrich_candidates.py`

This generates enriched training data in `data/reranker_pairs_enriched_v2.jsonl`.

### Training

Training uses Google Colab with a T4 GPU (free tier). Upload the enriched data files and run:

```python
!python two_stage_enriched.py \
  --stage1-data reranker_pairs_enriched_tumeteor_v2.jsonl \
  --stage2-data reranker_pairs_enriched_v2.jsonl \
  --stage1-epochs 1 \
  --stage2-epochs 2 \
  --output ./enriched_model
```

Total training time: approximately 16 minutes on T4 GPU.

## Repository Structure

```
graph_alignment/reranker/
│
├── README.md                          # This file
│
├── # ── Core System ──
├── finetune_production.py             # Training script (single-stage)
├── enrich_candidates.py               # ATT&CK description enrichment from STIX
├── hierarchical_postprocess.py        # ATT&CK taxonomy post-processing
├── explain.py                         # LOO token perturbation interpretability
├── flatten_contexts.py                # Data preprocessing
│
├── # ── Evaluation Suite ──
├── llm_baseline_final.py              # GPT-5.4-mini comparison (zero/few-shot)
├── tumeteor_eval.py                   # Secondary benchmark (20,604 queries)
├── mcnemar_test.py                    # Statistical significance testing
├── multilabel_eval.py                 # Multi-label precision/recall/F1
├── auto_word_categorize.py            # Corpus-frequency word categorization
├── tram_comparison.py                 # TRAM-style classification baseline
├── mitre_procedure_benchmark.py       # MITRE procedure examples (zero-shot)
│
├── # ── Ablation Studies ──
├── error_analysis_v2.py               # Categorization of remaining errors
├── hard_negative_mining.py            # Hard negative mining experiment
├── clean_markdown.py                  # Markdown cleaning experiment
├── securebert_comparison.py           # SecureBERT vs MiniLM base model
├── deployment_eval.py                 # Deployment-realistic evaluation
├── bm25_recall_check.py               # BM25 recall diagnostic
│
├── # ── Data ──
├── data/
│   ├── reranker_pairs_enriched.jsonl          # Original enriched training data
│   └── reranker_pairs_enriched_v2.jsonl       # Best enriched data (gitignored, regenerable)
├── gold_id_unbalanced_nobom.jsonl     # Gold evaluation labels (by ID)
├── gold_label_unbalanced_nobom.jsonl  # Gold evaluation labels (by label)
│
├── # ── Model Checkpoints ──
├── checkpoints/
│   ├── best/                          # Single-stage model (87.67% P@1)
│   ├── best_two_stage/                # Two-stage, short labels (91.10% P@1)
│   └── best_two_stage_v2/             # Two-stage, enriched (94.52% P@1) ← BEST
│
├── # ── Results ──
├── eval_results/                      # Interpretability analysis results
├── error_analysis_v2/                 # Error categorization (8 errors)
├── hard_neg_results/                  # Hard negative mining report
├── hierarchical_results/              # Hierarchical post-processing results
├── llm_baseline_results/              # GPT-5.4-mini comparison results
├── multilabel_results/                # Multi-label evaluation results
├── tumeteor_eval_results/             # Tumeteor benchmark results
├── mitre_procedure_results/           # MITRE procedure benchmark results
└── deployment_eval_results/           # Deployment evaluation results
```

## Error Analysis

At 94.52% P@1, the model makes 8 errors out of 146 test queries. Detailed analysis reveals that 5 of these are defensible alternative mappings where the model identified a legitimate ATT&CK technique matching a different aspect of the same behavior (method vs. purpose). For example, predicting T1047 (WMI — the method) when the gold label is T1082 (System Information Discovery — the purpose). Seven of 8 errors have the gold answer in the top 3 predictions. Only 2 represent genuine misclassifications.

Hard negative mining and markdown cleaning were explored as improvement strategies; neither improved P@1 beyond 94.52%, confirming the remaining errors reflect annotation ambiguity rather than learnable model failures.

## Improvements Since ICAIC 2026 Publication

The published paper (IEEE ICAIC 2026, Best Research Paper) reported 87.67% P@1 with short technique labels. Post-publication improvements include:

**Candidate text enrichment** (87.67% → 94.52%): Replacing short "ID — Name" labels with full ATT&CK descriptions from MITRE STIX data. This is the single largest improvement, achieved through better input representation rather than architectural changes.

**Hierarchical ATT&CK post-processing** (94.52% → 95.21%): Inference-time module encoding the ATT&CK taxonomy. Zero retraining, zero additional latency. Improves multi-label recall from 78.86% to 81.57%.

**Comprehensive evaluation portfolio**: LLM baseline comparison (GPT-5.4-mini), 5-fold cross-validation (90.7% ± 2.0%), tumeteor generalization benchmark (20,604 queries), McNemar's statistical significance testing (p < 0.001), multi-label evaluation (80.43% F1), automated corpus-frequency word categorization, TRAM classification baseline (39.04% vs 94.52%), SecureBERT base model comparison (90.41% vs 94.52%), and zero-shot evaluation on 500 MITRE procedure examples (53.80% P@1 against all 625 techniques).

## Citation

```bibtex
@inproceedings{waldrop2026digital,
  title     = {Digital Precognition: Teaching Transformers to Map Cyber Threats Before Analysts Can},
  author    = {Waldrop, Shane and Dogdu, Erdogan and Choupani, Roya and Mitchell, William},
  booktitle = {5th IEEE International Conference on AI in Cybersecurity (ICAIC)},
  year      = {2026},
  address   = {Houston, TX},
  note      = {Best Research Paper Award. Updated results: 95.21\% P@1}
}
```

## Acknowledgments

This research was conducted at Angelo State University under ARL Cooperative Agreement W911NF-24-2-0180, with supervision from Dr. Erdogan Dogdu and technical guidance from William Mitchell.

---

*Last updated: April 2026*
*Best model: checkpoints/best_two_stage_v2 (94.52% P@1, 95.21% with post-processing)*
*Training data: 27,976 enriched CTI query-candidate pairs across 7 threat actors*
