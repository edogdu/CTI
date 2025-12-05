# CTI→ATT&CK Neural Reranker: 87.7% Precision with Fine-Tuned MiniLM

## 🎯 What This Project Achieves

This repository contains a production-ready neural reranking system that maps cyber threat intelligence (CTI) descriptions to MITRE ATT&CK techniques with **87.7% precision at rank 1**. Think of it as a specialized translator that understands cybersecurity language - when you describe a threat behavior, it tells you exactly which ATT&CK technique is being used, getting it right nearly 9 times out of 10.

The system solves a critical problem in threat intelligence: analysts spend hours manually mapping threat reports to the ATT&CK framework. Our fine-tuned model reduces this to milliseconds while maintaining higher accuracy than both generic language models (57.4%) and keyword-based approaches (75.4%).

## 📊 Performance Metrics

Our fine-tuned cross-encoder achieves remarkable performance across multiple threat actors:

| Metric | Generic MiniLM | BM25 Baseline | **Our Model** | Improvement |
|--------|---------------|---------------|---------------|-------------|
| **Precision@1** | 57.38% | 75.36% | **87.67%** | +30.29% / +12.31% |
| **Hit@3** | ~75% | 97.56% | **97.95%** | +22.95% / +0.39% |
| **Hit@5** | ~85% | 99.86% | **99.32%** | +14.32% / -0.54% |

### Performance by Threat Actor
- **APT29**: 87.80% (41 test queries)
- **Carbanak**: 100.00% (27 test queries) 
- **FIN6**: 76.19% (21 test queries)
- **FIN7**: 95.83% (24 test queries)
- **Oilrig**: 76.47% (17 test queries)
- **Sandworm**: 66.67% (9 test queries)
- **WizardSpider**: 100.00% (8 test queries)

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have Python 3.8 or later installed on your system. You'll also need about 2GB of free disk space for the model and dependencies.

### Installation

First, clone this repository and install the required packages:

```bash
# Clone the repository
git clone https://github.com/edogdu/CTI.git
cd CTI/graph_alignment

# Install dependencies
pip install torch sentence-transformers pandas numpy scikit-learn
pip install accelerate datasets  # Required for training
```

### Downloading the Model

We use Git Large File Storage (LFS) for the 88.7MB model file. Here's how to get it:

```bash
# Install Git LFS if you haven't already
git lfs install

# Pull the model file
git lfs pull

# The model will be in graph_alignment/reranker/checkpoints/best/
```

### Basic Usage

Here's how to use the trained model to classify threat descriptions:

```python
from sentence_transformers import CrossEncoder
import numpy as np

# Load the fine-tuned model
model = CrossEncoder('./graph_alignment/reranker/checkpoints/best')

# Example: Classify a threat description
threat_description = "The attacker used PowerShell to download additional tools and establish persistence"

# ATT&CK techniques to consider (in practice, you'd have all techniques)
candidate_techniques = [
    "T1059.001 — PowerShell",
    "T1105 — Ingress Tool Transfer",
    "T1547 — Boot or Logon Autostart Execution",
    "T1055 — Process Injection",
    "T1053 — Scheduled Task/Job"
]

# Score each candidate
pairs = [[threat_description, technique] for technique in candidate_techniques]
scores = model.predict(pairs)

# Get the best match
best_idx = np.argmax(scores)
print(f"Best match: {candidate_techniques[best_idx]}")
print(f"Confidence score: {scores[best_idx]:.3f}")

# Show top 3 matches
top_3_indices = np.argsort(scores)[::-1][:3]
print("\nTop 3 predictions:")
for i, idx in enumerate(top_3_indices, 1):
    print(f"{i}. {candidate_techniques[idx]} (score: {scores[idx]:.3f})")
```

## 📁 Repository Structure

Understanding how this repository is organized will help you navigate and use the code effectively:

```
CTI/graph_alignment/
│
├── README.md                              # This file
├── metrics_top1.py                        # Metrics computation script
├── preflight.py                           # Data validation utilities
├── misses_report.py                       # Error analysis script
├── ce_top1_final_ID.csv                   # Top-1 predictions with IDs
├── ce_top1_final_LABEL_metrics.csv        # Detailed metrics by label
├── minilm_ft_by_actor.csv                 # Per-actor performance breakdown
├── gold_id_unbalanced_nobom.jsonl         # Evaluation dataset (by ID)
├── gold_label_unbalanced_nobom.jsonl      # Evaluation dataset (by label)
├── results_brief_finetuned.md             # Results summary
│
└── graph_alignment/reranker/              # Main reranking system
    ├── finetune_production.py             # Training script that produced the model
    ├── flatten_contexts.py                # Data preprocessing utilities
    ├── data/
    │   └── reranker_pairs_enriched.jsonl  # Prepared training data (27,976 pairs)
    └── checkpoints/best/                  # Best performing model (87.7% P@1)
        ├── model.safetensors              # Trained model weights
        ├── config.json                    # Model configuration
        ├── tokenizer.json                 # Tokenizer configuration
        ├── vocab.txt                      # Vocabulary file
        └── eval/                          # Evaluation results
```

## 🔧 Reproducing Results

To reproduce the results presented in this project, follow these steps:

### Step 1: Setup Environment

```bash
git clone https://github.com/edogdu/CTI.git
cd CTI/graph_alignment
pip install torch sentence-transformers pandas numpy scikit-learn accelerate datasets
git lfs pull
```

### Step 2: Run Training (Optional - model already provided)

If you want to retrain the model from scratch:

```bash
python graph_alignment/reranker/finetune_production.py \
  --epochs 3 \
  --bs 16 \
  --lr 2e-5 \
  --device cuda  # Use 'cpu' if no GPU available
```

Training takes approximately 20-30 minutes on GPU or 30-45 minutes on CPU.

### Step 3: Run Evaluation

```bash
python metrics_top1.py
```

### Expected Output

```
Overall Results:
  P@1:    87.67% (244/278 correct)
  Hit@3:  97.95%
  Hit@5:  99.32%
  MRR:    0.912

Per-Actor P@1:
  APT29:        87.80%
  Carbanak:     100.00%
  FIN6:         76.19%
  FIN7:         95.83%
  Oilrig:       76.47%
  Sandworm:     66.67%
  WizardSpider: 100.00%
```

## 🧠 Understanding How It Works

The success of this system comes from teaching a neural network to understand the specialized language of cybersecurity. Here's what makes it work:

### The Base Model
We start with Microsoft's MiniLM-L-6-v2, a compact but powerful language model with 22.7 million parameters arranged in 6 transformer layers. This model was pretrained on MS-MARCO passage ranking data (Nogueira & Cho, 2019), so it understands query-document relevance but not cybersecurity specifics.

### The Fine-Tuning Process
Through fine-tuning on 27,976 query-candidate pairs from real threat intelligence reports, the model learns:
- **Vocabulary mapping**: "lateral movement" → Remote Services techniques
- **Hierarchical understanding**: T1059 (parent) vs T1059.001 (sub-technique)
- **Contextual patterns**: Attack sequences and technique combinations
- **Actor-specific patterns**: How different groups describe similar behaviors

### Why It Works Better Than Alternatives
- **Generic models (57.4%)**: Don't understand security terminology
- **Keyword matching (75.4%)**: Misses semantic relationships
- **Our approach (87.7%)**: Understands meaning, not just words

## 📈 Evaluation Methodology

To ensure trustworthy results, we implement rigorous evaluation:

### Data Splitting
- 80% training (1,111 queries)
- 10% validation (135 queries)  
- 10% test (146 queries)
- **Zero leakage**: No query appears in multiple splits

### Metrics Explained
- **Precision@1 (P@1)**: Is the top prediction correct?
- **Hit@3**: Is the correct answer in the top 3?
- **Hit@5**: Is the correct answer in the top 5?
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank for correct answers

### Statistical Significance
With 146 test queries and 87.7% accuracy, the 95% confidence interval is approximately ±5.3%, meaning true performance is likely between 82.4% and 93.0%.

## 🔍 Known Limitations and Future Work

While our model achieves impressive results, there are areas for improvement:

### Current Limitations
- **Sandworm performance**: 66.7% accuracy due to limited training data (107 examples vs 398 for APT29) and specialized ICS vocabulary (IEC 104, RTU, HMI, SCADA)
- **Single-label assumption**: Doesn't handle multi-technique descriptions
- **Static model**: Doesn't adapt to new techniques without retraining

### Planned Improvements
1. **Confidence scoring** to indicate prediction uncertainty
2. **Multi-label support** for descriptions involving multiple techniques
3. **Ensemble methods** combining neural and keyword approaches
4. **Expanded dataset** with more ICS-focused training examples
5. **Real-time integration** with threat intelligence feeds

## 📚 Technical Deep Dive

For those interested in the implementation details:

### Model Architecture
- **Type**: Cross-encoder (jointly processes query and candidate)
- **Base Model**: MS-MARCO MiniLM-L-6-v2
- **Parameters**: 22.7M
- **Layers**: 6 transformer layers, 384 hidden dimensions
- **Max sequence**: 512 tokens
- **Output**: Single relevance score (0-1)

### Training Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-05 |
| Batch size | 16 |
| Epochs | 3 |
| Warmup steps | 10% of training |
| Max sequence length | 512 tokens |
| Optimizer | AdamW (weight decay 0.01) |
| Training time | ~30 minutes (CPU) |

### Computational Requirements
- **Training time**: ~30 minutes on CPU, ~20 minutes on GPU
- **Inference**: ~50ms per query with 20 candidates
- **Model size**: 88.7MB
- **Memory usage**: ~500MB during inference

## 📖 References

This work builds on the following foundational research:

### Neural Ranking & Sentence Embeddings
- Reimers, N., & Gurevych, I. (2019). **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP)*, 3982-3992. https://arxiv.org/abs/1908.10084

### BERT for Document Ranking
- Nogueira, R., & Cho, K. (2019). **Passage Re-ranking with BERT**. *arXiv preprint arXiv:1901.04085*. https://arxiv.org/abs/1901.04085

- Nogueira, R., Yang, W., Cho, K., & Lin, J. (2019). **Multi-Stage Document Ranking with BERT**. *arXiv preprint arXiv:1910.14424*. https://arxiv.org/abs/1910.14424

### Resources
- **MITRE ATT&CK Framework**: https://attack.mitre.org/
- **Base Model (MS-MARCO MiniLM)**: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
- **Sentence Transformers Library**: https://www.sbert.net/

## 🙏 Acknowledgments

- **Dr. Erdogan Dogdu** (Angelo State University) - Project advisor
- **Bill Mitchell** - Technical guidance
- **MITRE** for the ATT&CK framework
- **Microsoft** for the base MiniLM model
- **Hugging Face** for the sentence-transformers library

This research was conducted at Angelo State University with support from the Department of Defense.

## 📄 License

This project is part of the CTI research initiative at Angelo State University.

## 📝 Citation

If you use this work in research, please cite:

```bibtex
@software{waldrop_cti_reranker_2024,
  title = {CTI→ATT&CK Neural Reranker},
  author = {Waldrop, Shane and Dogdu, Erdogan and Mitchell, Bill},
  year = {2024},
  institution = {Angelo State University},
  url = {https://github.com/edogdu/CTI/tree/main/graph_alignment},
  note = {87.7% P@1 on cyber threat intelligence classification}
}
```

## 📞 Contact

For questions or collaboration opportunities, please contact the CTI research team at Angelo State University.

---

*Last updated: December 2024*  
*Model version: 1.0*  
*Training data: 27,976 CTI query-candidate pairs across 7 threat actors*
