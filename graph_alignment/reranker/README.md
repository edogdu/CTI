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
git clone https://github.com/Wittyusername12/cti-jsonl-pipeline.git
cd cti-jsonl-pipeline

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

If you prefer not to use Git LFS, you can also download the model directly from the [Releases page](https://github.com/Wittyusername12/cti-jsonl-pipeline/releases).

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
cti-jsonl-pipeline/
│
├── graph_alignment/reranker/     # Main reranking system
│   ├── checkpoints/              # Trained models
│   │   ├── best/                 # Best performing model (87.7% P@1)
│   │   └── last/                 # Final training checkpoint
│   ├── finetune_production.py   # Training script that produced the model
│   ├── evaluate_model.py        # Evaluation utilities
│   └── data/
│       └── reranker_pairs_enriched.jsonl  # Prepared training data
│
├── ce_top1_final_ID.csv         # Top-1 predictions with IDs
├── ce_top1_final_LABEL_metrics.csv  # Detailed metrics by label
├── metrics_top1.py              # Metrics computation script
├── preflight.py                 # Data validation utilities
└── README.md                    # This file
```

## 🔧 Training Your Own Model

If you want to reproduce our results or fine-tune on your own data, here's the complete process:

### Preparing Your Data

Your data should be in JSONL format with each line containing:
```json
{
  "query_raw": "The threat description text",
  "query_norm": "normalized_query_for_deduplication",
  "candidate_id": "T1059.001",
  "candidate_text": "T1059.001 — PowerShell",
  "label": 1,  // 1 for correct match, 0 for incorrect
  "actor": "apt29"
}
```

### Running Fine-Tuning

```bash
# Run the fine-tuning script
python graph_alignment/reranker/finetune_production.py \
  --epochs 3 \
  --bs 16 \
  --lr 2e-5 \
  --device cuda  # Use 'cpu' if no GPU available

# Training takes approximately:
# - 20-30 minutes on GPU
# - 30-45 minutes on CPU
```

The script automatically:
- Validates data quality (rejects if >5% of candidates lack proper text)
- Creates stratified train/val/test splits with zero leakage
- Balances classes during training (1:2.5 positive:negative ratio)
- Saves the best model based on validation performance
- Evaluates on held-out test set

## 🧠 Understanding How It Works

The success of this system comes from teaching a neural network to understand the specialized language of cybersecurity. Here's what makes it work:

### The Base Model
We start with Microsoft's MiniLM-L-6-v2, a compact but powerful language model with 22.7 million parameters arranged in 6 transformer layers. This model was originally trained on search queries and web passages, so it understands general language but not cybersecurity specifics.

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
- **Recall@5**: What fraction of all correct answers appear in top 5?

### Statistical Significance
With 146 test queries and 87.7% accuracy, the 95% confidence interval is approximately ±5.3%, meaning true performance is likely between 82.4% and 93.0%.

## 🔍 Known Limitations and Future Work

While our model achieves impressive results, there are areas for improvement:

### Current Limitations
- **Sandworm performance**: 66.7% accuracy (needs targeted improvement)
- **Single-label assumption**: Doesn't handle multi-technique descriptions
- **Static model**: Doesn't adapt to new techniques without retraining

### Planned Improvements
1. **Actor-specific fine-tuning** for underperforming groups
2. **Ensemble methods** combining neural and keyword approaches
3. **Confidence scoring** to indicate prediction uncertainty
4. **Continuous learning** as new ATT&CK techniques emerge

## 📚 Technical Deep Dive

For those interested in the implementation details:

### Model Architecture
- **Type**: Cross-encoder (jointly processes query and candidate)
- **Parameters**: 22.7M
- **Layers**: 6 transformer layers, 384 hidden dimensions
- **Max sequence**: 512 tokens
- **Output**: Single relevance score (0-1)

### Training Hyperparameters
- **Learning rate**: 2e-05 with linear warmup
- **Batch size**: 16
- **Epochs**: 3
- **Optimizer**: AdamW with weight decay 0.01
- **Class balancing**: 1:2.5 positive:negative ratio
- **Early stopping**: Based on validation P@1

### Computational Requirements
- **Training time**: ~30 minutes on CPU
- **Inference**: ~50ms per query with 20 candidates
- **Model size**: 88.7MB
- **Memory usage**: ~500MB during inference

## 🤝 Contributing

We welcome contributions! Areas where you can help:

- Improving performance on underperforming actors
- Adding new evaluation metrics
- Creating visualization tools
- Extending to other threat intelligence frameworks

Please open an issue to discuss major changes before submitting a PR.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MITRE for the ATT&CK framework
- Microsoft for the base MiniLM model
- Hugging Face for the sentence-transformers library

## 📖 Citation

If you use this work in research, please cite:

```bibtex
@software{cti_neural_reranker_2024,
  title = {CTI→ATT&CK Neural Reranker},
  author = {Wittyusername12},
  year = {2024},
  url = {https://github.com/Wittyusername12/cti-jsonl-pipeline},
  note = {87.7% P@1 on cyber threat intelligence classification}
}
```

## 📞 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

*Last updated: November 2024*  
*Model version: 1.0*  
*Training data: 27,976 CTI query-candidate pairs across 7 threat actors*
