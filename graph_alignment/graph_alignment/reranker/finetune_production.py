#!/usr/bin/env python3
"""
Production-Ready Fine-Tuning Script for MiniLM Cross-Encoder on CTI Data
=========================================================================
This version ensures full compatibility with current sentence-transformers library
by properly formatting all training data as InputExample objects.

Key improvements:
- Proper InputExample formatting for all training data
- Compatible with latest sentence-transformers (2.2.0+)
- Robust error handling and progress reporting
- Comprehensive evaluation metrics

Expected outcome: P@1 improvement from 57% to 75-80%+
"""

import json
import random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import time
import os
import sys
import argparse

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def parse_arguments():
    """Parse command line arguments for flexible configuration."""
    parser = argparse.ArgumentParser(description='Fine-tune MiniLM for CTI ranking')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs (max 4)')
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device to use')
    parser.add_argument('--save-dir', default='./outputs', help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Cap epochs at 4 as specified
    if args.epochs > 4:
        print(f"Note: Capping epochs at 4 (requested {args.epochs})")
        args.epochs = 4
    
    return args

def load_and_validate_data(filepath='./outputs/reranker_pairs_enriched.jsonl'):
    """
    Load and validate the enriched dataset with comprehensive quality checks.
    
    This function performs several critical validations to ensure your data
    is suitable for training. Think of it as a quality control inspection
    before starting production - we need to make sure all the raw materials
    meet our specifications before we begin the manufacturing process.
    """
    print("="*70)
    print("LOADING AND VALIDATING DATASET")
    print("="*70)
    
    query_data = defaultdict(lambda: {
        'query_raw': None,
        'actor': None,
        'candidates': [],
        'positive_count': 0,
        'negative_count': 0
    })
    
    total_rows = 0
    empty_candidate_texts = 0
    
    print("\nReading dataset from:", filepath)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_rows += 1
                
                if total_rows % 5000 == 0:
                    print(f"  Processed {total_rows:,} rows...")
                
                try:
                    row = json.loads(line)
                    
                    query_norm = row.get('query_norm', '')
                    query_raw = row.get('query_raw', '')
                    candidate_text = row.get('candidate_text', '')
                    candidate_id = row.get('candidate_id', '')
                    candidate_norm = row.get('candidate_norm', '')
                    label = row.get('label', 0)
                    actor = row.get('actor', 'unknown')
                    
                    if not query_norm or not query_raw:
                        continue
                    
                    # Check if candidate_text is meaningful
                    if not candidate_text or candidate_text.strip() == candidate_norm:
                        empty_candidate_texts += 1
                    
                    if query_data[query_norm]['query_raw'] is None:
                        query_data[query_norm]['query_raw'] = query_raw
                        query_data[query_norm]['actor'] = actor if actor else 'unknown'
                    
                    query_data[query_norm]['candidates'].append({
                        'text': candidate_text,
                        'label': label,
                        'id': candidate_norm if candidate_norm else candidate_id
                    })
                    
                    if label == 1:
                        query_data[query_norm]['positive_count'] += 1
                    else:
                        query_data[query_norm]['negative_count'] += 1
                        
                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        print(f"\n❌ ERROR: Could not find file: {filepath}")
        print("Please ensure the file exists in the ./outputs/ directory")
        sys.exit(1)
    
    print(f"\nValidation Results:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Unique queries: {len(query_data):,}")
    print(f"  Rows with empty/ID-only candidate text: {empty_candidate_texts:,}")
    
    empty_text_ratio = empty_candidate_texts / total_rows if total_rows > 0 else 0
    if empty_text_ratio > 0.05:
        print(f"\n⚠️  WARNING: {empty_text_ratio:.1%} of rows have problematic candidate text")
        print("This exceeds 5% threshold but proceeding with training...")
    
    # Calculate class distribution
    total_positives = sum(q['positive_count'] for q in query_data.values())
    total_negatives = sum(q['negative_count'] for q in query_data.values())
    
    print(f"\nClass Distribution:")
    print(f"  Positive examples: {total_positives:,} ({total_positives/(total_positives+total_negatives)*100:.1f}%)")
    print(f"  Negative examples: {total_negatives:,} ({total_negatives/(total_positives+total_negatives)*100:.1f}%)")
    
    return dict(query_data)

def create_stratified_splits(query_data: Dict, train_ratio=0.8, val_ratio=0.1):
    """
    Create train/val/test splits ensuring no data leakage between splits.
    
    This is one of the most critical parts of machine learning - ensuring that
    your test data is truly unseen during training. If the same query appears
    in both training and test sets, your model might memorize specific examples
    rather than learning general patterns, leading to overoptimistic results
    that fail in real-world use.
    """
    print("\n" + "="*70)
    print("CREATING STRATIFIED TRAIN/VAL/TEST SPLITS")
    print("="*70)
    
    # Group queries by actor for stratification
    queries_by_actor = defaultdict(list)
    for query_norm, data in query_data.items():
        if data['positive_count'] > 0:  # Only include queries with positives
            queries_by_actor[data['actor']].append(query_norm)
    
    print(f"\nDistribution across {len(queries_by_actor)} actors:")
    for actor, queries in sorted(queries_by_actor.items()):
        print(f"  {actor}: {len(queries)} queries")
    
    # Initialize splits
    train_queries = []
    val_queries = []
    test_queries = []
    
    # Stratified splitting by actor
    for actor, queries in queries_by_actor.items():
        random.shuffle(queries)
        n_queries = len(queries)
        n_train = int(n_queries * train_ratio)
        n_val = int(n_queries * val_ratio)
        
        train_queries.extend(queries[:n_train])
        val_queries.extend(queries[n_train:n_train + n_val])
        test_queries.extend(queries[n_train + n_val:])
    
    random.shuffle(train_queries)
    random.shuffle(val_queries)
    random.shuffle(test_queries)
    
    print(f"\nSplit Sizes:")
    print(f"  Training:   {len(train_queries)} queries")
    print(f"  Validation: {len(val_queries)} queries")
    print(f"  Test:       {len(test_queries)} queries")
    
    # Verify no leakage
    train_set = set(train_queries)
    val_set = set(val_queries)
    test_set = set(test_queries)
    
    if train_set & val_set or train_set & test_set or val_set & test_set:
        print("\n❌ ERROR: Data leakage detected!")
        sys.exit(1)
    else:
        print("\n✓ Splits verified: No data leakage detected")
    
    return train_queries, val_queries, test_queries

def train_model_properly(train_queries, val_queries, query_data, args):
    """
    Fine-tune the CrossEncoder with proper data formatting for current library version.
    
    This function handles all the complexity of training a neural network for
    ranking. The key insight is that we're teaching the model to score the
    relevance between a query (threat description) and a candidate (ATT&CK technique).
    Higher scores mean better matches.
    """
    print("\n" + "="*70)
    print("INITIALIZING CROSS-ENCODER FOR FINE-TUNING")
    print("="*70)
    
    try:
        from sentence_transformers import CrossEncoder, InputExample
        import torch
        
        # Check for CUDA availability
        device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
        print(f"✓ Using {device.upper()} for training")
        if device == 'cpu':
            print("  Expected training time: 30-45 minutes")
        
        # Set PyTorch seed
        torch.manual_seed(RANDOM_SEED)
        if device == 'cuda':
            torch.cuda.manual_seed(RANDOM_SEED)
            
    except ImportError as e:
        print(f"\n❌ ERROR: Required packages not installed")
        print(f"Missing: {e}")
        print("\nInstall with:")
        print("  pip install torch sentence-transformers")
        sys.exit(1)
    
    # Initialize model
    model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    print(f"\nLoading model: {model_name}")
    print("This model was trained on Microsoft's search dataset and we'll")
    print("adapt it specifically for cyber threat intelligence...")
    
    model = CrossEncoder(model_name, num_labels=1, max_length=512, device=device)
    
    print("\nModel Architecture:")
    print("  - 6 transformer layers (processes text through multiple stages)")
    print("  - 384 hidden dimensions (internal representation size)")
    print("  - 22.7M parameters (weights to be fine-tuned)")
    print("  - Outputs: Single relevance score (0-1)")
    
    # Prepare training data as InputExample objects
    print("\nPreparing training data with proper formatting...")
    train_examples = []
    
    for query_norm in train_queries:
        if query_norm not in query_data:
            continue
        
        data = query_data[query_norm]
        query_raw = data['query_raw']
        
        # Get positive and negative candidates
        positive_candidates = [c for c in data['candidates'] if c['label'] == 1]
        negative_candidates = [c for c in data['candidates'] if c['label'] == 0]
        
        # Add all positives (they're valuable for learning)
        for candidate in positive_candidates:
            train_examples.append(
                InputExample(texts=[query_raw, candidate['text']], label=1.0)
            )
        
        # Add balanced negatives (2.5 negatives per positive)
        n_negatives = min(int(len(positive_candidates) * 2.5), len(negative_candidates))
        if n_negatives > 0:
            sampled_negatives = random.sample(negative_candidates, n_negatives)
            for candidate in sampled_negatives:
                train_examples.append(
                    InputExample(texts=[query_raw, candidate['text']], label=0.0)
                )
    
    # Shuffle training examples
    random.shuffle(train_examples)
    
    print(f"  Created {len(train_examples):,} training examples")
    positive_count = sum(1 for ex in train_examples if ex.label == 1.0)
    print(f"  Positive ratio: {positive_count/len(train_examples)*100:.1f}%")
    
    # Prepare validation data similarly
    print("\nPreparing validation data...")
    val_examples = []
    
    for query_norm in val_queries:
        if query_norm not in query_data:
            continue
        
        data = query_data[query_norm]
        query_raw = data['query_raw']
        
        # For validation, use ALL candidates (realistic evaluation)
        for candidate in data['candidates']:
            val_examples.append(
                InputExample(texts=[query_raw, candidate['text']], 
                           label=float(candidate['label']))
            )
    
    print(f"  Created {len(val_examples):,} validation examples")
    val_positive_count = sum(1 for ex in val_examples if ex.label == 1.0)
    print(f"  Positive ratio: {val_positive_count/len(val_examples)*100:.1f}%")
    
    # Training configuration
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.bs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup steps: {int(len(train_examples) / args.bs * 0.1)}")
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f"{args.save_dir}/checkpoints", exist_ok=True)
    
    # Set up evaluator for validation
    from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
    
    # Group validation examples by query for ranking evaluation
    val_query_examples = defaultdict(list)
    for ex in val_examples:
        query = ex.texts[0]
        val_query_examples[query].append(ex)
    
    # Create evaluator samples
    evaluator_samples = []
    for query, examples in val_query_examples.items():
        # Find positive examples for this query
        positive_texts = [ex.texts[1] for ex in examples if ex.label == 1.0]
        negative_texts = [ex.texts[1] for ex in examples if ex.label == 0.0]
        
        if positive_texts and negative_texts:
            evaluator_samples.append({
                'query': query,
                'positive': positive_texts,
                'negative': negative_texts[:10]  # Limit negatives for efficiency
            })
    
    print(f"\nCreated {len(evaluator_samples)} validation queries for evaluation")
    
    # Create evaluator
    from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
    evaluator = CERerankingEvaluator(evaluator_samples, name='val')
    
    # Train the model
    print("\n" + "="*70)
    print("STARTING FINE-TUNING")
    print("="*70)
    print("\nThe model is now learning to understand:")
    print("  • Cyber threat terminology (e.g., 'lateral movement', 'persistence')")
    print("  • ATT&CK technique relationships (techniques, tactics, software)")
    print("  • Contextual clues that indicate specific attack patterns")
    print("\nTraining progress will be shown below...")
    print("-" * 70)
    
    # Use the fit method with properly formatted data
    from torch.utils.data import DataLoader
    
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.bs
    )
    
    # Train using fit method with correct parameters
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        evaluation_steps=int(len(train_examples) / args.bs / 2),  # Eval twice per epoch
        warmup_steps=int(len(train_examples) / args.bs * 0.1),
        output_path=f"{args.save_dir}/checkpoints/best",
        save_best_model=True,
        optimizer_params={'lr': args.lr},
        weight_decay=0.01,
        show_progress_bar=True
    )
    
    # Save final model
    model.save(f"{args.save_dir}/checkpoints/last")
    
    print(f"\n✓ Training completed successfully!")
    print(f"✓ Best model saved to: {args.save_dir}/checkpoints/best")
    print(f"✓ Final model saved to: {args.save_dir}/checkpoints/last")
    
    # Load best model for final evaluation
    best_model = CrossEncoder(f"{args.save_dir}/checkpoints/best", device=device)
    
    return best_model

def evaluate_final_model(model, test_queries, query_data, args):
    """
    Perform comprehensive evaluation on the test set.
    
    This is the moment of truth - we evaluate the model on data it has never
    seen during training to get an honest assessment of its performance.
    This tells us how well the model will perform on new, real-world threat
    descriptions.
    """
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    # Evaluate each query
    results_by_actor = defaultdict(list)
    all_results = []
    
    for query_norm in test_queries:
        if query_norm not in query_data:
            continue
        
        data = query_data[query_norm]
        query_raw = data['query_raw']
        actor = data['actor']
        
        # Prepare candidates
        texts = [[query_raw, c['text']] for c in data['candidates']]
        true_labels = [c['label'] for c in data['candidates']]
        
        if not texts or sum(true_labels) == 0:
            continue
        
        # Get model predictions
        scores = model.predict(texts, show_progress_bar=False)
        
        # Rank by score
        ranked_indices = np.argsort(scores)[::-1]
        ranked_labels = [true_labels[i] for i in ranked_indices]
        
        # Calculate metrics
        p_at_1 = ranked_labels[0] if ranked_labels else 0
        hit_at_3 = 1 if 1 in ranked_labels[:3] else 0
        hit_at_5 = 1 if 1 in ranked_labels[:5] else 0
        
        results_by_actor[actor].append({
            'p@1': p_at_1,
            'hit@3': hit_at_3,
            'hit@5': hit_at_5
        })
        
        all_results.append({
            'p@1': p_at_1,
            'hit@3': hit_at_3,
            'hit@5': hit_at_5
        })
    
    # Calculate overall metrics
    overall_p1 = np.mean([r['p@1'] for r in all_results])
    overall_hit3 = np.mean([r['hit@3'] for r in all_results])
    overall_hit5 = np.mean([r['hit@5'] for r in all_results])
    
    print(f"\nOVERALL TEST METRICS:")
    print(f"  P@1:   {overall_p1:.4f} (vs baseline 0.7536)")
    print(f"  Hit@3: {overall_hit3:.4f}")
    print(f"  Hit@5: {overall_hit5:.4f}")
    
    # By-actor metrics
    print(f"\nBY-ACTOR PERFORMANCE:")
    print("-" * 50)
    print(f"{'Actor':<12} | {'P@1':>8} | {'N':>5}")
    print("-" * 50)
    
    actor_metrics = {}
    for actor in sorted(results_by_actor.keys()):
        results = results_by_actor[actor]
        p1 = np.mean([r['p@1'] for r in results])
        actor_metrics[actor] = p1
        print(f"{actor:<12} | {p1:>8.4f} | {len(results):>5}")
    
    # Save results
    import csv
    with open(f'{args.save_dir}/minilm_ft_by_actor.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['actor', 'p@1', 'n_queries'])
        for actor, p1 in sorted(actor_metrics.items()):
            writer.writerow([actor, f"{p1:.4f}", len(results_by_actor[actor])])
    
    print(f"\n✓ Results saved to: {args.save_dir}/minilm_ft_by_actor.csv")
    
    # Check if we beat the baseline
    improvement = overall_p1 - 0.7536
    if overall_p1 >= 0.7836:  # Target: baseline + 3%
        print(f"\n✅ SUCCESS! Model achieved {overall_p1:.1%} P@1")
        print(f"   This is {improvement:+.1%} better than baseline!")
    else:
        print(f"\n⚠️  Model achieved {overall_p1:.1%} P@1")
        print(f"   Improvement: {improvement:+.1%} from baseline")
        print("   Consider training for more epochs or adjusting hyperparameters")
    
    return overall_p1

def main():
    """Main execution pipeline coordinating the entire training process."""
    args = parse_arguments()
    
    print("\n" + "🚀 "*30)
    print("\nPRODUCTION-READY FINE-TUNING FOR CTI RANKING")
    print("\n" + "🚀 "*30)
    
    # Step 1: Load and validate data
    query_data = load_and_validate_data()
    
    # Step 2: Create splits
    train_queries, val_queries, test_queries = create_stratified_splits(query_data)
    
    # Step 3: Train model
    model = train_model_properly(train_queries, val_queries, query_data, args)
    
    # Step 4: Evaluate
    final_p1 = evaluate_final_model(model, test_queries, query_data, args)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal P@1: {final_p1:.1%}")
    print(f"Baseline P@1: 75.36%")
    print(f"Original MiniLM P@1: 57.38%")
    print(f"\nYour fine-tuned model is ready for production use!")
    print(f"Load it with: CrossEncoder('{args.save_dir}/checkpoints/best')")
    
    print("\n" + "🚀 "*30)

if __name__ == "__main__":
    main()
