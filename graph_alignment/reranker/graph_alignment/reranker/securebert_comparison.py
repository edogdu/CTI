#!/usr/bin/env python3
"""
SecureBERT Base Model Comparison
==================================
Ablation study: does cybersecurity-specific pre-training improve
cross-encoder reranking for CTI-to-ATT&CK mapping?

This runs the EXACT same two-stage training pipeline that produced
the 94.52% P@1 result, but swaps the base model:

  Experiment A (already done): ms-marco-MiniLM-L-6-v2 (22.7M params)
    - Pre-trained on MS MARCO web search queries
    - Result: 94.52% P@1

  Experiment B (this script): ehsanaghaei/SecureBERT (110M params)
    - Pre-trained on cybersecurity text corpus
    - Result: ???

If SecureBERT improves accuracy: domain pre-training matters for CTI retrieval
If SecureBERT matches: fine-tuning overcomes the pre-training gap
If SecureBERT is worse: MS MARCO's query-document matching transfers better

All three outcomes are publishable and demonstrate rigorous methodology.

Usage (on Colab with T4 GPU):
  python securebert_comparison.py \
    --stage1-data reranker_pairs_enriched_tumeteor_v2.jsonl \
    --stage2-data reranker_pairs_enriched_v2.jsonl

Author: Shane Waldrop — Angelo State University / ARL Grant W911NF-24-2-0180
"""

import json
import random
import numpy as np
import os
import sys
import time
import argparse
from collections import defaultdict
from pathlib import Path

RANDOM_SEED = 42


def load_and_group_data(filepath):
    """Load enriched JSONL and group by query_norm."""
    query_data = defaultdict(lambda: {
        'query_raw': None, 'actor': None, 'candidates': [],
        'positive_count': 0, 'negative_count': 0,
    })
    total_rows = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qn = row.get('query_norm', '')
            qr = row.get('query_raw', '')
            ct = row.get('candidate_text', '')
            ci = row.get('candidate_id', '')
            cn = row.get('candidate_norm', '')
            label = row.get('label', 0)
            actor = row.get('actor', 'unknown')
            if not qn or not qr:
                continue
            if query_data[qn]['query_raw'] is None:
                query_data[qn]['query_raw'] = qr
                query_data[qn]['actor'] = actor if actor else 'unknown'
            query_data[qn]['candidates'].append({
                'text': ct, 'label': label,
                'id': cn if cn else ci,
            })
            if label == 1:
                query_data[qn]['positive_count'] += 1
            else:
                query_data[qn]['negative_count'] += 1
    return dict(query_data), total_rows


def create_test_split(query_data, train_ratio=0.8, val_ratio=0.1):
    """Recreate exact test split from finetune_production.py."""
    random.seed(RANDOM_SEED)
    queries_by_actor = defaultdict(list)
    for qn, data in query_data.items():
        actor = data['actor']
        if actor.startswith('external_'):
            continue
        if data['positive_count'] > 0:
            queries_by_actor[actor].append(qn)
    train_queries, val_queries, test_queries = [], [], []
    for actor, queries in queries_by_actor.items():
        random.shuffle(queries)
        n = len(queries)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_queries.extend(queries[:n_train])
        val_queries.extend(queries[n_train:n_train + n_val])
        test_queries.extend(queries[n_train + n_val:])
    random.shuffle(train_queries)
    random.shuffle(val_queries)
    random.shuffle(test_queries)
    return train_queries, val_queries, test_queries


def build_training_examples(queries, query_data, neg_ratio=2.5):
    """Build InputExample list with negative sampling."""
    from sentence_transformers import InputExample
    examples = []
    for qn in queries:
        if qn not in query_data:
            continue
        data = query_data[qn]
        qr = data['query_raw']
        positives = [c for c in data['candidates'] if c['label'] == 1]
        negatives = [c for c in data['candidates'] if c['label'] == 0]
        for c in positives:
            examples.append(InputExample(texts=[qr, c['text']], label=1.0))
        n_neg = min(int(len(positives) * neg_ratio), len(negatives))
        if n_neg > 0:
            sampled = random.sample(negatives, n_neg)
            for c in sampled:
                examples.append(InputExample(texts=[qr, c['text']], label=0.0))
    random.shuffle(examples)
    return examples


def build_evaluator(queries, query_data):
    """Build CERerankingEvaluator."""
    from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
    samples = []
    for qn in queries:
        if qn not in query_data:
            continue
        data = query_data[qn]
        pos = [c['text'] for c in data['candidates'] if c['label'] == 1]
        neg = [c['text'] for c in data['candidates'] if c['label'] == 0]
        if pos and neg:
            samples.append({'query': data['query_raw'], 'positive': pos, 'negative': neg[:10]})
    return CERerankingEvaluator(samples, name='val')


def evaluate_model(model, test_queries, query_data, label="Model"):
    """Evaluate P@1 and per-actor breakdown."""
    results_by_actor = defaultdict(list)
    all_results = []
    for qn in test_queries:
        if qn not in query_data:
            continue
        data = query_data[qn]
        texts = [[data['query_raw'], c['text']] for c in data['candidates']]
        labels = [c['label'] for c in data['candidates']]
        if not texts or sum(labels) == 0:
            continue
        scores = model.predict(texts, show_progress_bar=False)
        ranked = [labels[i] for i in np.argsort(scores)[::-1]]
        p1 = ranked[0] if ranked else 0
        h3 = 1 if 1 in ranked[:3] else 0
        h5 = 1 if 1 in ranked[:5] else 0
        r = {'p@1': p1, 'hit@3': h3, 'hit@5': h5}
        all_results.append(r)
        results_by_actor[data['actor']].append(r)

    if not all_results:
        return 0, {}

    p1 = np.mean([r['p@1'] for r in all_results])
    h3 = np.mean([r['hit@3'] for r in all_results])
    h5 = np.mean([r['hit@5'] for r in all_results])
    n_correct = sum(r['p@1'] for r in all_results)

    print(f"\n  {label} RESULTS:")
    print(f"    P@1:   {p1:.4f}  ({int(n_correct)}/{len(all_results)})")
    print(f"    Hit@3: {h3:.4f}")
    print(f"    Hit@5: {h5:.4f}")

    print(f"\n    Per-actor P@1:")
    print(f"    {'Actor':<14} | {'P@1':>8} | {'Correct':>8} | {'Total':>6}")
    print(f"    {'-'*14}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    actor_metrics = {}
    for actor in sorted(results_by_actor):
        results = results_by_actor[actor]
        c = sum(r['p@1'] for r in results)
        t = len(results)
        ap1 = c / t if t > 0 else 0
        actor_metrics[actor] = ap1
        print(f"    {actor:<14} | {ap1:>8.4f} | {int(c):>8} | {t:>6}")

    return p1, actor_metrics


def main():
    parser = argparse.ArgumentParser(description="SecureBERT base model comparison")
    parser.add_argument('--stage1-data', required=True)
    parser.add_argument('--stage2-data', required=True)
    parser.add_argument('--stage1-epochs', type=int, default=1)
    parser.add_argument('--stage2-epochs', type=int, default=2)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--output', default='./securebert_model')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  SECUREBERT BASE MODEL COMPARISON")
    print("  Ablation: does cybersecurity pre-training help CTI retrieval?")
    print("=" * 70)

    # ── The key variable: which base model to use ──
    # SecureBERT: cybersecurity domain-adapted RoBERTa (~110M params)
    # MiniLM comparison already done: 94.52% P@1
    SECUREBERT_MODEL = "ehsanaghaei/SecureBERT"

    try:
        from sentence_transformers import CrossEncoder, InputExample
        from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
        from torch.utils.data import DataLoader
        import torch
    except ImportError:
        sys.exit("[FAIL] pip install torch sentence-transformers")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device.upper()}")
    print(f"  Base model: {SECUREBERT_MODEL}")
    print(f"  Comparison baseline: ms-marco-MiniLM-L-6-v2 → 94.52% P@1")

    torch.manual_seed(RANDOM_SEED)
    if device == 'cuda':
        torch.cuda.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Load CTI-HAL data ──
    print(f"\n  Loading CTI-HAL data: {args.stage2_data}")
    ctihal_data, ctihal_rows = load_and_group_data(args.stage2_data)
    print(f"  Loaded {ctihal_rows:,} rows, {len(ctihal_data):,} queries")
    train_queries, val_queries, test_queries = create_test_split(ctihal_data)
    print(f"  Train: {len(train_queries)}  Val: {len(val_queries)}  Test: {len(test_queries)}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 1: PRE-TRAIN ON tüMETEOR (with SecureBERT base)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  STAGE 1: PRE-TRAINING ON tüMETEOR (SecureBERT base)")
    print(f"  {'='*60}")

    print(f"\n  Loading tumeteor data: {args.stage1_data}")
    tumeteor_data, tumeteor_rows = load_and_group_data(args.stage1_data)
    print(f"  Loaded {tumeteor_rows:,} rows, {len(tumeteor_data):,} queries")

    tumeteor_queries = [qn for qn, qd in tumeteor_data.items()
                        if qd['positive_count'] > 0]
    print(f"  Tumeteor queries with positives: {len(tumeteor_queries)}")

    stage1_examples = build_training_examples(tumeteor_queries, tumeteor_data,
                                               neg_ratio=1.5)
    pos_count = sum(1 for ex in stage1_examples if ex.label == 1.0)
    print(f"  Stage 1 examples: {len(stage1_examples):,} "
          f"({pos_count:,} pos, {len(stage1_examples)-pos_count:,} neg)")

    val_subset = random.sample(tumeteor_queries, min(200, len(tumeteor_queries)))
    stage1_evaluator = build_evaluator(val_subset, tumeteor_data)

    # Initialize SecureBERT as cross-encoder
    print(f"\n  Loading SecureBERT: {SECUREBERT_MODEL}")
    print(f"  (This is ~110M params vs MiniLM's 22.7M — expect slower training)")
    model = CrossEncoder(SECUREBERT_MODEL, num_labels=1, max_length=512, device=device)

    # Count parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    stage1_dataloader = DataLoader(stage1_examples, shuffle=True, batch_size=args.bs)
    stage1_output = f"{args.output}/stage1_checkpoint"
    os.makedirs(stage1_output, exist_ok=True)

    print(f"\n  Training Stage 1 ({args.stage1_epochs} epoch(s))...")
    start = time.time()

    model.fit(
        train_dataloader=stage1_dataloader,
        evaluator=stage1_evaluator,
        epochs=args.stage1_epochs,
        evaluation_steps=int(len(stage1_examples) / args.bs / 3),
        warmup_steps=int(len(stage1_examples) / args.bs * 0.1),
        output_path=stage1_output,
        save_best_model=True,
        optimizer_params={'lr': args.lr},
        weight_decay=0.01,
        show_progress_bar=True,
    )
    stage1_time = time.time() - start
    print(f"\n  Stage 1 complete in {stage1_time:.0f}s ({stage1_time/60:.1f} min)")

    # Quick eval after Stage 1
    stage1_model = CrossEncoder(stage1_output, device=device)
    stage1_p1, _ = evaluate_model(stage1_model, test_queries, ctihal_data,
                                   label="SecureBERT after Stage 1")
    del stage1_model

    # Free memory
    del tumeteor_data, stage1_examples, stage1_dataloader
    if device == 'cuda':
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # STAGE 2: FINE-TUNE ON CTI-HAL (from SecureBERT Stage 1)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  STAGE 2: FINE-TUNING ON CTI-HAL (SecureBERT base)")
    print(f"  {'='*60}")

    stage2_examples = build_training_examples(train_queries, ctihal_data, neg_ratio=2.5)
    pos_count = sum(1 for ex in stage2_examples if ex.label == 1.0)
    print(f"  Stage 2 examples: {len(stage2_examples):,} "
          f"({pos_count:,} pos, {len(stage2_examples)-pos_count:,} neg)")

    stage2_evaluator = build_evaluator(val_queries, ctihal_data)

    print(f"\n  Loading Stage 1 checkpoint: {stage1_output}")
    model = CrossEncoder(stage1_output, num_labels=1, max_length=512, device=device)

    stage2_dataloader = DataLoader(stage2_examples, shuffle=True, batch_size=args.bs)
    stage2_output = f"{args.output}/best"
    os.makedirs(stage2_output, exist_ok=True)

    print(f"\n  Training Stage 2 ({args.stage2_epochs} epochs)...")
    start = time.time()

    model.fit(
        train_dataloader=stage2_dataloader,
        evaluator=stage2_evaluator,
        epochs=args.stage2_epochs,
        evaluation_steps=int(len(stage2_examples) / args.bs / 2),
        warmup_steps=int(len(stage2_examples) / args.bs * 0.1),
        output_path=stage2_output,
        save_best_model=True,
        optimizer_params={'lr': args.lr},
        weight_decay=0.01,
        show_progress_bar=True,
    )
    stage2_time = time.time() - start
    print(f"\n  Stage 2 complete in {stage2_time:.0f}s ({stage2_time/60:.1f} min)")

    # ═══════════════════════════════════════════════════════════════
    # FINAL EVALUATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  FINAL EVALUATION: SecureBERT vs MiniLM")
    print(f"  {'='*60}")

    final_model = CrossEncoder(stage2_output, device=device)
    final_p1, actor_metrics = evaluate_model(
        final_model, test_queries, ctihal_data,
        label="SecureBERT two-stage enriched (FINAL)"
    )

    # Measure inference latency
    print(f"\n  Measuring inference latency...")
    sample_queries = test_queries[:20]
    latencies = []
    for qn in sample_queries:
        qd = ctihal_data[qn]
        texts = [[qd['query_raw'], c['text']] for c in qd['candidates']]
        t0 = time.time()
        _ = final_model.predict(texts, show_progress_bar=False)
        latencies.append((time.time() - t0) * 1000)  # ms
    avg_latency = np.mean(latencies)
    print(f"  SecureBERT avg latency: {avg_latency:.0f}ms per query")
    print(f"  MiniLM avg latency:     ~55ms per query")

    # ── Head-to-head comparison ──
    print(f"\n  {'='*60}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"  {'='*60}")

    minilm_actors = {
        'apt29': 0.9512, 'carbanak': 1.0000, 'fin6': 0.9048,
        'fin7': 1.0000, 'oilrig': 0.8235, 'sandworm': 0.8889,
        'wizardspider': 1.0000,
    }

    print(f"\n  {'Model':<35} | {'Params':>8} | {'P@1':>8} | {'Latency':>8}")
    print(f"  {'-'*35}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    print(f"  {'MiniLM (ms-marco, 6 layers)':<35} | {'22.7M':>8} | {'94.52%':>8} | {'~55ms':>8}")
    print(f"  {'SecureBERT (cyber, 12 layers)':<35} | {f'{total_params/1e6:.1f}M':>8} | "
          f"{final_p1:>7.2%} | {f'~{avg_latency:.0f}ms':>8}")

    print(f"\n  Per-actor comparison:")
    print(f"  {'Actor':<14} | {'MiniLM':>10} | {'SecureBERT':>10} | {'Winner':>10}")
    print(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    minilm_wins = 0
    secure_wins = 0
    ties = 0
    for actor in sorted(actor_metrics):
        m = minilm_actors.get(actor, 0)
        s = actor_metrics.get(actor, 0)
        if abs(m - s) < 0.001:
            winner = "TIE"
            ties += 1
        elif m > s:
            winner = "MiniLM"
            minilm_wins += 1
        else:
            winner = "SecureBERT"
            secure_wins += 1
        print(f"  {actor:<14} | {m:>9.2%} | {s:>9.2%} | {winner:>10}")

    print(f"\n  Overall: MiniLM wins {minilm_wins}, SecureBERT wins {secure_wins}, Ties {ties}")

    # ── Interpretation for the paper ──
    print(f"\n  {'='*60}")
    print(f"  INTERPRETATION FOR THE PAPER")
    print(f"  {'='*60}")

    if final_p1 > 0.9452:
        print(f"\n  SecureBERT OUTPERFORMS MiniLM ({final_p1:.2%} vs 94.52%)")
        print(f"  → Domain-adapted pre-training provides additional value")
        print(f"  → Tradeoff: {total_params/1e6:.0f}M params vs 22.7M, {avg_latency:.0f}ms vs 55ms")
    elif final_p1 > 0.93:
        print(f"\n  SecureBERT is COMPETITIVE with MiniLM ({final_p1:.2%} vs 94.52%)")
        print(f"  → Fine-tuning largely overcomes the pre-training gap")
        print(f"  → MiniLM preferred for deployment: 5x fewer params, {55/avg_latency*100:.0f}% faster")
    else:
        print(f"\n  MiniLM OUTPERFORMS SecureBERT ({final_p1:.2%} vs 94.52%)")
        print(f"  → MS MARCO's query-document matching pre-training transfers better")
        print(f"     to the retrieval task than cybersecurity domain knowledge")
        print(f"  → This supports our framing: the task is RETRIEVAL, not just NLP")

    # ── Save results ──
    results = {
        'securebert_p_at_1': round(final_p1, 4),
        'securebert_params': total_params,
        'securebert_latency_ms': round(avg_latency, 1),
        'minilm_p_at_1': 0.9452,
        'minilm_params': 22_700_000,
        'minilm_latency_ms': 55,
        'securebert_actor_metrics': {k: round(v, 4) for k, v in actor_metrics.items()},
        'minilm_actor_metrics': minilm_actors,
        'stage1_time_s': round(stage1_time, 1),
        'stage2_time_s': round(stage2_time, 1),
    }
    results_path = Path(args.output) / "comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    print(f"  Total training time: {(stage1_time+stage2_time)/60:.1f} min")
    print(f"\n  [DONE] SecureBERT comparison complete.")


if __name__ == '__main__':
    main()
