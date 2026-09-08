#!/usr/bin/env python3
"""
Full-Corpus Cross-Encoder Evaluation (No First-Stage Filter)
==============================================================
Tests the CTI-HAL test queries against ALL 625 ATT&CK techniques,
bypassing the candidate pool entirely. This answers Bill's question
from the April 9 meeting:

  "Take the data set, ignore the candidate pool and measure it
   against every [technique]. We could just start with every
   technique and see if it works or maybe it doesn't work."

This quantifies the value of the first-stage retrieval:
  - With 20-candidate pool (normal):     94.52% P@1
  - Without pool (this script):          ???% P@1

If the cross-encoder still performs well without pre-filtering,
it could potentially be used as a standalone classifier. If it
drops significantly, that validates the retrieve-then-rerank
architecture — the first stage narrows the search space, and
the cross-encoder discriminates within it.

Designed to run on Bill's RTX 4090. Should take 5-15 minutes
depending on GPU speed.

Usage:
  python full_corpus_eval.py

  Expects (all relative to this directory):
    enterprise-attack-v14.json          # MITRE STIX data
    data/reranker_pairs_enriched_v2.jsonl   # Training/test data
    checkpoints/best_two_stage_v2/      # Best model checkpoint

Author: Shane Waldrop — Angelo State University / ARL Grant W911NF-24-2-0180
"""

import json
import random
import re
import sys
import time
import numpy as np
from collections import defaultdict
from pathlib import Path

RANDOM_SEED = 42
ATTACK_ID_RE = re.compile(r'(TA\d{4}|T\d{4}(?:\.\d{3})?|S\d{4})')


def extract_id(raw_id):
    """Pull the ATT&CK ID from a candidate identifier string."""
    match = ATTACK_ID_RE.match(raw_id)
    return match.group(1) if match else raw_id


def clean_stix_text(text):
    """Strip markdown links, citations, and noise from STIX descriptions."""
    if not text:
        return ''
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\(Citation:\s*[^)]*\)', '', text)
    text = re.sub(r'https?://\S+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def build_technique_corpus(stix_path):
    """Build technique ID -> enriched text mapping from STIX data.
    
    This mirrors what enrich_candidates.py does: each technique gets
    its full description so the cross-encoder has rich semantic content
    to compare against the CTI query.
    """
    with open(stix_path, 'r', encoding='utf-8') as f:
        stix = json.load(f)

    corpus = {}
    for obj in stix['objects']:
        if obj.get('revoked', False) or obj.get('x_mitre_deprecated', False):
            continue

        obj_type = obj.get('type', '')
        attack_id = None
        name = obj.get('name', '')
        desc = clean_stix_text(obj.get('description', ''))

        if obj_type == 'attack-pattern':
            for ref in obj.get('external_references', []):
                if ref.get('source_name') == 'mitre-attack':
                    eid = ref.get('external_id', '')
                    if re.match(r'^T\d{4}(\.\d{3})?$', eid):
                        attack_id = eid

        elif obj_type == 'x-mitre-tactic':
            for ref in obj.get('external_references', []):
                if ref.get('source_name') == 'mitre-attack':
                    eid = ref.get('external_id', '')
                    if re.match(r'^TA\d{4}$', eid):
                        attack_id = eid

        elif obj_type in ('malware', 'tool'):
            for ref in obj.get('external_references', []):
                if ref.get('source_name') == 'mitre-attack':
                    eid = ref.get('external_id', '')
                    if re.match(r'^S\d{4}$', eid):
                        attack_id = eid

        if attack_id and attack_id not in corpus:
            if desc:
                enriched = f"{attack_id} — {name}: {desc}"
            else:
                enriched = f"{attack_id} — {name}"
            corpus[attack_id] = enriched

    return corpus


def load_test_queries(data_path):
    """Load the enriched JSONL data and extract test split queries
    with their gold labels, using the same split as finetune_production.py.
    """
    # Group by query_norm
    query_data = defaultdict(lambda: {
        'query_raw': None, 'actor': None, 'gold_ids': set(),
    })
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qn = row.get('query_norm', '')
            qr = row.get('query_raw', '')
            cn = row.get('candidate_norm', '') or row.get('candidate_id', '')
            label = row.get('label', 0)
            actor = row.get('actor', 'unknown')
            if not qn or not qr or actor.startswith('external_'):
                continue
            if query_data[qn]['query_raw'] is None:
                query_data[qn]['query_raw'] = qr
                query_data[qn]['actor'] = actor
            if label == 1:
                cid = extract_id(cn)
                query_data[qn]['gold_ids'].add(cid)

    # Reproduce the same train/val/test split
    random.seed(RANDOM_SEED)
    queries_by_actor = defaultdict(list)
    for qn, data in query_data.items():
        if data['gold_ids']:
            queries_by_actor[data['actor']].append(qn)

    test_queries = []
    for actor, queries in queries_by_actor.items():
        random.shuffle(queries)
        n = len(queries)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        test_queries.extend(queries[n_train + n_val:])
    random.shuffle(test_queries)

    # Build test set: list of (query_text, gold_ids, actor)
    test_set = []
    for qn in test_queries:
        qd = query_data[qn]
        if qd['gold_ids']:
            test_set.append({
                'query': qd['query_raw'],
                'gold_ids': qd['gold_ids'],
                'actor': qd['actor'],
            })

    return test_set


def main():
    print("\n" + "=" * 70)
    print("  FULL-CORPUS CROSS-ENCODER EVALUATION")
    print("  CTI-HAL test queries scored against ALL ATT&CK entities")
    print("  No first-stage candidate filtering")
    print("=" * 70)

    # ── Check paths ──
    stix_path = "enterprise-attack-v14.json"
    data_path = "data/reranker_pairs_enriched_v2.jsonl"
    model_path = "checkpoints/best_two_stage_v2"

    for p, name in [(stix_path, "STIX"), (data_path, "Data"), (model_path, "Model")]:
        if not Path(p).exists():
            sys.exit(f"[FAIL] {name} not found: {p}\n"
                     f"       Run from: graph_alignment/reranker/")

    # ── Build full technique corpus ──
    print(f"\n  Building technique corpus from: {stix_path}")
    corpus = build_technique_corpus(stix_path)
    print(f"  Total ATT&CK entities: {len(corpus)}")

    # Break down by type
    techniques = [k for k in corpus if k.startswith('T')]
    tactics = [k for k in corpus if k.startswith('TA')]
    software = [k for k in corpus if k.startswith('S')]
    print(f"    Techniques: {len(techniques)}")
    print(f"    Tactics:    {len(tactics)}")
    print(f"    Software:   {len(software)}")

    # ── Load test queries ──
    print(f"\n  Loading test queries from: {data_path}")
    test_set = load_test_queries(data_path)
    print(f"  Test queries: {len(test_set)}")

    # Collect all gold IDs to check corpus coverage
    all_gold = set()
    for t in test_set:
        all_gold.update(t['gold_ids'])
    gold_in_corpus = all_gold & set(corpus.keys())
    gold_missing = all_gold - set(corpus.keys())
    print(f"  Unique gold IDs: {len(all_gold)}")
    print(f"  Gold IDs in corpus: {len(gold_in_corpus)}/{len(all_gold)}")
    if gold_missing:
        print(f"  Gold IDs NOT in corpus: {gold_missing}")
        print(f"  (These will be automatic misses)")

    # ── Load model ──
    print(f"\n  Loading model from: {model_path}")
    import torch
    from sentence_transformers import CrossEncoder

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device.upper()}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    model = CrossEncoder(model_path, device=device)
    print(f"  Model loaded.")

    # ── Build candidate arrays ──
    entity_ids = sorted(corpus.keys())
    entity_texts = [corpus[eid] for eid in entity_ids]
    total_pairs = len(test_set) * len(entity_ids)

    print(f"\n  Entities per query: {len(entity_ids)}")
    print(f"  Total pairs to score: {total_pairs:,}")

    # ── Evaluate ──
    print(f"\n  {'='*60}")
    print(f"  RUNNING FULL-CORPUS EVALUATION")
    print(f"  {'='*60}")

    correct_at_1 = 0
    correct_at_3 = 0
    correct_at_5 = 0
    total = 0
    results_by_actor = defaultdict(lambda: {'correct': 0, 'total': 0})
    start_time = time.time()

    for i, test in enumerate(test_set):
        query = test['query']
        gold_ids = test['gold_ids']
        actor = test['actor']

        # Score query against ALL entities
        pairs = [[query, t] for t in entity_texts]
        scores = model.predict(pairs, show_progress_bar=False)

        # Rank by score (highest first)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_ids = [entity_ids[idx] for idx in ranked_indices]

        # Check if ANY gold ID is in top-K
        total += 1
        top1_hit = ranked_ids[0] in gold_ids
        top3_hit = any(rid in gold_ids for rid in ranked_ids[:3])
        top5_hit = any(rid in gold_ids for rid in ranked_ids[:5])

        if top1_hit:
            correct_at_1 += 1
        if top3_hit:
            correct_at_3 += 1
        if top5_hit:
            correct_at_5 += 1

        results_by_actor[actor]['total'] += 1
        if top1_hit:
            results_by_actor[actor]['correct'] += 1

        # Progress reporting every 10 queries
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(test_set) - i - 1) / rate
            p1 = correct_at_1 / total
            print(f"    [{i+1}/{len(test_set)}] {elapsed:.0f}s elapsed, "
                  f"~{remaining:.0f}s remaining, "
                  f"P@1: {p1:.2%}")

    elapsed = time.time() - start_time

    # ── Results ──
    p1 = correct_at_1 / total
    h3 = correct_at_3 / total
    h5 = correct_at_5 / total

    print(f"\n  {'='*60}")
    print(f"  RESULTS: FULL-CORPUS EVALUATION (NO CANDIDATE FILTERING)")
    print(f"  {'='*60}")

    print(f"\n  P@1:   {p1:.4f} ({correct_at_1}/{total})")
    print(f"  Hit@3: {h3:.4f} ({correct_at_3}/{total})")
    print(f"  Hit@5: {h5:.4f} ({correct_at_5}/{total})")
    print(f"  Time:  {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Per-actor breakdown ──
    print(f"\n  Per-actor P@1:")
    print(f"  {'Actor':<14} | {'Full Corpus':>12} | {'With Pool':>12} | {'Queries':>8}")
    print(f"  {'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")

    # Hardcoded pool-filtered results for comparison
    pool_results = {
        'apt29': 0.9512, 'carbanak': 1.0000, 'fin6': 0.9048,
        'fin7': 1.0000, 'oilrig': 0.8235, 'sandworm': 0.8889,
        'wizardspider': 1.0000,
    }

    for actor in sorted(results_by_actor):
        ar = results_by_actor[actor]
        actor_p1 = ar['correct'] / ar['total'] if ar['total'] > 0 else 0
        pool_p1 = pool_results.get(actor, 0)
        print(f"  {actor:<14} | {actor_p1:>11.2%} | {pool_p1:>11.2%} | {ar['total']:>8}")

    # ── Architecture impact analysis ──
    print(f"\n  {'='*60}")
    print(f"  ARCHITECTURE IMPACT: First-Stage Retrieval Value")
    print(f"  {'='*60}")

    pool_p1 = 0.9452
    delta = pool_p1 - p1

    print(f"\n  With 20-candidate pool:    {pool_p1:.2%} P@1")
    print(f"  Without pool (all {len(entity_ids)}):  {p1:.2%} P@1")
    print(f"  Difference:                {delta:+.2%}")

    if delta > 0.05:
        print(f"\n  INTERPRETATION: The first-stage retrieval provides a")
        print(f"  significant {delta:.1%} boost. The retrieve-then-rerank")
        print(f"  architecture is validated — the cross-encoder works best")
        print(f"  when given a pre-filtered candidate set.")
    elif delta > 0:
        print(f"\n  INTERPRETATION: The first-stage retrieval provides a")
        print(f"  modest {delta:.1%} boost. The cross-encoder is robust")
        print(f"  even without pre-filtering, but still benefits from it.")
    else:
        print(f"\n  INTERPRETATION: The cross-encoder performs as well or")
        print(f"  better without pre-filtering. The first-stage retrieval")
        print(f"  may not be necessary for this dataset size.")

    # ── Save results ──
    out_dir = Path("full_corpus_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'evaluation': 'Full-corpus cross-encoder (no first-stage filter)',
        'test_queries': total,
        'candidate_pool_size': len(entity_ids),
        'p_at_1': round(p1, 4),
        'hit_at_3': round(h3, 4),
        'hit_at_5': round(h5, 4),
        'evaluation_time_s': round(elapsed, 1),
        'device': device,
        'comparison': {
            'with_pool_p1': 0.9452,
            'without_pool_p1': round(p1, 4),
            'delta': round(delta, 4),
        },
        'per_actor': {
            actor: {
                'full_corpus_p1': round(r['correct'] / r['total'], 4) if r['total'] > 0 else 0,
                'with_pool_p1': pool_results.get(actor, 0),
                'queries': r['total'],
            }
            for actor, r in results_by_actor.items()
        },
    }

    with open(out_dir / "full_corpus_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {out_dir / 'full_corpus_results.json'}")
    print(f"\n  [DONE] Full-corpus evaluation complete.")


if __name__ == '__main__':
    main()
