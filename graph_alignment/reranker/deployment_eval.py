#!/usr/bin/env python3
"""
Deployment-Realistic Evaluation
=================================
Measures the gap between "oracle" evaluation (gold labels guaranteed
to be in the candidate pool) and "deployment" evaluation (BM25 retrieves
candidates without gold injection).

The current pipeline injects gold labels into the BM25 candidate pool
during data generation:
    candidates_for_query = set(candidate_pool)
    candidates_for_query.update(normalized_gold)  # ← gold injection

This guarantees 100% Hit@20 by construction, but in real deployment
there's no gold to inject. This script measures what actually happens
when BM25 must find the correct technique on its own.

Three metrics compared:
  1. BM25 Recall@20: Does BM25 even retrieve the gold label?
  2. Deployment P@1:  After cross-encoder reranking of BM25-only candidates
  3. Oracle P@1:      Current 94.52% with gold injection (for comparison)

Usage:
  python deployment_eval.py

  Expects:
    enterprise-attack-v14.json           (MITRE ATT&CK STIX data)
    data/reranker_pairs_enriched_v2.jsonl (for test queries and gold labels)
    checkpoints/best_two_stage_v2/       (the 94.52% model)

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


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: BUILD ATT&CK CORPUS FROM STIX DATA
# ═══════════════════════════════════════════════════════════════════════

def clean_stix_description(desc):
    """Clean STIX description: remove citations, truncate, collapse whitespace."""
    if not desc:
        return ''
    desc = re.sub(r'\(Citation:\s*[^)]*\)', '', desc)
    desc = re.sub(r'\s+', ' ', desc).strip()
    if len(desc) > 1000:
        truncated = desc[:1000]
        last_space = truncated.rfind(' ')
        if last_space > 800:
            truncated = truncated[:last_space]
        desc = truncated + '...'
    return desc


def build_attack_corpus(stix_path):
    """Parse STIX data and build the full ATT&CK corpus for BM25 indexing.
    
    Returns a list of dicts, each with:
      - id: ATT&CK ID (e.g., T1059.001)
      - name: Human-readable name
      - description: Full cleaned description
      - enriched_text: Full text as the cross-encoder would see it
                       (matching the format in reranker_pairs_enriched_v2.jsonl)
    """
    print(f"  Loading STIX data from: {stix_path}")
    with open(stix_path, 'r', encoding='utf-8') as f:
        stix = json.load(f)

    objects = stix.get('objects', [])
    corpus = []

    for obj in objects:
        obj_type = obj.get('type', '')
        if obj.get('revoked', False) or obj.get('x_mitre_deprecated', False):
            continue

        attack_id = None
        ext_refs = obj.get('external_references', [])
        for ref in ext_refs:
            if ref.get('source_name') == 'mitre-attack':
                eid = ref.get('external_id', '')
                if re.match(r'^(T\d{4}(\.\d{3})?|TA\d{4}|S\d{4})$', eid):
                    attack_id = eid

        if not attack_id:
            continue

        if obj_type not in ('attack-pattern', 'x-mitre-tactic', 'malware', 'tool'):
            continue

        name = obj.get('name', '')
        description = clean_stix_description(obj.get('description', ''))

        # Build enriched text matching the format the cross-encoder was trained on
        # This matches what enrich_candidates.py produces
        if description:
            enriched_text = f"{attack_id} — {name}: {description}"
        else:
            enriched_text = f"{attack_id} — {name}"

        corpus.append({
            'id': attack_id,
            'name': name,
            'description': description,
            'enriched_text': enriched_text,
        })

    print(f"  Corpus size: {len(corpus)} ATT&CK entities")

    # Count by type
    techs = sum(1 for c in corpus if c['id'].startswith('T') and '.' not in c['id'])
    subs = sum(1 for c in corpus if '.' in c['id'])
    tactics = sum(1 for c in corpus if c['id'].startswith('TA'))
    software = sum(1 for c in corpus if c['id'].startswith('S'))
    print(f"    Techniques: {techs}, Sub-techniques: {subs}, "
          f"Tactics: {tactics}, Software: {software}")

    return corpus


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: BUILD BM25 INDEX
# ═══════════════════════════════════════════════════════════════════════

def build_bm25_index(corpus):
    """Build a BM25 index over the ATT&CK corpus.
    
    Uses the same parameters as the original pipeline:
      - rank_bm25 library
      - k1=1.5, b=0.75
      - Whitespace tokenization + lowercasing (no stemming/stopwords)
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        sys.exit("[FAIL] rank_bm25 not installed. Run: pip install rank_bm25")

    print(f"\n  Building BM25 index...")

    # Tokenize each document: simple whitespace split + lowercase
    # This matches the ICAIC paper: "whitespace splitting, lowercasing,
    # without stemming or stopword removal"
    tokenized_corpus = []
    for entry in corpus:
        # BM25 should index the enriched text (same content cross-encoder sees)
        tokens = entry['enriched_text'].lower().split()
        tokenized_corpus.append(tokens)

    # Build BM25 with the same parameters as the original pipeline
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

    print(f"  BM25 index built over {len(tokenized_corpus)} documents")
    print(f"  Parameters: k1=1.5, b=0.75 (matching original pipeline)")

    return bm25


def bm25_retrieve(bm25, corpus, query_text, top_k=20):
    """Retrieve top-K candidates from BM25 for a given query.
    Returns list of (attack_id, enriched_text, bm25_score) tuples."""
    # Tokenize query same way as corpus
    query_tokens = query_text.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Get top-K indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'id': corpus[idx]['id'],
            'enriched_text': corpus[idx]['enriched_text'],
            'bm25_score': float(scores[idx]),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: LOAD TEST DATA AND MODEL
# ═══════════════════════════════════════════════════════════════════════

def load_test_data(jsonl_path):
    """Load test queries and their gold labels from the enriched JSONL."""
    query_data = defaultdict(lambda: {
        'query_raw': None, 'actor': None, 'candidates': [],
        'positive_count': 0,
    })
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
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
            if actor.startswith('external_'):
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
    return dict(query_data)


def create_test_split(query_data):
    """Same split as finetune_production.py."""
    random.seed(RANDOM_SEED)
    queries_by_actor = defaultdict(list)
    for qn, data in query_data.items():
        if data['positive_count'] > 0:
            queries_by_actor[data['actor']].append(qn)
    test_queries = []
    for actor, queries in queries_by_actor.items():
        random.shuffle(queries)
        n = len(queries)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        test_queries.extend(queries[n_train + n_val:])
    random.shuffle(test_queries)
    return test_queries


def extract_id(raw_id):
    match = ATTACK_ID_RE.match(raw_id)
    return match.group(1) if match else raw_id


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: DEPLOYMENT-REALISTIC EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  DEPLOYMENT-REALISTIC EVALUATION")
    print("  Measuring the gap between oracle and deployment performance")
    print("=" * 70)

    stix_path = "enterprise-attack-v14.json"
    data_path = "data/reranker_pairs_enriched_v2.jsonl"
    model_path = "checkpoints/best_two_stage_v2"

    for p, name in [(stix_path, "STIX"), (data_path, "Data"), (model_path, "Model")]:
        if not Path(p).exists():
            sys.exit(f"[FAIL] {name} not found: {p}")

    # ── Build ATT&CK corpus and BM25 index ──
    print()
    corpus = build_attack_corpus(stix_path)
    bm25 = build_bm25_index(corpus)

    # Build ID→enriched_text lookup for cross-encoder input
    id_to_text = {entry['id']: entry['enriched_text'] for entry in corpus}

    # ── Load test data ──
    print(f"\n  Loading test data from: {data_path}")
    query_data = load_test_data(data_path)
    test_queries = create_test_split(query_data)
    print(f"  Test queries: {len(test_queries)}")

    # ── Load cross-encoder ──
    print(f"  Loading model from: {model_path}")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_path)
    print(f"  Model loaded.")

    # ── Run evaluation ──
    print(f"\n  {'='*60}")
    print(f"  RUNNING DEPLOYMENT EVALUATION")
    print(f"  {'='*60}")

    TOP_K = 20
    oracle_correct = 0
    deploy_correct = 0
    bm25_recall_hits = 0
    bm25_recall_total = 0
    total = 0

    # Per-actor tracking
    oracle_by_actor = defaultdict(list)
    deploy_by_actor = defaultdict(list)
    recall_by_actor = defaultdict(lambda: {'hit': 0, 'total': 0})

    # Detailed miss analysis
    deploy_misses = []
    bm25_misses = []

    start_time = time.time()

    for i, qn in enumerate(test_queries):
        qd = query_data[qn]
        if qd['positive_count'] == 0:
            continue

        query_raw = qd['query_raw']
        actor = qd['actor']

        # Get gold IDs from the JSONL (these are the "correct" answers)
        gold_ids = set()
        for c in qd['candidates']:
            if c['label'] == 1:
                gold_ids.add(extract_id(c['id']))

        if not gold_ids:
            continue

        total += 1

        # ── Oracle evaluation (current method: BM25 + gold injection) ──
        oracle_texts = [[query_raw, c['text']] for c in qd['candidates']]
        oracle_labels = [c['label'] for c in qd['candidates']]
        oracle_scores = model.predict(oracle_texts, show_progress_bar=False)
        oracle_ranked = [oracle_labels[j] for j in np.argsort(oracle_scores)[::-1]]
        oracle_hit = oracle_ranked[0] == 1 if oracle_ranked else False
        if oracle_hit:
            oracle_correct += 1
        oracle_by_actor[actor].append(oracle_hit)

        # ── BM25-only retrieval (deployment mode) ──
        bm25_results = bm25_retrieve(bm25, corpus, query_raw, top_k=TOP_K)

        # Check BM25 recall: does any gold ID appear in BM25's top-K?
        bm25_ids = set(r['id'] for r in bm25_results)
        gold_in_bm25 = gold_ids & bm25_ids

        for g in gold_ids:
            bm25_recall_total += 1
            if g in bm25_ids:
                bm25_recall_hits += 1
                recall_by_actor[actor]['hit'] += 1
            recall_by_actor[actor]['total'] += 1

        any_gold_in_bm25 = len(gold_in_bm25) > 0

        # ── Deployment evaluation (BM25-only candidates through cross-encoder) ──
        deploy_texts = [[query_raw, r['enriched_text']] for r in bm25_results]
        deploy_scores = model.predict(deploy_texts, show_progress_bar=False)

        # Check if top-1 after reranking is a gold label
        deploy_ranked_indices = np.argsort(deploy_scores)[::-1]
        deploy_top1_id = bm25_results[deploy_ranked_indices[0]]['id']
        deploy_hit = deploy_top1_id in gold_ids

        if deploy_hit:
            deploy_correct += 1
        deploy_by_actor[actor].append(deploy_hit)

        # Track misses
        if not deploy_hit and oracle_hit:
            deploy_misses.append({
                'query': query_raw[:80],
                'actor': actor,
                'gold_ids': sorted(gold_ids),
                'deploy_top1': deploy_top1_id,
                'gold_in_bm25': any_gold_in_bm25,
                'n_gold_in_bm25': len(gold_in_bm25),
                'n_gold_total': len(gold_ids),
            })

        if not any_gold_in_bm25:
            bm25_misses.append({
                'query': query_raw[:80],
                'actor': actor,
                'gold_ids': sorted(gold_ids),
                'bm25_top5': [r['id'] for r in bm25_results[:5]],
            })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"    [{i+1}/{len(test_queries)}] {elapsed:.0f}s elapsed")

    elapsed = time.time() - start_time
    print(f"\n  Evaluation complete in {elapsed:.0f}s")

    # ═══════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════

    oracle_p1 = oracle_correct / total if total > 0 else 0
    deploy_p1 = deploy_correct / total if total > 0 else 0
    bm25_recall = bm25_recall_hits / bm25_recall_total if bm25_recall_total > 0 else 0
    gap = oracle_p1 - deploy_p1

    print(f"\n  {'='*60}")
    print(f"  RESULTS: ORACLE vs DEPLOYMENT")
    print(f"  {'='*60}")

    print(f"\n  {'Metric':<35} | {'Value':>10}")
    print(f"  {'-'*35}-+-{'-'*10}")
    print(f"  {'BM25 Recall@20 (per gold label)':<35} | {bm25_recall:>9.2%}")
    print(f"  {'Oracle P@1 (gold injected)':<35} | {oracle_p1:>9.2%}")
    print(f"  {'Deployment P@1 (BM25 only)':<35} | {deploy_p1:>9.2%}")
    print(f"  {'GAP (oracle - deployment)':<35} | {gap:>9.2%}")

    # Per-actor breakdown
    print(f"\n  Per-actor comparison:")
    print(f"  {'Actor':<14} | {'BM25 Recall':>12} | {'Oracle P@1':>10} | "
          f"{'Deploy P@1':>10} | {'Gap':>8}")
    print(f"  {'-'*14}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    for actor in sorted(oracle_by_actor):
        o = np.mean(oracle_by_actor[actor]) if oracle_by_actor[actor] else 0
        d = np.mean(deploy_by_actor[actor]) if deploy_by_actor[actor] else 0
        ra = recall_by_actor[actor]
        r = ra['hit'] / ra['total'] if ra['total'] > 0 else 0
        g = o - d
        print(f"  {actor:<14} | {r:>11.2%} | {o:>9.2%} | {d:>9.2%} | {g:>+7.2%}")

    # BM25 misses detail
    if bm25_misses:
        print(f"\n  {'='*60}")
        print(f"  QUERIES WHERE BM25 MISSED ALL GOLD LABELS ({len(bm25_misses)})")
        print(f"  {'='*60}")
        for miss in bm25_misses[:10]:
            print(f"\n    Query: {miss['query']}")
            print(f"    Actor: {miss['actor']}")
            print(f"    Gold:  {', '.join(miss['gold_ids'])}")
            print(f"    BM25 top-5: {', '.join(miss['bm25_top5'])}")

    # Deployment-only misses (oracle got it right, deployment didn't)
    if deploy_misses:
        print(f"\n  {'='*60}")
        print(f"  QUERIES WHERE DEPLOYMENT FAILS BUT ORACLE SUCCEEDS ({len(deploy_misses)})")
        print(f"  {'='*60}")
        for miss in deploy_misses[:10]:
            gold_status = "in BM25" if miss['gold_in_bm25'] else "NOT in BM25"
            print(f"\n    Query:       {miss['query']}")
            print(f"    Actor:       {miss['actor']}")
            print(f"    Gold:        {', '.join(miss['gold_ids'])} ({gold_status})")
            print(f"    Deploy top1: {miss['deploy_top1']}")
            if not miss['gold_in_bm25']:
                print(f"    → CAUSE: BM25 didn't retrieve gold label")
            else:
                print(f"    → CAUSE: Cross-encoder ranked wrong candidate higher")

    # ── Interpretation ──
    print(f"\n  {'='*60}")
    print(f"  INTERPRETATION")
    print(f"  {'='*60}")

    if gap < 0.02:
        print(f"\n  Gap is SMALL ({gap:.2%}). BM25 retrieval is NOT a bottleneck.")
        print(f"  The system performs nearly identically in deployment as in evaluation.")
        print(f"  Cross-encoder improvements remain the primary optimization target.")
    elif gap < 0.05:
        print(f"\n  Gap is MODERATE ({gap:.2%}). BM25 misses some gold labels,")
        print(f"  causing {int(gap * total)} additional errors in deployment.")
        print(f"  Consider expanding K from 20 to 50 or adding hybrid retrieval.")
    else:
        print(f"\n  Gap is LARGE ({gap:.2%}). BM25 is a significant bottleneck.")
        print(f"  {int(gap * total)} queries that succeed with gold injection fail")
        print(f"  in deployment because BM25 doesn't retrieve the correct technique.")
        print(f"  PRIORITY: Implement hybrid retrieval (BM25 + bi-encoder) before")
        print(f"  any further cross-encoder optimization.")

    # ── Save results ──
    out_dir = Path("deployment_eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'bm25_recall_at_20': round(bm25_recall, 4),
        'oracle_p_at_1': round(oracle_p1, 4),
        'deployment_p_at_1': round(deploy_p1, 4),
        'gap': round(gap, 4),
        'bm25_misses': len(bm25_misses),
        'deployment_only_misses': len(deploy_misses),
        'total_queries': total,
        'corpus_size': len(corpus),
    }

    with open(out_dir / "deployment_eval.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {out_dir / 'deployment_eval.json'}")
    print(f"\n  [DONE] Deployment-realistic evaluation complete.")


if __name__ == '__main__':
    main()
