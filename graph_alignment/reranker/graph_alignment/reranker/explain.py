#!/usr/bin/env python3
"""
Explainability Module for CTI Cross-Encoder Reranking Pipeline
===============================================================
Provides multi-layered explanations for technique predictions:

  Layer 1 - Token Importance (Leave-One-Out perturbation)
            Which query words most influenced the neural match?

  Layer 2 - BM25 Lexical Overlap
            Which query terms directly appear in the technique description?

  Layer 3 - Score Margins (Confidence)
            How far ahead is the top prediction from the runner-up?

  Layer 4 - ATT&CK Hierarchy
            Where does the predicted technique sit in the taxonomy?

  Layer 5 - Alternative Candidates
            What else did the model consider?

Usage:
  # Batch mode - explain all 146 test queries
  python explain.py

  # Single-query mode - explain one specific query
  python explain.py --query "PowerShell commands with base64 encoding"

  # Demo mode - show curated examples (correct, incorrect, borderline)
  python explain.py --demo

  # Custom model/data paths
  python explain.py --model checkpoints/best_two_stage --data data/reranker_pairs_enriched.jsonl

Method:
  Leave-One-Out (LOO) input perturbation. For each word in the query,
  we remove it and measure how much the cross-encoder's relevance score
  changes. A large drop means that word was critical to the match.

  Computational cost: ~15 forward passes per query (one per word).
  On CPU (MiniLM, 22.7M params): ~75ms per explanation.

Reference:
  - Li et al. (2016). "Understanding Neural Networks through Representation Erasure"
  - Zintgraf et al. (2017). "Visualizing Deep Neural Network Decisions"

Author: Shane Waldrop - Angelo State University / DoD CTI Research
"""

import json
import random
import re
import csv
import argparse
import sys
import os
import time
import numpy as np
from collections import defaultdict
from pathlib import Path

# --- Configuration ---
RANDOM_SEED = 42
DEFAULT_MODEL = './checkpoints/best_two_stage'
DEFAULT_DATA  = './data/reranker_pairs_enriched.jsonl'
OUTPUT_DIR    = './eval_results'

# ATT&CK ID pattern
ATTACK_ID_RE = re.compile(r'^(TA\d{4}|T\d{4}(?:\.\d{3})?|S\d{4})$')

# Stopwords to exclude from BM25 overlap
STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'and', 'but', 'or', 'nor',
    'not', 'so', 'than', 'that', 'this', 'these', 'those', 'it', 'its',
    'if', 'each', 'which', 'their', 'there', 'they', 'them', 'such', 'also',
    'when', 'where', 'who', 'what', 'how', 'all', 'both', 'other', 'more',
    'most', 'only', 'very', 'just', 'about', 'up', 'down', 'no', 'any',
    'some', 'using', 'used', 'uses', 'use',
})


# --- Data Loading (mirrors finetune_production.py exactly) ---

def load_and_group_data(filepath):
    """Load JSONL and group by query_norm. Same logic as finetune_production.py."""
    query_data = defaultdict(lambda: {
        'query_raw': None,
        'actor': None,
        'candidates': [],
        'positive_count': 0,
        'negative_count': 0,
    })

    total_rows = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            query_norm = row.get('query_norm', '')
            query_raw  = row.get('query_raw', '')
            candidate_text = row.get('candidate_text', '')
            candidate_id   = row.get('candidate_id', '')
            candidate_norm = row.get('candidate_norm', '')
            label = row.get('label', 0)
            actor = row.get('actor', 'unknown')

            if not query_norm or not query_raw:
                continue

            if query_data[query_norm]['query_raw'] is None:
                query_data[query_norm]['query_raw'] = query_raw
                query_data[query_norm]['actor'] = actor if actor else 'unknown'

            query_data[query_norm]['candidates'].append({
                'text': candidate_text,
                'label': label,
                'id': candidate_norm if candidate_norm else candidate_id,
                'raw_id': candidate_id,
            })

            if label == 1:
                query_data[query_norm]['positive_count'] += 1
            else:
                query_data[query_norm]['negative_count'] += 1

    return dict(query_data), total_rows


def create_test_split(query_data, train_ratio=0.8, val_ratio=0.1):
    """
    Recreate the exact same test split as finetune_production.py.
    CRITICAL: must use same seed and same logic to get the same 146 queries.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    queries_by_actor = defaultdict(list)
    for query_norm, data in query_data.items():
        actor = data['actor']
        if actor.startswith('external_'):
            continue
        if data['positive_count'] > 0:
            queries_by_actor[actor].append(query_norm)

    test_queries = []
    for actor, queries in queries_by_actor.items():
        random.shuffle(queries)
        n = len(queries)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        test_queries.extend(queries[n_train + n_val:])

    random.shuffle(test_queries)
    return test_queries


# --- ATT&CK Hierarchy ---

def extract_attack_id(candidate_id_str):
    """Extract ATT&CK ID from candidate_norm string."""
    m = re.match(r'(TA\d{4}|T\d{4}(?:\.\d{3})?|S\d{4})', candidate_id_str)
    return m.group(1) if m else None


def build_attack_hierarchy(query_data):
    """Build parent/child/sibling relationships from the candidate pool."""
    all_ids = set()
    id_to_name = {}

    for qd in query_data.values():
        for c in qd['candidates']:
            aid = extract_attack_id(c['id'])
            if aid:
                all_ids.add(aid)
                if aid not in id_to_name:
                    id_to_name[aid] = c['id']

    hierarchy = {}
    for aid in all_ids:
        entry = {'parent': None, 'children': [], 'siblings': [], 'type': 'unknown'}
        if aid.startswith('TA'):
            entry['type'] = 'tactic'
        elif aid.startswith('S'):
            entry['type'] = 'software'
        elif '.' in aid:
            entry['type'] = 'sub-technique'
            entry['parent'] = aid.split('.')[0]
        else:
            entry['type'] = 'technique'
        hierarchy[aid] = entry

    # Populate children and siblings
    for aid, entry in hierarchy.items():
        if entry['parent'] and entry['parent'] in hierarchy:
            parent = hierarchy[entry['parent']]
            if aid not in parent['children']:
                parent['children'].append(aid)

    for aid, entry in hierarchy.items():
        if entry['parent'] and entry['parent'] in hierarchy:
            entry['siblings'] = [c for c in hierarchy[entry['parent']]['children'] if c != aid]

    return hierarchy, id_to_name


# --- Explainability Core ---

def compute_token_importance(model, query, candidate_text):
    """
    Leave-One-Out token importance.

    For each word in the query, remove it and measure the score change.
    importance(word) = score(full_query) - score(query_without_word)

    Positive = word SUPPORTS the match (removing it hurts the score)
    Negative = word HURTS the match (removing it helps the score)
    """
    baseline = float(model.predict([(query, candidate_text)], show_progress_bar=False)[0])

    words = query.split()
    if len(words) <= 1:
        return {0: {'word': words[0] if words else '', 'importance': 0.0}}, baseline

    # Batch all perturbations for efficiency
    perturbed_pairs = []
    for i in range(len(words)):
        perturbed = ' '.join(words[:i] + words[i+1:])
        perturbed_pairs.append((perturbed, candidate_text))

    perturbed_scores = model.predict(perturbed_pairs, show_progress_bar=False)

    importances = {}
    for i, word in enumerate(words):
        importances[i] = {'word': word, 'importance': baseline - float(perturbed_scores[i])}

    return importances, baseline


def compute_bm25_overlap(query, candidate_text):
    """Find content-word overlap between query and candidate."""
    def tokenize(text):
        return set(re.findall(r'[a-z0-9]+', text.lower()))
    q_tokens = tokenize(query) - STOPWORDS
    c_tokens = tokenize(candidate_text) - STOPWORDS
    return sorted(q_tokens & c_tokens)


def classify_importance(norm_value):
    v = abs(norm_value)
    if v >= 0.5:   return 'HIGH'
    if v >= 0.2:   return 'MEDIUM'
    if v >= 0.05:  return 'LOW'
    return 'NONE'


def explain_single_query(model, query_raw, candidates, hierarchy=None, id_to_name=None):
    """Generate a complete multi-layered explanation for one query."""

    # Layer 0: Score all candidates and rank
    pairs = [(query_raw, c['text']) for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    top1_cand, top1_score = ranked[0]
    top1_id_str = top1_cand['id']
    top1_id = extract_attack_id(top1_id_str)

    gold_cands = [c for c in candidates if c['label'] == 1]
    gold_id_str = gold_cands[0]['id'] if gold_cands else '?'
    gold_id = extract_attack_id(gold_id_str)
    is_correct = top1_cand['label'] == 1

    # Layer 1: Token Importance (LOO)
    importances, baseline = compute_token_importance(model, query_raw, top1_cand['text'])

    max_abs = max((abs(v['importance']) for v in importances.values()), default=1.0)
    if max_abs == 0:
        max_abs = 1.0

    token_importance = []
    for idx in sorted(importances.keys()):
        entry = importances[idx]
        norm = entry['importance'] / max_abs
        token_importance.append({
            'position': idx,
            'token': entry['word'],
            'raw_importance': round(entry['importance'], 6),
            'normalized': round(norm, 4),
            'impact': classify_importance(norm),
        })

    # Layer 2: BM25 Lexical Overlap
    bm25_matches = compute_bm25_overlap(query_raw, top1_cand['text'])

    # Layer 3: Score Margin / Confidence
    margin = float(top1_score - ranked[1][1]) if len(ranked) > 1 else 0.0
    if margin >= 0.30:    confidence = 'HIGH'
    elif margin >= 0.10:  confidence = 'MEDIUM'
    else:                 confidence = 'LOW'

    # Layer 4: ATT&CK Hierarchy
    hier_info = {}
    if hierarchy and top1_id and top1_id in hierarchy:
        h = hierarchy[top1_id]
        hier_info = {
            'type': h.get('type', 'unknown'),
            'parent': h.get('parent'),
            'parent_name': id_to_name.get(h['parent'], '') if h.get('parent') else None,
            'children': h.get('children', [])[:5],
            'siblings': h.get('siblings', [])[:5],
        }

    # Layer 5: Alternatives (top 5)
    alternatives = []
    for cand, score in ranked[:5]:
        alternatives.append({
            'rank': len(alternatives) + 1,
            'id': cand['id'],
            'score': round(float(score), 6),
            'is_gold': cand['label'] == 1,
            'text_preview': cand['text'][:100],
        })

    return {
        'query': query_raw,
        'prediction': {
            'id': top1_id_str,
            'attack_id': top1_id,
            'score': round(float(top1_score), 6),
            'correct': is_correct,
        },
        'gold': {
            'id': gold_id_str,
            'attack_id': gold_id,
        },
        'confidence': confidence,
        'margin': round(margin, 6),
        'token_importance': token_importance,
        'bm25_overlap': bm25_matches,
        'hierarchy': hier_info,
        'alternatives': alternatives,
    }


# --- Display Formatting ---

def format_bar(value, width=16):
    """Visual bar for importance values."""
    filled = int(abs(value) * width)
    filled = min(filled, width)
    return '#' * filled + '.' * (width - filled)


def print_explanation(expl, index=None):
    """Pretty-print a single explanation to console."""
    correct_marker = 'CORRECT' if expl['prediction']['correct'] else 'MISS'

    print()
    print('=' * 80)
    if index is not None:
        print(f"  EXPLANATION #{index}  [{correct_marker}]")
    else:
        print(f"  EXPLANATION  [{correct_marker}]")
    print('=' * 80)

    print(f"\n  Query:     {expl['query']}")
    print(f"  Predicted: {expl['prediction']['id']}")
    print(f"  Gold:      {expl['gold']['id']}")
    print(f"  Score:     {expl['prediction']['score']:.4f}")
    print(f"  Margin:    {expl['margin']:.4f}  (Confidence: {expl['confidence']})")

    # Token importance
    print(f"\n  TOKEN IMPORTANCE (Leave-One-Out):")
    print(f"  {'Token':<20} | {'Importance':>10} | {'Bar':<18} | Impact")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*18}-+--------")

    sorted_tokens = sorted(expl['token_importance'],
                           key=lambda x: abs(x['raw_importance']), reverse=True)
    for t in sorted_tokens:
        sign = '+' if t['raw_importance'] >= 0 else ''
        bar = format_bar(t['normalized'])
        print(f"  {t['token']:<20} | {sign}{t['raw_importance']:>9.4f} | {bar} | {t['impact']}")

    # BM25 overlap
    matches_str = ', '.join(expl['bm25_overlap']) if expl['bm25_overlap'] else '(none)'
    print(f"\n  BM25 LEXICAL MATCHES: {matches_str}")

    # Hierarchy
    if expl['hierarchy']:
        h = expl['hierarchy']
        parts = [f"Type: {h['type']}"]
        if h.get('parent'):
            pname = f" ({h['parent_name'][:50]})" if h.get('parent_name') else ''
            parts.append(f"Parent: {h['parent']}{pname}")
        if h.get('siblings'):
            parts.append(f"Siblings: {', '.join(h['siblings'][:3])}")
        print(f"\n  ATT&CK CONTEXT: {' | '.join(parts)}")

    # Alternatives
    print(f"\n  TOP CANDIDATES:")
    for alt in expl['alternatives'][:5]:
        gold_mark = ' <-- GOLD' if alt['is_gold'] else ''
        sel_mark  = ' ***' if alt['rank'] == 1 else '    '
        id_display = alt['id'][:60]
        print(f"  {sel_mark} #{alt['rank']}  {id_display:<60}  score={alt['score']:.4f}{gold_mark}")

    print()


# --- Aggregate Statistics ---

def compute_aggregate_stats(explanations):
    """
    Summary statistics across all explanations.
    For the paper: shows model learns domain-appropriate features.
    """
    all_tokens = []
    for expl in explanations:
        for t in expl['token_importance']:
            word_lower = t['token'].lower()
            is_stop = word_lower in STOPWORDS
            is_cyber = bool(re.match(
                r'(powershell|cmd|wmi|mimikatz|cobalt|beacon|macro|phishing|'
                r'malware|trojan|backdoor|exploit|vulnerability|credential|'
                r'lateral|persistence|exfiltrat|encrypt|decode|base64|'
                r'registry|dll|exe|script|shell|command|remote|http|dns|'
                r'email|attachment|download|upload|inject|hook|dump|scan|'
                r'brute|privilege|escalat|keylog|screenshot|c2|proxy|tunnel|'
                r'ransomware|rootkit|botnet|spearphish|watering|supply|'
                r'execute|executed|execution|invoke|wmic|scheduled|service|'
                r'token|process|thread|memory|api|rdp|ssh|smb|sql|ftp|'
                r'certificates?|reconnaissance|discovery|collection|'
                r'exfiltration|impact|defense|evasion|initial|access)',
                word_lower
            ))
            all_tokens.append({
                'word': t['token'],
                'importance': t['raw_importance'],
                'normalized': t['normalized'],
                'is_stopword': is_stop,
                'is_cyber_term': is_cyber,
                'query_correct': expl['prediction']['correct'],
            })

    if not all_tokens:
        return {}

    cyber_imps = [t['importance'] for t in all_tokens if t['is_cyber_term']]
    stop_imps  = [t['importance'] for t in all_tokens if t['is_stopword']]
    other_imps = [t['importance'] for t in all_tokens
                  if not t['is_cyber_term'] and not t['is_stopword']]

    def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
    def safe_std(lst):  return float(np.std(lst)) if lst else 0.0

    # Top words by average importance
    word_total = defaultdict(list)
    for t in all_tokens:
        word_total[t['word'].lower()].append(t['importance'])
    word_avg = {w: safe_mean(imps) for w, imps in word_total.items()}
    top_words = sorted(word_avg.items(), key=lambda x: x[1], reverse=True)[:20]

    # Confidence distribution
    conf_dist = defaultdict(int)
    correct_by_conf = defaultdict(int)
    total_by_conf = defaultdict(int)
    for expl in explanations:
        conf = expl['confidence']
        conf_dist[conf] += 1
        total_by_conf[conf] += 1
        if expl['prediction']['correct']:
            correct_by_conf[conf] += 1

    return {
        'total_queries': len(explanations),
        'correct': sum(1 for e in explanations if e['prediction']['correct']),
        'incorrect': sum(1 for e in explanations if not e['prediction']['correct']),
        'importance_by_category': {
            'cyber_terms': {'mean': safe_mean(cyber_imps), 'std': safe_std(cyber_imps), 'n': len(cyber_imps)},
            'stopwords':   {'mean': safe_mean(stop_imps),  'std': safe_std(stop_imps),  'n': len(stop_imps)},
            'other_words': {'mean': safe_mean(other_imps), 'std': safe_std(other_imps), 'n': len(other_imps)},
        },
        'confidence_distribution': dict(conf_dist),
        'accuracy_by_confidence': {
            conf: round(correct_by_conf[conf] / total_by_conf[conf], 4)
            for conf in total_by_conf
        },
        'top_20_important_words': top_words,
    }


# --- Output Writers ---

def save_explanations_json(explanations, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(explanations)} explanations to {path}")


def save_token_importance_csv(explanations, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'query_idx', 'query_preview', 'actor', 'correct',
            'predicted_id', 'gold_id', 'confidence', 'margin',
            'token_position', 'token', 'importance_raw', 'importance_norm', 'impact',
        ])
        for i, expl in enumerate(explanations):
            for t in expl['token_importance']:
                writer.writerow([
                    i, expl['query'][:60], expl.get('actor', ''),
                    expl['prediction']['correct'],
                    expl['prediction'].get('attack_id', ''),
                    expl['gold'].get('attack_id', ''),
                    expl['confidence'], expl['margin'],
                    t['position'], t['token'],
                    t['raw_importance'], t['normalized'], t['impact'],
                ])
    print(f"  Saved token importance to {path}")


def save_stats_report(stats, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("EXPLAINABILITY AGGREGATE STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total queries analyzed: {stats['total_queries']}\n")
        f.write(f"Correct predictions:    {stats['correct']}\n")
        f.write(f"Incorrect predictions:  {stats['incorrect']}\n\n")

        f.write("IMPORTANCE BY WORD CATEGORY\n")
        f.write("-" * 40 + "\n")
        for cat, vals in stats['importance_by_category'].items():
            f.write(f"  {cat:<16}: mean={vals['mean']:+.4f}  std={vals['std']:.4f}  n={vals['n']}\n")

        f.write("\nCONFIDENCE DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for conf in ['HIGH', 'MEDIUM', 'LOW']:
            count = stats['confidence_distribution'].get(conf, 0)
            acc = stats['accuracy_by_confidence'].get(conf, 0)
            f.write(f"  {conf:<8}: {count:>4} queries  (accuracy: {acc:.1%})\n")

        f.write("\nTOP 20 MOST IMPORTANT WORDS (avg importance)\n")
        f.write("-" * 40 + "\n")
        for word, imp in stats['top_20_important_words']:
            f.write(f"  {word:<25}  {imp:+.4f}\n")

    print(f"  Saved statistics report to {path}")


# --- Main Pipeline ---

def main():
    parser = argparse.ArgumentParser(
        description='Explainability module for CTI cross-encoder reranking')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='Path to saved cross-encoder model')
    parser.add_argument('--data', default=DEFAULT_DATA,
                        help='Path to enriched JSONL dataset')
    parser.add_argument('--query', type=str, default=None,
                        help='Explain a single query (substring match)')
    parser.add_argument('--demo', action='store_true',
                        help='Show curated demo explanations')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of queries (for testing)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print every explanation to console')
    args = parser.parse_args()

    print()
    print('=' * 80)
    print('  EXPLAINABILITY MODULE')
    print(f'  Model: {args.model}')
    print(f'  Data:  {args.data}')
    print('=' * 80)

    # Step 1: Load data
    print('\n[1/5] Loading data...')
    query_data, total_rows = load_and_group_data(args.data)
    n_ctihal = sum(1 for qd in query_data.values()
                   if not qd['actor'].startswith('external_'))
    print(f'  Loaded {total_rows:,} rows, {len(query_data):,} unique queries '
          f'({n_ctihal:,} CTI-HAL)')

    # Step 2: Create test split
    print(f'\n[2/5] Creating test split (seed={RANDOM_SEED})...')
    test_queries = create_test_split(query_data)
    print(f'  Test split: {len(test_queries)} queries')

    actor_counts = defaultdict(int)
    for qn in test_queries:
        actor_counts[query_data[qn]['actor']] += 1
    for actor in sorted(actor_counts):
        print(f'    {actor}: {actor_counts[actor]}')

    # Step 3: Build ATT&CK hierarchy
    print('\n[3/5] Building ATT&CK hierarchy from candidate pool...')
    hierarchy, id_to_name = build_attack_hierarchy(query_data)
    n_tech = sum(1 for h in hierarchy.values() if h['type'] == 'technique')
    n_sub  = sum(1 for h in hierarchy.values() if h['type'] == 'sub-technique')
    n_tac  = sum(1 for h in hierarchy.values() if h['type'] == 'tactic')
    print(f'  {len(hierarchy)} ATT&CK entries: '
          f'{n_tac} tactics, {n_tech} techniques, {n_sub} sub-techniques')

    # Step 4: Load model
    print('\n[4/5] Loading cross-encoder model...')
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print('\n  ERROR: sentence-transformers not installed.')
        print('  Run: pip install sentence-transformers')
        sys.exit(1)

    model = CrossEncoder(args.model)
    print(f'  Model loaded from {args.model}')

    # Step 5: Generate explanations
    print('\n[5/5] Generating explanations...')

    if args.query:
        matches = [qn for qn in test_queries
                   if args.query.lower() in query_data[qn]['query_raw'].lower()]
        if not matches:
            matches = [qn for qn, qd in query_data.items()
                       if args.query.lower() in qd['query_raw'].lower()
                       and not qd['actor'].startswith('external_')]
        if not matches:
            print(f'  No queries matching "{args.query}"')
            sys.exit(1)
        target_queries = matches[:5]
        print(f'  Single-query mode: found {len(target_queries)} match(es)')
    else:
        target_queries = test_queries

    if args.limit:
        target_queries = target_queries[:args.limit]

    explanations = []
    start_time = time.time()

    for i, qn in enumerate(target_queries):
        qd = query_data[qn]
        if not qd['candidates'] or qd['positive_count'] == 0:
            continue

        expl = explain_single_query(
            model, qd['query_raw'], qd['candidates'],
            hierarchy=hierarchy, id_to_name=id_to_name
        )
        expl['actor'] = qd['actor']
        expl['query_norm'] = qn
        explanations.append(expl)

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(target_queries) - i - 1) / rate if rate > 0 else 0
            print(f'  [{i+1}/{len(target_queries)}] '
                  f'{elapsed:.1f}s elapsed, ~{eta:.0f}s remaining')

    elapsed = time.time() - start_time
    per_query_ms = elapsed / len(explanations) * 1000 if explanations else 0
    print(f'\n  Completed {len(explanations)} explanations in {elapsed:.1f}s '
          f'({per_query_ms:.0f}ms per query)')

    # Demo mode
    if args.demo:
        correct_high = [e for e in explanations
                        if e['prediction']['correct'] and e['confidence'] == 'HIGH']
        correct_low  = [e for e in explanations
                        if e['prediction']['correct'] and e['confidence'] == 'LOW']
        incorrect    = [e for e in explanations if not e['prediction']['correct']]

        demo_set = []
        if correct_high:
            demo_set.append(('HIGH-CONFIDENCE CORRECT', correct_high[0]))
        if correct_low:
            demo_set.append(('LOW-CONFIDENCE CORRECT', correct_low[0]))
        if len(incorrect) >= 2:
            demo_set.append(('MISS #1', incorrect[0]))
            demo_set.append(('MISS #2', incorrect[1]))
        elif incorrect:
            demo_set.append(('MISS', incorrect[0]))

        print('\n' + '=' * 80)
        print('  DEMO MODE: Curated Examples')
        print('=' * 80)
        for label, expl in demo_set:
            print(f'\n  --- {label} ---')
            print_explanation(expl)

    elif args.verbose or args.query:
        for i, expl in enumerate(explanations):
            print_explanation(expl, index=i+1)

    # Always show misses
    misses = [e for e in explanations if not e['prediction']['correct']]
    if misses and not args.query:
        print('\n' + '=' * 80)
        print(f'  MISS EXPLANATIONS ({len(misses)} misses)')
        print('=' * 80)
        for i, expl in enumerate(misses):
            print_explanation(expl, index=i+1)

    # Aggregate statistics
    stats = compute_aggregate_stats(explanations)

    print('\n' + '=' * 80)
    print('  AGGREGATE STATISTICS')
    print('=' * 80)

    print(f'\n  Accuracy: {stats["correct"]}/{stats["total_queries"]} '
          f'({stats["correct"]/stats["total_queries"]:.1%})')

    print(f'\n  Word Category Importance (avg raw score drop when removed):')
    for cat, vals in stats['importance_by_category'].items():
        bar_len = int(max(0, vals['mean']) * 200)
        bar = '#' * min(bar_len, 30)
        print(f'    {cat:<16}: {vals["mean"]:+.4f}  (n={vals["n"]})  {bar}')

    print(f'\n  Confidence vs Accuracy:')
    for conf in ['HIGH', 'MEDIUM', 'LOW']:
        count = stats['confidence_distribution'].get(conf, 0)
        acc = stats['accuracy_by_confidence'].get(conf, 0)
        print(f'    {conf:<8}: {count:>3} queries, accuracy = {acc:.1%}')

    print(f'\n  Top 10 Most Important Words (avg importance across all queries):')
    for word, imp in stats['top_20_important_words'][:10]:
        bar_len = int(max(0, imp) * 200)
        bar = '#' * min(bar_len, 30)
        print(f'    {word:<25}  {imp:+.4f}  {bar}')

    # Save outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_path  = os.path.join(OUTPUT_DIR, 'explanations.json')
    csv_path   = os.path.join(OUTPUT_DIR, 'token_importance.csv')
    stats_path = os.path.join(OUTPUT_DIR, 'explainability_stats.txt')

    save_explanations_json(explanations, json_path)
    save_token_importance_csv(explanations, csv_path)
    save_stats_report(stats, stats_path)

    print('\n' + '=' * 80)
    print('  DONE')
    print('=' * 80)
    print(f'\n  Outputs saved to {OUTPUT_DIR}/')
    print(f'    explanations.json       - Full structured explanations (for integration)')
    print(f'    token_importance.csv     - Per-token scores (for analysis/paper)')
    print(f'    explainability_stats.txt - Aggregate statistics (for paper)')
    print()


if __name__ == '__main__':
    main()
