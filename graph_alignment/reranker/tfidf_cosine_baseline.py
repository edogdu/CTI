#!/usr/bin/env python3
"""
TF-IDF + Cosine Similarity Baseline for the CTI-to-ATT&CK Reranker
==================================================================

This is the "embarrassingly simple comparator" baseline that the ACSAC playbook
calls for under the inappropriate-baselines Rieck pitfall. It scores each test
query against its per-query candidate pool using cosine similarity over TF-IDF
vectors, with no learned model on top.

Design notes that distinguish this from the previous broken attempt:

1. Per-query candidate pools, not a global corpus index. Each test query is
   scored against its own pre-curated candidate set drawn from
   reranker_pairs_enriched_v2.jsonl, exactly the way v2_reeval.py scores the
   cross-encoder. This is the apples-to-apples comparison: same retrieval
   setup, only the scoring function differs.

2. TF-IDF + cosine, NOT TF-IDF + logistic regression. The previous attempt
   used element-wise products of TF-IDF vectors as input to a logistic
   regression classifier, which lost signal and produced 27% P@1. The clean
   version is just cosine similarity, the standard simple comparator.

3. IDF weights fit on training+validation queries and candidates only.
   The 146 test queries never appear in the vectorizer fit, so there's no
   test-set data leaking into the IDF weights.

4. Same test split as v2_reeval.py via direct import. We import
   load_and_group_data and create_test_split from v2_reeval so the test
   queries are bit-identical to the canonical cross-encoder evaluation.

Expected runtime: roughly 20 seconds on a laptop CPU. No GPU required.

Usage from cmd:
    cd C:\\Users\\shane\\Downloads\\CTI\\graph_alignment\\reranker\\graph_alignment\\reranker
    python tfidf_cosine_baseline.py
"""

import json
import os
import csv
import time
import sys
from collections import defaultdict, Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import the canonical data loading and test split from v2_reeval.
# This guarantees bit-identical test split to the cross-encoder evaluation.
# If this import fails, the script aborts because the comparison wouldn't
# be valid.
from v2_reeval import load_and_group_data, create_test_split

# ============================================================
# CONFIGURATION
# ============================================================
# Paths are relative to the inner reranker directory where this script lives,
# matching the convention used by v2_reeval.py and the other eval scripts.
DATA_PATH = "data/reranker_pairs_enriched_v2.jsonl"
OUTPUT_DIR = "tfidf_cosine_results"

# These TfidfVectorizer settings match common defaults for short-text retrieval
# on technical corpora. We use both unigrams and bigrams because CTI passages
# often contain multi-word technique signatures like "powershell command" or
# "credential theft" that benefit from bigram representation. lowercase=True
# matches how BM25 typically tokenizes for this kind of comparison.
TFIDF_PARAMS = {
    "ngram_range": (1, 2),       # Unigrams and bigrams
    "lowercase": True,           # Normalize casing
    "min_df": 1,                 # Keep all terms (corpus is small)
    "max_df": 1.0,               # Keep all terms (no document-frequency cap)
    "sublinear_tf": True,        # Use log(tf)+1 instead of raw tf, matches
                                 # standard practice and is closer to BM25's
                                 # term-frequency saturation behavior.
    "norm": "l2",                # L2 normalize so cosine = dot product
}


# ============================================================
# PREFLIGHT CHECKS
# ============================================================
def preflight():
    """
    Verify required files exist before doing anything expensive.
    
    The vectorizer fit and per-query scoring are fast, but it's still better
    to fail loud if the data file is missing rather than silently producing
    nonsense. This mirrors v2_reeval.py's preflight pattern.
    """
    print("=" * 65)
    print("  PREFLIGHT: Checking required files")
    print("=" * 65)

    required = [
        (DATA_PATH, "enriched CTI-HAL data"),
        ("v2_reeval.py", "canonical eval script (for split functions)"),
    ]

    all_ok = True
    for path, desc in required:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  [{status:>7}] {desc}: {path}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n  [FATAL] Required files missing. Cannot proceed.")
        print("  Run this script from the inner reranker directory.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n  Output directory: {OUTPUT_DIR}/")
    print("  Preflight passed.\n")


# ============================================================
# CORPUS BUILDING
# ============================================================
def build_corpus(query_data, test_queries):
    """
    Build the corpus for fitting the TF-IDF vectorizer.
    
    The corpus consists of (a) all queries that are NOT in the test set, and
    (b) all candidate texts from those non-test queries. Candidate texts that
    also appear in test-query candidate pools are intentionally included
    because they're part of the broader training distribution the vectorizer
    needs to learn IDF weights for. Test query *texts* are excluded — that's
    the data-leakage check that matters.
    
    Returns the list of corpus documents (strings).
    """
    test_query_set = set(test_queries)
    
    corpus = []
    n_queries = 0
    n_candidates = 0
    
    for query_norm, qdata in query_data.items():
        # Skip test queries entirely
        if query_norm in test_query_set:
            continue
        # Skip queries from external augmentation sources (tumeteor, zenodo)
        # because they have different actor labels and aren't part of the
        # CTI-HAL split structure. v2_reeval.py's create_test_split also
        # skips these.
        if qdata.get('actor', '').startswith('external_'):
            continue
        
        # Add this query's text to the corpus
        if qdata.get('query_raw'):
            corpus.append(qdata['query_raw'])
            n_queries += 1
        
        # Add all of this query's candidate texts to the corpus
        for cand in qdata.get('candidates', []):
            if cand.get('text'):
                corpus.append(cand['text'])
                n_candidates += 1
    
    print(f"  Corpus size: {len(corpus)} documents")
    print(f"    Queries: {n_queries}")
    print(f"    Candidate texts: {n_candidates}")
    
    return corpus


# ============================================================
# EVALUATION
# ============================================================
def evaluate_tfidf_cosine(vectorizer, query_data, test_queries):
    """
    Score each test query against its per-query candidate pool using cosine
    similarity over TF-IDF vectors. Returns a list of per-query result dicts.
    
    For each test query:
      1. Transform the query text into a TF-IDF vector.
      2. Transform each of the query's candidate texts into TF-IDF vectors.
      3. Compute cosine similarity between the query vector and each
         candidate vector.
      4. Rank candidates by similarity (descending).
      5. Record top-1, top-3, top-5 hits against the gold ID set.
    
    The ranking-only output (no learned classifier on top) is what makes this
    the "embarrassingly simple" baseline.
    """
    print("=" * 65)
    print("  EVALUATION: TF-IDF + Cosine on CTI-HAL Test Set")
    print("=" * 65)
    
    n = len(test_queries)
    print(f"  Test queries: {n}")
    
    actor_counts = Counter(query_data[qn]['actor'] for qn in test_queries)
    for actor in sorted(actor_counts):
        print(f"    {actor}: {actor_counts[actor]}")
    
    results = []
    t0 = time.time()
    
    for i, qn in enumerate(test_queries):
        qdata = query_data[qn]
        query_raw = qdata['query_raw']
        gold_ids = qdata['gold_ids']
        candidates = qdata['candidates']
        
        # Transform query and all candidates into TF-IDF vectors
        query_vec = vectorizer.transform([query_raw])
        cand_texts = [c['text'] for c in candidates]
        cand_vecs = vectorizer.transform(cand_texts)
        
        # Cosine similarity between the query and each candidate.
        # cosine_similarity returns a (1, n_candidates) matrix; we flatten
        # to a 1D array of scores aligned with the candidates list.
        sims = cosine_similarity(query_vec, cand_vecs).flatten()
        
        # Rank candidates by similarity (descending). zip + sorted gives us
        # the same shape as v2_reeval.py's ranked-by-score output.
        ranked = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)
        
        top1_id = ranked[0][0]['id']
        top1_score = float(ranked[0][1])
        
        # Compute hit metrics. correct = top-1 in gold set; hit3/hit5 = any
        # of the top 3/5 candidate IDs in gold set.
        correct = top1_id in gold_ids
        ranked_ids = [c[0]['id'] for c in ranked]
        hit3 = any(rid in gold_ids for rid in ranked_ids[:3])
        hit5 = any(rid in gold_ids for rid in ranked_ids[:5])
        
        # Top-5 candidate detail for downstream comparison (e.g. checking
        # whether TF-IDF and cross-encoder pick the same runners-up).
        top5 = [(c['id'], float(s)) for c, s in ranked[:5]]
        
        results.append({
            'query_norm': qn,
            'query_raw': query_raw,
            'actor': qdata['actor'],
            'gold_ids': sorted(gold_ids),
            'predicted': top1_id,
            'predicted_score': top1_score,
            'correct': correct,
            'hit3': hit3,
            'hit5': hit5,
            'top5': top5,
        })
        
        # Progress print every 30 queries, matching v2_reeval.py's pattern
        if (i + 1) % 30 == 0:
            elapsed = time.time() - t0
            cur_p1 = sum(r['correct'] for r in results) / len(results)
            print(f"  [{i+1}/{n}] P@1 so far: {cur_p1:.4f} | {elapsed:.1f}s")
    
    elapsed = time.time() - t0
    print(f"\n  Evaluation complete in {elapsed:.1f} seconds")
    
    return results


# ============================================================
# REPORTING
# ============================================================
def report_results(results):
    """
    Print overall and per-actor metrics; return a summary dict for JSON output.
    """
    n = len(results)
    p1 = sum(r['correct'] for r in results) / n
    h3 = sum(r['hit3'] for r in results) / n
    h5 = sum(r['hit5'] for r in results) / n
    n_correct = sum(r['correct'] for r in results)
    
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"\n  Overall:")
    print(f"    P@1:   {p1:.4f}  ({n_correct}/{n})")
    print(f"    Hit@3: {h3:.4f}")
    print(f"    Hit@5: {h5:.4f}")
    
    # Per-actor breakdown
    actor_results = defaultdict(list)
    for r in results:
        actor_results[r['actor']].append(r)
    
    print(f"\n  Per-actor P@1:")
    print(f"    {'Actor':<15} | {'P@1':>7} | {'Correct':>7} | {'Total':>5}")
    print(f"    {'-'*15}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}")
    
    per_actor_summary = {}
    for actor in sorted(actor_results):
        ar = actor_results[actor]
        ap1 = sum(r['correct'] for r in ar) / len(ar)
        ac = sum(r['correct'] for r in ar)
        print(f"    {actor:<15} | {ap1:>6.2%} | {ac:>7} | {len(ar):>5}")
        per_actor_summary[actor] = {
            'p@1': ap1,
            'correct': ac,
            'total': len(ar),
        }
    
    summary = {
        'method': 'TF-IDF + cosine similarity',
        'n_test_queries': n,
        'p@1': p1,
        'hit@3': h3,
        'hit@5': h5,
        'correct': n_correct,
        'per_actor': per_actor_summary,
        'tfidf_params': TFIDF_PARAMS,
    }
    
    return summary


def save_outputs(results, summary):
    """
    Write the per-query CSV and the summary JSON to OUTPUT_DIR.
    """
    # Per-query CSV
    csv_path = os.path.join(OUTPUT_DIR, "per_query.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'query_norm', 'query_raw', 'actor', 'gold_ids',
            'predicted', 'predicted_score', 'correct', 'hit3', 'hit5',
            'top5_ids', 'top5_scores',
        ])
        for r in results:
            top5_ids = '|'.join([t[0] for t in r['top5']])
            top5_scores = '|'.join([f"{t[1]:.6f}" for t in r['top5']])
            writer.writerow([
                r['query_norm'], r['query_raw'], r['actor'],
                '|'.join(r['gold_ids']),
                r['predicted'], f"{r['predicted_score']:.6f}",
                r['correct'], r['hit3'], r['hit5'],
                top5_ids, top5_scores,
            ])
    print(f"\n  Wrote per-query CSV: {csv_path}")
    
    # Summary JSON
    json_path = os.path.join(OUTPUT_DIR, "tfidf_cosine_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote summary JSON: {json_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 65)
    print("  TF-IDF + COSINE BASELINE")
    print("  CTI-to-ATT&CK Reranker — Embarrassingly Simple Comparator")
    print("=" * 65 + "\n")
    
    preflight()
    
    # Step 1: Load and split data exactly as v2_reeval does
    print("=" * 65)
    print("  STEP 1: Load data and recreate canonical test split")
    print("=" * 65)
    print(f"  Loading {DATA_PATH}...")
    query_data, total_rows = load_and_group_data(DATA_PATH)
    print(f"  Loaded {total_rows} rows, {len(query_data)} unique queries")
    
    test_queries = create_test_split(query_data)
    print(f"  Test split: {len(test_queries)} queries")
    
    # Step 2: Build corpus from non-test queries and their candidates
    print("\n" + "=" * 65)
    print("  STEP 2: Build corpus from training+validation data")
    print("=" * 65)
    corpus = build_corpus(query_data, test_queries)
    
    # Step 3: Fit the TF-IDF vectorizer on the non-test corpus
    print("\n" + "=" * 65)
    print("  STEP 3: Fit TfidfVectorizer")
    print("=" * 65)
    print(f"  Parameters: {TFIDF_PARAMS}")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    t0 = time.time()
    vectorizer.fit(corpus)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Fit time: {time.time() - t0:.1f}s")
    
    # Step 4: Evaluate against the test queries' per-query candidate pools
    print()
    results = evaluate_tfidf_cosine(vectorizer, query_data, test_queries)
    
    # Step 5: Report and save
    summary = report_results(results)
    save_outputs(results, summary)
    
    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)
    print(f"  TF-IDF + Cosine P@1: {summary['p@1']:.4f}")
    print(f"  Compare to BM25 P@1: 0.7534 (from README)")
    print(f"  Compare to v2 cross-encoder P@1: 0.9452 (from v2_reeval)")
    print()


if __name__ == "__main__":
    main()
