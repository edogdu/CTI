#!/usr/bin/env python3
"""
Token Importance Analysis for the CTI-to-ATT&CK Reranker
=========================================================

This script produces the empirical claims that ground the ACSAC threat-model
paragraph and supporting explainability analysis. It reads the existing LOO
token-importance data, classifies every token using MITRE ATT&CK vocabulary
extracted from STIX (primary) plus a corpus-frequency cross-check (secondary),
and computes eight aggregations that map to specific paper claims.

Headline outputs (cited in the threat-model paragraph):
  H1: Percent of correct vs incorrect predictions whose top-importance token
      is technique-relevant (the spurious-correlations defense).
  H2: Mean LOO importance for actor-name tokens vs technique-relevant tokens
      (quantifies that the model attends to behavior, not actor identity).

Supporting aggregations (used elsewhere in the explainability section):
  A1: Per-actor breakdown of H1 and H2.
  A2: Detailed correctness stratification with confidence intervals.
  A3: Margin-vs-importance-entropy correlation (deployment readiness).
  A4: BM25 overlap correspondence (Layer 1 LOO vs Layer 2 BM25 agreement).
  A5: Position-effect analysis (transformer position-bias check).
  A6: Per-query importance entropy distribution.

Categorization scheme:
  ATTACK_VOCAB - token appears in MITRE ATT&CK technique names, descriptions,
                 software names, or tactic names (extracted from STIX bundle)
  ACTOR_NAME   - token matches conservative list of 9 actor identifiers
  STOPWORD     - token is in the standard English stopword set
  OTHER        - everything else (numbers, generic verbs, rare proper nouns)

Run from the inner reranker directory:
    cd C:\\Users\\shane\\Downloads\\CTI\\graph_alignment\\reranker
    python token_importance_analysis.py

Expected runtime: 30-60 seconds. No GPU required.
"""

import json
import os
import csv
import sys
import re
import time
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================
TOKEN_CSV = "eval_results/token_importance.csv"
ATTACK_STIX = "enterprise-attack-v14.json"
TRAINING_DATA = "data/reranker_pairs_enriched_v2.jsonl"
OUTPUT_DIR = "token_analysis_results"

# Conservative actor name list. We deliberately exclude common-word aliases
# like "duke", "wizard", "kitten", "carbon", "spider" because they are too
# generic and would cause false-positive actor categorization. We also
# exclude "ryuk" because it is a malware family name (technique-relevant
# vocabulary), not an actor reference per se.
CANONICAL_ACTORS = {
    "apt29",
    "carbanak",
    "anunak",       # Carbanak alias
    "fin6",
    "fin7",
    "oilrig",
    "apt34",        # OilRig alias
    "sandworm",
    "wizardspider",
}

# Standard English stopword list (matches what explain.py uses for BM25 overlap).
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "in", "is", "it",
    "its", "of", "on", "or", "that", "the", "their", "they", "this", "to",
    "was", "were", "will", "with", "would", "you", "your", "we", "our",
    "do", "does", "did", "than", "then", "so", "if", "not", "no", "all",
    "any", "some", "such", "these", "those", "there", "here", "what", "when",
    "where", "which", "who", "why", "how", "can", "could", "should", "may",
    "might", "must", "shall", "ought", "into", "onto", "upon", "via",
    "while", "during", "after", "before", "between", "through", "across",
    "also", "including", "include", "include", "mainly", "very", "such",
}

# Threshold for "high-importance" tokens. Using the existing categorical
# `impact` column from explain.py rather than redefining the threshold.
HIGH_IMPACT_VALUES = {"HIGH"}


# ============================================================
# HELPERS
# ============================================================
def normalize_token(token):
    """
    Lowercase and strip non-alphanumeric characters. The LOO tokens in
    token_importance.csv come from query.split() which preserves casing
    and may include attached punctuation, so we normalize here.
    """
    return re.sub(r'[^a-z0-9]', '', str(token).lower())


def tokenize_text(text):
    """
    Split text into normalized lowercase alphanumeric tokens, dropping
    stopwords and single-character tokens. Used for STIX vocabulary
    extraction and for BM25 overlap computation.
    """
    if not text:
        return []
    raw_tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in raw_tokens if t not in STOPWORDS and len(t) > 1]


def bootstrap_ci(values, n_bootstrap=1000, alpha=0.05, statistic=np.mean):
    """
    Compute a bootstrap confidence interval for any statistic over a set
    of values. Used to put uncertainty bounds around our headline percentages.
    """
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed=42)
    boots = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(statistic(sample))
    lower = np.percentile(boots, 100 * alpha / 2)
    upper = np.percentile(boots, 100 * (1 - alpha / 2))
    return (lower, upper)


def safe_entropy(values):
    """
    Compute Shannon entropy of an importance distribution. Negative values
    are clipped to zero (they represent tokens whose removal helped the
    score, which doesn't fit a probability interpretation), then the
    distribution is normalized to sum to 1.
    """
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, 0, None)
    total = arr.sum()
    if total == 0:
        return 0.0
    p = arr / total
    p = p[p > 0]  # Avoid log(0)
    return float(-np.sum(p * np.log2(p)))


# ============================================================
# PREFLIGHT
# ============================================================
def preflight():
    """
    Verify all required files exist before doing any expensive work.
    Mirrors the pattern from v2_reeval.py and tfidf_cosine_baseline.py.
    """
    print("=" * 65)
    print("  PREFLIGHT: Checking required files")
    print("=" * 65)

    required = [
        (TOKEN_CSV, "LOO token importance CSV"),
        (ATTACK_STIX, "MITRE ATT&CK STIX bundle"),
        (TRAINING_DATA, "enriched CTI-HAL training data"),
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
# STIX VOCABULARY EXTRACTION (PRIMARY CATEGORIZATION)
# ============================================================
def build_stix_vocabulary(stix_path):
    """
    Extract technique-relevant vocabulary from MITRE ATT&CK STIX bundle.

    We pull tokens from:
      - attack-pattern names and descriptions (techniques + sub-techniques)
      - malware names and descriptions
      - tool names and descriptions
      - x-mitre-tactic names and descriptions

    We deliberately EXCLUDE intrusion-set names because those are actor
    names and we want them in a separate category.

    Returns a set of normalized lowercase tokens that constitute the
    "technique-relevant" vocabulary.
    """
    print("  Extracting MITRE ATT&CK vocabulary from STIX...")
    with open(stix_path, 'r', encoding='utf-8') as f:
        bundle = json.load(f)

    vocabulary = set()
    counts_by_type = Counter()

    relevant_types = {"attack-pattern", "malware", "tool", "x-mitre-tactic"}

    for obj in bundle.get("objects", []):
        obj_type = obj.get("type")
        if obj_type not in relevant_types:
            continue

        # Skip revoked or deprecated objects (these have x_mitre_deprecated
        # or revoked flags). Including them would add stale vocabulary.
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue

        name = obj.get("name", "")
        description = obj.get("description", "")

        for token in tokenize_text(name):
            vocabulary.add(token)
            counts_by_type[obj_type] += 1
        for token in tokenize_text(description):
            vocabulary.add(token)

    print(f"    Vocabulary size: {len(vocabulary)} unique tokens")
    print(f"    Object counts by type:")
    for t, c in sorted(counts_by_type.items()):
        print(f"      {t}: {c} name-token contributions")

    return vocabulary


# ============================================================
# CORPUS-FREQUENCY VOCABULARY (CROSS-CHECK)
# ============================================================
def build_corpus_vocabulary(data_path):
    """
    Reproduce the auto_word_categorize.py logic to build a cyber lexicon
    from the training corpus. Used as a cross-check against the STIX-based
    vocabulary.

    A token is in the corpus lexicon if:
      (a) it appears in any candidate text (which are ATT&CK entity names
          and descriptions), OR
      (b) it appears in at least 5 distinct queries and is not a common
          English word.
    """
    print("\n  Building corpus-frequency vocabulary (cross-check)...")

    candidate_terms = set()
    query_term_counts = Counter()
    seen_candidates = set()
    seen_queries = set()

    common_english = {
        'also', 'new', 'first', 'last', 'time', 'way', 'day', 'part',
        'used', 'use', 'using', 'make', 'made', 'set', 'run', 'get',
        'well', 'back', 'much', 'end', 'own', 'still', 'found', 'since',
        'long', 'work', 'three', 'need', 'like', 'even', 'right', 'look',
        'think', 'next', 'keep', 'let', 'begin', 'name', 'show', 'try',
        'start', 'point', 'move', 'same', 'tell', 'help', 'turn', 'hand',
        'high', 'place', 'small', 'large', 'line', 'open', 'number',
        'group', 'order', 'case', 'system', 'possible', 'within', 'however',
        'different', 'include', 'general', 'specific', 'following', 'several',
        'another', 'known', 'included', 'able', 'often', 'report', 'reports',
        'based', 'information', 'order', 'example', 'two', 'one',
    }

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            ctext = row.get('candidate_text', '')
            if ctext and ctext not in seen_candidates:
                seen_candidates.add(ctext)
                # Strip the "ID — " prefix common to enriched candidate text
                name_part = re.sub(r'^[A-Z0-9.]+ [—-] ', '', ctext)
                for token in tokenize_text(name_part):
                    candidate_terms.add(token)

            qraw = row.get('query_raw', '')
            if qraw and qraw not in seen_queries:
                seen_queries.add(qraw)
                for token in tokenize_text(qraw):
                    query_term_counts[token] += 1

    # Start with all candidate terms (these are the "official" vocabulary)
    corpus_vocabulary = set(candidate_terms)

    # Add high-frequency query terms that aren't common English
    for term, count in query_term_counts.items():
        if count >= 5 and term not in common_english and len(term) > 2:
            corpus_vocabulary.add(term)

    print(f"    Candidates processed: {len(seen_candidates)}")
    print(f"    Queries processed: {len(seen_queries)}")
    print(f"    Corpus vocabulary size: {len(corpus_vocabulary)}")

    return corpus_vocabulary


# ============================================================
# TOKEN CATEGORIZATION
# ============================================================
def categorize_token(token, stix_vocab, corpus_vocab, actor_set):
    """
    Classify a single token into one of four categories using a priority
    cascade: actor names take priority (so "apt29" is ACTOR_NAME even though
    it might also appear in some technique descriptions); then stopwords;
    then ATT&CK vocabulary; then everything else.

    Returns a tuple (primary_category, in_stix, in_corpus) where:
      primary_category: ACTOR_NAME | STOPWORD | ATTACK_VOCAB | OTHER
      in_stix: True if token is in STIX-derived vocabulary
      in_corpus: True if token is in corpus-frequency vocabulary

    The (in_stix, in_corpus) flags let us cross-check categorization
    consistency between the two methods.
    """
    norm = normalize_token(token)

    if not norm:
        return ("OTHER", False, False)

    in_stix = norm in stix_vocab
    in_corpus = norm in corpus_vocab

    if norm in actor_set:
        return ("ACTOR_NAME", in_stix, in_corpus)
    if norm in STOPWORDS:
        return ("STOPWORD", in_stix, in_corpus)
    if in_stix:
        return ("ATTACK_VOCAB", in_stix, in_corpus)
    return ("OTHER", in_stix, in_corpus)


# ============================================================
# PER-QUERY ANALYSIS (entropy, BM25 overlap)
# ============================================================
def load_gold_candidate_texts(data_path):
    """
    Load gold candidate texts indexed by candidate_id. We need this to
    compute BM25 overlap between query LOO tokens and gold candidate
    text. We collect ALL candidate texts (a candidate may appear as
    gold for multiple queries) keyed by candidate_id.
    """
    print("\n  Loading gold candidate texts for BM25 overlap analysis...")
    candidate_texts = {}
    seen = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen += 1
            cid = row.get('candidate_norm') or row.get('candidate_id')
            ctext = row.get('candidate_text', '')
            if cid and ctext and cid not in candidate_texts:
                candidate_texts[cid] = ctext

    print(f"    Loaded {len(candidate_texts)} unique candidate texts "
          f"from {seen} rows")
    return candidate_texts


def compute_per_query_metrics(token_df, candidate_texts):
    """
    For each query (grouped by query_idx), compute:
      - top_token: the token with highest importance_norm
      - top_token_category: its categorization
      - importance_entropy: Shannon entropy of normalized importance
      - bm25_overlap_count: count of HIGH-impact tokens that also appear
        in the gold candidate text after stopword removal
      - bm25_overlap_rate: bm25_overlap_count / count of HIGH-impact tokens

    Returns a per-query DataFrame.
    """
    rows = []
    for query_idx, group in token_df.groupby('query_idx'):
        # Per-query metadata (same for all tokens of this query)
        actor = group['actor'].iloc[0]
        correct = bool(group['correct'].iloc[0])
        margin = float(group['margin'].iloc[0])
        confidence = group['confidence'].iloc[0]
        gold_id = group['gold_id'].iloc[0]
        predicted_id = group['predicted_id'].iloc[0]
        query_preview = group['query_preview'].iloc[0]

        # Top token (highest importance_norm)
        top_idx = group['importance_norm'].idxmax()
        top_token = group.loc[top_idx, 'token']
        top_token_category = group.loc[top_idx, 'category']
        top_token_importance = float(group.loc[top_idx, 'importance_norm'])

        # Importance entropy (concentration measure)
        entropy = safe_entropy(group['importance_raw'].values)

        # BM25 overlap: HIGH-impact tokens that appear in gold candidate text
        high_impact_mask = group['impact'].isin(HIGH_IMPACT_VALUES)
        high_tokens = [normalize_token(t) for t in group.loc[high_impact_mask, 'token'].values]
        high_tokens = [t for t in high_tokens if t and t not in STOPWORDS]
        n_high = len(high_tokens)

        gold_text = candidate_texts.get(gold_id, '')
        gold_tokens = set(tokenize_text(gold_text))

        if n_high > 0:
            overlap_count = sum(1 for t in high_tokens if t in gold_tokens)
            overlap_rate = overlap_count / n_high
        else:
            overlap_count = 0
            overlap_rate = np.nan

        rows.append({
            'query_idx': query_idx,
            'query_preview': query_preview,
            'actor': actor,
            'correct': correct,
            'margin': margin,
            'confidence': confidence,
            'gold_id': gold_id,
            'predicted_id': predicted_id,
            'top_token': top_token,
            'top_token_category': top_token_category,
            'top_token_importance': top_token_importance,
            'importance_entropy': entropy,
            'n_high_impact_tokens': n_high,
            'bm25_overlap_count': overlap_count,
            'bm25_overlap_rate': overlap_rate,
            'gold_text_available': gold_id in candidate_texts,
        })

    return pd.DataFrame(rows)


# ============================================================
# AGGREGATIONS
# ============================================================
def compute_aggregations(token_df, query_df):
    """
    Compute all eight aggregations for the threat-model paragraph and the
    explainability section. Returns a nested dict suitable for JSON output.

    Each aggregation includes both the headline number and supporting
    statistics (sample size, confidence intervals where applicable).
    """
    print("\n  Computing aggregations...")
    agg = {}

    # ---- H1: Top-token category by correctness ----
    h1 = {}
    for label, mask in [("correct", query_df['correct'] == True),
                        ("incorrect", query_df['correct'] == False)]:
        subset = query_df[mask]
        n = len(subset)
        if n == 0:
            h1[label] = {"n": 0, "pct_attack_vocab": None,
                         "pct_actor_name": None, "pct_other": None}
            continue
        cats = subset['top_token_category'].value_counts(normalize=True)
        # Boolean indicator series for bootstrap
        is_attack = (subset['top_token_category'] == 'ATTACK_VOCAB').astype(int).values
        ci = bootstrap_ci(is_attack)
        h1[label] = {
            "n": int(n),
            "pct_attack_vocab": float(cats.get('ATTACK_VOCAB', 0.0)),
            "pct_actor_name": float(cats.get('ACTOR_NAME', 0.0)),
            "pct_stopword": float(cats.get('STOPWORD', 0.0)),
            "pct_other": float(cats.get('OTHER', 0.0)),
            "ci_attack_vocab_95": [float(ci[0]), float(ci[1])],
        }
    agg["H1_top_token_by_correctness"] = h1

    # ---- H2: Mean importance by category ----
    h2 = {}
    for cat in ["ATTACK_VOCAB", "ACTOR_NAME", "STOPWORD", "OTHER"]:
        subset = token_df[token_df['category'] == cat]
        n = len(subset)
        if n == 0:
            h2[cat] = {"n": 0, "mean_importance_raw": None,
                       "mean_importance_norm": None}
            continue
        vals_raw = subset['importance_raw'].values
        vals_norm = subset['importance_norm'].values
        ci_raw = bootstrap_ci(vals_raw)
        h2[cat] = {
            "n": int(n),
            "mean_importance_raw": float(np.mean(vals_raw)),
            "mean_importance_norm": float(np.mean(vals_norm)),
            "median_importance_raw": float(np.median(vals_raw)),
            "std_importance_raw": float(np.std(vals_raw)),
            "ci_raw_95": [float(ci_raw[0]), float(ci_raw[1])],
        }
    # Compute the headline ratio: ATTACK_VOCAB mean / ACTOR_NAME mean
    if (h2.get("ACTOR_NAME", {}).get("mean_importance_raw") not in (None, 0)
            and h2.get("ATTACK_VOCAB", {}).get("mean_importance_raw") is not None):
        ratio = (h2["ATTACK_VOCAB"]["mean_importance_raw"] /
                 max(h2["ACTOR_NAME"]["mean_importance_raw"], 1e-9))
        h2["attack_vs_actor_ratio"] = float(ratio)
    agg["H2_mean_importance_by_category"] = h2

    # ---- A1: Per-actor breakdown of H1 and H2 ----
    a1 = {}
    for actor in sorted(query_df['actor'].unique()):
        actor_queries = query_df[query_df['actor'] == actor]
        actor_tokens = token_df[token_df['actor'] == actor]
        n_q = len(actor_queries)
        n_t = len(actor_tokens)

        # H1-style: % top-tokens that are ATTACK_VOCAB
        n_correct = int((actor_queries['correct'] == True).sum())
        if n_q > 0:
            cats = actor_queries['top_token_category'].value_counts(normalize=True)
            pct_attack = float(cats.get('ATTACK_VOCAB', 0.0))
            pct_actor = float(cats.get('ACTOR_NAME', 0.0))
        else:
            pct_attack = pct_actor = None

        # H2-style: mean importance by category, restricted to this actor
        per_cat = {}
        for cat in ["ATTACK_VOCAB", "ACTOR_NAME"]:
            sub = actor_tokens[actor_tokens['category'] == cat]
            if len(sub) > 0:
                per_cat[cat] = {
                    "n": int(len(sub)),
                    "mean_importance_raw": float(sub['importance_raw'].mean()),
                }
            else:
                per_cat[cat] = {"n": 0, "mean_importance_raw": None}

        a1[actor] = {
            "n_queries": int(n_q),
            "n_tokens": int(n_t),
            "n_correct": n_correct,
            "p_at_1": float(n_correct / n_q) if n_q > 0 else None,
            "pct_top_token_attack_vocab": pct_attack,
            "pct_top_token_actor_name": pct_actor,
            "by_category": per_cat,
        }
    agg["A1_per_actor"] = a1

    # ---- A2: Detailed correctness stratification ----
    a2 = {}
    n_correct = int((query_df['correct'] == True).sum())
    n_incorrect = int((query_df['correct'] == False).sum())
    a2["n_correct"] = n_correct
    a2["n_incorrect"] = n_incorrect
    a2["overall_p_at_1"] = float(n_correct / max(len(query_df), 1))

    # Top-token category distribution for each subset (already in H1, but
    # we add absolute counts here for completeness)
    for label, mask in [("correct", query_df['correct'] == True),
                        ("incorrect", query_df['correct'] == False)]:
        subset = query_df[mask]
        counts = subset['top_token_category'].value_counts().to_dict()
        a2[f"{label}_category_counts"] = {k: int(v) for k, v in counts.items()}
    agg["A2_correctness_stratification"] = a2

    # ---- A3: Margin-vs-entropy correlation ----
    valid = query_df.dropna(subset=['margin', 'importance_entropy'])
    if len(valid) >= 3:
        pearson_r, pearson_p = stats.pearsonr(valid['margin'], valid['importance_entropy'])
        spearman_r, spearman_p = stats.spearmanr(valid['margin'], valid['importance_entropy'])
        agg["A3_margin_entropy_correlation"] = {
            "n": int(len(valid)),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "mean_entropy_correct": float(query_df.loc[query_df['correct'], 'importance_entropy'].mean()),
            "mean_entropy_incorrect": float(query_df.loc[~query_df['correct'], 'importance_entropy'].mean()) if (~query_df['correct']).any() else None,
        }
    else:
        agg["A3_margin_entropy_correlation"] = {"n": int(len(valid)), "error": "insufficient data"}

    # ---- A4: BM25 overlap correspondence ----
    valid = query_df.dropna(subset=['bm25_overlap_rate'])
    a4 = {
        "n_queries_with_high_tokens": int(len(valid)),
        "n_queries_missing_gold_text": int((~query_df['gold_text_available']).sum()),
        "mean_overlap_rate_overall": float(valid['bm25_overlap_rate'].mean()),
    }
    if (valid['correct'] == True).any() and (valid['correct'] == False).any():
        correct_overlap = valid.loc[valid['correct'] == True, 'bm25_overlap_rate']
        incorrect_overlap = valid.loc[valid['correct'] == False, 'bm25_overlap_rate']
        a4["mean_overlap_rate_correct"] = float(correct_overlap.mean())
        a4["mean_overlap_rate_incorrect"] = float(incorrect_overlap.mean())
        # Independent samples t-test for the difference
        try:
            t_stat, t_p = stats.ttest_ind(correct_overlap, incorrect_overlap, equal_var=False)
            a4["correct_vs_incorrect_t"] = float(t_stat)
            a4["correct_vs_incorrect_p"] = float(t_p)
        except Exception:
            pass
    agg["A4_bm25_overlap"] = a4

    # ---- A5: Position effect ----
    # Normalize position to fraction-through-query so different query lengths
    # can be compared on a common scale.
    pos_df = token_df.copy()
    pos_df['query_length'] = pos_df.groupby('query_idx')['token_position'].transform('max') + 1
    pos_df['position_fraction'] = pos_df['token_position'] / pos_df['query_length'].clip(lower=1)
    valid = pos_df.dropna(subset=['position_fraction', 'importance_raw'])
    if len(valid) >= 3:
        pearson_r, pearson_p = stats.pearsonr(valid['position_fraction'], valid['importance_raw'])
        spearman_r, spearman_p = stats.spearmanr(valid['position_fraction'], valid['importance_raw'])
        agg["A5_position_effect"] = {
            "n_tokens": int(len(valid)),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        }
    else:
        agg["A5_position_effect"] = {"error": "insufficient data"}

    # ---- A6: Per-query entropy distribution ----
    a6 = {
        "n_queries": int(len(query_df)),
        "mean_entropy": float(query_df['importance_entropy'].mean()),
        "median_entropy": float(query_df['importance_entropy'].median()),
        "std_entropy": float(query_df['importance_entropy'].std()),
        "min_entropy": float(query_df['importance_entropy'].min()),
        "max_entropy": float(query_df['importance_entropy'].max()),
    }
    agg["A6_per_query_entropy"] = a6

    # ---- Cross-check: STIX vs corpus categorization agreement ----
    in_stix = token_df['in_stix'].astype(int).values
    in_corpus = token_df['in_corpus'].astype(int).values
    agreement = (in_stix == in_corpus).mean()
    agg["categorization_cross_check"] = {
        "n_tokens": int(len(token_df)),
        "stix_only_pct": float((in_stix & ~in_corpus.astype(bool)).mean()),
        "corpus_only_pct": float((in_corpus & ~in_stix.astype(bool)).mean()),
        "both_pct": float((in_stix & in_corpus).mean()),
        "neither_pct": float((~in_stix.astype(bool) & ~in_corpus.astype(bool)).mean()),
        "agreement_rate": float(agreement),
    }

    return agg


# ============================================================
# OUTPUT WRITING
# ============================================================
def write_csvs(token_df, query_df, output_dir):
    """Write the per-token and per-query CSV outputs."""
    token_path = os.path.join(output_dir, "per_token_classification.csv")
    token_df.to_csv(token_path, index=False, encoding='utf-8')
    print(f"  Wrote {token_path} ({len(token_df)} rows)")

    query_path = os.path.join(output_dir, "per_query_summary.csv")
    query_df.to_csv(query_path, index=False, encoding='utf-8')
    print(f"  Wrote {query_path} ({len(query_df)} rows)")


def write_aggregations_json(agg, output_dir):
    """Write the machine-readable JSON of all aggregations."""
    path = os.path.join(output_dir, "aggregations.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2)
    print(f"  Wrote {path}")


def write_summary_markdown(agg, output_dir):
    """
    Write the human-readable markdown summary that the threat-model paragraph
    will reference. This is the document a paper writer (or reviewer) would
    open to check the empirical claims.
    """
    path = os.path.join(output_dir, "summary.md")
    h1 = agg["H1_top_token_by_correctness"]
    h2 = agg["H2_mean_importance_by_category"]
    a1 = agg["A1_per_actor"]
    a3 = agg["A3_margin_entropy_correlation"]
    a4 = agg["A4_bm25_overlap"]
    a5 = agg["A5_position_effect"]
    a6 = agg["A6_per_query_entropy"]
    cc = agg["categorization_cross_check"]

    lines = []
    lines.append("# Token Importance Analysis - Summary")
    lines.append("")
    lines.append("Empirical claims for the ACSAC threat-model paragraph and the")
    lines.append("explainability section. Numbers below are computed from")
    lines.append("`eval_results/token_importance.csv` (LOO output of explain.py)")
    lines.append("with categorization grounded in MITRE ATT&CK STIX vocabulary.")
    lines.append("")
    lines.append("## H1: Top-token category by prediction correctness")
    lines.append("")
    lines.append(f"On the {h1['correct']['n']} queries the v2 model predicted correctly,")
    lines.append(f"the highest-importance LOO token was an ATT&CK-vocabulary term in")
    lines.append(f"**{h1['correct']['pct_attack_vocab']*100:.1f}%** of cases (95% CI: "
                 f"{h1['correct']['ci_attack_vocab_95'][0]*100:.1f}%-"
                 f"{h1['correct']['ci_attack_vocab_95'][1]*100:.1f}%).")
    lines.append("")
    lines.append(f"On the {h1['incorrect']['n']} queries the v2 model predicted incorrectly,")
    lines.append(f"the same was true in **{h1['incorrect']['pct_attack_vocab']*100:.1f}%** ")
    lines.append(f"of cases (95% CI: {h1['incorrect']['ci_attack_vocab_95'][0]*100:.1f}%-"
                 f"{h1['incorrect']['ci_attack_vocab_95'][1]*100:.1f}%).")
    lines.append("")
    lines.append(f"Actor-name top-token rates: {h1['correct']['pct_actor_name']*100:.1f}% on")
    lines.append(f"correct predictions versus {h1['incorrect']['pct_actor_name']*100:.1f}% on")
    lines.append(f"incorrect predictions.")
    lines.append("")

    lines.append("## H2: Mean LOO importance by token category")
    lines.append("")
    lines.append("| Category | n_tokens | mean_importance_raw | std | 95% CI |")
    lines.append("|----------|---------:|--------------------:|----:|--------|")
    for cat in ["ATTACK_VOCAB", "ACTOR_NAME", "STOPWORD", "OTHER"]:
        row = h2[cat]
        if row["n"] == 0:
            continue
        ci = row["ci_raw_95"]
        lines.append(f"| {cat} | {row['n']} | {row['mean_importance_raw']:.4f} | "
                     f"{row['std_importance_raw']:.4f} | "
                     f"[{ci[0]:.4f}, {ci[1]:.4f}] |")
    lines.append("")
    if "attack_vs_actor_ratio" in h2:
        ratio = h2["attack_vs_actor_ratio"]
        lines.append(f"**ATT&CK-vocabulary tokens carried mean importance "
                     f"{ratio:.2f}x that of actor-name tokens.**")
    lines.append("")

    lines.append("## A1: Per-actor breakdown")
    lines.append("")
    lines.append("| Actor | n_queries | P@1 | Top-token ATT&CK% | Top-token Actor% |")
    lines.append("|-------|----------:|----:|------------------:|-----------------:|")
    for actor in sorted(a1.keys()):
        row = a1[actor]
        if row["n_queries"] == 0:
            continue
        p1 = row["p_at_1"] if row["p_at_1"] is not None else 0
        att = row["pct_top_token_attack_vocab"] if row["pct_top_token_attack_vocab"] is not None else 0
        actorpct = row["pct_top_token_actor_name"] if row["pct_top_token_actor_name"] is not None else 0
        lines.append(f"| {actor} | {row['n_queries']} | {p1*100:.1f}% | "
                     f"{att*100:.1f}% | {actorpct*100:.1f}% |")
    lines.append("")

    lines.append("## A3: Margin vs importance-entropy correlation")
    lines.append("")
    if "error" in a3:
        lines.append(f"_{a3['error']}_")
    else:
        lines.append(f"Pearson r = {a3['pearson_r']:.4f} (p = {a3['pearson_p']:.4g})")
        lines.append(f"Spearman r = {a3['spearman_r']:.4f} (p = {a3['spearman_p']:.4g})")
        lines.append(f"")
        lines.append(f"Mean entropy on correct predictions: {a3['mean_entropy_correct']:.4f}")
        if a3['mean_entropy_incorrect'] is not None:
            lines.append(f"Mean entropy on incorrect predictions: {a3['mean_entropy_incorrect']:.4f}")
        lines.append("")
        lines.append("Negative correlation indicates concentrated importance (low entropy)")
        lines.append("co-occurs with high model confidence (high margin), which is the")
        lines.append("deployment-readiness pattern we want.")
    lines.append("")

    lines.append("## A4: BM25 overlap correspondence")
    lines.append("")
    lines.append(f"Queries with at least one HIGH-impact token: {a4['n_queries_with_high_tokens']}")
    lines.append(f"Queries where gold candidate text was unavailable: {a4['n_queries_missing_gold_text']}")
    lines.append(f"Mean overlap rate (HIGH tokens that also appear in gold candidate): "
                 f"{a4['mean_overlap_rate_overall']:.4f}")
    if "mean_overlap_rate_correct" in a4:
        lines.append(f"On correct predictions: {a4['mean_overlap_rate_correct']:.4f}")
        lines.append(f"On incorrect predictions: {a4['mean_overlap_rate_incorrect']:.4f}")
        if "correct_vs_incorrect_p" in a4:
            lines.append(f"Welch's t-test: t = {a4['correct_vs_incorrect_t']:.3f}, "
                         f"p = {a4['correct_vs_incorrect_p']:.4g}")
    lines.append("")

    lines.append("## A5: Position effect")
    lines.append("")
    if "error" in a5:
        lines.append(f"_{a5['error']}_")
    else:
        lines.append(f"Correlation between token position (normalized to fraction through query)")
        lines.append(f"and importance: Pearson r = {a5['pearson_r']:.4f} (p = {a5['pearson_p']:.4g}),")
        lines.append(f"Spearman r = {a5['spearman_r']:.4f} (p = {a5['spearman_p']:.4g})")
        lines.append("")
        lines.append("Values close to zero indicate the model treats positions roughly equally,")
        lines.append("which argues against systematic position bias.")
    lines.append("")

    lines.append("## A6: Per-query importance entropy")
    lines.append("")
    lines.append(f"Mean entropy across {a6['n_queries']} queries: {a6['mean_entropy']:.4f} bits")
    lines.append(f"Median: {a6['median_entropy']:.4f} | Std: {a6['std_entropy']:.4f}")
    lines.append(f"Range: [{a6['min_entropy']:.4f}, {a6['max_entropy']:.4f}]")
    lines.append("")

    lines.append("## Categorization cross-check")
    lines.append("")
    lines.append(f"STIX-vs-corpus vocabulary agreement: {cc['agreement_rate']*100:.1f}% of "
                 f"{cc['n_tokens']} tokens classified the same way by both methods.")
    lines.append(f"In STIX only: {cc['stix_only_pct']*100:.1f}% | "
                 f"In corpus only: {cc['corpus_only_pct']*100:.1f}% | "
                 f"In both: {cc['both_pct']*100:.1f}% | "
                 f"In neither: {cc['neither_pct']*100:.1f}%")
    lines.append("")

    lines.append("## Methodology notes")
    lines.append("")
    lines.append("**Primary categorization (STIX):** Tokens classified as ATTACK_VOCAB if")
    lines.append("they appear in the names or descriptions of attack-pattern, malware, tool,")
    lines.append("or x-mitre-tactic objects in the MITRE ATT&CK STIX bundle (enterprise-attack-v14.json).")
    lines.append("Revoked and deprecated objects are excluded.")
    lines.append("")
    lines.append("**Cross-check (corpus frequency):** Tokens also marked as in_corpus if they")
    lines.append("appear in any candidate text or in 5+ distinct queries (matching the")
    lines.append("auto_word_categorize.py methodology).")
    lines.append("")
    lines.append("**Actor names (conservative list, 9 tokens):**")
    lines.append(", ".join(sorted(CANONICAL_ACTORS)))
    lines.append("")
    lines.append("**HIGH-impact tokens:** Those whose `impact` column is 'HIGH' in")
    lines.append("token_importance.csv (preserved from explain.py's categorization).")
    lines.append("")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  Wrote {path}")


def write_visualizations(token_df, query_df, output_dir):
    """
    Generate two figures:
      1. importance_by_category.png - bar chart of mean importance by category
      2. correctness_stratified.png - same but stratified by correctness
    """
    # Figure 1: Mean importance by category
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["ATTACK_VOCAB", "ACTOR_NAME", "STOPWORD", "OTHER"]
    means = []
    errors = []
    counts = []
    for cat in categories:
        sub = token_df[token_df['category'] == cat]['importance_raw']
        if len(sub) > 0:
            means.append(sub.mean())
            errors.append(sub.std() / np.sqrt(len(sub)))  # SEM
            counts.append(len(sub))
        else:
            means.append(0)
            errors.append(0)
            counts.append(0)

    colors = ['#534AB7', '#D97757', '#888780', '#AFA9EC']
    bars = ax.bar(categories, means, yerr=errors, capsize=4, color=colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean LOO Importance (raw)')
    ax.set_xlabel('Token Category')
    ax.set_title('Mean LOO Importance by Token Category')
    ax.axhline(y=0, color='black', linewidth=0.3)
    for bar, count in zip(bars, counts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + (0.02 if h >= 0 else -0.05),
                f'n={count}', ha='center', fontsize=8)
    plt.tight_layout()
    path1 = os.path.join(output_dir, "importance_by_category.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"  Wrote {path1}")

    # Figure 2: Correctness-stratified
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(categories))
    width = 0.35

    means_c = []
    means_i = []
    sems_c = []
    sems_i = []
    for cat in categories:
        sub_c = token_df[(token_df['category'] == cat) & (token_df['correct'] == True)]['importance_raw']
        sub_i = token_df[(token_df['category'] == cat) & (token_df['correct'] == False)]['importance_raw']
        means_c.append(sub_c.mean() if len(sub_c) > 0 else 0)
        means_i.append(sub_i.mean() if len(sub_i) > 0 else 0)
        sems_c.append(sub_c.std() / np.sqrt(len(sub_c)) if len(sub_c) > 0 else 0)
        sems_i.append(sub_i.std() / np.sqrt(len(sub_i)) if len(sub_i) > 0 else 0)

    ax.bar(x - width/2, means_c, width, yerr=sems_c, capsize=3,
           label='Correct predictions', color='#0F6E56', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, means_i, width, yerr=sems_i, capsize=3,
           label='Incorrect predictions', color='#A0411C', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Mean LOO Importance (raw)')
    ax.set_xlabel('Token Category')
    ax.set_title('Mean LOO Importance by Category, Stratified by Correctness')
    ax.axhline(y=0, color='black', linewidth=0.3)
    ax.legend()
    plt.tight_layout()
    path2 = os.path.join(output_dir, "correctness_stratified.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"  Wrote {path2}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 65)
    print("  TOKEN IMPORTANCE ANALYSIS")
    print("  CTI-to-ATT&CK Reranker - Threat-Model Paragraph Numbers")
    print("=" * 65 + "\n")

    preflight()

    # Step 1: Build the two vocabularies
    print("=" * 65)
    print("  STEP 1: Build categorization vocabularies")
    print("=" * 65)
    stix_vocab = build_stix_vocabulary(ATTACK_STIX)
    corpus_vocab = build_corpus_vocabulary(TRAINING_DATA)

    # Step 2: Load the token CSV and categorize each row
    print("\n" + "=" * 65)
    print("  STEP 2: Load and categorize tokens")
    print("=" * 65)
    print(f"  Reading {TOKEN_CSV}...")
    token_df = pd.read_csv(TOKEN_CSV, encoding='utf-8')
    print(f"  Loaded {len(token_df)} token rows")

    print("  Categorizing tokens...")
    cats = []
    in_stix_flags = []
    in_corpus_flags = []
    for token in token_df['token']:
        cat, in_stix, in_corpus = categorize_token(token, stix_vocab, corpus_vocab, CANONICAL_ACTORS)
        cats.append(cat)
        in_stix_flags.append(in_stix)
        in_corpus_flags.append(in_corpus)
    token_df['category'] = cats
    token_df['in_stix'] = in_stix_flags
    token_df['in_corpus'] = in_corpus_flags

    cat_counts = Counter(cats)
    print(f"  Category distribution:")
    for cat in ["ATTACK_VOCAB", "ACTOR_NAME", "STOPWORD", "OTHER"]:
        n = cat_counts.get(cat, 0)
        pct = 100 * n / len(token_df) if len(token_df) > 0 else 0
        print(f"    {cat}: {n} ({pct:.1f}%)")

    # Step 3: Per-query analysis
    print("\n" + "=" * 65)
    print("  STEP 3: Per-query analysis (entropy, BM25 overlap)")
    print("=" * 65)
    candidate_texts = load_gold_candidate_texts(TRAINING_DATA)
    query_df = compute_per_query_metrics(token_df, candidate_texts)
    print(f"  Computed metrics for {len(query_df)} queries")

    # Step 4: Aggregations
    print("\n" + "=" * 65)
    print("  STEP 4: Compute aggregations")
    print("=" * 65)
    agg = compute_aggregations(token_df, query_df)

    # Step 5: Output
    print("\n" + "=" * 65)
    print("  STEP 5: Write outputs")
    print("=" * 65)
    write_csvs(token_df, query_df, OUTPUT_DIR)
    write_aggregations_json(agg, OUTPUT_DIR)
    write_summary_markdown(agg, OUTPUT_DIR)
    write_visualizations(token_df, query_df, OUTPUT_DIR)

    # Step 6: Headline summary printed to console
    print("\n" + "=" * 65)
    print("  HEADLINE NUMBERS")
    print("=" * 65)
    h1 = agg["H1_top_token_by_correctness"]
    h2 = agg["H2_mean_importance_by_category"]
    print(f"\n  H1: Top-token is ATT&CK-vocabulary in...")
    print(f"      {h1['correct']['pct_attack_vocab']*100:.1f}% of "
          f"correct predictions ({h1['correct']['n']} queries)")
    print(f"      {h1['incorrect']['pct_attack_vocab']*100:.1f}% of "
          f"incorrect predictions ({h1['incorrect']['n']} queries)")
    print(f"\n  H2: Mean LOO importance by category:")
    for cat in ["ATTACK_VOCAB", "ACTOR_NAME", "STOPWORD", "OTHER"]:
        row = h2.get(cat, {})
        if row.get('n', 0) > 0:
            print(f"      {cat}: {row['mean_importance_raw']:.4f} (n={row['n']})")
    if "attack_vs_actor_ratio" in h2:
        print(f"\n  ATT&CK-vocab importance is {h2['attack_vs_actor_ratio']:.2f}x "
              f"actor-name importance")

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)
    print(f"  All outputs in: {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
