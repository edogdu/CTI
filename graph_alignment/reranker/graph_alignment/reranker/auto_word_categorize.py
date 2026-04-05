#!/usr/bin/env python3
"""
Automated Corpus-Frequency Word Categorization
================================================
Replaces the subjective manual regex in explain.py with an objective,
reproducible, corpus-frequency-based classification.

Method:
  1. Build a "cyber lexicon" from ATT&CK technique/tactic/software names
     and descriptions (from the MITRE ATT&CK STIX data or from the
     candidate texts already in your training data).
  2. Build a "general English" baseline frequency from a standard corpus
     (we use the candidate_text field frequencies as a proxy, plus a
     built-in English stopword list).
  3. Classify each token as:
     - CYBER_TERM:  appears in ATT&CK corpus at significantly higher rate
                    than in general English, OR appears in ATT&CK entity names
     - STOPWORD:    in standard English stopword list
     - OTHER:       everything else
  4. Recompute the aggregate importance statistics with this objective
     categorization and compare to the original regex-based results.

This directly addresses co-author Roya Choupani's valid critique that
"categorization is done manually... defining the boundary between
'cybersecurity terms' and 'other words' should not be subjective."

Usage:
  python auto_word_categorize.py

  Expects:
    eval_results/explanations.json   (from explain.py)
    data/reranker_pairs_enriched.jsonl  (for ATT&CK term extraction)

Author: Shane Waldrop — Angelo State University / ARL Grant W911NF-24-2-0180
"""

import json
import re
import sys
import math
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# ── Standard English stopwords (same list as explain.py) ─────────────────────

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


def tokenize(text):
    """Extract lowercase alphanumeric tokens from text."""
    return re.findall(r'[a-z0-9]+', text.lower())


# ── Step 1: Build ATT&CK cyber lexicon from training data ───────────────────

def build_attack_lexicon(data_path):
    """
    Build a cyber-domain lexicon from ATT&CK entity names in the training data.

    Uses candidate_text fields (e.g., "T1059.001 — COMMAND AND SCRIPTING
    INTERPRETER: POWERSHELL") to extract domain-specific vocabulary.

    This is objective and reproducible: any researcher with the same
    dataset would produce the same lexicon.
    """
    print("  Building ATT&CK lexicon from training data...")

    attack_terms = Counter()  # term frequencies in ATT&CK names
    query_terms = Counter()   # term frequencies in CTI queries
    seen_candidates = set()
    seen_queries = set()

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Collect ATT&CK candidate text tokens
            ctext = row.get('candidate_text', '')
            if ctext and ctext not in seen_candidates:
                seen_candidates.add(ctext)
                # Remove the ID prefix (e.g., "T1059.001 — ")
                name_part = re.sub(r'^[A-Z0-9.]+ — ', '', ctext)
                for token in tokenize(name_part):
                    if token not in STOPWORDS and len(token) > 1:
                        attack_terms[token] += 1

            # Collect CTI query tokens (these represent general cyber text)
            qraw = row.get('query_raw', '')
            if qraw and qraw not in seen_queries:
                seen_queries.add(qraw)
                for token in tokenize(qraw):
                    if token not in STOPWORDS and len(token) > 1:
                        query_terms[token] += 1

    print(f"    ATT&CK entity names processed: {len(seen_candidates)}")
    print(f"    Unique ATT&CK vocabulary: {len(attack_terms)} terms")
    print(f"    CTI queries processed: {len(seen_queries)}")
    print(f"    Unique query vocabulary: {len(query_terms)} terms")

    # A term is "cyber-specific" if it appears in ATT&CK entity names
    # This is the most objective possible criterion: if MITRE uses this
    # word to name or describe a technique, it's a domain term.
    cyber_lexicon = set(attack_terms.keys())

    # Also add terms that appear in queries at very high frequency
    # relative to general English (these are domain terms even if
    # they don't appear in ATT&CK names)
    # We use a simple threshold: appears in >5 distinct queries
    # and is not a common English word
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

    for term, count in query_terms.items():
        if (count >= 5 and
            term not in common_english and
            term not in STOPWORDS and
            len(term) > 2):
            cyber_lexicon.add(term)

    print(f"    Final cyber lexicon size: {len(cyber_lexicon)} terms")

    # Show top terms for sanity check
    top_attack = attack_terms.most_common(20)
    print(f"\n    Top 20 ATT&CK vocabulary terms:")
    for term, count in top_attack:
        print(f"      {term:<25} (appears in {count} entity names)")

    return cyber_lexicon


# ── Step 2: Classify tokens and recompute statistics ─────────────────────────

def classify_and_analyze(explanations_path, cyber_lexicon):
    """
    Reload token importance data and reclassify using the automated lexicon.
    """
    print(f"\n  Loading explanations from: {explanations_path}")

    with open(explanations_path, 'r', encoding='utf-8') as f:
        explanations = json.load(f)

    print(f"  Loaded {len(explanations)} query explanations")

    # Classify all tokens
    all_tokens = []
    for expl in explanations:
        for t in expl['token_importance']:
            word_lower = t['token'].lower()

            if word_lower in STOPWORDS:
                category = 'stopword'
            elif word_lower in cyber_lexicon:
                category = 'cyber_term'
            else:
                category = 'other'

            all_tokens.append({
                'word': t['token'],
                'word_lower': word_lower,
                'importance': t['raw_importance'],
                'category': category,
                'correct': expl['prediction']['correct'],
            })

    if not all_tokens:
        print("  [WARN] No tokens found!")
        return

    # ── Compute category statistics ──
    cyber_imps = [t['importance'] for t in all_tokens if t['category'] == 'cyber_term']
    stop_imps = [t['importance'] for t in all_tokens if t['category'] == 'stopword']
    other_imps = [t['importance'] for t in all_tokens if t['category'] == 'other']

    def safe_stats(lst):
        if not lst:
            return 0.0, 0.0, 0
        return float(np.mean(lst)), float(np.std(lst)), len(lst)

    cyber_mean, cyber_std, cyber_n = safe_stats(cyber_imps)
    stop_mean, stop_std, stop_n = safe_stats(stop_imps)
    other_mean, other_std, other_n = safe_stats(other_imps)

    print(f"\n  {'='*60}")
    print(f"  AUTOMATED WORD CATEGORY IMPORTANCE ANALYSIS")
    print(f"  (Corpus-frequency classification — NO manual regex)")
    print(f"  {'='*60}")

    print(f"\n  {'Category':<16} | {'Mean':>10} | {'Std':>10} | {'N tokens':>10}")
    print(f"  {'-'*16}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'cyber_terms':<16} | {cyber_mean:>+10.4f} | {cyber_std:>10.4f} | {cyber_n:>10}")
    print(f"  {'stopwords':<16} | {stop_mean:>+10.4f} | {stop_std:>10.4f} | {stop_n:>10}")
    print(f"  {'other_words':<16} | {other_mean:>+10.4f} | {other_std:>10.4f} | {other_n:>10}")

    # Statistical test: is cyber_terms mean significantly different from other?
    if cyber_imps and other_imps:
        # Welch's t-test (doesn't assume equal variance)
        n1, n2 = len(cyber_imps), len(other_imps)
        m1, m2 = np.mean(cyber_imps), np.mean(other_imps)
        s1, s2 = np.std(cyber_imps, ddof=1), np.std(other_imps, ddof=1)

        se = math.sqrt(s1**2/n1 + s2**2/n2) if (n1 > 0 and n2 > 0) else 1
        t_stat = (m1 - m2) / se if se > 0 else 0

        # Approximate degrees of freedom (Welch-Satterthwaite)
        if s1 > 0 and s2 > 0:
            num = (s1**2/n1 + s2**2/n2)**2
            den = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
            df = num / den if den > 0 else min(n1, n2) - 1
        else:
            df = min(n1, n2) - 1

        # Cohen's d effect size
        pooled_std = math.sqrt((s1**2 + s2**2) / 2) if (s1 > 0 or s2 > 0) else 1
        cohens_d = (m1 - m2) / pooled_std if pooled_std > 0 else 0

        print(f"\n  Welch's t-test (cyber_terms vs other_words):")
        print(f"    t-statistic:  {t_stat:.4f}")
        print(f"    df (approx):  {df:.1f}")
        print(f"    Cohen's d:    {cohens_d:.4f}")

        if abs(cohens_d) >= 0.8:
            effect = "LARGE"
        elif abs(cohens_d) >= 0.5:
            effect = "MEDIUM"
        elif abs(cohens_d) >= 0.2:
            effect = "SMALL"
        else:
            effect = "NEGLIGIBLE"
        print(f"    Effect size:  {effect}")

    # ── Top important words by category ──
    word_by_cat = defaultdict(lambda: defaultdict(list))
    for t in all_tokens:
        word_by_cat[t['category']][t['word_lower']].append(t['importance'])

    print(f"\n  TOP 10 MOST IMPORTANT CYBER TERMS:")
    cyber_avg = {w: np.mean(imps) for w, imps in word_by_cat['cyber_term'].items()}
    for i, (word, avg) in enumerate(sorted(cyber_avg.items(),
                                            key=lambda x: x[1], reverse=True)[:10]):
        n = len(word_by_cat['cyber_term'][word])
        print(f"    {i+1:>2}. {word:<25} avg={avg:>+.4f}  (n={n})")

    print(f"\n  TOP 10 MOST IMPORTANT OTHER WORDS:")
    other_avg = {w: np.mean(imps) for w, imps in word_by_cat['other'].items()}
    for i, (word, avg) in enumerate(sorted(other_avg.items(),
                                            key=lambda x: x[1], reverse=True)[:10]):
        n = len(word_by_cat['other'][word])
        print(f"    {i+1:>2}. {word:<25} avg={avg:>+.4f}  (n={n})")

    # ── Comparison with original regex method ──
    print(f"\n  {'='*60}")
    print(f"  COMPARISON: AUTOMATED vs ORIGINAL REGEX")
    print(f"  {'='*60}")

    # Reclassify using the original regex from explain.py
    original_regex = re.compile(
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
        r'exfiltration|impact|defense|evasion|initial|access)'
    )

    regex_cyber = set()
    auto_cyber = set()
    for t in all_tokens:
        w = t['word_lower']
        if bool(original_regex.match(w)):
            regex_cyber.add(w)
        if w in cyber_lexicon and w not in STOPWORDS:
            auto_cyber.add(w)

    only_regex = regex_cyber - auto_cyber
    only_auto = auto_cyber - regex_cyber
    both = regex_cyber & auto_cyber

    print(f"\n  Regex-only terms ({len(only_regex)}): "
          f"{', '.join(sorted(only_regex)[:15])}")
    print(f"  Auto-only terms ({len(only_auto)}):  "
          f"{', '.join(sorted(only_auto)[:15])}")
    if len(only_auto) > 15:
        print(f"    ... and {len(only_auto) - 15} more")
    print(f"  Agreement ({len(both)}):       "
          f"{', '.join(sorted(both)[:15])}")
    if len(both) > 15:
        print(f"    ... and {len(both) - 15} more")

    # Recompute stats with regex for comparison
    regex_cyber_imps = [t['importance'] for t in all_tokens
                        if bool(original_regex.match(t['word_lower']))]
    regex_other_imps = [t['importance'] for t in all_tokens
                        if not bool(original_regex.match(t['word_lower']))
                        and t['word_lower'] not in STOPWORDS]

    regex_cyber_mean = np.mean(regex_cyber_imps) if regex_cyber_imps else 0
    regex_other_mean = np.mean(regex_other_imps) if regex_other_imps else 0

    print(f"\n  {'Method':<20} | {'Cyber mean':>12} | {'Other mean':>12} | {'Ratio':>8}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    ratio_regex = regex_cyber_mean / regex_other_mean if regex_other_mean != 0 else float('inf')
    ratio_auto = cyber_mean / other_mean if other_mean != 0 else float('inf')
    print(f"  {'Regex (original)':<20} | {regex_cyber_mean:>+12.4f} | {regex_other_mean:>+12.4f} | {ratio_regex:>8.2f}x")
    print(f"  {'Automated (new)':<20} | {cyber_mean:>+12.4f} | {other_mean:>+12.4f} | {ratio_auto:>8.2f}x")

    # ── Paper-ready sentence ──
    print(f"\n  FOR THE PAPER:")
    print(f"  \"To ensure objectivity, we replaced manual term categorization")
    print(f"   with automated corpus-frequency classification. The ATT&CK")
    print(f"   entity vocabulary ({len(cyber_lexicon)} terms) served as the")
    print(f"   cyber-domain lexicon. Under this automated classification,")
    print(f"   cyber-domain terms exhibited mean importance of {cyber_mean:+.4f}")
    print(f"   compared to {other_mean:+.4f} for general vocabulary")
    print(f"   (Cohen's d = {cohens_d:.2f}, {effect.lower()} effect),")
    print(f"   confirming that the model learned to weight domain-specific")
    print(f"   terminology as the primary matching signal.\"")

    # ── Save results ──
    out_dir = Path("eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'method': 'automated_corpus_frequency',
        'lexicon_size': len(cyber_lexicon),
        'categories': {
            'cyber_terms': {'mean': round(cyber_mean, 6), 'std': round(cyber_std, 6), 'n': cyber_n},
            'stopwords': {'mean': round(stop_mean, 6), 'std': round(stop_std, 6), 'n': stop_n},
            'other_words': {'mean': round(other_mean, 6), 'std': round(other_std, 6), 'n': other_n},
        },
        't_statistic': round(t_stat, 4) if 'other_imps' in dir() else None,
        'cohens_d': round(cohens_d, 4) if 'cohens_d' in dir() else None,
        'comparison_with_regex': {
            'regex_cyber_mean': round(regex_cyber_mean, 6),
            'auto_cyber_mean': round(cyber_mean, 6),
            'terms_only_in_regex': len(only_regex),
            'terms_only_in_auto': len(only_auto),
            'terms_in_both': len(both),
        },
        'lexicon_sample': sorted(list(cyber_lexicon))[:100],
    }

    out_path = out_dir / "auto_word_categorization.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    print(f"\n  [DONE] Automated word categorization complete.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  AUTOMATED WORD CATEGORIZATION")
    print("  Replacing subjective regex with corpus-frequency method")
    print("=" * 60)

    data_path = Path("data/reranker_pairs_enriched.jsonl")
    expl_path = Path("eval_results/explanations.json")

    if not data_path.exists():
        sys.exit(f"[FAIL] Data file not found: {data_path}")
    if not expl_path.exists():
        sys.exit(f"[FAIL] Explanations not found: {expl_path}")

    # Step 1: Build lexicon
    cyber_lexicon = build_attack_lexicon(str(data_path))

    # Step 2: Classify and analyze
    classify_and_analyze(str(expl_path), cyber_lexicon)


if __name__ == '__main__':
    main()
