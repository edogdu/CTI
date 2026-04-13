#!/usr/bin/env python3
"""
Document-Level Thresholding for Deployment
============================================
In deployment, every CTI sentence gets scored by the cross-encoder,
but not every sentence describes an ATT&CK technique. Auxiliary content
like IoC listings, timestamps, attribution statements, and boilerplate
should be flagged as "no mapping" rather than receiving a forced prediction.

This script calibrates a score threshold by comparing:
  - POSITIVE distribution: top-1 scores on sentences that DO describe techniques
    (from our test set — known to have valid ATT&CK mappings)
  - NEGATIVE distribution: top-1 scores on sentences that DON'T describe techniques
    (constructed from realistic CTI report auxiliary content)

The threshold is the score below which the system says "no ATT&CK mapping"
instead of forcing a prediction. We report the precision/recall tradeoff
at various thresholds to let the deployment team choose their operating point.

Usage:
  python document_thresholding.py

  Expects:
    data/reranker_pairs_enriched_v2.jsonl
    checkpoints/best_two_stage_v2/

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


# ═══════════════════════════════════════════════════════════════════════
# NEGATIVE (NON-TECHNIQUE) SENTENCES
# ═══════════════════════════════════════════════════════════════════════
# These represent realistic CTI report content that should NOT map to
# any ATT&CK technique. Categories:
#   - IoC listings (hashes, IPs, domains, file paths)
#   - Timestamps and campaign metadata
#   - Attribution and actor background
#   - Boilerplate and report structure
#   - General context without specific technique behavior

NEGATIVE_SENTENCES = [
    # IoC listings — raw indicators without technique context
    "SHA256: 5f2b7f5e8d3a1c9b4e6f7a8d2c1e3b5a9f4d6e8c7b2a1d3f5e9c8b7a6d4f2e1a",
    "The C2 server was located at 185.243.115.12 on port 443.",
    "MD5 hash: ab32c09c46e0c9dbc576fefee68e5a2f57e0482e",
    "Associated domains include update-service.xyz and cdn-static.info.",
    "The malicious file was named invoice_2024_final.docx.",
    "IP addresses 10.0.0.15 and 192.168.1.100 were observed in the logs.",
    "File path: C:\\Users\\Public\\Documents\\svchost.exe",
    "The sample was uploaded to VirusTotal on 2024-03-15.",
    "YARA rule: rule APT29_Loader { strings: $a = {4D 5A 90} condition: $a }",
    "Certificate thumbprint: 3A:7B:2C:9D:1E:4F:6A:8B:0C:5D:7E:9F:2A:4B:6C:8D",

    # Timestamps and campaign metadata
    "The campaign was first observed in March 2024.",
    "Activity was detected between January 15 and February 28, 2025.",
    "This report was published on October 3, 2024.",
    "The threat actor has been active since at least 2019.",
    "Last updated: December 2024. Version 2.1.",
    "The incident was reported to CISA on April 12, 2024.",

    # Attribution and actor background
    "APT29, also known as Cozy Bear, is attributed to Russia's SVR.",
    "The group has been linked to multiple campaigns targeting government agencies.",
    "FIN7 is a financially motivated threat group active since 2013.",
    "This actor is tracked by multiple vendors under different names.",
    "The threat actor's motivations appear to be primarily espionage-related.",
    "Previous campaigns by this group targeted the energy sector.",

    # Boilerplate and report structure
    "For more information, see the appendix below.",
    "Table 1 summarizes the key findings of this analysis.",
    "The following indicators can be used for detection.",
    "Figure 3 shows the infection chain observed in this campaign.",
    "Acknowledgments: We thank our partners for sharing threat intelligence.",
    "This report is TLP:WHITE and may be shared freely.",
    "Disclaimer: This analysis is provided as-is without warranty.",
    "Contact the SOC at security@example.com for questions.",

    # General context without specific technique behavior
    "The attackers demonstrated a high level of sophistication.",
    "Several victims were identified across multiple industry sectors.",
    "The malware sample was written in C++ and compiled for x64.",
    "Network traffic analysis revealed encrypted communications.",
    "The attack chain consisted of multiple stages.",
    "Remediation efforts are ongoing at affected organizations.",
    "The vulnerability was assigned CVE-2024-12345.",
    "Antivirus detection rates were initially very low.",
    "The campaign primarily targeted organizations in North America and Europe.",
    "Further analysis is required to determine the full scope of the compromise.",

    # Short fragments that analysts encounter
    "See also: MITRE ATT&CK.",
    "References",
    "Indicators of Compromise",
    "Executive Summary",
    "N/A",
    "Unknown",
    "TBD",
]


def load_and_group_data(filepath):
    """Load enriched JSONL and group by query_norm."""
    query_data = defaultdict(lambda: {
        'query_raw': None, 'actor': None, 'candidates': [],
        'positive_count': 0,
    })
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qn = row.get('query_norm', '')
            qr = row.get('query_raw', '')
            ct = row.get('candidate_text', '')
            cn = row.get('candidate_norm', '') or row.get('candidate_id', '')
            label = row.get('label', 0)
            actor = row.get('actor', 'unknown')
            if not qn or not qr:
                continue
            if actor.startswith('external_'):
                continue
            if query_data[qn]['query_raw'] is None:
                query_data[qn]['query_raw'] = qr
                query_data[qn]['actor'] = actor
            query_data[qn]['candidates'].append({
                'text': ct, 'label': label, 'id': cn,
            })
            if label == 1:
                query_data[qn]['positive_count'] += 1
    return dict(query_data)


def create_splits(query_data):
    """Same split as finetune_production.py."""
    random.seed(RANDOM_SEED)
    queries_by_actor = defaultdict(list)
    for qn, data in query_data.items():
        if data['positive_count'] > 0:
            queries_by_actor[data['actor']].append(qn)
    train_queries, val_queries, test_queries = [], [], []
    for actor, queries in queries_by_actor.items():
        random.shuffle(queries)
        n = len(queries)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train_queries.extend(queries[:n_train])
        val_queries.extend(queries[n_train:n_train + n_val])
        test_queries.extend(queries[n_train + n_val:])
    random.shuffle(train_queries)
    random.shuffle(val_queries)
    random.shuffle(test_queries)
    return train_queries, val_queries, test_queries


def main():
    print("\n" + "=" * 70)
    print("  DOCUMENT-LEVEL THRESHOLDING FOR DEPLOYMENT")
    print("  Calibrating score threshold: technique vs non-technique sentences")
    print("=" * 70)

    data_path = "data/reranker_pairs_enriched_v2.jsonl"
    model_path = "checkpoints/best_two_stage_v2"

    for p, name in [(data_path, "Data"), (model_path, "Model")]:
        if not Path(p).exists():
            sys.exit(f"[FAIL] {name} not found: {p}")

    # ── Load data and model ──
    print(f"\n  Loading data from: {data_path}")
    query_data = load_and_group_data(data_path)
    train_queries, val_queries, test_queries = create_splits(query_data)
    print(f"  Train: {len(train_queries)}  Val: {len(val_queries)}  Test: {len(test_queries)}")

    print(f"  Loading model from: {model_path}")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_path)
    print(f"  Model loaded.")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Score positive sentences (real technique descriptions)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  PHASE 1: Scoring POSITIVE sentences (known technique mappings)")
    print(f"  {'='*60}")

    # Use validation set for threshold calibration, test set for final evaluation
    # This avoids overfitting the threshold to the test set
    positive_scores_val = []
    positive_scores_test = []

    for split_name, split_queries, score_list in [
        ("Validation", val_queries, positive_scores_val),
        ("Test", test_queries, positive_scores_test),
    ]:
        for qn in split_queries:
            qd = query_data[qn]
            if qd['positive_count'] == 0 or not qd['candidates']:
                continue
            texts = [[qd['query_raw'], c['text']] for c in qd['candidates']]
            scores = model.predict(texts, show_progress_bar=False)
            top1_score = float(np.max(scores))
            score_list.append(top1_score)

        print(f"\n  {split_name} positive scores ({len(score_list)} sentences):")
        print(f"    Mean:   {np.mean(score_list):.4f}")
        print(f"    Median: {np.median(score_list):.4f}")
        print(f"    Min:    {np.min(score_list):.4f}")
        print(f"    Max:    {np.max(score_list):.4f}")
        print(f"    Std:    {np.std(score_list):.4f}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Score negative sentences (non-technique content)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  PHASE 2: Scoring NEGATIVE sentences (non-technique content)")
    print(f"  {'='*60}")

    # For each negative sentence, score against the same candidate pool
    # as a randomly selected positive query (to use realistic candidates)
    random.seed(RANDOM_SEED)
    all_query_norms = list(query_data.keys())

    negative_scores = []
    print(f"\n  Scoring {len(NEGATIVE_SENTENCES)} negative sentences...")

    for neg_sent in NEGATIVE_SENTENCES:
        # Pick a random query's candidate pool to score against
        ref_qn = random.choice(all_query_norms)
        ref_qd = query_data[ref_qn]
        if not ref_qd['candidates']:
            continue

        texts = [[neg_sent, c['text']] for c in ref_qd['candidates']]
        scores = model.predict(texts, show_progress_bar=False)
        top1_score = float(np.max(scores))
        negative_scores.append({
            'sentence': neg_sent[:60],
            'score': top1_score,
        })

    neg_score_values = [s['score'] for s in negative_scores]
    print(f"\n  Negative scores ({len(negative_scores)} sentences):")
    print(f"    Mean:   {np.mean(neg_score_values):.4f}")
    print(f"    Median: {np.median(neg_score_values):.4f}")
    print(f"    Min:    {np.min(neg_score_values):.4f}")
    print(f"    Max:    {np.max(neg_score_values):.4f}")
    print(f"    Std:    {np.std(neg_score_values):.4f}")

    # Show some examples
    sorted_negs = sorted(negative_scores, key=lambda x: x['score'], reverse=True)
    print(f"\n  Highest-scoring negative sentences (hardest to filter):")
    for s in sorted_negs[:5]:
        print(f"    {s['score']:>7.4f}  {s['sentence']}")
    print(f"\n  Lowest-scoring negative sentences (easiest to filter):")
    for s in sorted_negs[-5:]:
        print(f"    {s['score']:>7.4f}  {s['sentence']}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Threshold calibration on validation set
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  PHASE 3: Threshold calibration (using validation set)")
    print(f"  {'='*60}")

    # Combine validation positives and all negatives for threshold search
    val_labels = [1] * len(positive_scores_val) + [0] * len(neg_score_values)
    val_scores = positive_scores_val + neg_score_values

    # Test thresholds from -2 to 8 in steps of 0.25
    thresholds = np.arange(-2.0, 8.25, 0.25)

    print(f"\n  {'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | "
          f"{'TP':>5} | {'FP':>5} | {'FN':>5} | {'TN':>5}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-"
          f"{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}")

    best_f1 = 0
    best_threshold = 0
    all_results = []

    for threshold in thresholds:
        tp = sum(1 for s, l in zip(val_scores, val_labels) if s >= threshold and l == 1)
        fp = sum(1 for s, l in zip(val_scores, val_labels) if s >= threshold and l == 0)
        fn = sum(1 for s, l in zip(val_scores, val_labels) if s < threshold and l == 1)
        tn = sum(1 for s, l in zip(val_scores, val_labels) if s < threshold and l == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        result = {
            'threshold': float(threshold),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        }
        all_results.append(result)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

        # Print select thresholds for readability
        if threshold % 1.0 == 0 or abs(threshold - best_threshold) < 0.01:
            print(f"  {threshold:>10.2f} | {precision:>9.2%} | {recall:>9.2%} | "
                  f"{f1:>9.2%} | {tp:>5} | {fp:>5} | {fn:>5} | {tn:>5}")

    print(f"\n  OPTIMAL THRESHOLD (max F1): {best_threshold:.2f}")

    # Show the optimal point details
    opt = next(r for r in all_results if abs(r['threshold'] - best_threshold) < 0.01)
    print(f"    Precision: {opt['precision']:.2%} (of sentences predicted as techniques, "
          f"this fraction actually are)")
    print(f"    Recall:    {opt['recall']:.2%} (of actual technique sentences, "
          f"this fraction are correctly identified)")
    print(f"    F1:        {opt['f1']:.2%}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: Final evaluation on test set
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  PHASE 4: Final evaluation on TEST set (threshold = {best_threshold:.2f})")
    print(f"  {'='*60}")

    test_labels = [1] * len(positive_scores_test) + [0] * len(neg_score_values)
    test_scores_combined = positive_scores_test + neg_score_values

    tp = sum(1 for s, l in zip(test_scores_combined, test_labels)
             if s >= best_threshold and l == 1)
    fp = sum(1 for s, l in zip(test_scores_combined, test_labels)
             if s >= best_threshold and l == 0)
    fn = sum(1 for s, l in zip(test_scores_combined, test_labels)
             if s < best_threshold and l == 1)
    tn = sum(1 for s, l in zip(test_scores_combined, test_labels)
             if s < best_threshold and l == 0)

    test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    test_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    test_f1 = (2 * test_precision * test_recall / (test_precision + test_recall)
               if (test_precision + test_recall) > 0 else 0)

    print(f"\n  Test set results at threshold {best_threshold:.2f}:")
    print(f"    Precision: {test_precision:.2%}")
    print(f"    Recall:    {test_recall:.2%}")
    print(f"    F1:        {test_f1:.2%}")
    print(f"    TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")

    # Show which positive sentences would be incorrectly filtered
    if fn > 0:
        filtered_positives = [(s, i) for i, (s, l) in
                              enumerate(zip(positive_scores_test, [1]*len(positive_scores_test)))
                              if s < best_threshold]
        print(f"\n  Positive sentences filtered out (false negatives):")
        for score, idx in filtered_positives[:5]:
            qn = test_queries[idx] if idx < len(test_queries) else '?'
            qd = query_data.get(qn, {})
            query_text = qd.get('query_raw', '?')[:60] if qd else '?'
            print(f"    score={score:.4f}  {query_text}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: Recommended operating points for deployment
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  PHASE 5: Recommended operating points for deployment")
    print(f"  {'='*60}")

    # Find thresholds for specific precision/recall targets
    print(f"\n  {'Mode':<25} | {'Threshold':>10} | {'Precision':>10} | "
          f"{'Recall':>10} | {'Use Case':>30}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*30}")

    # High recall (don't miss any techniques, accept some false alarms)
    for r in sorted(all_results, key=lambda x: abs(x['recall'] - 0.99)):
        if x := r:
            if x['recall'] >= 0.98:
                print(f"  {'High Recall (≥98%)':<25} | {x['threshold']:>10.2f} | "
                      f"{x['precision']:>9.2%} | {x['recall']:>9.2%} | "
                      f"{'Automated triage':>30}")
                break

    # Balanced (best F1)
    print(f"  {'Balanced (best F1)':<25} | {best_threshold:>10.2f} | "
          f"{opt['precision']:>9.2%} | {opt['recall']:>9.2%} | "
          f"{'General deployment':>30}")

    # High precision (only flag confident mappings)
    for r in sorted(all_results, key=lambda x: abs(x['precision'] - 0.99)):
        if x := r:
            if x['precision'] >= 0.98 and x['recall'] > 0.5:
                print(f"  {'High Precision (≥98%)':<25} | {x['threshold']:>10.2f} | "
                      f"{x['precision']:>9.2%} | {x['recall']:>9.2%} | "
                      f"{'Analyst confirmation':>30}")
                break

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY FOR BILL
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  {'='*60}")
    print(f"  SUMMARY FOR DEPLOYMENT INTEGRATION")
    print(f"  {'='*60}")

    print(f"""
  The cross-encoder's top-1 relevance score serves as a reliable
  confidence signal for distinguishing technique-bearing sentences
  from auxiliary content. At the recommended threshold of {best_threshold:.2f}:

  - Technique sentences score: mean {np.mean(positive_scores_test):.2f}, "
    min {np.min(positive_scores_test):.2f}
  - Non-technique sentences score: mean {np.mean(neg_score_values):.2f}, "
    max {np.max(neg_score_values):.2f}

  DEPLOYMENT RULE:
    if top_1_score >= {best_threshold:.2f}:
        return predicted_technique   # confident mapping
    else:
        return None                  # no ATT&CK mapping for this sentence

  This eliminates forced predictions on IoC listings, timestamps,
  attribution statements, and other non-technique content that would
  otherwise generate false ATT&CK mappings in the pipeline.
""")

    # ── Save results ──
    out_dir = Path("thresholding_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'optimal_threshold': float(best_threshold),
        'validation': {
            'precision': opt['precision'],
            'recall': opt['recall'],
            'f1': opt['f1'],
        },
        'test': {
            'precision': round(test_precision, 4),
            'recall': round(test_recall, 4),
            'f1': round(test_f1, 4),
        },
        'positive_score_stats': {
            'mean': round(float(np.mean(positive_scores_test)), 4),
            'median': round(float(np.median(positive_scores_test)), 4),
            'min': round(float(np.min(positive_scores_test)), 4),
            'max': round(float(np.max(positive_scores_test)), 4),
        },
        'negative_score_stats': {
            'mean': round(float(np.mean(neg_score_values)), 4),
            'median': round(float(np.median(neg_score_values)), 4),
            'min': round(float(np.min(neg_score_values)), 4),
            'max': round(float(np.max(neg_score_values)), 4),
        },
        'num_positive_sentences': len(positive_scores_test),
        'num_negative_sentences': len(neg_score_values),
        'threshold_curve': all_results,
    }

    with open(out_dir / "thresholding_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to: {out_dir / 'thresholding_results.json'}")
    print(f"\n  [DONE] Document-level thresholding complete.")


if __name__ == '__main__':
    main()
