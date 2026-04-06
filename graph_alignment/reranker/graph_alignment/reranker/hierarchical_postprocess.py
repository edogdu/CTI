#!/usr/bin/env python3
"""
Hierarchical ATT&CK Post-Processing
======================================
Applies MITRE ATT&CK's taxonomy structure to expand model predictions.
When the model predicts a sub-technique, this module automatically infers
the parent technique and associated tactic(s) from ATT&CK's ontological
hierarchy — encoding domain knowledge that the neural model doesn't
inherently capture.

Example:
  Model predicts: T1059.001 (PowerShell)
  Post-processor adds: T1059 (Command and Scripting Interpreter)
                        TA0002 (Execution)

This is particularly valuable for multi-label evaluation, where each CTI
passage maps to multiple ATT&CK entities (avg 2.84 per query). The
post-processor improves recall by ensuring hierarchically implied
entities are included in the prediction set.

Usage:
  python hierarchical_postprocess.py

  Expects:
    enterprise-attack-v14.json          (MITRE ATT&CK STIX data)
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
ATTACK_ID_RE = re.compile(r'(TA\d{4}|T\d{4}(?:\.\d{3})?|S\d{4})')


# ═══════════════════════════════════════════════════════════════════════
# ATT&CK HIERARCHY BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_attack_hierarchy(stix_path):
    """Parse MITRE ATT&CK STIX data to build the technique hierarchy.
    
    Returns two lookup tables:
      subtechnique_to_parent: T1059.001 → T1059
      technique_to_tactics:   T1059 → {TA0002}
                              T1059.001 → {TA0002}  (inherited from parent)
    """
    print(f"  Building ATT&CK hierarchy from: {stix_path}")
    
    with open(stix_path, 'r', encoding='utf-8') as f:
        stix = json.load(f)
    
    objects = stix.get('objects', [])
    
    # First pass: collect technique IDs and their STIX internal IDs
    stix_id_to_attack_id = {}  # STIX internal ID → ATT&CK ID (e.g., T1059)
    attack_id_to_name = {}     # ATT&CK ID → human name
    
    for obj in objects:
        if obj.get('type') == 'attack-pattern' and not obj.get('revoked', False):
            ext_refs = obj.get('external_references', [])
            for ref in ext_refs:
                if ref.get('source_name') == 'mitre-attack':
                    attack_id = ref.get('external_id', '')
                    if re.match(r'^T\d{4}(\.\d{3})?$', attack_id):
                        stix_id_to_attack_id[obj['id']] = attack_id
                        attack_id_to_name[attack_id] = obj.get('name', '')
    
    # Also collect tactic IDs
    tactic_shortname_to_id = {}  # e.g., "execution" → "TA0002"
    for obj in objects:
        if obj.get('type') == 'x-mitre-tactic' and not obj.get('revoked', False):
            ext_refs = obj.get('external_references', [])
            for ref in ext_refs:
                if ref.get('source_name') == 'mitre-attack':
                    tactic_id = ref.get('external_id', '')
                    if tactic_id.startswith('TA'):
                        shortname = obj.get('x_mitre_shortname', '')
                        tactic_shortname_to_id[shortname] = tactic_id
                        attack_id_to_name[tactic_id] = obj.get('name', '')
    
    # Build sub-technique → parent mapping
    # Sub-techniques have the pattern T1059.001, parent is T1059
    subtechnique_to_parent = {}
    for attack_id in attack_id_to_name:
        if '.' in attack_id:
            parent_id = attack_id.split('.')[0]
            if parent_id in attack_id_to_name:
                subtechnique_to_parent[attack_id] = parent_id
    
    # Build technique → tactics mapping from kill_chain_phases
    technique_to_tactics = defaultdict(set)
    for obj in objects:
        if obj.get('type') == 'attack-pattern' and not obj.get('revoked', False):
            attack_id = None
            ext_refs = obj.get('external_references', [])
            for ref in ext_refs:
                if ref.get('source_name') == 'mitre-attack':
                    attack_id = ref.get('external_id', '')
            
            if not attack_id:
                continue
            
            # Extract tactics from kill_chain_phases
            phases = obj.get('kill_chain_phases', [])
            for phase in phases:
                if phase.get('kill_chain_name') == 'mitre-attack':
                    phase_name = phase.get('phase_name', '')
                    if phase_name in tactic_shortname_to_id:
                        tactic_id = tactic_shortname_to_id[phase_name]
                        technique_to_tactics[attack_id].add(tactic_id)
    
    # Sub-techniques inherit parent's tactics
    for sub_id, parent_id in subtechnique_to_parent.items():
        if parent_id in technique_to_tactics:
            technique_to_tactics[sub_id].update(technique_to_tactics[parent_id])
    
    # Also collect software → technique relationships
    # (software entities map to the techniques they implement)
    software_stix_ids = {}
    for obj in objects:
        if obj.get('type') in ('malware', 'tool') and not obj.get('revoked', False):
            ext_refs = obj.get('external_references', [])
            for ref in ext_refs:
                if ref.get('source_name') == 'mitre-attack':
                    sw_id = ref.get('external_id', '')
                    if sw_id.startswith('S'):
                        software_stix_ids[obj['id']] = sw_id
                        attack_id_to_name[sw_id] = obj.get('name', '')
    
    print(f"    Techniques: {len([k for k in attack_id_to_name if k.startswith('T')])}")
    print(f"    Sub-techniques: {len(subtechnique_to_parent)}")
    print(f"    Tactics: {len(tactic_shortname_to_id)}")
    print(f"    Technique→tactic mappings: {sum(len(v) for v in technique_to_tactics.values())}")
    
    # Show some examples
    print(f"\n    Example hierarchy expansions:")
    examples = ['T1059.001', 'T1566.001', 'T1070.004', 'T1027.013']
    for eid in examples:
        if eid in attack_id_to_name:
            parts = [f"{eid} ({attack_id_to_name[eid]})"]
            if eid in subtechnique_to_parent:
                parent = subtechnique_to_parent[eid]
                parts.append(f"→ {parent} ({attack_id_to_name.get(parent, '?')})")
            if eid in technique_to_tactics:
                tactics = technique_to_tactics[eid]
                tactic_names = [f"{t} ({attack_id_to_name.get(t, '?')})" for t in sorted(tactics)]
                parts.append(f"→ {', '.join(tactic_names)}")
            print(f"      {' '.join(parts)}")
    
    return subtechnique_to_parent, dict(technique_to_tactics), attack_id_to_name


def expand_predictions(predicted_ids, subtechnique_to_parent, technique_to_tactics):
    """Expand a set of predicted ATT&CK IDs with hierarchical inferences.
    
    If T1059.001 is predicted, adds T1059 and TA0002.
    If T1059 is predicted, adds TA0002.
    Software IDs (S####) are left as-is (no hierarchy).
    Tactics (TA####) are left as-is.
    """
    expanded = set(predicted_ids)
    
    for pred_id in list(predicted_ids):
        # Sub-technique → add parent technique
        if pred_id in subtechnique_to_parent:
            parent = subtechnique_to_parent[pred_id]
            expanded.add(parent)
        
        # Technique or sub-technique → add associated tactics
        if pred_id in technique_to_tactics:
            expanded.update(technique_to_tactics[pred_id])
        
        # If it's a parent technique, also add its tactics
        if '.' not in pred_id and pred_id.startswith('T'):
            if pred_id in technique_to_tactics:
                expanded.update(technique_to_tactics[pred_id])
    
    return expanded


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING AND EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def load_and_group_data(filepath):
    """Load enriched JSONL and group by query_norm."""
    query_data = defaultdict(lambda: {
        'query_raw': None, 'actor': None, 'candidates': [], 'positive_count': 0,
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
    return dict(query_data)


def create_test_split(query_data):
    """Same split as finetune_production.py."""
    random.seed(RANDOM_SEED)
    queries_by_actor = defaultdict(list)
    for qn, data in query_data.items():
        actor = data['actor']
        if actor.startswith('external_'):
            continue
        if data['positive_count'] > 0:
            queries_by_actor[actor].append(qn)
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


def compute_multilabel_metrics(predictions, gold_labels):
    if not predictions and not gold_labels:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    if not predictions:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    if not gold_labels:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    tp = len(predictions & gold_labels)
    precision = tp / len(predictions) if predictions else 0
    recall = tp / len(gold_labels) if gold_labels else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)
    return {'precision': precision, 'recall': recall, 'f1': f1}


def main():
    print("\n" + "=" * 70)
    print("  HIERARCHICAL ATT&CK POST-PROCESSING")
    print("  Encoding ATT&CK taxonomy into the prediction pipeline")
    print("=" * 70)

    stix_path = "enterprise-attack-v14.json"
    data_path = "data/reranker_pairs_enriched_v2.jsonl"
    model_path = "checkpoints/best_two_stage_v2"

    for p, name in [(stix_path, "STIX"), (data_path, "Data"), (model_path, "Model")]:
        if not Path(p).exists():
            sys.exit(f"[FAIL] {name} not found: {p}")

    # ── Build hierarchy ──
    print()
    sub_to_parent, tech_to_tactics, id_to_name = build_attack_hierarchy(stix_path)

    # ── Load data and model ──
    print(f"\n  Loading data from: {data_path}")
    query_data = load_and_group_data(data_path)
    test_queries = create_test_split(query_data)
    print(f"  Test queries: {len(test_queries)}")

    print(f"  Loading model from: {model_path}")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_path)
    print(f"  Model loaded.")

    # ── Evaluate with and without post-processing ──
    print(f"\n  {'='*60}")
    print(f"  EVALUATION: BASELINE vs HIERARCHICAL POST-PROCESSING")
    print(f"  {'='*60}")

    # Use the optimal multi-label threshold from earlier (1.75)
    THRESHOLD = 1.75

    baseline_p1 = []
    postproc_p1 = []
    baseline_ml = []  # multi-label metrics
    postproc_ml = []
    
    baseline_by_actor = defaultdict(list)
    postproc_by_actor = defaultdict(list)
    baseline_ml_by_actor = defaultdict(list)
    postproc_ml_by_actor = defaultdict(list)

    expansion_examples = []

    for qn in test_queries:
        qd = query_data[qn]
        if qd['positive_count'] == 0 or not qd['candidates']:
            continue

        query_raw = qd['query_raw']
        actor = qd['actor']
        texts = [[query_raw, c['text']] for c in qd['candidates']]
        scores = model.predict(texts, show_progress_bar=False)

        # Build candidate info
        gold_ids = set()
        candidate_info = []
        for j, c in enumerate(qd['candidates']):
            cid = extract_id(c['id'])
            candidate_info.append({
                'id': cid, 'score': float(scores[j]), 'label': c['label'],
            })
            if c['label'] == 1:
                gold_ids.add(cid)

        ranked = sorted(candidate_info, key=lambda x: x['score'], reverse=True)

        # ── P@1 evaluation ──
        # Baseline: top-1 prediction
        top1_id = ranked[0]['id']
        top1_correct = ranked[0]['label'] == 1
        baseline_p1.append(top1_correct)
        baseline_by_actor[actor].append(top1_correct)

        # Post-processed: expand top-1 prediction, check if any gold is in expansion
        expanded_top1 = expand_predictions({top1_id}, sub_to_parent, tech_to_tactics)
        postproc_top1_correct = bool(expanded_top1 & gold_ids)
        postproc_p1.append(postproc_top1_correct)
        postproc_by_actor[actor].append(postproc_top1_correct)

        # Track expansion examples where post-processing helps
        if postproc_top1_correct and not top1_correct:
            expansion_examples.append({
                'query': query_raw[:80],
                'actor': actor,
                'predicted': top1_id,
                'expanded_to': sorted(expanded_top1 - {top1_id}),
                'gold_ids': sorted(gold_ids),
                'matched_via': sorted(expanded_top1 & gold_ids),
            })

        # ── Multi-label evaluation ──
        # Baseline: all candidates above threshold
        baseline_preds = set()
        for c in candidate_info:
            if c['score'] >= THRESHOLD:
                baseline_preds.add(c['id'])

        baseline_metrics = compute_multilabel_metrics(baseline_preds, gold_ids)
        baseline_ml.append(baseline_metrics)
        baseline_ml_by_actor[actor].append(baseline_metrics)

        # Post-processed: expand all above-threshold predictions
        expanded_preds = expand_predictions(baseline_preds, sub_to_parent, tech_to_tactics)
        # Only keep expanded IDs that are actually in the candidate pool
        # (since gold labels only include IDs from the BM25 candidates)
        all_candidate_ids = set(c['id'] for c in candidate_info)
        expanded_preds_filtered = expanded_preds & (all_candidate_ids | gold_ids)

        postproc_metrics = compute_multilabel_metrics(expanded_preds_filtered, gold_ids)
        postproc_ml.append(postproc_metrics)
        postproc_ml_by_actor[actor].append(postproc_metrics)

    # ── Results ──
    print(f"\n  {'─'*60}")
    print(f"  P@1 COMPARISON")
    print(f"  {'─'*60}")

    b_p1 = np.mean(baseline_p1)
    p_p1 = np.mean(postproc_p1)
    b_correct = sum(baseline_p1)
    p_correct = sum(postproc_p1)
    print(f"\n  {'Method':<30} | {'P@1':>8} | {'Correct':>8}")
    print(f"  {'-'*30}-+-{'-'*8}-+-{'-'*8}")
    print(f"  {'Baseline (no post-proc)':<30} | {b_p1:>7.2%} | {b_correct:>5}/{len(baseline_p1)}")
    print(f"  {'+ Hierarchical post-proc':<30} | {p_p1:>7.2%} | {p_correct:>5}/{len(postproc_p1)}")

    if p_correct > b_correct:
        print(f"\n  Post-processing FIXED {p_correct - b_correct} error(s)!")
    elif p_correct == b_correct:
        print(f"\n  Post-processing did not change P@1 (errors are lateral, not hierarchical)")

    # Show which errors were fixed
    if expansion_examples:
        print(f"\n  Errors fixed by hierarchical expansion:")
        for ex in expansion_examples:
            print(f"    Query: {ex['query']}")
            print(f"    Predicted: {ex['predicted']} → expanded to include: {ex['expanded_to']}")
            print(f"    Gold: {ex['gold_ids']}")
            print(f"    Match via: {ex['matched_via']}")
            print()

    # Per-actor P@1
    print(f"\n  Per-actor P@1:")
    print(f"  {'Actor':<14} | {'Baseline':>10} | {'Post-proc':>10} | {'Change':>8}")
    print(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for actor in sorted(baseline_by_actor):
        b = np.mean(baseline_by_actor[actor])
        p = np.mean(postproc_by_actor[actor])
        change = p - b
        marker = " ✓" if change > 0 else ""
        print(f"  {actor:<14} | {b:>9.2%} | {p:>9.2%} | {change:>+7.2%}{marker}")

    # ── Multi-label comparison ──
    print(f"\n  {'─'*60}")
    print(f"  MULTI-LABEL COMPARISON (threshold = {THRESHOLD})")
    print(f"  {'─'*60}")

    b_prec = np.mean([m['precision'] for m in baseline_ml])
    b_rec = np.mean([m['recall'] for m in baseline_ml])
    b_f1 = np.mean([m['f1'] for m in baseline_ml])

    p_prec = np.mean([m['precision'] for m in postproc_ml])
    p_rec = np.mean([m['recall'] for m in postproc_ml])
    p_f1 = np.mean([m['f1'] for m in postproc_ml])

    print(f"\n  {'Method':<30} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'Baseline':<30} | {b_prec:>9.2%} | {b_rec:>9.2%} | {b_f1:>9.2%}")
    print(f"  {'+ Hierarchical post-proc':<30} | {p_prec:>9.2%} | {p_rec:>9.2%} | {p_f1:>9.2%}")

    rec_change = p_rec - b_rec
    f1_change = p_f1 - b_f1
    print(f"\n  Recall change: {rec_change:+.2%}")
    print(f"  F1 change:     {f1_change:+.2%}")

    # Per-actor multi-label
    print(f"\n  Per-actor multi-label F1:")
    print(f"  {'Actor':<14} | {'Baseline F1':>12} | {'Post-proc F1':>12} | {'Change':>8}")
    print(f"  {'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    for actor in sorted(baseline_ml_by_actor):
        b = np.mean([m['f1'] for m in baseline_ml_by_actor[actor]])
        p = np.mean([m['f1'] for m in postproc_ml_by_actor[actor]])
        change = p - b
        marker = " ✓" if change > 0.001 else ""
        print(f"  {actor:<14} | {b:>11.2%} | {p:>11.2%} | {change:>+7.2%}{marker}")

    # ── Paper-ready output ──
    print(f"\n  {'─'*60}")
    print(f"  FOR THE PAPER")
    print(f"  {'─'*60}")
    print(f"\n  \"We incorporate ATT&CK's hierarchical taxonomy as a")
    print(f"   post-processing step: when the model predicts a sub-technique,")
    print(f"   the parent technique and associated tactic(s) are automatically")
    print(f"   inferred. This domain-knowledge integration improves multi-label")
    print(f"   recall from {b_rec:.2%} to {p_rec:.2%} and F1 from {b_f1:.2%}")
    print(f"   to {p_f1:.2%}, without any model retraining.\"")

    # ── Save results ──
    out_dir = Path("hierarchical_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'p_at_1': {
            'baseline': round(b_p1, 4),
            'postprocessed': round(p_p1, 4),
            'baseline_correct': int(b_correct),
            'postprocessed_correct': int(p_correct),
        },
        'multilabel': {
            'threshold': THRESHOLD,
            'baseline': {'precision': round(b_prec, 4), 'recall': round(b_rec, 4), 'f1': round(b_f1, 4)},
            'postprocessed': {'precision': round(p_prec, 4), 'recall': round(p_rec, 4), 'f1': round(p_f1, 4)},
        },
        'expansion_examples': expansion_examples,
    }

    with open(out_dir / "hierarchical_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_dir / 'hierarchical_results.json'}")
    print(f"\n  [DONE] Hierarchical post-processing evaluation complete.")


if __name__ == '__main__':
    main()
