#!/usr/bin/env python3
"""
Third Benchmark: MITRE ATT&CK Procedure Examples (Zero-Shot)
==============================================================
Evaluates the reranker on MITRE's own procedure examples — real-world
descriptions of how malware and threat groups use specific ATT&CK
techniques. These passages are completely independent from both the
CTI-HAL training data and the tumeteor generalization benchmark.

The STIX data contains 13,444 "uses" relationships with descriptions
like: "Explosive has collected the MAC address from the victim's machine."
Each is linked to a specific technique (e.g., T1082 System Information
Discovery). We strip the markdown, use the text as a query, score all
technique descriptions with the cross-encoder, and check if the correct
technique ranks #1.

This is the strongest possible zero-shot generalization claim:
  - Data source: MITRE (the creators of ATT&CK)
  - Annotation quality: Expert-curated by MITRE staff
  - Independence: No overlap with CTI-HAL or tumeteor training data
  - Scale: 500 sampled from 13,444 available examples

Usage:
  python mitre_procedure_benchmark.py

  Expects:
    enterprise-attack-v14.json
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
ATTACK_ID_RE = re.compile(r'(T\d{4}(?:\.\d{3})?)')
SAMPLE_SIZE = 500  # Number of procedure examples to evaluate


def clean_stix_text(text):
    """Strip markdown links, citations, and noise from STIX text."""
    if not text:
        return ''
    # Convert markdown links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove citations
    text = re.sub(r'\(Citation:\s*[^)]*\)', '', text)
    # Remove bare URLs
    text = re.sub(r'https?://\S+', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_technique_corpus(stix_objects):
    """Build technique ID -> description mapping from STIX data."""
    corpus = {}
    for obj in stix_objects:
        if obj.get('type') != 'attack-pattern':
            continue
        if obj.get('revoked', False) or obj.get('x_mitre_deprecated', False):
            continue

        attack_id = None
        for ref in obj.get('external_references', []):
            if ref.get('source_name') == 'mitre-attack':
                eid = ref.get('external_id', '')
                if re.match(r'^T\d{4}(\.\d{3})?$', eid):
                    attack_id = eid

        if not attack_id:
            continue

        name = obj.get('name', '')
        desc = clean_stix_text(obj.get('description', ''))

        # Build enriched text matching training format
        if desc:
            enriched = f"{attack_id} — {name}: {desc}"
        else:
            enriched = f"{attack_id} — {name}"

        corpus[attack_id] = {
            'name': name,
            'description': desc,
            'enriched_text': enriched,
            'stix_id': obj['id'],
        }

    return corpus


def extract_procedure_examples(stix_objects, technique_corpus):
    """Extract procedure examples from STIX 'uses' relationships.
    
    Each 'uses' relationship links a source (malware/group) to a
    target (technique) with a description of how it's used.
    """
    # Build STIX ID -> ATT&CK ID mapping for techniques
    stix_to_attack = {}
    for attack_id, info in technique_corpus.items():
        stix_to_attack[info['stix_id']] = attack_id

    # Build STIX ID -> name mapping for sources (malware/groups)
    source_names = {}
    for obj in stix_objects:
        if obj.get('type') in ('malware', 'tool', 'intrusion-set'):
            source_names[obj['id']] = obj.get('name', 'Unknown')

    examples = []
    for obj in stix_objects:
        if obj.get('type') != 'relationship':
            continue
        if obj.get('relationship_type') != 'uses':
            continue
        if 'description' not in obj:
            continue

        target_ref = obj.get('target_ref', '')
        source_ref = obj.get('source_ref', '')

        # Only keep relationships targeting techniques
        if target_ref not in stix_to_attack:
            continue

        technique_id = stix_to_attack[target_ref]
        description = clean_stix_text(obj['description'])
        source_name = source_names.get(source_ref, 'Unknown')

        # Skip very short descriptions (< 20 chars)
        if len(description) < 20:
            continue

        # Truncate very long descriptions to match typical CTI query length
        if len(description) > 500:
            # Find a sentence boundary near 500 chars
            truncated = description[:500]
            last_period = truncated.rfind('.')
            if last_period > 300:
                description = truncated[:last_period + 1]
            else:
                description = truncated + '...'

        examples.append({
            'query': description,
            'gold_technique': technique_id,
            'source_name': source_name,
        })

    return examples


def main():
    print("\n" + "=" * 70)
    print("  THIRD BENCHMARK: MITRE ATT&CK PROCEDURE EXAMPLES (ZERO-SHOT)")
    print("  Completely independent evaluation on MITRE's own data")
    print("=" * 70)

    stix_path = "enterprise-attack-v14.json"
    model_path = "checkpoints/best_two_stage_v2"

    for p, name in [(stix_path, "STIX"), (model_path, "Model")]:
        if not Path(p).exists():
            sys.exit(f"[FAIL] {name} not found: {p}")

    # ── Parse STIX data ──
    print(f"\n  Loading STIX data from: {stix_path}")
    with open(stix_path, 'r', encoding='utf-8') as f:
        stix = json.load(f)
    objects = stix['objects']
    print(f"  Total STIX objects: {len(objects)}")

    # Build technique corpus
    technique_corpus = build_technique_corpus(objects)
    print(f"  Techniques in corpus: {len(technique_corpus)}")

    # Extract procedure examples
    all_examples = extract_procedure_examples(objects, technique_corpus)
    print(f"  Procedure examples extracted: {len(all_examples)}")

    # Filter to examples whose gold technique is in our corpus
    valid_examples = [ex for ex in all_examples
                      if ex['gold_technique'] in technique_corpus]
    print(f"  Valid examples (gold in corpus): {len(valid_examples)}")

    # Sample for evaluation
    random.seed(RANDOM_SEED)
    if len(valid_examples) > SAMPLE_SIZE:
        sampled = random.sample(valid_examples, SAMPLE_SIZE)
    else:
        sampled = valid_examples
    print(f"  Sampled for evaluation: {len(sampled)}")

    # Check technique coverage in sample
    gold_techniques = set(ex['gold_technique'] for ex in sampled)
    print(f"  Unique gold techniques in sample: {len(gold_techniques)}")

    # ── Build candidate list ──
    # Score each query against ALL techniques (not just top-20)
    # This is the fairest possible evaluation — no first-stage filter
    technique_ids = sorted(technique_corpus.keys())
    technique_texts = [technique_corpus[tid]['enriched_text'] for tid in technique_ids]
    id_to_idx = {tid: idx for idx, tid in enumerate(technique_ids)}

    print(f"\n  Candidates per query: {len(technique_ids)} techniques")
    print(f"  Total pairs to score: {len(sampled) * len(technique_ids):,}")
    print(f"  Estimated time: {len(sampled) * len(technique_ids) * 0.003 / 60:.0f} min on CPU")

    # ── Load model ──
    print(f"\n  Loading model from: {model_path}")
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_path)
    print(f"  Model loaded.")

    # ── Evaluate ──
    print(f"\n  {'='*60}")
    print(f"  RUNNING ZERO-SHOT EVALUATION")
    print(f"  {'='*60}")

    correct_at_1 = 0
    correct_at_3 = 0
    correct_at_5 = 0
    total = 0
    start_time = time.time()

    for i, ex in enumerate(sampled):
        query = ex['query']
        gold = ex['gold_technique']

        # Score query against all techniques
        pairs = [[query, t] for t in technique_texts]
        scores = model.predict(pairs, show_progress_bar=False)

        # Rank by score
        ranked_indices = np.argsort(scores)[::-1]
        ranked_ids = [technique_ids[idx] for idx in ranked_indices]

        # Check P@1, Hit@3, Hit@5
        total += 1
        if ranked_ids[0] == gold:
            correct_at_1 += 1
        if gold in ranked_ids[:3]:
            correct_at_3 += 1
        if gold in ranked_ids[:5]:
            correct_at_5 += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(sampled) - i - 1) / rate
            current_p1 = correct_at_1 / total
            print(f"    [{i+1}/{len(sampled)}] {elapsed:.0f}s elapsed, "
                  f"~{remaining:.0f}s remaining, "
                  f"running P@1: {current_p1:.2%}")

    elapsed = time.time() - start_time

    # ── Results ──
    p1 = correct_at_1 / total
    h3 = correct_at_3 / total
    h5 = correct_at_5 / total

    print(f"\n  {'='*60}")
    print(f"  RESULTS: MITRE PROCEDURE EXAMPLES (ZERO-SHOT)")
    print(f"  {'='*60}")

    print(f"\n  P@1:   {p1:.4f} ({correct_at_1}/{total})")
    print(f"  Hit@3: {h3:.4f} ({correct_at_3}/{total})")
    print(f"  Hit@5: {h5:.4f} ({correct_at_5}/{total})")
    print(f"  Time:  {elapsed:.0f}s ({elapsed/60:.1f} min)")

    print(f"\n  CROSS-BENCHMARK COMPARISON:")
    print(f"  {'Benchmark':<40} | {'P@1':>8} | {'Queries':>8} | {'Source':>15}")
    print(f"  {'-'*40}-+-{'-'*8}-+-{'-'*8}-+-{'-'*15}")
    print(f"  {'CTI-HAL (primary, with training)':<40} | {'94.52%':>8} | {'146':>8} | {'Expert CTI':>15}")
    print(f"  {'Tumeteor (generalization, 2-stage)':<40} | {'93.85%':>8} | {'20,604':>8} | {'Derived CTI':>15}")
    print(f"  {'MITRE Procedures (zero-shot)':<40} | {f'{p1:.2%}':>8} | {f'{total}':>8} | {'MITRE':>15}")

    print(f"\n  Note: MITRE procedure benchmark scores ALL {len(technique_ids)} techniques")
    print(f"  per query (no first-stage filter, no gold injection).")
    print(f"  This is the hardest possible evaluation setting.")

    # ── Save results ──
    out_dir = Path("mitre_procedure_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'benchmark': 'MITRE ATT&CK Procedure Examples',
        'source': 'enterprise-attack-v14.json',
        'total_available': len(valid_examples),
        'sampled': len(sampled),
        'unique_gold_techniques': len(gold_techniques),
        'candidate_pool_size': len(technique_ids),
        'p_at_1': round(p1, 4),
        'hit_at_3': round(h3, 4),
        'hit_at_5': round(h5, 4),
        'evaluation_time_s': round(elapsed, 1),
        'model': model_path,
        'note': 'Zero-shot: model never trained on procedure examples. '
                'All techniques scored per query (no first-stage filter).',
    }

    with open(out_dir / "procedure_benchmark.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {out_dir / 'procedure_benchmark.json'}")
    print(f"\n  [DONE] MITRE procedure benchmark complete.")


if __name__ == '__main__':
    main()
