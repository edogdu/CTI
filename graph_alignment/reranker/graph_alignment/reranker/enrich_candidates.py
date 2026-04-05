#!/usr/bin/env python3
"""
Enrich Candidate Text with Full ATT&CK Descriptions
=====================================================
Replaces short candidate labels like "T1203 — EXPLOITATION FOR CLIENT EXECUTION"
with rich descriptions from the MITRE ATT&CK STIX data, giving the cross-encoder
real semantic content to match against CTI passages.

Before: candidate_text = "T1203 — EXPLOITATION FOR CLIENT EXECUTION"
After:  candidate_text = "T1203 — Exploitation for Client Execution:
        Adversaries may exploit software vulnerabilities in client
        applications to execute code. Vulnerabilities can exist in
        software due to insecure coding practices..."

This is the single highest-impact system improvement identified:
the cross-encoder currently achieves 91.1% P@1 matching rich CTI prose
against terse 69-character labels. Giving it full descriptions should
significantly improve accuracy.

Usage:
  python enrich_candidates.py

  Expects:
    enterprise-attack-v14.json              (MITRE ATT&CK STIX bundle)
    data/reranker_pairs_enriched.jsonl      (existing training pairs)

  Produces:
    data/reranker_pairs_enriched_v2.jsonl   (enriched training pairs)

Author: Shane Waldrop — Angelo State University / ARL Grant W911NF-24-2-0180
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# Max characters for description text to keep cross-encoder input under 512 tokens.
# ~300 tokens ≈ ~1200 characters. Leave room for query (up to ~200 chars) + ID/name.
MAX_DESC_CHARS = 1000


def extract_attack_id(external_references):
    """Extract ATT&CK ID (T1234, T1234.001, TA0001, S0001) from STIX external_references."""
    if not external_references:
        return None
    for ref in external_references:
        if ref.get('source_name') == 'mitre-attack':
            eid = ref.get('external_id', '')
            if re.match(r'^(T\d{4}(\.\d{3})?|TA\d{4}|S\d{4})$', eid):
                return eid
    return None


def clean_description(desc):
    """Clean STIX description text for use as cross-encoder input.
    Removes citation markers like (Citation: Name) and excess whitespace."""
    if not desc:
        return ''
    # Remove citation markers: (Citation: Whatever Text)
    desc = re.sub(r'\(Citation:\s*[^)]*\)', '', desc)
    # Collapse whitespace
    desc = re.sub(r'\s+', ' ', desc).strip()
    # Truncate to max length, breaking at word boundary
    if len(desc) > MAX_DESC_CHARS:
        truncated = desc[:MAX_DESC_CHARS]
        # Break at last space to avoid cutting mid-word
        last_space = truncated.rfind(' ')
        if last_space > MAX_DESC_CHARS * 0.8:  # Don't cut too aggressively
            truncated = truncated[:last_space]
        desc = truncated + '...'
    return desc


def build_attack_lookup(stix_path):
    """Parse MITRE ATT&CK STIX bundle and build ID → (name, description) mapping.
    
    Covers three entity types:
      - attack-pattern → Techniques and sub-techniques (T-numbers)
      - x-mitre-tactic → Tactics (TA-numbers)  
      - malware + tool → Software (S-numbers)
    """
    print(f"  Loading STIX data from: {stix_path}")
    with open(stix_path, 'r', encoding='utf-8') as f:
        stix = json.load(f)

    objects = stix.get('objects', [])
    print(f"  Total STIX objects: {len(objects)}")

    lookup = {}  # ATT&CK ID → {'name': ..., 'description': ...}

    # Track counts by type for reporting
    counts = defaultdict(int)

    for obj in objects:
        obj_type = obj.get('type', '')

        # Skip revoked or deprecated entries
        if obj.get('revoked', False) or obj.get('x_mitre_deprecated', False):
            continue

        attack_id = None
        name = obj.get('name', '')
        description = obj.get('description', '')

        if obj_type == 'attack-pattern':
            # Techniques and sub-techniques
            attack_id = extract_attack_id(obj.get('external_references', []))
            if attack_id:
                counts['technique'] += 1

        elif obj_type == 'x-mitre-tactic':
            # Tactics
            attack_id = extract_attack_id(obj.get('external_references', []))
            if attack_id:
                counts['tactic'] += 1

        elif obj_type in ('malware', 'tool'):
            # Software entries
            attack_id = extract_attack_id(obj.get('external_references', []))
            if attack_id:
                counts['software'] += 1

        if attack_id and name:
            cleaned_desc = clean_description(description)
            lookup[attack_id] = {
                'name': name,
                'description': cleaned_desc,
                'type': obj_type,
            }

    print(f"  Built lookup with {len(lookup)} ATT&CK entities:")
    for entity_type, count in sorted(counts.items()):
        print(f"    {entity_type}: {count}")

    # Show a few examples for sanity check
    print(f"\n  Sample entries:")
    examples = ['T1059', 'T1059.001', 'T1566.001', 'TA0002', 'S0051']
    for eid in examples:
        if eid in lookup:
            entry = lookup[eid]
            desc_preview = entry['description'][:100] + '...' if len(entry['description']) > 100 else entry['description']
            print(f"    {eid} — {entry['name']}")
            print(f"      {desc_preview}")
        else:
            print(f"    {eid} — NOT FOUND in STIX data")

    return lookup


def enrich_jsonl(input_path, output_path, lookup):
    """Read existing training pairs JSONL and enrich candidate_text with
    full ATT&CK descriptions from the lookup table."""
    
    print(f"\n  Reading: {input_path}")
    print(f"  Writing: {output_path}")

    total = 0
    enriched = 0
    not_found = 0
    not_found_ids = set()
    unchanged = 0

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line)
                continue

            candidate_id = row.get('candidate_id', '')

            # Try to match the candidate_id against the lookup
            # candidate_id might be "T1059.001" or "S0051" directly
            match = re.match(r'^(T\d{4}(?:\.\d{3})?|TA\d{4}|S\d{4})', candidate_id)

            if match:
                attack_id = match.group(1)
                if attack_id in lookup:
                    entry = lookup[attack_id]
                    # Build enriched candidate_text:
                    # "T1203 — Exploitation for Client Execution: Adversaries may..."
                    if entry['description']:
                        new_text = f"{attack_id} — {entry['name']}: {entry['description']}"
                    else:
                        # Some entries might not have descriptions (rare)
                        new_text = f"{attack_id} — {entry['name']}"

                    row['candidate_text'] = new_text
                    # Preserve original for reference
                    row['candidate_text_original'] = row.get('candidate_text', '')
                    enriched += 1
                else:
                    not_found += 1
                    not_found_ids.add(attack_id)
            else:
                unchanged += 1

            fout.write(json.dumps(row, ensure_ascii=False) + '\n')

            if total % 5000 == 0:
                print(f"    Processed {total:,} rows...")

    print(f"\n  Results:")
    print(f"    Total rows:      {total:,}")
    print(f"    Enriched:        {enriched:,} ({enriched/total*100:.1f}%)")
    print(f"    Not in lookup:   {not_found:,}")
    print(f"    No ID matched:   {unchanged:,}")

    if not_found_ids:
        sample = sorted(not_found_ids)[:20]
        print(f"    IDs not found (sample): {', '.join(sample)}")

    # Show a sample enriched row
    print(f"\n  Sample enriched row:")
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if len(row.get('candidate_text', '')) > 100:
                print(f"    query_raw:      {row['query_raw'][:80]}...")
                print(f"    candidate_id:   {row['candidate_id']}")
                print(f"    candidate_text: {row['candidate_text'][:150]}...")
                print(f"    label:          {row['label']}")
                break

    return total, enriched


def main():
    print("\n" + "=" * 70)
    print("  CANDIDATE TEXT ENRICHMENT")
    print("  Adding full ATT&CK descriptions to training pairs")
    print("=" * 70)

    stix_path = Path("enterprise-attack-v14.json")
    input_path = Path("data/reranker_pairs_enriched.jsonl")
    output_path = Path("data/reranker_pairs_enriched_v2.jsonl")

    if not stix_path.exists():
        sys.exit(f"[FAIL] STIX file not found: {stix_path}\n"
                 f"  Download from: https://github.com/mitre-attack/attack-stix-data")
    if not input_path.exists():
        sys.exit(f"[FAIL] Training data not found: {input_path}")

    # Step 1: Build ATT&CK lookup from STIX data
    print()
    lookup = build_attack_lookup(str(stix_path))

    # Step 2: Enrich the training pairs
    total, enriched = enrich_jsonl(str(input_path), str(output_path), lookup)

    # Step 3: Compare file sizes
    orig_size = input_path.stat().st_size
    new_size = output_path.stat().st_size
    print(f"\n  File sizes:")
    print(f"    Original: {orig_size:,} bytes ({orig_size/1024/1024:.1f} MB)")
    print(f"    Enriched: {new_size:,} bytes ({new_size/1024/1024:.1f} MB)")
    print(f"    Growth:   {(new_size-orig_size)/orig_size*100:.0f}%")

    # Also enrich tumeteor data if it exists
    tumeteor_input = Path("data/reranker_pairs_enriched_tumeteor.jsonl")
    tumeteor_output = Path("data/reranker_pairs_enriched_tumeteor_v2.jsonl")
    if tumeteor_input.exists():
        print(f"\n  Also enriching tumeteor data...")
        enrich_jsonl(str(tumeteor_input), str(tumeteor_output), lookup)

    print(f"\n  NEXT STEPS:")
    print(f"  1. Retrain the model using the enriched data:")
    print(f"     python finetune_production.py --data data/reranker_pairs_enriched_v2.jsonl")
    print(f"  2. Evaluate the retrained model on CTI-HAL test set")
    print(f"  3. Compare P@1 against the original 91.1% baseline")
    print(f"\n  [DONE] Enrichment complete.")


if __name__ == '__main__':
    main()
