#!/usr/bin/env python3
"""
Clean Markdown from Enriched Candidate Text
=============================================
The MITRE ATT&CK STIX descriptions contain markdown link formatting
like [MiniDuke](https://attack.mitre.org/software/S0051) and raw URLs.
These waste tokens in the cross-encoder's 512-token window without
adding semantic value.

This script reads the enriched JSONL and strips:
  - Markdown links: [text](url) → text
  - Raw ATT&CK URLs
  - Residual citation markers missed by the first pass
  - Double spaces and other whitespace artifacts

Usage:
  python clean_markdown.py

  Reads:  data/reranker_pairs_enriched_v2.jsonl
  Writes: data/reranker_pairs_enriched_v3.jsonl

  Also cleans tumeteor if present:
  Reads:  data/reranker_pairs_enriched_tumeteor_v2.jsonl
  Writes: data/reranker_pairs_enriched_tumeteor_v3.jsonl

Author: Shane Waldrop — Angelo State University / ARL Grant W911NF-24-2-0180
"""

import json
import re
import sys
from pathlib import Path


def clean_markdown(text):
    """Strip markdown formatting and URLs from ATT&CK description text.
    
    Before: "[MiniDuke](https://attack.mitre.org/software/S0051) is malware 
             that was used by [APT29](https://attack.mitre.org/groups/G0016)"
    After:  "MiniDuke is malware that was used by APT29"
    """
    if not text:
        return text

    # 1. Convert markdown links [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # 2. Remove any remaining bare URLs
    text = re.sub(r'https?://\S+', '', text)

    # 3. Remove residual citation markers: (Citation: Whatever)
    text = re.sub(r'\(Citation:\s*[^)]*\)', '', text)

    # 4. Remove empty parentheses left behind
    text = re.sub(r'\(\s*\)', '', text)

    # 5. Collapse multiple spaces into one
    text = re.sub(r'  +', ' ', text)

    # 6. Clean up spaces before punctuation
    text = re.sub(r'\s+([.,;:])', r'\1', text)

    # 7. Strip leading/trailing whitespace
    text = text.strip()

    return text


def process_file(input_path, output_path):
    """Clean markdown from all candidate_text fields in a JSONL file."""
    print(f"\n  Reading:  {input_path}")
    print(f"  Writing:  {output_path}")

    total = 0
    modified = 0
    chars_saved = 0

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line)
                continue

            ct = row.get('candidate_text', '')
            if ct:
                cleaned = clean_markdown(ct)
                if cleaned != ct:
                    chars_saved += len(ct) - len(cleaned)
                    modified += 1
                    row['candidate_text'] = cleaned

            fout.write(json.dumps(row, ensure_ascii=False) + '\n')

            if total % 10000 == 0:
                print(f"    Processed {total:,} rows...")

    print(f"\n  Results:")
    print(f"    Total rows:     {total:,}")
    print(f"    Rows modified:  {modified:,} ({modified/total*100:.1f}%)")
    print(f"    Characters saved: {chars_saved:,} ({chars_saved/1024:.0f} KB)")

    # Show before/after example
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            ct = row.get('candidate_text', '')
            if 'https' not in ct and len(ct) > 100 and '—' in ct:
                print(f"\n  Sample cleaned text:")
                print(f"    {ct[:150]}...")
                break

    # File size comparison
    orig_size = Path(input_path).stat().st_size
    new_size = Path(output_path).stat().st_size
    print(f"\n  File size: {orig_size/1024/1024:.1f} MB → {new_size/1024/1024:.1f} MB "
          f"(saved {(orig_size-new_size)/1024:.0f} KB)")


def main():
    print("\n" + "=" * 70)
    print("  CLEAN MARKDOWN FROM ENRICHED DESCRIPTIONS")
    print("  Removing [text](url) links, raw URLs, and noise tokens")
    print("=" * 70)

    # Clean CTI-HAL data
    ctihal_in = Path("data/reranker_pairs_enriched_v2.jsonl")
    ctihal_out = Path("data/reranker_pairs_enriched_v3.jsonl")

    if not ctihal_in.exists():
        sys.exit(f"[FAIL] Input not found: {ctihal_in}")

    process_file(str(ctihal_in), str(ctihal_out))

    # Clean tumeteor data if present
    tumeteor_in = Path("data/reranker_pairs_enriched_tumeteor_v2.jsonl")
    tumeteor_out = Path("data/reranker_pairs_enriched_tumeteor_v3.jsonl")

    if tumeteor_in.exists():
        print(f"\n  Also cleaning tumeteor data...")
        process_file(str(tumeteor_in), str(tumeteor_out))
    else:
        print(f"\n  Tumeteor file not found, skipping: {tumeteor_in}")

    # Quick sanity check — verify no markdown links remain
    print(f"\n  Sanity check — scanning for residual markdown links...")
    residual = 0
    with open(str(ctihal_out), 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            ct = row.get('candidate_text', '')
            if re.search(r'\[.*\]\(http', ct):
                residual += 1
    print(f"  Residual markdown links found: {residual}")
    if residual == 0:
        print(f"  ✓ All markdown links successfully cleaned")

    print(f"\n  NEXT STEPS:")
    print(f"  1. Upload cleaned files to Google Drive:")
    print(f"     {ctihal_out}")
    if tumeteor_out.exists():
        print(f"     {tumeteor_out}")
    print(f"  2. Retrain two-stage model on Colab with cleaned data")
    print(f"  3. Compare P@1 against current 94.52% baseline")

    print(f"\n  [DONE] Markdown cleaning complete.")


if __name__ == '__main__':
    main()
