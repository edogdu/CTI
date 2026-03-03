#!/usr/bin/env python3
"""
zenodo_to_reranker.py — Dataset Expansion Phase 2
===================================================
Converts the Zenodo MITRE ATT&CK TTP Dataset (v15.1) into reranker_pairs
format compatible with finetune_production.py.

Source: https://zenodo.org/records/14907305
  - 19,747 procedure examples, 780 unique TTP IDs
  - Columns: Key (TTP ID), Value (procedure description)
  - Extracted from enterprise-attack-15.1.json by Hamzic et al.
  - CC-BY-4.0 license

This script:
  1. Downloads the CSV from Zenodo (or uses a local copy)
  2. Reuses ATT&CK v14 STIX vocabulary (cached from tumeteor script)
  3. Filters v15.1 IDs to v14 vocabulary
  4. Converts to reranker pairs with random negatives
  5. Outputs JSONL in the exact format finetune_production.py expects

Usage:
    python zenodo_to_reranker.py
    python zenodo_to_reranker.py --negatives 10
    python zenodo_to_reranker.py --csv MITRE_ATTACKv15-1_TTP_Dataset.csv

Branch: dataset-expansion
Author: Shane Waldrop
"""

import json
import os
import sys
import csv
import random
import argparse
from collections import defaultdict
from urllib.request import urlopen, Request

# ── Constants ────────────────────────────────────────────────────────────
RANDOM_SEED = 43          # Different seed than tumeteor for negative diversity
EM_DASH = "\u2014"
ACTOR_NAME = "external_zenodo"

ZENODO_CSV_URL = (
    "https://zenodo.org/records/14907305/files/"
    "MITRE_ATTACKv15-1_TTP_Dataset.csv?download=1"
)

# ATT&CK v14.1 STIX bundle (same as tumeteor script)
STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/"
    "ATT%26CK-v14.1/enterprise-attack/enterprise-attack.json"
)


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Build ATT&CK v14 Vocabulary (reused from tumeteor script)
# ═══════════════════════════════════════════════════════════════════════

def download_stix(url=STIX_URL, cache_path=None):
    """Download ATT&CK STIX bundle, with optional local cache."""
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached STIX from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"  Downloading ATT&CK v14.1 STIX bundle from MITRE GitHub...")
    req = Request(url, headers={"User-Agent": "CTI-Reranker/1.0"})
    with urlopen(req, timeout=60) as resp:
        raw = resp.read()
    data = json.loads(raw.decode("utf-8"))

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"  Cached to {cache_path} ({len(raw)/1024/1024:.1f} MB)")

    return data


def build_vocab(stix_bundle):
    """Build {attack_id: "ID — NAME"} mapping from STIX objects."""
    vocab = {}
    type_counts = defaultdict(int)
    skipped = {"revoked": 0, "deprecated": 0, "no_id": 0}

    for obj in stix_bundle.get("objects", []):
        obj_type = obj.get("type", "")
        if obj_type not in ("attack-pattern", "x-mitre-tactic", "tool", "malware"):
            continue

        if obj.get("revoked", False):
            skipped["revoked"] += 1
            continue
        if obj.get("x_mitre_deprecated", False):
            skipped["deprecated"] += 1
            continue

        attack_id = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                attack_id = ref.get("external_id")
                break

        if not attack_id:
            skipped["no_id"] += 1
            continue

        name = obj.get("name", "")
        vocab[attack_id] = f"{attack_id} {EM_DASH} {name.upper()}"

        if obj_type == "attack-pattern":
            type_counts["sub-techniques" if "." in attack_id else "techniques"] += 1
        elif obj_type == "x-mitre-tactic":
            type_counts["tactics"] += 1
        else:
            type_counts["software"] += 1

    return vocab, dict(type_counts), skipped


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Load Zenodo CSV
# ═══════════════════════════════════════════════════════════════════════

def download_zenodo_csv(url=ZENODO_CSV_URL, cache_path="MITRE_ATTACKv15-1_TTP_Dataset.csv"):
    """Download the Zenodo CSV, with local cache."""
    if os.path.exists(cache_path):
        print(f"  Using cached CSV: {cache_path}")
        return cache_path

    print(f"  Downloading Zenodo TTP dataset...")
    req = Request(url, headers={"User-Agent": "CTI-Reranker/1.0"})
    with urlopen(req, timeout=120) as resp:
        raw = resp.read()

    with open(cache_path, "wb") as f:
        f.write(raw)

    print(f"  Saved to {cache_path} ({len(raw)/1024/1024:.1f} MB)")
    return cache_path


def load_zenodo_csv(csv_path):
    """
    Load the Zenodo TTP CSV.

    Format: Key (TTP ID), Value (procedure description)
    Example row: T1059.001, "Adversaries may abuse PowerShell..."
    """
    rows = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Verify columns
        if "Key" not in reader.fieldnames or "Value" not in reader.fieldnames:
            print(f"  ERROR: Expected columns 'Key' and 'Value'")
            print(f"  Found: {reader.fieldnames}")
            sys.exit(1)

        for row in reader:
            ttp_id = row["Key"].strip()
            description = row["Value"].strip()
            if ttp_id and description:
                rows.append({"ttp_id": ttp_id, "text": description})

    # Summary stats
    unique_ids = set(r["ttp_id"] for r in rows)
    text_lengths = [len(r["text"]) for r in rows]
    avg_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0

    print(f"  Loaded {len(rows):,} rows")
    print(f"  Unique TTP IDs: {len(unique_ids)}")
    print(f"  Text length: avg {avg_len:.0f} chars, "
          f"min {min(text_lengths)}, max {max(text_lengths)}")

    return rows


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Convert to Reranker Pairs
# ═══════════════════════════════════════════════════════════════════════

def convert_to_pairs(rows, vocab, n_negatives=8):
    """
    Convert Zenodo rows to reranker pair format.

    Each Zenodo row is a single (ttp_id, description) pair — simpler
    than tumeteor's multi-label format. For each row:
      - Creates one positive pair (description → technique)
      - Samples n_negatives random negatives from full vocab
    """
    random.seed(RANDOM_SEED)
    all_ids = sorted(vocab.keys())
    output = []

    stats = {
        "input_rows": 0,
        "skipped_empty": 0,
        "skipped_not_in_v14": 0,
        "output_queries": 0,
        "positive_pairs": 0,
        "negative_pairs": 0,
    }

    dropped_ids = defaultdict(int)

    for row in rows:
        stats["input_rows"] += 1

        ttp_id = row["ttp_id"]
        text = row["text"]

        if not text.strip():
            stats["skipped_empty"] += 1
            continue

        # Filter to v14 vocabulary
        if ttp_id not in vocab:
            dropped_ids[ttp_id] += 1
            stats["skipped_not_in_v14"] += 1
            continue

        stats["output_queries"] += 1
        query_raw = text
        query_norm = text.lower()

        # ── Positive pair ──
        output.append({
            "query_raw": query_raw,
            "query_norm": query_norm,
            "candidate_id": ttp_id,
            "candidate_norm": ttp_id,
            "candidate_text": vocab[ttp_id],
            "label": 1,
            "actor": ACTOR_NAME,
        })
        stats["positive_pairs"] += 1

        # ── Negative pairs ──
        neg_pool = [aid for aid in all_ids if aid != ttp_id]
        n_neg = min(n_negatives, len(neg_pool))
        sampled = random.sample(neg_pool, n_neg)

        for neg_id in sampled:
            output.append({
                "query_raw": query_raw,
                "query_norm": query_norm,
                "candidate_id": neg_id,
                "candidate_norm": neg_id,
                "candidate_text": vocab[neg_id],
                "label": 0,
                "actor": ACTOR_NAME,
            })
            stats["negative_pairs"] += 1

    stats["top_dropped_ids"] = sorted(dropped_ids.items(),
                                       key=lambda x: -x[1])[:20]
    return output, stats


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Write Output
# ═══════════════════════════════════════════════════════════════════════

def write_jsonl(pairs, output_path):
    """Write pairs to JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def combine_jsonl(file_list, combined_path):
    """Concatenate multiple JSONL files into one."""
    total = 0
    os.makedirs(os.path.dirname(combined_path) or ".", exist_ok=True)
    with open(combined_path, "w", encoding="utf-8") as out:
        for src in file_list:
            if not os.path.exists(src):
                print(f"  WARNING: {src} not found, skipping")
                continue
            with open(src, "r", encoding="utf-8") as inp:
                for line in inp:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        total += 1
            print(f"  + {src}: added to combined file")
    return total


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert Zenodo TTP dataset to reranker pairs format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python zenodo_to_reranker.py
  python zenodo_to_reranker.py --negatives 10
  python zenodo_to_reranker.py --csv MITRE_ATTACKv15-1_TTP_Dataset.csv
  python zenodo_to_reranker.py --combine-all
        """
    )
    parser.add_argument(
        "--negatives", type=int, default=8,
        help="Random negatives per query (default: 8)"
    )
    parser.add_argument(
        "--output", type=str,
        default="data/reranker_pairs_enriched_zenodo.jsonl",
        help="Output JSONL path"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to pre-downloaded Zenodo CSV (skips download)"
    )
    parser.add_argument(
        "--stix-cache", type=str, default="enterprise-attack-v14.json",
        help="Local cache path for STIX bundle"
    )
    parser.add_argument(
        "--combine-all", action="store_true",
        help="Combine CTI-HAL + tumeteor + zenodo into one file for training"
    )
    parser.add_argument(
        "--combined-output", type=str,
        default="data/reranker_pairs_all_combined.jsonl",
        help="Combined output path"
    )
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  ZENODO TTP -> RERANKER PAIRS CONVERSION")
    print("  Dataset Expansion Phase 2")
    print("  Branch: dataset-expansion")
    print("=" * 70)

    # ── Step 1: ATT&CK Vocabulary ─────────────────────────────────────
    print("\n[1/4] Building ATT&CK v14 vocabulary...")
    stix = download_stix(cache_path=args.stix_cache)
    vocab, type_counts, skip_counts = build_vocab(stix)

    print(f"  Vocabulary: {len(vocab)} entries")
    for k, v in sorted(type_counts.items()):
        print(f"    {k}: {v}")

    assert len(vocab) >= 600, f"Expected 600+ vocab entries, got {len(vocab)}"
    print(f"  Sample: {vocab.get('T1059', 'MISSING')}")

    # ── Step 2: Load Zenodo CSV ───────────────────────────────────────
    print("\n[2/4] Loading Zenodo TTP dataset...")
    csv_path = args.csv if args.csv else download_zenodo_csv()
    rows = load_zenodo_csv(csv_path)

    if not rows:
        print("\n  ERROR: No data loaded.")
        sys.exit(1)

    # ── Step 3: Convert ───────────────────────────────────────────────
    print(f"\n[3/4] Converting to reranker pairs...")
    print(f"  Negatives per query: {args.negatives}")
    pairs, stats = convert_to_pairs(rows, vocab, n_negatives=args.negatives)

    print(f"\n  Conversion Results:")
    print(f"  {'Input rows:':<30} {stats['input_rows']:>10,}")
    print(f"  {'Skipped (empty):':<30} {stats['skipped_empty']:>10,}")
    print(f"  {'Skipped (not in v14):':<30} {stats['skipped_not_in_v14']:>10,}")
    print(f"  {'Output queries:':<30} {stats['output_queries']:>10,}")
    print(f"  {'Positive pairs:':<30} {stats['positive_pairs']:>10,}")
    print(f"  {'Negative pairs:':<30} {stats['negative_pairs']:>10,}")
    print(f"  {'Total output rows:':<30} {len(pairs):>10,}")

    if pairs:
        pos_rate = stats['positive_pairs'] / len(pairs) * 100
        print(f"  {'Positive rate:':<30} {pos_rate:>9.1f}%")

    if stats["top_dropped_ids"]:
        print(f"\n  Top dropped v15.1 IDs (not in v14):")
        for tid, cnt in stats["top_dropped_ids"][:10]:
            print(f"    {tid}: {cnt} occurrences")

    # ── Step 4: Write Output ──────────────────────────────────────────
    print(f"\n[4/4] Writing output...")
    write_jsonl(pairs, args.output)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  Wrote {args.output} ({size_mb:.1f} MB, {len(pairs):,} rows)")

    # ── Optional: Combine all datasets ────────────────────────────────
    if args.combine_all:
        print(f"\n  Combining all datasets...")
        files_to_combine = [
            "data/reranker_pairs_enriched.jsonl",          # CTI-HAL
            "data/reranker_pairs_enriched_tumeteor.jsonl",  # Tumeteor
            args.output,                                     # Zenodo
        ]
        total = combine_jsonl(files_to_combine, args.combined_output)
        combined_mb = os.path.getsize(args.combined_output) / (1024 * 1024)
        print(f"  Combined: {args.combined_output}")
        print(f"  Total rows: {total:,} ({combined_mb:.1f} MB)")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)

    if not args.combine_all:
        print(f"\n  Next steps:")
        print(f"  1. Combine all datasets:")
        print(f"       python zenodo_to_reranker.py --combine-all")
        print(f"  2. Upload to Colab for two-stage training:")
        print(f"       Stage 1: tumeteor + zenodo (broad vocabulary)")
        print(f"       Stage 2: CTI-HAL (specialization)")
    else:
        print(f"\n  Ready for two-stage training!")
        print(f"  Upload these to Colab:")
        print(f"    - {args.combined_output} (Stage 1 + 2 combined)")
        print(f"    - data/reranker_pairs_enriched.jsonl (CTI-HAL for Stage 2)")
        print(f"    - finetune_production.py")

    print()


if __name__ == "__main__":
    main()
