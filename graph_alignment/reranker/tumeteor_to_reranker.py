#!/usr/bin/env python3
"""
tumeteor_to_reranker.py — Dataset Expansion Phase 1
====================================================
Converts the tumeteor/Security-TTP-Mapping dataset into reranker_pairs
format compatible with finetune_production.py.

This script:
  1. Downloads ATT&CK v14.1 STIX bundle from MITRE's GitHub
  2. Builds the "{ID} — {NAME}" vocabulary matching the pipeline format
  3. Loads all tumeteor data via HuggingFace datasets library
  4. Converts to reranker pairs with positive + random negative candidates
  5. Outputs JSONL in the exact format finetune_production.py expects

Prerequisites:
    pip install datasets

Usage:
    python tumeteor_to_reranker.py
    python tumeteor_to_reranker.py --negatives 10
    python tumeteor_to_reranker.py --stix-cache enterprise-attack-v14.json
    python tumeteor_to_reranker.py --combine data/reranker_pairs_enriched.jsonl

Branch: dataset-expansion
Author: Shane Waldrop
"""

import json
import os
import sys
import random
import argparse
import ast
from pathlib import Path
from collections import defaultdict
from urllib.request import urlopen, Request

# ── Constants ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
EM_DASH = "\u2014"                   # Unicode em dash, matches pipeline
ACTOR_NAME = "external_tumeteor"     # Keeps CTI-HAL per-actor eval clean

# ATT&CK v14.1 STIX bundle — matches the version stated in Paper Id 611
STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/"
    "ATT%26CK-v14.1/enterprise-attack/enterprise-attack.json"
)


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Build ATT&CK v14 Vocabulary
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
    """
    Build {attack_id: "ID — NAME"} mapping from STIX objects.

    Includes techniques, sub-techniques, tactics, and software/malware.
    Skips revoked and deprecated entries.
    Formats names in ALL CAPS with em dash to match pipeline exactly.
    """
    vocab = {}
    type_counts = defaultdict(int)
    skipped = {"revoked": 0, "deprecated": 0, "no_id": 0}

    for obj in stix_bundle.get("objects", []):
        obj_type = obj.get("type", "")
        if obj_type not in ("attack-pattern", "x-mitre-tactic", "tool", "malware"):
            continue

        # Skip revoked or deprecated
        if obj.get("revoked", False):
            skipped["revoked"] += 1
            continue
        if obj.get("x_mitre_deprecated", False):
            skipped["deprecated"] += 1
            continue

        # Extract ATT&CK ID from external references
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

        # Count by type
        if obj_type == "attack-pattern":
            type_counts["sub-techniques" if "." in attack_id else "techniques"] += 1
        elif obj_type == "x-mitre-tactic":
            type_counts["tactics"] += 1
        else:
            type_counts["software"] += 1

    return vocab, dict(type_counts), skipped


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Load Tumeteor Dataset
# ═══════════════════════════════════════════════════════════════════════

def load_tumeteor():
    """
    Load all tumeteor data via HuggingFace datasets library.

    Loads the default config with all splits (train/validation/test).
    Everything becomes training data for us — we evaluate only on CTI-HAL.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("\n  ERROR: 'datasets' library not installed.")
        print("  Run: pip install datasets")
        sys.exit(1)

    all_rows = []
    split_counts = {}

    print("    Loading 'tumeteor/Security-TTP-Mapping' (default config)...")
    try:
        ds = load_dataset("tumeteor/Security-TTP-Mapping")
    except Exception as e:
        print(f"    ERROR: Could not load dataset: {e}")
        return all_rows

    for split_name in ds.keys():
        count = 0
        for row in ds[split_name]:
            all_rows.append({
                "text1": row["text1"],
                "labels": row["labels"],
                "source": split_name,
            })
            count += 1
        split_counts[split_name] = count

    print(f"  Loaded by split:")
    for split, cnt in split_counts.items():
        print(f"    {split}: {cnt:,} rows")

    return all_rows


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Parse Labels & Convert to Reranker Pairs
# ═══════════════════════════════════════════════════════════════════════

def parse_labels(label_val):
    """
    Parse the labels field from tumeteor dataset.

    Handles:
      - Python list objects: ['T1057']
      - String repr:         "['T1057', 'T1113']"
      - Comma-separated:     "T1057, T1113"
    """
    if isinstance(label_val, list):
        return [str(x).strip() for x in label_val if str(x).strip()]

    if isinstance(label_val, str):
        try:
            result = ast.literal_eval(label_val)
            if isinstance(result, list):
                return [str(x).strip() for x in result]
            return [str(result).strip()]
        except (ValueError, SyntaxError):
            cleaned = label_val.strip("[]' ")
            if not cleaned:
                return []
            return [t.strip().strip("'\"") for t in cleaned.split(",")]

    return []


def convert_to_pairs(rows, vocab, n_negatives=8):
    """
    Convert tumeteor rows to reranker pair format.

    For each (text, [gold_ids]) row:
      - Creates one positive pair per gold ID that exists in v14 vocab
      - Samples n_negatives random negatives from full vocab
      - Negatives exclude ALL gold labels for that query
      - Formats output exactly like reranker_pairs_enriched.jsonl

    Returns (list_of_pairs, stats_dict).
    """
    random.seed(RANDOM_SEED)
    all_ids = sorted(vocab.keys())
    output = []

    stats = {
        "input_rows": 0,
        "skipped_empty_text": 0,
        "skipped_no_valid_labels": 0,
        "v12_labels_total": 0,
        "v12_labels_kept": 0,
        "v12_labels_dropped": 0,
        "positive_pairs": 0,
        "negative_pairs": 0,
        "output_queries": 0,
        "multi_label_queries": 0,
    }

    dropped_ids = defaultdict(int)  # Track which IDs got dropped

    for row in rows:
        stats["input_rows"] += 1

        text = row["text1"].strip() if row.get("text1") else ""
        if not text:
            stats["skipped_empty_text"] += 1
            continue

        raw_labels = parse_labels(row["labels"])
        stats["v12_labels_total"] += len(raw_labels)

        # Filter to v14 vocabulary
        valid_labels = []
        for lbl in raw_labels:
            if lbl in vocab:
                valid_labels.append(lbl)
            else:
                dropped_ids[lbl] += 1
                stats["v12_labels_dropped"] += 1

        stats["v12_labels_kept"] += len(valid_labels)

        if not valid_labels:
            stats["skipped_no_valid_labels"] += 1
            continue

        if len(valid_labels) > 1:
            stats["multi_label_queries"] += 1

        stats["output_queries"] += 1
        query_raw = text
        query_norm = text.lower()
        gold_set = set(valid_labels)

        # ── Positive pairs (one per gold label) ──
        for label_id in valid_labels:
            output.append({
                "query_raw": query_raw,
                "query_norm": query_norm,
                "candidate_id": label_id,
                "candidate_norm": label_id,
                "candidate_text": vocab[label_id],
                "label": 1,
                "actor": ACTOR_NAME,
            })
            stats["positive_pairs"] += 1

        # ── Negative pairs (random from full vocab, excluding golds) ──
        neg_pool = [aid for aid in all_ids if aid not in gold_set]
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

    # Add top dropped IDs to stats
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


def combine_jsonl(original_path, tumeteor_path, combined_path):
    """
    Concatenate original CTI-HAL data with tumeteor data.

    Writes a combined JSONL file that finetune_production.py can
    load directly from its default path.
    """
    total = 0
    os.makedirs(os.path.dirname(combined_path) or ".", exist_ok=True)
    with open(combined_path, "w", encoding="utf-8") as out:
        for src in [original_path, tumeteor_path]:
            with open(src, "r", encoding="utf-8") as inp:
                for line in inp:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        total += 1
    return total


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert tumeteor dataset to reranker pairs format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tumeteor_to_reranker.py
  python tumeteor_to_reranker.py --negatives 10
  python tumeteor_to_reranker.py --combine data/reranker_pairs_enriched.jsonl
        """
    )
    parser.add_argument(
        "--negatives", type=int, default=8,
        help="Random negatives per query (default: 8)"
    )
    parser.add_argument(
        "--output", type=str,
        default="data/reranker_pairs_enriched_tumeteor.jsonl",
        help="Output JSONL path (default: data/reranker_pairs_enriched_tumeteor.jsonl)"
    )
    parser.add_argument(
        "--stix-cache", type=str, default="enterprise-attack-v14.json",
        help="Local cache path for STIX bundle (avoids re-download)"
    )
    parser.add_argument(
        "--combine", type=str, default=None, metavar="ORIGINAL_JSONL",
        help="Path to original reranker_pairs_enriched.jsonl to produce combined file"
    )
    parser.add_argument(
        "--combined-output", type=str,
        default="data/reranker_pairs_combined.jsonl",
        help="Combined output path (default: data/reranker_pairs_combined.jsonl)"
    )
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  TUMETEOR -> RERANKER PAIRS CONVERSION")
    print("  Dataset Expansion Phase 1")
    print("  Branch: dataset-expansion")
    print("=" * 70)

    # ── Step 1: ATT&CK Vocabulary ─────────────────────────────────────
    print("\n[1/4] Building ATT&CK v14 vocabulary...")
    stix = download_stix(cache_path=args.stix_cache)
    vocab, type_counts, skip_counts = build_vocab(stix)

    print(f"  Vocabulary: {len(vocab)} entries")
    for k, v in sorted(type_counts.items()):
        print(f"    {k}: {v}")
    if any(skip_counts.values()):
        print(f"  Skipped: {skip_counts}")

    # Sanity check
    assert len(vocab) >= 600, f"Expected 600+ vocab entries, got {len(vocab)}"
    assert "T1059" in vocab, "T1059 (Command and Scripting Interpreter) missing"
    assert "T1059.001" in vocab, "T1059.001 (PowerShell) missing"
    print(f"  Sample: {vocab.get('T1059', 'MISSING')}")
    print(f"  Sample: {vocab.get('T1059.001', 'MISSING')}")
    print(f"  Sample: {vocab.get('TA0002', 'MISSING')}")

    # ── Step 2: Load Tumeteor ─────────────────────────────────────────
    print("\n[2/4] Loading tumeteor dataset from HuggingFace...")
    rows = load_tumeteor()
    print(f"  Total rows loaded: {len(rows):,}")

    if not rows:
        print("\n  ERROR: No data loaded. Check network connection.")
        sys.exit(1)

    # ── Step 3: Convert ───────────────────────────────────────────────
    print(f"\n[3/4] Converting to reranker pairs...")
    print(f"  Negatives per query: {args.negatives}")
    pairs, stats = convert_to_pairs(rows, vocab, n_negatives=args.negatives)

    print(f"\n  Conversion Results:")
    print(f"  {'Input rows:':<30} {stats['input_rows']:>10,}")
    print(f"  {'Skipped (empty text):':<30} {stats['skipped_empty_text']:>10,}")
    print(f"  {'Skipped (no v14 labels):':<30} {stats['skipped_no_valid_labels']:>10,}")
    print(f"  {'v12 labels seen:':<30} {stats['v12_labels_total']:>10,}")
    print(f"  {'v12 labels kept (in v14):':<30} {stats['v12_labels_kept']:>10,}")
    print(f"  {'v12 labels dropped:':<30} {stats['v12_labels_dropped']:>10,}")
    print(f"  {'Multi-label queries:':<30} {stats['multi_label_queries']:>10,}")
    print(f"  {'Output queries:':<30} {stats['output_queries']:>10,}")
    print(f"  {'Positive pairs:':<30} {stats['positive_pairs']:>10,}")
    print(f"  {'Negative pairs:':<30} {stats['negative_pairs']:>10,}")
    print(f"  {'Total output rows:':<30} {len(pairs):>10,}")

    if pairs:
        pos_rate = stats['positive_pairs'] / len(pairs) * 100
        print(f"  {'Positive rate:':<30} {pos_rate:>9.1f}%")

    if stats["top_dropped_ids"]:
        print(f"\n  Top dropped v12 IDs (not in v14):")
        for tid, cnt in stats["top_dropped_ids"][:10]:
            print(f"    {tid}: {cnt} occurrences")

    # ── Step 4: Write Output ──────────────────────────────────────────
    print(f"\n[4/4] Writing output...")
    write_jsonl(pairs, args.output)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  Wrote {args.output} ({size_mb:.1f} MB, {len(pairs):,} rows)")

    # ── Optional: Combine with original data ──────────────────────────
    if args.combine:
        if not os.path.exists(args.combine):
            print(f"\n  WARNING: Original file not found: {args.combine}")
            print(f"  Skipping combine step.")
        else:
            print(f"\n  Combining with original data...")
            print(f"    Original: {args.combine}")
            print(f"    Tumeteor: {args.output}")
            total = combine_jsonl(args.combine, args.output, args.combined_output)
            combined_mb = os.path.getsize(args.combined_output) / (1024 * 1024)
            print(f"    Combined: {args.combined_output}")
            print(f"    Total rows: {total:,} ({combined_mb:.1f} MB)")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)

    if not args.combine:
        print(f"\n  Next steps:")
        print(f"  1. Review output:")
        print(f"       python -c \"import json; f=open('{args.output}'); "
              f"[print(json.dumps(json.loads(next(f)),indent=2)) for _ in range(3)]\"")
        print(f"  2. Combine with original data:")
        print(f"       python tumeteor_to_reranker.py --combine data\\reranker_pairs_enriched.jsonl")
        print(f"  3. Train on combined data:")
        print(f"       python finetune_production.py --epochs 3 --bs 16")
        print(f"       (update filepath in load_and_validate_data() to point to combined file)")
    else:
        print(f"\n  Ready to train!")
        print(f"  Update finetune_production.py load_and_validate_data() default path to:")
        print(f"    filepath='{args.combined_output}'")
        print(f"  Then run:")
        print(f"    python finetune_production.py --epochs 3 --bs 16")

    print()


if __name__ == "__main__":
    main()
