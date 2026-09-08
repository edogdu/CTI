"""Deduplicate DNRTI dataset.

For duplicate texts:
- Identical annotations: keep one copy (first occurrence)
- Different annotations: keep the richest (most entities + relations)

All copies of the same text are grouped so they land in the same split partition.

Usage:
    python tools/deduplicate_dnrti.py \
        --input datasets/DNRTI/dnrti_aug_stix2_je.json \
        --output datasets/DNRTI/dnrti_dedup.json \
        --report results/finetuning/data/dedup_report.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def richest_doc(docs: list[dict]) -> dict:
    """Return the doc with the most entities + relations."""
    return max(docs, key=lambda d: len(d.get("entities", [])) + len(d.get("relations", [])))


def deduplicate(data: list[dict]) -> tuple[list[dict], dict]:
    """Deduplicate by text, keeping the richest annotation per unique text.

    Returns (deduped_data, report_dict).
    """
    text_groups = defaultdict(list)
    for i, doc in enumerate(data):
        text_groups[doc["text"]].append((i, doc))

    deduped = []
    stats = {
        "original_count": len(data),
        "unique_texts": len(text_groups),
        "pure_duplicates_removed": 0,
        "annotation_variants_resolved": 0,
        "singletons": 0,
        "examples_removed": [],
    }

    for text, group in text_groups.items():
        if len(group) == 1:
            # Unique text — keep as-is
            deduped.append(group[0][1])
            stats["singletons"] += 1
        else:
            # Duplicate text — check if annotations differ
            ent_signatures = set()
            for idx, doc in group:
                sig = str(sorted([str(e) for e in doc.get("entities", [])]))
                ent_signatures.add(sig)

            if len(ent_signatures) == 1:
                # All copies have identical annotations — pure dupe
                deduped.append(group[0][1])
                stats["pure_duplicates_removed"] += len(group) - 1
            else:
                # Different annotations — keep the richest
                best = richest_doc([doc for _, doc in group])
                deduped.append(best)
                stats["annotation_variants_resolved"] += 1
                stats["pure_duplicates_removed"] += len(group) - 1

                # Log what we chose
                if len(stats["examples_removed"]) < 5:
                    stats["examples_removed"].append({
                        "text": text[:120] + "...",
                        "copies": len(group),
                        "kept_entities": len(best.get("entities", [])),
                        "kept_relations": len(best.get("relations", [])),
                        "other_entity_counts": [
                            len(doc.get("entities", []))
                            for _, doc in group
                            if doc is not best
                        ],
                    })

    stats["final_count"] = len(deduped)
    return deduped, stats


def main():
    parser = argparse.ArgumentParser(description="Deduplicate DNRTI dataset")
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output deduplicated JSON path")
    parser.add_argument("--report", default=None, help="Optional report JSON path")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    deduped, stats = deduplicate(data)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=None)

    print(f"Original:    {stats['original_count']}")
    print(f"Unique texts: {stats['unique_texts']}")
    print(f"Pure dupes removed:       {stats['pure_duplicates_removed']}")
    print(f"Annotation variants resolved: {stats['annotation_variants_resolved']}")
    print(f"Final:       {stats['final_count']}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Report saved: {args.report}")


if __name__ == "__main__":
    main()
