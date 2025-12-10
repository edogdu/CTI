# metrics_top1.py
# Lightweight metrics for your reranker outputs.
# - Works with ce_k3_top1_full.csv (or ce_top1_final.csv)
# - Computes totals, keep-rate, flips (if post_keep exists), and type distribution
# - If you also pass a gold JSONL (see --gold), computes P@1 overall and by actor group

import argparse, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            yield {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}

def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def pick(row, *cands, default=None):
    for c in cands:
        if c in row and row[c] not in (None, ""):
            return row[c]
    return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top1_csv", required=True,
                    help="Path to ce_k3_top1_full.csv or ce_top1_final.csv")
    ap.add_argument("--gold_jsonl", default=None,
                    help="Optional: JSONL with gold labels to compute P@1. "
                         "Expected fields: group/actor (optional), and either "
                         " (q or query) + (gold_label or gold_id)")
    ap.add_argument("--group_map", default=None,
                    help="Optional CSV mapping query to actor/group if not in gold file. "
                         "Expected columns: query,group (or actor)")
    ap.add_argument("--out", default="exports/metrics_summary.txt",
                    help="Where to write a human-readable summary.")
    args = ap.parse_args()

    top1 = list(read_csv(args.top1_csv))
    if not top1:
        print("No rows in", args.top1_csv)
        sys.exit(1)

    # Flexible column names (handles both ce_k3_top1_full.csv and ce_top1_final.csv)
    # Try to locate relevant columns
    q_cols     = ["query", "q", "text_a"]
    cand_cols  = ["typed_cand", "candidate", "cand", "base_cand"]
    type_cols  = ["type", "typed_type"]
    keep_cols  = ["post_keep", "keep"]
    reason_col = "reason"

    # Compute metrics without gold
    total = len(top1)
    keep_count = 0
    pre_keep_count = 0
    flips = 0
    reasons = Counter()
    type_counts = Counter()

    # group map (optional)
    q2group = {}
    if args.group_map:
        for row in read_csv(args.group_map):
            qtxt = pick(row, "query", "q", default=None)
            grp  = pick(row, "group", "actor", default=None)
            if qtxt and grp:
                q2group[qtxt] = grp

    rows_norm = []
    for r in top1:
        q = pick(r, *q_cols, default="")
        cand = pick(r, *cand_cols, default="")
        rtype = pick(r, *type_cols, default="")
        # Keep logic: prefer post_keep if present else keep
        keep_val = pick(r, *keep_cols, default="")
        pre_keep_val = r.get("keep", "")
        reason = r.get(reason_col, "")

        # Normalize keep to {0,1}
        def as01(v):
            vs = str(v).strip().lower()
            return 1 if vs in ("1","true","keep","yes") else 0

        keep01 = as01(keep_val)
        pre_keep01 = as01(pre_keep_val)

        keep_count += keep01
        pre_keep_count += pre_keep01
        if keep01 != pre_keep01:
            flips += 1
            if reason:
                reasons[reason] += 1
            else:
                reasons["(no_reason)"] += 1

        if rtype:
            type_counts[rtype] += 1

        rows_norm.append({"query": q, "candidate": cand, "type": rtype,
                          "keep": keep01, "pre_keep": pre_keep01})

    # Optional: compute P@1 if gold file provided
    p_at_1 = None
    p_at_1_by_group = None
    if args.gold_jsonl:
        # We’ll accept either exact label match or ID match if present
        # Expected fields:
        #  - query text: one of ["q", "query"]
        #  - gold label: ["gold_label", "gold"] (string matching your candidate label)
        #  - optional: actor/group: ["group","actor"]
        gold = {}
        q2actor = {}
        for g in read_jsonl(args.gold_jsonl):
            qtxt = (g.get("q") or g.get("query") or "").strip()
            gold_label = (g.get("gold_label") or g.get("gold") or "").strip()
            actor = (g.get("group") or g.get("actor") or "").strip()
            if qtxt:
                gold[qtxt] = gold_label
                if actor:
                    q2actor[qtxt] = actor

        # fill group from map if not in gold
        if q2group and not q2actor:
            q2actor = q2group

        correct = 0
        total_eval = 0
        correct_by_grp = Counter()
        total_by_grp = Counter()

        for r in rows_norm:
            q = r["query"]
            cand = r["candidate"]
            if not q or q not in gold:
                continue
            total_eval += 1
            grp = q2actor.get(q, "(unknown)")
            total_by_grp[grp] += 1
            if cand and gold[q] and cand == gold[q]:
                correct += 1
                correct_by_grp[grp] += 1

        if total_eval > 0:
            p_at_1 = correct / total_eval
            p_at_1_by_group = {g: (correct_by_grp[g] / total_by_grp[g]
                                   if total_by_grp[g] else 0.0)
                               for g in sorted(total_by_grp)}

    # Write a concise, human-readable summary
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        def w(s=""):
            f.write(str(s) + "\n")
        w("Re-ranker metrics")
        w("-----------------")
        w(f"file: {args.top1_csv}")
        w(f"total queries: {total}")
        w(f"keep (pre): {pre_keep_count}  ({pre_keep_count/total:.2%})")
        w(f"keep (post): {keep_count}  ({keep_count/total:.2%})")
        w(f"flips caused by type-aware rule: {flips}")
        if reasons:
            w("flip reasons (top 5):")
            for k, v in reasons.most_common(5):
                w(f"  {k}: {v}")

        if type_counts:
            w("top-1 type distribution:")
            for t, c in type_counts.most_common():
                w(f"  {t}: {c} ({c/total:.2%})")

        if p_at_1 is not None:
            w(f"P@1 (overall): {p_at_1:.3f}")
            if p_at_1_by_group:
                w("P@1 by group:")
                width = max((len(g) for g in p_at_1_by_group), default=10)
                for g in sorted(p_at_1_by_group):
                    w(f"  {g:<{width}}  {p_at_1_by_group[g]:.3f}")

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
