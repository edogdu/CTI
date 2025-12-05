# preflight.py (v2) -- canonicalize + fail-fast checks + manifest
import csv, json, sys, re, argparse, pathlib, hashlib, random, datetime

ID_RX = re.compile(r'^(TA\d{4}|T\d{4}(?:\.\d{3})?|S\d{4})$')

def read_jsonl(path):
    # BOM-tolerant
    for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except Exception as e:
            raise RuntimeError(f"Bad JSON on line {i} of {path}: {e}") from e

def sha256sum(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top1", required=True, help="Input Top-1 CSV (q/d or query/candidate)")
    ap.add_argument("--gold", required=True, help="Gold JSONL with query + (gold|gold_id|gold_label)")
    ap.add_argument("--mode", choices=["id","label"], default="id", help="Which gold representation to target")
    ap.add_argument("--out-csv", default="top1_eval.csv")
    ap.add_argument("--out-gold", default="gold_eval.jsonl")
    ap.add_argument("--overlap-floor", type=int, default=1, help="Minimum overlapping rows to proceed")
    ap.add_argument("--sample-out", default="sample50_eval.csv", help="Micro-eval subset (written for Stage 0)")
    args = ap.parse_args()

    top1 = pathlib.Path(args.top1).resolve()
    gold = pathlib.Path(args.gold).resolve()
    if not top1.exists(): sys.exit(f"[FAIL] Missing top1 CSV: {top1}")
    if not gold.exists(): sys.exit(f"[FAIL] Missing gold JSONL: {gold}")

    # read gold; canonicalize to 'gold' string field
    gold_map, rows_gold = {}, 0
    want_label = (args.mode == "label")
    for rec in read_jsonl(gold):
        rows_gold += 1
        q = (rec.get("query") or "").strip()
        if not q: continue
        g = (rec.get("gold") or
             (rec.get("gold_label") if want_label else rec.get("gold_id")) or
             (rec.get("gold_id") if not want_label else rec.get("gold_label")) or
             "").strip()
        if not g: continue
        gold_map[q] = g
    if not gold_map: sys.exit("[FAIL] Gold JSONL had no usable (query,gold) pairs.")

    # read top1
    with top1.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            sys.exit("[FAIL] Top-1 CSV has no header.")
        cols = {c.lower(): c for c in reader.fieldnames}
        qcol = cols.get("query") or cols.get("q") or cols.get("text_a") or cols.get("name")
        dcol = cols.get("candidate") or cols.get("d") or cols.get("text_b") or cols.get("id")
        if not qcol or not dcol:
            sys.exit(f"[FAIL] Could not find query/candidate columns. Headers={reader.fieldnames}")
        in_rows = list(reader)

    if not in_rows: sys.exit("[FAIL] Top-1 CSV is empty.")
    # canonicalize
    out_rows = []
    nonempty = 0
    ids_like = 0
    for r in in_rows:
        q = (r.get(qcol) or "").strip()
        d = (r.get(dcol) or "").strip()
        if d: nonempty += 1
        if ID_RX.match(d): ids_like += 1
        out_rows.append({"query": q, "candidate": d})

    if nonempty == 0:
        sys.exit("[FAIL] All candidate cells are empty.")

    # overlap
    overlap_keys = [r["query"] for r in out_rows if r["query"] in gold_map]
    overlap = len(overlap_keys)
    if overlap < args.overlap_floor:
        sys.exit(f"[FAIL] Query overlap with gold is {overlap} (< {args.overlap_floor}).")

    # write canonical eval files
    out_csv = pathlib.Path(args.out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query","candidate"])
        w.writeheader(); w.writerows(out_rows)

    out_gold = pathlib.Path(args.out_gold)
    with out_gold.open("w", encoding="utf-8") as f:
        for q in overlap_keys:
            f.write(json.dumps({"query": q, "gold": gold_map[q]}, ensure_ascii=False) + "\n")

    # micro-sample for Stage 0 (50 overlapped)
    samp = overlap_keys[:]
    random.shuffle(samp)
    samp = set(samp[:min(50, len(samp))])
    with pathlib.Path(args.sample_out).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query","candidate"])
        w.writeheader()
        for r in out_rows:
            if r["query"] in samp:
                w.writerow(r)

    # manifest
    manifest = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "mode": args.mode,
        "top1_csv": str(top1), "top1_sha256": sha256sum(top1),
        "gold_jsonl": str(gold), "gold_sha256": sha256sum(gold),
        "rows_top1": len(out_rows),
        "rows_gold": rows_gold,
        "overlap": overlap,
        "ids_like": ids_like,
        "ids_like_ratio": round(ids_like/len(out_rows), 3),
        "outputs": {"top1_eval_csv": str(out_csv), "gold_eval_jsonl": str(out_gold), "sample50_csv": args.sample_out}
    }
    pathlib.Path("manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] preflight passed. rows={len(out_rows)} overlap={overlap} ids_like={ids_like} -> {out_csv}, {out_gold}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
