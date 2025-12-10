# misses_report.py
import csv, json, sys, argparse

def load_gold(path):
    out = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                out[o["query"]] = o["gold"]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top1_csv", required=True)
    ap.add_argument("--gold_jsonl", required=True)
    ap.add_argument("--out_csv", default="debug_misses_top40.csv")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--p_col", default="p_relevant")
    ap.add_argument("--pirr_col", default="p_irrelevant")
    args = ap.parse_args()

    gold = load_gold(args.gold_jsonl)
    rows = list(csv.DictReader(open(args.top1_csv, newline="", encoding="utf-8")))
    keep_cols = set(["query","candidate",args.p_col,args.pirr_col,"margin"])
    enriched = []
    for r in rows:
        q = r.get("query","")
        g = gold.get(q,"")
        if not g: continue
        pred = r.get("candidate","")
        if pred == g: continue
        rr = {k: r.get(k,"") for k in keep_cols if k in r}
        rr["gold"] = g
        try:
            rr["p_relevant"] = float(rr.get(args.p_col,0))
        except: rr["p_relevant"] = 0.0
        enriched.append(rr)
    enriched.sort(key=lambda x: x.get("p_relevant",0), reverse=True)
    out = enriched[:args.k]
    with open(args.out_csv,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted(out[0].keys()) if out else ["query","candidate","gold"])
        w.writeheader(); w.writerows(out)
    print(f"Wrote {args.out_csv} (n={len(out)})")

if __name__ == "__main__":
    sys.exit(main())
