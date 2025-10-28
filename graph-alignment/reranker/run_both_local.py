# run_both_local.py  —  robust ID + LABEL evaluation with preflight & optional positional join
import csv, json, os, re, sys
from pathlib import Path

ROOT = Path.cwd()

# Inputs
ID_CSV   = ROOT / "ce_top1_final_ID.csv"
ID_GOLD  = ROOT / "gold_id_unbalanced_nobom.jsonl"
LAB_CSV  = ROOT / "ce_top1_final_LABEL_metrics.csv"
LAB_GOLD = ROOT / "gold_label_unbalanced_nobom.jsonl"

# Artifacts
TOP1_ID      = ROOT / "top1_eval_id.csv"
TOP1_LABEL   = ROOT / "top1_eval_label.csv"
METRICS_ID   = ROOT / "metrics_id.txt"
METRICS_LAB  = ROOT / "metrics_label.txt"
LABEL_DEBUG  = ROOT / "label_vs_gold_overlap_report.txt"
SUMMARY      = ROOT / "both_summary.txt"

# Behavior: allow positional join to fill missing label queries from ID CSV
ALLOW_POSITIONAL_JOIN = True  # set to False if you do not want this fallback

# Utility
def norm(s:str)->str:
    if s is None: return ""
    return re.sub(r"\s+"," ",str(s).strip().lower())

def read_csv_head(p:Path, n=2):
    rows=[]
    with p.open("r",encoding="utf-8-sig",newline="") as f:
        r=csv.DictReader(f); hdr=r.fieldnames or []
        for i,row in enumerate(r):
            rows.append(row)
            if i+1>=n: break
    return hdr, rows

def sniff_cols(hdr:list[str], prefer:dict)->dict:
    """Map logical names -> actual columns; prefer exact matches, else first match by substring"""
    hset = {h.lower(): h for h in hdr}
    out={}
    for role,names in prefer.items():
        found=None
        # exact first
        for nm in names:
            if nm.lower() in hset:
                found = hset[nm.lower()]; break
        if not found:
            # substring
            for h in hdr:
                for nm in names:
                    if nm.lower() in h.lower():
                        found=h; break
                if found: break
        out[role]=found
    return out

def read_jsonl_map(p:Path)->dict:
    m={}
    if not p.exists(): return m
    with p.open("r",encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try:
                o=json.loads(ln)
                qn=norm(o.get("query",""))
                g=o.get("gold","")
                if qn and g is not None: m[qn]=str(g)
            except Exception:
                pass
    return m

def p_at_1(eval_rows:list[dict], gold_map:dict, key_q="query", key_pred="candidate")->tuple:
    overlap=0; correct=0
    for r in eval_rows:
        qn=norm(r.get(key_q))
        if not qn: continue
        if qn in gold_map:
            overlap+=1
            if norm(r.get(key_pred))==norm(gold_map[qn]):
                correct+=1
    p1 = None if overlap==0 else correct/overlap
    return p1,overlap,correct

def write_csv(p:Path,hdr:list[str],rows:list[dict]):
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=hdr); w.writeheader()
        for r in rows: w.writerow(r)

def write_jsonl(p:Path,rows:list[dict]):
    with p.open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

def main():
    print("=== Local ID + LABEL evaluation (robust) ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"CWD   : {ROOT}")

    # Preflight presence
    print("\nPreflight presence:")
    for p in [ID_CSV, LAB_CSV, ID_GOLD, LAB_GOLD]:
        print(f"  {p}: {'OK' if p.exists() else 'MISSING'}")

    # Read heads + sample rows
    id_hdr,id_sample = read_csv_head(ID_CSV, n=2) if ID_CSV.exists() else ([],[])
    lab_hdr,lab_sample= read_csv_head(LAB_CSV, n=2) if LAB_CSV.exists() else ([],[])

    # Print detected headers and samples
    print(f"[ID] Detected headers in {ID_CSV.name}: {id_hdr}")
    if id_sample:
        print(f"[ID] Sample rows (first 2):")
        for r in id_sample:
            print("       ", {k:r.get(k,"") for k in id_hdr[:min(6,len(id_hdr))]})
    print(f"[LABEL] Detected headers in {LAB_CSV.name}: {lab_hdr}")
    if lab_sample:
        print(f"[LABEL] Sample rows (first 2):")
        for r in lab_sample:
            print("       ", {k:r.get(k,"") for k in lab_hdr[:min(6,len(lab_hdr))]})

    # ---- ID evaluator ----
    id_cols = sniff_cols(id_hdr, {"q":["query","q"], "d":["candidate","cand","id","name"], "p":["p_relevant","p","score","prob"]})
    id_eval=[]
    if ID_CSV.exists() and id_cols["q"] and id_cols["d"]:
        with ID_CSV.open("r",encoding="utf-8-sig",newline="") as f:
            r=csv.DictReader(f)
            for row in r:
                q=row.get(id_cols["q"],"")
                d=row.get(id_cols["d"],"")
                if q and d:
                    id_eval.append({"query":q,"candidate":d})
    write_csv(TOP1_ID, ["query","candidate"], id_eval)

    id_gold = read_jsonl_map(ID_GOLD) if ID_GOLD.exists() else {}
    id_p1,id_overlap,id_correct = p_at_1(id_eval,id_gold,"query","candidate")

    # ---- LABEL evaluator ----
    lab_cols = sniff_cols(lab_hdr, {"q":["query","q"], "d":["candidate","label","name"], "p":["p_relevant","p","prob","score"]})
    lab_eval=[]
    label_note=""

    if LAB_CSV.exists() and lab_cols["d"]:
        # fill query field directly if present, else optional positional join
        q_col = lab_cols["q"]
        if q_col:
            with LAB_CSV.open("r",encoding="utf-8-sig",newline="") as f:
                r=csv.DictReader(f)
                for row in r:
                    q=row.get(q_col,"")
                    d=row.get(lab_cols["d"],"")
                    if q and d: lab_eval.append({"query":q,"candidate":d})
        elif ALLOW_POSITIONAL_JOIN and id_eval:
            # positional join: take queries from ID in same order
            with LAB_CSV.open("r",encoding="utf-8-sig",newline="") as f:
                r=csv.DictReader(f)
                for i,row in enumerate(r):
                    d=row.get(lab_cols["d"],"")
                    q = id_eval[i]["query"] if i < len(id_eval) else ""
                    if q and d: lab_eval.append({"query":q,"candidate":d})
            label_note="(used positional join from ID for missing label queries)"
        else:
            label_note="(label CSV had no query column and positional join disabled)"

    write_csv(TOP1_LABEL, ["query","candidate"], lab_eval)

    # Build label gold map
    lab_gold = read_jsonl_map(LAB_GOLD) if LAB_GOLD.exists() else {}
    lab_p1,lab_overlap,lab_correct = p_at_1(lab_eval, lab_gold, "query", "candidate")

    # Small overlap report for LABEL
    with LABEL_DEBUG.open("w",encoding="utf-8") as f:
        f.write("LABEL overlap diagnostic\n")
        f.write(f"eval rows: {len(lab_eval)}  gold rows: {len(lab_gold)}  overlap: {lab_overlap}\n")
        if label_note: f.write(f"NOTE: {label_note}\n")
        f.write("\nEval examples (first 8):\n")
        for r in lab_eval[:8]:
            f.write(f"  q='{r['query'][:80]}'  cand='{r['candidate']}'\n")
        f.write("\nGold examples (first 8):\n")
        k=0
        for q,g in lab_gold.items():
            f.write(f"  q='{q[:80]}'  gold='{g}'\n"); k+=1
            if k>=8: break

    # Write metrics text files (simple, local P@1)
    with METRICS_ID.open("w",encoding="utf-8") as f:
        f.write("Re-ranker metrics (ID)\n----------------------\n")
        f.write(f"file: {TOP1_ID}\n")
        f.write(f"eval rows: {len(id_eval)}\n")
        f.write(f"gold rows: {len(id_gold)}\n")
        f.write(f"overlap: {id_overlap}\n")
        f.write(f"P@1 (overall): {'NA' if id_p1 is None else f'{id_p1:.3f}'}\n")
    with METRICS_LAB.open("w",encoding="utf-8") as f:
        f.write("Re-ranker metrics (LABEL)\n-------------------------\n")
        f.write(f"file: {TOP1_LABEL}\n")
        f.write(f"eval rows: {len(lab_eval)}\n")
        f.write(f"gold rows: {len(lab_gold)}\n")
        f.write(f"overlap: {lab_overlap}\n")
        f.write(f"P@1 (exact): {'NA' if lab_p1 is None else f'{lab_p1:.3f}'}\n")
        if label_note: f.write(f"NOTE: {label_note}\n")

    # Final summary
    summary = f"""
SUMMARY
-------
[ID]
  built eval rows : {len(id_eval)}
  gold rows       : {len(id_gold)}
  overlap         : {id_overlap}
  P@1 (overall)   : {'NA' if id_p1 is None else f'{id_p1:.3f}'}

[LABEL]
  built eval rows : {len(lab_eval)}
  gold rows       : {len(lab_gold)}
  overlap         : {lab_overlap}
  P@1 (exact)     : {'NA' if lab_p1 is None else f'{lab_p1:.3f}'}
  {label_note}
Artifacts:
  - {TOP1_ID.name}
  - {TOP1_LABEL.name}
  - {METRICS_ID.name}
  - {METRICS_LAB.name}
  - {LABEL_DEBUG.name}
"""
    print(summary)
    SUMMARY.write_text(summary, encoding="utf-8")

if __name__ == "__main__":
    main()
