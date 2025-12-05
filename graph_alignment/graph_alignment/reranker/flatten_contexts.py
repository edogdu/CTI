# flatten_contexts.py
# Convert s_contexts (list) -> a single query string and add a stable query_id.

import json, hashlib, argparse
from pathlib import Path

def first_nonempty(strings):
    if isinstance(strings, list):
        for s in strings:
            if isinstance(s, str) and s.strip():
                return s.strip()
        return ""
    if isinstance(strings, str):
        return strings.strip()
    return ""

def make_qid(report_id: str, query_text: str) -> str:
    base = f"{report_id or ''}|{query_text or ''}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]

def run(in_path: Path, out_path: Path):
    n_in, n_out, n_skipped = 0, 0, 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)

            # 1) Flatten s_contexts -> query (take the first non-empty string)
            query_text = first_nonempty(rec.get("s_contexts"))

            # If nothing usable, try any fallback single-string fields you might have
            if not query_text:
                query_text = first_nonempty(rec.get("s_context"))

            if not query_text:
                n_skipped += 1
                continue

            # 2) Stable query_id from report_id + query_text
            report_id = rec.get("s_report_id") or rec.get("report_id") or ""
            qid = make_qid(report_id, query_text)

            # 3) Keep useful candidate fields; don’t drop originals if present
            out = {
                "query_id": qid,
                "query": query_text,
                "s_report_id": report_id,
                "s_report_name": rec.get("s_report_name"),
                "candidate_id": rec.get("c_clean_id") or rec.get("candidate_id"),
                "candidate_label": rec.get("c_name") or rec.get("candidate_label"),
                "candidate_type": rec.get("c_type") or rec.get("candidate_type"),
                "label": int(rec.get("label", 0)),
            }

            # keep similarity if present
            if "cosine" in rec:
                try:
                    out["cosine"] = float(rec["cosine"])
                except Exception:
                    pass

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[OK] read={n_in} wrote={n_out} skipped_empty_context={n_skipped}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Flatten s_contexts to a single query string and add query_id.")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL (e.g., reranker_dataset.jsonl)")
    ap.add_argument("--out", dest="out", default="flattened_queries.jsonl", help="Output JSONL")
    args = ap.parse_args()
    run(Path(args.inp), Path(args.out))
