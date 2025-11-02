# scripts/reranker/flatten_and_tag.py
# Reads datasets/reranker/reranker_dataset.jsonl
# Writes datasets/reranker/reranker_dataset_flat.jsonl
import json, pathlib

BASE = pathlib.Path(__file__).resolve().parents[2]  # repo root
src = BASE / "datasets" / "reranker" / "reranker_dataset.jsonl"
dst = BASE / "datasets" / "reranker" / "reranker_dataset_flat.jsonl"

def gen_rows():
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            contexts = row.get("s_contexts") or []
            # Fallback if someone already flattened to a single string:
            if isinstance(contexts, str):
                contexts = [contexts]
            for j, ctx in enumerate(contexts):
                out = {
                    # stable query id = report id + local index
                    "query_id": f"{row.get('s_report_id','UNK')}:{j:04d}",
                    "q_text": ctx,                          # the single context string
                    "candidate_id": row.get("c_clean_id"),  # stable code (TACTIC/TECHNIQUE/SOFTWARE…)
                    "candidate_name": row.get("c_name"),
                    "candidate_type": row.get("c_type"),
                    "label": int(row.get("label", 0)),      # 1 or 0
                    "cosine": row.get("cosine"),
                    # retain provenance
                    "source_name": row.get("s_name"),
                    "source_type": row.get("s_type"),
                    "report_id": row.get("s_report_id"),
                    "report_name": row.get("s_report_name"),
                }
                yield out

count = 0
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as w:
    for out in gen_rows():
        w.write(json.dumps(out, ensure_ascii=False) + "\n")
        count += 1

print(f"Wrote {count} rows to {dst}")
