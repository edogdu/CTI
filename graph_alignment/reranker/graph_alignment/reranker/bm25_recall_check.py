import json
from collections import defaultdict

gold_path = "C:/Users/shane/Downloads/CTI/graph_alignment/reranker/gold_id_unbalanced_nobom.jsonl"
cand_path = "data/reranker_pairs_enriched_v2.jsonl"

# Load original gold labels
gold_by_query = defaultdict(set)
for line in open(gold_path, 'r', encoding='utf-8'):
    r = json.loads(line)
    gold_by_query[r['query']].add(r['gold'])

print(f"Gold file: {len(gold_by_query)} unique queries, "
      f"{sum(len(v) for v in gold_by_query.values())} total gold labels")

# Load BM25 candidate pools
cand_by_query = defaultdict(set)
for line in open(cand_path, 'r', encoding='utf-8'):
    r = json.loads(line)
    qr = r.get('query_raw', '')
    cid = r.get('candidate_norm', '') or r.get('candidate_id', '')
    if qr:
        cand_by_query[qr].add(cid)

print(f"JSONL file: {len(cand_by_query)} unique queries")

# Measure BM25 recall
total_gold = 0
found_gold = 0
missed_gold = 0
missed_examples = []

for query, golds in gold_by_query.items():
    cands = cand_by_query.get(query, set())
    if not cands:
        continue
    for g in golds:
        total_gold += 1
        if any(g in c for c in cands):
            found_gold += 1
        else:
            missed_gold += 1
            if len(missed_examples) < 15:
                missed_examples.append((query[:80], g))

print(f"\nBM25 RECALL ANALYSIS")
print(f"Total gold labels checked: {total_gold}")
print(f"Found in BM25 top-20:     {found_gold} ({found_gold/total_gold*100:.1f}%)")
print(f"MISSED by BM25:           {missed_gold} ({missed_gold/total_gold*100:.1f}%)")
print(f"BM25 Recall@20:           {found_gold/total_gold*100:.1f}%")

if missed_examples:
    print(f"\nExamples of missed gold labels:")
    for q, g in missed_examples:
        print(f"  {g:<15} | {q}")
