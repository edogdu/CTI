#!/usr/bin/env python3
"""
AnnoCTR Unseen-Eval Runner for the CTI-to-ATT&CK Cross-Encoder Reranker
=======================================================================
Scores the query files produced by annoctr_to_reranker.py with the
best_two_stage_v2 checkpoint and reports metrics in the conventions of
v2_reeval.py / hierarchical_postprocess.py.

Two candidate-pool variants, both reported:
  * reranking pool  -- gold union BM25 top-20 (gold guaranteed present).
                       Comparable to the canonical 146-query CTI-HAL eval,
                       where every query's pool contains its gold.
  * deploy pool     -- pure BM25 top-20 (gold NOT injected). The honest
                       end-to-end deployment measurement; bounded above by
                       the BM25 recall ceiling, which is also reported.

Two scoring layers, both reported (mirrors the 94.52% / 95.21% convention):
  * cross-encoder layer      -- top-1 in gold set.
  * hierarchical layer (+)   -- top-1 expanded with its parent technique and
                                associated tactic(s) per ATT&CK's ontology
                                (same expansion as hierarchical_postprocess.py)
                                before checking against the gold set.

Baselines included: BM25 rank-1 (deploy pool), random-expectation per pool.

Usage:
  # plumbing check without the model (deterministic stand-in scorer):
  python annoctr_eval.py --queries annoctr_eval/annoctr_queries_test.jsonl \
      --hierarchy annoctr_eval/attack_hierarchy.json --self-test

  # real run:
  python annoctr_eval.py --queries annoctr_eval/annoctr_queries_test.jsonl \
      --hierarchy annoctr_eval/attack_hierarchy.json \
      --checkpoint <path>/checkpoints/best_two_stage_v2 \
      --out-dir annoctr_eval/results
"""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------

class SelfTestScorer:
    """Deterministic stand-in: score = hash(query || candidate) -> [0, 1).
    Validates plumbing, output shapes, and determinism only."""
    name = 'self-test-hash'

    def predict(self, pairs, **kw):
        out = []
        for q, c in pairs:
            h = hashlib.sha256((q + '\x00' + c).encode('utf-8')).digest()
            out.append(int.from_bytes(h[:8], 'big') / 2**64)
        return out


class CrossEncoderScorer:
    def __init__(self, checkpoint):
        from sentence_transformers import CrossEncoder  # lazy import
        self.model = CrossEncoder(checkpoint, max_length=512)
        self.name = f'cross-encoder:{checkpoint}'

    def predict(self, pairs, **kw):
        return self.model.predict(pairs, batch_size=32,
                                  show_progress_bar=False).tolist()


# --------------------------------------------------------------------------
# Hierarchical expansion (mirrors hierarchical_postprocess.py)
# --------------------------------------------------------------------------

def expand_prediction(pred_id, hierarchy):
    expanded = {pred_id}
    sub_to_parent = hierarchy['sub_to_parent']
    tech_to_tactics = hierarchy['tech_to_tactics']
    if pred_id in sub_to_parent:
        parent = sub_to_parent[pred_id]
        expanded.add(parent)
        expanded.update(tech_to_tactics.get(parent, []))
    expanded.update(tech_to_tactics.get(pred_id, []))
    return expanded


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def evaluate_pool(queries, pool_key, scorer, hierarchy, tag, progress_every=200):
    rows = []
    t0 = time.time()
    n = len(queries)
    for i, q in enumerate(queries):
        pool = q[pool_key]
        gold = set(q['gold_ids'])
        if not pool:
            continue
        pairs = [[q['query_raw'], c['text']] for c in pool]
        scores = scorer.predict(pairs)
        ranked = sorted(zip(pool, scores), key=lambda x: -x[1])
        ranked_ids = [c['id'] for c, _ in ranked]
        top1, top1_score = ranked_ids[0], float(ranked[0][1])

        correct = top1 in gold
        correct_plus = bool(expand_prediction(top1, hierarchy) & gold)
        hit3 = any(r in gold for r in ranked_ids[:3])
        hit5 = any(r in gold for r in ranked_ids[:5])
        rank = next((k + 1 for k, r in enumerate(ranked_ids) if r in gold), None)
        rr = 1.0 / rank if rank else 0.0
        gold_in_pool = any(c['label'] for c in pool)
        bm25_top1 = q['pool_deploy'][0]['id'] if q['pool_deploy'] else None

        rows.append({
            'query_norm': q['query_norm'], 'document': q['document'],
            'category': q['category'], 'n_mentions': q['n_mentions'],
            'gold_ids': ';'.join(sorted(gold)),
            'pool': tag, 'pool_size': len(pool),
            'gold_in_pool': gold_in_pool,
            'predicted': top1, 'predicted_score': round(top1_score, 6),
            'correct': correct, 'correct_plus': correct_plus,
            'hit3': hit3, 'hit5': hit5,
            'gold_rank': rank if rank else '',
            'rr': round(rr, 6),
            'bm25_top1': bm25_top1,
            'bm25_top1_correct': bm25_top1 in gold,
            'bm25_top1_correct_plus': bool(
                expand_prediction(bm25_top1, hierarchy) & gold) if bm25_top1 else False,
        })
        if progress_every and (i + 1) % progress_every == 0:
            p1 = sum(r['correct'] for r in rows) / len(rows)
            print(f"    [{tag}] {i+1}/{n}  P@1 so far {p1:.4f}  "
                  f"({time.time()-t0:.0f}s)")
    return rows


def aggregate(rows, keyfn=None):
    if keyfn is None:
        groups = {'all': rows}
    else:
        groups = defaultdict(list)
        for r in rows:
            groups[keyfn(r)].append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        n = len(rs)
        out[k] = {
            'n': n,
            'p1': round(sum(r['correct'] for r in rs) / n, 4),
            'p1_plus': round(sum(r['correct_plus'] for r in rs) / n, 4),
            'hit3': round(sum(r['hit3'] for r in rs) / n, 4),
            'hit5': round(sum(r['hit5'] for r in rs) / n, 4),
            'mrr': round(sum(r['rr'] for r in rs) / n, 4),
            'gold_in_pool_rate': round(sum(r['gold_in_pool'] for r in rs) / n, 4),
            'bm25_top1_p1': round(sum(r['bm25_top1_correct'] for r in rs) / n, 4),
            'bm25_top1_p1_plus': round(
                sum(r['bm25_top1_correct_plus'] for r in rs) / n, 4),
        }
    return out


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', required=True)
    ap.add_argument('--hierarchy', required=True)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--pools', nargs='+',
                    default=['reranking', 'deploy'],
                    choices=['reranking', 'deploy'])
    ap.add_argument('--limit', type=int, default=0,
                    help='evaluate only the first N queries (smoke run)')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    if not args.self_test and not args.checkpoint:
        ap.error('provide --checkpoint or use --self-test')

    queries = [json.loads(l) for l in open(args.queries, encoding='utf-8')]
    if args.limit:
        queries = queries[:args.limit]
    hierarchy = json.load(open(args.hierarchy, encoding='utf-8'))
    split = queries[0]['split'] if queries else 'unknown'

    scorer = SelfTestScorer() if args.self_test else \
        CrossEncoderScorer(args.checkpoint)
    mode = 'SELF-TEST (stand-in scorer — numbers are meaningless)' \
        if args.self_test else 'REAL'
    print(f"AnnoCTR eval | split={split} | queries={len(queries)} | mode={mode}")

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(args.queries).parent / ('results_selftest' if args.self_test
                                     else 'results')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary = {}
    for pool in args.pools:
        key = f'pool_{pool}'
        print(f"  Scoring pool: {pool}")
        rows = evaluate_pool(queries, key, scorer, hierarchy, pool)
        all_rows.extend(rows)
        summary[pool] = {
            'overall': aggregate(rows)['all'],
            'by_category': aggregate(rows, lambda r: r['category']),
            'by_document': aggregate(rows, lambda r: r['document']),
        }
        o = summary[pool]['overall']
        print(f"    -> P@1 {o['p1']:.4f} | P@1+ {o['p1_plus']:.4f} | "
              f"Hit@3 {o['hit3']:.4f} | MRR {o['mrr']:.4f} | "
              f"gold-in-pool {o['gold_in_pool_rate']:.2%} | "
              f"BM25-rank1 P@1 {o['bm25_top1_p1']:.4f}")

    results = {
        'meta': {
            'split': split, 'n_queries': len(queries),
            'scorer': scorer.name, 'self_test': args.self_test,
            'queries_file': str(args.queries),
            'queries_sha256_16': hashlib.sha256(
                open(args.queries, 'rb').read()).hexdigest()[:16],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'summary': summary,
    }
    jpath = out_dir / f'annoctr_results_{split}.json'
    with open(jpath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1, sort_keys=True)

    cpath = out_dir / f'annoctr_per_query_{split}.csv'
    if all_rows:
        with open(cpath, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    print(f"  Wrote {jpath}")
    print(f"  Wrote {cpath}")


if __name__ == '__main__':
    main()
