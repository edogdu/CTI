#!/usr/bin/env python3
"""
Fix 4, part 1: AnnoCTR Stage-3 Training-Pair Generator
======================================================
Builds the stage-3 fine-tuning data from the AnnoCTR TRAIN split, in the
locked mention-first (mf) query format, mirroring the stage-2 recipe of
finetune_production.py:

  * pairs schema identical to reranker training data
  * negatives = BM25 top-20 minus golds, sampled at 2.5x positives
    (deterministic per-query RNG, seed 42)
  * candidate cards = standard "{ID} - {Name}: {desc<=1000}" (ep was
    rejected by the dev sweep; no Example: suffix anywhere)
  * training-time validation = held-out TRAIN documents (last 10% by
    sorted doc name), NOT dev -- dev stays a pure post-training check

Safety rails baked in:
  * asserts train docs are fully disjoint from dev and test docs (loads
    all three splits and proves it; refuses to write outputs otherwise)
  * asserts no group links or unresolvable golds leak into pairs
  * deterministic: identical reruns produce byte-identical files

Outputs (into --out-dir, default annoctr_eval\\stage3):
  annoctr_stage3_train.jsonl      flat training pairs
  annoctr_stage3_val.jsonl        val queries with pools (for CERerankingEvaluator)
  stage3_provenance.json          counts, hashes, disjointness proof
  stage3_colab_bundle.zip         everything Colab needs in one upload:
                                  the two files above + annoctr_queries_dev_mf.jsonl
                                  + attack_hierarchy.json + annoctr_eval.py
                                  + stage3_annoctr_colab.py + this provenance

Run from the reranker directory:
  python annoctr_eval\\annoctr_train_pairs.py --annoctr-dir <ANNOCTR>\\AnnoCTR ^
      --stix enterprise-attack-v14.json
"""

import argparse
import hashlib
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from annoctr_to_reranker import (  # noqa: E402
    build_v14, parse_label_link, clean_text, load_split, sha256_head,
    BM25_TOP_K)

SEED = 42
NEG_RATIO = 2.5
VAL_DOC_FRACTION = 0.10


def group_queries(recs, lookup, revoked_map, split):
    """Group mentions into mf-format queries; returns list of dicts with
    qid, query_raw (mf), gold_ids, document. Mirrors the eval builder's
    grouping exactly (same base_norm key, same crosswalk rules)."""
    stats = Counter()
    groups = {}
    for r in recs:
        stats['mentions_total'] += 1
        kind, aid = parse_label_link(r.get('label_link', ''))
        if kind is None:
            stats['non_mitre'] += 1
            continue
        if kind == 'group':
            stats['group_excluded'] += 1
            continue
        gold = None
        if aid in lookup:
            gold = aid
        elif aid in revoked_map and revoked_map[aid]:
            gold = revoked_map[aid]
            stats['mapped_forward'] += 1
        else:
            stats['unresolvable'] += 1
            continue
        base_raw = clean_text(
            f"{r.get('context_left','')} {r.get('mention','')} {r.get('context_right','')}")
        if not base_raw:
            stats['empty_query'] += 1
            continue
        base_norm = base_raw.lower()
        doc = r.get('document', 'unknown')
        key = (doc, base_norm)
        g = groups.setdefault(key, {
            'base_raw': base_raw, 'base_norm': base_norm, 'document': doc,
            'gold_ids': set(), 'mentions': set(), 'kinds': Counter()})
        g['gold_ids'].add(gold)
        m = clean_text(r.get('mention', ''))
        if m:
            g['mentions'].add(m)
        g['kinds'][kind] += 1
        stats['mentions_kept'] += 1

    out = []
    for (doc, base_norm), g in sorted(groups.items()):
        qid = hashlib.sha1((doc + '|' + base_norm).encode('utf-8')).hexdigest()[:16]
        mentions = sorted(g['mentions'])
        query_raw = '; '.join(mentions) + ' — ' + g['base_raw']   # mf format
        out.append({
            'qid': qid, 'query_raw': query_raw,
            'query_norm': query_raw.lower(),
            'document': doc, 'split': split,
            'category': list(g['kinds'])[0] if len(g['kinds']) == 1 else 'mixed',
            'gold_ids': sorted(g['gold_ids']),
        })
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annoctr-dir', required=True)
    ap.add_argument('--stix', required=True)
    ap.add_argument('--out-dir', default=str(HERE / 'stage3'))
    ap.add_argument('--dev-mf-queries',
                    default=str(HERE / 'sweep_dev' / 'annoctr_queries_dev_mf.jsonl'))
    ap.add_argument('--hierarchy',
                    default=str(HERE / 'sweep_dev' / 'attack_hierarchy.json'))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('[1/6] ATT&CK v14 lookup (standard cards, no ep)')
    lookup, revoked_map, _ = build_v14(args.stix, enrich_procedures=False)
    print(f'      {len(lookup)} active candidates')

    print('[2/6] Loading splits and proving document disjointness')
    docs = {}
    recs_by_split = {}
    for split in ('train', 'dev', 'test'):
        recs, _ = load_split(args.annoctr_dir, split)
        recs_by_split[split] = recs
        docs[split] = {r.get('document') for r in recs}
    inter_td = docs['train'] & docs['test']
    inter_tv = docs['train'] & docs['dev']
    inter_vt = docs['dev'] & docs['test']
    print(f"      docs: train {len(docs['train'])} dev {len(docs['dev'])} "
          f"test {len(docs['test'])}")
    print(f"      overlaps: train-test {len(inter_td)}, train-dev {len(inter_tv)}, "
          f"dev-test {len(inter_vt)}")
    assert not inter_td and not inter_tv and not inter_vt, \
        'DOCUMENT OVERLAP DETECTED - refusing to build training data'

    print('[3/6] Grouping train mentions into mf queries')
    queries, stats = group_queries(recs_by_split['train'], lookup, revoked_map, 'train')
    print(f"      {stats['mentions_total']} mentions -> {stats['mentions_kept']} kept "
          f"({stats['group_excluded']} group-excluded, "
          f"{stats['unresolvable']} unresolvable) -> {len(queries)} queries")
    cat = Counter(q['category'] for q in queries)
    print(f"      categories: {dict(cat)}")

    # held-out validation documents: deterministic stride over the sorted doc
    # list (vendor-diverse; tail-of-list selection would have grabbed a single
    # query-dense vendor cluster)
    train_docs_sorted = sorted({q['document'] for q in queries})
    n_val_docs = max(2, int(len(train_docs_sorted) * VAL_DOC_FRACTION))
    stride = max(1, len(train_docs_sorted) // n_val_docs)
    val_docs = set(train_docs_sorted[::stride][:n_val_docs])
    n_val_q = sum(1 for q in queries if q['document'] in val_docs)
    print(f"[4/6] Held-out training-val docs ({len(val_docs)}, stride {stride}): "
          f"{n_val_q}/{len(queries)} queries "
          f"({100*n_val_q/len(queries):.1f}% of query mass)")
    if n_val_q > 0.15 * len(queries):
        print('      WARNING: val query mass exceeds 15% — check doc distribution')

    print('[5/6] BM25 negative mining (top-20 minus golds, 2.5x, seed 42)')
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        sys.exit('[FAIL] rank_bm25 not installed. Run: pip install rank_bm25')
    corpus_ids = sorted(lookup.keys())
    corpus_texts = [lookup[i]['enriched_text'] for i in corpus_ids]
    bm25 = BM25Okapi([t.lower().split() for t in corpus_texts], k1=1.5, b=0.75)

    train_pairs, val_queries = [], []
    n_pos = n_neg = 0
    for q in queries:
        golds = set(q['gold_ids'])
        scores = bm25.get_scores(q['query_raw'].lower().split())
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:BM25_TOP_K]
        pool_ids = [corpus_ids[i] for i in order]
        negatives = [c for c in pool_ids if c not in golds]
        if q['document'] in val_docs:
            # full pool record for CERerankingEvaluator (pos + up to 10 negs)
            val_queries.append({
                'qid': q['qid'], 'query_raw': q['query_raw'],
                'document': q['document'], 'category': q['category'],
                'positives': [lookup[g]['enriched_text'] for g in sorted(golds)],
                'negatives': [lookup[c]['enriched_text'] for c in negatives[:10]],
            })
            continue
        rng = random.Random(f'{SEED}:{q["qid"]}')
        k = min(int(len(golds) * NEG_RATIO), len(negatives))
        sampled = rng.sample(negatives, k) if k > 0 else []
        for g in sorted(golds):
            train_pairs.append({
                'qid': q['qid'], 'query_raw': q['query_raw'],
                'query_norm': q['query_norm'], 'candidate_id': g,
                'candidate_norm': g, 'candidate_text': lookup[g]['enriched_text'],
                'label': 1, 'actor': f"annoctr:{q['document']}"})
            n_pos += 1
        for c in sorted(sampled):
            train_pairs.append({
                'qid': q['qid'], 'query_raw': q['query_raw'],
                'query_norm': q['query_norm'], 'candidate_id': c,
                'candidate_norm': c, 'candidate_text': lookup[c]['enriched_text'],
                'label': 0, 'actor': f"annoctr:{q['document']}"})
            n_neg += 1

    assert all(p['label'] in (0, 1) for p in train_pairs)
    print(f"      train pairs: {len(train_pairs)} "
          f"({n_pos} pos + {n_neg} neg, ratio {n_neg/max(n_pos,1):.2f}) | "
          f"val queries: {len(val_queries)}")

    tpath = out_dir / 'annoctr_stage3_train.jsonl'
    with open(tpath, 'w', encoding='utf-8') as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False, sort_keys=True) + '\n')
    vpath = out_dir / 'annoctr_stage3_val.jsonl'
    with open(vpath, 'w', encoding='utf-8') as f:
        for v in val_queries:
            f.write(json.dumps(v, ensure_ascii=False, sort_keys=True) + '\n')

    prov = {
        'seed': SEED, 'neg_ratio': NEG_RATIO,
        'stix_sha256_16': sha256_head(args.stix),
        'disjointness': {'train_docs': len(docs['train']),
                         'dev_docs': len(docs['dev']),
                         'test_docs': len(docs['test']),
                         'overlaps': 0},
        'mention_stats': dict(stats),
        'queries': len(queries), 'categories': dict(cat),
        'val_docs': sorted(val_docs),
        'train_pairs': {'total': len(train_pairs), 'pos': n_pos, 'neg': n_neg},
        'files': {'train': sha256_head(tpath), 'val': sha256_head(vpath)},
    }
    with open(out_dir / 'stage3_provenance.json', 'w', encoding='utf-8') as f:
        json.dump(prov, f, indent=1, sort_keys=True)

    print('[6/6] Building the one-upload Colab bundle')
    bundle = out_dir / 'stage3_colab_bundle.zip'
    needed = [
        (tpath, 'annoctr_stage3_train.jsonl'),
        (vpath, 'annoctr_stage3_val.jsonl'),
        (out_dir / 'stage3_provenance.json', 'stage3_provenance.json'),
        (Path(args.dev_mf_queries), 'annoctr_queries_dev_mf.jsonl'),
        (Path(args.hierarchy), 'attack_hierarchy.json'),
        (HERE / 'annoctr_eval.py', 'annoctr_eval.py'),
        (HERE / 'stage3_annoctr_colab.py', 'stage3_annoctr_colab.py'),
    ]
    missing = [str(src) for src, _ in needed if not src.exists()]
    if missing:
        sys.exit('[FAIL] bundle inputs missing:\n  ' + '\n  '.join(missing))
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        for src, name in needed:
            z.write(src, name)
    print(f'      wrote {bundle} '
          f'({bundle.stat().st_size/1e6:.1f} MB) — upload THIS ONE FILE to Colab')


if __name__ == '__main__':
    main()
