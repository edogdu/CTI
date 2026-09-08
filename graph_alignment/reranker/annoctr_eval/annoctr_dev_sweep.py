#!/usr/bin/env python3
"""
AnnoCTR Dev Sweep: 8-config controlled comparison of query/candidate fixes
==========================================================================
Builds (unless present) and scores the DEV split under all combinations of:
  cw = --context-window      (neighboring sentences added to the query)
  mf = --mention-first       (annotated mention span(s) prepended)
  ep = --enrich-procedures   (procedure example appended to technique cards)

Controlled design: every variant reuses the BASELINE's candidate pools
(--freeze-pools-from), so all 8 configs score the *same* candidate IDs per
query — only the query/candidate TEXTS differ. Differences in P@1 are
therefore attributable to the text changes, not to retrieval drift.
A separate "BM25 R@20" column reports what each variant's retrieval recall
WOULD be if pools were rebuilt (the system-level view); the final locked
configuration gets consistently rebuilt pools in the one-time test run.

Guardrail: refuses to touch the test split unless --i-am-certifying-final.

Usage (self-test plumbing, no model):
  python annoctr_dev_sweep.py --annoctr-dir <AnnoCTR> --stix <v14.json> --self-test
Real run:
  python annoctr_dev_sweep.py --annoctr-dir <AnnoCTR> --stix <v14.json> \
      --checkpoint <path>/checkpoints/best_two_stage_v2
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from annoctr_eval import SelfTestScorer, CrossEncoderScorer, evaluate_pool  # noqa: E402
from annoctr_to_reranker import build_v14  # noqa: E402

CONFIGS = [
    ('base',      []),
    ('cw',        ['--context-window']),
    ('mf',        ['--mention-first']),
    ('ep',        ['--enrich-procedures']),
    ('cw_mf',     ['--context-window', '--mention-first']),
    ('cw_ep',     ['--context-window', '--enrich-procedures']),
    ('mf_ep',     ['--mention-first', '--enrich-procedures']),
    ('cw_mf_ep',  ['--context-window', '--mention-first', '--enrich-procedures']),
]


def ensure_built(args, qdir):
    qdir.mkdir(parents=True, exist_ok=True)
    base_path = qdir / f'annoctr_queries_{args.split}_base.jsonl'
    builder = str(HERE / 'annoctr_to_reranker.py')

    def run(extra, suffix, freeze=None):
        cmd = [sys.executable, builder,
               '--annoctr-dir', args.annoctr_dir, '--stix', args.stix,
               '--splits', args.split, '--out-dir', str(qdir),
               '--suffix', suffix] + extra
        if freeze:
            cmd += ['--freeze-pools-from', str(freeze)]
        print('  [build]', suffix or '_base')
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:]); print(r.stderr[-2000:])
            sys.exit(f'builder failed for {suffix}')

    if not base_path.exists() or args.rebuild:
        run([], '_base')
    for name, flags in CONFIGS[1:]:
        p = qdir / f'annoctr_queries_{args.split}_{name}.jsonl'
        if not p.exists() or args.rebuild:
            run(flags, f'_{name}', freeze=base_path)


def agg(rows):
    n = len(rows)
    return {
        'n': n,
        'p1': sum(r['correct'] for r in rows) / n,
        'p1_plus': sum(r['correct_plus'] for r in rows) / n,
        'hit3': sum(r['hit3'] for r in rows) / n,
        'mrr': sum(r['rr'] for r in rows) / n,
    }


def bm25_recall(queries, lookup, top_k=20):
    """System-view retrieval recall for this variant's query text against
    this variant's candidate texts (fresh index, not the frozen pools)."""
    from rank_bm25 import BM25Okapi
    ids = sorted(lookup.keys())
    texts = [lookup[i]['enriched_text'] for i in ids]
    bm25 = BM25Okapi([t.lower().split() for t in texts], k1=1.5, b=0.75)
    hits = 0
    for q in queries:
        scores = bm25.get_scores(q['query_raw'].lower().split())
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        got = {ids[i] for i in order}
        if got & set(q['gold_ids']):
            hits += 1
    return hits / len(queries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annoctr-dir', required=True)
    ap.add_argument('--stix', required=True)
    ap.add_argument('--split', default='dev')
    ap.add_argument('--queries-dir', default='sweep_dev')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--skip-retrieval-column', action='store_true')
    ap.add_argument('--i-am-certifying-final', action='store_true')
    args = ap.parse_args()

    if 'test' in args.split and not args.i_am_certifying_final:
        sys.exit('REFUSED: sweeping the test split is not allowed. '
                 'All tuning runs on dev. (Override only for the one final '
                 'certification run: --i-am-certifying-final)')
    if not args.self_test and not args.checkpoint:
        ap.error('provide --checkpoint or use --self-test')

    qdir = Path(args.queries_dir)
    print(f'== Sweep on split "{args.split}" | mode='
          + ('SELF-TEST (numbers meaningless)' if args.self_test else 'REAL'))
    ensure_built(args, qdir)

    hierarchy = json.load(open(qdir / 'attack_hierarchy.json', encoding='utf-8'))
    scorer = SelfTestScorer() if args.self_test else CrossEncoderScorer(args.checkpoint)

    # retrieval column needs the two candidate universes (ep off / ep on)
    lookups = {}
    if not args.skip_retrieval_column:
        lookups[False] = build_v14(args.stix, enrich_procedures=False)[0]
        lookups[True] = build_v14(args.stix, enrich_procedures=True)[0]

    results = {}
    for name, flags in CONFIGS:
        qfile = qdir / f'annoctr_queries_{args.split}_{name}.jsonl'
        queries = [json.loads(l) for l in open(qfile, encoding='utf-8')]
        t0 = time.time()
        rows = evaluate_pool(queries, 'pool_reranking', scorer, hierarchy,
                             name, progress_every=0)
        dt = time.time() - t0
        by_cat = defaultdict(list)
        for r in rows:
            by_cat[r['category']].append(r)
        entry = {
            'flags': flags,
            'overall': agg(rows),
            'technique': agg(by_cat['technique']),
            'software': agg(by_cat['software']),
            'tactic': agg(by_cat['tactic']),
            'mixed': agg(by_cat['mixed']),
            'seconds': round(dt, 1),
        }
        if not args.skip_retrieval_column:
            entry['bm25_r20'] = bm25_recall(
                queries, lookups['--enrich-procedures' in flags])
        results[name] = entry
        print(f'  [{name:9s}] overall P@1 {entry["overall"]["p1"]:.4f} | '
              f'technique P@1 {entry["technique"]["p1"]:.4f} | {dt:.0f}s')

    base = results['base']
    print()
    hdr = ('config    | ovr P@1  d(base) | tech P@1  d(base) | tech P@1+ '
           '| tech H@3 | ovr P@1+ | ovr MRR | BM25R@20')
    print(hdr); print('-' * len(hdr))
    lines = []
    for name, _ in CONFIGS:
        r = results[name]
        d_o = r['overall']['p1'] - base['overall']['p1']
        d_t = r['technique']['p1'] - base['technique']['p1']
        line = ('%-9s | %7.4f  %+6.4f | %8.4f  %+6.4f | %9.4f | %8.4f | '
                '%8.4f | %7.4f | %s' % (
                    name, r['overall']['p1'], d_o, r['technique']['p1'], d_t,
                    r['technique']['p1_plus'], r['technique']['hit3'],
                    r['overall']['p1_plus'], r['overall']['mrr'],
                    ('%8.4f' % r['bm25_r20']) if 'bm25_r20' in r else '     n/a'))
        print(line); lines.append(line)

    out = qdir / f'sweep_results_{args.split}.json'
    payload = {'meta': {'split': args.split, 'scorer': scorer.name,
                        'self_test': args.self_test,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
               'results': results}
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    with open(qdir / f'sweep_table_{args.split}.txt', 'w', encoding='utf-8') as f:
        f.write(hdr + '\n' + '-' * len(hdr) + '\n' + '\n'.join(lines) + '\n')
    print(f'\nWrote {out}')
    print(f'Wrote {qdir / ("sweep_table_%s.txt" % args.split)}')


if __name__ == '__main__':
    main()
