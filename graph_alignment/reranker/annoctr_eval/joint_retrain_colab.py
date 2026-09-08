#!/usr/bin/env python3
"""
Option 2: Full Joint Retrain (Colab) — the final pre-declared attempt
=====================================================================
Faithfully reproduces best_two_stage_v2's curriculum with ONE change:
Stage 2 trains on CTI-HAL-train AND AnnoCTR-train combined.

  Stage 1: cross-encoder/ms-marco-MiniLM-L-6-v2
           -> tumeteor + zenodo combined (351,515 rows; the exact short-card
              files of the historical run), 3 epochs, lr 2e-5, bs 16
  Stage 2: stage-1 best -> [10,727 CTI-HAL train pairs (verbatim seed-42
           reconstruction) + 8,456 AnnoCTR train pairs], 2 epochs, lr 1e-5,
           union validation evaluator (CTI-HAL val + AnnoCTR train-val)

Then tuning-set readouts (AnnoCTR dev, CTI-HAL val), and finally THE ONE
PRE-DECLARED TEST LOOK: CTI-HAL test (146) scored once, verdict printed
against the bar of >=136/146. This is the last test look for this model
family. AnnoCTR test is NOT touched here (local step, only if the gate
passes; bar there: overall P@1 >= 72.9).

Colab (T4), after uploading joint_colab_bundle.zip:

    !unzip -o joint_colab_bundle.zip -d /content/joint
    !pip -q install sentence-transformers rank_bm25
    !python /content/joint/joint_retrain_colab.py

Outputs: /content/joint_results.json and /content/joint_candidate.zip
(archived regardless of verdict; becomes canonical ONLY on a full pass).
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
CONTENT = Path('/content') if Path('/content').exists() else BUNDLE.parent
REPO = CONTENT / 'CTI_repo'
RERANKER = REPO / 'graph_alignment' / 'reranker'
DATA = RERANKER / 'data'
WORK = CONTENT / 'joint_work'

SEED = 42
BATCH = 16
S1_LR, S1_EPOCHS = 2e-5, 3
S2_LR, S2_EPOCHS = 1e-5, 2
CTIHAL_TEST_BAR = 136
LOG = {'asserts': []}


def note(msg):
    print('  ' + msg)
    LOG['asserts'].append(msg)


def sh(cmd, cwd=None):
    print('  $', cmd)
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
        sys.exit(f'command failed: {cmd}')


# ---- verified verbatim CTI-HAL machinery (same code as the rehearsal kit;
# ---- reproduction of 1111/135/146 and the 10,727 pairs proven pre-ship) ----

def load_and_group(filepath):
    qd = defaultdict(lambda: {'query_raw': None, 'actor': None, 'candidates': [],
                              'positive_count': 0, 'negative_count': 0})
    for line in open(filepath, encoding='utf-8'):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        qn, qr = row.get('query_norm', ''), row.get('query_raw', '')
        if not qn or not qr:
            continue
        d = qd[qn]
        if d['query_raw'] is None:
            d['query_raw'] = qr
            d['actor'] = row.get('actor', 'unknown') or 'unknown'
        cid = row.get('candidate_norm') or row.get('candidate_id', '')
        d['candidates'].append({'text': row.get('candidate_text', ''),
                                'label': row.get('label', 0), 'id': cid})
        if row.get('label', 0) == 1:
            d['positive_count'] += 1
        else:
            d['negative_count'] += 1
    return dict(qd)


def finetune_splits(query_data, train_ratio=0.8, val_ratio=0.1):
    random.seed(SEED)
    import numpy as np
    np.random.seed(SEED)
    qba = defaultdict(list)
    for qn, d in query_data.items():
        if d['positive_count'] > 0:
            qba[d['actor']].append(qn)
    tr, va, te = [], [], []
    for actor, queries in qba.items():
        random.shuffle(queries)
        n = len(queries); nt = int(n * train_ratio); nv = int(n * val_ratio)
        tr.extend(queries[:nt]); va.extend(queries[nt:nt + nv])
        te.extend(queries[nt + nv:])
    random.shuffle(tr); random.shuffle(va); random.shuffle(te)
    return tr, va, te


def build_train_pairs(train_queries, query_data):
    """Continues the finetune_splits RNG stream (verbatim path)."""
    out = []
    for qn in train_queries:
        if qn not in query_data:
            continue
        d = query_data[qn]; qr = d['query_raw']
        pos = [c for c in d['candidates'] if c['label'] == 1]
        neg = [c for c in d['candidates'] if c['label'] == 0]
        for c in pos:
            out.append({'q': qr, 't': c['text'], 'y': 1.0})
        k = min(int(len(pos) * 2.5), len(neg))
        if k > 0:
            for c in random.sample(neg, k):
                out.append({'q': qr, 't': c['text'], 'y': 0.0})
    random.shuffle(out)
    return out


def val_eval_samples(query_data, val_keys, cap=None):
    samples = []
    for k in val_keys if cap is None else val_keys[:cap]:
        q = query_data[k]
        pos = [c['text'] for c in q['candidates'] if c['label'] == 1]
        neg = [c['text'] for c in q['candidates'] if c['label'] == 0][:10]
        if pos and neg:
            samples.append({'query': q['query_raw'], 'positive': pos,
                            'negative': neg})
    return samples


# ---------------------------------------------------------------------------

def fit_stage(base, examples, evaluator, epochs, lr, out_dir, tag):
    import torch
    import random as _r
    import numpy as np
    _r.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader
    ex = [InputExample(texts=[p['q'], p['t']], label=p['y']) for p in examples]
    print(f'  [{tag}] {len(ex)} pairs | {epochs} epoch(s) @ lr {lr}')
    model = CrossEncoder(str(base), num_labels=1, max_length=512)
    loader = DataLoader(ex, shuffle=True, batch_size=BATCH)
    steps = len(ex) // BATCH
    out_dir.mkdir(parents=True, exist_ok=True)
    model.fit(train_dataloader=loader, evaluator=evaluator, epochs=epochs,
              evaluation_steps=max(1, steps // 2),
              warmup_steps=int(steps * epochs * 0.1),
              output_path=str(out_dir / 'best'), save_best_model=True,
              optimizer_params={'lr': lr}, weight_decay=0.01,
              show_progress_bar=True)
    return out_dir / 'best'


def make_rerank_eval(samples, name):
    try:
        from sentence_transformers.cross_encoder.evaluation import (
            CERerankingEvaluator as RerankEval)
    except ImportError:
        from sentence_transformers.cross_encoder.evaluation import (
            CrossEncoderRerankingEvaluator as RerankEval)
    return RerankEval(samples, name=name)


def eval_annoctr_dev(ckpt, tag):
    sys.path.insert(0, str(BUNDLE))
    from annoctr_eval import CrossEncoderScorer, evaluate_pool
    queries = [json.loads(l) for l in
               open(BUNDLE / 'annoctr_queries_dev_mf.jsonl', encoding='utf-8')]
    hierarchy = json.load(open(BUNDLE / 'attack_hierarchy.json', encoding='utf-8'))
    rows = evaluate_pool(queries, 'pool_reranking', CrossEncoderScorer(str(ckpt)),
                         hierarchy, tag, progress_every=0)
    tech = [r for r in rows if r['category'] == 'technique']
    return {'overall_p1': round(sum(r['correct'] for r in rows) / len(rows), 4),
            'technique_p1': round(sum(r['correct'] for r in tech) / len(tech), 4)}


def eval_ctihal(ckpt, qd, keys):
    from sentence_transformers import CrossEncoder
    m = CrossEncoder(str(ckpt), max_length=512)
    c = 0
    for k in keys:
        q = qd[k]
        s = m.predict([[q['query_raw'], x['text']] for x in q['candidates']],
                      batch_size=64, show_progress_bar=False)
        top = max(range(len(s)), key=lambda i: s[i])
        if q['candidates'][top]['id'] in {x['id'] for x in q['candidates']
                                          if x['label'] == 1}:
            c += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    t0 = time.time()
    results = {'meta': {'seed': SEED, 'batch': BATCH,
                        'stage1': {'lr': S1_LR, 'epochs': S1_EPOCHS},
                        'stage2': {'lr': S2_LR, 'epochs': S2_EPOCHS},
                        'ctihal_test_bar': CTIHAL_TEST_BAR,
                        'timestamp': time.strftime('%F %T')}}

    print('[1/7] Repo (no LFS needed: stage-1 data are regular blobs)')
    if not REPO.exists():
        sh(f'GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --branch dataset-expansion '
           f'https://github.com/edogdu/CTI {REPO}')

    print('[2/7] Stage-1 data: tumeteor + zenodo (historical short-card files)')
    tum = DATA / 'reranker_pairs_enriched_tumeteor.jsonl'
    zen = DATA / 'reranker_pairs_enriched_zenodo.jsonl'
    n_tum = sum(1 for _ in open(tum, encoding='utf-8'))
    n_zen = sum(1 for _ in open(zen, encoding='utf-8'))
    assert n_tum == 187823, f'tumeteor rows {n_tum} != 187823'
    assert n_zen == 163692, f'zenodo rows {n_zen} != 163692'
    note(f'stage-1 sources verified: tumeteor {n_tum} + zenodo {n_zen} '
         f'= {n_tum + n_zen} rows (historical 351,515)')
    s1_file = WORK / 'stage1_combined.jsonl'
    WORK.mkdir(parents=True, exist_ok=True)
    with open(s1_file, 'w', encoding='utf-8') as f:
        for src in (tum, zen):
            for line in open(src, encoding='utf-8'):
                f.write(line)
    qd1 = load_and_group(str(s1_file))
    tr1, va1, te1 = finetune_splits(qd1)
    s1_pairs = build_train_pairs(tr1, qd1)
    note(f'stage-1: {len(qd1)} queries -> train {len(tr1)} / val {len(va1)} / '
         f'test {len(te1)} -> {len(s1_pairs)} training pairs')

    print('[3/7] Stage-2 data: CTI-HAL (verbatim 10,727) + AnnoCTR (8,456)')
    v2file = DATA / 'reranker_pairs_enriched_v2.jsonl'
    if not v2file.exists():
        print('  regenerating v2 pairs via repo enrich_candidates.py')
        sh(f'{sys.executable} enrich_candidates.py', cwd=RERANKER)
    qd2 = load_and_group(str(v2file))
    tr2, va2, te2 = finetune_splits(qd2)
    assert (len(tr2), len(va2), len(te2)) == (1111, 135, 146), \
        f'CTI-HAL split reproduction failed: {len(tr2)}/{len(va2)}/{len(te2)}'
    ctihal_pairs = build_train_pairs(tr2, qd2)
    npos = sum(1 for p in ctihal_pairs if p['y'] == 1.0)
    assert (len(ctihal_pairs), npos) == (10727, 3190), \
        f'10,727 reproduction failed: {len(ctihal_pairs)} ({npos} pos)'
    note('CTI-HAL split 1111/135/146 and 10,727 pairs reproduced')
    annoctr_pairs = [
        {'q': r['query_raw'], 't': r['candidate_text'], 'y': float(r['label'])}
        for r in map(json.loads,
                     open(BUNDLE / 'annoctr_stage3_train.jsonl', encoding='utf-8'))]
    assert len(annoctr_pairs) == 8456, f'AnnoCTR pairs {len(annoctr_pairs)} != 8456'
    joint = ctihal_pairs + annoctr_pairs
    note(f'joint stage-2 set: {len(joint)} pairs '
         f'({len(ctihal_pairs)} CTI-HAL + {len(annoctr_pairs)} AnnoCTR), '
         f'interleaved via shuffled DataLoader')

    if args.dry_run:
        print('[4-7/7] [dry] skipping training/eval phases')
        print(json.dumps(LOG['asserts'], indent=1))
        return

    print('[4/7] Stage 1 fit (~30 min on T4 — the long part)')
    ev1 = make_rerank_eval(val_eval_samples(qd1, va1), 'stage1val')
    s1_best = fit_stage('cross-encoder/ms-marco-MiniLM-L-6-v2', s1_pairs, ev1,
                        S1_EPOCHS, S1_LR, WORK / 'stage1', 'stage1')

    print('[5/7] Stage 2 joint fit (from stage-1 best)')
    union = val_eval_samples(qd2, va2)
    n_c = len(union)
    for line in open(BUNDLE / 'annoctr_stage3_val.jsonl', encoding='utf-8'):
        r = json.loads(line)
        if r['positives'] and r['negatives']:
            union.append({'query': r['query_raw'], 'positive': r['positives'],
                          'negative': r['negatives']})
    note(f'union stage-2 evaluator: {n_c} CTI-HAL-val + {len(union)-n_c} '
         f'AnnoCTR-trainval queries')
    ev2 = make_rerank_eval(union, 'unionval')
    joint_best = fit_stage(s1_best, joint, ev2, S2_EPOCHS, S2_LR,
                           WORK / 'joint', 'stage2-joint')

    print('[6/7] Tuning-set readouts (context)')
    a = eval_annoctr_dev(joint_best, 'joint')
    cv = eval_ctihal(joint_best, qd2, va2)
    results['annoctr_dev'] = a
    results['ctihal_val'] = {'correct': cv, 'n': len(va2),
                             'p1': round(cv / len(va2), 4)}
    print(f'  [joint] AnnoCTR dev {a["overall_p1"]:.4f} '
          f'(tech {a["technique_p1"]:.4f}) | CTI-HAL val {cv}/{len(va2)} '
          f'= {cv/len(va2):.4f}')

    print('[7/7] THE PRE-DECLARED TEST GATE: CTI-HAL test (146), scored ONCE')
    ct = eval_ctihal(joint_best, qd2, te2)
    results['ctihal_test'] = {'correct': ct, 'n': 146,
                              'p1': round(ct / 146, 4),
                              'bar': CTIHAL_TEST_BAR,
                              'pass': ct >= CTIHAL_TEST_BAR}
    results['test_looks'] = {'ctihal_test': 1, 'annoctr_test': 0}
    verdict = 'PASS' if ct >= CTIHAL_TEST_BAR else 'FAIL'
    print(f'\n  ====== CTI-HAL test: {ct}/146 = {ct/146:.4f} '
          f'(bar >= {CTIHAL_TEST_BAR}) -> {verdict} ======')
    if verdict == 'PASS':
        print('  Next: local AnnoCTR test certification '
              '(bar: overall P@1 >= 0.729).')
    else:
        print('  Pre-declared fallback applies permanently: '
              'two-checkpoint framing; candidate archived, never canonical.')

    results['asserts'] = LOG['asserts']
    with open(CONTENT / 'joint_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1, sort_keys=True)
    z = shutil.make_archive(str(CONTENT / 'joint_candidate'), 'zip', joint_best)
    print(f'  wrote {CONTENT / "joint_results.json"}')
    print(f'  wrote {z}')
    print(f'Done in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
