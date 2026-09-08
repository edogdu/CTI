#!/usr/bin/env python3
"""
Fix 4, part 2: Stage-3 Fine-Tuning on AnnoCTR (Colab)
=====================================================
Run inside Google Colab (GPU runtime) AFTER uploading and unzipping
stage3_colab_bundle.zip into /content. One cell:

    !unzip -o stage3_colab_bundle.zip -d /content/stage3
    !pip -q install sentence-transformers rank_bm25
    !python /content/stage3/stage3_annoctr_colab.py

What it does, in order:
  1. Clones edogdu/CTI (dataset-expansion) and LFS-pulls ONLY the
     best_two_stage_v2 checkpoint (~91 MB).
  2. Trains stage 3: 1 epoch @ lr 5e-6, batch 16, warmup 10%, weight
     decay 0.01, seed 42 -- the exact finetune_production.py recipe with
     the curriculum's next lr halving. Val = held-out TRAIN docs.
     Saves checkpoints/annoctr_stage3_v1/{best,last}. The base
     checkpoint is never modified.
  3. Eval A -- AnnoCTR dev (mf format), reranking pool: base vs adapted.
  4. Eval B -- forgetting probe: regenerates the CTI-HAL v2 pairs with the
     repo's own enrich_candidates.py, reproduces the canonical seed-42
     146-query test split via v2_reeval, scores base (must be 138/146)
     and adapted.
  5. Writes /content/stage3_results.json and zips the adapted checkpoint
     to /content/annoctr_stage3_v1_best.zip for download.

Flags: --dry-run (data plumbing only, no training/eval, no GPU needed).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
CONTENT = Path('/content') if Path('/content').exists() else BUNDLE.parent
REPO = CONTENT / 'CTI_repo'
RERANKER = REPO / 'graph_alignment' / 'reranker'
BASE_CKPT = RERANKER / 'checkpoints' / 'best_two_stage_v2'
OUT_CKPT = CONTENT / 'checkpoints' / 'annoctr_stage3_v1'
SEED = 42
LR = 5e-6
EPOCHS = 1
BATCH = 16


def sh(cmd, cwd=None):
    print('  $', cmd)
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
        sys.exit(f'command failed: {cmd}')
    return r


def step1_get_repo_and_checkpoint():
    print('[1/5] Repo + checkpoint')
    if not REPO.exists():
        sh(f'GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --branch dataset-expansion '
           f'https://github.com/edogdu/CTI {REPO}')
    sh('git lfs install --skip-smudge', cwd=REPO)
    sh('git lfs pull -I "graph_alignment/reranker/checkpoints/best_two_stage_v2/*"',
       cwd=REPO)
    sz = (BASE_CKPT / 'model.safetensors').stat().st_size
    assert sz > 80_000_000, f'checkpoint pull failed (size {sz})'
    print(f'      checkpoint ready ({sz/1e6:.1f} MB)')


def load_train_examples():
    from sentence_transformers import InputExample
    examples = []
    pos = neg = 0
    for line in open(BUNDLE / 'annoctr_stage3_train.jsonl', encoding='utf-8'):
        r = json.loads(line)
        examples.append(InputExample(
            texts=[r['query_raw'], r['candidate_text']], label=float(r['label'])))
        if r['label'] == 1:
            pos += 1
        else:
            neg += 1
    print(f'      train pairs: {len(examples)} ({pos} pos / {neg} neg)')
    return examples


def load_val_samples():
    samples = []
    for line in open(BUNDLE / 'annoctr_stage3_val.jsonl', encoding='utf-8'):
        r = json.loads(line)
        if r['positives'] and r['negatives']:
            samples.append({'query': r['query_raw'],
                            'positive': r['positives'],
                            'negative': r['negatives']})
    print(f'      val queries for CERerankingEvaluator: {len(samples)}')
    return samples


def step2_train(dry):
    print('[2/5] Stage-3 training '
          f'(lr {LR}, epochs {EPOCHS}, bs {BATCH}, seed {SEED})')
    examples = None
    if not dry:
        import random as _random
        import numpy as np
        import torch
        _random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
        from sentence_transformers import CrossEncoder
        try:
            from sentence_transformers.cross_encoder.evaluation import (
                CERerankingEvaluator as RerankEval)
        except ImportError:  # renamed in newer sentence-transformers
            from sentence_transformers.cross_encoder.evaluation import (
                CrossEncoderRerankingEvaluator as RerankEval)
        from torch.utils.data import DataLoader
        examples = load_train_examples()
        evaluator = RerankEval(load_val_samples(), name='stage3val')
        model = CrossEncoder(str(BASE_CKPT), num_labels=1, max_length=512)
        loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
        steps = len(examples) // BATCH
        OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
        model.fit(train_dataloader=loader,
                  evaluator=evaluator,
                  epochs=EPOCHS,
                  evaluation_steps=max(1, steps // 2),
                  warmup_steps=int(steps * 0.1),
                  output_path=str(OUT_CKPT / 'best'),
                  save_best_model=True,
                  optimizer_params={'lr': LR},
                  weight_decay=0.01,
                  show_progress_bar=True)
        model.save(str(OUT_CKPT / 'last'))
        print(f'      saved {OUT_CKPT}/best and /last')
    else:
        n = sum(1 for _ in open(BUNDLE / 'annoctr_stage3_train.jsonl'))
        v = sum(1 for _ in open(BUNDLE / 'annoctr_stage3_val.jsonl'))
        print(f'      [dry] would train on {n} pairs, validate on {v} queries')


def eval_annoctr_dev(ckpt_path, tag):
    sys.path.insert(0, str(BUNDLE))
    from annoctr_eval import CrossEncoderScorer, evaluate_pool
    queries = [json.loads(l) for l in
               open(BUNDLE / 'annoctr_queries_dev_mf.jsonl', encoding='utf-8')]
    hierarchy = json.load(open(BUNDLE / 'attack_hierarchy.json', encoding='utf-8'))
    scorer = CrossEncoderScorer(str(ckpt_path))
    rows = evaluate_pool(queries, 'pool_reranking', scorer, hierarchy, tag,
                         progress_every=0)
    def agg(rs):
        n = len(rs)
        return {'n': n,
                'p1': round(sum(r['correct'] for r in rs) / n, 4),
                'p1_plus': round(sum(r['correct_plus'] for r in rs) / n, 4),
                'hit3': round(sum(r['hit3'] for r in rs) / n, 4),
                'mrr': round(sum(r['rr'] for r in rs) / n, 4)}
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r['category']].append(r)
    res = {'overall': agg(rows)}
    for c, rs in sorted(by_cat.items()):
        res[c] = agg(rs)
    per_query = [{'qid': q.get('qid', ''), 'correct': r['correct'],
                  'category': r['category']}
                 for q, r in zip(queries, rows)]
    print(f'      [{tag}] dev overall P@1 {res["overall"]["p1"]:.4f} | '
          f'technique P@1 {res["technique"]["p1"]:.4f}')
    return res, per_query


def eval_ctihal(ckpt_path, tag):
    """Forgetting probe on the canonical 146-query CTI-HAL test split."""
    v2 = RERANKER / 'data' / 'reranker_pairs_enriched_v2.jsonl'
    if not v2.exists():
        print('      regenerating v2 pairs via repo enrich_candidates.py')
        sh(f'{sys.executable} enrich_candidates.py', cwd=RERANKER)
    sys.path.insert(0, str(RERANKER))
    import v2_reeval
    query_data, _ = v2_reeval.load_and_group_data(str(v2))
    _, _, test_keys = v2_reeval.create_test_split(query_data)
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(str(ckpt_path), max_length=512)
    correct = 0
    for k in test_keys:
        q = query_data[k]
        pairs = [[q['query_raw'], c['text']] for c in q['candidates']]
        scores = model.predict(pairs, show_progress_bar=False)
        top = max(range(len(scores)), key=lambda i: scores[i])
        if q['candidates'][top]['id'] in q['gold_ids']:
            correct += 1
    n = len(test_keys)
    print(f'      [{tag}] CTI-HAL {correct}/{n} = {correct/n:.4f}')
    return {'correct': correct, 'n': n, 'p1': round(correct / n, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    t0 = time.time()

    if not args.dry_run:
        step1_get_repo_and_checkpoint()
    else:
        print('[1/5] [dry] skipping repo/checkpoint')
    step2_train(args.dry_run)

    results = {'meta': {'lr': LR, 'epochs': EPOCHS, 'batch': BATCH,
                        'seed': SEED, 'timestamp': time.strftime('%F %T')}}
    if not args.dry_run:
        print('[3/5] Eval A: AnnoCTR dev (mf), base vs adapted')
        results['annoctr_dev_base'], pq_base = eval_annoctr_dev(BASE_CKPT, 'base')
        results['annoctr_dev_adapted'], pq_adpt = eval_annoctr_dev(
            OUT_CKPT / 'best', 'adapted')
        flips = sum(1 for a, b in zip(pq_base, pq_adpt)
                    if a['correct'] != b['correct'])
        results['annoctr_dev_flips'] = flips
        print('[4/5] Eval B: CTI-HAL forgetting probe, base vs adapted')
        results['ctihal_base'] = eval_ctihal(BASE_CKPT, 'base')
        if results['ctihal_base']['correct'] != 138:
            print('      WARNING: base != 138/146 — split reproduction is off; '
                  'treat the probe as relative-only')
        results['ctihal_adapted'] = eval_ctihal(OUT_CKPT / 'best', 'adapted')
        print('[5/5] Packaging')
        out = CONTENT / 'stage3_results.json'
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=1, sort_keys=True)
        import shutil
        zpath = shutil.make_archive(str(CONTENT / 'annoctr_stage3_v1_best'),
                                    'zip', OUT_CKPT / 'best')
        print(f'      wrote {out}')
        print(f'      wrote {zpath}  <- download this (Files pane, left sidebar)')
    else:
        print('[3-5/5] [dry] skipping evals/packaging')
    print(f'Done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
