#!/usr/bin/env python3
"""
Checkpoint Unification: Interpolation Sweep + Rehearsal Fine-Tune (Colab)
=========================================================================
Produces ONE unified-candidate checkpoint from the parent (best_two_stage_v2)
and the AnnoCTR-adapted descendant (annoctr_stage3_v1), following the
research-report recipe: Stage A weight-interpolation sweep, Stage B rehearsal
fine-tunes at 25% and 45% replay, Stage C candidate selection on validation
data only (AnnoCTR dev + CTI-HAL val). No test set is ever scored.

Run in Colab (T4 GPU) after uploading rehearsal_colab_bundle.zip:

    !unzip -o rehearsal_colab_bundle.zip -d /content/reh
    !pip -q install sentence-transformers rank_bm25
    !python /content/reh/rehearsal_unify_colab.py

Selection gates (declared in advance, all on validation data):
  G1  AnnoCTR dev overall P@1 >= (descendant's dev overall) - 2.0 pts
  G2  AnnoCTR dev technique P@1 >= (descendant's dev technique) - 2.0 pts
  G3  CTI-HAL val P@1 >= (parent's val P@1) - 1.5 pts
Winner = gate-passer maximizing min(margin_old, margin_new), where
  margin_old = CTI-HAL val (cand) - CTI-HAL val (parent)
  margin_new = AnnoCTR dev overall (cand) - AnnoCTR dev overall (descendant).
If no candidate passes all gates, no winner is zipped; the full table is
reported and the pre-declared fallback (two-checkpoint framing) applies.

Downloads at the end: /content/rehearsal_results.json and, if a winner
exists, /content/best_three_stage_v1_CANDIDATE.zip ("CANDIDATE" until the
one-time test certification).
"""

import argparse
import copy
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
PARENT_DIR = RERANKER / 'checkpoints' / 'best_two_stage_v2'
DESC_DIR = BUNDLE / 'annoctr_stage3_v1'
WORK = CONTENT / 'unify_work'

SEED = 42
LR = 5e-6
BATCH = 16
ANNOCTR_GATE_PTS = 2.0     # G1/G2 tolerance vs descendant (dev)
CTIHAL_GATE_PTS = 1.5      # G3 tolerance vs parent (val)
REPLAY_MIXES = (0.25, 0.45)
ALPHA_COARSE = [round(a * 0.1, 2) for a in range(11)]

LOG = {'asserts': [], 'seeds': {'python_random': SEED, 'torch': SEED}}


def note(msg):
    print('  ' + msg)
    LOG['asserts'].append(msg)


def sh(cmd, cwd=None):
    print('  $', cmd)
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
        sys.exit(f'command failed: {cmd}')
    return r


# ---------------------------------------------------------------------------
# CTI-HAL data machinery (VERBATIM RNG path of finetune_production.py;
# reproduction of the 1111/135/146 split and the 10,727-pair training set
# verified against the repository data before this script was shipped)
# ---------------------------------------------------------------------------

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


def build_stage2_pairs(train_queries, query_data):
    """Continues the same RNG stream as finetune_splits (do not reseed)."""
    out = []
    for qn in train_queries:
        if qn not in query_data:
            continue
        d = query_data[qn]; qr = d['query_raw']
        pos = [c for c in d['candidates'] if c['label'] == 1]
        neg = [c for c in d['candidates'] if c['label'] == 0]
        for c in pos:
            out.append({'qn': qn, 'q': qr, 't': c['text'], 'y': 1.0})
        k = min(int(len(pos) * 2.5), len(neg))
        if k > 0:
            for c in random.sample(neg, k):
                out.append({'qn': qn, 'q': qr, 't': c['text'], 'y': 0.0})
    random.shuffle(out)
    return out


def prepare_ctihal():
    v2file = RERANKER / 'data' / 'reranker_pairs_enriched_v2.jsonl'
    if not v2file.exists():
        print('  regenerating v2 pairs via repo enrich_candidates.py')
        sh(f'{sys.executable} enrich_candidates.py', cwd=RERANKER)
    qd = load_and_group(str(v2file))
    tr, va, te = finetune_splits(qd)
    assert (len(tr), len(va), len(te)) == (1111, 135, 146), \
        f'split reproduction failed: {len(tr)}/{len(va)}/{len(te)}'
    note(f'CTI-HAL split reproduced: 1111/135/146')
    pairs = build_stage2_pairs(tr, qd)
    npos = sum(1 for p in pairs if p['y'] == 1.0)
    assert (len(pairs), npos) == (10727, 3190), \
        f'stage-2 pair reproduction failed: {len(pairs)} ({npos} pos)'
    note('stage-2 training set reproduced: 10,727 pairs (3,190 pos / 7,537 neg)')
    # leak asserts
    trs, vas, tes = set(tr), set(va), set(te)
    assert all(p['qn'] in trs for p in pairs), 'replay source leaked outside train'
    assert not (trs & vas) and not (trs & tes) and not (vas & tes), 'split leakage'
    note('leak asserts passed: replay source strictly train; splits disjoint')
    return qd, tr, va, te, pairs


def sample_replay(stage2_pairs, mix, n_new):
    """Whole-query sampling from the reproduced stage-2 set until the pair
    budget for the requested old:new mix is met. Deterministic per mix."""
    target = round(n_new * mix / (1.0 - mix))
    byq = defaultdict(list)
    for p in stage2_pairs:
        byq[p['qn']].append(p)
    order = sorted(byq)
    rng = random.Random(f'{SEED}:replay:{mix}')
    rng.shuffle(order)
    out, i = [], 0
    while len(out) < target and i < len(order):
        out.extend(byq[order[i]]); i += 1
    note(f'replay mix {mix:.0%}: target {target} pairs -> sampled {len(out)} '
         f'from {i} train queries')
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def make_ce(state_dict=None, base_dir=None):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(str(base_dir or PARENT_DIR), max_length=512)
    if state_dict is not None:
        ce.model.load_state_dict(state_dict, strict=True)
        ce.model.eval()
    return ce


def eval_annoctr_dev(ce, tag):
    sys.path.insert(0, str(BUNDLE))
    from annoctr_eval import evaluate_pool
    queries = [json.loads(l) for l in
               open(BUNDLE / 'annoctr_queries_dev_mf.jsonl', encoding='utf-8')]
    hierarchy = json.load(open(BUNDLE / 'attack_hierarchy.json', encoding='utf-8'))

    class W:
        name = tag
        def predict(self, pairs, **kw):
            return ce.predict(pairs, batch_size=64, show_progress_bar=False).tolist()

    rows = evaluate_pool(queries, 'pool_reranking', W(), hierarchy, tag,
                         progress_every=0)
    tech = [r for r in rows if r['category'] == 'technique']
    res = {'overall_p1': round(sum(r['correct'] for r in rows) / len(rows), 4),
           'technique_p1': round(sum(r['correct'] for r in tech) / len(tech), 4)}
    return res


def eval_ctihal_val(ce, qd, val_keys):
    correct = 0
    for k in val_keys:
        q = qd[k]
        scores = ce.predict([[q['query_raw'], c['text']] for c in q['candidates']],
                            batch_size=64, show_progress_bar=False)
        top = max(range(len(scores)), key=lambda i: scores[i])
        gold = {c['id'] for c in q['candidates'] if c['label'] == 1}
        if q['candidates'][top]['id'] in gold:
            correct += 1
    return round(correct / len(val_keys), 4)


def eval_candidate(name, ce, qd, val_keys):
    t0 = time.time()
    a = eval_annoctr_dev(ce, name)
    c = eval_ctihal_val(ce, qd, val_keys)
    r = {'name': name, 'annoctr_dev_overall': a['overall_p1'],
         'annoctr_dev_technique': a['technique_p1'], 'ctihal_val': c,
         'seconds': round(time.time() - t0, 1)}
    print('  [%-22s] AnnoCTR dev %.4f (tech %.4f) | CTI-HAL val %.4f'
          % (name, r['annoctr_dev_overall'], r['annoctr_dev_technique'], c))
    return r


# ---------------------------------------------------------------------------
# Weight-space operations
# ---------------------------------------------------------------------------

def load_sd(model_dir):
    from safetensors.torch import load_file
    return load_file(str(Path(model_dir) / 'model.safetensors'))


def check_compat(sd_a, sd_b):
    assert set(sd_a) == set(sd_b), 'state-dict key sets differ'
    for k in sd_a:
        assert sd_a[k].shape == sd_b[k].shape, f'shape mismatch at {k}'
    ca = json.load(open(PARENT_DIR / 'config.json'))
    cb = json.load(open(DESC_DIR / 'config.json'))
    for f in ('hidden_size', 'vocab_size', 'num_hidden_layers'):
        assert ca.get(f) == cb.get(f), f'config mismatch: {f}'
    assert len(ca.get('id2label', {})) == len(cb.get('id2label', {})) == 1, \
        'head is not single-logit on both checkpoints'
    note('compat asserts passed: keys, shapes, config, single-logit head')


def blend(sd_a, sd_b, alpha):
    import torch
    out = {}
    for k in sd_a:
        ta, tb = sd_a[k], sd_b[k]
        if ta.dtype.is_floating_point:
            out[k] = (1.0 - alpha) * ta + alpha * tb
        else:
            assert (ta == tb).all(), f'non-float buffer differs at {k}'
            out[k] = ta.clone()
    return out


def soup(sds):
    import torch
    out = {}
    for k in sds[0]:
        if sds[0][k].dtype.is_floating_point:
            out[k] = sum(sd[k] for sd in sds) / float(len(sds))
        else:
            out[k] = sds[0][k].clone()
    return out


# ---------------------------------------------------------------------------
# Rehearsal training
# ---------------------------------------------------------------------------

def train_rehearsal(mix, annoctr_pairs, replay_pairs, val_samples, out_dir):
    import torch
    import random as _random
    import numpy as np
    _random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    from sentence_transformers import CrossEncoder, InputExample
    try:
        from sentence_transformers.cross_encoder.evaluation import (
            CERerankingEvaluator as RerankEval)
    except ImportError:
        from sentence_transformers.cross_encoder.evaluation import (
            CrossEncoderRerankingEvaluator as RerankEval)
    from torch.utils.data import DataLoader

    examples = ([InputExample(texts=[p['q'], p['t']], label=p['y'])
                 for p in replay_pairs] +
                [InputExample(texts=[r['query_raw'], r['candidate_text']],
                              label=float(r['label'])) for r in annoctr_pairs])
    print(f'  rehearsal mix {mix:.0%}: {len(examples)} pairs '
          f'({len(replay_pairs)} replay + {len(annoctr_pairs)} AnnoCTR), '
          f'interleaved via shuffled DataLoader')
    model = CrossEncoder(str(DESC_DIR), num_labels=1, max_length=512)
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
    steps = len(examples) // BATCH
    evaluator = RerankEval(val_samples, name='unifyval')
    out_dir.mkdir(parents=True, exist_ok=True)
    model.fit(train_dataloader=loader, evaluator=evaluator, epochs=1,
              evaluation_steps=max(1, steps // 2),
              warmup_steps=int(steps * 0.1),
              output_path=str(out_dir / 'best'), save_best_model=True,
              optimizer_params={'lr': LR}, weight_decay=0.01,
              show_progress_bar=True)
    model.save(str(out_dir / 'last'))
    return out_dir / 'best'


def build_union_val_samples(qd, val_keys):
    """Training-time evaluator spanning BOTH domains: AnnoCTR training-val
    (bundled) + CTI-HAL val queries (pos + up to 10 negs)."""
    samples = []
    for line in open(BUNDLE / 'annoctr_stage3_val.jsonl', encoding='utf-8'):
        r = json.loads(line)
        if r['positives'] and r['negatives']:
            samples.append({'query': r['query_raw'], 'positive': r['positives'],
                            'negative': r['negatives']})
    n_a = len(samples)
    for k in val_keys:
        q = qd[k]
        pos = [c['text'] for c in q['candidates'] if c['label'] == 1]
        neg = [c['text'] for c in q['candidates'] if c['label'] == 0][:10]
        if pos and neg:
            samples.append({'query': q['query_raw'], 'positive': pos,
                            'negative': neg})
    note(f'union training evaluator: {n_a} AnnoCTR-trainval + '
         f'{len(samples)-n_a} CTI-HAL-val queries')
    return samples


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='data plumbing + asserts only; no GPU, no evals')
    args = ap.parse_args()
    t0 = time.time()
    results = {'meta': {'seed': SEED, 'lr': LR, 'batch': BATCH,
                        'gates': {'annoctr_pts': ANNOCTR_GATE_PTS,
                                  'ctihal_pts': CTIHAL_GATE_PTS},
                        'replay_mixes': list(REPLAY_MIXES),
                        'timestamp': time.strftime('%F %T')},
               'candidates': []}

    print('[1/6] Repo + parent checkpoint')
    if not args.dry_run:
        if not REPO.exists():
            sh(f'GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --branch dataset-expansion '
               f'https://github.com/edogdu/CTI {REPO}')
        sh('git lfs install --skip-smudge', cwd=REPO)
        sh('git lfs pull -I "graph_alignment/reranker/checkpoints/best_two_stage_v2/*"',
           cwd=REPO)
        assert (PARENT_DIR / 'model.safetensors').stat().st_size > 80_000_000
    assert (DESC_DIR / 'model.safetensors').exists(), \
        'descendant checkpoint missing from bundle'

    print('[2/6] CTI-HAL data (verified verbatim reproduction path)')
    if args.dry_run and not (RERANKER / 'data').exists():
        print('  [dry] repo absent; skipping CTI-HAL phase')
        qd = None
    else:
        qd, tr, va, te, stage2 = prepare_ctihal()
        replay = {m: sample_replay(stage2, m, n_new=sum(
            1 for _ in open(BUNDLE / 'annoctr_stage3_train.jsonl'))) 
            for m in REPLAY_MIXES}
        # test keys computed ONLY for the leak assert above; never scored.
        results['meta']['test_contact'] = 'none (test keys used for leak assert only)'

    annoctr_pairs = [json.loads(l) for l in
                     open(BUNDLE / 'annoctr_stage3_train.jsonl', encoding='utf-8')]
    note(f'AnnoCTR training pairs loaded: {len(annoctr_pairs)}')

    if args.dry_run:
        print('[3-6/6] [dry] skipping model phases')
        print(json.dumps(LOG['asserts'], indent=1))
        return

    print('[3/6] Stage A: interpolation sweep')
    sd_p, sd_d = load_sd(PARENT_DIR), load_sd(DESC_DIR)
    check_compat(sd_p, sd_d)
    # endpoint sanity: blended alpha=0 must equal parent, alpha=1 descendant
    ce_p = make_ce(); ce_d = make_ce(base_dir=DESC_DIR)
    base_p = eval_candidate('parent(v2)', ce_p, qd, va)
    base_d = eval_candidate('descendant(s3)', ce_d, qd, va)
    results['candidates'] += [base_p, base_d]
    ce_a0 = make_ce(blend(sd_p, sd_d, 0.0))
    a0 = eval_ctihal_val(ce_a0, qd, va)
    assert abs(a0 - base_p['ctihal_val']) < 1e-9, \
        f'endpoint check failed: alpha=0 ({a0}) != parent ({base_p["ctihal_val"]})'
    del ce_a0
    ce_a1 = make_ce(blend(sd_p, sd_d, 1.0))
    a1 = eval_ctihal_val(ce_a1, qd, va)
    assert abs(a1 - base_d['ctihal_val']) < 1e-9, \
        f'endpoint check failed: alpha=1 ({a1}) != descendant ({base_d["ctihal_val"]})'
    note('endpoint checks passed: alpha=0 == parent, alpha=1 == descendant')
    del ce_a1

    recipes = {}
    sweep = []
    for a in ALPHA_COARSE[1:-1]:
        name = f'interp a={a:.2f}'
        recipes[name] = ('blend_pd', a)
        ce = make_ce(blend(sd_p, sd_d, a))
        sweep.append((a, eval_candidate(name, ce, qd, va)))
        del ce
    def min_margin(r):
        return min(r['ctihal_val'] - base_p['ctihal_val'],
                   r['annoctr_dev_overall'] - base_d['annoctr_dev_overall'])
    best_a, best_r = max(sweep, key=lambda x: min_margin(x[1]))
    for a in (round(best_a - 0.05, 2), round(best_a + 0.05, 2)):
        if 0 < a < 1:
            name = f'interp a={a:.2f}'
            recipes[name] = ('blend_pd', a)
            ce = make_ce(blend(sd_p, sd_d, a))
            sweep.append((a, eval_candidate(name, ce, qd, va)))
            del ce
    best_a, best_r = max(sweep, key=lambda x: min_margin(x[1]))
    results['candidates'] += [r for _, r in sweep]
    results['interp_best_alpha'] = best_a
    print(f'  best alpha by min-margin: {best_a:.2f}')

    print('[4/6] Stage B: rehearsal fine-tunes (from descendant)')
    val_samples = build_union_val_samples(qd, va)
    reh = {}
    for m in REPLAY_MIXES:
        out = train_rehearsal(m, annoctr_pairs, replay[m], val_samples,
                              WORK / f'rehearsal_{int(m*100)}')
        name = f'rehearsal {int(m*100)}%'
        recipes[name] = ('dir', str(out))
        ce = make_ce(base_dir=out)
        reh[m] = (out, eval_candidate(name, ce, qd, va))
        results['candidates'].append(reh[m][1])
        del ce

    print('[5/6] Stage C: soup, conditional blend-backs, selection')
    sd_r = {m: load_sd(reh[m][0]) for m in REPLAY_MIXES}
    recipes['uniform soup (4)'] = ('soup4', None)
    ce_soup = make_ce(soup([sd_p, sd_d] + list(sd_r.values())))
    r_soup = eval_candidate('uniform soup (4)', ce_soup, qd, va)
    results['candidates'].append(r_soup)
    del ce_soup

    def gates(r):
        return (r['annoctr_dev_overall'] >= base_d['annoctr_dev_overall'] - ANNOCTR_GATE_PTS/100 and
                r['annoctr_dev_technique'] >= base_d['annoctr_dev_technique'] - ANNOCTR_GATE_PTS/100 and
                r['ctihal_val'] >= base_p['ctihal_val'] - CTIHAL_GATE_PTS/100)

    best_mix = max(REPLAY_MIXES, key=lambda m: min_margin(reh[m][1]))
    br = reh[best_mix][1]
    if not gates(br):
        target, tsd = ((sd_p, 'v2') if br['ctihal_val'] <
                       base_p['ctihal_val'] - CTIHAL_GATE_PTS/100
                       else (sd_d, 's3'))
        for b in (0.25, 0.5):
            name = f'reh{int(best_mix*100)}->%s b=%.2f' % (tsd, b)
            recipes[name] = ('blendback', (best_mix, tsd, b))
            ce = make_ce(blend(sd_r[best_mix], target, b))
            results['candidates'].append(eval_candidate(name, ce, qd, va))
            del ce

    def realize(name):
        kind, arg = recipes[name]
        if kind == 'blend_pd':
            return ('sd', blend(sd_p, sd_d, arg))
        if kind == 'soup4':
            return ('sd', soup([sd_p, sd_d] + list(sd_r.values())))
        if kind == 'dir':
            return ('dir', arg)
        if kind == 'blendback':
            m, tname, b = arg
            tgt = sd_p if tname == 'v2' else sd_d
            return ('sd', blend(sd_r[m], tgt, b))
        raise KeyError(name)

    passers = [r for r in results['candidates']
               if gates(r) and r['name'] not in ('parent(v2)', 'descendant(s3)')]
    results['gate_passers'] = [r['name'] for r in passers]
    if passers:
        winner = max(passers, key=min_margin)
        results['winner'] = winner['name']
        results['winner_margins'] = {
            'old': round(winner['ctihal_val'] - base_p['ctihal_val'], 4),
            'new': round(winner['annoctr_dev_overall'] - base_d['annoctr_dev_overall'], 4)}
        print(f'  WINNER: {winner["name"]}  '
              f'(margins old {results["winner_margins"]["old"]:+.4f}, '
              f'new {results["winner_margins"]["new"]:+.4f})')
        print('[6/6] Saving winner as best_three_stage_v1_CANDIDATE')
        out = WORK / 'best_three_stage_v1_CANDIDATE'
        kind, payload = realize(winner['name'])
        if kind == 'sd':
            ce = make_ce(payload)
            ce.save(str(out))
        else:
            shutil.copytree(payload, out, dirs_exist_ok=True)
        z = shutil.make_archive(str(CONTENT / 'best_three_stage_v1_CANDIDATE'),
                                'zip', out)
        print(f'  wrote {z}')
    else:
        results['winner'] = None
        print('  NO gate-passer: pre-declared fallback applies '
              '(two-checkpoint framing); table reported, nothing zipped.')

    results['asserts'] = LOG['asserts']
    with open(CONTENT / 'rehearsal_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1, sort_keys=True)
    print(f'  wrote {CONTENT / "rehearsal_results.json"}')
    print(f'Done in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
