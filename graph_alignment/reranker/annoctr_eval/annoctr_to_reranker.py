#!/usr/bin/env python3
"""
AnnoCTR -> Reranker Unseen-Eval Bridge (v2: experiment flags)
=============================================================
v1 behavior (no flags) is preserved: same grouping, same golds, same pools.
v2 adds three query/candidate construction experiments and pool freezing:

  --context-window     Query = prev sentence + mention-sentence + next sentence
                       (AnnoCTR sentence_left/right), capped at 300 words with
                       the mention-sentence always kept whole (neighbors are
                       trimmed first: left neighbor from its start, right
                       neighbor from its end). Mirrors the pipeline's semantic-
                       chunking cap.
  --mention-first      Query = "<mention(s)> - <sentence>" -- prepends the
                       annotated mention span(s) (sorted, '; '-joined when a
                       sentence has several), mimicking the entity-anchored
                       query style of the CTI-HAL training pairs.
  --enrich-procedures  Technique candidate cards become
                       "{ID} - {Name}: {desc<=600} Example: {procedure<=300}"
                       using ATT&CK 'uses'-relationship descriptions (cleaned,
                       deterministic shortest adequate example). Techniques
                       without an example (108/625) keep the standard card.
                       Software/tactic cards unchanged.
  --freeze-pools-from FILE
                       Reuse candidate-ID pools per query from a baseline
                       queries JSONL (matched by stable qid) instead of running
                       BM25. Candidate TEXTS still come from the current
                       variant's lookup. This makes variant sweeps a controlled
                       comparison: identical negatives, only texts differ.

Every query record now carries:
  qid        stable id = sha1(document | base-sentence-norm)[:16]; invariant
             across variants (grouping is ALWAYS by the base mention-sentence,
             so multi-gold groups never split under --mention-first).
  base_norm  the baseline mention-sentence normalization (grouping key).
  mentions   sorted unique annotated mention spans in the group.

Outputs and all v1 fields are otherwise unchanged.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

MAX_DESC_CHARS = 1000       # standard card (matches enrich_candidates.py)
ENRICH_DESC_CHARS = 600     # shortened description when an example is appended
ENRICH_EX_MIN = 40          # minimum cleaned example length
ENRICH_EX_MAX = 300         # example truncation cap
CONTEXT_CAP_WORDS = 300     # mirrors the pipeline's semantic-chunking cap
BM25_TOP_K = 20             # matches deployment_eval.py / ICAIC recipe

ATTACK_ID_RE = re.compile(r'^(T\d{4}(?:\.\d{3})?|TA\d{4}|S\d{4})$')
MITRE_URL_RE = re.compile(
    r'attack\.mitre\.org/(techniques|software|tactics|groups)/'
    r'(T\d{4}|TA\d{4}|S\d{4}|G\d{4})(?:/(\d{3}))?', re.I)
MD_LINK_RE = re.compile(r'\[([^\]]*)\]\([^)]*\)')


def extract_attack_id(external_references):
    if not external_references:
        return None
    for ref in external_references:
        if ref.get('source_name') == 'mitre-attack':
            eid = ref.get('external_id', '')
            if ATTACK_ID_RE.match(eid):
                return eid
    return None


def truncate_words(desc, max_chars):
    """Word-boundary truncation, identical policy to enrich_candidates.py."""
    if len(desc) <= max_chars:
        return desc
    truncated = desc[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + '...'


def clean_description(desc, max_chars=MAX_DESC_CHARS):
    if not desc:
        return ''
    desc = re.sub(r'\(Citation:\s*[^)]*\)', '', desc)
    desc = re.sub(r'\s+', ' ', desc).strip()
    return truncate_words(desc, max_chars)


def clean_rel_text(desc):
    """Clean a relationship (procedure-example) description: citations out,
    markdown links -> anchor text, whitespace collapsed."""
    if not desc:
        return ''
    desc = re.sub(r'\(Citation:\s*[^)]*\)', '', desc)
    desc = MD_LINK_RE.sub(r'\1', desc)
    desc = re.sub(r'\s+', ' ', desc).strip()
    return desc


def build_v14(stix_path, enrich_procedures=False):
    """Returns (lookup, revoked_map, hierarchy).
    lookup[attack_id] = {name, description, type, enriched_text}
    With enrich_procedures, technique cards embed one procedure example."""
    with open(stix_path, 'r', encoding='utf-8') as f:
        stix = json.load(f)
    objects = stix.get('objects', [])

    lookup = {}
    stix_uid_to_attack = {}
    revoked_ids = {}
    tactic_shortname_to_id = {}
    raw_desc = {}

    for obj in objects:
        obj_type = obj.get('type', '')
        if obj_type not in ('attack-pattern', 'x-mitre-tactic', 'malware', 'tool'):
            continue
        attack_id = extract_attack_id(obj.get('external_references', []))
        if not attack_id:
            continue
        stix_uid_to_attack[obj['id']] = attack_id
        is_dead = obj.get('revoked', False) or obj.get('x_mitre_deprecated', False)
        if is_dead:
            revoked_ids[attack_id] = obj['id']
            continue
        name = obj.get('name', '')
        if not name:
            continue
        raw_desc[attack_id] = obj.get('description', '')
        lookup[attack_id] = {'name': name, 'type': obj_type}
        if obj_type == 'x-mitre-tactic':
            short = obj.get('x_mitre_shortname', '')
            if short:
                tactic_shortname_to_id[short] = attack_id

    # Procedure examples per technique (deterministic: shortest adequate,
    # ties broken by text), from 'uses' relationships with descriptions.
    examples = {}
    if enrich_procedures:
        cand = defaultdict(list)
        for obj in objects:
            if obj.get('type') != 'relationship':
                continue
            if obj.get('relationship_type') != 'uses':
                continue
            tgt = stix_uid_to_attack.get(obj.get('target_ref'))
            if not tgt or tgt not in lookup or lookup[tgt]['type'] != 'attack-pattern':
                continue
            ex = clean_rel_text(obj.get('description', ''))
            if len(ex) >= ENRICH_EX_MIN:
                cand[tgt].append(truncate_words(ex, ENRICH_EX_MAX))
        for tid, exs in cand.items():
            examples[tid] = sorted(exs, key=lambda e: (len(e), e))[0]

    # Final card texts
    for aid, entry in lookup.items():
        name = entry['name']
        if enrich_procedures and entry['type'] == 'attack-pattern' and aid in examples:
            desc = clean_description(raw_desc.get(aid, ''), ENRICH_DESC_CHARS)
            entry['description'] = desc
            entry['example'] = examples[aid]
            entry['enriched_text'] = (
                f"{aid} — {name}: {desc} Example: {examples[aid]}" if desc
                else f"{aid} — {name} Example: {examples[aid]}")
        else:
            desc = clean_description(raw_desc.get(aid, ''), MAX_DESC_CHARS)
            entry['description'] = desc
            entry['enriched_text'] = (f"{aid} — {name}: {desc}" if desc
                                      else f"{aid} — {name}")

    revoked_map = {}
    for obj in objects:
        if obj.get('type') == 'relationship' and obj.get('relationship_type') == 'revoked-by':
            src_aid = stix_uid_to_attack.get(obj.get('source_ref'))
            tgt_aid = stix_uid_to_attack.get(obj.get('target_ref'))
            if src_aid and tgt_aid and tgt_aid in lookup:
                revoked_map[src_aid] = tgt_aid
    for aid in revoked_ids:
        revoked_map.setdefault(aid, None)

    sub_to_parent = {}
    for aid in lookup:
        if aid.startswith('T') and '.' in aid:
            parent = aid.split('.')[0]
            if parent in lookup:
                sub_to_parent[aid] = parent
    tech_to_tactics = defaultdict(set)
    for obj in objects:
        if obj.get('type') != 'attack-pattern':
            continue
        if obj.get('revoked', False) or obj.get('x_mitre_deprecated', False):
            continue
        aid = extract_attack_id(obj.get('external_references', []))
        if not aid:
            continue
        for phase in obj.get('kill_chain_phases', []):
            if phase.get('kill_chain_name') == 'mitre-attack':
                tid = tactic_shortname_to_id.get(phase.get('phase_name', ''))
                if tid:
                    tech_to_tactics[aid].add(tid)
    for sub, parent in sub_to_parent.items():
        if parent in tech_to_tactics:
            tech_to_tactics[sub].update(tech_to_tactics[parent])

    hierarchy = {
        'sub_to_parent': sub_to_parent,
        'tech_to_tactics': {k: sorted(v) for k, v in tech_to_tactics.items()},
    }
    return lookup, revoked_map, hierarchy


def parse_label_link(url):
    m = MITRE_URL_RE.search(url or '')
    if not m:
        return None, None
    kind, base, sub = m.group(1).lower(), m.group(2).upper(), m.group(3)
    if kind == 'techniques' and sub:
        return 'technique', f"{base}.{sub}"
    return {'techniques': 'technique', 'software': 'software',
            'tactics': 'tactic', 'groups': 'group'}[kind], base


def clean_text(s):
    if not s:
        return ''
    s = MD_LINK_RE.sub(r'\1', s)
    s = s.replace('\xa0', ' ')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def load_split(annoctr_dir, split):
    path = Path(annoctr_dir) / 'linking_mitre_only' / f'{split}.jsonl'
    recs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs, path


def sha256_head(path, n=16):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:n]


def make_query_text(q, context_window, mention_first):
    """Construct the variant query text from a grouped query record.
    Base (both flags off) is exactly the v1 text: the cleaned mention-sentence."""
    text = q['base_raw']
    if context_window:
        base_words = text.split()
        left_words = q['sent_left'].split()
        right_words = q['sent_right'].split()
        budget = CONTEXT_CAP_WORDS - len(base_words)
        if budget < 0:
            budget = 0
        half = budget // 2
        # deterministic split: give each side half the budget, spill leftover
        lw = min(len(left_words), half)
        rw = min(len(right_words), budget - lw)
        lw = min(len(left_words), budget - rw)          # spill back to left
        left = ' '.join(left_words[len(left_words) - lw:]) if lw else ''
        right = ' '.join(right_words[:rw]) if rw else ''
        text = ' '.join(p for p in (left, text, right) if p)
    if mention_first:
        text = '; '.join(q['mentions']) + ' — ' + text
    return text


def build_split(recs, lookup, revoked_map, split,
                context_window=False, mention_first=False,
                bm25=None, corpus_ids=None, corpus_texts=None,
                frozen_pools=None):
    stats = Counter()
    crosswalk = {'passed': Counter(), 'mapped_forward': {}, 'dropped': Counter(),
                 'group_links_excluded': Counter()}
    queries = {}

    for r in recs:
        stats['mentions_total'] += 1
        kind, aid = parse_label_link(r.get('label_link', ''))
        if kind is None:
            stats['mentions_non_mitre'] += 1
            continue
        if kind == 'group':
            stats['mentions_group_excluded'] += 1
            crosswalk['group_links_excluded'][aid] += 1
            continue
        gold = None
        if aid in lookup:
            gold = aid
            crosswalk['passed'][kind] += 1
        elif aid in revoked_map and revoked_map[aid]:
            gold = revoked_map[aid]
            crosswalk['mapped_forward'][aid] = gold
            stats['mentions_gold_mapped_forward'] += 1
        else:
            crosswalk['dropped'][aid] += 1
            stats['mentions_gold_unresolvable'] += 1
            continue
        stats['mentions_kept'] += 1

        base_raw = clean_text(
            f"{r.get('context_left','')} {r.get('mention','')} {r.get('context_right','')}")
        if not base_raw:
            stats['mentions_empty_query'] += 1
            continue
        base_norm = base_raw.lower()
        doc = r.get('document', 'unknown')
        key = (doc, base_norm)
        q = queries.setdefault(key, {
            'base_raw': base_raw, 'base_norm': base_norm,
            'document': doc, 'split': split,
            'gold_ids': set(), 'gold_ids_raw': set(),
            'mention_kinds': Counter(), 'n_mentions': 0,
            'mentions': set(), 'sent_left': '', 'sent_right': '',
        })
        q['gold_ids'].add(gold)
        q['gold_ids_raw'].add(aid)
        q['mention_kinds'][kind] += 1
        q['n_mentions'] += 1
        m = clean_text(r.get('mention', ''))
        if m:
            q['mentions'].add(m)
        for fld, dst in (('sentence_left', 'sent_left'),
                         ('sentence_right', 'sent_right')):
            s = clean_text(r.get(fld, ''))
            if (len(s), s) > (len(q[dst]), q[dst]):
                q[dst] = s

    out = []
    for (doc, base_norm), q in sorted(queries.items()):
        q['mentions'] = sorted(q['mentions'])
        gold_ids = sorted(q['gold_ids'])
        qid = hashlib.sha1((doc + '|' + base_norm).encode('utf-8')).hexdigest()[:16]
        query_raw = make_query_text(q, context_window, mention_first)
        if context_window:
            allowance = (len('; '.join(q['mentions']).split()) + 1
                         if mention_first else 0)
            cap = max(CONTEXT_CAP_WORDS, len(q['base_raw'].split())) + allowance
            assert len(query_raw.split()) <= cap, f"context cap violated for {qid}"

        if frozen_pools is not None:
            fp = frozen_pools[qid]
            assert sorted(fp['gold_ids']) == gold_ids, f"gold drift at {qid}"
            deploy = [{'id': cid, 'text': lookup[cid]['enriched_text'],
                       'label': 1 if cid in q['gold_ids'] else 0}
                      for cid in fp['deploy_ids']]
            rerank = [{'id': cid, 'text': lookup[cid]['enriched_text'],
                       'label': 1 if cid in q['gold_ids'] else 0}
                      for cid in fp['rerank_ids']]
        else:
            tokens = query_raw.lower().split()
            scores = bm25.get_scores(tokens)
            order = sorted(range(len(scores)), key=lambda i: -scores[i])[:BM25_TOP_K]
            deploy = [{'id': corpus_ids[i], 'text': corpus_texts[i],
                       'label': 1 if corpus_ids[i] in q['gold_ids'] else 0}
                      for i in order]
            pool_ids = {c['id'] for c in deploy}
            rerank = list(deploy)
            for g in gold_ids:
                if g not in pool_ids:
                    rerank.append({'id': g, 'text': lookup[g]['enriched_text'],
                                   'label': 1})

        kinds = q['mention_kinds']
        category = list(kinds)[0] if len(kinds) == 1 else 'mixed'
        out.append({
            'qid': qid,
            'query_raw': query_raw, 'query_norm': query_raw.lower(),
            'base_norm': base_norm, 'mentions': q['mentions'],
            'document': doc, 'split': split, 'category': category,
            'n_mentions': q['n_mentions'],
            'gold_ids': gold_ids, 'gold_ids_raw': sorted(q['gold_ids_raw']),
            'pool_deploy': deploy, 'pool_reranking': rerank,
        })
    return out, stats, crosswalk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annoctr-dir', required=True)
    ap.add_argument('--stix', required=True)
    ap.add_argument('--splits', nargs='+', default=['test'])
    ap.add_argument('--out-dir', default='annoctr_eval')
    ap.add_argument('--context-window', action='store_true')
    ap.add_argument('--mention-first', action='store_true')
    ap.add_argument('--enrich-procedures', action='store_true')
    ap.add_argument('--freeze-pools-from', default=None,
                    help='baseline queries JSONL to reuse candidate pools from')
    ap.add_argument('--suffix', default='',
                    help='filename suffix for variant outputs, e.g. _cw_mf')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flags = {'context_window': args.context_window,
             'mention_first': args.mention_first,
             'enrich_procedures': args.enrich_procedures,
             'frozen_pools': bool(args.freeze_pools_from)}
    print(f"[1/4] Building ATT&CK v14 lookup (enrich_procedures={args.enrich_procedures})")
    lookup, revoked_map, hierarchy = build_v14(
        args.stix, enrich_procedures=args.enrich_procedures)
    n_by_type = Counter(v['type'] for v in lookup.values())
    n_ex = sum(1 for v in lookup.values() if 'example' in v)
    print(f"      Active candidate universe: {len(lookup)} ({dict(n_by_type)})"
          + (f" | technique cards with example: {n_ex}" if args.enrich_procedures else ''))
    with open(out_dir / 'attack_hierarchy.json', 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, indent=1, sort_keys=True)

    frozen = None
    bm25 = corpus_ids = corpus_texts = None
    if args.freeze_pools_from:
        print(f"[2/4] Freezing pools from {args.freeze_pools_from}")
        frozen = {}
        for line in open(args.freeze_pools_from, encoding='utf-8'):
            q = json.loads(line)
            frozen[q['qid']] = {
                'deploy_ids': [c['id'] for c in q['pool_deploy']],
                'rerank_ids': [c['id'] for c in q['pool_reranking']],
                'gold_ids': q['gold_ids'],
            }
        print(f"      {len(frozen)} frozen query pools loaded")
    else:
        print(f"[2/4] Building BM25 index (k1=1.5, b=0.75, whitespace+lowercase)")
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            sys.exit("[FAIL] rank_bm25 not installed. Run: pip install rank_bm25")
        corpus_ids = sorted(lookup.keys())
        corpus_texts = [lookup[i]['enriched_text'] for i in corpus_ids]
        bm25 = BM25Okapi([t.lower().split() for t in corpus_texts], k1=1.5, b=0.75)
        print(f"      Indexed {len(corpus_ids)} candidates")

    prov = {
        'stix': {'path': str(args.stix), 'sha256_16': sha256_head(args.stix)},
        'flags': flags,
        'params': {'bm25': {'k1': 1.5, 'b': 0.75, 'top_k': BM25_TOP_K},
                   'max_desc_chars': MAX_DESC_CHARS,
                   'enrich_desc_chars': ENRICH_DESC_CHARS,
                   'context_cap_words': CONTEXT_CAP_WORDS},
        'candidate_universe': {'total': len(lookup), 'by_type': dict(n_by_type)},
        'splits': {},
    }

    for split in args.splits:
        print(f"[3/4] Processing split '{split}'")
        recs, src = load_split(args.annoctr_dir, split)
        queries, stats, crosswalk = build_split(
            recs, lookup, revoked_map, split,
            context_window=args.context_window, mention_first=args.mention_first,
            bm25=bm25, corpus_ids=corpus_ids, corpus_texts=corpus_texts,
            frozen_pools=frozen)

        sfx = args.suffix
        qpath = out_dir / f'annoctr_queries_{split}{sfx}.jsonl'
        with open(qpath, 'w', encoding='utf-8') as f:
            for q in queries:
                f.write(json.dumps(q, ensure_ascii=False, sort_keys=True) + '\n')

        ppath = out_dir / f'annoctr_pairs_reranking_{split}{sfx}.jsonl'
        with open(ppath, 'w', encoding='utf-8') as f:
            for q in queries:
                for c in q['pool_reranking']:
                    f.write(json.dumps({
                        'query_raw': q['query_raw'], 'query_norm': q['query_norm'],
                        'candidate_id': c['id'], 'candidate_norm': c['id'],
                        'candidate_text': c['text'], 'label': c['label'],
                        'actor': f"annoctr:{q['document']}",
                    }, ensure_ascii=False, sort_keys=True) + '\n')

        cw = {
            'passed_by_kind': dict(crosswalk['passed']),
            'mapped_forward_v12_to_v14': crosswalk['mapped_forward'],
            'dropped_unresolvable': dict(crosswalk['dropped']),
            'group_links_excluded': dict(crosswalk['group_links_excluded']),
            'stats': dict(stats),
        }
        with open(out_dir / f'crosswalk_report_{split}{sfx}.json', 'w',
                  encoding='utf-8') as f:
            json.dump(cw, f, indent=1, sort_keys=True)

        cat = Counter(q['category'] for q in queries)
        docs = len({q['document'] for q in queries})
        gold_in_pool = sum(1 for q in queries
                           if any(c['label'] for c in q['pool_deploy']))
        print(f"      mentions: {stats['mentions_total']} total, "
              f"{stats['mentions_kept']} kept, "
              f"{stats['mentions_group_excluded']} group-excluded, "
              f"{stats['mentions_gold_unresolvable']} unresolvable")
        print(f"      queries: {len(queries)} across {docs} docs | "
              f"categories: {dict(cat)}")
        print(f"      BM25 gold-in-top-{BM25_TOP_K}: {gold_in_pool}/{len(queries)} "
              f"= {gold_in_pool/len(queries):.2%}"
              + ("  [frozen from baseline]" if frozen else ""))
        prov['splits'][split] = {
            'source': {'path': str(src), 'sha256_16': sha256_head(src)},
            'mention_stats': dict(stats), 'queries': len(queries),
            'documents': docs, 'categories': dict(cat),
            'bm25_gold_in_pool_rate': round(gold_in_pool / len(queries), 4),
        }

    with open(out_dir / f'build_provenance{args.suffix}.json', 'w',
              encoding='utf-8') as f:
        json.dump(prov, f, indent=1, sort_keys=True)
    print(f"[4/4] Done. Outputs in {out_dir}/ (suffix: '{args.suffix}')")


if __name__ == '__main__':
    main()
