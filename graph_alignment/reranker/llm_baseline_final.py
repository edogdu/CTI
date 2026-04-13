#!/usr/bin/env python3
"""
LLM Baseline Evaluation for CTI-to-ATT&CK Mapping
====================================================
Paper evaluation infrastructure — NOT part of the deployed reranker system.

Compares frontier LLMs (zero-shot, generative) against the fine-tuned
MiniLM cross-encoder (discriminative, 22.7M params) on the same 146
CTI-HAL test queries.

Three phases (can run independently):
  PREPARE  - Extract test queries + gold labels from enriched JSONL
  RUN      - Send queries to OpenAI API, save responses
  EVAL     - Score responses, compute P@1, per-actor breakdown

Usage:
  python llm_baseline_final.py prepare --data data\reranker_pairs_enriched.jsonl
  python llm_baseline_final.py run --model gpt-5.4-mini
  python llm_baseline_final.py eval --model gpt-5.4-mini

Requirements:
  pip install openai    # Only needed for 'run' phase

Author: Shane Waldrop — Angelo State University / ARL Grant W911NF-24-2-0180
"""

import json
import re
import os
import sys
import time
import random
import argparse
import csv
from collections import defaultdict
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
OUTPUT_DIR = Path("./llm_baseline_results")
ATTACK_ID_RE = re.compile(r'(TA\d{4}|T\d{4}(?:\.\d{3})?|S\d{4})')

# ── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_ZERO_SHOT = """You are a cybersecurity analyst specializing in the MITRE ATT&CK framework.

Given a passage from a cyber threat intelligence (CTI) report, identify the single most applicable MITRE ATT&CK entity. This may be:
- A technique or sub-technique (e.g., T1059.001)
- A tactic (e.g., TA0002)
- A software entry (e.g., S0051)

Respond with ONLY two lines:
Line 1: The ATT&CK ID (e.g., T1059.001)
Line 2: The entity name (e.g., Command and Scripting Interpreter: PowerShell)

Do not include any explanation, reasoning, or additional text."""

SYSTEM_PROMPT_FEW_SHOT = """You are a cybersecurity analyst specializing in the MITRE ATT&CK framework.

Given a passage from a cyber threat intelligence (CTI) report, identify the single most applicable MITRE ATT&CK entity. This may be:
- A technique or sub-technique (e.g., T1059.001)
- A tactic (e.g., TA0002)
- A software entry (e.g., S0051)

Here are three examples:

CTI passage: "The adversary used PowerShell scripts to download additional payloads from a remote server"
T1059.001
Command and Scripting Interpreter: PowerShell

CTI passage: "The group gained initial access to the network through spearphishing emails containing malicious attachments"
T1566.001
Phishing: Spearphishing Attachment

CTI passage: "The malware established persistence by creating a new Windows service"
T1543.003
Create or Modify System Process: Windows Service

Now identify the ATT&CK entity for the following passage. Respond with ONLY the ATT&CK ID on the first line and the entity name on the second line."""


# ── Phase 1: PREPARE ─────────────────────────────────────────────────────────

def load_and_group_data(filepath):
    """Load enriched JSONL and group by query_norm."""
    query_data = defaultdict(lambda: {
        'query_raw': None,
        'actor': None,
        'candidates': [],
        'positive_count': 0,
        'negative_count': 0,
    })
    total_rows = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            query_norm = row.get('query_norm', '')
            query_raw = row.get('query_raw', '')
            candidate_id = row.get('candidate_id', '')
            candidate_norm = row.get('candidate_norm', '')
            candidate_text = row.get('candidate_text', '')
            label = row.get('label', 0)
            actor = row.get('actor', 'unknown')
            if not query_norm or not query_raw:
                continue
            if query_data[query_norm]['query_raw'] is None:
                query_data[query_norm]['query_raw'] = query_raw
                query_data[query_norm]['actor'] = actor if actor else 'unknown'
            query_data[query_norm]['candidates'].append({
                'id': candidate_norm if candidate_norm else candidate_id,
                'text': candidate_text,
                'label': label,
            })
            if label == 1:
                query_data[query_norm]['positive_count'] += 1
            else:
                query_data[query_norm]['negative_count'] += 1
    return dict(query_data), total_rows


def create_test_split(query_data, train_ratio=0.8, val_ratio=0.1):
    """Recreate the EXACT same test split as finetune_production.py."""
    random.seed(RANDOM_SEED)
    queries_by_actor = defaultdict(list)
    for query_norm, data in query_data.items():
        actor = data['actor']
        if actor.startswith('external_'):
            continue
        if data['positive_count'] > 0:
            queries_by_actor[actor].append(query_norm)
    test_queries = []
    for actor, queries in queries_by_actor.items():
        random.shuffle(queries)
        n = len(queries)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        test_queries.extend(queries[n_train + n_val:])
    random.shuffle(test_queries)
    return test_queries


def phase_prepare(args):
    """Extract 146 test queries with gold labels into a clean JSON file."""
    print("\n" + "=" * 70)
    print("  PHASE 1: PREPARE TEST QUERIES")
    print("=" * 70)
    data_path = args.data
    if not Path(data_path).exists():
        sys.exit(f"[FAIL] Data file not found: {data_path}")
    print(f"\n  Loading data from: {data_path}")
    query_data, total_rows = load_and_group_data(data_path)
    print(f"  Loaded {total_rows:,} rows, {len(query_data):,} unique queries")
    print(f"\n  Creating test split (seed={RANDOM_SEED})...")
    test_queries = create_test_split(query_data)
    print(f"  Test split: {len(test_queries)} queries")
    test_items = []
    actor_counts = defaultdict(int)
    for qn in test_queries:
        qd = query_data[qn]
        if qd['positive_count'] == 0:
            continue
        gold_ids = []
        for c in qd['candidates']:
            if c['label'] == 1:
                match = ATTACK_ID_RE.match(c['id'])
                if match:
                    gold_ids.append(match.group(1))
                else:
                    gold_ids.append(c['id'])
        first_gold = gold_ids[0] if gold_ids else ''
        if first_gold.startswith('TA'):
            entity_type = 'tactic'
        elif first_gold.startswith('S'):
            entity_type = 'software'
        elif '.' in first_gold:
            entity_type = 'sub-technique'
        else:
            entity_type = 'technique'
        test_items.append({
            'query_norm': qn,
            'query_raw': qd['query_raw'],
            'actor': qd['actor'],
            'gold_ids': gold_ids,
            'entity_type': entity_type,
        })
        actor_counts[qd['actor']] += 1
    print(f"\n  Per-actor distribution:")
    for actor in sorted(actor_counts):
        print(f"    {actor}: {actor_counts[actor]}")
    type_counts = defaultdict(int)
    for item in test_items:
        type_counts[item['entity_type']] += 1
    print(f"\n  Per-entity-type distribution:")
    for etype in sorted(type_counts):
        print(f"    {etype}: {type_counts[etype]}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "test_queries.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(test_items, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(test_items)} test queries to: {out_path}")
    print(f"  [DONE] Phase 1 complete.")
    return test_items


# ── Phase 2: RUN ─────────────────────────────────────────────────────────────

def call_openai(client, model, query_raw, system_prompt, max_retries=3):
    """Send a single query to the OpenAI API with retry and timing."""
    for attempt in range(max_retries):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_completion_tokens=100,
                messages=[
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": query_raw},
                ],
            )
            latency_ms = (time.time() - start) * 1000
            content = response.choices[0].message.content or ""
            usage = response.usage
            return {
                'content': content.strip(),
                'latency_ms': round(latency_ms, 1),
                'input_tokens': usage.prompt_tokens if usage else 0,
                'output_tokens': usage.completion_tokens if usage else 0,
                'model': response.model,
                'error': None,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt+1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                return {
                    'content': '',
                    'latency_ms': 0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'model': model,
                    'error': str(e),
                }


def phase_run(args):
    """Send test queries to OpenAI API and save raw responses."""
    print("\n" + "=" * 70)
    print("  PHASE 2: RUN LLM BASELINE")
    print("=" * 70)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "[FAIL] OPENAI_API_KEY environment variable not set.\n"
            "  Set it with: set OPENAI_API_KEY=your-key-here\n"
            "  Get a key at: https://platform.openai.com/api-keys"
        )
    queries_path = OUTPUT_DIR / "test_queries.json"
    if not queries_path.exists():
        sys.exit(
            f"[FAIL] Prepared queries not found at {queries_path}\n"
            "  Run 'python llm_baseline_final.py prepare' first."
        )
    with open(queries_path, 'r') as f:
        test_items = json.load(f)
    model = args.model
    mode = args.prompt_mode
    system_prompt = (SYSTEM_PROMPT_FEW_SHOT if mode == 'few_shot'
                     else SYSTEM_PROMPT_ZERO_SHOT)
    print(f"\n  Model:       {model}")
    print(f"  Prompt mode: {mode}")
    print(f"  Queries:     {len(test_items)}")
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("[FAIL] openai package not installed. Run: pip install openai")
    client = OpenAI(api_key=api_key)
    results_path = OUTPUT_DIR / f"responses_{model}_{mode}.jsonl"
    completed = set()
    if results_path.exists():
        with open(results_path, 'r') as f:
            for line in f:
                rec = json.loads(line)
                completed.add(rec.get('query_norm', ''))
        print(f"  Resuming: {len(completed)} queries already completed")
    remaining = [q for q in test_items if q['query_norm'] not in completed]
    print(f"  Remaining:   {len(remaining)} queries to process")
    if not remaining:
        print(f"\n  All queries already completed.")
        return
    total_input_tokens = 0
    total_output_tokens = 0
    latencies = []
    with open(results_path, 'a', encoding='utf-8') as f_out:
        for i, item in enumerate(remaining):
            result = call_openai(client, model, item['query_raw'], system_prompt)
            record = {
                'query_norm': item['query_norm'],
                'query_raw': item['query_raw'],
                'actor': item['actor'],
                'gold_ids': item['gold_ids'],
                'entity_type': item['entity_type'],
                'llm_response': result['content'],
                'latency_ms': result['latency_ms'],
                'input_tokens': result['input_tokens'],
                'output_tokens': result['output_tokens'],
                'model_used': result['model'],
                'error': result['error'],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            total_input_tokens += result['input_tokens']
            total_output_tokens += result['output_tokens']
            if result['latency_ms'] > 0:
                latencies.append(result['latency_ms'])
            if (i + 1) % 10 == 0 or i == 0:
                avg_lat = sum(latencies) / len(latencies) if latencies else 0
                eta = avg_lat * (len(remaining) - i - 1) / 1000
                print(f"    [{i+1}/{len(remaining)}] "
                      f"avg latency: {avg_lat:.0f}ms, ETA: {eta:.0f}s")
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"\n  Completed {len(remaining)} API calls.")
    print(f"  Avg latency:    {avg_latency:.0f}ms per query")
    print(f"  Total input:    {total_input_tokens:,} tokens")
    print(f"  Total output:   {total_output_tokens:,} tokens")
    print(f"  Results saved:  {results_path}")
    input_cost = total_input_tokens * 1.75 / 1_000_000
    output_cost = total_output_tokens * 14.0 / 1_000_000
    print(f"\n  Estimated cost (GPT-5.2 pricing):")
    print(f"    Input:  ${input_cost:.4f}")
    print(f"    Output: ${output_cost:.4f}")
    print(f"    Total:  ${input_cost + output_cost:.4f}")
    print(f"\n  [DONE] Phase 2 complete.")


# ── Phase 3: EVAL ────────────────────────────────────────────────────────────

def extract_attack_id(text):
    """Extract ATT&CK ID from LLM response text."""
    match = ATTACK_ID_RE.search(text)
    return match.group(1) if match else None


def is_parent_child(predicted_id, gold_id):
    """Check if predicted and gold are parent/child of each other."""
    if not predicted_id or not gold_id:
        return False
    if '.' in gold_id and predicted_id == gold_id.split('.')[0]:
        return True
    if '.' in predicted_id and gold_id == predicted_id.split('.')[0]:
        return True
    return False


def phase_eval(args):
    """Score LLM responses against gold labels and compute metrics."""
    print("\n" + "=" * 70)
    print("  PHASE 3: EVALUATE LLM BASELINE")
    print("=" * 70)
    model = args.model
    mode = args.prompt_mode
    results_path = OUTPUT_DIR / f"responses_{model}_{mode}.jsonl"
    if not results_path.exists():
        sys.exit(f"[FAIL] Response file not found: {results_path}\n"
                 f"  Run 'python llm_baseline_final.py run --model {model}' first.")
    responses = []
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    print(f"\n  Loaded {len(responses)} responses for {model} ({mode})")
    results_by_actor = defaultdict(list)
    results_by_type = defaultdict(list)
    all_results = []
    parse_failures = 0
    api_errors = 0
    near_misses = 0
    latencies = []
    for resp in responses:
        if resp.get('error'):
            api_errors += 1
            continue
        predicted_id = extract_attack_id(resp['llm_response'])
        gold_ids = resp['gold_ids']
        actor = resp['actor']
        entity_type = resp['entity_type']
        if predicted_id is None:
            parse_failures += 1
            correct = False
            near_miss = False
        else:
            correct = predicted_id in gold_ids
            near_miss = (not correct and
                         any(is_parent_child(predicted_id, gid) for gid in gold_ids))
        if near_miss:
            near_misses += 1
        result = {
            'query_raw': resp['query_raw'],
            'query_norm': resp['query_norm'],
            'predicted_id': predicted_id,
            'gold_ids': gold_ids,
            'correct': correct,
            'near_miss': near_miss,
            'actor': actor,
            'entity_type': entity_type,
            'llm_response': resp['llm_response'],
        }
        all_results.append(result)
        results_by_actor[actor].append(result)
        results_by_type[entity_type].append(result)
        if resp.get('latency_ms', 0) > 0:
            latencies.append(resp['latency_ms'])
    n_total = len(all_results)
    n_correct = sum(1 for r in all_results if r['correct'])
    p_at_1 = n_correct / n_total if n_total > 0 else 0
    n_correct_lenient = sum(1 for r in all_results if r['correct'] or r['near_miss'])
    p_at_1_lenient = n_correct_lenient / n_total if n_total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print(f"\n  {'='*55}")
    print(f"  RESULTS: {model} ({mode})")
    print(f"  {'='*55}")
    print(f"  P@1 (strict):   {p_at_1:.4f}  ({n_correct}/{n_total})")
    print(f"  P@1 (lenient):  {p_at_1_lenient:.4f}  ({n_correct_lenient}/{n_total})")
    print(f"  Near-misses:    {near_misses}")
    print(f"  Parse failures: {parse_failures}")
    print(f"  API errors:     {api_errors}")
    print(f"  Avg latency:    {avg_latency:.0f}ms")

    print(f"\n  PER-ACTOR P@1:")
    print(f"  {'Actor':<14} | {'P@1':>8} | {'Correct':>8} | {'Total':>6}")
    print(f"  {'-'*14}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    actor_metrics = {}
    for actor in sorted(results_by_actor):
        results = results_by_actor[actor]
        c = sum(1 for r in results if r['correct'])
        t = len(results)
        p1 = c / t if t > 0 else 0
        actor_metrics[actor] = {'p_at_1': p1, 'correct': c, 'total': t}
        print(f"  {actor:<14} | {p1:>8.4f} | {c:>8} | {t:>6}")

    print(f"\n  PER-ENTITY-TYPE P@1:")
    print(f"  {'Type':<16} | {'P@1':>8} | {'Correct':>8} | {'Total':>6}")
    print(f"  {'-'*16}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    for etype in sorted(results_by_type):
        results = results_by_type[etype]
        c = sum(1 for r in results if r['correct'])
        t = len(results)
        p1 = c / t if t > 0 else 0
        print(f"  {etype:<16} | {p1:>8.4f} | {c:>8} | {t:>6}")

    print(f"\n  {'='*55}")
    print(f"  COMPARISON TABLE (for paper)")
    print(f"  {'='*55}")
    print(f"  {'System':<22} | {'P@1':>8} | {'Latency':>10} | {'Params':>10} | {'Cloud':>6}")
    print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")
    print(f"  {'BM25 only':<22} | {'75.34%':>8} | {'5ms':>10} | {'0':>10} | {'No':>6}")
    print(f"  {'MiniLM (off-shelf)':<22} | {'57.53%':>8} | {'50ms':>10} | {'22.7M':>10} | {'No':>6}")
    print(f"  {'MiniLM (fine-tuned)':<22} | {'91.10%':>8} | {'55ms':>10} | {'22.7M':>10} | {'No':>6}")
    print(f"  {model:<22} | {p_at_1:>7.2%} | {avg_latency:>8.0f}ms | {'>>1B':>10} | {'Yes':>6}")

    # Save per-query results for McNemar's test
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mcnemar_path = OUTPUT_DIR / f"per_query_{model}_{mode}.csv"
    with open(mcnemar_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['query_norm', 'actor', 'entity_type',
                         'gold_ids', 'predicted_id',
                         'correct', 'near_miss', 'llm_response'])
        for r in all_results:
            writer.writerow([
                r['query_norm'], r['actor'], r['entity_type'],
                '|'.join(r['gold_ids']), r['predicted_id'] or '',
                int(r['correct']), int(r['near_miss']),
                r['llm_response'][:200],
            ])
    print(f"\n  Per-query results: {mcnemar_path}")

    summary = {
        'model': model, 'prompt_mode': mode, 'n_queries': n_total,
        'p_at_1_strict': round(p_at_1, 4),
        'p_at_1_lenient': round(p_at_1_lenient, 4),
        'near_misses': near_misses, 'parse_failures': parse_failures,
        'api_errors': api_errors, 'avg_latency_ms': round(avg_latency, 1),
        'actor_metrics': actor_metrics,
    }
    summary_path = OUTPUT_DIR / f"summary_{model}_{mode}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary:           {summary_path}")

    misses = [r for r in all_results if not r['correct']]
    if misses:
        n_show = min(10, len(misses))
        print(f"\n  SAMPLE MISSES ({n_show} of {len(misses)}):")
        print(f"  {'-'*70}")
        for r in misses[:n_show]:
            print(f"  Query:     {r['query_raw'][:80]}")
            print(f"  Predicted: {r['predicted_id']}")
            print(f"  Gold:      {', '.join(r['gold_ids'])}")
            print(f"  Near-miss: {r['near_miss']}")
            print(f"  Response:  {r['llm_response'][:100]}")
            print(f"  {'-'*70}")
    print(f"\n  [DONE] Phase 3 complete.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM baseline evaluation for CTI-to-ATT&CK mapping")
    parser.add_argument('phase', choices=['prepare', 'run', 'eval', 'all'],
                        help="Which phase to execute")
    parser.add_argument('--data', default='data/reranker_pairs_enriched.jsonl',
                        help="Path to enriched JSONL (for prepare phase)")
    parser.add_argument('--model', default='gpt-5.4-mini',
                        help="OpenAI model to test")
    parser.add_argument('--prompt', dest='prompt_mode',
                        choices=['zero_shot', 'few_shot'], default='zero_shot',
                        help="Prompt strategy")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  LLM BASELINE EVALUATION")
    print("  For: CTI-to-ATT&CK Mapping Journal Paper")
    print("=" * 70)

    if args.phase in ('prepare', 'all'):
        phase_prepare(args)
    if args.phase in ('run', 'all'):
        phase_run(args)
    if args.phase in ('eval', 'all'):
        phase_eval(args)


if __name__ == '__main__':
    main()
