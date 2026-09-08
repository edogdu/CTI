#!/usr/bin/env python3
"""
Option 2 kit, part 1: local bundle builder (lean — no checkpoint needed).
Run from the reranker directory:
  cd C:\\Users\\shane\\Downloads\\CTI\\graph_alignment\\reranker
  python annoctr_eval\\build_joint_bundle.py
Writes: annoctr_eval\\joint_colab_bundle.zip (~5 MB)
"""
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = [
    (HERE / 'stage3' / 'annoctr_stage3_train.jsonl', 'annoctr_stage3_train.jsonl'),
    (HERE / 'stage3' / 'annoctr_stage3_val.jsonl', 'annoctr_stage3_val.jsonl'),
    (HERE / 'sweep_dev' / 'annoctr_queries_dev_mf.jsonl', 'annoctr_queries_dev_mf.jsonl'),
    (HERE / 'sweep_dev' / 'attack_hierarchy.json', 'attack_hierarchy.json'),
    (HERE / 'annoctr_eval.py', 'annoctr_eval.py'),
    (HERE / 'joint_retrain_colab.py', 'joint_retrain_colab.py'),
]

missing = [str(s) for s, _ in FILES if not s.exists()]
if missing:
    sys.exit('[FAIL] missing inputs:\n  ' + '\n  '.join(missing))
out = HERE / 'joint_colab_bundle.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for src, arc in FILES:
        z.write(src, arc)
        print('[ok] +', arc)
print(f'\nWrote {out} ({out.stat().st_size/1e6:.1f} MB)')
print('Upload THIS ONE FILE to Colab (Files pane).')
