#!/usr/bin/env python3
"""
Rehearsal Kit, part 1: local bundle builder
===========================================
Collects everything the Colab unification run needs into ONE upload file.
Run from the reranker directory:

  cd C:\\Users\\shane\\Downloads\\CTI\\graph_alignment\\reranker
  python annoctr_eval\\build_rehearsal_bundle.py

Writes: annoctr_eval\\rehearsal_colab_bundle.zip  (~90 MB; includes the
annoctr_stage3_v1 checkpoint so Colab uses the EXACT certified descendant,
not a retrain).
"""

import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # ...\annoctr_eval
RERANKER = HERE.parent                          # ...\reranker
CKPT = RERANKER / 'checkpoints' / 'annoctr_stage3_v1'
EXPECTED_SAFETENSORS_BYTES = 90_866_404

FILES = [
    (HERE / 'stage3' / 'annoctr_stage3_train.jsonl', 'annoctr_stage3_train.jsonl'),
    (HERE / 'stage3' / 'annoctr_stage3_val.jsonl', 'annoctr_stage3_val.jsonl'),
    (HERE / 'sweep_dev' / 'annoctr_queries_dev_mf.jsonl', 'annoctr_queries_dev_mf.jsonl'),
    (HERE / 'sweep_dev' / 'attack_hierarchy.json', 'attack_hierarchy.json'),
    (HERE / 'annoctr_eval.py', 'annoctr_eval.py'),
    (HERE / 'rehearsal_unify_colab.py', 'rehearsal_unify_colab.py'),
]


def main():
    missing = [str(src) for src, _ in FILES if not src.exists()]
    if not CKPT.exists():
        missing.append(str(CKPT))
    if missing:
        sys.exit('[FAIL] missing inputs:\n  ' + '\n  '.join(missing))

    st = CKPT / 'model.safetensors'
    size = st.stat().st_size
    if size != EXPECTED_SAFETENSORS_BYTES:
        sys.exit(f'[FAIL] {st} is {size} bytes; expected '
                 f'{EXPECTED_SAFETENSORS_BYTES} (the certified annoctr_stage3_v1). '
                 'Wrong or corrupted checkpoint — do not proceed.')
    print(f'[ok] descendant checkpoint verified ({size} bytes)')

    ckpt_files = [p for p in CKPT.iterdir()
                  if p.is_file() and (p.suffix == '.json' or p.name == 'model.safetensors')]
    names = {p.name for p in ckpt_files}
    for req in ('model.safetensors', 'config.json', 'tokenizer.json',
                'tokenizer_config.json'):
        if req not in names:
            sys.exit(f'[FAIL] checkpoint missing required file: {req}')

    out = HERE / 'rehearsal_colab_bundle.zip'
    with zipfile.ZipFile(out, 'w') as z:
        for src, arc in FILES:
            z.write(src, arc, compress_type=zipfile.ZIP_DEFLATED)
            print(f'[ok] + {arc}')
        for p in sorted(ckpt_files):
            ct = zipfile.ZIP_STORED if p.suffix == '.safetensors' \
                else zipfile.ZIP_DEFLATED
            z.write(p, f'annoctr_stage3_v1/{p.name}', compress_type=ct)
            print(f'[ok] + annoctr_stage3_v1/{p.name}')
    print(f'\nWrote {out} ({out.stat().st_size/1e6:.1f} MB)')
    print('Upload THIS ONE FILE to Colab (Files pane), then run the cell in '
          'REHEARSAL_README.md')


if __name__ == '__main__':
    main()
