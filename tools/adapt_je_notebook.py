"""Create JE_BERT_CRF_DNRTI.ipynb — adapted for 19-type DNRTI labels."""
import json, copy

with open("notebooks/JE_BERT_CRF.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

nb2 = copy.deepcopy(nb)

# ── DNRTI label definitions ──────────────────────────────────────────────
DNRTI_TYPES = [
    "APT", "MAL", "TOOL", "ACT", "IDTY", "LOC", "TIME", "FILE",
    "SECTEAM", "OS", "VULID", "VULNAME", "HASH", "DOM", "ENCR",
    "IP", "URL", "PROT", "EMAIL",
]

RE_PREDS = [
    "noRelation", "uses", "usedBy", "targets", "targetedBy",
    "affiliatedWith", "associatedWith", "identifies", "identifiedBy",
    "monitors", "monitoredBy", "hasLocation", "hasAttackLocation",
    "hasAttackTime", "hasVulnerability", "contains",
]

# ── Cell 4: data loading ─────────────────────────────────────────────────
nb2["cells"][4]["source"] = [
    "# Use pre-split DNRTI dataset (19 entity types, 16 predicates)\n",
    "# Split: stratified multi-label (Sechidis et al., 2011)\n",
    "import json\n",
    "\n",
    "data_dir = 'results/finetuning/data'  # Update for Colab/server\n",
    "\n",
    "with open(f'{data_dir}/train_dnrti.json', 'r') as f:\n",
    "    train_data = json.load(f)\n",
    "with open(f'{data_dir}/val_dnrti.json', 'r') as f:\n",
    "    val_data = json.load(f)\n",
    "with open(f'{data_dir}/test_dnrti.json', 'r') as f:\n",
    "    test_data = json.load(f)\n",
    "\n",
    "print(f'Training data size: {len(train_data)}')\n",
    "print(f'Validation data size: {len(val_data)}')\n",
    "print(f'Test data size: {len(test_data)}')\n",
]

# ── Cell 5: skip file writing ────────────────────────────────────────────
nb2["cells"][5]["source"] = [
    "# Data already pre-split — no need to write files\n",
    "pass\n",
]

# ── Cell 10: label mappings ──────────────────────────────────────────────
lines = [
    "import os\n",
    "\n",
    "bert_encoder = BERTTextEncoder(model_name='bert-base-cased')\n",
    "\n",
    "projectName = 'je_bert_crf_dnrti_19types'\n",
    "\n",
    "# --- NER labels: 19 DNRTI types x 2 (B/I) + null + O = 40 ---\n",
    "neid2type = {\n",
    "    0: 'null',\n",
    "    1: 'O',\n",
]
idx = 2
for t in DNRTI_TYPES:
    lines.append(f"    {idx}: 'B-{t}',\n")
    lines.append(f"    {idx+1}: 'I-{t}',\n")
    idx += 2
lines += [
    "}\n",
    "\n",
    "netype2id = {v: k for k, v in neid2type.items()}\n",
    "\n",
    "# --- RE labels: 16 DNRTI predicates + null = 17 ---\n",
    "reid2type = {\n",
    "    0: 'null',\n",
]
for ri, rp in enumerate(RE_PREDS, 1):
    lines.append(f"    {ri}: '{rp}',\n")
lines += [
    "}\n",
    "\n",
    "retype2id = {v: k for k, v in reid2type.items()}\n",
    "\n",
    "print(f'NER labels: {len(neid2type)} ({len(neid2type)-2} BIO tags + null + O)')\n",
    "print(f'RE labels:  {len(reid2type)} ({len(reid2type)-1} predicates + null)')\n",
]
nb2["cells"][10]["source"] = lines

# ── Cell 20: use pre-loaded data ─────────────────────────────────────────
src20 = "".join(nb2["cells"][20]["source"])
if "data_path" in src20 and "json.load" in src20:
    nb2["cells"][20]["source"] = [
        "# Data already loaded from cell 4\n",
        "print(f'Train: {len(train_data)}, Dev: {len(val_data)}, Test: {len(test_data)}')\n",
        "\n",
        "x_train, y_train = re_data_to_input_output(train_data, bert_encoder, max_len)\n",
        "x_val, y_val = re_data_to_input_output(val_data, bert_encoder, max_len)\n",
        "x_test, y_test = re_data_to_input_output(test_data, bert_encoder, max_len)\n",
    ]

# ── Cell 26: auto-balanced oversampling ──────────────────────────────────
nb2["cells"][26]["source"] = [
    "from collections import Counter\n",
    "import numpy as np\n",
    "\n",
    "# Current RE class distribution\n",
    "re_labels_flat = y_train[1].tolist() if hasattr(y_train[1], 'tolist') else list(y_train[1])\n",
    "re_class_counts = Counter(re_labels_flat)\n",
    "print('Original RE class distribution:')\n",
    "for cls_id, count in sorted(re_class_counts.items(), key=lambda x: -x[1]):\n",
    "    label = reid2type.get(cls_id, f'unk_{cls_id}')\n",
    "    print(f'  {label:20s} (id={cls_id}): {count}')\n",
    "\n",
    "# Auto-balance: oversample minority classes to median count\n",
    "counts = [c for cid, c in re_class_counts.items() if cid != 0]\n",
    "median_count = int(np.median(counts))\n",
    "target_min = max(median_count, 2000)\n",
    "\n",
    "target_oversample = {}\n",
    "for cls_id, count in re_class_counts.items():\n",
    "    if cls_id == 0:\n",
    "        continue\n",
    "    target_oversample[cls_id] = max(count, target_min)\n",
    "\n",
    "# Cap noRelation\n",
    "norel_id = retype2id.get('noRelation', 1)\n",
    "if norel_id in target_oversample:\n",
    "    target_oversample[norel_id] = min(\n",
    "        target_oversample[norel_id],\n",
    "        max(target_min * 3, re_class_counts.get(norel_id, 0))\n",
    "    )\n",
    "\n",
    "print(f'\\nTarget oversample (min={target_min}):')\n",
    "for cls_id, target in sorted(target_oversample.items()):\n",
    "    label = reid2type.get(cls_id, f'unk_{cls_id}')\n",
    "    print(f'  {label:20s}: {re_class_counts.get(cls_id, 0)} -> {target}')\n",
    "\n",
    "def oversample_classes(x_data, y_data, targets):\n",
    "    input_ids, attn_mask, ent_mask, ent_type_ids = x_data\n",
    "    ner_labels, re_labels = y_data\n",
    "    input_ids = input_ids.tolist()\n",
    "    attn_mask = attn_mask.tolist()\n",
    "    ent_mask = ent_mask.tolist()\n",
    "    ent_type_ids = ent_type_ids.tolist()\n",
    "    ner_labels = ner_labels.tolist()\n",
    "    re_labels = re_labels.tolist()\n",
    "\n",
    "    by_class = {}\n",
    "    for i, lbl in enumerate(re_labels):\n",
    "        v = lbl if isinstance(lbl, int) else int(np.argmax(lbl))\n",
    "        by_class.setdefault(v, []).append(i)\n",
    "\n",
    "    out = {k: list(v) for k, v in zip(\n",
    "        ['iid','am','em','et','nl','rl'],\n",
    "        [input_ids, attn_mask, ent_mask, ent_type_ids, ner_labels, re_labels]\n",
    "    )}\n",
    "\n",
    "    for cls, tgt in targets.items():\n",
    "        if cls not in by_class:\n",
    "            continue\n",
    "        cur = len(by_class[cls])\n",
    "        if cur >= tgt:\n",
    "            continue\n",
    "        extra = np.random.choice(by_class[cls], size=tgt - cur, replace=True)\n",
    "        for idx in extra:\n",
    "            out['iid'].append(input_ids[idx])\n",
    "            out['am'].append(attn_mask[idx])\n",
    "            out['em'].append(ent_mask[idx])\n",
    "            out['et'].append(ent_type_ids[idx])\n",
    "            out['nl'].append(ner_labels[idx])\n",
    "            out['rl'].append(re_labels[idx])\n",
    "\n",
    "    return (\n",
    "        (np.array(out['iid']), np.array(out['am']),\n",
    "         np.array(out['em']), np.array(out['et'])),\n",
    "        (np.array(out['nl']), np.array(out['rl']))\n",
    "    )\n",
    "\n",
    "x_train_resampled, y_train_resampled = oversample_classes(\n",
    "    x_train, y_train, target_oversample\n",
    ")\n",
    "print(f'\\nAfter resampling: {len(x_train_resampled[0])} samples '\n",
    "      f'(was {len(x_train[0])})')\n",
]

# ── Clear all outputs ────────────────────────────────────────────────────
for cell in nb2["cells"]:
    if "outputs" in cell:
        cell["outputs"] = []
    if "execution_count" in cell:
        cell["execution_count"] = None

# ── Save ─────────────────────────────────────────────────────────────────
out_path = "notebooks/JE_BERT_CRF_DNRTI.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=1, ensure_ascii=False)

print(f"Created {out_path}")
print(f"  Cell  4: data loading -> pre-split DNRTI files")
print(f"  Cell 10: {len(DNRTI_TYPES)} NER types (40 BIO labels), {len(RE_PREDS)} RE predicates")
print(f"  Cell 20: uses pre-loaded data variables")
print(f"  Cell 26: auto-balanced oversampling")
