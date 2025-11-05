# cti/datasets/loaders.py
from __future__ import annotations
import os, re, json, random, subprocess
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Callable, Optional

# ---------------- CTI-HAL ----------------
def ensure_cti_hal(dst: str):
    """Clone CTI-HAL if missing."""
    if not os.path.exists(dst):
        print("Cloning CTI-HAL...")
        subprocess.run(f"git clone https://github.com/dessertlab/CTI-HAL.git {dst}", shell=True, check=False)
    else:
        print(" CTI-HAL already present.")
    print("CTI-HAL ready at:", dst)

def load_cti_hal_samples(dataset_path: str, num_samples: int = 10) -> List[Dict]:
    data_path = Path(dataset_path) / 'data'
    if not data_path.exists():
        print(" CTI-HAL 'data' folder missing:", data_path)
        return []
    json_files = sorted(list(data_path.rglob("*.json")))
    if not json_files:
        print(" No CTI-HAL json files found under", data_path)
        return []

    random.seed(42)
    sample_files = random.sample(json_files, min(num_samples, len(json_files)))
    samples = []
    for jf in sample_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                anns = json.load(f)
            threat_actor = jf.parent.parent.name if 'annotator' not in str(jf) else jf.parent.parent.parent.name
            contexts, techniques, tools = [], [], []
            for ann in anns:
                if 'context' in ann: contexts.append(ann['context'])
                if 'technique' in ann:
                    tech = ann['technique']
                    if isinstance(tech, list): techniques.extend(tech)
                    else: techniques.append(tech)
                if 'metadata' in ann and ann['metadata'].get('tool_name'):
                    t = ann['metadata']['tool_name']
                    if t and t not in tools: tools.append(t)
            text = " ".join(contexts)

            gt_triples = []
            if tools:
                for tool in tools:
                    for tech in set(techniques):
                        gt_triples.append((str(tool).lower(), "executes", str(tech).lower()))
            else:
                for tech in set(techniques):
                    if threat_actor:
                        gt_triples.append((str(threat_actor).lower(), "uses", str(tech).lower()))

            samples.append({
                "id": jf.stem,
                "file_path": str(jf),
                "threat_actor": threat_actor,
                "text": text,
                "techniques": list(set(techniques)),
                "tools": tools,
                "annotations": anns,
                "dataset": "CTI-HAL",
                "ground_truth_triples": gt_triples
            })
        except Exception as e:
            print(f"   error reading {jf}: {e}")
    print(f" Loaded {len(samples)} CTI-HAL samples")
    return samples

# ---------------- ANNO-CTR ----------------
def ensure_anno_ctr(dst: str):
    """Clone ANNO-CTR if missing."""
    if not os.path.exists(dst):
        print("Cloning ANNO-CTR...")
        subprocess.run(f"git clone https://github.com/boschresearch/anno-ctr-lrec-coling-2024.git {dst}", shell=True, check=False)
    else:
        print(" ANNO-CTR already present.")
    print("ANNO-CTR ready at:", dst)

def _guess_anno_ctr_json(base: str, split: str) -> Optional[Path]:
    basep = Path(base)
    candidates = [
        basep / "ner_json" / f"{split}.json",
        basep / "data" / "ner_json" / f"{split}.json",
        basep / f"{split}.json",
        basep / "AnnoCTR" / "ner_json" / f"{split}.json",
        basep / "AnnoCTR" / f"{split}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def _find_any_anno_ctr_json(base: str) -> Optional[Path]:
    """If split files are not found, return the first json under any ner_json folder."""
    basep = Path(base)
    for p in basep.rglob("ner_json"):
        js = sorted(p.glob("*.json"))
        if js:
            return js[0]
    # as a last resort, any json under base
    any_jsons = sorted(basep.rglob("*.json"))
    return any_jsons[0] if any_jsons else None

_ANNOLBL_TO_STIX = {
    "ORG": "identity", "LOC": "location", "TOOL": "tool", "MALWARE": "malware",
    "TECHNIQUE": "attack-pattern", "TACTIC": "tactic", "SECTOR": "identity",
}
_DEFAULT_ID2BIO = {
    0:'O', 1:'B-ORG', 2:'I-ORG', 3:'B-LOC', 4:'I-LOC', 5:'B-TOOL', 6:'I-TOOL',
    7:'B-MALWARE', 8:'I-MALWARE', 9:'B-TECHNIQUE', 10:'I-TECHNIQUE', 11:'B-TACTIC',
    12:'I-TACTIC', 13:'B-SECTOR', 14:'I-SECTOR'
}

def _annoctr_id2bio_from_features(payload_first_item: Any) -> Dict[int, str] | None:
    try:
        feats = payload_first_item.get('features') or {}
        ner = feats.get('ner_tags') or feats.get('ner') or None
        if not ner: return None
        id2lab = {int(i): str(l) for i, l in enumerate(ner.get('labels', []))}
        if any('-' in v for v in id2lab.values()):
            return id2lab
        return None
    except Exception:
        return None

def _bio_tag_to_type(tag: str) -> str | None:
    if not tag or tag == 'O': return None
    m = re.match(r'^[BI]-(.+)$', tag.upper())
    if not m: return None
    raw = m.group(1).strip()
    return _ANNOLBL_TO_STIX.get(raw, None)

def _extract_entities_from_bio(tokens: List[str], bio_seq: List[str]) -> List[Dict]:
    ents, cur_tokens, cur_type, start = [], [], None, None
    for i, (tok, tag) in enumerate(zip(tokens, bio_seq)):
        if tag == 'O' or not tag:
            if cur_tokens:
                ents.append({"text": " ".join(cur_tokens), "type": cur_type, "start": start, "end": i})
                cur_tokens, cur_type, start = [], None, None
            continue
        if tag.startswith('B-'):
            if cur_tokens:
                ents.append({"text": " ".join(cur_tokens), "type": cur_type, "start": start, "end": i})
            cur_type = _bio_tag_to_type(tag); cur_tokens = [tok]; start = i
        elif tag.startswith('I-'):
            t2 = _bio_tag_to_type(tag)
            if t2 and t2 == cur_type and cur_tokens:
                cur_tokens.append(tok)
            else:
                if cur_tokens:
                    ents.append({"text": " ".join(cur_tokens), "type": cur_type, "start": start, "end": i})
                cur_type = t2; cur_tokens = [tok]; start = i
        else:
            if cur_tokens:
                ents.append({"text": " ".join(cur_tokens), "type": cur_type, "start": start, "end": i})
            cur_tokens, cur_type, start = [], None, None
    if cur_tokens:
        ents.append({"text": " ".join(cur_tokens), "type": cur_type, "start": start, "end": len(tokens)})
    return [e for e in ents if e.get("type")]

def _anno_ctr_to_triples_stix(entities: List[Dict]) -> List[Tuple[str,str,str]]:
    triples = []
    by_type = defaultdict(list)
    for e in entities:
        t = (e.get("type") or "").lower()
        if t: by_type[t].append(e)
    for src in by_type.get("tool", []) + by_type.get("malware", []):
        for tech in by_type.get("attack-pattern", []):
            triples.append((src["text"].lower(), "uses", tech["text"].lower()))
    for tech in by_type.get("attack-pattern", []):
        for tact in by_type.get("tactic", []):
            triples.append((tech["text"].lower(), "part-of", tact["text"].lower()))
    main = [e for e in entities if (e.get("type") or "").lower() in {"malware","tool","attack-pattern","identity"}]
    for i, e1 in enumerate(main):
        for e2 in main[i+1:i+3]:
            if e1["text"].lower() != e2["text"].lower():
                triples.append((e1["text"].lower(), "related-to", e2["text"].lower()))
    return triples

def _coerce_ner_tags_to_bio(tags: List[Any], id2bio: Dict[int, str] | None) -> List[str] | None:
    if not tags: return None
    if isinstance(tags[0], str):
        if all((t == 'O' or re.match(r'^[BI]-[A-Za-z]+$', t)) for t in tags): return tags
        return None
    if isinstance(tags[0], int):
        mapper = id2bio or _DEFAULT_ID2BIO
        return [mapper.get(int(t), 'O') for t in tags]
    return None

def _read_json_or_jsonl(jpath: Path) -> List[Dict]:
    txt = jpath.read_text(encoding="utf-8").strip()
    # try array or dict-with-data
    try:
        data = json.loads(txt)
        if isinstance(data, list): return data
        if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list): return data['data']
    except Exception:
        pass
    # JSONL with tolerant tail
    objs = []
    for line in txt.splitlines():
        line=line.strip()
        if not line: continue
        try:
            objs.append(json.loads(line))
        except Exception:
            # ignore a single bad/truncated tail line
            break
    if objs:
        return objs
    # concatenated objects as array
    repaired = "[" + re.sub(r"}\s*{", "},{", txt) + "]"
    try:
        data = json.loads(repaired)
        if isinstance(data, list): return data
    except Exception:
        pass
    # final: try to close ] or }
    t = txt
    if t.startswith("[") and not t.endswith("]"): t = t + "]"
    if t.startswith("{") and not t.endswith("}"): t = t + "}"
    try:
        data = json.loads(t)
        if isinstance(data, list): return data
        if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list): return data['data']
    except Exception:
        pass
    raise ValueError("Unrecognized JSON/JSONL format")


def _pick_tokens_and_tags(item: Dict[str, Any]) -> Tuple[Optional[List[str]], Optional[List[Any]]]:
    """
    ANNO-CTR (your file) schema:
      tokens: List[str]
      all_tags: List[str] (BIO)  <-- prefer this
      ne_tags, nc_tags, te_tags, ce_tags, ci_tags: List[str] (alternates)
      text: raw string (we won't use it since tokens align with tags)
    """
    tokens = item.get("tokens") or item.get("words") or item.get("sentence")
    if isinstance(tokens, str):
        tokens = tokens.split()
    if not isinstance(tokens, list):
        return None, None

    # Prefer 'all_tags' first
    tag_keys = ("all_tags", "ner_tags", "tags", "labels", "ne_tags", "nc_tags", "te_tags", "ce_tags", "ci_tags")
    for k in tag_keys:
        if k in item and isinstance(item[k], list):
            return tokens, item[k]

    # Sometimes tags are nested in 'features'
    feats = item.get("features") or {}
    for k in ("all_tags", "ner_tags", "ner", "tags", "labels"):
        v = feats.get(k)
        if isinstance(v, dict) and "labels" in v: return tokens, v["labels"]
        if isinstance(v, list): return tokens, v

    return None, None


def load_anno_ctr_samples(base: str, num_samples: int = 10, split: str = "auto") -> List[Dict]:
    # Resolve split path
    jpath = None
    if split and split.lower() != "auto":
        jpath = _guess_anno_ctr_json(base, split)
    if jpath is None:
        for s in ("test","dev","valid","val","train"):
            jpath = _guess_anno_ctr_json(base, s)
            if jpath: break
    if jpath is None:
        jpath = _find_any_anno_ctr_json(base)

    if jpath is None or not jpath.exists():
        print(" ANNO-CTR json not found under", base)
        return []

    # Read JSON/JSONL/concatenated
    try:
        data = _read_json_or_jsonl(jpath)
    except Exception as e:
        print(f" ANNO-CTR: failed to parse {jpath.name}: {e}")
        return []

    if not isinstance(data, list) or not data:
        print(f" ANNO-CTR: {jpath.name} is empty or not a list.")
        return []

    id2bio_feat = _annoctr_id2bio_from_features(data[0])

    out, processed = [], 0
    for item in data:
        if num_samples and processed >= num_samples:
            break

        tokens, tags_raw = _pick_tokens_and_tags(item)
        if not tokens or not tags_raw:
            continue

        bio_seq = _coerce_ner_tags_to_bio(tags_raw, id2bio_feat)
        if not bio_seq:
            if isinstance(tags_raw, dict) and "labels" in tags_raw:
                bio_seq = _coerce_ner_tags_to_bio(tags_raw["labels"], id2bio_feat)
        if not bio_seq:
            continue

        if len(tokens) != len(bio_seq):
            m = min(len(tokens), len(bio_seq))
            tokens, bio_seq = tokens[:m], bio_seq[:m]
            if m == 0: continue

        text = " ".join(tokens)
        ents = _extract_entities_from_bio(tokens, bio_seq)
        triples = _anno_ctr_to_triples_stix(ents)
        out.append({
            "id": f"annoctr_{item.get('id', processed)}",
            "text": text,
            "ground_truth_triples": triples,
            "entities": ents,
            "dataset": "ANNO-CTR"
        })
        processed += 1

    if not out:
        first = data[0] if isinstance(data, list) and data else {}
        print(f" ANNO-CTR: no usable (tokens,BIO) pairs in {jpath}. First item keys: {list(first.keys())}")
        return []

    print(f" Loaded {len(out)} ANNO-CTR samples from {jpath.name}")
    return out

# ---------------- DNRTI ----------------
def ensure_dnrti(dst: str):
    """Clone DNRTI repo if missing (and try to extract rar if present)."""
    if not os.path.exists(dst):
        print("Cloning DNRTI...")
        subprocess.run("git clone https://github.com/SCreaMxp/DNRTI-A-Large-scale-Dataset-for-Named-Entity-Recognition-in-Threat-Intelligence.git {}".format(dst), shell=True, check=False)
    else:
        print(" DNRTI already present:", dst)
    rar = Path(dst) / "DNRTI.rar"
    if rar.exists():
        print(" Found DNRTI.rar - extracting...")
        subprocess.run("apt-get update -qq && apt-get install -y -qq unrar", shell=True, check=False, timeout=180)
        try:
            subprocess.run(f"unrar x -o+ {rar} {dst}/", shell=True, check=True, timeout=300)
            print(" Extraction complete.")
        except subprocess.TimeoutExpired:
            print(" unrar timed out; trying fallback extractor...")
            subprocess.run("apt-get install -y -qq p7zip-full", shell=True, check=False)
            subprocess.run(f"7z x -y -o{dst} {rar}", shell=True, check=False)
    else:
        print("ℹ DNRTI.rar not found; proceeding with raw files.")

def _dnrti_find_split_file(base: str, split: str) -> Optional[Path]:
    basep = Path(base)
    candidates = [
        basep / f"{split}.txt",
        basep / "data" / f"{split}.txt",
        basep / "DNRTI" / f"{split}.txt",
        basep / split / "data.txt",
    ]
    for c in candidates:
        if c.exists(): return c
    # fallback: first .txt under base
    txts = sorted(basep.rglob("*.txt"))
    return txts[0] if txts else None

def _bio_str_to_id(tag: str) -> int:
    tag = tag.upper()
    if tag == "O": return 0
    if tag.startswith("B-"): return 1
    if tag.startswith("I-"): return 2
    return 0

def _entities_from_bio(tokens: List[str], bio_tags: List[int]) -> List[Dict]:
    ents, cur, typ, start = [], [], None, 0
    def flush(i):
        nonlocal ents, cur, typ, start
        if cur:
            ents.append({"text":" ".join(cur), "type": typ or "ENTITY", "start":start, "end":i})
            cur, typ = [], None
    for i,(tok,tid) in enumerate(zip(tokens, bio_tags)):
        if tid == 1:
            flush(i); cur = [tok]; typ = "ENTITY"; start = i
        elif tid == 2 and cur:
            cur.append(tok)
        else:
            flush(i)
    flush(len(tokens))
    return ents

def _dnrti_to_triples(entities: List[Dict]) -> List[Tuple[str,str,str]]:
    triples = []
    main = [e for e in entities]
    for i,e1 in enumerate(main):
        for e2 in main[i+1:i+3]:
            if e1["text"].lower()!=e2["text"].lower():
                triples.append((e1["text"].lower(), "related-to", e2["text"].lower()))
    return triples

def load_dnrti_samples(base: str, num_samples: int = 10, split: str = "test") -> List[Dict]:
    f = _dnrti_find_split_file(base, split)
    if f is None or not f.exists():
        print(f" Could not locate DNRTI {split} file under {base}."); return []
    samples, sent_tokens, sent_tags, count = [], [], [], 0
    def normalize_dnrti_text(t: str) -> str: return " ".join(t.split())
    with open(f, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                if sent_tokens:
                    text = normalize_dnrti_text(" ".join(sent_tokens))
                    ents = _entities_from_bio(sent_tokens, sent_tags)
                    triples = _dnrti_to_triples(ents)
                    samples.append({
                        "id": f"dnrti_{count}",
                        "text": text,
                        "ground_truth_triples": triples,
                        "entities": ents,
                        "dataset": "DNRTI"
                    })
                    count += 1
                    if num_samples and count>=num_samples: break
                    sent_tokens, sent_tags = [], []
            else:
                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts)>=2:
                    w, tag = parts[0], parts[1]
                    sent_tokens.append(w); sent_tags.append(_bio_str_to_id(tag))
    if sent_tokens and (not num_samples or count<num_samples):
        text = normalize_dnrti_text(" ".join(sent_tokens))
        ents = _entities_from_bio(sent_tokens, sent_tags)
        triples = _dnrti_to_triples(ents)
        samples.append({"id": f"dnrti_{count}","text": text,"ground_truth_triples": triples,"entities": ents,"dataset": "DNRTI"})
    print(f" Loaded {len(samples)} DNRTI samples from {f.name}")
    return samples

# ------------- unified access -------------
DatasetLoader = Callable[..., List[Dict]]
def get_dataset_loader(name: str) -> DatasetLoader:
    n = name.strip().lower()
    if n == "cti-hal":
        return lambda base, n_samples=10, **kw: load_cti_hal_samples(base, num_samples=n_samples)
    if n == "anno-ctr":
        return lambda base, n_samples=10, split="auto", **kw: load_anno_ctr_samples(base, num_samples=n_samples, split=split)
    if n == "dnrti":
        return lambda base, n_samples=10, split="test", **kw: load_dnrti_samples(base, num_samples=n_samples, split=split)
    raise ValueError(f"Unknown dataset: {name}")

