## extraction_semantic_chunk_v2.py
# Semantic-chunk CTI triple extraction with:
# - Embedding-based chunking (max-min, self-contained)
# - Bigger chunk targets with coalescing
# - Boilerplate filtering
# - Page spillover context for continuity
# - Chunk-only LLM extraction with optional consensus
# - MALONT-lite validation
# - Embedding cache (on-disk)
# - Chunk previews and rich metrics

import os, re, json, time, math, hashlib, pickle
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import requests
import spacy
import numpy as np
from spacy.matcher import PhraseMatcher
from docling.document_converter import DocumentConverter
from langchain_community.llms import Ollama

# ---------- optional consensus helper ----------
try:
    from triple_extraction.consensus import consensus_filter
    _HAVE_CONSENSUS = True
except Exception:
    _HAVE_CONSENSUS = False


# =========================
# Config and utilities
# =========================

def ensure_ollama_model(model_name="gemma2:9b", base_url="http://localhost:11434"):
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=20)
        resp.raise_for_status()
        models = [m.get("name") for m in resp.json().get("models", [])]
        if model_name not in models:
            print(f"[ollama] Model '{model_name}' not found. Pulling...")
            pull_resp = requests.post(f"{base_url}/api/pull", json={"name": model_name}, timeout=1800)
            pull_resp.raise_for_status()
            print(f"[ollama] Model '{model_name}' pulled.")
        else:
            print(f"[ollama] Model '{model_name}' available.")
    except Exception as e:
        print(f"[warn] Ollama check failed: {e}")


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


# =========================
# NLP setup
# =========================

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")


# =========================
# Boilerplate filtering
# =========================

_BOILERPLATE_PATTERNS = [
    r"\b(cookie|cookies|privacy|manage cookies|accept|reject)\b",
    r"\b(all microsoft|sign in|search the blog|more|solutions|products|services)\b",
    r"\bsubscribe\b|\bshare\b|\bread more\b|\bheader\b|\bfooter\b",
    r"^\s*[A-Z]{2,}(?:\s+[A-Z]{2,})*\s*$",           # ALL CAPS menus
    r"^\s*©\s*\d{4}\s",                               # copyright
    r"^\s*[\uE000-\uF8FF\uf000-\uf8ff].*$",          # icon fonts
]

_BP = [re.compile(p, re.I) for p in _BOILERPLATE_PATTERNS]

def is_boilerplate(sent: str) -> bool:
    s = sent.strip()
    if len(s) < 8:
        return True
    return any(rx.search(s) for rx in _BP)

def filter_sentences(sentences: List[str]) -> List[str]:
    return [s for s in sentences if not is_boilerplate(s)]


# =========================
# MALONT-lite schema
# =========================

MALONT_CLASSES = [
    'Staging','Adware','CommandAndControl','Spyware','DDoS','DomainName','Dropper','Port','MD5',
    'Protocol','VirusScanner','Downloader','Ransomware','OperatingSystem','Rootkit',
    'AttackPattern_SmallDescription','IPAddress','Bootkit','Hardware','SSDeep','Application',
    'AttackPattern','Phishing','Campaign','SHA-256','System','Vulnerability_Desc','Anonymization',
    'Backdoor','Location','Organization','Reconnaissance','Exploit-kit','Time','MalwareAnalysis',
    'ResourceExploitation','SHA','HostingMalware','SHA-1','Unknown','HostingTargetLists','Hash',
    'AttackPattern_LargeDescription','Software','Network','Indicator','Trojan','Botnet','Worm',
    'EmailAddress','Malware','RogueSecuritySoftware','vHash','Filepath','Region','Report','Virus',
    'ThreatActor','Keylogger','Browser','ScreenCapture','Vulnerability_CVEID','URL','Wiper',
    'Filename','Infrastructure','MalwareFamily','Person','Webshell','Vulnerability','Bot',
    'RemoteAccessTrojan-RAT','Country','Exfiltration','Amplification'
]

MALONT_PREDS = [
    "targets","communicatesWith","uses","has","hasAlias","hasVulnerability",
    "indicates","exploits","hasAuthor","belongsTo"
]

# seed matcher for cheap relevance checks
try:
    matcher.add("MALONT", [nlp.make_doc(term) for term in MALONT_CLASSES])
except Exception:
    pass

def is_valid_triple(t: dict) -> bool:
    if not isinstance(t, dict):
        return False
    s, o, p = t.get("subject"), t.get("object"), t.get("predicate")
    if not isinstance(s, dict) or not isinstance(o, dict) or not isinstance(p, str):
        return False
    st, ot = (s.get("type"), o.get("type"))
    sn, on = (s.get("name"), o.get("name"))
    if not sn or not on:
        return False
    if st not in MALONT_CLASSES or ot not in MALONT_CLASSES:
        return False
    if p.strip() not in MALONT_PREDS:
        return False
    q = (t.get("evidence") or {}).get("quote", "")
    if not isinstance(q, str) or not q.strip():
        return False
    return True


# =========================
# Embedding client with cache
# =========================

class Embedder:
    def __init__(self, base_url="http://localhost:11434", model="nomic-embed-text",
                 cache_dir=".embed_cache", batch_size=64, timeout=60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_index_path = os.path.join(cache_dir, "index.pkl")
        self.cache: Dict[str, List[float]] = {}
        if os.path.exists(self.cache_index_path):
            try:
                with open(self.cache_index_path, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception:
                self.cache = {}

    def save_cache(self):
        try:
            with open(self.cache_index_path, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception:
            pass

    def _embed_one(self, text: str) -> List[float]:
        try:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            r.raise_for_status()
            v = r.json().get("embedding") or []
            if not v:
                return []
            # L2 normalize
            arr = np.array(v, dtype=float)
            n = np.linalg.norm(arr) or 1.0
            return (arr / n).tolist()
        except Exception:
            return []

    def embed(self, sentences: List[str], doc_id="") -> List[List[float]]:
        out: List[List[float]] = []
        to_compute_idx = []
        # locate in cache
        for i, s in enumerate(sentences):
            key = f"{doc_id}:{sha1(s)}"
            if key in self.cache:
                out.append(self.cache[key])
            else:
                out.append(None)
                to_compute_idx.append((i, key, s))

        # batch compute remaining
        for start in range(0, len(to_compute_idx), self.batch_size):
            batch = to_compute_idx[start:start+self.batch_size]
            for i, key, s in batch:
                vec = self._embed_one(s)
                if vec:
                    self.cache[key] = vec
                out[i] = vec if vec else None
        return out


# =========================
# Semantic chunking (max-min)
# =========================

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a/na, b/nb))

def maxmin_semantic_chunks(
    sentences: List[str],
    embeddings: List[List[float]],
    min_sim_threshold: float = 0.25,
    max_words: int = 380,
    overlap_words: int = 120,
    lookahead: int = 6,
) -> List[Dict[str, Any]]:
    """
    Simple max-min chunker:
    - Start a chunk
    - iteratively add sentences whose sim to current centroid >= threshold
    - stop when max_words reached
    - add overlap_words context between chunks by sentence spill
    """
    # pack valid vectors
    vecs = [np.array(v, dtype=np.float32) if isinstance(v, list) else None for v in embeddings]
    n = len(sentences)
    chunks = []
    i = 0
    while i < n:
        # start a new chunk at i
        start = i
        # skip empty or extremely short sentences
        cur_sents = [sentences[i]]
        cur_vecs = [vecs[i]] if vecs[i] is not None else []
        # running centroid
        if cur_vecs:
            centroid = np.mean(cur_vecs, axis=0)
        else:
            centroid = None

        # grow the chunk
        j = i + 1
        while j < n:
            # compute tentative size
            new_text = " ".join(cur_sents + [sentences[j]])
            if len(new_text.split()) > max_words:
                break
            # check similarity
            ok = True
            if centroid is not None and vecs[j] is not None:
                sim = cosine(centroid, vecs[j])
                if sim < min_sim_threshold:
                    ok = False
            # allow a small lookahead to keep topical continuity
            if not ok and lookahead > 0:
                window = min(n, j + lookahead)
                found = False
                for k in range(j+1, window):
                    if vecs[k] is not None and centroid is not None:
                        if cosine(centroid, vecs[k]) >= min_sim_threshold:
                            found = True
                            break
                if not found:
                    break

            # accept sentence j
            cur_sents.append(sentences[j])
            if vecs[j] is not None:
                cur_vecs.append(vecs[j])
                centroid = np.mean(cur_vecs, axis=0)
            j += 1

        # record chunk
        text = " ".join(cur_sents).strip()
        size_words = len(text.split())
        # min pair sim inside chunk for preview
        mps = 1.0
        if len(cur_vecs) >= 2:
            mps = 1.0
            for a in range(len(cur_vecs)):
                for b in range(a+1, len(cur_vecs)):
                    mps = min(mps, cosine(cur_vecs[a], cur_vecs[b]))
        chunks.append({
            "start": start,
            "end": j,
            "text": text,
            "size_words": size_words,
            "min_pair_sim": mps
        })

        # overlap: step back by overlap_words worth of sentences
        if j >= n:
            break
        # compute how many sentences approximate overlap_words
        back = 0
        wcount = 0
        k = len(cur_sents) - 1
        while k >= 0 and wcount < overlap_words:
            wcount += len(cur_sents[k].split())
            back += 1
            k -= 1
        # next chunk starts with last back sentences minus 1 to avoid zero progress
        i = max(start + len(cur_sents) - back, start + 1)

    return chunks


# =========================
# Extractor
# =========================

class CyberTripleExtractor:
    def __init__(
        self,
        file_path: str,
        model_name="gemma2:9b",
        ollama_base_url="http://localhost:11434",
    ):
        self.file_path = file_path
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url

        # IO
        self.converter = DocumentConverter()
        ensure_ollama_model(model_name, ollama_base_url)
        self.llm = Ollama(
            model=model_name,
            base_url=ollama_base_url,
            num_ctx=3072,
            format="json",
            stop=["</think>", "<think>"],
        )

        # Embeddings
        self.embedder = Embedder(
            base_url=ollama_base_url,
            model="nomic-embed-text",
            cache_dir=".embed_cache",
            batch_size=64,
            timeout=60
        )

        # Chunking knobs
        self.MM_MIN_SIM_THRESH = 0.25
        self.MM_MAX_WORDS = 380
        self.MM_OVERLAP_WORDS = 140
        self.MM_LOOKAHEAD = 6

        # Coalescing target sizes
        self.TARGET_MIN_WORDS = 160
        self.TARGET_MAX_WORDS = 420

        # Page spillover
        self.SPILLOVER_SENTENCES = 2

        # Extraction
        self.MAX_RELATIONS_PER_CHUNK = 18
        self.CHUNK_CONTEXT_LIMIT = 2200
        self.PROMPT_VARIANTS = 3

        # Scoping
        self.PAGE_LIMIT = None  # set to an int to debug fewer pages

        # Metrics
        self.pages_parsed = 0
        self.raw_triples = 0
        self.valid_triples = []
        self.suspicious_triples = 0
        self.runtime_seconds = 0
        self.rejection_stats = {"bad_structure": 0, "invalid_class_or_predicate": 0}
        self.chunk_data = []
        self._chunks_preview = []

    # ---------- relevance ----------
    def _ner_hit(self, text):
        doc = nlp(text)
        labels = {e.label_ for e in doc.ents}
        return bool(labels & {"ORG","PRODUCT","GPE","DATE"})

    def _onto_hit(self, text):
        try:
            return len(matcher(nlp(text))) > 0
        except Exception:
            return False

    def _chunk_relevant(self, sentences: List[str]) -> bool:
        block = " ".join(sentences).lower()
        has_kw = any(
            kw in block for kw in [
                "apt","malware","c2","command and control","phishing","ransomware",
                "exploit","cve-","lateral movement","persistence","beacon","cobalt strike",
                "exfiltration","loader","backdoor","webshell","ioc","infrastructure"
            ]
        )
        ner = any(self._ner_hit(s) for s in sentences)
        onto = any(self._onto_hit(s) for s in sentences)
        return has_kw or ner or onto

    # ---------- doc to page text ----------
    def chunk_by_page(self, doc):
        page_chunks = defaultdict(str)
        for ti in doc.texts:
            if not getattr(ti, "prov", None):
                continue
            if hasattr(ti, "content_layer") and ti.content_layer != "body":
                continue
            try:
                p = ti.prov[0].page_no
                page_chunks[p] += " " + ti.text.strip()
            except Exception:
                continue
        return sorted(page_chunks.items())

    # ---------- coalesce undersized chunks ----------
    def _coalesce_short(self, chunks, sentences, min_words=160, max_words=420):
        if not chunks:
            return []
        out = []
        buf = None
        for ch in chunks:
            start, end = int(ch["start"]), int(ch["end"])
            text = ch.get("text") or " ".join(sentences[start:end])
            size = len(text.split())
            if buf is None:
                buf = {"start": start, "end": end, "text": text, "size_words": size, "min_pair_sim": ch.get("min_pair_sim")}
                continue
            if buf["size_words"] < min_words or (size < min_words and buf["size_words"] + size <= max_words):
                buf["end"] = end
                buf["text"] = (buf["text"] + " " + text).strip()
                buf["size_words"] = len(buf["text"].split())
                mp = ch.get("min_pair_sim")
                if buf.get("min_pair_sim") is None or (mp is not None and mp < buf["min_pair_sim"]):
                    buf["min_pair_sim"] = mp
            else:
                out.append(buf)
                buf = {"start": start, "end": end, "text": text, "size_words": size, "min_pair_sim": ch.get("min_pair_sim")}
        if buf:
            out.append(buf)
        # hard cap
        final = []
        for ch in out:
            if ch["size_words"] <= max_words:
                final.append(ch)
            else:
                s_idx, e_idx = ch["start"], ch["end"]
                cur, cur_words, cur_start = [], 0, s_idx
                for i in range(s_idx, e_idx):
                    w = len(sentences[i].split())
                    if cur and cur_words + w > max_words:
                        txt = " ".join(cur)
                        final.append({"start": cur_start, "end": i, "text": txt, "size_words": len(txt.split()), "min_pair_sim": ch.get("min_pair_sim")})
                        cur, cur_words, cur_start = [], 0, i
                    cur.append(sentences[i])
                    cur_words += w
                if cur:
                    txt = " ".join(cur)
                    final.append({"start": cur_start, "end": e_idx, "text": txt, "size_words": len(txt.split()), "min_pair_sim": ch.get("min_pair_sim")})
        return final

    # ---------- prompts ----------
    def _schema(self):
        return """
[
  {
    "subject": {"name": "<string>", "type": "<type-from-list>"},
    "predicate": "<predicate-from-list>",
    "object": {"name": "<string>", "type": "<type-from-list>"},
    "evidence": {"quote": "<exact substring from the text>"}
  }
]
""".strip()

    def _prompts_for_chunk(self, chunk_text: str) -> List[str]:
        types = ", ".join(MALONT_CLASSES)
        preds = ", ".join(MALONT_PREDS)
        schema = self._schema()
        snippet = chunk_text[: self.CHUNK_CONTEXT_LIMIT]

        p1 = f"""
Extract up to {self.MAX_RELATIONS_PER_CHUNK} explicit CTI triples across the text.
Use ONLY types in [{types}] and predicates in [{preds}].
Each triple must include evidence.quote as an exact substring from the text.
Text:
\"\"\"{snippet}\"\"\"
Output {schema}
""".strip()

        p2 = f"""
Work predicate-first across the entire text. Do not output uncertain fields.
Types: [{types}]
Predicates: [{preds}]
Text:
\"\"\"{snippet}\"\"\"
Output {schema}
""".strip()

        p3 = f"""
Extract ALL explicit CTI triples. No guessing. Exact entity names only.
Types allowed: [{types}]
Predicates allowed: [{preds}]
Each triple must include an evidence.quote as an exact substring.
Text:
\"\"\"{snippet}\"\"\"
Output {schema}
""".strip()

        if self.PROMPT_VARIANTS == 1:
            return [p1]
        if self.PROMPT_VARIANTS == 2:
            return [p1, p3]
        return [p1, p2, p3]

    # ---------- JSON parse helpers ----------
    _JARR = re.compile(r"\[[\s\S]*\]")
    _JOBJ = re.compile(r"\{[\s\S]*\}")

    def _parse_any_json(self, resp) -> Any:
        try:
            if isinstance(resp, (list, dict)):
                return resp
            s = str(resp)
            try:
                return json.loads(s)
            except Exception:
                pass
            m = self._JARR.search(s)
            if m:
                return json.loads(m.group(0))
            m = self._JOBJ.search(s)
            if m:
                return json.loads(m.group(0))
        except Exception:
            return None
        return None

    def _as_triple_list(self, obj):
        if obj is None:
            return []
        if isinstance(obj, list):
            return [t for t in obj if isinstance(t, dict)]
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, str) and obj.strip().upper() in {"NO_TRIPLES","NONE"}:
            return []
        return []

    # ---------- run ----------
    def run(self):
        t0 = time.time()
        print(f"[load] Converting document {os.path.basename(self.file_path)}")
        result = self.converter.convert(self.file_path)
        doc = result.document

        page_chunks = self.chunk_by_page(doc)
        self.pages_parsed = len(page_chunks)
        print(f"[load] Found {self.pages_parsed} pages")

        # page spillover buffer
        prev_page_tail: List[str] = []

        chunk_results = []

        for page_no, page_text in page_chunks:
            if self.PAGE_LIMIT and page_no > self.PAGE_LIMIT:
                print(f"[info] Stopping at PAGE_LIMIT={self.PAGE_LIMIT}")
                break

            # split sentences
            doc_spacy = nlp(page_text)
            sentences = [s.text for s in doc_spacy.sents]

            # prepend spillover sentences from previous page for embedding continuity
            spillover_prefix = prev_page_tail.copy()
            full_for_embed = spillover_prefix + sentences

            # filter boilerplate after building full list so spillover can match
            sentences = filter_sentences(sentences)
            full_for_embed = filter_sentences(full_for_embed)

            if not sentences:
                prev_page_tail = []
                continue

            print(f"[chunking] Page {page_no}: {len(sentences)} sentences after filtering")
            # embed
            doc_id = f"{os.path.basename(self.file_path)}:p{page_no}"
            vecs = self.embedder.embed(full_for_embed, doc_id=doc_id)

            # slice back to current page portion
            spill = len(spillover_prefix)
            vecs = vecs[spill:] if spill > 0 else vecs

            # build semantic chunks
            start_t = time.time()
            chunks = maxmin_semantic_chunks(
                sentences=sentences,
                embeddings=vecs,
                min_sim_threshold=self.MM_MIN_SIM_THRESH,
                max_words=self.MM_MAX_WORDS,
                overlap_words=self.MM_OVERLAP_WORDS,
                lookahead=self.MM_LOOKAHEAD
            )
            # coalesce to target sizes
            chunks = self._coalesce_short(chunks, sentences, self.TARGET_MIN_WORDS, self.TARGET_MAX_WORDS)
            dt = time.time() - start_t
            print(f"[chunking] Page {page_no}: built {len(chunks)} chunks in {dt:.1f}s")

            # record preview
            self._chunks_preview.append({
                "page": page_no,
                "num_chunks": len(chunks),
                "chunks": [
                    {
                        "chunk_id": idx,
                        "start_sent_idx": int(ch["start"]),
                        "end_sent_idx": int(ch["end"]),
                        "size_words": ch.get("size_words"),
                        "min_pair_sim": ch.get("min_pair_sim"),
                        "text_preview": (ch.get("text","")[:260] + "...") if len(ch.get("text","")) > 263 else ch.get("text","")
                    } for idx, ch in enumerate(chunks)
                ]
            })

            # extract per chunk
            for ch_id, ch in enumerate(chunks):
                s_idx, e_idx = int(ch["start"]), int(ch["end"])
                ch_sents = sentences[s_idx:e_idx]
                if not self._chunk_relevant(ch_sents):
                    continue

                chunk_text = ch.get("text","")
                prompts = self._prompts_for_chunk(chunk_text)
                lists = []
                for p in prompts:
                    try:
                        resp = self.llm.invoke(p)
                        parsed = self._parse_any_json(resp)
                        triples = self._as_triple_list(parsed)
                    except Exception:
                        triples = []
                    lists.append(triples)
                    self.raw_triples += len(triples)

                if _HAVE_CONSENSUS:
                    fused = consensus_filter(
                        lists,
                        allowed_types=MALONT_CLASSES,
                        allowed_preds=MALONT_PREDS,
                        m=2,
                        tau_name=0.90
                    )
                else:
                    # simple union
                    fused, seen = [], set()
                    for lst in lists:
                        for t in lst:
                            k = json.dumps(t, sort_keys=True)
                            if k not in seen:
                                seen.add(k)
                                fused.append(t)

                accepted = []
                for t in fused:
                    if is_valid_triple(t):
                        tt = dict(t)
                        tt["_validity"] = "strict"
                        tt["_chunk"] = {
                            "page": page_no,
                            "chunk_id": ch_id,
                            "start": s_idx,
                            "end": e_idx,
                            "size_words": ch.get("size_words"),
                            "min_pair_sim": ch.get("min_pair_sim")
                        }
                        accepted.append(tt)
                        self.valid_triples.append(tt)
                        print(f"[ACCEPT] {tt['subject']} —{tt['predicate']}→ {tt['object']}  [p{page_no} c{ch_id}]")
                    else:
                        self.suspicious_triples += 1

                if accepted:
                    chunk_results.append(("[CHUNK]", page_no, ch_id, accepted))

            # prepare spillover for next page
            prev_page_tail = sentences[-self.SPILLOVER_SENTENCES:] if len(sentences) >= self.SPILLOVER_SENTENCES else sentences

        # persist cache
        self.embedder.save_cache()
        self.runtime_seconds = time.time() - t0
        return chunk_results

    # ---------- build + save ----------
    def build_dict(self, chunk_results):
        self.chunk_data = []
        for ctx, page_no, ch_id, triples in chunk_results:
            valids = [t for t in triples if t.get("_validity") == "strict" or is_valid_triple(t)]
            if not valids:
                continue
            self.chunk_data.append({
                "context": ctx,
                "triple": valids,
                "metadata": {
                    "page_number": page_no,
                    "id": str(ch_id).zfill(3),
                    "source": "CHUNK"
                }
            })
        return self.chunk_data

    def _write_chunks_preview(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        jp = os.path.join(out_dir, "chunks_preview.json")
        tp = os.path.join(out_dir, "chunks_preview.txt")
        try:
            with open(jp, "w", encoding="utf-8") as f:
                json.dump(self._chunks_preview, f, indent=2, ensure_ascii=False)
            with open(tp, "w", encoding="utf-8") as f:
                for page in self._chunks_preview:
                    print(f"Page {page['page']} — {page['num_chunks']} chunks", file=f)
                    for ch in page["chunks"]:
                        print(f"  [Chunk {ch['chunk_id']}] sidx={ch['start_sent_idx']} eidx={ch['end_sent_idx']} size={ch['size_words']} min_pair_sim={ch['min_pair_sim']}", file=f)
                        print(f"    preview: {ch['text_preview']}", file=f)
                    print("", file=f)
            print(f"[write] Chunk previews → {jp} and {tp}")
        except Exception as e:
            print(f"[warn] Failed to write chunk previews: {e}")

    def safe_filename(self, name: str) -> str:
        return re.sub(r'[<>:\"/\\|?*]', '_', name)

    def save_to_json(self, output_filename="chunk_data.json", base_dir=None):
        try:
            # Extract document name (no extension)
            doc_name = os.path.splitext(os.path.basename(self.file_path))[0]

            # Get full path and split into parts
            abs_path = os.path.abspath(self.file_path)
            parts = abs_path.split(os.sep)

            # Detect base branch folder
            anchor_folders = ["dataset", "similarity_scoring", "vec_results"]
            anchor_index = None
            for folder in anchor_folders:
                if folder in parts:
                    anchor_index = parts.index(folder)
                    break

            # Mirror structure under /output/
            if anchor_index is not None:
                rel_parts = parts[anchor_index + 1:-1]  # e.g., CTI-HAL/apt29
                rel_path = os.path.join(*rel_parts) if rel_parts else ""
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                output_dir = os.path.join(project_root, "output", rel_path, doc_name, "vec_results")
            else:
                # fallback
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                output_dir = os.path.join(project_root, "output", doc_name, "vec_results")

            # Allow override
            if base_dir:
                output_dir = base_dir

            os.makedirs(output_dir, exist_ok=True)
            filename = self.safe_filename(os.path.basename(output_filename))
            output_path = os.path.join(output_dir, filename)

            # ---- METRICS ----
            strict = sum(
                1
                for item in self.chunk_data
                for t in item.get("triple", [])
                if t.get("_validity") == "strict"
            )

            metrics = {
                "file_name": os.path.basename(self.file_path),
                "model_used": self.model_name,
                "num_pages": self.pages_parsed,
                "num_raw_triples": self.raw_triples,
                "num_valid_triples": len(self.valid_triples),
                "num_suspicious_triples": self.suspicious_triples,
                "runtime_seconds": self.runtime_seconds,
                "num_valid_triples_strict": strict
            }

            payload = {"metrics": metrics, "data": self.chunk_data}

            # ---- SAVE ----
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            # Save chunk previews in same directory
            self._write_chunks_preview(output_dir)

            print(f"[write] Results saved to: {output_path}")

        except Exception as e:
            print(f"[warn] Failed to save JSON: {e}")

if __name__ == "__main__":
    # Adjust path and model as needed
    pdf_path = "cti-analysis/AnalysisOfCyberattackOnUS.pdf"
    extractor = CyberTripleExtractor(
        pdf_path,
        model_name="gemma2:9b",
        ollama_base_url="http://localhost:11434"
    )
    results = extractor.run()
    extractor.build_dict(results)
    out_name = extractor.safe_filename(f"chunk_data_{extractor.model_name}.json")
    # Save all results to the unified output folder
    extractor.save_to_json(
        output_filename=out_name,
        base_dir=os.path.join("output", "extracted_triples")
    )
