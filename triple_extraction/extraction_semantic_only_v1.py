# triple_extraction/extraction_semantic_only_v1.py
from __future__ import annotations

import os, re, json, time, hashlib, logging, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

# ----------------------- Optional sentence splitter --------------------------
try:
    import spacy
    _NLP = None
    def _nlp():
        global _NLP
        if _NLP is None:
            try:
                _NLP = spacy.load("en_core_web_sm")
            except Exception:
                _NLP = spacy.blank("en")
                if "sentencizer" not in _NLP.pipe_names:
                    _NLP.add_pipe("sentencizer")
        return _NLP
    _HAVE_SPACY = True
except Exception:
    _HAVE_SPACY = False
    def _nlp(): return None

# ----------------------- Lightweight TF-IDF embeddings -----------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _sk_cos
    _HAVE_SCIKIT = True
except Exception:
    _HAVE_SCIKIT = False

# ----------------------- HTTP client (Ollama) --------------------------------
import requests

# =============================================================================
# Config
# =============================================================================

# In extraction_semantic_only_v1.py


@dataclass
class Config:
    # LLM
    model_name: str = os.environ.get("CTI_MODEL_NAME", "gemma2:9b")
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    use_llm: bool = bool(int(os.environ.get("USE_LLM", "1")))
    # prompting & consensus
    num_prompts: int = int(os.environ.get("NUM_PROMPTS", "3"))   # prompts per chunk
    consensus_m: int = int(os.environ.get("CONSENSUS_M", "2"))   # appear in >= m prompts
    # chunking
    TARGET_MIN_WORDS: int = int(os.environ.get("CHUNK_MIN_WORDS", "1"))
    TARGET_MAX_WORDS: int = int(os.environ.get("CHUNK_MAX_WORDS", "220"))
    MM_MIN_SIM_THRESH: float = float(os.environ.get("CHUNK_MIN_SIM", "0.12"))
    MM_OVERLAP_WORDS: int = int(os.environ.get("CHUNK_OVERLAP", "25"))
    MM_LOOKAHEAD: int = int(os.environ.get("CHUNK_LOOKAHEAD", "3"))
    MIN_CTITERM_DENSITY: float = float(os.environ.get("MIN_CTI_DENSITY", "0.005"))
    

config = Config()


# =============================================================================
# STIX ontology + notebook-normalization (matches your evaluation)
# =============================================================================
STIX_ENTITY_TYPES = {
    "malware","tool","attack-pattern","threat-actor","intrusion-set",
    "infrastructure","campaign","indicator","vulnerability","identity",
    "tactic","technique","sub-technique","c2","domain","ip"
}
STIX_REL_TYPES = {
    "uses","delivers","drops","downloads","installs","exploits","targets",
    "communicates-with","beacons-to","hosts-on","controls","indicates",
    "attributed-to","mitigates","part-of","related-to","located-in"
}

RE_TECHNIQUE     = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.I)
RE_CVE           = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
RE_IPV4          = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")
RE_DOMAIN        = re.compile(r"\b(?=.{4,253}\b)(?!-)(?:[a-z0-9-]{1,63}\.)+[a-z]{2,63}\b", re.I)

def canonical_entity_type(t: str) -> str:
    t = (t or "").lower().strip()
    alias = {
        "attack pattern":"attack-pattern", "attack-pattern":"attack-pattern",
        "technique":"attack-pattern", "sub-technique":"attack-pattern",
        "tactic":"tactic", "c2":"infrastructure", "server":"infrastructure",
        "actor":"threat-actor", "group":"intrusion-set",
        "vuln":"vulnerability", "cve":"vulnerability",
        "program":"tool", "utility":"tool",
        "org":"identity", "organization":"identity", "company":"identity",
        "sector":"identity", "industry":"identity"
    }
    return alias.get(t, t if t in STIX_ENTITY_TYPES else t)

def detect_implicit_type(text: str) -> str | None:
    s = (text or "").strip()
    if RE_TECHNIQUE.search(s):     return "attack-pattern"
    if RE_CVE.search(s):           return "vulnerability"
    if RE_IPV4.search(s):          return "infrastructure"
    if RE_DOMAIN.search(s):        return "infrastructure"
    return None

_MAL_TOOL_ALIASES = {
    "powershell.exe":"powershell", "pwsh":"powershell",
    "mimikatz.exe":"mimikatz", "cobalt strike":"cobalt_strike",
    "x-agent":"xagent","x-tunnel":"xtunnel",
    "apt28":"fancy bear","sednit":"fancy bear",
    "cozy bear":"apt29","cozybear":"apt29","cozyduke":"apt29","the dukes":"apt29",
    "trick bot":"trickbot", "trickbot.exe":"trickbot",
    "cobaltstrike":"cobalt_strike", "fancybear":"fancy bear", "ta505":"ta505"
}

def normalize_entity_label(text: str, etype: str | None = None) -> str:
    if not text: return ""
    s = text.strip()
    etype = canonical_entity_type(etype or "") or detect_implicit_type(s) or ""

    m = RE_TECHNIQUE.search(s)
    if m: return m.group(0).upper()
    m = RE_CVE.search(s)
    if m: return m.group(0).upper()

    if etype == "infrastructure":
        s2 = s.lower()
        s2 = re.sub(r"^https?://", "", s2)
        s2 = s2.split("/")[0]
        return s2

    base = re.sub(r"[\"'`]", "", s).strip()
    base = re.sub(r"\s+", " ", base)
    base_l = base.lower()
    for ext in [".exe",".dll",".sys",".bin",".dat",".tmp",".ps1",".bat",".cmd",".js",".jar",".zip",".rar"]:
        if base_l.endswith(ext): base = base[:-len(ext)]
    base_l = base.lower()
    base_l = _MAL_TOOL_ALIASES.get(base_l, base_l)
    return base_l

_PRED_MAP = {
    "use":"uses","uses":"uses","used":"uses","using":"uses",
    "execute":"uses","executes":"uses","executed":"uses","run":"uses","runs":"uses","leverages":"uses","utilizes":"uses","invokes":"uses","calls":"uses","implements":"uses",
    "deliver":"delivers","delivers":"delivers","drops":"drops","drop":"drops","download":"downloads","downloads":"downloads","install":"installs","installs":"installs",
    "beacon":"beacons-to","beacons":"beacons-to","beacons-to":"beacons-to","communicates":"communicates-with","communicates-with":"communicates-with","contacts":"communicates-with","connects to":"communicates-with",
    "hosts on":"hosts-on","hosts-on":"hosts-on","hosted on":"hosts-on",
    "target":"targets","targets":"targets","attack":"targets","attacks":"targets","compromises":"targets",
    "exploit":"exploits","exploits":"exploits","abuses":"exploits",
    "attributed to":"attributed-to","linked to":"related-to","associated with":"related-to","part of":"part-of","part-of":"part-of",
    "indicates":"indicates","mitigates":"mitigates","controls":"controls","located in":"located-in","located-in":"located-in"
}
def normalize_predicate_for_embedding(p: str) -> str:
    if not p: return "uses"
    q = p.strip().lower()
    q = re.sub(r"\s+", " ", q)
    return _PRED_MAP.get(q, q if q in STIX_REL_TYPES else "related-to")

def normalize_predicate(p: str) -> str:
    return normalize_predicate_for_embedding(p)

def _canon_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()
def infer_entity_type_from_name(name: str) -> str:
    n = (name or "").strip()
    n_l = n.lower()

    # hard signals
    if RE_TECHNIQUE.search(n):     return "attack-pattern"
    if RE_CVE.search(n):           return "vulnerability"
    if RE_IPV4.search(n) or RE_DOMAIN.search(n): return "infrastructure"

    # heuristics
    if re.search(r"\b(apt|group|crew|bear|team|lazarus|ta\d+)\b", n_l): return "intrusion-set"
    if re.search(r"\b(phish|spear.?phish|credential|brute|exfiltrat|payload|lateral|persistence|t\d{4})\b", n_l): return "attack-pattern"
    if re.search(r"\b(malware|trojan|ransom|worm|bot|backdoor|loader|infostealer)\b", n_l): return "malware"
    if re.search(r"\b(tool|framework|beacon|ps1|powershell|mimikatz|cobalt ?strike|metasploit)\b", n_l): return "tool"
    if re.search(r"\b(c2|command[- ]and[- ]control|server|ip|domain|dns|infrastructure)\b", n_l): return "infrastructure"
    if re.search(r"\b(inc|corp|llc|ltd|company|organization|sector|bank|government|ministry)\b", n_l): return "identity"

    # default fallback: identity (entity mentioned)
    return "identity"

# =============================================================================
# Sentence splitting & filtering
# =============================================================================
def sent_tokenize(text: str) -> List[str]:
    if _HAVE_SPACY:
        doc = _nlp()(text)
        return [s.text for s in doc.sents if s.text.strip()]
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s and s.strip()]

_CTI_TERMS = re.compile(r"\b(APT|MITRE|CVE-\d{4}-\d{4,7}|phishing|lateral|Cobalt ?Strike|beacon|payload|ransom|exfiltrat|persistence|C2|command[- ]and[- ]control|TTPs?)\b", re.I)
def _word_count(s: str) -> int:
    return len(re.findall(r"\w+", s))

def filter_sentences(sents: List[str]) -> List[str]:
    out = []
    for s in sents:
        s2 = s.strip()
        if len(s2) >= 3:
            out.append(s2)
    return out

def _chunk_relevant(sents: List[str]) -> bool:
    text = " ".join(sents)
    if not text.strip(): return False
    dense = len(_CTI_TERMS.findall(text)) / max(1, _word_count(text))
    return dense >= config.MIN_CTITERM_DENSITY or _word_count(text) >= config.TARGET_MIN_WORDS

# =============================================================================
# TF-IDF embedder
# =============================================================================
class TfidfEmbedder:
    def __init__(self):
        self._vec = None
        self._cache: Dict[str, Any] = {}
    def embed(self, sentences: List[str], doc_id: str):
        key = f"{doc_id}:{len(sentences)}"
        if key in self._cache:
            return self._cache[key][1]
        if not _HAVE_SCIKIT:
            mat = [[1.0] for _ in sentences]
            self._cache[key] = (sentences, mat)
            return mat
        self._vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        mat = self._vec.fit_transform(sentences)
        self._cache[key] = (sentences, mat)
        return mat
    def cosine(self, A, B):
        if not _HAVE_SCIKIT:
            return [[1.0 if i==j else 0.0 for j in range(len(B))] for i in range(len(A))]
        return _sk_cos(A, B)

# =============================================================================
# Semantic chunking (max-min style)
# =============================================================================
def maxmin_semantic_chunks(
    sentences: List[str],
    embeddings,
    min_sim_threshold: float,
    max_words: int,
    overlap_words: int,
    lookahead: int
) -> List[Dict[str, Any]]:
    n = len(sentences)
    if n == 0: return []
    def cos_ii(i, j):
        if _HAVE_SCIKIT:
            return float(embeddings[i].dot(embeddings[j].T).toarray().ravel()[0])
        return 1.0 if i==j else 0.0
    chunks = []
    i = 0
    while i < n:
        start = i
        cur_words = _word_count(sentences[i])
        last_i = i
        j = i + 1
        while j < n and cur_words < max_words:
            ok = True
            for k in range(max(i, j - lookahead), j):
                if cos_ii(last_i, k) < min_sim_threshold:
                    ok = False; break
            if not ok: break
            cur_words += _word_count(sentences[j])
            last_i = j
            j += 1
        end = j
        text = " ".join(sentences[start:end])
        chunks.append({"start": start, "end": end, "text": text})
        if end >= n: break
        # overlap
        if overlap_words > 0:
            words, k = 0, end - 1
            while k > start and words < overlap_words:
                words += _word_count(sentences[k]); k -= 1
            i = max(k + 1, start + 1)
        else:
            i = end
    return chunks

def _coalesce_short(chunks: List[Dict[str, Any]], sentences: List[str], min_words: int, max_words: int) -> List[Dict[str, Any]]:
    if not chunks: return []
    out, buf = [], None
    for ch in chunks:
        if buf is None:
            buf = ch
        elif _word_count(buf["text"]) < min_words:
            buf = {"start": buf["start"], "end": ch["end"], "text": " ".join(sentences[buf["start"]:ch["end"]])}
        else:
            out.append(buf); buf = ch
    if buf: out.append(buf)
    capped = []
    for ch in out:
        if _word_count(ch["text"]) <= max_words:
            capped.append(ch)
        else:
            mid = (ch["start"] + ch["end"]) // 2
            capped.append({"start": ch["start"], "end": mid, "text": " ".join(sentences[ch["start"]:mid])})
            capped.append({"start": mid, "end": ch["end"], "text": " ".join(sentences[mid:ch["end"]])})
    return capped

# =============================================================================
# Ollama client (Gemma 2 9B)
# =============================================================================
class OllamaClient:
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base = base_url.rstrip("/")
    def invoke(self, prompt: str) -> str:
        print(">>> sending prompt to Ollama:", prompt[:120])
        try:
            

            resp = requests.post(
                f"{self.base}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "") or ""
        except Exception as e:
            logging.getLogger(__name__).warning("LLM call failed: %s", e)
            return '[]'  # JSON parse will just yield empty triples

# =============================================================================
# Prompts 
# =============================================================================
SYSTEM_STIX = (
    "Extract cyber threat intelligence triples in JSON.\n"
    "Use ONLY these entity types and relationship types.\n"
    "Entity types: threat-actor | intrusion-set | malware | tool | campaign | infrastructure | indicator | vulnerability | attack-pattern | x-mitre-tactic | identity\n"
    "Relationship types: uses | targets | indicates | exploits | communicates-with | part-of | related-to\n"
    "Return strictly this JSON array of objects:\n"
    "[{ \"subject\": {\"name\":\"...\",\"type\":\"<entity-type>\"}, \"predicate\":\"<relationship-type>\", \"object\": {\"name\":\"...\",\"type\":\"<entity-type>\"} }]"
)

def _schema() -> str:
    return """
[
  {
    "subject": {"name": "<string>", "type": "<entity-type>"},
    "predicate": "<relationship-type>",
    "object": {"name": "<string>", "type": "<entity-type>"}
  }
]
""".strip()

def _prompts_for_chunk(text: str, n: int) -> List[str]:
    text = text.strip()
    schema = _schema()
    # Three distinct “personalities” — consensus works best with diversity
    p1 = f"""{SYSTEM_STIX}

Text:
\"\"\"{text[:2200]}\"\"\"

Rules:
- Only output triples explicitly supported by the text
- No guesses, no invented entities
- Names should be lowercase strings when possible
- Use allowed types/predicates only
Output {schema}
"""
    p2 = f"""{SYSTEM_STIX}

Strategy:
- Read the whole passage first
- Extract a small set of high-confidence triples (<= 18)
- If a field is uncertain, omit the triple

Text:
\"\"\"{text[:2200]}\"\"\"
Output {schema}
"""
    p3 = f"""{SYSTEM_STIX}

Constraints:
- Every triple must be justified by an explicit phrase
- Use 'attack-pattern' for MITRE techniques (e.g., T1059, spearphishing)
- Map organizations/companies/sectors to 'identity'
- Map IPs/domains/C2 to 'infrastructure'
- Do not include duplicates

Text:
\"\"\"{text[:2200]}\"\"\"
Output {schema}
"""
    variants = [p1, p2, p3]
    random.seed(len(text))
    random.shuffle(variants)
    return variants[:max(1, n)]

# =============================================================================
# Parse/normalize/consensus
# =============================================================================
def _parse_any_json(text: str) -> Any:
    if not text: return []
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(json)?", "", t).strip()
        t = re.sub(r"```$", "", t).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, list): return obj
        if isinstance(obj, dict): return [obj]
    except Exception:
        pass
    m = re.search(r"\[[\s\S]*\]", t)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    return []

def _as_triple_list(obj: Any) -> List[Dict[str, Any]]:
    if not isinstance(obj, list): return []
    out = []
    for t in obj:
        if not isinstance(t, dict): continue
        s = t.get("subject") or {}
        o = t.get("object") or {}
        out.append({
            "subject":{"name": s.get("name") or s.get("text") or "", "type": s.get("type") or ""},
            "predicate": t.get("predicate") or t.get("relation") or "",
            "object":{"name": o.get("name") or o.get("text") or "", "type": o.get("type") or ""},
        })
    return out

ALLOWED_STIX_TYPES = {
    "threat-actor","intrusion-set","malware","tool","campaign","infrastructure",
    "indicator","vulnerability","attack-pattern","x-mitre-tactic","identity"
}
ALLOWED_STIX_PREDS = {"uses","targets","indicates","exploits","communicates-with","part-of","related-to"}

def _valid_stix_triple_dict(t: dict) -> bool:
    try:
        sname = _canon_text((t.get("subject") or {}).get("name"))
        oname = _canon_text((t.get("object") or {}).get("name"))
        if not (sname and oname): return False

        st = normalize_entity_label((t.get("subject") or {}).get("type"))
        ot = normalize_entity_label((t.get("object") or {}).get("type"))
        if not st: st = normalize_entity_label(infer_entity_type_from_name(sname))
        if not ot: ot = normalize_entity_label(infer_entity_type_from_name(oname))

        p  = normalize_predicate((t.get("predicate") or ""))
        return (p in ALLOWED_STIX_PREDS) and (st in ALLOWED_STIX_TYPES) and (ot in ALLOWED_STIX_TYPES)
    except Exception:
        return False


def normalize_triple_dict(t: dict) -> Tuple[str,str,str]:
    s = normalize_entity_label((t.get("subject") or {}).get("name"), (t.get("subject") or {}).get("type"))
    p = normalize_predicate((t.get("predicate") or ""))
    o = normalize_entity_label((t.get("object") or {}).get("name"), (t.get("object") or {}).get("type"))
    return (_canon_text(s), p, _canon_text(o))

def consensus_filter(lists: List[List[Dict[str, Any]]], m: int = 2) -> List[Dict[str, Any]]:
    """Majority vote on normalized (s,p,o,st,ot)."""
    from collections import Counter
    ctr, canon_map = Counter(), {}
    for li in lists:
        seen = set()
        for t in li:
            s = _canon_text((t.get("subject") or {}).get("name"))
            st = canonical_entity_type((t.get("subject") or {}).get("type"))
            p  = normalize_predicate(t.get("predicate"))
            o = _canon_text((t.get("object")  or {}).get("name"))
            ot = canonical_entity_type((t.get("object")  or {}).get("type"))
            key = json.dumps({"s":s,"st":st,"p":p,"o":o,"ot":ot}, sort_keys=True)
            if key not in seen:
                seen.add(key); ctr[key]+=1
                canon_map[key] = {"subject":{"name":s,"type":st},"predicate":p,"object":{"name":o,"type":ot}}
    return [canon_map[k] for k,c in ctr.items() if c>=m]

# =============================================================================
# Extractor class
# =============================================================================
class CyberTripleExtractor:
    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name = model_name or config.model_name
        self.base_url   = (base_url or config.ollama_base_url).rstrip("/")
        self.embedder   = TfidfEmbedder()
        self.llm        = OllamaClient(self.model_name, self.base_url)

        # expose chunking params
        self.TARGET_MIN_WORDS = config.TARGET_MIN_WORDS
        self.TARGET_MAX_WORDS = config.TARGET_MAX_WORDS
        self.MM_MIN_SIM_THRESH = config.MM_MIN_SIM_THRESH
        self.MM_OVERLAP_WORDS  = config.MM_OVERLAP_WORDS
        self.MM_LOOKAHEAD      = config.MM_LOOKAHEAD

    def chunk(self, text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        sents = filter_sentences(sent_tokenize(text or ""))
        if not sents: return [], []
        doc_id = f"eval:{hashlib.sha1((text[:1000] or '').encode('utf-8','ignore')).hexdigest()}"
        mats = self.embedder.embed(sents, doc_id=doc_id)
        chunks = maxmin_semantic_chunks(
            sentences=sents,
            embeddings=mats,
            min_sim_threshold=self.MM_MIN_SIM_THRESH,
            max_words=self.TARGET_MAX_WORDS,
            overlap_words=self.MM_OVERLAP_WORDS,
            lookahead=self.MM_LOOKAHEAD
        )
        chunks = _coalesce_short(chunks, sents, self.TARGET_MIN_WORDS, self.TARGET_MAX_WORDS)
        # annotate relevance
        for ch in chunks:
            ch["relevant"] = _chunk_relevant(sents[ch["start"]:ch["end"]])
        return chunks, sents

# =============================================================================
# Public entrypoints (used by eval_cli)
# =============================================================================
def extract_triples_from_text(
    text: str,
    model_name: str = None,
    ollama_base_url: str = None
) -> List[Tuple[str,str,str]]:
    """
    Semantic chunking → multi-prompt LLM extraction (Gemma2-9B via Ollama) → consensus → STIX-normalized triples.
    """
    ex = CyberTripleExtractor(model_name=model_name, base_url=ollama_base_url)
    chunks, sents = ex.chunk(text)
    if not chunks: return []

    triples_out: List[Tuple[str,str,str]] = []

    for ch in chunks:
        if not ch.get("relevant") and not getattr(config, "RELEVANCE_OFF", False):
        
            continue
        chunk_text = ch.get("text") or " ".join(sents[ch["start"]:ch["end"]])

        if config.use_llm:
            prompts = _prompts_for_chunk(chunk_text, n=max(1, config.num_prompts))
            lists: List[List[Dict[str, Any]]] = []
            for p in prompts:
                resp   = ex.llm.invoke(p)
                parsed = _parse_any_json(resp)
                triples = _as_triple_list(parsed)
                for t in triples:
                    subj = t.get("subject") or {}
                    obj  = t.get("object")  or {}
                    # fill missing types
                    if not subj.get("type"):
                        subj["type"] = infer_entity_type_from_name(subj.get("name",""))
                    if not obj.get("type"):
                        obj["type"]  = infer_entity_type_from_name(obj.get("name",""))

                    # normalize to your STIX set
                    subj["type"] = normalize_entity_label(subj.get("type"))
                    obj["type"]  = normalize_entity_label(obj.get("type"))
                    t["predicate"] = normalize_predicate(t.get("predicate"))

                # keep raw; consensus works on normalized keys
                lists.append(triples)

            fused = consensus_filter(lists, m=config.consensus_m) if config.consensus_m > 1 else [t for li in lists for t in li]

            # validate + normalize final
            for t in fused:
                if _valid_stix_triple_dict(t):
                    s,p,o = normalize_triple_dict(t)
                    if s and p and o:
                        triples_out.append((s,p,o))
        else:
            # deterministic fallback (not used when USE_LLM=1)
            pass

    # dedupe
    seen = set(); deduped = []
    for t in triples_out:
        k = json.dumps(t, sort_keys=True)
        if k not in seen:
            seen.add(k); deduped.append(t)
    return deduped

def extract_triples_from_file(
    file_path: str,
    model_name: str = None,
    ollama_base_url: str = None
) -> List[Tuple[str,str,str]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    return extract_triples_from_text(raw, model_name=model_name, ollama_base_url=ollama_base_url)

# =============================================================================
# CLI smoke test
# =============================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="Inline text to extract from")
    ap.add_argument("--file", help="Path to a text file")
    ap.add_argument("--model", default=config.model_name)
    ap.add_argument("--base", default=config.ollama_base_url)
    ap.add_argument("--prompts", type=int, default=config.num_prompts)
    ap.add_argument("--consensus", type=int, default=config.consensus_m)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    config.num_prompts = max(1, int(args.prompts))
    config.consensus_m = max(1, int(args.consensus))
    config.use_llm = not args.no_llm

    sample = args.text
    if args.file and not sample:
        with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read()
    if not sample:
        sample = "APT29 used spearphishing to deploy Cobalt Strike and communicated with a C2 server."

    t0 = time.time()
    triples = extract_triples_from_text(sample, model_name=args.model, ollama_base_url=args.base)
    dt = time.time() - t0
    for t in triples:
        print(t)
    print(f"\nExtracted {len(triples)} triples in {dt:.2f}s (model={args.model})")
