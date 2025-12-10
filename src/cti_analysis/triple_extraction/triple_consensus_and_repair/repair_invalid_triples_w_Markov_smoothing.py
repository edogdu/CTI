import os, re, json, time, requests, sys
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict


#Create one unified Session for requests.get() and requests.post()
SESSION = requests.Session()
#look for invalid_triples.json inside extracted_triples/
script_dir = Path(__file__).parent
input_dir = script_dir / "../extracted_triples"
INPUT_PATH = input_dir / "invalid_triples_gemma2_9b.json"

if not INPUT_PATH.exists():
    print(f"[error] Could not find invalid_triples JSON at: {INPUT_PATH}")
    print("Tip: make sure you ran extraction_consensus_plus_inva...py first; it saves invalid triples under 'extracted_triples/'.")
    sys.exit(1)

os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
#os.environ["OLLAMA_MODEL"] = "foundation-sec-8b-instruct"

os.environ["OLLAMA_MODEL"] = "gemma2:9b"
BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "gemma2:9b")
OUT_BASE = os.path.splitext(str(INPUT_PATH))[0]
LIM = 500  # max string length sanity
print(f"BASE = {BASE}")
print(f"MODEL = {MODEL}")
#STIX 2.1 Names and
TYPES = {
    "threat-actor","intrusion-set","campaign","malware","tool","infrastructure","attack-pattern",
    "course-of-action","indicator","vulnerability","software","product","organization","identity",
    "sector","country","location","city","ipv4-addr","ipv6-addr","domain-name","url","file",
    "email-addr","observed-data","report","sighting","x-mitre-data-source","x-mitre-data-component",
    "observable","user-account","directory","autonomous-system","date","malware-analysis"
}
PREDS = {
    "uses","targets","attributed_to","exploits","affects","mitigates","indicates","detects","based_on",
    "derived_from","observed_in","communicates_with","hosts","delivers","drops","located_at",
    "uses_technique","subtechnique_of","revoked_by","duplicate_of","originates_from","impacts",
    "resolves_to","downloads_from","writes_to","reads_from","emails_to","observed_on",
    "analysis_of","characterizes","sighting_of","sighted_at","sighted_by","variant_of"
}
#Domain/range constraints
SCHEMA = {
    "uses": ({"threat-actor","intrusion-set","campaign"}, {"malware","tool","infrastructure","attack-pattern"}),
    "targets": ({"threat-actor","intrusion-set","campaign","malware","attack-pattern"}, {"identity","organization","sector","location","country"}),
    "attributed_to": ({"campaign","intrusion-set"}, {"threat-actor"}),
    "exploits": ({"threat-actor","malware","tool"}, {"vulnerability"}),
    "uses_technique": ({"threat-actor","malware","tool","campaign"}, {"attack-pattern"}),
    "originates_from": ({"threat-actor","intrusion-set"}, {"location","country"}),
    "based_on": ({"indicator"}, {"file","url","domain-name","ipv4-addr","ipv6-addr","email-addr","observable","artifact"}),
    "derived_from": ({"indicator"}, {"report","observed-data","artifact","log"}),
    "detects": ({"indicator","tool","x-mitre-data-source"}, {"malware","campaign","intrusion-set","infrastructure","tool","vulnerability"}),
    "indicates": ({"indicator"}, {"malware","campaign","intrusion-set","attack-pattern","vulnerability","organization","sector","location"}),
    "resolves_to": ({"domain-name"}, {"ipv4-addr","ipv6-addr"}),
    "hosts": ({"infrastructure"}, {"domain-name","ipv4-addr","ipv6-addr","url","file"}),
    "delivers": ({"infrastructure","malware","tool"}, {"malware","tool","file","url"}),
    "drops": ({"malware","tool"}, {"malware","tool","file"}),
    "located_at": ({"infrastructure","organization","identity"}, {"location","country","city"}),
    "observed_on": ({"threat-actor","malware","tool","infrastructure","indicator","attack-pattern"}, {"date"}),
    "impacts": ({"threat-actor","malware","attack-pattern"}, {"identity","organization","sector"}),
}

# ========= Canonicalization helpers =========
TYPE_ALIASES = {
    "threatactor":"threat-actor", "threat actor":"threat-actor",
    "intrusionset":"intrusion-set", "intrusion set":"intrusion-set",
    "attackpattern":"attack-pattern", "attack pattern":"attack-pattern",
    "courseofaction":"course-of-action", "course of action":"course-of-action",
    "domain":"domain-name","domainname":"domain-name",
    "ipv4":"ipv4-addr","ipv4addr":"ipv4-addr","ipv4-address":"ipv4-addr",
    "ipv6":"ipv6-addr","ipv6addr":"ipv6-addr","ipv6-address":"ipv6-addr",
    "email":"email-addr","emailaddress":"email-addr","email addr":"email-addr",
    "software":"software","product":"product","organization":"organization","organisation":"organization",
    "identity":"identity","sector":"sector","country":"country","location":"location","city":"city",
    "infrastructure":"infrastructure","malware":"malware","tool":"tool",
    "indicator":"indicator","vulnerability":"vulnerability",
    "x-mitre-data-source":"x-mitre-data-source","x_mitre_data_source":"x-mitre-data-source",
    "x-mitre-data-component":"x-mitre-data-component","x_mitre_data_component":"x-mitre-data-component",
    "useraccount":"user-account","user account":"user-account",
    "autonomoussystem":"autonomous-system","autonomous system":"autonomous-system",
    "observeddata":"observed-data","observed data":"observed-data",
    "report":"report","sighting":"sighting","date":"date","malwareanalysis":"malware-analysis",
}
PRED_ALIASES = {
    "communicateswith":"communicates_with",
    "usestechnique":"uses_technique",
    "subtechniqueof":"subtechnique_of",
    "revokedby":"revoked_by",
    "duplicateof":"duplicate_of",
    "originatesfrom":"originates_from",
    "resolvesto":"resolves_to",
    "downloadsfrom":"downloads_from",
    "writes_to":"writes_to","reads_from":"reads_from","emails_to":"emails_to",
    "observedon":"observed_on","analysisof":"analysis_of",
    "sightingof":"sighting_of","sightedat":"sighted_at","sightedby":"sighted_by",
    "variantof":"variant_of",
}

def canon_type(t):
    if not isinstance(t, str): return t
    t2 = t.strip().lower().replace(" ", "").replace("_", "-")
    if t2 in TYPE_ALIASES: t2 = TYPE_ALIASES[t2]
    return t2 if t2 in TYPES else t2

def canon_pred(p):
    if not isinstance(p, str): return p
    s = p.strip().lower().replace(" ", "_")
    snake = PRED_ALIASES.get(s, s)
    return snake if snake in PREDS else p

#Tiny string and length checkers
s=lambda x:isinstance(x,str) # check if string
ne=lambda x:s(x) and 0<len(x.strip())<=LIM # check if line is not empty with length limit (500) here

#Normalize everything, No LLM used here. Strip whitespace, cannonize names, ensures everything matches STIX 2.1 exactly
def det_fix(t):
    t = dict(t) if isinstance(t, dict) else {"_raw": t}
    t.setdefault("subject", {}); t.setdefault("object", {})
    for side in ("subject","object"):
        name = t[side].get("name")
        if isinstance(name, str):
            name = name.strip()
            t[side]["name"] = name if name else "UNKNOWN"

        typ = t[side].get("type")
        if isinstance(typ, str):
            t[side]["type"] = canon_type(typ)
    if isinstance(t.get("predicate"), str):
        t["predicate"] = canon_pred(t["predicate"].strip())
    return t

#Markov smoothing
def split_sentences(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [p for p in parts if p]

class MarkovEntitySmoother:
    def __init__(self, states=None, alpha=0.1):
        self.states = states or [
            "malware","tool","attack-pattern","threat-actor","intrusion-set",
            "vulnerability","indicator","org","tactic","other"
        ]
        n = len(self.states); self.alpha = alpha
        self.P = np.full((n, n), alpha)
        np.fill_diagonal(self.P, 1.0)
        self.P /= self.P.sum(axis=1, keepdims=True)
        self.pi = np.full(n, 1.0 / n)

    def _sid(self, t):
        return self.states.index(t) if t in self.states else self.states.index("other")

    def viterbi(self, obs_types, conf=None):
        n = len(self.states); T = len(obs_types)
        conf = conf or [0.8] * T
        E = np.zeros((T, n))
        for t in range(T):
            pid = self._sid(obs_types[t])
            for s in range(n):
                E[t, s] = conf[t] if s == pid else (1 - conf[t]) / (n - 1)
        V = np.log(self.pi + 1e-9) + np.log(E[0] + 1e-9)
        B = np.zeros((T, n), dtype=int)
        for t in range(1, T):
            V_new = np.empty(n)
            for s in range(n):
                sc = V + np.log(self.P[:, s] + 1e-9) + np.log(E[t, s] + 1e-9)
                B[t, s] = int(np.argmax(sc))
                V_new[s] = np.max(sc)
            V = V_new
        path = [int(np.argmax(V))]
        for t in range(T - 1, 0, -1):
            path.append(B[t, path[-1]])
        path = path[::-1]
        return [self.states[p] for p in path]

def _build_sentence_majorities_for_context(triples, ctx_text: str):
    sents = split_sentences(ctx_text)
    if not sents:
        return [], [], []
    per_sent_types = [[] for _ in sents]
    for e in triples:
        t = e.get("triple") if isinstance(e, dict) else e
        if not isinstance(t, dict):
            continue
        sub = t.get("subject") or {}
        obj = t.get("object") or {}
        sname = sub.get("name") if isinstance(sub.get("name"), str) else ""
        oname = obj.get("name") if isinstance(obj.get("name"), str) else ""
        stype = canon_type(sub.get("type")) if isinstance(sub.get("type"), str) else None
        otype = canon_type(obj.get("type")) if isinstance(obj.get("type"), str) else None
        for i, sent in enumerate(sents):
            if sname and sname in sent and stype:
                per_sent_types[i].append(stype)
            if oname and oname in sent and otype:
                per_sent_types[i].append(otype)
    types, conf = [], []
    for lst in per_sent_types:
        if lst:
            top, cnt = Counter(lst).most_common(1)[0]
            types.append(top)
            conf.append(cnt / max(1, len(lst)))
        else:
            types.append("other")
            conf.append(0.5)
    return sents, types, conf

def _smoothed_sentence_labels(ctx_text: str, triples):
    sents, types, conf = _build_sentence_majorities_for_context(triples, ctx_text)
    if not sents:
        return [], []
    sm = MarkovEntitySmoother()
    smooth_types = sm.viterbi(types, conf)
    return sents, smooth_types

def _apply_sentence_smoothing_to_triple(triple, ctx_text: str, sents, smooth_types):
    if not sents or not smooth_types:
        return triple
    t = dict(triple)
    for side in ("subject", "object"):
        side_obj = dict(t.get(side) or {})
        name = side_obj.get("name") if isinstance(side_obj.get("name"), str) else ""
        cur_type = canon_type(side_obj.get("type")) if isinstance(side_obj.get("type"), str) else None
        if not name:
            continue
        idx = next((i for i, s in enumerate(sents) if name in s), None)
        if idx is None:
            continue
        candidate = smooth_types[idx]
        cand_canon = canon_type(candidate)
        if (cur_type not in TYPES) or (cur_type in {"org", "tactic", "other"}):
            if cand_canon in TYPES:
                side_obj["type"] = cand_canon
                t[side] = side_obj
    return t
#End of Markov smoothing

def validate(t):
    if not isinstance(t, dict):
        return False, ["not_object"]

    sub, obj, p = t.get("subject"), t.get("object"), t.get("predicate")
    errs = []

    #Structure checks
    if not isinstance(sub, dict): errs.append("subject_not_object")
    if not isinstance(obj, dict): errs.append("object_not_object")
    pp = canon_pred(p) if isinstance(p, str) else p
    st = canon_type(sub.get("type") if isinstance(sub, dict) else "")
    ot = canon_type(obj.get("type") if isinstance(obj, dict) else "")
    if not isinstance(pp, str) or pp not in PREDS: errs.append("predicate_invalid")
    if st not in TYPES: errs.append("subject.type_not_allowed")
    if ot not in TYPES: errs.append("object.type_not_allowed")

    #Domain/Range checks
    if isinstance(pp, str) and pp in SCHEMA:
        dom, rng = SCHEMA[pp]
        if st not in dom: errs.append("subject.type_domain")
        if ot not in rng: errs.append("object.type_range")

    #Name sanity checks
    for side in ("subject","object"):
        ent = t.get(side) or {}
        nm = ent.get("name")
        if not ne(nm): errs.append(f"{side}.name_bad")

    return (len(errs) == 0), errs

#Reasons that are structural/schema and NOT fixable by paraphrasing
FATAL_FOR_LOOSE = {
    "predicate_invalid",
    "subject_not_object",
    "object_not_object",
}
#Reasons where a paraphrase/rename might help
FIXABLE_BY_LOOSE = {
    "subject.name_bad",
    "object.name_bad",
    "subject.type_not_allowed",
    "object.type_not_allowed",
    "subject.type_domain",
    "object.type_domain",
    "subject.type_range",
    "object.type_range",
}

#Gate the loose LLM call. Only run if at least one reason is fixable and none are fatal/schema-level problems.
def should_try_loose(reasons: list[str]) -> bool:
    if not reasons:
        return False
    if any(r in FATAL_FOR_LOOSE for r in reasons):
        return False
    return any(r in FIXABLE_BY_LOOSE for r in reasons)

def ensure_model():
    try:
        r=SESSION.get(f"{BASE}/api/tags",timeout=6) #Uses persistent SESSION now
        r.raise_for_status()
        names={m.get("name") for m in r.json().get("models",[])}
        if MODEL not in names:
            SESSION.post(f"{BASE}/api/pull",json={"name":MODEL},timeout=None) #Uses persistent SESSION now
    except Exception as e:
        print("[warn] model check:", e)

def _extract_json_list_loose(text: str):
    """Parse the first/last JSON array from text; ignore code fences/chatter."""
    if not isinstance(text, str): return []
    t=text.strip()
    if t.startswith("```"):
        t=t.strip("`")
    # naive scan for first/last brackets
    l=t.find("["); r=t.rfind("]")
    if l==-1 or r==-1 or r<=l: return []
    try:
        return json.loads(t[l:r+1])
    except Exception:
        return []

def prompt(t, ctx=""):
    """Builds a concise repair instruction for MODEL."""
    sub=t.get("subject") or {}; obj=t.get("object") or {}
    return (
        "You are a CTI triple repair assistant. Return ONLY a JSON array of "
        "candidate repaired triples (objects with subject, predicate, object). "
        "Rules:\n"
        f"- Subject/object 'type' must be from: {sorted(list(TYPES))}\n"
        f"- Predicate must be from: {sorted(list(PREDS))}\n"
        f"- Enforce domain/range per SCHEMA where applicable.\n"
        "- Keep names realistic; avoid UNKNOWN if possible.\n"
        "- Keep the core meaning consistent with the context.\n"
        f"Context: {ctx[:800]}\n"
        f"Current triple:\n{json.dumps(t, ensure_ascii=False)}\n"
        "Return 1–3 options."
    )

def ask_llm(ptxt):
    """Call Ollama compat endpoint; return list of triple dicts or []."""
    try:
        r = SESSION.post(
            f"{BASE}/api/generate",
            json={"model": MODEL, "prompt": ptxt, "stream": False},  # <— add stream=False
            timeout=120,
        )
        r.raise_for_status()
        payload = r.json()                      # single JSON object now
        txt = payload.get("response", "")
        arr = _extract_json_list_loose(txt)     # your existing extractor
        return arr if isinstance(arr, list) else []
    except Exception as e:
        print("[warn] LLM call failed:", e)
        return []


def load_invalids(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept raw list
    if isinstance(data, list):
        return data

    # Accept common dict shapes
    if isinstance(data, dict):
        # Your file uses this:
        if isinstance(data.get("invalid_triples"), list):
            return data["invalid_triples"]

        # Previous / other exporters:
        if isinstance(data.get("items"), list):
            return data["items"]
        if isinstance(data.get("invalid"), list):
            return data["invalid"]
        if isinstance(data.get("results"), list):
            return data["results"]

        # Last-resort: first list-valued field
        for k, v in data.items():
            if isinstance(v, list):
                return v

    return []


def main():
    ensure_model()
    invalids = load_invalids(INPUT_PATH)
    repaired, bad = [], []
    total=len(invalids)
    start=time.time()

    print(f"[start] Loaded {total} invalid triples\n")

    # Group invalids by context and precompute smoothed labels per context
    by_ctx = defaultdict(list)
    for e in invalids:
        ctx = e.get("context") if isinstance(e, dict) else ""
        by_ctx[ctx].append(e)

    ctx_cache = {}
    for ctx_text, group in by_ctx.items():
        try:
            sents, smooth_types = _smoothed_sentence_labels(ctx_text, group)
        except Exception:
            sents, smooth_types = [], []
        ctx_cache[ctx_text] = (sents, smooth_types)

    for i,e in enumerate(invalids,1):
        t=e.get("triple") if isinstance(e,dict) and "triple" in e else e
        ctx=e.get("context") if isinstance(e,dict) else "" #Get the "context" portion of the triple. 
#Updated repair passes
        # 1) deterministic fix
        t1 = det_fix(t)
        ok, reasons = validate(t1)
        if ok:
            repaired.append({"triple": t1, "repair": "deterministic"})
            # (optional) progress print here
            continue

        # 2) Markov smoothing attempt (sentence-level)
        sents, smooth_types = ctx_cache.get(ctx, ([], []))
        if sents:
            t2 = _apply_sentence_smoothing_to_triple(t1, ctx, sents, smooth_types)
            if t2 != t1:
                ok2, reasons2 = validate(t2)
                if ok2:
                    repaired.append({"triple": t2, "repair": "markov_smooth"})
                    # (optional) progress print here
                    continue

        # 3) Strict LLM pass
        accepted=False
        ptxt = prompt(t1, ctx)
        props = ask_llm(ptxt)
        if props:
            for c in props:
                c.setdefault("confidence", 0.5)
                c = det_fix(c)
                ok2, _ = validate(c)
                if ok2:
                    repaired.append({"triple": c, "repair": f"{MODEL}_strict"})
                    accepted = True
                    break

        #Only try loose LLM pass if strict failed and reasons are fixable
        if not accepted and should_try_loose(reasons):
            ptxt_loose = (
                prompt(t1, ctx)
                + "\nIf needed, paraphrase entity names slightly; "
                "types and predicates MUST remain from the allowed lists."
            )
            props = ask_llm(ptxt_loose)
            if props:
                for c in props:
                    c.setdefault("confidence", 0.5)
                    c = det_fix(c)
                    ok3, _ = validate(c)
                    if ok3:
                        repaired.append({"triple": c, "repair": f"{MODEL}_loose"})
                        accepted = True
                        break
        #Diag to see why the loose was skipped            
#           if not accepted and not should_try_loose(reasons):
#               print(f"skip loose {reasons}")

        if not accepted:
            bad.append({
                "original": t,
                "deterministic": t1,
                "deterministic_reasons": reasons,
                "context": ctx
            })

        if i % 50 == 0 or i == total:
            print(f"[progress] {i}/{total} processed; OK={len(repaired)} BAD={len(bad)}")

    print("\n[done] Saving results...")
    with open(f"{OUT_BASE}_repaired.json", "w", encoding="utf-8") as f:
        json.dump({"count": len(repaired), "items": repaired}, f, indent=2, ensure_ascii=False)
    with open(f"{OUT_BASE}_still_bad.json", "w", encoding="utf-8") as f:
        json.dump({"count":len(bad),"items":bad},f,indent=2,ensure_ascii=False)

    #Top failure reasons and other metrics
    reason_counts={}
    for item in bad:
        for r in item.get("deterministic_reasons",[]):
            reason_counts[r]=reason_counts.get(r,0)+1
    if reason_counts:
        top=sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print("[diagnostics] top failure reasons:", ", ".join(f"{k}:{v}" for k,v in top))

    print(f"[report] OK={len(repaired)} BAD={len(bad)} | Time={time.time()-start:.1f}s")

if __name__=="__main__":
    main()
