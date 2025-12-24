from __future__ import annotations

import json
import os
import re
import requests
import tempfile
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from docling.document_converter import DocumentConverter
from langchain_community.llms import Ollama
import spacy
from spacy.matcher import PhraseMatcher

from cti_analysis.models.documents import Chunk
from cti_analysis.models.triples import Triple, TripleBatch, triple_batch_for_doc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

SCHEMA: Dict[str, Tuple[set, set]] = {
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

# ===== spaCy bootstrap =====
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")


# ===== Ollama helper =====
def ensure_ollama_model(model_name="mistral", base_url="http://localhost:11434"):
    try:
        resp = requests.get(f"{base_url}/api/tags")
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if model_name not in models:
            print(f"[Model] '{model_name}' not found. Pulling...")
            pull_resp = requests.post(f"{base_url}/api/pull", json={"name": model_name})
            pull_resp.raise_for_status()
            print(f"[Model] Pulled '{model_name}'.")
        else:
            print(f"[Model] '{model_name}' available.")
    except Exception as e:
        print("[Model] Error checking/downloading model:", e)


# ===== Prompt builders =====
def _fmt_types(types: set) -> str:
    return ", ".join(sorted(types))


def _fmt_schema(schema: Dict[str, Tuple[set, set]]) -> str:
    parts = []
    for pred, (subj_set, obj_set) in schema.items():
        parts.append(f"- {pred}: subject in {{{', '.join(sorted(subj_set))}}}, object in {{{', '.join(sorted(obj_set))}}}")
    return "\n".join(parts)


def _extractor_sentence_prompt(text: str, entity_types: set, predicates: set, schema: Dict[str, Tuple[set, set]]) -> str:
    types_str = _fmt_types(entity_types)
    preds_str = _fmt_types(predicates)
    schema_str = _fmt_schema(schema)
    return f"""
You are a cybersecurity analyst extracting structured intelligence using STIX 2.1 ontology.

TASK:
From the sentence below, extract at most 5 subject–predicate–object triples.

RULES:
1. Subject.type and Object.type MUST be in: [{types_str}]
2. Predicate MUST be in: [{preds_str}]
3. Respect these domain/range rules (only choose Predicate: Subject.type, Object.type combinations from these sets):
{schema_str}
4. Use ONLY explicit relationships stated in the text (no guesses).
5. Return "NO_TRIPLES" if none.
6. Names must match the sentence exactly.
7. Include an exact quote from the sentence for each triple.

OUTPUT (JSON array only):
[
  {{
    "subject": {{"name": "<string>", "type": "<type>"}},
    "predicate": "<predicate>",
    "object": {{"name": "<string>", "type": "<type>"}},
    "evidence": {{"quote": "<exact substring>"}}
  }}
]

Sentence:
\"\"\"{text}\"\"\"""".strip()


def _chunk_prompt(chunk_text: str, entity_types: set, predicates: set, schema: Dict[str, Tuple[set, set]]) -> str:
    types_str = _fmt_types(entity_types)
    preds_str = _fmt_types(predicates)
    schema_str = _fmt_schema(schema)
    return f"""
You are a cybersecurity analyst extracting structured intelligence from a chunk of text using STIX 2.1 ontology.

TASK:
From the chunk below, extract at most 8 subject–predicate–object triples.
Scope only within this chunk.

RULES:
1. Subject.type and Object.type MUST be in: [{types_str}]
2. Predicate MUST be in: [{preds_str}]
3. Respect these domain/range hints (do not fabricate types): 
{schema_str}
4. Use ONLY explicit relations in the chunk.
5. Return "NO_TRIPLES" if none.
6. Names must match exactly.
7. Include an exact quote from the chunk for each triple.

OUTPUT (JSON array only) as above.

Chunk:
\"\"\"{chunk_text}\"\"\"""".strip()


def _consensus_variants(text: str, entity_types: str, predicates: str) -> List[str]:
    base_tail = f"""
Types: [{entity_types}]
Predicates: [{predicates}]
Return "NO_TRIPLES" if uncertain.
Output strictly as a JSON array with the schema in the instructions.
Sentence:
\"\"\"{text}\"\"\"""".strip()

    v1 = f"Fill slots exactly from the sentence. {base_tail}"
    v2 = f"Work predicate-first; exact names only. {base_tail}"
    v3 = f"Prefer explicit IOCs and named malware when present. {base_tail}"
    v4 = f"No guessing; any missing field => 'NO_TRIPLES'. {base_tail}"
    return [v1, v2, v3, v4]


# ===== Utils =====
def _dbg(on: bool, msg: str):
    if on:
        print("[DEBUG]", msg)


def _tri_key(t: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    s = t.get("subject", {}) or {}
    o = t.get("object", {}) or {}
    return (
        (s.get("name") or "").strip().lower(),
        (s.get("type") or "").strip(),
        (t.get("predicate") or "").strip(),
        (o.get("name") or "").strip().lower(),
        (o.get("type") or "").strip(),
    )


def _dedupe(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for t in triples:
        k = _tri_key(t)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def _normalize_rel(triple):
    if isinstance(triple, list):
        # If already a list of dicts, return as-is
        if triple and isinstance(triple[0], dict):
            return triple
        # Otherwise treat as list of tuples/lists
        return [{"subject": t[0], "predicate": t[1], "object": t[2]} for t in triple if len(t) >= 3]
    if isinstance(triple, dict):
        return [triple]
    return []


class CyberTripleExtractor:
    def __init__(self, file_path: str | None, document_id: str, model_name: str = "gemma2:9b", ollama_base_url: str = "http://localhost:11434"):
        self.file_path = file_path
        self.document_id = document_id
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.chunk_data = []
        self.valid_triples = []
        self.text_override: str | None = None

    def _load_file(self) -> str:
        if self.text_override is not None:
            return self.text_override
        # Minimal text load; for PDF use docling converter
        p = Path(self.file_path)
        if p.suffix.lower() == ".pdf":
            conv = DocumentConverter()
            doc = conv.convert(p)
            return doc.text_content
        return p.read_text(encoding="utf-8", errors="ignore")

    def _call_llm(self, prompt: str) -> List[Dict[str, Any]]:
        llm = Ollama(model=self.model_name, base_url=self.ollama_base_url)
        resp = llm.invoke(prompt)
        try:
            data = json.loads(resp)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _extract_sentence_level(self, text: str, opts=None) -> List[Dict[str, Any]]:
        doc = nlp(text)
        out = []
        for sent in doc.sents:
            prompt = _extractor_sentence_prompt(sent.text, TYPES, PREDS, SCHEMA)
            triples = self._call_llm(prompt)
            print(triples)
            print("working")
            out.extend(_normalize_rel(triples))
        return _dedupe(out)

    def _extract_chunk_level(self, text: str, opts=None) -> List[Dict[str, Any]]:
        prompt = _chunk_prompt(text, TYPES, PREDS, SCHEMA)
        triples = self._call_llm(prompt)
        return _dedupe(_normalize_rel(triples))

    def run(self, opts=None):
        text = self._load_file()
        triples = []
        triples.extend(self._extract_sentence_level(text, opts))
        triples.extend(self._extract_chunk_level(text, opts))
        triples = _dedupe(triples)
        self.valid_triples = triples
        # Build chunk_data in a shape similar to previous outputs
        self.chunk_data = [{"triple": triples, "context": ""}]
        return self.chunk_data

    def build_dict(self, raw_results):
        # no-op; compatibility
        return


def run_extraction(cfg, chunks: List[Chunk] | Union[str, Path]) -> List[TripleBatch] | Dict[str, Any]:
    """
    IR-friendly extractor using the main CyberTripleExtractor.
    - Preferred: pass a list of Chunk -> returns List[TripleBatch]
    - Path/str: runs extraction on a file and returns the legacy dict
    """
    if isinstance(chunks, (str, Path)):
        return _run_file(cfg, Path(chunks))

    print(f"[extractor] run_extraction on {len(chunks)} chunks")
    batches: List[TripleBatch] = []
    for ch in chunks:
        triples: List[Triple] = _extract_triples_with_llm(cfg, ch)

        # propagate chunk meta into the batch meta
        extra_meta = {"chunk_text": ch.text}
        if isinstance(ch.meta, dict):
            extra_meta.update(ch.meta)

        batches.append(
            triple_batch_for_doc(
                doc_id=ch.doc_id,
                triples=triples,
                chunk_ids=[ch.chunk_id],
                extra_meta=extra_meta,
            )
        )
    return batches


def _extract_triples_with_llm(cfg, chunk: Chunk) -> List[Triple]:
    """
    Run the CyberTripleExtractor over the chunk text to produce triples.
    For DNRTI, gold labels in chunk.meta (prealigned_triples/entities/relations)
    are treated as reference only; they are not used to seed predictions.
    """
    model_name = getattr(cfg, "model_name", "gemma2:9b")
    ollama_url = getattr(getattr(cfg, "ollama", None), "base_url", "http://localhost:11434")
    doc_id = chunk.doc_id

    extractor = CyberTripleExtractor(
        file_path=None,
        document_id=doc_id,
        model_name=model_name,
        ollama_base_url=ollama_url,
    )
    extractor.text_override = chunk.text

    try:
        raw_results = extractor.run()
        print("good")
        extractor.build_dict(raw_results)
        print("stillgood")
        chunk_data = getattr(extractor, "chunk_data", raw_results)
        print("stillstillgood")
    except Exception as e:
        chunk_data = []
        print(f"[extractor] LLM extraction failed for {chunk.chunk_id}: {e}")

    triples: List[Triple] = []
    for entry in chunk_data or []:
        if not isinstance(entry, dict):
            continue
        # Case 1: entry is a triple dict directly
        if "subject" in entry and "predicate" in entry and "object" in entry:
            rels = [entry]
        else:
            rels = entry.get("triple", [])
            if not isinstance(rels, list):
                rels = []

        for t in rels:
            if not isinstance(t, dict):
                continue
            subj = t.get("subject", {})
            obj = t.get("object", {})
            pred = t.get("predicate", "")
            s_name = subj.get("name") if isinstance(subj, dict) else subj
            o_name = obj.get("name") if isinstance(obj, dict) else obj
            s_type = subj.get("type") if isinstance(subj, dict) else None
            o_type = obj.get("type") if isinstance(obj, dict) else None
            if not (s_name and o_name and pred):
                continue
            triples.append(
                Triple(
                    subject={"name": str(s_name), "type": s_type},
                    predicate=str(pred),
                    object={"name": str(o_name), "type": o_type},
                    meta=t.get("evidence"),
                )
            )

    print(f"[extractor] chunk_id={chunk.chunk_id} triples={len(triples)}")
    return triples


def _run_file(cfg, chunk_source: Path) -> Dict[str, Any]:
    model_name = getattr(cfg, "model_name", "gemma2:9b")
    ollama_url = getattr(getattr(cfg, "ollama", None), "base_url", "http://localhost:11434")
    doc_id = getattr(cfg, "document_id", chunk_source.stem)

    extractor = CyberTripleExtractor(
        file_path=str(chunk_source),
        document_id=doc_id,
        model_name=model_name,
        ollama_base_url=ollama_url,
    )

    raw_results = extractor.run()
    extractor.build_dict(raw_results)
    chunk_data = getattr(extractor, "chunk_data", raw_results)
    return {"data": chunk_data, "metrics": {"document_id": doc_id}}

