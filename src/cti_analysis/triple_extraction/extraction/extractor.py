from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from cti_analysis.models.documents import Chunk
from cti_analysis.models.triples import Entity, Triple, TripleBatch, triple_batch_for_doc
from cti_analysis.llm_backend import LlamaCppBackend
from cti_analysis.ontology import (
    TYPES, PREDS, EXTRACTION_PROMPT,
    ENTITY_EXTRACTION_PROMPT, RELATION_EXTRACTION_PROMPT,
    format_entity_hints,
)

logger = logging.getLogger(__name__)


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def _strip_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned


def _parse_pipe_entities(text: str) -> List[Dict]:
    """Parse pipe-delimited entity lines: 'name | type' per line."""
    if not text or not text.strip():
        return []

    cleaned = _strip_fences(text)
    if cleaned.upper() == "NONE":
        return []

    results = []
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        if line.startswith("entity_name") or line.startswith("---"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2:
            results.append({"name": parts[0], "type": parts[1]})

    return results


def _parse_pipe_triples(text: str) -> List[Dict]:
    """Parse pipe-delimited triple lines: 'subj | stype | pred | obj | otype' per line."""
    if not text or not text.strip():
        return []

    cleaned = _strip_fences(text)
    if cleaned.upper() == "NONE":
        return []

    results = []
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        if line.startswith("subject_name") or line.startswith("---"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 5:
            results.append({
                "subject": {"name": parts[0], "type": parts[1]},
                "predicate": parts[2],
                "object": {"name": parts[3], "type": parts[4]},
            })

    return results


def _parse_response(text: str) -> List[Dict]:
    """Parse LLM response — tries pipe-delimited first, then JSON fallback.

    Pipe-delimited (primary, finetuned models):
      APT28 | threat-actor | uses | Mimikatz | tool

    JSON (fallback, base models):
      [{"subject": {"name": "APT28", "type": "threat-actor"}, ...}]
    """
    if not text or not text.strip():
        return []

    cleaned = _strip_fences(text)
    if cleaned.upper() == "NONE":
        return []

    has_pipes = "|" in cleaned
    has_json = cleaned.lstrip().startswith("[") or cleaned.lstrip().startswith("{")

    # Pipe-delimited first (finetuned model output)
    if has_pipes and not has_json:
        return _parse_pipe_triples(cleaned)

    # JSON fallback (base model output)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        if has_pipes:
            return _parse_pipe_triples(cleaned)
        return []

    json_str = cleaned[start:end + 1]

    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # GGUF quantization artifact fixes
    fixed = re.sub(r'},(\s*)(subject)', r'},{\1\2', json_str)
    fixed = re.sub(r'(subject|object)":(name)', r'\1":{"\2', fixed)
    fixed = re.sub(r'(?<=[{,:\[])(\s*)([\w-]+)"', r'\1"\2"', fixed)
    try:
        data = json.loads(fixed)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Final fallback: try pipe parsing
    if has_pipes:
        return _parse_pipe_triples(text)

    return []


# =============================================================================
# TRIPLE VALIDATION
# =============================================================================

def _validate_triple(t: Dict) -> Tuple[Optional[Dict], List[str]]:
    """Validate a raw triple dict against TYPES and PREDS.

    Returns (normalized_triple, errors).
    - If errors is empty, the triple is valid.
    - If errors is non-empty, the triple is invalid but still returned
      (normalized as best we can) so repair can work on it.
    - Returns (None, errors) only for structurally unparseable triples.
    """
    errors = []

    if not isinstance(t, dict):
        return None, ["not_dict"]

    subj = t.get("subject", {})
    obj = t.get("object", {})
    pred = t.get("predicate", "")

    # Coerce string subjects/objects to dicts
    if isinstance(subj, str):
        subj = {"name": subj, "type": ""}
    if isinstance(obj, str):
        obj = {"name": obj, "type": ""}
    if not isinstance(subj, dict):
        return None, ["subject_not_dict"]
    if not isinstance(obj, dict):
        return None, ["object_not_dict"]

    s_name = str(subj.get("name", "") or "").strip()
    o_name = str(obj.get("name", "") or "").strip()
    s_type = str(subj.get("type", "") or "").strip()
    o_type = str(obj.get("type", "") or "").strip()
    pred = str(pred or "").strip()

    # Name checks
    if not s_name or len(s_name) < 2:
        errors.append("subject_name_empty")
    if not o_name or len(o_name) < 2:
        errors.append("object_name_empty")
    if not pred:
        errors.append("predicate_empty")

    # Self-loop
    if s_name and o_name and s_name.lower() == o_name.lower():
        errors.append("self_loop")

    # Type validation
    if s_type and s_type not in TYPES:
        errors.append("subject_type_invalid")
    if o_type and o_type not in TYPES:
        errors.append("object_type_invalid")
    if not s_type:
        errors.append("subject_type_missing")
    if not o_type:
        errors.append("object_type_missing")

    # Predicate validation
    if pred and pred not in PREDS:
        errors.append("predicate_invalid")

    normalized = {
        "subject": {"name": s_name, "type": s_type},
        "predicate": pred,
        "object": {"name": o_name, "type": o_type},
        "evidence": t.get("evidence"),
    }

    return normalized, errors


def _triple_key(t: Dict) -> Tuple[str, str, str]:
    """Deduplication key for a triple dict."""
    return (
        t["subject"]["name"].strip().lower(),
        t["predicate"],
        t["object"]["name"].strip().lower(),
    )


# =============================================================================
# ENTITY EXTRACTION (Pass 1 of two-pass mode)
# =============================================================================

def _extract_entities_from_chunk(
    text: str,
    backend: LlamaCppBackend,
    temperature: float,
    max_tokens: int,
    lora_id: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Extract entities from text using ENTITY_EXTRACTION_PROMPT.

    Returns deduplicated list of {"name": ..., "type": ...} dicts
    with validated STIX types.
    """
    if not text or len(text.strip()) < 10:
        return []

    prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
    response = backend.generate(
        prompt, temperature=temperature, max_tokens=max_tokens,
        lora_id=lora_id,
    )

    # Try pipe-delimited first, then JSON fallback
    raw = _parse_pipe_entities(response)
    if not raw:
        raw = _parse_response(response)

    entities = []
    seen: set = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        etype = str(item.get("type", "") or "").strip()
        if not name or len(name) < 2:
            continue
        if etype not in TYPES:
            continue
        key = (name.lower(), etype)
        if key not in seen:
            seen.add(key)
            entities.append({"name": name, "type": etype})

    return entities


# =============================================================================
# TRIPLE EXTRACTION (single-pass or Pass 2 of two-pass)
# =============================================================================

def _extract_from_chunk(
    text: str,
    backend: LlamaCppBackend,
    temperature: float,
    max_tokens: int,
    lora_id: Optional[int] = None,
    entity_hints: Optional[List[Dict[str, str]]] = None,
) -> Tuple[List[Dict], List[Dict], int]:
    """Extract triples from a single chunk of text.

    If entity_hints is provided, uses RELATION_EXTRACTION_PROMPT (two-pass mode).
    Otherwise uses EXTRACTION_PROMPT (single-pass mode).

    Returns (valid_triples, invalid_triples, parse_failure_count).
    """
    if not text or len(text.strip()) < 10:
        return [], [], 0

    if entity_hints is not None:
        hint_text = format_entity_hints(entity_hints)
        prompt = RELATION_EXTRACTION_PROMPT.format(
            text=text, entity_hints=hint_text,
        )
    else:
        prompt = EXTRACTION_PROMPT.format(text=text)

    response = backend.generate(
        prompt, temperature=temperature, max_tokens=max_tokens,
        lora_id=lora_id,
    )

    raw_triples = _parse_response(response)

    valid = []
    invalid = []
    parse_failures = 0
    seen: set = set()

    for raw_t in raw_triples:
        normalized, errors = _validate_triple(raw_t)

        if normalized is None:
            parse_failures += 1
            continue

        # Deduplicate
        key = _triple_key(normalized)
        if key in seen:
            continue
        seen.add(key)

        if errors:
            normalized["errors"] = errors
            normalized["context"] = text
            invalid.append(normalized)
        else:
            valid.append(normalized)

    return valid, invalid, parse_failures


# =============================================================================
# PIPELINE INTERFACE
# =============================================================================

def run_extraction(cfg, chunks: List[Chunk], backend: Optional[LlamaCppBackend] = None) -> List[TripleBatch]:
    """Extract triples from chunks using llama.cpp server.

    Supports two modes:
    - Single-pass (default): EXTRACTION_PROMPT does NER + RE in one call
    - Two-pass (cfg.use_two_pass=True): Pass 1 extracts entities (NER LoRA),
      Pass 2 classifies relations with entity hints (RE LoRA)

    Each TripleBatch contains:
    - triples: valid triples (pass TYPES/PREDS validation)
    - meta["invalid_triples"]: invalid triples with error reasons (for repair)
    - meta["parse_failures"]: count of unparseable response fragments
    """
    temperature = getattr(cfg, "temperature", 0.1)
    max_tokens = getattr(cfg, "max_tokens", 2048)
    two_pass = getattr(cfg, "use_two_pass", True)

    # Backend and LoRA IDs from config
    backend_cfg = getattr(cfg, "_backend_cfg", None)
    if backend is None:
        url = backend_cfg.url if backend_cfg else "http://localhost:8080"
        timeout = backend_cfg.timeout if backend_cfg else 300
        backend = LlamaCppBackend(url=url, timeout=timeout)

    ner_lora_id = backend_cfg.ner_lora_id if backend_cfg else 0
    re_lora_id = backend_cfg.re_lora_id if backend_cfg else 1

    mode_str = "two-pass (NER→RE)" if two_pass else "single-pass"
    logger.info(
        "[extractor] Extracting from %d chunks (mode=%s, ner_lora=%s, re_lora=%s)",
        len(chunks), mode_str, ner_lora_id, re_lora_id,
    )

    batches: List[TripleBatch] = []
    for ci, ch in enumerate(chunks):
        entity_hints = None
        if two_pass:
            # Pass 1: extract entities using NER adapter
            entity_hints = _extract_entities_from_chunk(
                ch.text, backend=backend,
                temperature=temperature, max_tokens=max_tokens,
                lora_id=ner_lora_id,
            )
            print(f"  [extract {ci+1}/{len(chunks)}] NER: {len(entity_hints)} entities | {ch.chunk_id}",
                  flush=True)

        # Pass 2 (or single-pass): extract triples using RE adapter
        valid_triples, invalid_triples, parse_failures = _extract_from_chunk(
            ch.text, backend=backend,
            temperature=temperature, max_tokens=max_tokens,
            lora_id=re_lora_id if two_pass else None,
            entity_hints=entity_hints,
        )

        print(f"  [extract {ci+1}/{len(chunks)}] RE: {len(valid_triples)} valid, {len(invalid_triples)} invalid",
              flush=True)

        # Convert valid triples to Triple objects
        triples: List[Triple] = []
        for t in valid_triples:
            triples.append(Triple(
                subject=Entity(name=t["subject"]["name"], type=t["subject"]["type"]),
                predicate=t["predicate"],
                object=Entity(name=t["object"]["name"], type=t["object"]["type"]),
                meta={"evidence": t.get("evidence")},
            ))

        # Build batch meta
        extra_meta = {"chunk_text": ch.text}
        if isinstance(ch.meta, dict):
            extra_meta.update(ch.meta)
        extra_meta["invalid_triples"] = invalid_triples
        extra_meta["parse_failures"] = parse_failures
        if entity_hints is not None:
            extra_meta["entity_hints"] = entity_hints

        batches.append(
            triple_batch_for_doc(
                doc_id=ch.doc_id,
                triples=triples,
                chunk_ids=[ch.chunk_id],
                extra_meta=extra_meta,
            )
        )

        logger.info(
            "[extractor] %s: valid=%d, invalid=%d, parse_failures=%d",
            ch.chunk_id, len(triples), len(invalid_triples), parse_failures,
        )

    return batches
