import json
import os
import requests
from docling.document_converter import DocumentConverter
from collections import defaultdict
from langchain_community.llms import Ollama
import spacy
import re
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try package-relative first, then local file import
try:
    from .consensus import consensus_filter
except ImportError:
    from consensus import consensus_filter

# ===== STIX ontology profile and eval adapter =====
from dataclasses import dataclass

# Toggle between extraction processes
@dataclass
class ExtractionOptions:
    # Stage switches
    run_sentence: bool = True
    run_semantic: bool = True
    consensus_on_sentence: bool = True
    consensus_on_chunk: bool = True

    # Mix multiple active stages, or select exactly one by priority
    combine_stages: bool = True  # set False to avoid mixing "Single" + "Consensus"

    # Consensus config
    consensus_passes: int = 4
    consensus_m: int = 2

    # Debug
    debug: bool = False

    # Performance
    batch_size: int = 10  # Process sentences in batches
    save_checkpoints: bool = False  # Save progress periodically
    
    def validate(self):
        """Validate configuration options."""
        if self.consensus_passes < 1:
            raise ValueError("consensus_passes must be at least 1")
        if self.consensus_m < 1:
            raise ValueError("consensus_m must be at least 1")
        if self.consensus_m > self.consensus_passes:
            raise ValueError("consensus_m cannot be greater than consensus_passes")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not (self.run_sentence or self.run_semantic or self.consensus_on_sentence or self.consensus_on_chunk):
            logger.warning("No extraction stages enabled - no extraction will occur")
        return True

# -----------------------------
# spaCy bootstrap
# -----------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# -----------------------------
# Ollama helper
# -----------------------------
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

# -----------------------------
# Prompt builders
# -----------------------------
def _extractor_sentence_prompt(text: str, entity_types: str, predicates: str) -> str:
    return f"""
You are a cybersecurity analyst extracting structured intelligence using STIX 2.1 ontology.

TASK:
From the sentence below, extract at most 5 subject–predicate–object triples.

RULES:
1. Subject.type and Object.type MUST be in: [{entity_types}]
2. Predicate MUST be in: [{predicates}]
3. Use ONLY explicit relationships stated in the text (no guesses).
4. Return "NO_TRIPLES" if none.
5. Names must match the sentence exactly.
6. Include an exact quote from the sentence for each triple.

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

def _chunk_prompt(chunk_text: str, entity_types: str, predicates: str) -> str:
    return f"""
You are a cybersecurity analyst extracting structured intelligence from a chunk of text using STIX 2.1 ontology.

TASK:
From the chunk below, extract at most 8 subject–predicate–object triples.
Scope only within this chunk.

RULES:
1. Subject.type and Object.type MUST be in: [{entity_types}]
2. Predicate MUST be in: [{predicates}]
3. Use ONLY explicit relations in the chunk.
4. Return "NO_TRIPLES" if none.
5. Names must match exactly.
6. Include an exact quote from the chunk for each triple.

OUTPUT (JSON array only) as above.

Chunk:
\"\"\"{chunk_text}\"\"\"""".strip()

def _consensus_variants(text: str, entity_types: str, predicates: str) -> List[str]:
    # 4 stylistically different prompts to diversify reasoning, same schema
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

# -----------------------------
# Utility: debug + dedupe keys
# -----------------------------
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

def _options_dict(opts: "ExtractionOptions") -> Dict[str, Any]:
    if not isinstance(opts, ExtractionOptions):
        return {}
    return {
        "run_sentence": bool(opts.run_sentence),
        "run_semantic": bool(opts.run_semantic),
        "consensus_on_sentence": bool(opts.consensus_on_sentence),
        "consensus_on_chunk": bool(opts.consensus_on_chunk),
        "combine_stages": bool(opts.combine_stages),
        "consensus_passes": int(opts.consensus_passes),
        "consensus_m": int(opts.consensus_m),
    }

# -----------------------------
# Main extractor
# -----------------------------
class CyberTripleExtractor:
    def __init__(self, file_path, document_id, model_name, ollama_base_url="http://localhost:11434"):
        self.file_path = file_path
        self.ollama_base_url = ollama_base_url
        self.converter = DocumentConverter()
        self.model_name = model_name
        self.doc_sha256_id = document_id

        ensure_ollama_model(model_name=self.model_name, base_url=self.ollama_base_url)
        self.llm = Ollama(
            model=model_name,
            base_url=self.ollama_base_url,
            num_ctx=2048,
            format="json",
            stop=["</think>", "<think>"]
        )

        # STIX 2.1 ontology
        self.stix_classes = [
            'attack-pattern', 'campaign', 'course-of-action', 'identity', 'indicator', 
            'infrastructure', 'intrusion-set', 'location', 'malware', 'malware-analysis', 
            'report', 'threat-actor', 'tool', 'vulnerability', 'observed-data', 
            'grouping', 'note', 'opinion'
        ]
        self.stix_predicates = [
            "uses", "targets", "indicates", "attributed-to", "mitigates", "compromises",
            "delivers", "drops", "downloads", "communicates-with", "exfiltrates-to",
            "controls", "located-at", "part-of", "related-to"
        ]

        # phrase matcher for rule filter (using STIX classes)
        patterns = [nlp.make_doc(term) for term in self.stix_classes]
        matcher.add("STIX", patterns)

        # metrics
        self.valid_triples: List[Dict[str, Any]] = []
        self.invalid_triples: List[Dict[str, Any]] = []
        self.chunk_data: List[Dict[str, Any]] = []
        self.sentences_total = 0
        self.sentences_used = 0
        self.raw_triples = 0
        self.suspicious_triples = 0
        self.rejection_stats = {
            "bad_structure": 0,
            "invalid_class_or_predicate": 0
        }
        self.pages_parsed = 0
        self.runtime_seconds = 0
        self.per_page_stats: Dict[int, Dict[str, int]] = {}  # Track stats per page
        self.rejection_reasons: Dict[str, int] = defaultdict(int)  # Count each rejection reason

        # record options used in last run
        self._last_run_options: Dict[str, Any] = {}

    # -------------------------
    # Filters
    # -------------------------
    def is_relevant_with_ner(self, sentence: str) -> bool:
        doc = nlp(sentence)
        entity_labels = {ent.label_ for ent in doc.ents}
        return bool(entity_labels & {"ORG", "PRODUCT", "GPE", "DATE"})

    def rule_based_filter(self, sentence: str) -> bool:
        doc = nlp(sentence)
        matches = matcher(doc)
        return len(matches) > 0

    # -------------------------
    # Page chunking
    # -------------------------
    def chunk_by_page(self, doc):
        page_chunks = defaultdict(str)
        for text_item in doc.texts:
            if not getattr(text_item, "prov", None):
                continue
            if hasattr(text_item, "content_layer") and text_item.content_layer != "body":
                continue
            try:
                page_no = text_item.prov[0].page_no
                page_chunks[page_no] += " " + text_item.text.strip()
            except Exception:
                continue
        return sorted(page_chunks.items())

    # -------------------------
    # Internal helpers
    # -------------------------
    def _maxmin_chunks(self, sentences: List[str]) -> List[str]:
        from .Chunking_Methods.maxmin_chunking import maxmin_semantic_chunks
        meta = maxmin_semantic_chunks(sentences)
        return [m["text"] for m in meta]

    def _extract_single(self, prompt_text: str) -> List[Dict[str, Any]]:
        """invoke LLM once and safely parse to list of dict triples"""
        resp = self.llm.invoke(prompt_text)
        try:
            if not resp or resp.strip().upper() == "NO_TRIPLES":
                triples = []
            else:
                triples = json.loads(resp)
            if isinstance(triples, dict): triples = [triples]
            if isinstance(triples, str):  triples = []  # guard stray strings
            if not isinstance(triples, list): triples = []
        except Exception:
            triples = []
        self.raw_triples += len(triples)
        return triples

    def _consensus_runs(self, unit: str, entity_types: str, predicates: str, passes: int, debug: bool) -> List[List[Dict[str, Any]]]:
        all_lists: List[List[Dict[str, Any]]] = []
        variants = _consensus_variants(unit, entity_types, predicates)
        num_vars = len(variants) or 1
        for i in range(max(1, passes)):
            ptxt = variants[i % num_vars]
            print(f"      [Consensus] pass {i+1}/{passes} (variant {i % num_vars + 1}/{num_vars})")
            _dbg(debug, f"[CONSENSUS] pass {i+1}/{passes}")
            out = self._extract_single(ptxt)
            # keep only dicts
            out = [t for t in out if isinstance(t, dict)]
            all_lists.append(out)
        return all_lists

    def _consensus_merge(self, runs: List[List[Dict[str, Any]]], m: int) -> List[Dict[str, Any]]:
        """
        Delegate to the shared consensus filter which:
        - whitelists allowed types/preds
        - merges near-duplicates by name similarity (tau) and quote overlap
        - applies quorum m
        """
        # clean to list[list[dict]]
        cleaned = []
        for lst in runs or []:
            if not isinstance(lst, list):
                continue
            cleaned.append([t for t in lst if isinstance(t, dict)])

        kept = consensus_filter(
            candidate_lists=cleaned,
            allowed_types=set(self.stix_classes),
            allowed_preds=set(self.stix_predicates),
            m=max(2, m),
            tau_name=0.50,
        )
        print(f"      [Consensus] quorum m={m} -> kept={len(kept)}")
        return kept

    # -------------------------
    # RUN (multi-stage capable)
    # -------------------------
    def run(self, options: ExtractionOptions = ExtractionOptions()):
        # Validate options
        try:
            options.validate()
        except ValueError as e:
            logger.error(f"Invalid extraction options: {e}")
            raise
        
        t0 = time.time()
        logger.info("Converting document...")
        result = self.converter.convert(self.file_path)
        doc = result.document
        page_chunks = self.chunk_by_page(doc)
        self.pages_parsed = len(page_chunks)
        logger.info(f"Document converted. Pages detected: {self.pages_parsed}")

        # remember options for save_to_json
        self._last_run_options = _options_dict(options)

        print("[Stage] Extracting triples...")
        chunk_results = []
        entity_types = ", ".join(self.stix_classes)
        predicates = ", ".join(self.stix_predicates)

        # announce active stages
        print(f"[Config] consensus_passes={options.consensus_passes}  consensus_m={options.consensus_m}")

        for page_idx, (page_no, page_text) in enumerate(page_chunks, start=1):
            print(f"[Page] {page_idx}/{len(page_chunks)} -> page_no={page_no}")
            doc_spacy = nlp(page_text)
            sentences = [s.text for s in doc_spacy.sents]
            print(f"  [Page] sentences={len(sentences)}")

            units_sentence = sentences
            units_chunk = self._maxmin_chunks(sentences) if options.run_semantic or options.consensus_on_chunk else []
            if units_chunk:
                print(f"  [Page] chunk_units={len(units_chunk)}")

            # Stage registry candidates
            candidates: List[Tuple[str, List[str], bool]] = []
            if options.run_sentence:
                candidates.append(("SENTENCE_SINGLE", units_sentence, False))
            if options.consensus_on_sentence:
                candidates.append(("SENTENCE_CONSENSUS", units_sentence, True))
            if options.run_semantic:
                candidates.append(("CHUNK_SINGLE", units_chunk, False))
            if options.consensus_on_chunk:
                candidates.append(("CHUNK_CONSENSUS", units_chunk, True))

            # If exclusive mode, pick one by priority
            if not options.combine_stages and candidates:
                priority = ["SENTENCE_CONSENSUS", "SENTENCE_SINGLE", "CHUNK_CONSENSUS", "CHUNK_SINGLE"]
                chosen = None
                for p in priority:
                    for name, units, use_cons in candidates:
                        if name == p:
                            chosen = [(name, units, use_cons)]
                            break
                    if chosen:
                        break
                stages = chosen or []
                print("[Config] combine_stages=False → selecting only:", stages[0][0] if stages else "NONE")
            else:
                stages = candidates

            print("[Config] Active stages:", ", ".join(n for n,_,__ in stages) if stages else "NONE")

            for stage_name, units, use_consensus in stages:
                print(f"  [StageRun] {stage_name} units={len(units)}")
                for i, unit in enumerate(units):
                    if stage_name.startswith("SENTENCE"):
                        self.sentences_total += 1
                    if len(unit.strip()) < 40:
                        if i % 10 == 0:
                            print(f"    [Skip] unit {i}: too short")
                        continue
                    if not (self.rule_based_filter(unit) or self.is_relevant_with_ner(unit)):
                        if i % 10 == 0:
                            print(f"    [Skip] unit {i}: prefilter")
                        continue
                    if stage_name.startswith("SENTENCE"):
                        self.sentences_used += 1

                    # heartbeat every 10 units
                    if i % 10 == 0:
                        print(f"    [Run] unit {i}/{len(units)}")

                    try:
                        if use_consensus:
                            runs = self._consensus_runs(
                                unit=unit,
                                entity_types=entity_types,
                                predicates=predicates,
                                passes=options.consensus_passes,
                                debug=False  # prints are handled outside
                            )
                            triples = self._consensus_merge(runs, m=options.consensus_m)
                            print(f"      [Consensus] kept={len(triples)}")
                        else:
                            if stage_name.startswith("CHUNK"):
                                ptxt = _chunk_prompt(unit, entity_types, predicates)
                            else:
                                ptxt = _extractor_sentence_prompt(unit, entity_types, predicates)
                            triples = self._extract_single(ptxt)
                            print(f"      [Single] kept={len(triples)}")

                        kept = [t for t in triples if self._is_valid_triple(t)]
                        invalid = [t for t in triples if t not in kept]

                        # rejection accounting
                        self.rejection_stats["bad_structure"] += sum(
                            1 for t in invalid
                            if not isinstance(t, dict) or not isinstance(t.get("predicate"), str)
                        )
                        self.rejection_stats["invalid_class_or_predicate"] += sum(
                            1 for t in invalid
                            if isinstance(t, dict) and isinstance(t.get("predicate"), str)
                            and (
                                t.get("predicate", "").strip() not in self.stix_predicates
                                or (isinstance(t.get("subject"), dict) and t["subject"].get("type") not in self.stix_classes)
                                or (isinstance(t.get("object"), dict) and t["object"].get("type") not in self.stix_classes)
                            )
                        )

                        # Store invalid triples with context and categorized reason
                        for inv_triple in invalid:
                            rejection_reason = "unknown"
                            if not isinstance(inv_triple, dict):
                                rejection_reason = "not_dict"
                            elif not isinstance(inv_triple.get("predicate"), str):
                                rejection_reason = "invalid_predicate_type"
                            elif inv_triple.get("predicate", "").strip() not in self.stix_predicates:
                                rejection_reason = "predicate_not_in_ontology"
                            elif isinstance(inv_triple.get("subject"), dict) and inv_triple["subject"].get("type") not in self.stix_classes:
                                rejection_reason = "subject_type_not_in_ontology"
                            elif isinstance(inv_triple.get("object"), dict) and inv_triple["object"].get("type") not in self.stix_classes:
                                rejection_reason = "object_type_not_in_ontology"
                            else:
                                rejection_reason = "malformed_structure"

                            self.rejection_reasons[rejection_reason] += 1
                            self.invalid_triples.append({
                                "triple": inv_triple,
                                "context": unit,
                                "page_number": page_no,
                                "unit_index": i,
                                "rejection_reason": rejection_reason,
                                "stage": stage_name
                            })
                        
                        # Track per-page statistics
                        if page_no not in self.per_page_stats:
                            self.per_page_stats[page_no] = {"valid": 0, "invalid": 0, "total": 0}
                        self.per_page_stats[page_no]["valid"] += len(kept)
                        self.per_page_stats[page_no]["invalid"] += len(invalid)
                        self.per_page_stats[page_no]["total"] += len(triples)

                        final_triples = _dedupe(kept)
                        chunk_results.append((unit, page_no, i, final_triples))

                        for t in final_triples:
                            if self._is_valid_triple(t):
                                self.valid_triples.append(t)
                            else:
                                self.suspicious_triples += 1

                    except Exception as e:
                        print(f"      [LLM] error at page {page_no} {stage_name} unit {i}: {e}")

        self.runtime_seconds = round(time.time() - t0, 2)
        print(f"[Done] runtime_seconds={self.runtime_seconds}  valid={len(self.valid_triples)}  raw_seen={self.raw_triples}")
        return chunk_results

    # -------------------------
    # Output builders
    # -------------------------
    def build_dict(self, chunk_results):
        """Build chunk_data using only valid triples (suspicious ones are excluded)."""
        self.chunk_data = []
        for sentence, page_no, i, triples in chunk_results:
            valid_only = [t for t in triples if self._is_valid_triple(t)]
            if not valid_only:
                continue
            self.chunk_data.append({
                "context": sentence,
                "triple": valid_only,
                "metadata": {
                    "page_number": page_no,
                    "id": str(i).zfill(3),
                    "source": "TEXT"
                }
            })
        return self.chunk_data

    def safe_filename(self, name: str) -> str:
        return re.sub(r'[<>:"/\\|?*]', '_', name)

    def save_to_json(self, output_filename="chunk_data.json", base_dir=None):
        try:
            # Decide output path
            if base_dir is not None:
                os.makedirs(base_dir, exist_ok=True)
                filename = self.safe_filename(os.path.basename(output_filename))
                output_path = os.path.join(base_dir, filename)
            elif os.path.isabs(output_filename) or os.path.dirname(output_filename):
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                output_path = output_filename
            else:
                input_dir = os.path.dirname(self.file_path)
                output_dir = os.path.join(input_dir, "extracted_triples")
                os.makedirs(output_dir, exist_ok=True)
                filename = self.safe_filename(output_filename)
                output_path = os.path.join(output_dir, filename)

            metrics = {
                "file_name": os.path.basename(self.file_path),
                "model_used": self.model_name,
                "doc_sha256_id": self.doc_sha256_id,
                "num_pages": self.pages_parsed,
                "num_sentences_total": self.sentences_total,
                "num_sentences_used": self.sentences_used,
                "num_raw_triples": self.raw_triples,
                "num_valid_triples": len(self.valid_triples),
                "num_invalid_triples": len(self.invalid_triples),
                "num_suspicious_triples": self.suspicious_triples,
                "avg_triples_per_sentence": self.raw_triples / self.sentences_used if self.sentences_used else 0,
                "rejection_stats": self.rejection_stats,
                "rejection_reasons": dict(self.rejection_reasons),
                "per_page_stats": self.per_page_stats,
                "runtime_seconds": self.runtime_seconds,
                "processes_used": self._last_run_options,
            }

            metadata = {
                "metrics": metrics,
                "data": self.chunk_data
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"[IO] File saved to: {output_path}")
            
            # Save invalid triples to separate file
            if self.invalid_triples:
                invalid_output_path = output_path.replace(".json", "_invalid.json")
                invalid_metadata = {
                    "metrics": {
                        "file_name": os.path.basename(self.file_path),
                        "model_used": self.model_name,
                        "doc_sha256_id": self.doc_sha256_id,
                        "num_invalid_triples": len(self.invalid_triples),
                        "rejection_stats": self.rejection_stats,
                        "rejection_reasons": dict(self.rejection_reasons),
                    },
                    "invalid_triples": self.invalid_triples
                }
                with open(invalid_output_path, "w", encoding="utf-8") as f:
                    json.dump(invalid_metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"Invalid triples saved to: {invalid_output_path}")
        except Exception as e:
            print(f"[IO] Failed to save JSON: {e}")

    # -------------------------
    # Validation
    # -------------------------
    def _is_valid_triple(self, triple):
        if not isinstance(triple, dict):
            return False
        s = triple.get("subject")
        o = triple.get("object")
        p = triple.get("predicate")
        if not isinstance(p, str):
            return False
        if not isinstance(s, dict) or not isinstance(o, dict):
            return False
        subject_type = s.get("type")
        object_type = o.get("type")
        if not isinstance(subject_type, str) or not isinstance(object_type, str):
            return False
        return (
            p.strip() in self.stix_predicates and
            subject_type in self.stix_classes and
            object_type in self.stix_classes
        )

    def print_summary(self):
        """Print a summary of extraction results."""
        logger.info("=" * 60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"File: {os.path.basename(self.file_path)}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Pages processed: {self.pages_parsed}")
        logger.info(f"Runtime: {self.runtime_seconds}s")
        logger.info("-" * 60)
        logger.info(f"Sentences total: {self.sentences_total}")
        logger.info(f"Sentences used: {self.sentences_used} ({100*self.sentences_used/self.sentences_total:.1f}%)" if self.sentences_total > 0 else "Sentences used: 0")
        logger.info(f"Raw triples extracted: {self.raw_triples}")
        logger.info(f"Valid triples: {len(self.valid_triples)} ({100*len(self.valid_triples)/self.raw_triples:.1f}%)" if self.raw_triples > 0 else "Valid triples: 0")
        logger.info(f"Invalid triples: {len(self.invalid_triples)} ({100*len(self.invalid_triples)/self.raw_triples:.1f}%)" if self.raw_triples > 0 else "Invalid triples: 0")
        logger.info(f"Suspicious triples: {self.suspicious_triples}")
        logger.info("-" * 60)
        logger.info("Rejection reasons:")
        for reason, count in sorted(self.rejection_reasons.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {reason}: {count}")
        logger.info("-" * 60)
        logger.info("Top 3 pages by valid triples:")
        sorted_pages = sorted(self.per_page_stats.items(), key=lambda x: x[1]["valid"], reverse=True)[:3]
        for page, stats in sorted_pages:
            logger.info(f"  Page {page}: {stats['valid']} valid, {stats['invalid']} invalid")
        logger.info("=" * 60)

# -----------------------------
# Quick manual run (optional)
# -----------------------------
if __name__ == "__main__":
    # Example: run sentence + consensus (+ optional chunk) together
    model = "gemma2:9b"
    extractor = CyberTripleExtractor(
        "./datasets/CTI-HAL/reports/apt29/AnalysisOfCyberattackOnUS.pdf",
        document_id="APT29_US_DEMO",
        model_name=model,
        ollama_base_url="http://localhost:11434"
    )
    opts = ExtractionOptions(
        run_sentence=True,
        consensus_on_sentence=True,
        run_semantic=True,
        consensus_on_chunk=True,
        combine_stages=False,   # set to False to run only one stage
        consensus_passes=4,
        consensus_m=2,
        debug=False,
        batch_size=10,
        save_checkpoints=False
    )
    results = extractor.run(options=opts)
    extractor.build_dict(results)
    out_name = extractor.safe_filename(f"chunk_data_{model}.json")
    extractor.save_to_json(out_name)
    extractor.print_summary()
