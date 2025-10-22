import json
import os
import requests
from docling.document_converter import DocumentConverter
from collections import defaultdict
from langchain_community.llms import Ollama
import spacy
import re
from spacy.matcher import PhraseMatcher
import numpy as np
import math
from ontologies.validator import validate_triple_record, tally_validity
from ontologies.malont_adapter import normalize_and_map_malont
from ontologies.ontology_loader import get_ontology
from ontologies.malont import get_config 
from maxmin_chunking import maxmin_semantic_chunks 
# -------- spaCy setup --------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# -------- feature flags --------
USE_CONTEXT_WINDOW = True
CONTEXT_WINDOW_K = 1  # prev/next sentence window size

def ensure_ollama_model(model_name="mistral", base_url="http://localhost:11434"):
    try:
        resp = requests.get(f"{base_url}/api/tags")
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        if model_name not in models:
            print(f"Model '{model_name}' not found. Downloading...")
            pull_resp = requests.post(f"{base_url}/api/pull", json={"name": model_name})
            pull_resp.raise_for_status()
            print(f"Model '{model_name}' downloaded.")
        else:
            print(f"Model '{model_name}' already available.")
    except Exception as e:
        print("Error checking or downloading model:", e)

def _ollama_embed(texts, model="nomic-embed-text", base_url="http://localhost:11434", timeout=30):
            """Return L2-normalized embeddings from Ollama /api/embeddings."""
            vecs = []
            for t in texts:
                try:
                    r = requests.post(
                        f"{base_url.rstrip('/')}/api/embeddings",
                        json={"model": model, "prompt": t},
                        timeout=timeout
                    )
                    r.raise_for_status()
                    v = np.array(r.json()["embedding"], dtype=float)
                    n = np.linalg.norm(v) or 1.0
                    vecs.append(v / n)
                except Exception:
                    vecs.append(None)
            return vecs


class CyberTripleExtractor:
    def __init__(self, file_path, model_name="mistral", ollama_base_url="http://localhost:11434"):
        self.file_path = file_path
        self.ollama_base_url = ollama_base_url
        self.converter = DocumentConverter()
        self.model_name = model_name

        # Ensure the Ollama model is present before constructing the client
        ensure_ollama_model(model_name=self.model_name, base_url=self.ollama_base_url)
        self.llm = Ollama(
            model=model_name,
            base_url=self.ollama_base_url,
            num_ctx=2048,
            format="json",
            stop=["</think>", "<think>"],
        )

        self.ontology = get_ontology("connection")
        self.malont_onto = get_config()  # MALONT surface vocab for the prompt
        self.MALONT_TYPES = sorted(self.malont_onto.OBJECT_TYPES | self.malont_onto.LITERAL_TYPES)
        self.MALONT_PREDS = sorted(self.malont_onto.PREDICATES)


        self.SUBJECT_TYPES        = self.ontology.SUBJECT_TYPES
        self.OBJECT_TYPES         = self.ontology.OBJECT_TYPES
        self.PREDICATES           = self.ontology.PREDICATES
        self.LITERAL_TYPES        = self.ontology.LITERAL_TYPES
        self.ATTRIBUTE_PREDICATES = self.ontology.ATTRIBUTE_PREDICATES
        self.RELATION_PREDICATES  = self.ontology.RELATION_PREDICATES

        # Optional: if you have a hierarchy map in the ontology
        self.PARENT_OF = getattr(self.ontology, "PARENT_OF", {})

       
        # Seed an ontology term matcher for cheap pre-screening
        patterns = [nlp.make_doc(t.split(":")[-1]) for t in sorted(self.ontology.SUBJECT_TYPES)]
        if patterns:
            matcher.add("ONTO", patterns)


            # --- Semantic chunking knobs ---
        self.MM_LOOKAHEAD        = 6          # 4–8 is a sweet spot
        self.MM_MIN_SIM_THRESH   = 0.38       # 0.30–0.45 depending on topic tightness
        self.MM_MAX_WORDS        = 260        # soft size cap to avoid mega-chunks
        self.MM_OVERLAP_WORDS    = 50         # to preserve context across splits
        self.SAVE_CHUNKS_PREVIEW = True       # write .json/.csv/.html previews

        self.TOP_SENTENCES_PER_CHUNK = 3
        self.CHUNK_CONTEXT_LIMIT     = 1200

        # IOC/TTP detectors (cheap + effective)
        self._ioc_patterns = {
            "ipv4": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
            "domain": re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:[a-z]{2,})\b", re.I),
            "url": re.compile(r"\bhttps?://[^\s)>\]}]+", re.I),
            "cve": re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I),
            "md5": re.compile(r"\b[a-f0-9]{32}\b", re.I),
            "sha1": re.compile(r"\b[a-f0-9]{40}\b", re.I),
            "sha256": re.compile(r"\b[a-f0-9]{64}\b", re.I),
            "port": re.compile(r"\bport\s*(?:\d{1,5})\b", re.I),
        }
        self._cti_keywords = {
            "rat","remote access trojan","botnet","c2","command and control","dropper",
            "exploit kit","webshell","keylogger","wiper","exfiltration","phishing",
            "ddos","rootkit","backdoor","loader","persistence","lateral movement"
        }

        # Metrics
        self.suspicious_triples = 0
        self.valid_triples = []
        self.chunk_data = []
        self.sentences_total = 0
        self.sentences_used = 0
        self.raw_triples = 0
        self.rejection_stats = {"bad_structure": 0, "invalid_class_or_predicate": 0}
        self.pages_parsed = 0
        self.runtime_seconds = 0
        self.lenient_accept = True  # or False if you only want strict


    # ---------- prompt with optional context window ----------
    def generate_prompt(self, text, ctx_before="", ctx_after=""):
        entity_types = ", ".join(sorted(self.MALONT_TYPES))
        predicates = ", ".join(sorted(self.MALONT_PREDS))
        return f"""
You are a cybersecurity analyst extracting structured intelligence.

TASK:
Extract at most 5 subject–predicate–object triples from the TARGET sentence only.

CONSTRAINTS:
1) Subject.type and Object.type MUST be in: [{entity_types}]
2) Predicate MUST be in: [{predicates}]
3) Use ONLY relationships explicitly stated in the TARGET sentence (no guessing).
4) You MAY use the CONTEXT (neighboring sentences) only to resolve references (e.g., pronouns, aliases).
5) Every triple MUST include an exact quote from the TARGET sentence as evidence.
6) If no valid triples exist in the TARGET sentence, return "NO_TRIPLES".

CONTEXT (optional, for disambiguation only):
BEFORE: {ctx_before}
AFTER: {ctx_after}

TARGET SENTENCE:
\"\"\"{text}\"\"\"

OUTPUT (JSON array only):
[
  {{
    "subject": {{"name": "<string>", "type": "<type-from-list>"}},
    "predicate": "<predicate-from-list>",
    "object": {{"name": "<string>", "type": "<type-from-list>"}},
    "evidence": {{"quote": "<exact substring from TARGET sentence>"}}
  }}
]
""".strip()

    # ---------- relevance gate ----------
    def _has_ioc_or_cti_terms(self, text: str) -> bool:
        s = text.lower()
        if any(k in s for k in self._cti_keywords):
            return True
        for rx in self._ioc_patterns.values():
            if rx.search(text):
                return True
        return False

    def is_relevant(self, sentence: str) -> bool:
        doc = nlp(sentence)
        entity_labels = {ent.label_ for ent in doc.ents}
        ner_hit  = bool(entity_labels & {"ORG", "PRODUCT", "GPE", "DATE"})
        onto_hit = len(matcher(doc)) > 0
        ioc_hit  = self._has_ioc_or_cti_terms(sentence)
        return ner_hit or onto_hit or ioc_hit

    def _neighbor_context(self, sentences, idx, k=1):
        left = sentences[max(0, idx - k): idx]
        right = sentences[idx + 1: idx + 1 + k]
        return " ".join(left).strip(), " ".join(right).strip()

    # ---------- conversion & chunking ----------
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
    def ollama_embed(self, texts, model="nomic-embed-text", base_url=None, timeout=30):
        return _ollama_embed(
            texts, model=model, base_url=base_url or self.ollama_base_url, timeout=timeout
    )

    


    def run(self):
        import time, json as _json
        start_time = time.time()
        print("Loading and converting document...")
        result = self.converter.convert(self.file_path)
        doc = result.document

        page_chunks = self.chunk_by_page(doc)
        chunk_results = []
        self.pages_parsed = len(page_chunks)

        print("Extracting triples using semantic chunks...")
        for page_no, page_text in page_chunks:
            # 1) sentence split for the page
            doc_spacy = nlp(page_text)
            sentences = [sent.text for sent in doc_spacy.sents]

            # 2) build Max–Min chunks
            chunks = maxmin_semantic_chunks(
                sentences,
                base_url=self.ollama_base_url,
                embed_model="nomic-embed-text",
                lookahead=self.MM_LOOKAHEAD,
                min_sim_threshold=self.MM_MIN_SIM_THRESH,
                max_words=self.MM_MAX_WORDS,
                overlap_words=self.MM_OVERLAP_WORDS,
            )

            # 3) loop through chunks
            for ch_id, ch in enumerate(chunks):
                s_idx, e_idx = ch["start"], ch["end"]
                ch_sents = sentences[s_idx:e_idx]

                # quick relevance gate at chunk level (any IOC/CTI keyword inside?)
                if not any(self._has_ioc_or_cti_terms(s) for s in ch_sents):
                    continue

                # 4) sentence-level scoring inside the chunk
                sent_embs = self.ollama_embed(ch_sents, model="nomic-embed-text")
                scored = []
                for local_i, s in enumerate(ch_sents):
                    global_i = s_idx + local_i
                    self.sentences_total += 1
                    if len(s.strip()) < 40:
                        continue
                    onto_hit = len(matcher(nlp(s))) > 0
                    ioc_hit  = self._has_ioc_or_cti_terms(s)
                    if not (onto_hit or ioc_hit):
                        continue

                    # cosine to chunk centroid (if available)
                    cos_to_centroid = 0.0
                    if ch["centroid"] is not None and sent_embs[local_i] is not None:
                        cos_to_centroid = float(np.dot(ch["centroid"], sent_embs[local_i]))

                    # simple weighted score
                    score = 0.6 * cos_to_centroid + 0.25 * (1.0 if ioc_hit else 0.0) + 0.15 * (1.0 if onto_hit else 0.0)
                    scored.append((score, global_i, s))

                if not scored:
                    continue

                scored.sort(reverse=True, key=lambda x: x[0])
                topN = scored[: min(self.TOP_SENTENCES_PER_CHUNK, len(scored))]

                # 5) extract from each top sentence with the chunk as context
                for _, global_i, sentence in topN:
                    self.sentences_used += 1
                    try:
                        # Pass chunk text as compact context (no neighbor sentences)
                        ctx_text = ch["text"][:self.CHUNK_CONTEXT_LIMIT]
                        prompt = self.generate_prompt(
                            sentence,
                            ctx_before="",
                            ctx_after=f"[CHUNK CONTEXT] {ctx_text}"
                        )

                        print(f"\n--- Page {page_no} | Chunk {ch_id} | Sentence {global_i} ---")
                        print("Prompt sent to LLM:")
                        response = self.llm.invoke(prompt)
                        print("Raw LLM response:")
                        print(response)

                        # 6) robust JSON handling
                        # try:
                        #     triples = _json.loads(response)
                        # except Exception:
                        #     triples = []
                        # if isinstance(triples, dict):
                        #     triples = [triples]
                        # elif isinstance(triples, str):
                        #     triples = [] if triples.strip().upper() in {"NO_TRIPLES","NO RELATED ENTITIES AND RELATIONS.","NONE"} else []
                        # elif not isinstance(triples, list):
                        #     triples = []

                        # self.raw_triples += len(triples)
                        def _coerce_triple_list(x):
                            import json as _json
                            if x is None:
                                return []
                            if isinstance(x, list):
                                return [t for t in x if isinstance(t, dict)]
                            if isinstance(x, dict):
                                return [x]
                            if isinstance(x, str):
                                s = x.strip()
                                if s.upper() in {"NO_TRIPLES", "NONE", "NO RELATED ENTITIES AND RELATIONS."}:
                                    return []
                                try:
                                    j = _json.loads(s)
                                    return _coerce_triple_list(j)
                                except Exception:
                                    return []
                            return []

                        raw = response  # whatever self.llm.invoke(...) returned
                        triples = _coerce_triple_list(raw)

                        # 7) normalize + validate
                        norm_pack = []
                        for raw_t in (triples or []):
                            clean_malont, mapped = normalize_and_map_malont(raw_t)
                            rec = dict(clean_malont)
                            rec["mapped"] = mapped
                            outcome = validate_triple_record(
                                rec, self.ontology,
                                lenient_accept=self.lenient_accept,
                                target_field="mapped"
                            )
                            if outcome != "reject":
                                print(f"{rec['subject']} —{rec['predicate']}→ {rec['object']}  [{outcome}]")
                                rec["_chunk"] = {
                                    "page": page_no,
                                    "chunk_id": ch_id,
                                    "start": s_idx,
                                    "end": e_idx,
                                    "min_pair_sim": ch.get("min_pair_sim", None),
                                    "size_words": ch.get("size_words", None),
                                }
                                norm_pack.append(rec)
                                self.valid_triples.append(rec)
                            else:
                                print(f"Suspicious triple (rejected): {rec}")
                                self.suspicious_triples += 1

                        # 8) store accepted recs for this sentence
                        chunk_results.append((sentence, page_no, global_i, norm_pack, "", ""))

                    except Exception as e:
                        print(f"LLM error on page {page_no} sentence {global_i}: {e}")

        self.runtime_seconds = time.time() - start_time
        return chunk_results


    def build_dict(self, chunk_results):
        self.chunk_data = []
        for sentence, page_no, i, triples, ctx_before, ctx_after in chunk_results:
            valid_only = [t for t in triples if self._is_valid_triple(t)]
            if not valid_only:
                continue
            self.chunk_data.append({
                "context": sentence,
                "triple": valid_only,
                "metadata": {
                    "page_number": page_no,
                    "id": str(i).zfill(3),
                    "source": "TEXT",
                    "context_before": ctx_before,
                    "context_after": ctx_after,
                    "window_k": CONTEXT_WINDOW_K if USE_CONTEXT_WINDOW else 0,
                }
            })
        return self.chunk_data

    def safe_filename(self, name: str) -> str:
        return re.sub(r'[<>:"/\\|?*]', '_', name)

    def save_to_json(self, output_filename="chunk_data.json", base_dir=None):
        try:
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
            
            
            strict = lenient = reject = 0
            for item in self.chunk_data:
                for t in item.get("triple", []):
                    v = t.get("_validity", "reject")
                    if v == "strict":
                        strict += 1
                    elif isinstance(v, str) and v.startswith("lenient:"):
                        lenient += 1
                    else:
                        reject += 1

            metrics = {
                "file_name": os.path.basename(self.file_path),
                 "model_used": self.model_name,
                 "num_pages": self.pages_parsed,
                 "num_sentences_total": self.sentences_total,
                 "num_sentences_used": self.sentences_used,
                 "num_raw_triples": self.raw_triples,
                 "num_valid_triples": len(self.valid_triples),
                 "num_suspicious_triples": self.suspicious_triples,
                 "avg_triples_per_sentence": self.raw_triples / self.sentences_used if self.sentences_used else 0,
                 "rejection_stats": self.rejection_stats,
                 "runtime_seconds": self.runtime_seconds,
                 "num_valid_triples_strict": strict,
                 "num_valid_triples_lenient": lenient,
                 "num_rejected_triples": reject,
            }

            metadata = {"metrics": metrics, "data": self.chunk_data}

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"File saved successfully to: {output_path}")
        except Exception as e:
            print(f"Failed to save JSON: {e}")

    # -------- single source of truth: ontology validator --------
    def _is_valid_triple(self, triple):
    # no longer used; kept for compatibility
        return True


if __name__ == "__main__":
    models = ["gemma2:9b"]
    for model in models:
        extractor = CyberTripleExtractor("cti-analysis/AnalysisOfCyberattackOnUS.pdf", model, "http://localhost:11434")
        raw_chunk_results = extractor.run()
        extractor.build_dict(raw_chunk_results)
        out_name = extractor.safe_filename(f"chunk_data_{model}.json")
        extractor.save_to_json(out_name)
