# Triple Extraction

## Baseline Extraction

The main extraction process lives inside [extraction.py](https://github.com/edogdu/CTI/blob/main/triple_extraction/extraction.py). This process...

### Baseline Results

Record Valid vs Invalid triples, # of extracted triples, Avg contexts per entity, etc

## Methods

### Consensus

...

### Semantic Chunking

...

### Markov Filtering & Skew Zero Forcing

##  Triple Validation and Repair

Located in the **`/triple_validation_and_repair/`** folder, this method refines CTI triple extraction results by filtering, validating, and repairing triples that fail schema or consensus checks.

---

### triple_validation_and_repair.py

Loads invalid triples from  
`triple_extraction/extracted_triples/invalid_triples_gemma2_9b.json`,  
applies deterministic normalization (`det_fix`), applies Markov Smoothing, validates against **STIX/UCO schema**, and performs a strict LLM repair pass using **`gemma2:9b`** - followed by a **loose LLM pass only if beneficial, additionally runs a SZF pass**.    
Repaired triples are saved to `invalid_triples_gemma2_9b_repaired.json`, and remaining invalid ones to `invalid_triples_gemma2_9b_still_bad.json`, improving overall dataset accuracy and schema compliance.

##  Validation & Repair Results Comparison

## Extraction & Repair Results – STIX 2.1

| Ontology | Original Invalid | Processing Pipeline                          | Time (s) | Repaired | Invalid | Notes                                                                                                                                                                       |
|----------|------------------|----------------------------------------------|----------|----------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| STIX 2.1 | 272              | Deterministic + Markov + 2 LLM passes        | 814.4    | 91       | 181     | Markov smoothing improved contextual alignment and reduced repair needs. Lower invalid rate indicates higher validity in the original extraction while maintaining STIX 2.1’s strict relationship standards. |
| STIX 2.1 | 272              | Deterministic + Markov + 1 Unified LLM pass  | 288.6    | 84       | 194     | Unified prompt and selective LLM usage improved time by more than 150%, with only a minimal reduction in repaired triples.                                                 |

---

### Summary Insights

- STIX 2.1 enforces strict entity typing and relationship standards, ensuring schema compliance but yielding higher initial invalid counts.
- Markov smoothing enhances contextual continuity and predicate coherence, reducing correction needs across the pipeline.
- A unified LLM prompt with selective invalid triple nomination allows for faster processing with minimal drawbacks in repair effectiveness.




---

### Semantic Chunking Triple Extraction

This pipeline performs semantic chunk-level CTI triple extraction using embeddings and large language models (LLMs).  
It converts CTI reports into structured knowledge graphs by following these steps:

1. **Document Conversion** → Clean text extracted via `docling.DocumentConverter`.
2. **Sentence Splitting** → Text divided into sentences using `spaCy`.
3. **Boilerplate Filtering** → Removes irrelevant lines (headers, menus, cookie text, etc.).
4. **Embedding Generation** → Each sentence converted into a vector using `Ollama`'s `nomic-embed-text` model.
5. **Semantic Chunking (Max–Min)** → Groups semantically similar sentences using cosine similarity.
6. **Chunk Coalescing** → Ensures minimum and maximum chunk lengths.
7. **LLM Extraction** → Runs Gemma 2 (9B) on each chunk to extract CTI triples.
8. **Consensus Filtering** → Keeps only triples that multiple prompt variants agree on.
9. **MALONT-lite Validation** → Confirms entities/predicates are valid CTI ontology terms.
10. **Output Saving** → JSON results and chunk previews are written to `output`.

**Output example:**
- `chunk_data_gemma2_9b.json`
- `chunks_preview.json`
- `chunks_preview.txt`


