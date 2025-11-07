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

##  Triple Consensus and Repair

Located in the **`/triple_consensus_and_repair/`** folder, this method refines CTI triple extraction results by filtering, validating, and repairing triples that fail schema or consensus checks.

---

### extraction_consensus_plus_invalid.py

Performs automated triple extraction and consensus filtering on CTI reports in PDF format.  
Uses **Docling** for text conversion, **SpaCy** for segmentation, and a local **`gemma2:9b`** model for multiple prompt-based extractions.  
Valid triples are saved to:  
`triple_extraction/extracted_triples/chunk_data_gemma2_9b.json`  
Invalid or non-consensus triples are stored in:  
`triple_extraction/extracted_triples/invalid_triples_gemma2_9b.json`  
for later repair.

---

### repair_invalid_triples.py

Loads invalid triples from  
`triple_extraction/extracted_triples/invalid_triples_gemma2_9b.json`,  
applies deterministic normalization (`det_fix`), validates against **STIX/UCO schema**, and performs a strict LLM repair pass using **`gemma2:9b`** - followed by a **loose LLM pass only if beneficial**.    
Repaired triples are saved to `invalid_triples_gemma2_9b_repaired_valid.json`, and remaining invalid ones to `invalid_triples_gemma2_9b_still_invalid.json`, improving overall dataset accuracy and schema compliance.

### repair_invalid_triples_w_Markov_smoothing.py

Loads invalid triples from  
`triple_extraction/extracted_triples/invalid_triples_gemma2_9b.json`,  
applies deterministic normalization (`det_fix`), applies Markov Smoothing, validates against **STIX/UCO schema**, and performs a strict LLM repair pass using **`gemma2:9b`** - followed by a **loose LLM pass only if beneficial**.    
Repaired triples are saved to `invalid_triples_gemma2_9b_repaired.json`, and remaining invalid ones to `invalid_triples_gemma2_9b_still_bad.json`, improving overall dataset accuracy and schema compliance.



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


