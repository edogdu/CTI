Description of all the scripts related to triple extraction

***extraction_consensus_plus_invalid.py*** -

Modification of extraction_consensus.py script that performs automated triple extraction and consensus filtering from Cyber Threat Intelligence (CTI) reports in PDF format.
  It converts documents into text using Docling, splits content by page and sentence with SpaCy, and uses multiple prompt variants with a local gemma2:9b worker to extract candidate subject–predicate–object triples.
  It still uses MALOnt ontology in this version but other ontologies can be applied with further modification.
  Extracted triples are compared across prompt outputs using a consensus filter, keeping only those that appear in at least two prompt results.
  BIGGEST CHANGE from the original extraction_consensus.py:
      Valid triples are saved to chunk_data_gemma2_9b.json, while non-consensus or schema-invalid ones (e.g., wrong type, malformed structure) are stored separately in invalid_triples_gemma2_9b.json for later repair.
      The location of all stored JSONs is CTI/scripts/triple_extraction/extracted_triples folder

***repair_invalid_triples.py*** -

Loads previously failed triples from CTI/scripts/trirple_extraction/extracted_triples/invalid_triples_gemma2_9b.json.
  applies deterministic normalization (det_fix) to clean names, types, and predicates, and validates them against a predefined STIX/UCO schema. 
  Triples that still fail are re-evaluated using a local LLM (via Ollama) in two passes: a strict repair (schema-compliant only) and, if needed, a loose repair (allowing slight paraphrasing of entity names).
  All repaired triples are saved to _repaired_valid.json, while unfixable ones go to _still_invalid.json inside the extracted_triples folder, along with top failure reasons and runtime metrics. 
  This process increases dataset quality and ensures final triples conform to STIX 2.1 domain and range rules.
