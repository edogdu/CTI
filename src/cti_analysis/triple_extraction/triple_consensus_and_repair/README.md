Description of Triple Extraction with consensus with validation and repair.




###**Old files, for reference and old Powerpoint presentation only.**





**extraction_consensus_plus_invalid.py**

Performs automated triple extraction and consensus filtering on CTI reports in PDF format.  
Uses **Docling** for text conversion, **SpaCy** for segmentation, and a local **`gemma2:9b`** model for multiple prompt-based extractions.  
Valid triples are saved to:  
`triple_extraction/extracted_triples/chunk_data_gemma2_9b.json`  
Invalid or non-consensus triples are stored in:  
`triple_extraction/extracted_triples/invalid_triples_gemma2_9b.json`  
for later repair.
