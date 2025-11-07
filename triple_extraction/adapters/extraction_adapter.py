# cti/entity_extraction/adapters/semantic_only_adapter.py
from __future__ import annotations
from typing import List, Tuple
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "CTI/triple_extraction/extraction_semantic_only_v1")))

# Try local package import first; if not present, load from /mnt/data (your upload)
try:
    from triple_extraction.extraction_semantic_consensus import extract_triples_from_text
except Exception:
    import importlib.util
    spec = importlib.util.spec_from_file_location("extraction_semantic_only_v1")
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)                 # type: ignore
    extract_triples_from_text = getattr(mod, "extract_triples_from_text", None)

def run_semantic_extractor(text: str) -> List[Tuple[str, str, str]]:
    if extract_triples_from_text is None:
        raise RuntimeError("extraction_semantic_only_v1.extract_triples_from_text not found")
    triples = extract_triples_from_text(text)
    return [(str(s).lower().strip(), str(p).lower().strip(), str(o).lower().strip()) for s,p,o in triples]
