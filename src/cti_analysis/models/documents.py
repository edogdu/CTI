from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import copy


@dataclass
class RawDocument:
    doc_id: str
    source_path: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedDocument:
    doc_id: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


def to_json(obj) -> Dict[str, Any]:
    """Shallow dataclass -> dict serializer for JSON emission."""
    return asdict(obj)


def chunk_from_text(
    doc: NormalizedDocument,
    chunk_id: str,
    text: str,
    meta_override: Optional[Dict[str, Any]] = None,
) -> Chunk:
    """
    Create a Chunk inheriting metadata from the parent document.
    Use meta_override to add/replace chunk-specific fields (e.g., span indices).
    """
    meta = copy.deepcopy(doc.meta)
    if meta_override:
        meta.update(meta_override)
    return Chunk(doc_id=doc.doc_id, chunk_id=chunk_id, text=text, meta=meta)

