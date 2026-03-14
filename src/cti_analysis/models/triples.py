from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class Entity:
    name: str
    type: str

    def __hash__(self):
        return hash((self.name.lower(), self.type))

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.name.lower() == other.name.lower() and self.type == other.type


@dataclass
class Triple:
    subject: Entity
    predicate: str
    object: Entity
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TripleBatch:
    doc_id: str
    triples: List[Triple]
    meta: Dict[str, Any] = field(default_factory=dict)
    version: Optional[str] = None


def to_json(obj) -> Dict[str, Any]:
    return asdict(obj)


def triple_batch_for_doc(
    doc_id: str,
    triples: List[Triple],
    *,
    chunk_ids: Optional[List[str]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
) -> TripleBatch:
    """
    Helper to stamp doc_id (and optional chunk lineage) into batch metadata.
    """
    meta = {"doc_id": doc_id}
    if chunk_ids:
        meta["chunk_ids"] = list(chunk_ids)
    if extra_meta:
        meta.update(extra_meta)
    return TripleBatch(doc_id=doc_id, triples=triples, meta=meta, version=version)

