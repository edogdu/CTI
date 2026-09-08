from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from cti_analysis.models.documents import (
    RawDocument,
    NormalizedDocument,
)


def run_normalization(cfg, raw_docs: List[RawDocument] | Iterable[Path] | Path) -> List[NormalizedDocument]:
    """
    Normalize input documents into a common in-memory representation.
    Accepts RawDocument list (preferred) or file paths for backward compatibility.
    """
    dataset = getattr(cfg, "dataset_name", "").lower()

    # Path/iterable path support (legacy)
    if isinstance(raw_docs, (str, Path)) or (
        not raw_docs or isinstance(next(iter(raw_docs)), Path)  # type: ignore
    ):
        return _normalize_files(raw_docs)  # type: ignore

    # Structured IR input
    if dataset == "dnrti":
        normalized: List[NormalizedDocument] = []
        for doc in raw_docs:  # type: ignore
            meta = dict(doc.meta or {})
            # If relations/entities present, build prealigned_triples
            entities = meta.get("entities") or []
            relations = meta.get("relations") or []
            prealigned = meta.get("prealigned_triples") or []
            if not prealigned and entities and relations:
                ents = entities
                for rel in relations:
                    if not (isinstance(rel, list) and len(rel) >= 3):
                        continue
                    pred, head_idx, tail_idx = rel[0], rel[1], rel[2]
                    subj = ents[tail_idx] if 0 <= tail_idx < len(ents) else None
                    obj = ents[head_idx] if 0 <= head_idx < len(ents) else None
                    if not subj or not obj:
                        continue
                    prealigned.append(
                        {
                            "subject": {"name": subj[2], "type": subj[3]},
                            "predicate": pred,
                            "object": {"name": obj[2], "type": obj[3]},
                        }
                    )
            meta["prealigned_triples"] = prealigned
            normalized.append(
                NormalizedDocument(doc_id=doc.doc_id, text=doc.text.strip(), meta=meta)
            )
        return normalized

    return [
        NormalizedDocument(doc_id=doc.doc_id, text=doc.text.strip(), meta=doc.meta)
        for doc in raw_docs  # type: ignore
    ]


def _normalize_files(raw_docs: Iterable[Path] | Path) -> List[NormalizedDocument]:
    # legacy path-based normalization
    docs_iter = [raw_docs] if isinstance(raw_docs, (str, Path)) else raw_docs
    normalized: List[NormalizedDocument] = []
    for doc_path in docs_iter:
        doc_path = Path(doc_path)
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
        normalized.append(
            NormalizedDocument(
                doc_id=doc_path.stem,
                text=text.strip(),
                meta={"source": "raw_file", "path": str(doc_path)},
            )
        )
    return normalized
