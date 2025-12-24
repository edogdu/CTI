from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

from cti_analysis.models.documents import (
    RawDocument,
    NormalizedDocument,
)


def run_normalization(cfg, raw_docs: List[RawDocument] | Iterable[Path] | Path) -> List[NormalizedDocument]:
    """
    Normalize input documents into a common in-memory representation.
    Accepts RawDocument list (preferred) or file paths for backward compatibility.
    """
    dataset = getattr(getattr(cfg, "dataset", None), "name", "").lower()

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


def _coerce_single_path(raw_input: Iterable[Path] | Path) -> Path:
    if isinstance(raw_input, (str, Path)):
        return Path(raw_input)
    raw_list = list(raw_input)
    if not raw_list:
        raise ValueError("No input provided for normalization.")
    if len(raw_list) > 1:
        raise ValueError("Expected a single input path for this dataset.")
    return Path(raw_list[0])


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


def _normalize_dnrti(json_path: Path) -> List[NormalizedDocument]:
    """
    Normalize DNRTI labeled sentences into documents with entity/relation metadata.
    Input format:
      {
        "text": "...",
        "entities": [ [start_idx, end_idx, surface, label], ...],
        "relations": [ [predicate, head_idx, tail_idx], ... ]
      }
    Relation indices refer into the entities list; we treat tail_idx as subject,
    head_idx as object (per sample semantics).
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    docs: List[NormalizedDocument] = []
    for i, entry in enumerate(data):
        text = entry.get("text", "")
        raw_entities = entry.get("entities") or []
        raw_relations = entry.get("relations") or []

        entities = [
            {
                "name": ent[2],
                "type": ent[3],
                "span": [ent[0], ent[1]],
            }
            for ent in raw_entities
            if isinstance(ent, list) and len(ent) >= 4
        ]

        relations: List[Dict[str, Any]] = []
        prealigned_triples: List[Dict[str, Any]] = []
        for rel in raw_relations:
            if not (isinstance(rel, list) and len(rel) >= 3):
                continue
            pred, head_idx, tail_idx = rel[0], rel[1], rel[2]
            subj = entities[tail_idx] if 0 <= tail_idx < len(entities) else None
            obj = entities[head_idx] if 0 <= head_idx < len(entities) else None
            if not subj or not obj:
                continue
            rel_rec = {
                "predicate": pred,
                "subject_idx": tail_idx,
                "object_idx": head_idx,
            }
            relations.append(rel_rec)
            prealigned_triples.append(
                {
                    "subject": {"name": subj["name"], "type": subj["type"]},
                    "predicate": pred,
                    "object": {"name": obj["name"], "type": obj["type"]},
                }
            )

        doc_id = f"dnrti_{i}"
        docs.append(
            NormalizedDocument(
                doc_id=doc_id,
                text=text,
                meta={
                    "source": "dnrti",
                    "entities": entities,
                    "relations": relations,
                    "prealigned_triples": prealigned_triples,
                    "index": i,
                    "path": str(json_path),
                },
            )
        )

    return docs
