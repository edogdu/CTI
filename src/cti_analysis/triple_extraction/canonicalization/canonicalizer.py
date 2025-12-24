from __future__ import annotations

from typing import List, Dict, Any
from cti_analysis.models.triples import TripleBatch


def run_canonicalization(cfg, triple_batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stub canonicalization stage.

    Replace this with the real canonicalization logic as it becomes available.
    """
    return triple_batches


def run_canonicalization_ir(cfg, batches: List[TripleBatch]) -> List[TripleBatch]:
    return batches

