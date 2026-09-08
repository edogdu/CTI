from __future__ import annotations
from pathlib import Path
import hashlib


def compute_document_id(pdf: Path) -> str:
    """Compute a stable content-based ID (first 16 hex of SHA-256)."""
    with open(pdf, "rb") as _f:
        _h = hashlib.sha256(_f.read()).hexdigest()
    return _h[:16]

