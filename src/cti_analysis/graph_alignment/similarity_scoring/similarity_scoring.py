"""
Backward-compatible shim to preserve legacy imports.
"""
from .scorer import run_similarity, run_from_manifest, run_scoring

__all__ = ["run_similarity", "run_from_manifest", "run_scoring"]

