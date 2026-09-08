"""
Backward-compatible shim for legacy imports.
"""
from cti_analysis.graph_alignment.graph_insertion.inserter import (
    store_in_neo4j,
    embed_cti_entities_from_chunk,
)

__all__ = ["store_in_neo4j", "embed_cti_entities_from_chunk"]

