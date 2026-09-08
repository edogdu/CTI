from typing import Dict, Any, List


def summarize_metrics(stages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Coarse aggregation placeholder for pipeline stage metrics."""
    merged: Dict[str, Any] = {}
    for stage in stages:
        merged.update(stage or {})
    return merged

