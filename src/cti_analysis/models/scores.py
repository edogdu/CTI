from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List


@dataclass
class SimilarityScore:
    query_id: str
    target_id: str
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResult:
    query_id: str
    ranked_targets: List[SimilarityScore]
    meta: Dict[str, Any] = field(default_factory=dict)


def to_json(obj) -> Dict[str, Any]:
    return asdict(obj)

