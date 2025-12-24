from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List


@dataclass
class NodeIR:
    id: str
    labels: List[str]
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeIR:
    id: str
    start_id: str
    end_id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphInsertBatch:
    nodes: List[NodeIR] = field(default_factory=list)
    edges: List[EdgeIR] = field(default_factory=list)


def to_json(obj) -> Dict[str, Any]:
    return asdict(obj)

