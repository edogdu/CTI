# extractors/registry.py
from typing import Dict, Callable
from core.pipeline import Extractor
from core.config import Config
from extractors.multi_agent_extractor import MultiAgentExtractor
from adapters.semantic_extraction_adapter import SemanticExtractorAdapter

REGISTRY: Dict[str, Callable[[Config], Extractor]] = {
    "multi-agent": lambda cfg: MultiAgentExtractor(cfg),
    "semantic":    lambda cfg: SemanticExtractorAdapter(cfg),
}

def build_extractor(name: str, cfg: Config) -> Extractor:
    key = name.strip().lower()
    if key not in REGISTRY:
        raise ValueError(f"Unknown extractor '{name}'. Options: {', '.join(REGISTRY)}")
    return REGISTRY[key](cfg)
