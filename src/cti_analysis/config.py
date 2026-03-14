from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


# Resolve config files relative to the repository root (not the CWD)
_PKG_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PKG_ROOT / "config" / "config.yaml"
_DEFAULT_PIPELINE = _PKG_ROOT / "config" / "pipeline.yaml"


def _locate_config(default_path: Path, filename: str) -> Path:
    """
    Find the config file even when the process is launched outside the repo root.
    Checks the package root first, then the current working directory.
    """
    candidates = [
        default_path,
        Path.cwd() / "config" / filename,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return default_path


CONFIG_PATH = _locate_config(_DEFAULT_CONFIG, "config.yaml")
PIPELINE_PATH = _locate_config(_DEFAULT_PIPELINE, "pipeline.yaml")


@dataclass
class ExtractionConfig:
    model_name: str = "gemma2:9b"
    max_tokens: int = 1200
    temperature: float = 0.1
    ollama_base_url: str = "http://localhost:11434"
    use_consensus: bool = True
    ollama_timeout: int = 120
    use_two_pass: bool = False  # two-pass: NER then RE with entity hints


@dataclass
class RepairConfig:
    enabled: bool = True
    dedupe_threshold: float = 0.0


@dataclass
class SemanticChunkingConfig:
    enabled: bool = True


@dataclass
class CanonicalizationConfig:
    enabled: bool = True
    enable_markov: bool = False
    markov_alpha: float = 0.1
    enable_szf: bool = False
    szf_entity_threshold: float = 0.80
    szf_relation_threshold: float = 0.75
    szf_max_iters: int = 50


@dataclass
class GraphInsertionConfig:
    enabled: bool = True


@dataclass
class SimilarityScoringConfig:
    enabled: bool = True


@dataclass
class RerankingConfig:
    enabled: bool = True


@dataclass
class PipelineConfig:
    extraction: ExtractionConfig
    semantic_chunking: SemanticChunkingConfig
    repair: RepairConfig
    canonicalization: CanonicalizationConfig
    graph_insertion: GraphInsertionConfig
    similarity_scoring: SimilarityScoringConfig
    reranking: RerankingConfig
    dataset_name: str = ""
    dataset_file: Optional[str] = None
    output_dir: Path | str = "results"
    datasets_dir: Path | str = "datasets"
    dataset_mode: str = "document"  # "sentence" or "document"


def _safe_load(path: Path, *, required: bool = True) -> dict:
    """
    Read a YAML file safely. If required and missing, raise so we fail fast
    instead of running with empty configs (which silently disables datasets/modules).
    """
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Configuration file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    config_path: Path = CONFIG_PATH,
    pipeline_path: Path = PIPELINE_PATH,
) -> PipelineConfig:
    """
    Load structured pipeline config from YAML files.
    - config.yaml: global values (models, etc.)
    - pipeline.yaml: per-stage toggles (optional)
    """
    base = _safe_load(config_path, required=True)
    stages = _safe_load(pipeline_path, required=False)
    # stage toggles may live under "modules" in pipeline.yaml
    module_flags = stages.get("modules", stages) if isinstance(stages, dict) else {}

    def flag(name: str, default: bool = True) -> bool:
        val = module_flags.get(name, default)
        return bool(val) if val is not None else default

    # Extraction
    models = base.get("models", {}) if isinstance(base, dict) else {}
    extraction_cfg = ExtractionConfig(
        model_name=models.get("extraction_model", models.get("model_name", ExtractionConfig.model_name)),
        max_tokens=int(models.get("max_tokens", ExtractionConfig.max_tokens)),
        temperature=float(models.get("temperature", ExtractionConfig.temperature)),
        ollama_base_url=models.get("ollama_url", ExtractionConfig.ollama_base_url),
        use_consensus=flag("consensus", True),
        ollama_timeout=int(models.get("ollama_timeout", ExtractionConfig.ollama_timeout)),
        use_two_pass=bool(models.get("use_two_pass", ExtractionConfig.use_two_pass)),
    )

    semantic_chunking_cfg = SemanticChunkingConfig(enabled=flag("semantic_chunking", True))
    repair_cfg = RepairConfig(enabled=flag("repair", True))
    
    # Canonicalization config with optional Markov/SZF settings
    canonicalization_base = base.get("canonicalization", {}) if isinstance(base, dict) else {}
    canonicalization_cfg = CanonicalizationConfig(
        enabled=flag("canonicalization", True),
        enable_markov=bool(canonicalization_base.get("enable_markov", False)),
        markov_alpha=float(canonicalization_base.get("markov_alpha", 0.1)),
        enable_szf=bool(canonicalization_base.get("enable_szf", False)),
        szf_entity_threshold=float(canonicalization_base.get("szf_entity_threshold", 0.80)),
        szf_relation_threshold=float(canonicalization_base.get("szf_relation_threshold", 0.75)),
        szf_max_iters=int(canonicalization_base.get("szf_max_iters", 50)),
    )
    
    graph_insertion_cfg = GraphInsertionConfig(enabled=flag("graph_insertion", True))
    similarity_scoring_cfg = SimilarityScoringConfig(enabled=flag("similarity", True))
    reranking_cfg = RerankingConfig(enabled=flag("reranking", True))


    paths_cfg = base.get("paths", {}) if isinstance(base, dict) else {}
    dataset_cfg = base.get("dataset", {}) if isinstance(base, dict) else {}

    # dataset_mode from pipeline.yaml (top-level, outside modules)
    dataset_mode = stages.get("dataset_mode", "document") if isinstance(stages, dict) else "document"

    return PipelineConfig(
        extraction=extraction_cfg,
        semantic_chunking=semantic_chunking_cfg,
        repair=repair_cfg,
        canonicalization=canonicalization_cfg,
        graph_insertion=graph_insertion_cfg,
        similarity_scoring=similarity_scoring_cfg,
        reranking=reranking_cfg,
        dataset_name=str(dataset_cfg.get("name", "")),
        dataset_file=dataset_cfg.get("file"),
        output_dir=paths_cfg.get("output_dir", "results"),
        datasets_dir=paths_cfg.get("datasets_dir", "datasets"),
        dataset_mode=str(dataset_mode),
    )


__all__ = [
    "ExtractionConfig",
    "SemanticChunkingConfig",
    "RepairConfig",
    "CanonicalizationConfig",
    "GraphInsertionConfig",
    "SimilarityScoringConfig",
    "RerankingConfig",
    "PipelineConfig",
    "load_config",
]
