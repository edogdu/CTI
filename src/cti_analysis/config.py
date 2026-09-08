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
class BackendConfig:
    url: str = "http://localhost:8080"
    timeout: int = 300
    ner_lora_id: int = 0
    re_lora_id: int = 1
    base_model_path: str = ""
    ner_lora_path: str = ""
    re_lora_path: str = ""


@dataclass
class EmbeddingsConfig:
    method: str = "sentence-transformers"
    model: str = "all-MiniLM-L6-v2"
    dimensions: int = 768
    strategy: str = "entity-context"


@dataclass
class ExtractionConfig:
    max_tokens: int = 2048
    temperature: float = 0.1
    use_consensus: bool = True
    use_two_pass: bool = True


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
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "abcd90909090"


@dataclass
class GraphInsertionConfig:
    enabled: bool = True


@dataclass
class SimilarityConfig:
    top_k: int = 10
    index_name: str = "cti_entity_similarity"
    index_label: str = "SimilarityTarget"
    index_property: str = "embedding"
    similarity_function: str = "cosine"
    allowed_labels: tuple = ("UcoexMITREATTACK", "UcoexTACTICS", "UcoexSOFTWARE")


@dataclass
class SimilarityScoringConfig:
    enabled: bool = True


@dataclass
class RerankingConfig:
    enabled: bool = True


@dataclass
class PipelineConfig:
    extraction: ExtractionConfig
    backend: BackendConfig
    embeddings: EmbeddingsConfig
    semantic_chunking: SemanticChunkingConfig
    repair: RepairConfig
    canonicalization: CanonicalizationConfig
    graph_insertion: GraphInsertionConfig
    similarity_scoring: SimilarityScoringConfig
    reranking: RerankingConfig
    neo4j: Neo4jConfig = None
    similarity: SimilarityConfig = None
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

    # Backend (llama.cpp server)
    backend_raw = base.get("backend", {}) if isinstance(base, dict) else {}
    backend_cfg = BackendConfig(
        url=str(backend_raw.get("url", BackendConfig.url)),
        timeout=int(backend_raw.get("timeout", BackendConfig.timeout)),
        ner_lora_id=int(backend_raw.get("ner_lora_id", BackendConfig.ner_lora_id)),
        re_lora_id=int(backend_raw.get("re_lora_id", BackendConfig.re_lora_id)),
        base_model_path=str(backend_raw.get("base_model_path", "")),
        ner_lora_path=str(backend_raw.get("ner_lora_path", "")),
        re_lora_path=str(backend_raw.get("re_lora_path", "")),
    )

    # Embeddings
    embed_raw = base.get("embeddings", {}) if isinstance(base, dict) else {}
    embeddings_cfg = EmbeddingsConfig(
        method=str(embed_raw.get("method", EmbeddingsConfig.method)),
        model=str(embed_raw.get("model", EmbeddingsConfig.model)),
        dimensions=int(embed_raw.get("dimensions", EmbeddingsConfig.dimensions)),
        strategy=str(embed_raw.get("strategy", EmbeddingsConfig.strategy)),
    )

    # Extraction
    models = base.get("models", {}) if isinstance(base, dict) else {}
    extraction_cfg = ExtractionConfig(
        max_tokens=int(models.get("max_tokens", ExtractionConfig.max_tokens)),
        temperature=float(models.get("temperature", ExtractionConfig.temperature)),
        use_consensus=flag("consensus", True),
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

    # Neo4j
    neo4j_raw = base.get("neo4j", {}) if isinstance(base, dict) else {}
    neo4j_cfg = Neo4jConfig(
        uri=str(neo4j_raw.get("uri", Neo4jConfig.uri)),
        user=str(neo4j_raw.get("user", Neo4jConfig.user)),
        password=str(neo4j_raw.get("password", Neo4jConfig.password)),
    )

    # Similarity search config
    sim_raw = base.get("similarity", {}) if isinstance(base, dict) else {}
    similarity_cfg = SimilarityConfig(
        top_k=int(sim_raw.get("top_k", 10)),
        index_name=str(sim_raw.get("index_name", "cti_entity_similarity")),
        index_label=str(sim_raw.get("index_label", "SimilarityTarget")),
        index_property=str(sim_raw.get("index_property", "embedding")),
        similarity_function=str(sim_raw.get("similarity_function", "cosine")),
        allowed_labels=tuple(sim_raw.get("allowed_labels", ["UcoexMITREATTACK", "UcoexTACTICS", "UcoexSOFTWARE"])),
    )

    paths_cfg = base.get("paths", {}) if isinstance(base, dict) else {}
    dataset_cfg = base.get("dataset", {}) if isinstance(base, dict) else {}

    # dataset_mode from pipeline.yaml (top-level, outside modules)
    dataset_mode = stages.get("dataset_mode", "document") if isinstance(stages, dict) else "document"

    return PipelineConfig(
        extraction=extraction_cfg,
        backend=backend_cfg,
        embeddings=embeddings_cfg,
        semantic_chunking=semantic_chunking_cfg,
        repair=repair_cfg,
        canonicalization=canonicalization_cfg,
        graph_insertion=graph_insertion_cfg,
        similarity_scoring=similarity_scoring_cfg,
        reranking=reranking_cfg,
        neo4j=neo4j_cfg,
        similarity=similarity_cfg,
        dataset_name=str(dataset_cfg.get("name", "")),
        dataset_file=dataset_cfg.get("file"),
        output_dir=paths_cfg.get("output_dir", "results"),
        datasets_dir=paths_cfg.get("datasets_dir", "datasets"),
        dataset_mode=str(dataset_mode),
    )


__all__ = [
    "BackendConfig",
    "EmbeddingsConfig",
    "ExtractionConfig",
    "SemanticChunkingConfig",
    "RepairConfig",
    "CanonicalizationConfig",
    "Neo4jConfig",
    "GraphInsertionConfig",
    "SimilarityConfig",
    "SimilarityScoringConfig",
    "RerankingConfig",
    "PipelineConfig",
    "load_config",
]
