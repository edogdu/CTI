"""Configuration for QLoRA fine-tuning pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


_PKG_ROOT = Path(__file__).resolve().parents[4]  # c:/CTI
_DEFAULT_CONFIG = _PKG_ROOT / "config" / "finetuning.yaml"


@dataclass
class LoRAConfig:
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class TrainingConfig:
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    seed: int = 42
    completion_only: bool = True


@dataclass
class DataConfig:
    dataset_path: str = ""
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    min_triples_per_doc: int = 1
    include_negatives: bool = True
    negative_ratio: float = 0.15
    task_mix_ratio: float = 0.85  # fraction NER vs RE (0.85 = 85% NER, 15% RE)


@dataclass
class ExportConfig:
    quantization: str = "q4_k_m"
    ollama_model_name: str = "gemma2-cti"


@dataclass
class FinetuningConfig:
    base_model: str = "unsloth/gemma-2-9b-it-bnb-4bit"
    output_dir: str = "results/finetuning"
    use_unsloth: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def load_finetuning_config(
    config_path: Optional[Path] = None,
) -> FinetuningConfig:
    """Load fine-tuning config from YAML, falling back to defaults."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        return FinetuningConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    lora_raw = raw.get("lora", {})
    training_raw = raw.get("training", {})
    data_raw = raw.get("data", {})
    export_raw = raw.get("export", {})

    lora_cfg = LoRAConfig(
        rank=int(lora_raw.get("rank", LoRAConfig.rank)),
        alpha=int(lora_raw.get("alpha", LoRAConfig.alpha)),
        dropout=float(lora_raw.get("dropout", LoRAConfig.dropout)),
        target_modules=lora_raw.get("target_modules", LoRAConfig().target_modules),
    )

    training_cfg = TrainingConfig(
        epochs=int(training_raw.get("epochs", TrainingConfig.epochs)),
        batch_size=int(training_raw.get("batch_size", TrainingConfig.batch_size)),
        gradient_accumulation_steps=int(training_raw.get(
            "gradient_accumulation_steps",
            TrainingConfig.gradient_accumulation_steps,
        )),
        learning_rate=float(training_raw.get("learning_rate", TrainingConfig.learning_rate)),
        max_seq_length=int(training_raw.get("max_seq_length", TrainingConfig.max_seq_length)),
        warmup_ratio=float(training_raw.get("warmup_ratio", TrainingConfig.warmup_ratio)),
        weight_decay=float(training_raw.get("weight_decay", TrainingConfig.weight_decay)),
        bf16=bool(training_raw.get("bf16", TrainingConfig.bf16)),
        gradient_checkpointing=bool(training_raw.get(
            "gradient_checkpointing", TrainingConfig.gradient_checkpointing,
        )),
        logging_steps=int(training_raw.get("logging_steps", TrainingConfig.logging_steps)),
        save_steps=int(training_raw.get("save_steps", TrainingConfig.save_steps)),
        eval_steps=int(training_raw.get("eval_steps", TrainingConfig.eval_steps)),
        seed=int(training_raw.get("seed", TrainingConfig.seed)),
        completion_only=bool(training_raw.get("completion_only", TrainingConfig.completion_only)),
    )

    data_cfg = DataConfig(
        dataset_path=str(data_raw.get("dataset_path", DataConfig.dataset_path)),
        val_ratio=float(data_raw.get("val_ratio", DataConfig.val_ratio)),
        test_ratio=float(data_raw.get("test_ratio", DataConfig.test_ratio)),
        min_triples_per_doc=int(data_raw.get("min_triples_per_doc", DataConfig.min_triples_per_doc)),
        include_negatives=bool(data_raw.get("include_negatives", DataConfig.include_negatives)),
        negative_ratio=float(data_raw.get("negative_ratio", DataConfig.negative_ratio)),
        task_mix_ratio=float(data_raw.get("task_mix_ratio", DataConfig.task_mix_ratio)),
    )

    export_cfg = ExportConfig(
        quantization=str(export_raw.get("quantization", ExportConfig.quantization)),
        ollama_model_name=str(export_raw.get("ollama_model_name", ExportConfig.ollama_model_name)),
    )

    return FinetuningConfig(
        base_model=str(raw.get("base_model", FinetuningConfig.base_model)),
        output_dir=str(raw.get("output_dir", FinetuningConfig.output_dir)),
        use_unsloth=bool(raw.get("use_unsloth", FinetuningConfig.use_unsloth)),
        lora=lora_cfg,
        training=training_cfg,
        data=data_cfg,
        export=export_cfg,
    )
