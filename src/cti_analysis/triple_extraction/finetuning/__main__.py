"""CLI entry point for the QLoRA fine-tuning pipeline.

Usage:
    python -m cti_analysis.triple_extraction.finetuning --stage all
    python -m cti_analysis.triple_extraction.finetuning --stage all --task ner
    python -m cti_analysis.triple_extraction.finetuning --stage all --task re
    python -m cti_analysis.triple_extraction.finetuning --stage prep
    python -m cti_analysis.triple_extraction.finetuning --stage train
    python -m cti_analysis.triple_extraction.finetuning --stage export
    python -m cti_analysis.triple_extraction.finetuning --config path/to/config.yaml
    python -m cti_analysis.triple_extraction.finetuning --base-model unsloth/gemma-2-27b-it-bnb-4bit
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_finetuning_config
from .data_prep import prepare_dataset, regenerate_from_manifest, save_dataset, export_test_set_dnrti
from .trainer import run_training
from .export_gguf import export_to_gguf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _run_prep(cfg):
    """Run data preparation stage."""
    dataset_path = cfg.data.dataset_path
    if not dataset_path or not Path(dataset_path).exists():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    train, val, test_indices, stats = prepare_dataset(
        dataset_path=dataset_path,
        data_cfg=cfg.data,
        seed=cfg.training.seed,
    )

    # Save to shared data dir — task-specific filenames
    # Data always goes in the base output dir (not task subdirs)
    task = cfg.data.task
    base_output = Path(cfg.output_dir)
    if task != "joint":
        # cfg.output_dir may be .../finetuning/ner — go up one level for data
        data_dir = base_output.parent / "data"
    else:
        data_dir = base_output / "data"
    train_filename = f"train_{task}.jsonl" if task != "joint" else "train.jsonl"
    val_filename = f"val_{task}.jsonl" if task != "joint" else "val.jsonl"
    save_dataset(train, data_dir / train_filename)
    save_dataset(val, data_dir / val_filename)

    # Export test set as DNRTI-format JSON for pipeline evaluation
    test_dnrti_path = data_dir / "test_dnrti.json"
    export_test_set_dnrti(
        dataset_path=dataset_path,
        test_indices=test_indices,
        output_path=test_dnrti_path,
    )

    # Save split manifest for reproducibility
    manifest = {
        "seed": cfg.training.seed,
        "val_ratio": cfg.data.val_ratio,
        "test_ratio": cfg.data.test_ratio,
        "min_triples_per_doc": cfg.data.min_triples_per_doc,
        "source_dataset": str(dataset_path),
        "train_size": stats["train_size"],
        "val_size": stats["val_size"],
        "test_size": stats["test_size"],
        "train_indices": stats.get("train_indices", []),
        "val_indices": stats.get("val_indices", []),
        "test_indices": stats.get("test_indices", []),
    }
    manifest_path = data_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save stats (without large index lists)
    stats_clean = {k: v for k, v in stats.items()
                   if k not in ("train_indices", "val_indices", "test_indices")}
    stats_path = data_dir / "data_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_clean, f, indent=2)

    logger.info("Data prep complete:")
    logger.info("  Train: %d conversations", stats["train_size"])
    logger.info("  Val: %d conversations", stats["val_size"])
    logger.info("  Test: %d docs (DNRTI format at %s)", stats["test_size"], test_dnrti_path)
    logger.info("  Total triples: %d", stats["total_triples"])
    logger.info("  Predicate distribution: %s", stats["predicate_counts"])

    return train, val


def _load_cached_data(cfg):
    """Load previously prepared data from JSONL cache."""
    task = cfg.data.task
    base_output = Path(cfg.output_dir)
    if task != "joint":
        data_dir = base_output.parent / "data"
    else:
        data_dir = base_output / "data"
    train_filename = f"train_{task}.jsonl" if task != "joint" else "train.jsonl"
    val_filename = f"val_{task}.jsonl" if task != "joint" else "val.jsonl"
    train_path = data_dir / train_filename
    val_path = data_dir / val_filename

    if not train_path.exists():
        logger.error("Cached training data not found at %s. Run --stage prep first.", train_path)
        sys.exit(1)

    def load_jsonl(path):
        convos = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    convos.append(json.loads(line))
        return convos

    train = load_jsonl(train_path)
    val = load_jsonl(val_path)
    logger.info("Loaded cached data: %d train, %d val", len(train), len(val))
    return train, val


def _run_train(cfg, train=None, val=None):
    """Run training stage."""
    if train is None or val is None:
        train, val = _load_cached_data(cfg)

    adapter_dir = run_training(cfg, train, val)
    logger.info("Training complete. Adapter at %s", adapter_dir)
    return adapter_dir


def _run_export(cfg, adapter_dir=None):
    """Run export stage."""
    task = cfg.data.task
    gguf_path = export_to_gguf(cfg, adapter_dir, task=task)
    logger.info("GGUF LoRA adapter exported to %s", gguf_path)

    if gguf_path.exists():
        logger.info(
            "Use with llama-server: --lora %s", gguf_path,
        )
    else:
        logger.warning(
            "GGUF file not yet created (manual conversion needed). "
            "See export log for instructions."
        )

    return gguf_path


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning pipeline for CTI triple extraction",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "prep", "train", "export"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to finetuning config YAML",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset path",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model name",
    )
    parser.add_argument(
        "--task",
        choices=["ner", "re", "joint"],
        default=None,
        help="Task to train: ner, re, or joint (default: from config)",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_finetuning_config(args.config)

    # Apply CLI overrides
    if args.dataset:
        cfg.data.dataset_path = args.dataset
    if args.base_model:
        cfg.base_model = args.base_model
    if args.task:
        cfg.data.task = args.task

    # For task-specific runs, use a subdirectory for adapter output
    task = cfg.data.task
    if task != "joint":
        base_output = Path(cfg.output_dir)
        cfg.output_dir = str(base_output / task)

    logger.info("Fine-tuning config:")
    logger.info("  Base model: %s", cfg.base_model)
    logger.info("  Task: %s", task)
    logger.info("  Dataset: %s", cfg.data.dataset_path)
    logger.info("  Output dir: %s", cfg.output_dir)
    logger.info("  LoRA rank: %d, alpha: %d", cfg.lora.rank, cfg.lora.alpha)
    logger.info("  Epochs: %d, batch_size: %d, lr: %s", cfg.training.epochs, cfg.training.batch_size, cfg.training.learning_rate)
    logger.info("  Split: train=%.0f%% val=%.0f%% test=%.0f%%",
                (1 - cfg.data.val_ratio - cfg.data.test_ratio) * 100,
                cfg.data.val_ratio * 100, cfg.data.test_ratio * 100)

    stage = args.stage

    if stage in ("all", "prep"):
        train, val = _run_prep(cfg)
    else:
        train = val = None

    if stage in ("all", "train"):
        adapter_dir = _run_train(cfg, train, val)
    else:
        adapter_dir = None

    if stage in ("all", "export"):
        _run_export(cfg, adapter_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
