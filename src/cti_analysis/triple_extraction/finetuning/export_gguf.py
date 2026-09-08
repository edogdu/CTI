"""Export fine-tuned LoRA adapter to GGUF format for use with llama.cpp.

Primary method: convert PEFT adapter to GGUF LoRA format using llama.cpp's
convert_lora_to_gguf.py. The resulting adapter is loaded by llama-server
via --lora flag at inference time.

No HuggingFace downloads required.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from .config import FinetuningConfig

logger = logging.getLogger(__name__)

# Architecture configs for GGUF conversion without HF download.
# Add new model configs here as needed.
ARCH_CONFIGS = {
    "gemma-3-4b": {
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "head_dim": 256,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 10240,
        "max_position_embeddings": 8192,
        "num_attention_heads": 10,
        "num_hidden_layers": 34,
        "num_key_value_heads": 2,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "torch_dtype": "bfloat16",
        "vocab_size": 262144,
    },
    "gemma-2-9b": {
        "architectures": ["Gemma2ForCausalLM"],
        "model_type": "gemma2",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_logit_softcapping": 50.0,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "final_logit_softcapping": 30.0,
        "head_dim": 256,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 3584,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 8192,
        "num_attention_heads": 16,
        "num_hidden_layers": 42,
        "num_key_value_heads": 8,
        "pad_token_id": 0,
        "query_pre_attn_scalar": 256,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "sliding_window": 4096,
        "torch_dtype": "bfloat16",
        "vocab_size": 256000,
    },
}


def _resolve_arch_config(base_model: str) -> dict:
    """Find the architecture config matching the base model name."""
    for key, config in ARCH_CONFIGS.items():
        if key in base_model:
            return config
    raise ValueError(
        f"No local architecture config for {base_model}. "
        f"Known models: {list(ARCH_CONFIGS.keys())}. "
        "Add a config dict to export_gguf.py."
    )


def _write_base_config(cfg: FinetuningConfig, config_dir: Path) -> Path:
    """Write a local architecture config for the base model (no HF download)."""
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_data = _resolve_arch_config(cfg.base_model)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    return config_dir


def export_to_gguf(
    cfg: FinetuningConfig,
    adapter_dir: Optional[Path] = None,
    task: str = "joint",
) -> Path:
    """Convert PEFT LoRA adapter to GGUF format.

    Uses llama.cpp's convert_lora_to_gguf.py with a local architecture config.
    The resulting GGUF LoRA adapter can be loaded by llama-server via --lora.

    Args:
        cfg: Fine-tuning configuration.
        adapter_dir: Path to PEFT adapter directory. Defaults to cfg.output_dir/adapter.
        task: Task name for output file naming ("ner", "re", or "joint").

    Returns:
        Path to the GGUF adapter file.
    """
    adapter_path = adapter_dir or Path(cfg.output_dir) / "adapter"
    gguf_dir = Path(cfg.output_dir) / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    # Task-specific naming: adapter-ner-lora.gguf, adapter-re-lora.gguf, adapter-lora.gguf
    if task in ("ner", "re"):
        gguf_filename = f"adapter-{task}-lora.gguf"
    else:
        gguf_filename = "adapter-lora.gguf"
    gguf_path = gguf_dir / gguf_filename

    # Write local base model config (avoids HuggingFace download)
    base_config_dir = Path(cfg.output_dir) / "base_config"
    _write_base_config(cfg, base_config_dir)

    # Find convert_lora_to_gguf.py
    project_root = Path(__file__).resolve().parents[4]  # src/cti.../finetuning -> project root
    convert_script = project_root / "tools" / "llama.cpp" / "convert_lora_to_gguf.py"

    if not convert_script.exists():
        raise FileNotFoundError(
            f"llama.cpp conversion script not found at {convert_script}. "
            "Clone llama.cpp into tools/: git clone --depth 1 https://github.com/ggml-org/llama.cpp.git tools/llama.cpp"
        )

    logger.info("Converting PEFT adapter to GGUF LoRA format")
    logger.info("  Adapter: %s", adapter_path)
    logger.info("  Task: %s", task)
    logger.info("  Output: %s", gguf_path)

    result = subprocess.run(
        [
            "python", str(convert_script),
            str(adapter_path),
            "--outfile", str(gguf_path),
            "--outtype", "f16",
            "--base", str(base_config_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        logger.error("LoRA GGUF conversion failed:\n%s", result.stderr)
        raise RuntimeError(f"convert_lora_to_gguf.py failed: {result.stderr}")

    logger.info("GGUF LoRA adapter exported to %s", gguf_path)
    logger.info("Use with llama-server: --lora %s", gguf_path)
    return gguf_path
