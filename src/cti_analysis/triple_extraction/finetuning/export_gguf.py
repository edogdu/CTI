"""Export fine-tuned LoRA adapter to GGUF and register with Ollama.

Primary method: convert PEFT adapter to GGUF LoRA format using llama.cpp's
convert_lora_to_gguf.py, then layer it on the local Ollama base model.
No HuggingFace downloads required.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from cti_analysis.ontology import EXTRACTION_PROMPT as SYSTEM_MESSAGE

from .config import FinetuningConfig

logger = logging.getLogger(__name__)

# Gemma-2-9b-it architecture config (for LoRA GGUF conversion without HF download)
GEMMA2_9B_CONFIG = {
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
}


def _resolve_ollama_base_model(base_model: str) -> str:
    """Map training base model name to Ollama model tag."""
    if "gemma-2-9b" in base_model:
        return "gemma2:9b"
    if "gemma-2-27b" in base_model:
        return "gemma2:27b"
    # Fallback: strip prefix/suffix and guess
    name = base_model.replace("unsloth/", "").replace("-bnb-4bit", "").replace("-it", "")
    return name


def _write_base_config(cfg: FinetuningConfig, config_dir: Path) -> Path:
    """Write a local architecture config for the base model (no HF download)."""
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"

    if "gemma-2-9b" in cfg.base_model:
        config_data = GEMMA2_9B_CONFIG
    else:
        raise ValueError(
            f"No local architecture config for {cfg.base_model}. "
            "Add a config dict to export_gguf.py or provide --base to convert_lora_to_gguf.py."
        )

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    return config_dir


def export_to_gguf(
    cfg: FinetuningConfig,
    adapter_dir: Optional[Path] = None,
) -> Path:
    """Convert PEFT LoRA adapter to GGUF format and register with Ollama.

    Uses llama.cpp's convert_lora_to_gguf.py with a local architecture config.
    The resulting GGUF LoRA is layered on top of the local Ollama base model
    (no HuggingFace downloads).

    Returns path to the GGUF adapter file.
    """
    adapter_path = adapter_dir or Path(cfg.output_dir) / "adapter"
    gguf_dir = Path(cfg.output_dir) / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    gguf_path = gguf_dir / "adapter-lora.gguf"

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
    return gguf_path


def register_with_ollama(
    gguf_path: Path,
    model_name: str,
    base_model: str = "gemma2:9b",
    system_message: Optional[str] = None,
) -> bool:
    """Create an Ollama Modelfile and register the fine-tuned model.

    Uses the local Ollama base model (FROM gemma2:9b) with the GGUF LoRA
    adapter layered on top. No HuggingFace interaction.

    Returns True on success.
    """
    sys_msg = system_message or SYSTEM_MESSAGE
    ollama_base = _resolve_ollama_base_model(base_model)
    adapter_abs = str(gguf_path.resolve()).replace("\\", "/")

    modelfile_content = (
        f"FROM {ollama_base}\n"
        f"ADAPTER {adapter_abs}\n"
        f'SYSTEM """{sys_msg}"""\n'
        f"PARAMETER temperature 0.1\n"
        f"PARAMETER num_predict 2048\n"
    )

    # Write Modelfile next to the GGUF
    modelfile_path = gguf_path.parent / "Modelfile"
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    logger.info("Wrote Modelfile to %s", modelfile_path)

    # Register with Ollama
    logger.info("Registering model '%s' with Ollama (base: %s)...", model_name, ollama_base)
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            logger.info("Model '%s' registered successfully", model_name)
            return True
        else:
            logger.error("ollama create failed: %s", result.stderr)
            return False
    except FileNotFoundError:
        logger.error("ollama command not found. Register manually:\n  ollama create %s -f %s", model_name, modelfile_path)
        return False
    except subprocess.TimeoutExpired:
        logger.error("ollama create timed out")
        return False
