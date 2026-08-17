"""QLoRA fine-tuning for CTI triple extraction.

Primary backend: Unsloth (faster, lower VRAM).
Fallback: HuggingFace PEFT + TRL (if Unsloth is unavailable).

Supports completion-only masking: loss is computed only on the
assistant response tokens, not the prompt.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from .config import FinetuningConfig

logger = logging.getLogger(__name__)


def _make_loss_file_callback(output_dir: Path):
    """Return a TrainerCallback that writes loss entries to a clean JSONL file.

    Solves the problem where tqdm overwrites Trainer's loss table in the log,
    making loss values unreadable. Writes to training_losses.jsonl instead.
    """
    from transformers import TrainerCallback

    loss_path = output_dir / "training_losses.jsonl"

    class LossFileCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 3)}
            entry.update({k: round(v, 6) if isinstance(v, float) else v
                          for k, v in logs.items()
                          if k in ("loss", "eval_loss", "learning_rate", "grad_norm")})
            with open(loss_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            if "loss" in logs:
                logger.info("Step %d | loss=%.4f | lr=%.2e",
                            state.global_step, logs["loss"],
                            logs.get("learning_rate", 0))

    return LossFileCallback()


def _format_conversations_to_text(
    conversations: List[List[Dict[str, str]]],
    tokenizer: Any,
) -> List[str]:
    """Apply chat template to each conversation, returning full formatted strings."""
    return [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        for conv in conversations
    ]


def _train_unsloth(
    cfg: FinetuningConfig,
    train_conversations: List[List[Dict[str, str]]],
    val_conversations: List[List[Dict[str, str]]],
) -> Path:
    """Train using Unsloth backend."""
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    output_dir = Path(cfg.output_dir) / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model — use 4-bit only if model name indicates bnb-4bit
    use_4bit = "bnb-4bit" in cfg.base_model
    logger.info("Loading model %s with Unsloth (4bit=%s)", cfg.base_model, use_4bit)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.training.max_seq_length,
        load_in_4bit=use_4bit,
        dtype=None,  # auto-detect
        device_map={"": 0},  # force single GPU, avoid auto-split
    )

    # Apply LoRA (exclude vision tower to keep adapter text-only)
    exclude = getattr(cfg.lora, "exclude_modules", ["vision_tower"])
    logger.info(
        "Applying LoRA: rank=%d, alpha=%d, targets=%s, exclude=%s",
        cfg.lora.rank, cfg.lora.alpha, cfg.lora.target_modules, exclude,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        exclude_modules=exclude,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.training.seed,
    )

    # Unsloth forces vision_tower.vision_model.embeddings to require gradients
    # for VLMs (Gemma 3). We explicitly freeze it back since we only need text LoRA.
    frozen_count = 0
    for name, param in model.named_parameters():
        if "vision_tower" in name and param.requires_grad:
            param.requires_grad = False
            frozen_count += 1
    if frozen_count > 0:
        logger.info("Froze %d vision tower parameters (text-only LoRA)", frozen_count)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable parameters after vision freeze: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)

    # Format data as full text strings.
    # We use Unsloth's train_on_responses_only() for completion masking instead
    # of TRL's completion_only_loss/assistant_only_loss, because Gemma 3 is
    # detected as a VLM and Unsloth blocks those flags for VLMs.
    logger.info("Formatting %d train, %d val conversations", len(train_conversations), len(val_conversations))
    train_texts = _format_conversations_to_text(train_conversations, tokenizer)
    val_texts = _format_conversations_to_text(val_conversations, tokenizer)

    from datasets import Dataset as HFDataset
    train_dataset = HFDataset.from_dict({"text": train_texts})
    val_dataset = HFDataset.from_dict({"text": val_texts})

    # Configure trainer
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        seed=cfg.training.seed,
        max_seq_length=cfg.training.max_seq_length,
        packing=False,
        dataset_text_field="text",
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    if cfg.training.completion_only:
        # Detect chat template markers from the base model
        from unsloth.chat_templates import train_on_responses_only

        # Map known model families to their chat template markers
        base = cfg.base_model.lower()
        if "phi" in base:
            inst_part = "<|user|>"
            resp_part = "<|assistant|>"
        elif "llama" in base or "mistral" in base:
            inst_part = "[INST]"
            resp_part = "[/INST]"
        elif "qwen" in base:
            inst_part = "<|im_start|>user\n"
            resp_part = "<|im_start|>assistant\n"
        else:
            # Default: Gemma-style
            inst_part = "<start_of_turn>user\n"
            resp_part = "<start_of_turn>model\n"

        logger.info("Completion masking: instruction=%r, response=%r", inst_part, resp_part)
        trainer = train_on_responses_only(
            trainer,
            instruction_part=inst_part,
            response_part=resp_part,
        )
        logger.info("Completion masking applied via train_on_responses_only")

    trainer.add_callback(_make_loss_file_callback(output_dir))

    # Train (resume from checkpoint if available)
    checkpoint = None
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if checkpoints:
        checkpoint = str(checkpoints[-1])
        logger.info("Resuming from checkpoint: %s", checkpoint)
    else:
        logger.info("Starting training from scratch...")

    start = time.time()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    elapsed = time.time() - start
    logger.info("Training complete in %.1f minutes", elapsed / 60)

    # Save adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training stats
    stats = {
        "backend": "unsloth",
        "base_model": cfg.base_model,
        "training_time_seconds": elapsed,
        "train_loss": result.training_loss,
        "train_samples": len(train_conversations),
        "val_samples": len(val_conversations),
        "lora_rank": cfg.lora.rank,
        "lora_alpha": cfg.lora.alpha,
        "epochs": cfg.training.epochs,
        "learning_rate": cfg.training.learning_rate,
        "completion_only": cfg.training.completion_only,
    }
    stats_path = Path(cfg.output_dir) / "training_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Adapter saved to %s", output_dir)
    return output_dir


def _train_peft_trl(
    cfg: FinetuningConfig,
    train_conversations: List[List[Dict[str, str]]],
    val_conversations: List[List[Dict[str, str]]],
) -> Path:
    """Train using HuggingFace PEFT + TRL fallback."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    output_dir = Path(cfg.output_dir) / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve HuggingFace model name from Unsloth name
    hf_model_name = cfg.base_model
    if "unsloth/" in hf_model_name:
        hf_model_name = hf_model_name.replace("unsloth/", "")
        hf_model_name = hf_model_name.replace("-bnb-4bit", "")
        hf_model_name = f"google/{hf_model_name}"

    logger.info("Loading model %s with PEFT (fallback)", hf_model_name)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    lora_config = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Format data (same approach as Unsloth path)
    logger.info("Formatting %d train, %d val conversations", len(train_conversations), len(val_conversations))
    train_texts = _format_conversations_to_text(train_conversations, tokenizer)
    val_texts = _format_conversations_to_text(val_conversations, tokenizer)

    from datasets import Dataset as HFDataset
    train_dataset = HFDataset.from_dict({"text": train_texts})
    val_dataset = HFDataset.from_dict({"text": val_texts})

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        seed=cfg.training.seed,
        max_seq_length=cfg.training.max_seq_length,
        packing=False,
        dataset_text_field="text",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    if cfg.training.completion_only:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )
        logger.info("Completion masking applied via train_on_responses_only")

    trainer.add_callback(_make_loss_file_callback(output_dir))

    # Train
    logger.info("Starting training (PEFT fallback)...")
    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start
    logger.info("Training complete in %.1f minutes", elapsed / 60)

    # Save adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training stats
    stats = {
        "backend": "peft_trl",
        "base_model": hf_model_name,
        "training_time_seconds": elapsed,
        "train_loss": result.training_loss,
        "train_samples": len(train_conversations),
        "val_samples": len(val_conversations),
        "lora_rank": cfg.lora.rank,
        "lora_alpha": cfg.lora.alpha,
        "epochs": cfg.training.epochs,
        "learning_rate": cfg.training.learning_rate,
        "completion_only": cfg.training.completion_only,
    }
    stats_path = Path(cfg.output_dir) / "training_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Adapter saved to %s", output_dir)
    return output_dir


def run_training(
    cfg: FinetuningConfig,
    train_conversations: List[List[Dict[str, str]]],
    val_conversations: List[List[Dict[str, str]]],
) -> Path:
    """Run QLoRA fine-tuning. Returns path to saved adapter directory.

    Tries Unsloth first; falls back to PEFT+TRL if unavailable.
    """
    if cfg.use_unsloth:
        try:
            import unsloth  # noqa: F401
        except ImportError:
            logger.warning("Unsloth not installed, falling back to PEFT+TRL")
            return _train_peft_trl(cfg, train_conversations, val_conversations)

        logger.info("Using Unsloth backend")
        return _train_unsloth(cfg, train_conversations, val_conversations)

    return _train_peft_trl(cfg, train_conversations, val_conversations)
