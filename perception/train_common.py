"""Shared, dependency-light helpers for both trainers.
Authored by Shritej Gunda."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

INTERNAL_TRAINING_KEYS = {"max_train_samples", "max_eval_samples", "fp16_initial_scale"}


def configure_mixed_precision(
    trainer: Any,
    training: dict[str, Any],
    error_type: type[Exception],
) -> None:
    """Install the bounded FP16 scaler or verify scaler-free BF16."""
    scaler = trainer.accelerator.scaler
    if training.get("bf16") is True:
        if scaler is not None:
            raise error_type("BF16 training unexpectedly created a gradient scaler")
        return
    if training.get("fp16") is not True or scaler is None:
        raise error_type("FP16 training did not create a gradient scaler")
    import torch

    trainer.accelerator.scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=training["fp16_initial_scale"],
    )


def seed_training(seed: int) -> None:
    """Seed model initialization before Trainer receives an instantiated model."""
    if type(seed) is not int:
        raise TypeError("training seed must be an integer")
    from transformers import set_seed

    set_seed(seed)


def trainer_arguments(training: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Translate validated YAML fields into Hugging Face TrainingArguments."""
    arguments = {key: value for key, value in training.items() if key not in INTERNAL_TRAINING_KEYS}
    if "warmup_ratio" in arguments:
        if "warmup_steps" in arguments:
            raise ValueError("training config cannot define both warmup_ratio and warmup_steps")
        arguments["warmup_steps"] = arguments.pop("warmup_ratio")
    arguments.update(
        {
            "output_dir": str(output_dir.resolve()),
            "do_train": True,
            "do_eval": True,
            "report_to": [],
            "ddp_find_unused_parameters": False,
        }
    )
    return arguments


def load_manifest_metadata(path: Path) -> dict[str, Any]:
    """Read the frozen manifest metadata used to construct run provenance."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"manifest metadata must be a JSON object: {path}")
    return value
