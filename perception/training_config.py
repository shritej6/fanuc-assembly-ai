"""Validated YAML configuration for T13 training jobs."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


class ConfigError(ValueError):
    """Raised when a training configuration violates a project invariant."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _validate_precision(training: dict[str, Any]) -> None:
    if training.get("fp16") is not True:
        raise ConfigError("T4 training requires FP16")
    if training.get("bf16") is not False:
        raise ConfigError("BF16 is not supported by T4")
    if training.get("tf32") is not False:
        raise ConfigError("TF32 is not supported by T4")


def _validate_detector(model: dict[str, Any], training: dict[str, Any]) -> None:
    if model.get("name") != "PekingU/rtdetr_r50vd":
        raise ConfigError("detector model must be PekingU/rtdetr_r50vd")
    if not REVISION_PATTERN.fullmatch(str(model.get("revision", ""))):
        raise ConfigError("detector model revision must be a 40-character commit")
    if model.get("image_size") != 640:
        raise ConfigError("RT-DETR image size must be 640")
    required = {
        "fp16_initial_scale": 128,
        "remove_unused_columns": False,
        "eval_do_concat_batches": False,
        "metric_for_best_model": "eval_map",
        "greater_is_better": True,
    }
    for key, expected in required.items():
        if training.get(key) != expected:
            raise ConfigError(f"detector training requires {key}={expected!r}")


def _validate_classifier(model: dict[str, Any], training: dict[str, Any]) -> None:
    if model.get("architecture") != "resnet18" or model.get("weights") != "DEFAULT":
        raise ConfigError("classifier must use ImageNet-pretrained resnet18")
    if model.get("num_labels") != 30:
        raise ConfigError("classifier must keep exactly 30 labels")
    if model.get("image_size") != 224:
        raise ConfigError("classifier image size must be 224")
    if training.get("fp16_initial_scale") != 128:
        raise ConfigError("classifier training requires fp16_initial_scale=128")


def _validate_smoke(training: dict[str, Any]) -> None:
    bounds = {
        "max_train_samples": (1, 32),
        "max_eval_samples": (1, 16),
        "max_steps": (1, 5),
    }
    for key, (minimum, maximum) in bounds.items():
        value = training.get(key)
        if not isinstance(value, int) or not minimum <= value <= maximum:
            raise ConfigError(f"smoke {key} must be in [{minimum}, {maximum}]")


def apply_smoke_step_override(
    config: dict[str, Any],
    *,
    smoke: bool,
    max_steps: int | None,
) -> dict[str, Any]:
    """Allow a bounded third-step resume probe without changing production."""
    if max_steps is None:
        return config
    if not smoke:
        raise ConfigError("--max-steps is allowed only with --smoke")
    if type(max_steps) is not int or not 1 <= max_steps <= 5:
        raise ConfigError("smoke max_steps must be in [1, 5]")
    updated = copy.deepcopy(config)
    updated["training"]["max_steps"] = max_steps
    return updated


def load_training_config(path: Path, smoke: bool = False) -> dict[str, Any]:
    """Load, overlay optional smoke limits, and enforce fixed T13 choices."""
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read training config {path}: {exc}") from exc
    payload = copy.deepcopy(_mapping(payload, "config"))
    task = payload.get("task")
    model = _mapping(payload.get("model"), "model")
    training = _mapping(payload.get("training"), "training")
    if smoke:
        training.update(_mapping(payload.get("smoke"), "smoke"))
    _validate_precision(training)
    if task == "detector":
        _validate_detector(model, training)
    elif task == "classifier":
        _validate_classifier(model, training)
    else:
        raise ConfigError(f"unknown task {task!r}")
    if smoke:
        _validate_smoke(training)
    return {"task": task, "model": model, "training": training}
