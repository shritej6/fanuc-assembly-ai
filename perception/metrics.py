"""Measured classifier metrics, including explicit look-alike confusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class MetricInputError(ValueError):
    """Raised when predictions cannot be measured against the label map."""


def _validate(y_true: Sequence[int], y_pred: Sequence[int], labels: tuple[str, ...]) -> None:
    if len(y_true) != len(y_pred):
        raise MetricInputError("y_true and y_pred must have the same length")
    if not y_true:
        raise MetricInputError("metrics require at least one sample")
    if len(labels) != len(set(labels)):
        raise MetricInputError("labels must be unique")
    if any(not isinstance(value, int) or not 0 <= value < len(labels) for value in (*y_true, *y_pred)):
        raise MetricInputError("class index outside the label map")


def _confusion(y_true: Sequence[int], y_pred: Sequence[int], size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        matrix[truth, prediction] += 1
    return matrix


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _per_class(matrix: np.ndarray, labels: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    output = {}
    for index, label in enumerate(labels):
        true_positive = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        f1_denominator = 2 * true_positive + (predicted - true_positive) + (support - true_positive)
        output[label] = {
            "support": support,
            "predicted": predicted,
            "precision": _ratio(true_positive, predicted),
            "recall": _ratio(true_positive, support),
            "f1": _ratio(2 * true_positive, f1_denominator),
        }
    return output


def _lookalike_report(
    matrix: np.ndarray,
    labels: tuple[str, ...],
    groups: tuple[tuple[str, ...], ...],
) -> dict[str, dict[str, Any]]:
    label2id = {label: index for index, label in enumerate(labels)}
    output = {}
    for group in groups:
        if any(label not in label2id for label in group):
            raise MetricInputError(f"unknown look-alike label in {group}")
        indices = {label2id[label] for label in group}
        samples = sum(int(matrix[index].sum()) for index in indices)
        correct = sum(int(matrix[index, index]) for index in indices)
        within = sum(int(matrix[truth, pred]) for truth in indices for pred in indices if truth != pred)
        outside = sum(int(matrix[truth, pred]) for truth in indices for pred in range(len(labels)) if pred not in indices)
        output["|".join(group)] = {
            "labels": list(group),
            "samples": samples,
            "correct": correct,
            "confused_within_group": within,
            "predicted_outside_group": outside,
            "accuracy": _ratio(correct, samples),
        }
    return output


def _scenario_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    scenarios: Sequence[Mapping[str, Any]],
    keys: tuple[str, ...],
) -> dict[str, dict[str, dict[str, int | float | None]]]:
    output = {}
    for key in keys:
        values: dict[str, list[int]] = {}
        for index, scenario in enumerate(scenarios):
            values.setdefault(str(scenario.get(key)), []).append(index)
        output[key] = {}
        for value, indices in sorted(values.items()):
            correct = sum(y_true[index] == y_pred[index] for index in indices)
            output[key][value] = {
                "samples": len(indices),
                "correct": correct,
                "accuracy": _ratio(correct, len(indices)),
            }
    return output


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: tuple[str, ...],
    lookalike_groups: tuple[tuple[str, ...], ...],
    scenarios: Sequence[Mapping[str, Any]] | None = None,
    scenario_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return JSON-safe measured metrics without masking undefined values."""
    _validate(y_true, y_pred, labels)
    if scenarios is not None and len(scenarios) != len(y_true):
        raise MetricInputError("scenarios and predictions must have the same length")
    matrix = _confusion(y_true, y_pred, len(labels))
    per_class = _per_class(matrix, labels)
    f1_values = [item["f1"] for item in per_class.values() if item["f1"] is not None]
    correct = int(np.trace(matrix))
    return {
        "samples": len(y_true),
        "correct": correct,
        "accuracy": correct / len(y_true),
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "labels": list(labels),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "lookalike_groups": _lookalike_report(matrix, labels, lookalike_groups),
        "scenarios": _scenario_report(y_true, y_pred, scenarios, scenario_keys) if scenarios is not None else {},
    }
