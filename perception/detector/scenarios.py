"""Detector dataset slices for measured per-scenario evaluation.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

SCALAR_DIMENSIONS = (("set", "set"), ("camera", "camera"), ("variant", "variant"))
LIST_DIMENSIONS = (
    ("products", "product"),
    ("defect_types", "defect_type"),
    ("build_states", "build_state"),
)


def _slice(dataset: Any, predicate: Callable[[dict[str, Any]], bool]) -> Any:
    selected = copy.copy(dataset)
    selected.images = [image for image in dataset.images if predicate(image)]
    return selected


def _scalar_slices(dataset: Any, field: str, label: str) -> dict[str, Any]:
    values = sorted({image[field] for image in dataset.images if image.get(field) is not None})
    return {
        f"{label}={value}": _slice(dataset, lambda image, value=value: image.get(field) == value)
        for value in values
    }


def _list_slices(dataset: Any, field: str, label: str) -> dict[str, Any]:
    values = sorted(
        {value for image in dataset.images if isinstance(image.get(field), list) for value in image[field]}
    )
    return {
        f"{label}={value}": _slice(dataset, lambda image, value=value: value in image.get(field, []))
        for value in values
    }


def _trap_slices(dataset: Any) -> dict[str, Any]:
    output = {}
    for present, label in ((False, "absent"), (True, "present")):
        selected = _slice(dataset, lambda image, present=present: image.get("has_trap") is present)
        if len(selected):
            output[f"trap={label}"] = selected
    return output


def scenario_datasets(dataset: Any) -> dict[str, Any]:
    """Return deterministic, non-empty views while sharing annotations and categories."""
    output = {}
    for field, label in SCALAR_DIMENSIONS:
        output.update(_scalar_slices(dataset, field, label))
    for field, label in LIST_DIMENSIONS:
        output.update(_list_slices(dataset, field, label))
    output.update(_trap_slices(dataset))
    return {name: selected for name, selected in output.items() if len(selected)}
