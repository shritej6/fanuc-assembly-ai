"""Stable label mappings derived from the shared taxonomy contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FOREIGN_LABELS = (
    "foreign_screw_m3",
    "foreign_tape_roll",
    "foreign_cube_10mm",
)


@dataclass(frozen=True)
class Ontology:
    """Immutable class order and evaluation groups for T13."""

    classifier_labels: tuple[str, ...]
    detector_labels: tuple[str, ...]
    lookalike_groups: tuple[tuple[str, ...], ...]

    @property
    def classifier_label2id(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.classifier_labels)}

    @property
    def detector_label2id(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.detector_labels)}


def _connected_groups(
    labels: tuple[str, ...],
    links: dict[str, tuple[str, ...]],
) -> tuple[tuple[str, ...], ...]:
    order = {label: index for index, label in enumerate(labels)}
    visited: set[str] = set()
    groups: list[tuple[str, ...]] = []
    for start in labels:
        if start in visited or not links[start]:
            continue
        pending = [start]
        component: set[str] = set()
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(links[current])
        visited.update(component)
        groups.append(tuple(sorted(component, key=order.__getitem__)))
    return tuple(groups)


def load_ontology(path: Path) -> Ontology:
    """Load class order and reject ambiguous look-alike graphs."""
    data = json.loads(path.read_text())
    objects = data["objects"]
    labels = tuple(item["object_id"] for item in objects)
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate object_id in taxonomy")
    links = {item["object_id"]: tuple(item.get("lookalikes", ())) for item in objects}
    known = set(labels)
    for label, others in links.items():
        for other in others:
            if other not in known:
                raise ValueError(f"unknown look-alike {other!r} for {label}")
            if label not in links[other]:
                raise ValueError(f"look-alike link {label}->{other} is not symmetric")
    return Ontology(
        classifier_labels=labels,
        detector_labels=labels + FOREIGN_LABELS,
        lookalike_groups=_connected_groups(labels, links),
    )
