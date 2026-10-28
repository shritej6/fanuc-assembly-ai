"""Dependency-light value types and split policy for frozen manifests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestConfig:
    data_root: Path
    output_dir: Path
    taxonomy_path: Path
    detector_sets: tuple[str, ...] = ("A", "F", "G")
    classifier_sets: tuple[str, ...] = ("B", "H", "K")
    split_seed: int = 1337
    require_cosmos: bool = True
    include_cosmos: bool = True
    max_scenes_per_set: int | None = None
    min_cosmos_fraction: float = 0.10
    max_cosmos_fraction: float = 0.20


@dataclass(frozen=True)
class ManifestSummary:
    detector_path: Path
    classifier_path: Path
    metadata_path: Path
    detector_sha256: str
    classifier_sha256: str
    detector_records: int
    classifier_records: int
    skipped_classifier_targets: dict[str, int]


def split_for_scene(set_name: str, index: int, seed: int) -> str:
    """Assign an entire scene to an 80/10/10 split deterministically."""
    token = f"{seed}:{set_name}:{index}".encode()
    bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % 10_000
    if bucket < 8_000:
        return "train"
    if bucket < 9_000:
        return "validation"
    return "test"
