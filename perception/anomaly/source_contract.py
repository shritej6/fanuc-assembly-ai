"""Immutable source-corpus contract for T14 PatchCore preparation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

SOURCE_CHUNK_COUNTS = {"O": 48, "T": 3, "C": 16, "I": 12}
SOURCE_LABELS_PER_CHUNK = 500
REVIEWED_DEFECT_MANIFEST_SHA256 = "4918fec9d73a9a0fdbeb64bb878fe60d21e476ca4e9ce81c1a45665ca2e36659"


class DatasetBoundaryError(ValueError):
    """Raised when synthetic anomaly data cannot be proven safe to consume."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reviewed_variants(path: Path) -> frozenset[str]:
    """Load the exact reviewed CAD defect variant keys."""
    if path.is_symlink() or not path.is_file() or _sha256(path) != REVIEWED_DEFECT_MANIFEST_SHA256:
        raise DatasetBoundaryError(f"reviewed defect manifest is missing or changed: {path}")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetBoundaryError(f"reviewed defect manifest is invalid: {path}") from exc
    variants = payload.get("variants") if isinstance(payload, Mapping) else None
    if not isinstance(variants, Mapping) or not variants:
        raise DatasetBoundaryError(f"reviewed defect manifest is invalid: {path}")
    return frozenset(str(key) for key in variants)


def validate_variant(object_id: str, variant: str, reviewed: frozenset[str], label_path: Path) -> None:
    """Reject labels that do not name an exact reviewed part/defect mesh."""
    if variant != "ok" and f"{object_id}__{variant}" not in reviewed:
        raise DatasetBoundaryError(f"unreviewed defect variant in anomaly label: {label_path}")


def _expected_labels(chunk_index: int) -> set[str]:
    first = chunk_index * SOURCE_LABELS_PER_CHUNK
    return {f"{index:06d}.json" for index in range(first, first + SOURCE_LABELS_PER_CHUNK)}


def _inventory_set(set_name: str, set_path: Path, chunk_count: int) -> dict[str, int]:
    if set_path.is_symlink() or not set_path.is_dir():
        raise DatasetBoundaryError(f"required synthetic set is missing or symlinked: {set_path}")
    expected_chunks = {f"chunk_{index:04d}" for index in range(chunk_count)}
    chunks = {path.name: path for path in set_path.glob("chunk_*")}
    if set(chunks) != expected_chunks:
        raise DatasetBoundaryError(f"{set_name} source chunk inventory is incomplete or has extras")
    for index, chunk_name in enumerate(sorted(expected_chunks)):
        chunk = chunks[chunk_name]
        labels = chunk / "labels"
        invalid = chunk.is_symlink() or not chunk.is_dir() or labels.is_symlink() or not labels.is_dir()
        if invalid:
            raise DatasetBoundaryError(f"{set_name} source chunk is not a real complete directory: {chunk}")
        actual_labels = {path.name for path in labels.iterdir() if path.is_file() and not path.is_symlink()}
        if actual_labels != _expected_labels(index):
            raise DatasetBoundaryError(f"{set_name}/{chunk_name} label inventory is incomplete or has extras")
    return {"chunks": chunk_count, "labels": chunk_count * SOURCE_LABELS_PER_CHUNK}


def validate_patchcore_source_inventory(new_root: Path, old_root: Path) -> dict[str, dict[str, int]]:
    """Require the exact reviewed O/T/C/I chunk and label inventory before preparation."""
    new, old = Path(new_root), Path(old_root)
    if new.is_symlink() or old.is_symlink():
        raise DatasetBoundaryError("synthetic source roots must not be symlinks")
    roots = {"O": new, "T": new, "C": old, "I": old}
    return {
        set_name: _inventory_set(set_name, roots[set_name] / set_name, chunk_count)
        for set_name, chunk_count in SOURCE_CHUNK_COUNTS.items()
    }
