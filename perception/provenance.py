"""Run provenance and checkpoint-resume compatibility guards."""

from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
from pathlib import Path
from typing import Any

COMPATIBILITY_KEYS = (
    "manifest_sha256",
    "taxonomy_sha256",
    "split_seed",
    "labels",
    "pretrained_model",
    "pretrained_revision",
)


class ProvenanceMismatchError(RuntimeError):
    """Raised when a checkpoint belongs to different data or labels."""


def provenance_from_manifest(
    metadata: dict[str, Any],
    task: str,
    pretrained_model: str,
    pretrained_revision: str,
) -> dict[str, Any]:
    """Reduce frozen metadata to the identity required for safe resume."""
    if task not in {"detector", "classifier"}:
        raise ValueError(f"unknown perception task: {task}")
    try:
        return {
            "manifest_sha256": metadata[task]["sha256"],
            "taxonomy_sha256": metadata["taxonomy_sha256"],
            "split_seed": metadata["split_seed"],
            "labels": metadata["labels"][task],
            "pretrained_model": pretrained_model,
            "pretrained_revision": pretrained_revision,
            "code_commit": metadata["git_commit"],
        }
    except KeyError as exc:
        raise ValueError(f"manifest metadata is missing {exc.args[0]!r}") from exc


def assert_resume_compatible(
    current: dict[str, Any],
    checkpoint: dict[str, Any],
) -> None:
    """Refuse a checkpoint when any training identity field changed."""
    mismatches = []
    for key in COMPATIBILITY_KEYS:
        if key not in current or key not in checkpoint:
            mismatches.append(f"{key}: missing")
        elif current[key] != checkpoint[key]:
            mismatches.append(f"{key}: current={current[key]!r}, checkpoint={checkpoint[key]!r}")
    if mismatches:
        raise ProvenanceMismatchError("incompatible checkpoint provenance: " + "; ".join(mismatches))


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    """Record resolved packages without treating missing optional tools as versions."""
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_provenance(path: Path, provenance: dict[str, Any]) -> None:
    """Atomically persist JSON provenance next to a run or checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_provenance(path: Path) -> dict[str, Any]:
    """Read provenance as a JSON object."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"provenance must be a JSON object: {path}")
    return value
