"""Run provenance and checkpoint-resume compatibility guards."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
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
    "manifest_code_commit",
    "derived_data_sha256",
    "trainer_code_commit",
    "packages",
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
            "manifest_code_commit": metadata["git_commit"],
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
    config_error = _training_config_error(current.get("training_config"), checkpoint.get("training_config"))
    if config_error:
        mismatches.append(config_error)
    if mismatches:
        raise ProvenanceMismatchError("incompatible checkpoint provenance: " + "; ".join(mismatches))


def _training_config_error(current: Any, checkpoint: Any) -> str | None:
    if not isinstance(current, dict) or not isinstance(checkpoint, dict):
        return "training_config: missing"
    current_values, checkpoint_values = dict(current), dict(checkpoint)
    current_steps = current_values.pop("max_steps", None)
    checkpoint_steps = checkpoint_values.pop("max_steps", None)
    if current_values != checkpoint_values:
        return "training_config: fields changed other than max_steps"
    if current_steps == checkpoint_steps:
        return None
    bounded_increase = (
        type(current_steps) is int
        and type(checkpoint_steps) is int
        and 1 <= checkpoint_steps <= current_steps <= 5
    )
    return None if bounded_increase else "training_config.max_steps: not a bounded smoke increase"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_data_identity(paths: dict[str, Path]) -> dict[str, Any]:
    """Hash the exact derived metadata files consumed by a trainer."""
    hashes = {name: _sha256(path) for name, path in sorted(paths.items())}
    combined = hashlib.sha256()
    for name, value in hashes.items():
        combined.update(f"{name}\0{value}\n".encode())
    return {"derived_data_sha256": combined.hexdigest(), "derived_files": hashes}


def current_git_commit() -> str:
    """Return the exact trainer source revision, refusing an unknown identity."""
    root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot determine trainer git commit") from exc
    return result.stdout.strip()


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
