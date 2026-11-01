"""Derive C/I rest sidecars with an externally reviewed, hash-pinned algorithm."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

REVIEWED_TOOL_SHA256 = "b89e81c7bcc11f76ec59d88a8127d1997bf5871d849037346d3f0b7084e2d10f"
REVIEWED_GEOMETRY_SHA256 = "bf98f0473e9c3e5a5a9fbb1bae801434c5085b094b56d28eb03860386245d91c"
REVIEWED_RULE = "installed = same up axis as inside its product (yaw free); anything else = fallen"
REVIEWED_MAX_TILT_DEG = 2.0


class RestProvenanceError(ValueError):
    """Raised when installed-pose provenance cannot be reproduced safely."""


@dataclass(frozen=True)
class RestDerivationSummary:
    counts: dict[str, int]
    sidecar_sha256: dict[str, str]
    tool_sha256: str
    geometry_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_file(path: Path, expected: str, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise RestProvenanceError(f"reviewed {label} must be a real file: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise RestProvenanceError(f"reviewed {label} SHA-256 mismatch: {actual}")
    return actual


def _load_tool(path: Path, expected: str) -> tuple[ModuleType, str]:
    actual = _verified_file(path, expected, "tool")
    spec = importlib.util.spec_from_file_location(f"t14_rest_labels_{actual[:12]}", path)
    if spec is None or spec.loader is None:
        raise RestProvenanceError(f"cannot load reviewed rest tool: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "product_up_axes", None)) or not callable(getattr(module, "rest_states", None)):
        raise RestProvenanceError("reviewed rest tool lacks required callables")
    if getattr(module, "RULE", None) != REVIEWED_RULE or getattr(module, "MAX_TILT_DEG", None) != REVIEWED_MAX_TILT_DEG:
        raise RestProvenanceError("reviewed rest tool metadata disagrees with the approved rule")
    return module, actual


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RestProvenanceError(f"cannot read rest source {path}: {exc}") from exc


def _validated_states(value: Any, label_path: Path) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise RestProvenanceError(f"rest tool returned invalid states for {label_path}")
    states = dict(value)
    if any(not isinstance(uid, str) or rest not in ("installed", "fallen") for uid, rest in states.items()):
        raise RestProvenanceError(f"rest tool returned invalid states for {label_path}")
    return states


def _derive_chunk(
    module: ModuleType,
    up_axes: Mapping[str, list[float]],
    source: Path,
    destination: Path,
) -> tuple[dict[str, int], str]:
    labels = source / "labels"
    if labels.is_symlink() or not labels.is_dir():
        raise RestProvenanceError(f"rest source labels must be a real directory: {labels}")
    images: dict[str, dict[str, str]] = {}
    counts = {"installed": 0, "fallen": 0}
    rest_states: Callable[[dict, dict], dict] = module.rest_states
    for label_path in sorted(labels.glob("*.json")):
        if label_path.is_symlink():
            raise RestProvenanceError(f"rest source label must not be a symlink: {label_path}")
        payload = _json(label_path)
        spec = payload.get("spec") if isinstance(payload, Mapping) else None
        if not isinstance(spec, dict):
            raise RestProvenanceError(f"rest source lacks a spec object: {label_path}")
        states = _validated_states(rest_states(spec, dict(up_axes)), label_path)
        images[label_path.stem] = states
        for rest in states.values():
            counts[rest] += 1
    destination.parent.mkdir(parents=True)
    payload = {"rule": module.RULE, "max_tilt_deg": module.MAX_TILT_DEG, "images": images}
    destination.write_text(json.dumps(payload, separators=(",", ":")))
    return counts, _sha256(destination)


def derive_rest_sidecars(
    old_root: Path,
    rest_root: Path,
    *,
    tool_path: Path,
    geometry_path: Path,
    expected_tool_sha256: str = REVIEWED_TOOL_SHA256,
    expected_geometry_sha256: str = REVIEWED_GEOMETRY_SHA256,
) -> RestDerivationSummary:
    """Write derived C/I sidecars outside the immutable render roots."""
    tool, tool_sha256 = _load_tool(Path(tool_path), expected_tool_sha256)
    geometry = Path(geometry_path)
    geometry_sha256 = _verified_file(geometry, expected_geometry_sha256, "geometry")
    old, output = Path(old_root), Path(rest_root)
    if old.is_symlink() or output.exists():
        raise RestProvenanceError("rest source must be real and output must not exist")
    if output.resolve().is_relative_to(old.resolve()):
        raise RestProvenanceError("derived rest output must remain outside the immutable source")
    product_up_axes: Callable[[dict], dict] = tool.product_up_axes
    up_axes = product_up_axes(_json(geometry))
    counts, hashes = {"installed": 0, "fallen": 0}, {}
    for set_name in ("C", "I"):
        for chunk in sorted((old / set_name).glob("chunk_*")):
            if chunk.is_symlink() or not chunk.is_dir() or re.fullmatch(r"chunk_\d{4}", chunk.name) is None:
                raise RestProvenanceError(f"invalid rest source chunk: {chunk}")
            relative = Path(set_name) / chunk.name / "_REST.json"
            current, digest = _derive_chunk(tool, up_axes, chunk, output / relative)
            hashes[relative.as_posix()] = digest
            for key, value in current.items():
                counts[key] += value
    if not hashes:
        raise RestProvenanceError("no C/I chunks were available for rest derivation")
    return RestDerivationSummary(counts, hashes, tool_sha256, geometry_sha256)


def verify_rest_sidecars(rest_root: Path, summary: RestDerivationSummary) -> None:
    """Re-hash the complete derived sidecar set immediately before approval."""
    root = Path(rest_root)
    if root.is_symlink() or not root.is_dir():
        raise RestProvenanceError(f"derived rest root must be a real directory: {root}")
    entries = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise RestProvenanceError("derived rest output contains a symlink")
    actual = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in entries
        if path.is_file() and path.name == "_REST.json"
    }
    if actual != summary.sidecar_sha256:
        raise RestProvenanceError("derived rest sidecars changed after derivation")
