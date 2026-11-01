"""Validated nominal CAD assets for pose-estimation preflight."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


class PoseAssetError(ValueError):
    """Raised when the nominal pose-asset registry is unsafe or inconsistent."""


@dataclass(frozen=True)
class PoseAsset:
    object_id: str
    mesh_path: Path
    mesh_scale_to_m: float


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PoseAssetError(f"cannot read pose asset registry {path}: {exc}") from exc


def _taxonomy_ids(payload: Any) -> list[str]:
    try:
        object_ids = [item["object_id"] for item in payload["objects"]]
    except (KeyError, TypeError) as exc:
        raise PoseAssetError("taxonomy objects must contain object_id strings") from exc
    if not object_ids or not all(isinstance(item, str) and item for item in object_ids):
        raise PoseAssetError("taxonomy objects must contain object_id strings")
    if len(object_ids) != len(set(object_ids)):
        raise PoseAssetError("taxonomy object IDs must be unique")
    return object_ids


def _resolve_mesh(repo_root: Path, relative_value: Any) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise PoseAssetError("each object must name a non-empty STL path")
    relative = Path(relative_value)
    if relative.is_absolute() or relative.suffix.lower() != ".stl":
        raise PoseAssetError(f"mesh path must be a relative STL: {relative_value}")
    mesh_root = (repo_root / "cad/stl").resolve()
    candidate = (repo_root / relative).resolve()
    if not candidate.is_relative_to(mesh_root):
        raise PoseAssetError(f"mesh path must remain inside the repository CAD registry: {relative_value}")
    cursor = repo_root.resolve()
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise PoseAssetError(f"mesh path must not use symlinks: {relative_value}")
    if not candidate.is_file():
        raise PoseAssetError(f"mesh file does not exist: {relative_value}")
    return candidate


def load_pose_assets(repo_root: Path) -> Mapping[str, PoseAsset]:
    """Load the contract-complete nominal STL registry in taxonomy order."""
    root = Path(repo_root).resolve()
    taxonomy = _read_json(root / "contracts/taxonomy.json")
    geometry = _read_json(root / "sdg/data/object_geometry.json")
    object_ids = _taxonomy_ids(taxonomy)
    try:
        units = geometry["units"]
        geometry_objects = geometry["objects"]
    except (KeyError, TypeError) as exc:
        raise PoseAssetError("geometry registry must contain units and objects") from exc
    if units != "mm":
        raise PoseAssetError(f"unsupported geometry units: {units!r}")
    if not isinstance(geometry_objects, dict) or set(object_ids) != set(geometry_objects):
        raise PoseAssetError("taxonomy and geometry object IDs must match exactly")
    if any(not isinstance(geometry_objects[item], Mapping) for item in object_ids):
        raise PoseAssetError("each geometry entry must be an object")
    assets = {
        object_id: PoseAsset(
            object_id=object_id,
            mesh_path=_resolve_mesh(root, geometry_objects[object_id].get("stl")),
            mesh_scale_to_m=0.001,
        )
        for object_id in object_ids
    }
    return MappingProxyType(assets)
