"""Conservative held-part sweeps derived from T29 frames and printed CAD STLs."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path

from assembly.cell_runtime_safety import (
    ProductionSafetyConfigurationError,
    ProductionSafetyUnavailable,
)
from assembly.motion import Waypoint
from assembly.safety import AABB, MotionOwner


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ProductionSafetyConfigurationError(f"{label} must be a finite number")
    return float(value)


def _vector(value: object, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ProductionSafetyConfigurationError(f"{label} must contain {size} numbers")
    return tuple(_number(item, label) for item in value)


def _matrix(value: object, label: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != 4:
        raise ProductionSafetyConfigurationError(f"{label} must be a 4x4 transform")
    rows = tuple(_vector(row, 4, label) for row in value)
    if rows[3] != (0.0, 0.0, 0.0, 1.0):
        raise ProductionSafetyConfigurationError(f"{label} must be an affine transform")
    return rows


def _binary_vertices(data: bytes) -> tuple[tuple[float, float, float], ...] | None:
    if len(data) < 84:
        return None
    count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + count * 50:
        return None
    vertices: list[tuple[float, float, float]] = []
    for index in range(count):
        values = struct.unpack_from("<12fH", data, 84 + index * 50)
        vertices.extend((tuple(values[3:6]), tuple(values[6:9]), tuple(values[9:12])))
    return tuple(vertices)


def _ascii_vertices(data: bytes) -> tuple[tuple[float, float, float], ...]:
    try:
        lines = data.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ProductionSafetyConfigurationError("STL is neither valid binary nor ASCII") from error
    vertices = []
    for line in lines:
        fields = line.split()
        if fields[:1] == ["vertex"] and len(fields) == 4:
            try:
                vertex = tuple(float(item) for item in fields[1:])
            except ValueError as error:
                raise ProductionSafetyConfigurationError("STL vertex must be numeric") from error
            vertices.append(tuple(_number(item, "STL vertex") for item in vertex))
    return tuple(vertices)


def _vertices(path: Path) -> tuple[tuple[float, float, float], ...]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ProductionSafetyConfigurationError(f"cannot read held-part STL {path}") from error
    vertices = _binary_vertices(data) or _ascii_vertices(data)
    if len(vertices) < 3:
        raise ProductionSafetyConfigurationError(f"held-part STL {path} has no triangles")
    return vertices


def _transform(point: Sequence[float], matrix: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    homogeneous = (*point, 1.0)
    return tuple(sum(row[index] * homogeneous[index] for index in range(4)) for row in matrix[:3])


def _radius(entry: object, mesh: Path, object_id: str) -> float:
    if not isinstance(entry, dict):
        raise ProductionSafetyConfigurationError(f"T29 entry for {object_id} must be an object")
    transform = _matrix(entry.get("T_local_from_stl"), f"{object_id}.T_local_from_stl")
    grips = entry.get("grips")
    if not isinstance(grips, list) or len(grips) != 1 or not isinstance(grips[0], dict):
        raise ProductionSafetyConfigurationError(f"T29 entry for {object_id} must contain exactly one grip")
    tcp = _vector(grips[0].get("tcp_local_mm"), 3, f"{object_id}.tcp_local_mm")
    radius = max(math.dist(_transform(vertex, transform), tcp) for vertex in _vertices(mesh))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ProductionSafetyConfigurationError(f"held-part radius for {object_id} is invalid")
    return radius


def _load_frames(path: Path) -> tuple[dict[str, object], bytes, str]:
    try:
        data = path.read_bytes()
        raw = json.loads(data)
    except (OSError, json.JSONDecodeError) as error:
        raise ProductionSafetyConfigurationError(f"cannot read T29 grasp frames {path}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("source"), str) or not raw["source"].strip():
        raise ProductionSafetyConfigurationError("T29 grasp frames must record their source")
    if raw.get("units") != "mm, deg" or not isinstance(raw.get("objects"), dict):
        raise ProductionSafetyConfigurationError("T29 grasp frames must use mm, deg and contain objects")
    return raw["objects"], data, raw["source"].strip()


class HeldPartSTLSweepSource:
    """Bound any held-part rotation by its T29 TCP-centred CAD sphere."""

    def __init__(self, grasp_frames: Path, stl_by_object: Mapping[str, Path]) -> None:
        frame_path = grasp_frames.resolve()
        objects, frame_data, provenance = _load_frames(frame_path)
        if not stl_by_object:
            raise ProductionSafetyConfigurationError("at least one held-part STL is required")
        self._radii: dict[str, float] = {}
        source_hash = hashlib.sha256(frame_data)
        for object_id, path in sorted(stl_by_object.items()):
            mesh = path.resolve()
            if object_id not in objects:
                raise ProductionSafetyConfigurationError(f"T29 grasp frames do not contain {object_id}")
            self._radii[object_id] = _radius(objects[object_id], mesh, object_id)
            source_hash.update(object_id.encode())
            source_hash.update(mesh.read_bytes())
        self.source = (
            f"T29={frame_path}#sha256={_digest(frame_data)} ({provenance}); "
            f"held-STLs#sha256={source_hash.hexdigest()}"
        )

    def swept_bounds(self, start: Waypoint, end: Waypoint, owner: MotionOwner) -> AABB:
        if not isinstance(owner, MotionOwner) or not owner.attached:
            raise ProductionSafetyUnavailable("held-part sweep requires an attached motion owner")
        if not isinstance(start, Waypoint) or not isinstance(end, Waypoint):
            raise ProductionSafetyUnavailable("held-part sweep requires waypoint endpoints")
        try:
            radius = self._radii[owner.object_id]
        except KeyError as error:
            raise ProductionSafetyUnavailable(f"unknown held part {owner.object_id!r}") from error
        minimum = tuple(min(left, right) - radius for left, right in zip(start.xyz, end.xyz, strict=True))
        maximum = tuple(max(left, right) + radius for left, right in zip(start.xyz, end.xyz, strict=True))
        return AABB(f"held:{owner.object_id}", minimum, maximum)
