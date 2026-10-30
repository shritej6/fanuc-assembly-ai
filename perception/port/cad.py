"""The CAD facts the port measures against, from pinned, hash-checked snapshots (no hand-typed dimensions).

- `sources/objects.74d749a.yaml` is `sim/assembly_lab/objects/objects.yaml` at origin/feature/T12-mimic commit 74d749a
  (T12 agent 2, built from the printed STLs + cad/grasp_frames.json): installed-frame bbox, seated heights, insertion
  clearances, look-alike groups, `distinguishing_dims_mm` (with the top-surface cues) and `size_twins_within_1mm`.
- `sdg/data/object_geometry.json` @5a24e10 (the same file Codex pins as assembly/sources/object_geometry.5a24e10.json):
  the stable resting poses, used for the fallen threshold. With `T_local_from_stl` from the T12 snapshot, the
  smallest tilt between the installed pose and any other stable pose is 69.4 deg (M3_02, M1_02, M2_02); a part tilted
  more than half of its own smallest tilt cannot be resting on its installed face.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from pathlib import Path

import numpy as np
import yaml

from assembly.catalog import load_catalog
from perception.port.errors import CadSourceError

ROOT = Path(__file__).resolve().parents[2]
OBJECTS_SNAPSHOT = Path(__file__).resolve().parent / "sources" / "objects.74d749a.yaml"
OBJECTS_SHA256 = "6fe5558ccdea3e2ca6b8291b36886f98732a5e152484c5c07ff5674017579935"
STABLE_POSES = ROOT / "sdg" / "data" / "object_geometry.json"


@dataclass(frozen=True, slots=True)
class TopCue:
    """A part-local region whose top height differs between two look-alikes (objects.yaml `cues`)."""

    region_xy_mm: tuple[float, float, float, float]
    area_mm2: float
    top_z_here_mm: float
    top_z_other_mm: float


@dataclass(frozen=True, slots=True)
class ObjectCad:
    object_id: str
    bbox_min_mm: tuple[float, float, float]
    bbox_max_mm: tuple[float, float, float]
    local_from_stl: tuple[tuple[float, ...], ...]
    seated_origin_z_mm: float
    seated_top_z_mm: float
    tcp_height_mm: float | None       # local z of the fingertip TCP (grips[0].tcp_local_mm), None for bases
    insertion_onto: tuple[str, ...]
    lookalikes: tuple[str, ...]
    cues: dict[str, tuple[TopCue, ...]]
    fallen_tilt_deg: float

    @property
    def size_mm(self) -> tuple[float, float, float]:
        return tuple(hi - lo for lo, hi in zip(self.bbox_min_mm, self.bbox_max_mm, strict=True))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CadCatalog:
    objects: dict[str, ObjectCad]
    size_twins: tuple[tuple[str, str], ...]
    source: str

    def get(self, object_id: str) -> ObjectCad:
        try:
            return self.objects[object_id]
        except KeyError as error:
            raise CadSourceError(f"no pinned CAD entry for {object_id!r}") from error

    def candidates_for(self, object_id: str) -> tuple[str, ...]:
        """The object, its look-alikes and its size twins: everything a size measurement must tell it apart from."""
        twins = {b if a == object_id else a for a, b in self.size_twins if object_id in (a, b)}
        return tuple(dict.fromkeys((object_id, *self.get(object_id).lookalikes, *sorted(twins))))


def _read_snapshot(path: Path, expected_sha: str) -> dict:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CadSourceError(f"cannot read pinned CAD snapshot {path}: {error}") from error
    if sha256(raw).hexdigest() != expected_sha:
        raise CadSourceError(f"{path.name}: sha256 differs from the pinned T12 snapshot")
    return yaml.safe_load(raw)


def _cues(entry: dict) -> dict[str, tuple[TopCue, ...]]:
    vs = (entry.get("distinguishing_dims_mm") or {}).get("vs") or {}
    return {other: tuple(TopCue(tuple(c["region_xy_mm"]), float(c["area_mm2"]), float(c["top_z_here_mm"]),
                                float(c["top_z_other_mm"])) for c in data.get("cues") or ())
            for other, data in vs.items()}


def _authorized_tcp_heights() -> dict[str, float]:
    catalog = load_catalog()
    return {
        step.object_id: float(step.preferred_grasp.tcp_local_mm[2])
        for product in catalog.products
        for step in product.steps
    }


def _source_tcp_points(grips: list[dict]) -> tuple[object, ...]:
    if len(grips) != 1 or "tcp_local_mm" not in grips[0]:
        raise CadSourceError("movable object needs exactly one sourced TCP record")
    tcp = grips[0]["tcp_local_mm"]
    return tuple(tcp.values()) if isinstance(tcp, dict) else (tcp,)


def _validated_tcp_height(object_id: str, entry: dict, authorized: dict[str, float]) -> float | None:
    expected = authorized.get(object_id)
    grips = entry.get("grips") or []
    if expected is None:
        if grips:
            raise CadSourceError(f"base {object_id} unexpectedly has a grasp")
        return None
    try:
        heights = tuple(float(point[2]) for point in _source_tcp_points(grips))
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise CadSourceError(f"{object_id} has a malformed TCP record") from error
    if not heights or any(not math.isfinite(height) or abs(height - expected) > 1e-6 for height in heights):
        raise CadSourceError(f"{object_id} TCP height disagrees with the authorized T29 grasp")
    return expected


def _fallen_tilt(stable_poses: list[dict], local_from_stl: np.ndarray) -> float:
    installed_up = local_from_stl[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    tilts = [math.degrees(math.acos(float(np.clip((np.array(p["transform"])[:3, :3].T @ (0, 0, 1)) @ installed_up,
                                                  -1.0, 1.0)))) for p in stable_poses]
    other = [t for t in tilts if t > 1.0]
    if not other:
        raise CadSourceError("object has no stable pose other than its installed one")
    return min(other) / 2.0


def _object(object_id: str, entry: dict, stable: dict, authorized: dict[str, float]) -> ObjectCad:
    local_from_stl = np.array(entry["T_local_from_stl"], dtype=float)
    insertion = entry.get("insertion") or {}
    return ObjectCad(
        object_id=object_id,
        bbox_min_mm=tuple(float(v) for v in entry["bbox_mm"]["min"]),
        bbox_max_mm=tuple(float(v) for v in entry["bbox_mm"]["max"]),
        local_from_stl=tuple(tuple(float(v) for v in row) for row in local_from_stl),
        seated_origin_z_mm=float(entry["seated_origin_z_mm"]),
        seated_top_z_mm=float(entry["seated_top_z_mm"]),
        tcp_height_mm=_validated_tcp_height(object_id, entry, authorized),
        insertion_onto=tuple(insertion.get("onto") or ()),
        lookalikes=tuple(entry.get("lookalikes") or ()),
        cues=_cues(entry),
        fallen_tilt_deg=_fallen_tilt(stable[object_id]["stable_poses"], local_from_stl),
    )


def _add_twin_cues(objects: dict[str, ObjectCad], twin: dict) -> None:
    """Size twins outside a taxonomy group (M1_01/M3_01, M1_02/M2_03) carry their top cues only in
    `size_twins_within_1mm`; add them both ways unless the group table already has that pair."""
    a, b = twin["a"], twin["b"]
    cues = [TopCue(tuple(c["region_xy_mm"]), float(c["area_mm2"]), float(c["top_z_here_mm"]), float(c["top_z_other_mm"]))
            for c in twin.get("top_cue") or ()]
    if b not in objects[a].cues:
        objects[a].cues[b] = tuple(cues)
    if a not in objects[b].cues:
        objects[b].cues[a] = tuple(TopCue(c.region_xy_mm, c.area_mm2, c.top_z_other_mm, c.top_z_here_mm) for c in cues)


@cache
def load_cad(path: Path = OBJECTS_SNAPSHOT, expected_sha: str = OBJECTS_SHA256) -> CadCatalog:
    raw = _read_snapshot(path, expected_sha)
    stable = json.loads(STABLE_POSES.read_text())["objects"]
    authorized = _authorized_tcp_heights()
    try:
        objects = {oid: _object(oid, entry, stable, authorized) for oid, entry in raw["objects"].items()}
        twins = tuple((t["a"], t["b"]) for t in raw["size_twins_within_1mm"])
        for twin in raw["size_twins_within_1mm"]:
            _add_twin_cues(objects, twin)
    except (KeyError, TypeError, ValueError) as error:
        raise CadSourceError(f"{path.name}: malformed entry ({error})") from error
    return CadCatalog(objects, twins, f"{path.name} sha256 {expected_sha[:12]} + {STABLE_POSES.name}@5a24e10")
