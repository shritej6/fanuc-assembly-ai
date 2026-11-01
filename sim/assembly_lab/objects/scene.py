"""Scene model for the planner: object instances on the table or on a base, in UF0.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TABLE_Z_MM = -335.0          # INTERFACES.md: the table is at z ~ -335 in UF0
CELL_MM = 0.5


@dataclass(frozen=True)
class Instance:
    key: str
    oid: str
    x: float
    y: float
    z: float                 # world z of the object's local origin
    yaw: float
    state: str               # installed | fallen | on:<base key> | unknown
    base: str | None         # base key when on a base
    bbox_mm: tuple[float, float, float] | None = None   # only for UNKNOWN pieces

    def to_world(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c, s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        return self.x + c * x - s * y, self.y + s * x + c * y

    def local_to_world(self, p) -> np.ndarray:
        wx, wy = self.to_world(np.array([p[0]]), np.array([p[1]]))
        return np.array([wx[0], wy[0], self.z + p[2]])


@dataclass(frozen=True)
class Solid:
    """World-frame occupancy of one instance: cell centres and the top/bottom z of each."""

    key: str
    x: np.ndarray
    y: np.ndarray
    top: np.ndarray
    bottom: np.ndarray


def _entries(scene: dict) -> list[dict]:
    out = [dict(o) for o in scene.get("objects", [])]
    for i, s in enumerate(scene.get("spares", []), start=1):
        out.append({**s, "key": s.get("key", f"{s['id']}#spare{i}"), "state": s.get("state", "installed")})
    return out


def build_instances(scene: dict, catalog: dict) -> dict[str, Instance]:
    """Resolve every object to a world pose; bases first so parts on them can be placed."""
    table = float(scene.get("table_z_mm", TABLE_Z_MM))
    entries = _entries(scene)
    keys = [e.get("key", e["id"]) for e in entries]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate instance keys in scene: {sorted(k for k in keys if keys.count(k) > 1)}")
    out: dict[str, Instance] = {}
    pending = []
    for e in entries:
        key, state = e.get("key", e["id"]), e.get("state", "installed")
        if state.startswith("on:"):
            pending.append(e)
            continue
        if e["id"] not in catalog:
            bbox = tuple(float(v) for v in e.get("bbox_mm", (0, 0, 0)))
            out[key] = Instance(key, e["id"], float(e["x"]), float(e["y"]), table, float(e.get("yaw", 0.0)),
                                "unknown", None, bbox)
            continue
        z = table - catalog[e["id"]]["bbox_mm"]["min"][2]
        out[key] = Instance(key, e["id"], float(e["x"]), float(e["y"]), z, float(e.get("yaw", 0.0)), state, None)
    for e in pending:
        key, base_key = e.get("key", e["id"]), e["state"][3:]
        if base_key not in out:
            raise ValueError(f"{key}: base {base_key!r} is not in the scene")
        base = out[base_key]
        p = base.local_to_world(catalog[e["id"]]["installed_origin_mm"])
        out[key] = Instance(key, e["id"], float(p[0]), float(p[1]), float(p[2]), base.yaw, e["state"], base_key)
    return out


def _box_cells(inst: Instance, sx: float, sy: float) -> tuple[np.ndarray, np.ndarray]:
    gx = np.arange(-sx / 2 + CELL_MM / 2, sx / 2, CELL_MM)
    gy = np.arange(-sy / 2 + CELL_MM / 2, sy / 2, CELL_MM)
    lx, ly = np.meshgrid(gx, gy, indexing="ij")
    return inst.to_world(lx.ravel(), ly.ravel())


def solid_of(inst: Instance, catalog: dict, maps: dict) -> Solid:
    """World occupancy. A fallen part's pose is unknown, so it is a disc of radius = half its 3D diagonal and
    height = its longest side; an UNKNOWN piece is its measured box."""
    if inst.state == "unknown":
        x, y = _box_cells(inst, inst.bbox_mm[0], inst.bbox_mm[1])
        return Solid(inst.key, x, y, np.full(x.shape, inst.z + inst.bbox_mm[2]), np.full(x.shape, inst.z))
    entry = catalog[inst.oid]
    if inst.state == "fallen":
        size = np.asarray(entry["bbox_mm"]["size"], float)
        r = float(np.linalg.norm(size)) / 2
        x, y = _box_cells(inst, 2 * r, 2 * r)
        keep = (x - inst.x) ** 2 + (y - inst.y) ** 2 <= r * r
        table = inst.z + entry["bbox_mm"]["min"][2]
        x, y = x[keep], y[keep]
        return Solid(inst.key, x, y, np.full(x.shape, table + size.max()), np.full(x.shape, table))
    hm = maps[inst.oid]
    i, j = np.nonzero(~np.isnan(hm["top"]))
    lx = hm["origin"][0] + (i + 0.5) * hm["res"]
    ly = hm["origin"][1] + (j + 0.5) * hm["res"]
    x, y = inst.to_world(lx, ly)
    return Solid(inst.key, x, y, inst.z + hm["top"][i, j], inst.z + hm["bottom"][i, j])
