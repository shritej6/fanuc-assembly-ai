"""Load printed STL meshes in each object's installed frame and compute their height maps.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STL_DIR = ROOT / "cad" / "stl"
GRASP_FRAMES = ROOT / "cad" / "grasp_frames.json"
GRID_MM = 0.5


@dataclass(frozen=True)
class HeightMap:
    """Top and bottom surface z per cell; cell (i, j) centre = origin + (i + 0.5, j + 0.5) * res."""

    origin_xy: tuple[float, float]
    res: float
    top: np.ndarray       # shape (nx, ny), NaN where nothing
    bottom: np.ndarray

    def cells(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centres (x, y) and top z of every occupied cell, in the object's local frame."""
        i, j = np.nonzero(~np.isnan(self.top))
        x = self.origin_xy[0] + (i + 0.5) * self.res
        y = self.origin_xy[1] + (j + 0.5) * self.res
        return x, y, self.top[i, j]


def read_stl(path: Path) -> np.ndarray:
    """Binary STL -> (n, 3, 3) triangle vertices in mm."""
    data = path.read_bytes()
    (count,) = struct.unpack("<I", data[80:84])
    if len(data) != 84 + 50 * count:
        raise ValueError(f"{path.name}: not a binary STL of {count} triangles")
    rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                        count=count, offset=84)
    return rec["v"].astype(np.float64)


def stl_path(name: str) -> Path:
    return STL_DIR / f"{name}.stl"


def local_triangles(name: str, t_local_from_stl: list[list[float]]) -> np.ndarray:
    """The printed mesh in the object's installed local frame."""
    tri = read_stl(stl_path(name))
    t = np.asarray(t_local_from_stl, float)
    return tri @ t[:3, :3].T + t[:3, 3]


def mesh_volume_com(tri: np.ndarray) -> tuple[float, np.ndarray]:
    """Volume (mm^3) and centre of mass of a closed triangle mesh (signed tetrahedra from the origin)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    volume = vol6.sum() / 6.0
    com = (vol6[:, None] * (a + b + c)).sum(axis=0) / (4.0 * vol6.sum())
    return float(volume), com


def _surface_points(tri: np.ndarray, spacing: float) -> np.ndarray:
    """Points on every triangle no further apart than `spacing` (barycentric grid per triangle)."""
    edges = np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2).max(axis=1)
    levels = np.maximum(1, np.ceil(edges / spacing)).astype(int)
    out = []
    for n in np.unique(levels):
        sel = tri[levels == n]
        u, v = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (u + v) <= n
        u, v = u[keep] / n, v[keep] / n
        w = 1.0 - u - v
        pts = (w[None, :, None] * sel[:, None, 0] + u[None, :, None] * sel[:, None, 1]
               + v[None, :, None] * sel[:, None, 2])
        out.append(pts.reshape(-1, 3))
    return np.concatenate(out)


def height_map(tri: np.ndarray, res: float = GRID_MM) -> HeightMap:
    """Top and bottom surface z per res x res cell over the mesh's xy bounding box."""
    pts = _surface_points(tri, res / 2.0)
    lo = np.floor(tri[..., :2].reshape(-1, 2).min(axis=0) / res) * res
    hi = tri[..., :2].reshape(-1, 2).max(axis=0)
    shape = tuple(math.ceil((h - lo_) / res) + 1 for h, lo_ in zip(hi, lo))
    idx = np.floor((pts[:, :2] - lo) / res).astype(int)
    top = np.full(shape, -np.inf)
    bottom = np.full(shape, np.inf)
    np.maximum.at(top, (idx[:, 0], idx[:, 1]), pts[:, 2])
    np.minimum.at(bottom, (idx[:, 0], idx[:, 1]), pts[:, 2])
    top[np.isinf(top)] = np.nan
    bottom[np.isinf(bottom)] = np.nan
    return HeightMap((float(lo[0]), float(lo[1])), res, top, bottom)


def grasp_frames() -> dict:
    return json.loads(GRASP_FRAMES.read_text())["objects"]
