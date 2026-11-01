"""4x4 homogeneous transforms in mm and degrees, in the UF0 frame.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import math

import numpy as np


def translation(x: float, y: float, z: float) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, 3] = (x, y, z)
    return tf


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    tf = np.eye(4)
    tf[:2, :2] = ((c, -s), (s, c))
    return tf


def rot_axis(axis: tuple[float, float, float], deg: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis through the origin."""
    a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = math.radians(deg)
    tf = np.eye(4)
    tf[:3, :3] = np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)
    return tf


def about_point(tf: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply `tf` as if `point` were the origin."""
    return translation(*point) @ tf @ translation(*(-np.asarray(point)))


def world_aabb(tf: np.ndarray, bounds: list[list[float]]) -> np.ndarray:
    """Axis-aligned box (2x3) of a local box `bounds` (2x3) after `tf`."""
    lo, hi = np.asarray(bounds)
    corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pts = (tf @ corners.T).T[:, :3]
    return np.stack([pts.min(axis=0), pts.max(axis=0)])


def footprint(aabb: np.ndarray) -> tuple[np.ndarray, float]:
    """Conservative table footprint: circle around the xy box."""
    centre = (aabb[0, :2] + aabb[1, :2]) / 2
    return centre, float(np.linalg.norm(aabb[1, :2] - aabb[0, :2]) / 2)


def yaw_deg(tf: np.ndarray) -> float:
    """Heading of the local x axis projected on the table."""
    return math.degrees(math.atan2(tf[1, 0], tf[0, 0]))


def rest_on_table_dz(verts_mm: np.ndarray, tf: np.ndarray, table_z: float) -> float:
    """Return the vertical shift that puts the lowest vertex of the posed mesh on the table."""
    z = verts_mm @ tf[2, :3] + tf[2, 3]
    return float(table_z - z.min())
