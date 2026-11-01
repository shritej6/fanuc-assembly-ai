"""Collision scene pieces: obstacle boxes, the held-part check and work-area intrusion.
Authored by Shritej Gunda."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from assembly.safety import AABB
from perception.port.cad import CadCatalog, ObjectCad
from perception.port.errors import PerceptionPortError
from perception.port.frames import InstalledPose, posed_bbox_corners
from robot.cell.calibration import geometry

MIN_HELD_POINTS = 12   # perception/cell/localize.py MIN_POINTS
SLAB_TOP_GAP_MM = 1.0  # points closer than this below the fingertips may be fingertip edges


def box_from_pose(name: str, pose: InstalledPose, cad: ObjectCad) -> AABB:
    """Return the obstacle box of a posed CAD part."""
    corners = posed_bbox_corners(pose, cad)
    return AABB(name, tuple(float(v) for v in corners.min(axis=0)), tuple(float(v) for v in corners.max(axis=0)))


def box_from_footprint(name: str, x_mm: float, y_mm: float, top_z_mm: float, footprint_mm: tuple[float, float],
                       surface_z_mm: float) -> AABB:
    """Return an obstacle box from a measured centre and footprint."""
    half = math.hypot(*footprint_mm) / 2.0
    return AABB(name, (x_mm - half, y_mm - half, surface_z_mm), (x_mm + half, y_mm + half, top_z_mm))


def cad_footprint(cad: ObjectCad) -> tuple[float, float]:
    """Return the CAD footprint of a part in mm."""
    return cad.size_mm[0], cad.size_mm[1]


def held_part_visible(points_uf0: np.ndarray, tcp_xyz: tuple[float, float, float], cad: ObjectCad) -> bool:
    """Return True when depth points show the part in the fingers."""
    if cad.tcp_height_mm is None:
        raise PerceptionPortError(f"{cad.object_id} has no grip, so it cannot be held")
    below = cad.tcp_height_mm - cad.bbox_min_mm[2]
    radius = math.hypot(*cad.size_mm[:2]) / 2.0
    x, y, z = tcp_xyz
    near = np.hypot(points_uf0[:, 0] - x, points_uf0[:, 1] - y) <= radius
    slab = (points_uf0[:, 2] >= z - below) & (points_uf0[:, 2] <= z - SLAB_TOP_GAP_MM)
    return int((near & slab).sum()) >= MIN_HELD_POINTS


def tallest_stack_mm(cad: CadCatalog) -> float:
    """Tallest installed product above the table: base thickness below its origin + the highest seated top."""
    base_drop = max(-o.bbox_min_mm[2] for o in cad.objects.values() if o.object_id.startswith("F"))
    return base_drop + max(o.seated_top_z_mm for o in cad.objects.values())


@dataclass(frozen=True, slots=True)
class ArmCorridor:
    """Plan-view corridor around the arm used for intrusion checks."""
    radius_mm: float
    source: str

    def __post_init__(self) -> None:
        if not (math.isfinite(self.radius_mm) and self.radius_mm > 0) or not self.source:
            raise PerceptionPortError("arm corridor needs a positive radius and a source")


def _segment_distance(points_xy: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((points_xy - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(points_xy - (a + t[:, None] * ab), axis=1)


def intrusion(points_uf0: np.ndarray, joints: np.ndarray, work_x: tuple[float, float], work_y: tuple[float, float],
              min_height_mm: float, surface_z_mm: float, corridor: ArmCorridor) -> bool:
    """Return work-area points above the table that are outside the arm corridor."""
    flange_mm = geometry.flange_pose(joints)[:3, 3] * 1000.0
    tall = points_uf0[:, 2] > surface_z_mm + min_height_mm
    inside = (points_uf0[:, 0] >= work_x[0]) & (points_uf0[:, 0] <= work_x[1]) & \
             (points_uf0[:, 1] >= work_y[0]) & (points_uf0[:, 1] <= work_y[1])
    candidates = points_uf0[tall & inside, :2]
    if len(candidates) == 0:
        return False
    away = _segment_distance(candidates, np.zeros(2), flange_mm[:2]) > corridor.radius_mm
    return int(away.sum()) >= MIN_HELD_POINTS
