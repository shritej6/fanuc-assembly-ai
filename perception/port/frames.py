"""Convert FoundationPose STL-frame poses to the installed-local frame the controller uses.
Authored by Shritej Gunda."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from perception.port.cad import ObjectCad
from perception.port.errors import ModelOutputError

FRAME_SOURCE = "perception/port/frames.py: world_from_stl @ inv(T_local_from_stl), objects.74d749a.yaml"
RIGID_ATOL = 1e-4


@dataclass(frozen=True, slots=True)
class InstalledPose:
    """Canonical installed-local pose of one part in UF0 (mm, deg)."""

    world_from_local: np.ndarray
    x_mm: float
    y_mm: float
    z_mm: float
    yaw_deg: float
    tilt_deg: float

    def contract_pose(self) -> dict[str, float]:
        """Return the pose as a contract dict."""
        return {"x": self.x_mm, "y": self.y_mm, "z": self.z_mm, "yaw": self.yaw_deg}

    def contract_tuple(self) -> tuple[float, float, float, float]:
        """Return the pose as an (x, y, z, yaw) tuple."""
        return (self.x_mm, self.y_mm, self.z_mm, self.yaw_deg)

    def to_local(self, points_uf0_mm: np.ndarray) -> np.ndarray:
        """Return UF0 points in the part's installed-local frame."""
        inverse = np.linalg.inv(self.world_from_local)
        return points_uf0_mm @ inverse[:3, :3].T + inverse[:3, 3]


def _require_rigid(matrix: np.ndarray) -> None:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ModelOutputError("FoundationPose pose must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=RIGID_ATOL) or np.linalg.det(rotation) <= 0:
        raise ModelOutputError("FoundationPose pose is not a proper rigid transform")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0)):
        raise ModelOutputError("FoundationPose pose has a non-affine bottom row")


def installed_pose(world_from_stl: np.ndarray, cad: ObjectCad) -> InstalledPose:
    """Return the installed-local pose of a FoundationPose STL fit."""
    world_from_stl = np.asarray(world_from_stl, dtype=float)
    _require_rigid(world_from_stl)
    world_from_local = world_from_stl @ np.linalg.inv(np.array(cad.local_from_stl))
    rotation = world_from_local[:3, :3]
    tilt = math.degrees(math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0))))
    if math.hypot(rotation[0, 0], rotation[1, 0]) < 1e-6:
        raise ModelOutputError(f"{cad.object_id}: posed x axis is vertical, table yaw is undefined")
    yaw = (math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])) + 180.0) % 360.0 - 180.0
    x, y, z = (float(v) for v in world_from_local[:3, 3])
    return InstalledPose(world_from_local, x, y, z, yaw, tilt)


def resting_side(pose: InstalledPose, cad: ObjectCad) -> str:
    """'upright' on the installed face, 'side' or 'flipped' otherwise (contract detection.resting_side)."""
    if pose.tilt_deg <= cad.fallen_tilt_deg:
        return "upright"
    return "flipped" if pose.tilt_deg >= 180.0 - cad.fallen_tilt_deg else "side"


def posed_bbox_corners(pose: InstalledPose, cad: ObjectCad) -> np.ndarray:
    """The 8 corners of the installed-frame CAD bbox in UF0 mm."""
    lo, hi = cad.bbox_min_mm, cad.bbox_max_mm
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return corners @ pose.world_from_local[:3, :3].T + pose.world_from_local[:3, 3]
