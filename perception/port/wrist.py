"""Inspect one part from one wrist close-up: pose, box, classifier and size.
Authored by Shritej Gunda."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from assembly.controller_ports import current_waypoint, observation_time
from perception.cell.localize import _largest_group
from perception.port.backends import CellGeometry, PoseEstimator, WristClassifier, WristReading
from perception.port.cad import CadCatalog
from perception.port.errors import PerceptionPortError
from perception.port.frames import InstalledPose, installed_pose, posed_bbox_corners, resting_side
from perception.port.geometry import depth, image
from perception.port.size_gate import SizeUncertainty, SizeVerdict, gate, measure
from robot.adapter import Observation

ABOVE_SUPPORT_BOUNDS = 2.0   # perception/cell/localize.py ABOVE_TABLE_SIGMAS: two noise bounds above the support
SEARCH_MARGIN_MM = 12.0      # perception/cell/tracker.py GATE_MM: how far the part may be from its expected centre
MIN_MASK_PIXELS = 12         # perception/cell/localize.py MIN_POINTS

ImageSink = Callable[[str, np.ndarray], str]


@dataclass(frozen=True, slots=True)
class WristView:
    """Everything measured in one wrist close-up."""
    observed_at_s: float
    camera_tcp: tuple[float, float, float, float, float, float]
    image_ref: str
    pose: InstalledPose
    fit_score: float
    side: str
    bbox_px: tuple[float, float, float, float]
    reading: WristReading
    size: SizeVerdict


def _part_points(points: np.ndarray, pixels: np.ndarray, center_xy: tuple[float, float], radius_mm: float,
                 floor_z_mm: float) -> tuple[np.ndarray, np.ndarray]:
    near = (np.hypot(points[:, 0] - center_xy[0], points[:, 1] - center_xy[1]) <= radius_mm) & \
           (points[:, 2] > floor_z_mm)
    points, pixels = points[near], pixels[near]
    if len(points) == 0:
        return points, pixels
    keep = _largest_group(points[:, :2])
    return points[keep], pixels[keep]


def _mask(shape: tuple[int, ...], pixels: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape[:2], bool)
    mask[pixels[:, 1].astype(int), pixels[:, 0].astype(int)] = True
    if mask.sum() < MIN_MASK_PIXELS:
        raise PerceptionPortError(f"{int(mask.sum())} wrist depth pixels on the part; need {MIN_MASK_PIXELS}")
    return mask


class WristInspector:
    """Takes one wrist close-up of one part."""
    def __init__(self, geometry: CellGeometry, pose_estimator: PoseEstimator, classifier: WristClassifier,
                 cad: CadCatalog, uncertainty: SizeUncertainty, image_sink: ImageSink) -> None:
        self._geometry = geometry
        self._pose = pose_estimator
        self._classifier = classifier
        self._cad = cad
        self._unc = uncertainty
        self._sink = image_sink

    def look(self, observation: Observation, object_id: str, center_xy: tuple[float, float],
             support_z_mm: float) -> WristView:
        """Inspect the part near the centre and return the wrist view."""
        observed_at = observation_time(observation)
        cad = self._cad.get(object_id)
        radius = math.hypot(*cad.size_mm[:2]) / 2.0 + SEARCH_MARGIN_MM
        points, pixels = self._geometry.wrist_points(observation)
        points, pixels = _part_points(points, pixels, center_xy, radius,
                                      support_z_mm + ABOVE_SUPPORT_BOUNDS * self._unc.z_mm)
        mask = _mask(depth(observation, "wrist").shape, pixels)
        fit = self._pose(object_id, "wrist", observation, mask)
        pose = installed_pose(fit.world_from_stl, cad)
        bbox = self._geometry.wrist_box(observation, posed_bbox_corners(pose, cad))
        wrist_image = image(observation, "wrist")
        reading = self._classifier(wrist_image, bbox)
        cues = tuple(c for other in self._cad.candidates_for(object_id) for c in cad.cues.get(other, ()))
        size = gate(object_id, measure(pose.to_local(points), points, support_z_mm, cues), self._cad, self._unc)
        tcp = current_waypoint(observation)
        ref = self._sink(f"wrist_{observed_at:.3f}_{object_id}", wrist_image)
        return WristView(observed_at, (tcp.x, tcp.y, tcp.z, tcp.w, tcp.p, tcp.r), ref, pose, fit.fit_score,
                         resting_side(pose, cad), bbox, reading, size)
