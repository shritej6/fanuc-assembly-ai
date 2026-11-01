"""Locate table-camera detections in UF0 from the table depth stream and find their cell zone.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import yaml

from robot.cell.calibration import geometry

ROOT = Path(__file__).resolve().parents[2]
LAYOUT = ROOT / "sdg" / "configs" / "cell_layout.yaml"
UF0_Z_MM = geometry.UF0_Z_MM
# Table-plane depth noise is cameras.json table_depth rms_plane_mm (2.31 mm, solved on the cell); a point counts as part
# when it is this many noise sigmas above the surface. The lowest standing parts (the 04 guards) are 8 mm tall.
ABOVE_TABLE_SIGMAS = 2.0
# x, y grid for grouping part points: twice the table depth pixel footprint on the table (2.95-3.21 mm at 0.90-0.98 m
# range, fx 305.5), so one part never splits into groups; parts closer than about two cells merge into one group.
CLUSTER_CELL_MM = 6.5
MIN_POINTS = 12           # fewer depth points than this on a part is a depth hole, not a measurement
TOP_PERCENTILE = 95.0


class LocalizeError(ValueError):
    """The depth stream cannot place this detection; nothing is guessed."""


class Zone(str, Enum):
    """Cell zone a point lies in."""
    TABLE = "table"
    SPARE = "spare"
    REJECT = "reject"
    OUTSIDE = "outside"


@dataclass(frozen=True, slots=True)
class Located:
    """UF0 centre, top height and point count of one located part."""
    x_mm: float
    y_mm: float
    top_z_mm: float
    points: int


@dataclass(frozen=True, slots=True)
class CellZones:
    """Work area, reach limits, reject zone and store slots of the cell."""
    work_x: tuple[float, float]
    work_y: tuple[float, float]
    reach_min: float
    reach_max: float
    reject_x: tuple[float, float]
    reject_y: tuple[float, float]
    slots: tuple[tuple[str, float, float], ...]
    slot_radius: float

    @classmethod
    def load(cls, path: Path = LAYOUT) -> CellZones:
        """Load the cell zones from the layout file."""
        raw = yaml.safe_load(path.read_text())
        store = raw["spare_store"]
        return cls(work_x=tuple(raw["work_area"]["x"]), work_y=tuple(raw["work_area"]["y"]),
                   reach_min=float(raw["reach_radius_min"]), reach_max=float(raw["reach_radius_max"]),
                   reject_x=tuple(raw["reject_zone"]["x"]), reject_y=tuple(raw["reject_zone"]["y"]),
                   slots=tuple((s["id"], float(s["x"]), float(s["y"])) for s in store["slots"]),
                   slot_radius=float(store["slot_pitch"]) / 2.0)

    def zone(self, x_mm: float, y_mm: float) -> tuple[Zone, str | None]:
        """Return the zone of a UF0 point and the store slot id when it is in one."""
        if _within(x_mm, self.reject_x) and _within(y_mm, self.reject_y):
            return Zone.REJECT, None
        for slot_id, sx, sy in self.slots:
            if np.hypot(x_mm - sx, y_mm - sy) < self.slot_radius:
                return Zone.SPARE, slot_id
        reach = float(np.hypot(x_mm, y_mm))
        if (_within(x_mm, self.work_x) and _within(y_mm, self.work_y)
                and self.reach_min <= reach <= self.reach_max):
            return Zone.TABLE, None
        return Zone.OUTSIDE, None


def _within(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def above_table_mm(path: Path = geometry.HERE / "cameras.json") -> float:
    """Return the height above the surface (mm) that separates part points from depth noise."""
    return ABOVE_TABLE_SIGMAS * float(json.loads(path.read_text())["table_depth"]["rms_plane_mm"])


def table_points(cams: dict, depth_mm: np.ndarray, above_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Return base-frame points (m) above the table and their colour pixels."""
    pts = geometry.depth_to_base(cams["table_depth"], depth_mm)
    above = pts[:, 2] * 1000.0 > geometry.SURFACE_Z * 1000.0 + above_mm
    pts = pts[above]
    return pts, geometry.project(cams["table"], pts)


def _largest_group(xy_mm: np.ndarray) -> np.ndarray:
    """Return a mask of the points in the largest connected group of occupied cells."""
    cells = np.floor(xy_mm / CLUSTER_CELL_MM).astype(np.int64)
    occupied = {tuple(c) for c in cells}
    best: set[tuple[int, int]] = set()
    seen: set[tuple[int, int]] = set()
    for start in occupied:
        if start in seen:
            continue
        group, stack = set(), [start]
        while stack:
            cell = stack.pop()
            if cell in seen or cell not in occupied:
                continue
            seen.add(cell)
            group.add(cell)
            stack.extend((cell[0] + i, cell[1] + j) for i in (-1, 0, 1) for j in (-1, 0, 1))
        best = max(best, group, key=len)
    return np.array([tuple(c) in best for c in cells])


def locate(bbox_px: tuple[float, float, float, float], pts: np.ndarray, uv: np.ndarray) -> Located:
    """Return the UF0 centre and top of the part inside a table colour box."""
    x0, y0, x1, y1 = (float(v) for v in bbox_px)
    if not (x1 > x0 and y1 > y0):
        raise LocalizeError(f"empty box {bbox_px}")
    inside = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    mm = pts[inside] * 1000.0
    if len(mm):
        mm = mm[_largest_group(mm[:, :2])]
    if len(mm) < MIN_POINTS:
        raise LocalizeError(f"{len(mm)} depth points on the part inside {bbox_px}; need {MIN_POINTS}")
    return Located(x_mm=float(np.median(mm[:, 0])), y_mm=float(np.median(mm[:, 1])),
                   top_z_mm=float(np.percentile(mm[:, 2], TOP_PERCENTILE)) + UF0_Z_MM, points=len(mm))


def part_mask(cams: dict, depth_mm: np.ndarray, bbox_px: tuple[float, float, float, float],
              above_mm: float) -> np.ndarray:
    """Return the table depth pixels on the part inside a table colour box."""
    pts, pix = geometry.depth_to_base(cams["table_depth"], depth_mm, with_pixels=True)
    above = pts[:, 2] * 1000.0 > geometry.SURFACE_Z * 1000.0 + above_mm
    pts, pix = pts[above], pix[above].astype(np.int64)
    uv = geometry.project(cams["table"], pts)
    x0, y0, x1, y1 = bbox_px
    inside = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    pts, pix = pts[inside], pix[inside]
    mask = np.zeros(np.asarray(depth_mm).shape, bool)
    if len(pts):
        keep = _largest_group(pts[:, :2] * 1000.0)
        mask[pix[keep, 1], pix[keep, 0]] = True
    if mask.sum() < MIN_POINTS:
        raise LocalizeError(f"{int(mask.sum())} depth pixels on the part inside {bbox_px}; need {MIN_POINTS}")
    return mask


def wrist_bbox(cams: dict, points_uf0_mm: np.ndarray, joints) -> tuple[float, float, float, float]:
    """Return the wrist colour-image box around UF0 points for this arm pose."""
    pts = np.asarray(points_uf0_mm, float).reshape(-1, 3)
    if not np.isfinite(pts).all() or len(pts) == 0:
        raise LocalizeError("wrist_bbox needs finite UF0 points")
    base = (pts - (0.0, 0.0, UF0_Z_MM)) / 1000.0
    flange = np.linalg.inv(geometry.flange_pose(joints))
    in_flange = base @ flange[:3, :3].T + flange[:3, 3]
    cam_from_flange = np.linalg.inv(cams["wrist"]["T"])
    if (in_flange @ cam_from_flange[:3, :3].T + cam_from_flange[:3, 3])[:, 2].min() <= 0:
        raise LocalizeError("part is behind the wrist camera")
    uv = geometry.project(cams["wrist"], in_flange)
    w, h = cams["wrist"]["size"]
    x0, y0 = uv.min(0)
    x1, y1 = uv.max(0)
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        raise LocalizeError("part is outside the wrist image")
    return (max(0.0, float(x0)), max(0.0, float(y0)), min(float(w), float(x1)), min(float(h), float(y1)))
