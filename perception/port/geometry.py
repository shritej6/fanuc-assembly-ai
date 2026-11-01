"""Cell geometry on the real cell: the official calibration plus table localisation.
Authored by Shritej Gunda."""

from __future__ import annotations

import numpy as np

from perception.cell import localize
from perception.cell.localize import CellZones, LocalizeError, Located, Zone
from perception.port.errors import ObservationInputError, PerceptionPortError
from robot.adapter import Observation
from robot.cell.calibration import geometry

TABLE_SURFACE_Z_MM = geometry.SURFACE_Z * 1000.0 + geometry.UF0_Z_MM


def image(observation: Observation, camera: str) -> np.ndarray:
    """Return one camera image from an observation."""
    images = observation.get("images")
    if not isinstance(images, dict) or camera not in images:
        raise ObservationInputError(f"observation has no {camera} image")
    return np.asarray(images[camera])


def depth(observation: Observation, camera: str) -> np.ndarray:
    """Return one camera depth map from an observation."""
    maps = observation.get("depth")
    if not isinstance(maps, dict) or camera not in maps:
        raise ObservationInputError(f"observation has no {camera} depth map (the policy must request depth=[{camera!r}])")
    return np.asarray(maps[camera])


def joints(observation: Observation) -> np.ndarray:
    """Return the robot joints from an observation."""
    values = np.asarray(observation.get("joints"), dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ObservationInputError("observation joints must be 7 finite values")
    return values


def _scale(box: tuple[float, ...], from_size: tuple[int, int], to_size: tuple[int, int]) -> tuple[float, ...]:
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


class CalibratedCellGeometry:
    """Cell geometry backed by the official calibration."""
    table_surface_z_mm = TABLE_SURFACE_Z_MM

    def __init__(self) -> None:
        self._cams = geometry.load()
        self._zones = CellZones.load()
        self._above_mm = localize.above_table_mm()
        self.work_area = (self._zones.work_x, self._zones.work_y)

    def _table_box(self, observation: Observation, bbox_px) -> tuple[float, ...]:
        h, w = image(observation, "table").shape[:2]
        return _scale(bbox_px, (w, h), self._cams["table"]["size"])

    def table_locate(self, observation: Observation, bbox_px) -> Located:
        """Locate the part in a table box and return its UF0 position."""
        pts, uv = localize.table_points(self._cams, depth(observation, "table"), self._above_mm)
        try:
            return localize.locate(self._table_box(observation, bbox_px), pts, uv)
        except LocalizeError as error:
            raise PerceptionPortError(str(error)) from error

    def table_mask(self, observation: Observation, bbox_px) -> np.ndarray:
        """Return the depth mask of the part in a table box."""
        try:
            return localize.part_mask(self._cams, depth(observation, "table"), self._table_box(observation, bbox_px),
                                      self._above_mm)
        except LocalizeError as error:
            raise PerceptionPortError(str(error)) from error

    def table_points(self, observation: Observation, bbox_px) -> np.ndarray:
        """Return the UF0 points of the part in a table box."""
        pts, uv = localize.table_points(self._cams, depth(observation, "table"), self._above_mm)
        x0, y0, x1, y1 = self._table_box(observation, bbox_px)
        inside = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
        mm = pts[inside] * 1000.0
        if len(mm):
            mm = mm[localize._largest_group(mm[:, :2])]
        return mm + (0.0, 0.0, geometry.UF0_Z_MM)

    def zone(self, x_mm: float, y_mm: float) -> tuple[Zone, str | None]:
        """Return the cell zone of a UF0 position."""
        return self._zones.zone(x_mm, y_mm)

    def table_cloud(self, observation: Observation) -> np.ndarray:
        """Return all table depth points above the surface."""
        pts, _ = localize.table_points(self._cams, depth(observation, "table"), self._above_mm)
        return pts * 1000.0 + (0.0, 0.0, geometry.UF0_Z_MM)

    def table_box(self, observation: Observation, points_uf0_mm: np.ndarray) -> tuple[float, float, float, float]:
        """Return the table image box of the given points."""
        base_m = (np.asarray(points_uf0_mm, float) - (0.0, 0.0, geometry.UF0_Z_MM)) / 1000.0
        uv = geometry.project(self._cams["table"], base_m)
        (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
        h, w = image(observation, "table").shape[:2]
        return _scale((float(x0), float(y0), float(x1), float(y1)), self._cams["table"]["size"], (w, h))  # type: ignore[return-value]

    def wrist_points(self, observation: Observation) -> tuple[np.ndarray, np.ndarray]:
        """Return the wrist depth points in UF0 and their pixels."""
        pts, pix = geometry.depth_to_base(self._cams["wrist_depth"], depth(observation, "wrist"), with_pixels=True,
                                          joints=joints(observation))
        return pts * 1000.0 + (0.0, 0.0, geometry.UF0_Z_MM), pix

    def wrist_box(self, observation: Observation, points_uf0_mm: np.ndarray) -> tuple[float, float, float, float]:
        """Return the wrist image box of the given points."""
        try:
            box = localize.wrist_bbox(self._cams, points_uf0_mm, joints(observation))
        except LocalizeError as error:
            raise PerceptionPortError(str(error)) from error
        h, w = image(observation, "wrist").shape[:2]
        return _scale(box, self._cams["wrist"]["size"], (w, h))  # type: ignore[return-value]
