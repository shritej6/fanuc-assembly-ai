"""Reject drop spots and spare-store slots, in UF0, for parts the robot takes out of a build.
Authored by Shritej Gunda.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.cell.localize import CellZones
from robot.cell.calibration import geometry
from sdg.wrist_ik import reachable

TABLE_Z_UF0_MM = geometry.SURFACE_Z * 1000.0 + geometry.UF0_Z_MM
RELEASE_GAP_MM = 5.0        # a dropped PLA part falls this far; less risks the fingertips touching the table
APPROACH_ABOVE_MM = 120.0   # same approach height robot/placement.py checks for every placed object
REJECT_PITCH_MM = 40.0      # the largest part footprint radius is about 26 mm (housings); neighbours may touch
ROLLS = (-90.0, 0.0, 90.0, 180.0)
MAX_HELD_BELOW_TCP_MM = 60.0


class TargetError(ValueError):
    """No safe place for this part: the reject zone is full or the input is impossible."""


@dataclass(frozen=True, slots=True)
class Release:
    """Where to release a part and the height to approach from."""
    x_mm: float
    y_mm: float
    z_mm: float
    approach_z_mm: float


def _grid(bounds_x: tuple[float, float], bounds_y: tuple[float, float]) -> list[tuple[float, float]]:
    xs = np.arange(bounds_x[0] + REJECT_PITCH_MM / 2, bounds_x[1] - REJECT_PITCH_MM / 2 + 1e-9, REJECT_PITCH_MM)
    ys = np.arange(bounds_y[0] + REJECT_PITCH_MM / 2, bounds_y[1] - REJECT_PITCH_MM / 2 + 1e-9, REJECT_PITCH_MM)
    if len(ys) == 0:
        ys = np.array([(bounds_y[0] + bounds_y[1]) / 2])
    return [(float(x), float(y)) for y in ys for x in xs]


def _check_reach(label: str, x: float, y: float) -> None:
    for z in (TABLE_Z_UF0_MM + RELEASE_GAP_MM + 10.0, TABLE_Z_UF0_MM + APPROACH_ABOVE_MM):
        for roll in ROLLS:
            if not reachable([x, y, z, -180.0, 0.0, roll]):
                raise TargetError(f"{label} ({x:.0f}, {y:.0f}) is not reachable at z {z:.0f}, roll {roll:.0f}")


class CellTargets:
    """Reject spots and store slots, checked for reach, handed out one at a time."""
    def __init__(self, zones: CellZones) -> None:
        self.reject_spots = _grid(zones.reject_x, zones.reject_y)
        self.slots = {slot_id: (x, y) for slot_id, x, y in zones.slots}
        for i, (x, y) in enumerate(self.reject_spots):
            _check_reach(f"reject spot {i}", x, y)
        for slot_id, (x, y) in self.slots.items():
            _check_reach(f"store slot {slot_id}", x, y)
        self._used = 0

    def next_reject(self, held_below_tcp_mm: float) -> Release:
        """Return the next free reject spot for a part held this far below the fingertips."""
        if self._used >= len(self.reject_spots):
            raise TargetError(f"reject zone full ({len(self.reject_spots)} spots); operator must empty it")
        x, y = self.reject_spots[self._used]
        release = _release(x, y, held_below_tcp_mm)
        self._used += 1
        return release

    def slot(self, slot_id: str, held_below_tcp_mm: float) -> Release:
        """Return the release point for one spare-store slot."""
        if slot_id not in self.slots:
            raise TargetError(f"no store slot {slot_id!r}")
        return _release(*self.slots[slot_id], held_below_tcp_mm)


def _release(x: float, y: float, held_below_tcp_mm: float) -> Release:
    if not (np.isfinite(held_below_tcp_mm) and 0.0 <= held_below_tcp_mm <= MAX_HELD_BELOW_TCP_MM):
        raise TargetError(f"held part reaching {held_below_tcp_mm} mm below the fingertips is not possible")
    z = TABLE_Z_UF0_MM + RELEASE_GAP_MM + held_below_tcp_mm
    return Release(x, y, z, TABLE_Z_UF0_MM + APPROACH_ABOVE_MM + held_below_tcp_mm)
