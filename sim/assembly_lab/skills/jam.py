"""Watch for jams and twists during the insert contact phases.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import checks
from .gates import _world_delta
from .geometry import wrap_deg
from .plan import InsertPlan

ENTRY_PHASES = ("touch", "level", "twist", "slide", "push")
FIRST_CONTACT_N = 1.0  # axial contact above the in-hand baseline (~0.3 N): the part has met the base
TWIST_PHASES = ("twist", "slide", "push")
DESCENT_PHASES = ("slide", "push")
JAM_ROWS = 4  # 0.2 mm commanded at PRESS speed
JAM_MIN_SINK_FRAC = 0.25
JAM_RISE_N = 1.0
SEATED_MM = 0.3  # the seat z gate: closer than this is seating, not jamming


class InsertJam(RuntimeError):
    """The part wedged or jammed during the insert (counted as a robot failure)."""


def signed_turn_deg(a, b) -> float:
    """Signed turn of pose a relative to pose b about world z."""
    d = _world_delta(a, b)
    return float(wrap_deg(math.degrees(math.atan2(d[1, 0], d[0, 0]))))


@dataclass
class InsertWatch:
    plan: InsertPlan
    twist_start: tuple[np.ndarray, np.ndarray] | None = None  # (part pose, commanded TCP) at the first twist row
    rows: deque = field(default_factory=lambda: deque(maxlen=JAM_ROWS + 1))  # (commanded z, part z, axial N)
    max_follow_err_deg: float = 0.0
    first_contact: dict | None = None  # part offset from its seated pose at the first base contact (soft-corner audit)

    def observe(self, obs: dict, row: np.ndarray, phase: str, axial_n: float) -> None:
        """Raise InsertJam if the part stopped following the twist or jammed on the way down."""
        if phase not in ENTRY_PHASES:
            return
        part = checks.object_pose(obs, self.plan.object_id)
        if self.first_contact is None and axial_n > FIRST_CONTACT_N:
            seat = self.plan.seated_object_pose
            dx, dy = float(part[0] - seat[0]), float(part[1] - seat[1])
            self.first_contact = {"phase": phase, "xy_mm": round(math.hypot(dx, dy), 4), "dx_mm": round(dx, 4),
                                  "dy_mm": round(dy, 4), "z_above_seat_mm": round(float(part[2] - seat[2]), 4),
                                  "axial_N": round(float(axial_n), 3)}
        if phase not in TWIST_PHASES:
            return
        if self.twist_start is None:
            self.twist_start = (part, np.asarray(row[:6], float))
        self._check_twist(part, row)
        if phase in DESCENT_PHASES:
            self._check_jam(part, row, axial_n)

    def _check_twist(self, part: np.ndarray, row: np.ndarray) -> None:
        play = self.plan.yaw_tol_deg
        part0, tcp0 = self.twist_start
        err = abs(signed_turn_deg(part, part0) - float(wrap_deg(row[5] - tcp0[5])))
        self.max_follow_err_deg = max(self.max_follow_err_deg, err)
        if play is not None and err > play:
            raise InsertJam(f"jam: {self.plan.object_id} turned {err:.2f} deg off the commanded twist "
                            f"(CAD yaw play {play:.2f} deg)")

    def _check_jam(self, part: np.ndarray, row: np.ndarray, axial_n: float) -> None:
        self.rows.append((float(row[2]), float(part[2]), float(axial_n)))
        if len(self.rows) <= JAM_ROWS:
            return
        (cz0, pz0, f0), (cz1, pz1, f1) = self.rows[0], self.rows[-1]
        commanded, sank = cz0 - cz1, pz0 - pz1
        seat = self.plan.seated_object_pose
        above_seat = pz1 - seat[2]
        in_bore = math.hypot(part[0] - seat[0], part[1] - seat[1]) <= self.plan.capture_mm
        stuck = commanded > 0 and sank < JAM_MIN_SINK_FRAC * commanded and f1 - f0 > JAM_RISE_N
        if stuck and in_bore and above_seat > SEATED_MM:
            raise InsertJam(f"jam: {self.plan.object_id} sank {sank:.3f} of {commanded:.3f} mm commanded while the "
                            f"axial contact rose {f0:.2f} -> {f1:.2f} N, {above_seat:.2f} mm above its seat")
