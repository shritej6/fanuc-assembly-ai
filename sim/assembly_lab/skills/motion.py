"""Turn waypoints into `act_tcp` rows `[x, y, z, w, p, r, gripper]` at a fixed control step.

Every skill builds its motion through `Motion`, which enforces, for every row it emits:
- the workspace box (PLAN 3.1),
- the seated-height floor (never below `floor_z`, which skills set to seated TCP z - 1 mm),
- the slow last 5 mm above the active contact height,
- per-row step limits, so the speed is the same in every demo (one fixed dt for GR00T).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import interpolate, rotation_distance_deg
from .plan import SLOW_ZONE_MM

DT_S = 0.2  # one control step; act_tcp allows 0.1-1.0 s
WORKSPACE = ((150.0, 650.0), (-450.0, 450.0), (-334.0, 300.0))
OPEN, CLOSED = 0.0, 1.0


@dataclass(frozen=True)
class Speed:
    mm: float  # max translation per row
    deg: float  # max rotation per row


FREE = Speed(10.0, 6.0)  # 50 mm/s in clear space
APPROACH = Speed(2.0, 2.0)  # 10 mm/s near parts
SLOW = Speed(0.5, 0.5)  # 2.5 mm/s in the last 5 mm and in contact
GRIP_SETTLE_ROWS = 4  # 0.8 s for the fingers to open / close


class MotionError(ValueError):
    """A row would leave the workspace or go below the seated-height floor."""


@dataclass
class Segment:
    """Rows that make one labelled phase; `subtask` groups phases for Isaac Lab Mimic."""

    skill: str
    phase: str
    subtask: str
    object_ref: str
    rows: np.ndarray
    learn: bool = True  # False: rows exist only to stage a recovery case (masked out of training)


@dataclass
class Motion:
    pose: np.ndarray
    gripper: float
    skill: str
    subtask: str
    object_ref: str
    floor_z: float = WORKSPACE[2][0]
    contact_z: float | None = None  # slow zone reference height (seated or grasp TCP z)
    learn: bool = True
    segments: list[Segment] = field(default_factory=list)

    def _check(self, row: np.ndarray) -> None:
        for v, (lo, hi), name in zip(row[:3], WORKSPACE, "xyz"):
            if not lo <= v <= hi:
                raise MotionError(f"{self.skill}: {name}={v:.2f} mm outside the workspace [{lo}, {hi}]")
        if row[2] < self.floor_z - 1e-6:
            raise MotionError(f"{self.skill}: z={row[2]:.2f} below the floor {self.floor_z:.2f}")

    def _limit(self, z: float, speed: Speed) -> Speed:
        if self.contact_z is not None and z <= self.contact_z + SLOW_ZONE_MM + 1e-6:
            return Speed(min(speed.mm, SLOW.mm), min(speed.deg, SLOW.deg))
        return speed

    def _exceeds(self, a: np.ndarray, b: np.ndarray, speed: Speed) -> bool:
        lim = self._limit(min(a[2], b[2]), speed)
        return (np.linalg.norm(b[:3] - a[:3]) > lim.mm + 1e-9 or rotation_distance_deg(a, b) > lim.deg + 1e-9)

    def _path(self, goal: np.ndarray, speed: Speed) -> list[np.ndarray]:
        """Rows along the straight path, each as far as the local step limit allows and never farther."""
        dist = float(np.linalg.norm(goal[:3] - self.pose[:3]))
        ang = rotation_distance_deg(self.pose, goal)
        n = max(1, int(np.ceil(max(dist / 0.05, ang / 0.05))))  # dense samples (well under any limit)
        rows, last, prev = [], self.pose, self.pose
        for i in range(1, n + 1):
            p = interpolate(self.pose, goal, i / n)
            if self._exceeds(last, p, speed):
                rows.append(prev)
                last = prev
            prev = p
        if not rows or not np.allclose(rows[-1], prev):
            rows.append(prev)
        return rows

    def move(self, phase: str, goal, speed: Speed = FREE) -> None:
        goal = np.asarray(goal, dtype=float)
        poses = self._path(goal, speed)
        self._emit(phase, poses, self.gripper)

    def grip(self, phase: str, value: float, rows: int = GRIP_SETTLE_ROWS) -> None:
        self.gripper = value
        self._emit(phase, [self.pose] * rows, value)

    def hold(self, phase: str, rows: int) -> None:
        self._emit(phase, [self.pose] * rows, self.gripper)

    def _emit(self, phase: str, poses: list[np.ndarray], gripper: float) -> None:
        rows = np.array([[*p, gripper] for p in poses], dtype=float)
        for r in rows:
            self._check(r)
        if len(poses):
            self.pose = np.asarray(poses[-1], dtype=float)
        if self.segments and self.segments[-1].phase == phase and self.segments[-1].subtask == self.subtask:
            self.segments[-1].rows = np.vstack([self.segments[-1].rows, rows])
        else:
            self.segments.append(Segment(self.skill, phase, self.subtask, self.object_ref, rows, self.learn))


def above(pose, z: float) -> np.ndarray:
    out = np.asarray(pose, dtype=float).copy()
    out[2] = z
    return out


def rows_of(segments: list[Segment]) -> np.ndarray:
    return np.vstack([s.rows for s in segments]) if segments else np.zeros((0, 7))


def chunks(rows: np.ndarray, size: int = 100) -> list[np.ndarray]:
    """act_tcp takes 1-100 rows per call."""
    return [rows[i:i + size] for i in range(0, len(rows), size)]
