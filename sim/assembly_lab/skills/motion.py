"""Turn waypoints into act_tcp rows at a fixed control step.
Authored by Shritej Gunda.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import interpolate, rotation_distance_deg, tilt_deg
from .plan import SLOW_ZONE_MM

DT_S = 0.1  # one control step = one Isaac Lab env step (env step_dt 0.1 s), so Mimic replays rows 1:1
WORKSPACE = ((150.0, 650.0), (-450.0, 450.0), (-334.0, 300.0))
OPEN, CLOSED = 0.0, 1.0


@dataclass(frozen=True)
class Speed:
    mm: float  # max translation per row
    deg: float  # max rotation per row


FREE = Speed(10.0, 6.0)  # 100 mm/s in clear space
APPROACH = Speed(2.0, 2.0)  # 20 mm/s near parts (the last 5 mm is always SLOW)
SLOW = Speed(0.25, 0.25)  # 2.5 mm/s in the last 5 mm and in contact
GRIP_SETTLE_ROWS = 8  # 0.8 s for the fingers to open / close
ENTER_ROWS = 15  # 1.5 s to reach the first in-box target from the home pose
WORKSPACE_MARGIN_MM = 5.0
MAX_TOOL_TILT_DEG = 6.0  # |tool axis - straight down|: the real-arm guard rejects rows beyond this


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
        if tilt_deg(row[:6]) > MAX_TOOL_TILT_DEG + 1e-6:
            raise MotionError(f"{self.skill}: tool {tilt_deg(row[:6]):.2f} deg from straight down > {MAX_TOOL_TILT_DEG}")

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
        res_mm, res_deg = min(0.05, speed.mm), min(0.05, speed.deg)  # samples no coarser than the step limit
        n = max(1, int(np.ceil(max(dist / res_mm, ang / res_deg))))
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

    def enter(self) -> None:
        """From a start pose outside the workspace box (the cell's home is at x ~ 87 mm) or with the tool more than
        6 deg off straight down: one in-box, tool-down target, held while the arm gets there. act_tcp rows outside the box are refused by the real cell, so no row ever leaves it."""
        inside = start_pose(self.pose)
        if not np.allclose(inside, self.pose):
            self._emit("enter", [inside] * ENTER_ROWS, self.gripper)

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


def in_workspace(pose, margin: float = WORKSPACE_MARGIN_MM) -> np.ndarray:
    """The pose with x, y, z clamped `margin` inside the workspace box."""
    out = np.asarray(pose, dtype=float).copy()
    for k, (lo, hi) in enumerate(WORKSPACE):
        out[k] = min(max(out[k], lo + margin), hi - margin)
    return out


def start_pose(pose) -> np.ndarray:
    """The first pose every episode goes to and returns to: the start pose clamped inside the box, with the tool
    straightened to point down (roll kept) when it is more than 6 deg off (the cell's home is 50 deg off)."""
    out = in_workspace(pose)
    if tilt_deg(out) > MAX_TOOL_TILT_DEG:
        out[3], out[4] = 180.0, 0.0
    return out


def above(pose, z: float) -> np.ndarray:
    out = np.asarray(pose, dtype=float).copy()
    out[2] = z
    return out


def rows_of(segments: list[Segment]) -> np.ndarray:
    return np.vstack([s.rows for s in segments]) if segments else np.zeros((0, 7))


def chunks(rows: np.ndarray, size: int = 100) -> list[np.ndarray]:
    """act_tcp takes 1-100 rows per call."""
    return [rows[i:i + size] for i in range(0, len(rows), size)]
