"""Task state of one episode: target part, base, Mimic frames and subtask signals.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import math

import layout as L
import numpy as np

GRASP_N = 0.5
PULL_MM = 6.0
DROP_RADIUS_MM = 40.0
TOUCH_N = 0.05
SEAT = {"xy": 0.3, "z": 0.3, "ang": 1.5}      # skills/gates.py SEAT_XY_MM / SEAT_Z_MM / tilt+yaw 1 deg each
SPOT = {"xy": 1.0, "ang": 2.5}                # skills/gates.py SPOT_XY_MM / SPOT_YAW_DEG + SPOT_TILT_DEG
SKILL_SIGNALS = {"pick_insert": ("grasped", "seated"), "pull_out": ("grasped", "pulled", "placed"),
                 "place_free": ("grasped", "placed"), "defect_drop": ("grasped", "dropped")}
SUCCESS_SIGNAL = {"pick_insert": "seated", "pull_out": "placed", "place_free": "placed", "defect_drop": "dropped"}


def rot_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    rel = a[:3, :3].T @ b[:3, :3]
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rel) - 1) / 2))))


class TaskState:
    """Per-episode target bookkeeping. Poses are 4x4 UF0 mm (layout.py frames)."""

    def __init__(self, skill: str, solver: L.LayoutSolver):
        if skill not in SKILL_SIGNALS:
            raise ValueError(f"unknown skill {skill!r}; one of {sorted(SKILL_SIGNALS)}")
        self.skill, self.solver = skill, solver
        self.target: str | None = None
        self.base: str | None = None
        self.start_tf: np.ndarray | None = None
        self.grip_rel: np.ndarray | None = None
        self.latched: dict[str, bool] = {}
        self.was_held = False      # placed / dropped only count after the gripper held the target this episode

    # ------------------------------------------------------------------ inference
    def infer(self, poses: dict[str, np.ndarray], frames: dict[str, np.ndarray]) -> None:
        """Target = the object whose pose is nearest the grasp frame; base = the base whose CAD seat for the target
        is nearest the pose the seat frame implies. Called after every reset / reset_to / set_frames."""
        grasp = frames["grasp_frame"]
        movable = {u: tf for u, tf in poses.items() if L.type_id(u).startswith("M")}
        self.target = min(movable, key=lambda u: np.linalg.norm(movable[u][:3, 3] - grasp[:3, 3]))
        self.start_tf = poses[self.target]
        self.grip_rel = np.linalg.inv(self.start_tf) @ grasp
        want = frames["seat_frame"] @ np.linalg.inv(self.grip_rel)
        bases = [u for u in poses if L.type_id(u).startswith("F")]
        self.base = min(bases, key=lambda b: np.linalg.norm(self.seat_tf(poses, b)[:3, 3] - want[:3, 3])) if bases else None
        self.latched = {s: False for s in SKILL_SIGNALS[self.skill]}
        self.was_held = False

    def seat_tf(self, poses: dict[str, np.ndarray], base: str) -> np.ndarray:
        return self.solver.seat_tf_for(self.target, base, poses)

    # ------------------------------------------------------------------ signals
    def signals(self, poses: dict[str, np.ndarray], frames: dict[str, np.ndarray], jaw_n: tuple[float, float],
                touch_n: float, grip_cmd: float) -> dict[str, bool]:
        """Current (instantaneous) signals; `latched` keeps each one true once it fired (Mimic term signals)."""
        tf = poses[self.target]
        held = min(jaw_n) >= GRASP_N and grip_cmd > 0.5
        free = touch_n < TOUCH_N
        self.was_held = self.was_held or held
        released = self.was_held and free
        now = {"grasped": held}
        if self.base is not None:
            seat = self.seat_tf(poses, self.base)
            d = tf[:3, 3] - seat[:3, 3]
            now["seated"] = (math.hypot(d[0], d[1]) <= SEAT["xy"] and abs(d[2]) <= SEAT["z"]
                             and rot_angle_deg(tf, seat) <= SEAT["ang"])
        now["pulled"] = held and bool(tf[2, 3] - self.start_tf[2, 3] >= PULL_MM)
        spot_part = frames["spot_frame"] @ np.linalg.inv(self.grip_rel)
        d = tf[:3, 3] - spot_part[:3, 3]
        now["placed"] = released and math.hypot(d[0], d[1]) <= SPOT["xy"] and rot_angle_deg(tf, spot_part) <= SPOT["ang"]
        ds = tf[:2, 3] - frames["spot_frame"][:2, 3]
        now["dropped"] = released and math.hypot(*ds) <= DROP_RADIUS_MM and tf[2, 3] - self.solver.table_z < 60.0
        now = {k: bool(v) for k, v in now.items()}
        for s in self.latched:
            self.latched[s] = self.latched[s] or bool(now.get(s, False))
        return now

    def success(self, now: dict[str, bool], touch_n: float) -> bool:
        """Final state: the skill's last signal holds now and the robot no longer touches the target."""
        return bool(now.get(SUCCESS_SIGNAL[self.skill], False)) and touch_n < TOUCH_N
