"""Precision gates (sim/assembly_lab/INTERFACES.md "Precision gates", user: "like heart surgery").

Every gate is measured and logged per episode, kept or rejected, so the acceptance rate is measured, not assumed.
A demo is kept only if every gate passed. Values in mm / deg / N.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import pose_to_matrix, wpr_to_rot, wrap_deg

SEAT_XY_MM = 0.3  # final part xy vs the CAD installed pose
SEAT_YAW_DEG = 1.0
SEAT_Z_MM = 0.3  # seated height, ±
SEAT_TILT_DEG = 1.0
OTHER_MOVE_MM = 1.0  # no non-target object moved more than this
OTHER_MOVE_DEG = 2.0
GRIP_PAD_MM = 1.0  # fingers closed within this of the planned CAD pad TCP (pad is 7 mm wide)
GRIP_ROLL_DEG = 1.0  # at the planned roll
FINGER_TOUCH_N = 0.05  # a finger force above this on a non-target object counts as a touch (env's grip threshold)
SPOT_XY_MM = 1.0  # a part set down on a remembered test-bed spot
SPOT_YAW_DEG = 2.0
SPOT_TILT_DEG = 1.0


class GateFailed(RuntimeError):
    """A precision gate failed."""


WORLD_UP = (0.0, 0.0, 1.0)


def _world_delta(a, b) -> np.ndarray:
    """Rotation taking pose b's orientation to pose a's, expressed in the world (UF0) frame."""
    return wpr_to_rot(*np.asarray(a, float)[3:6]) @ wpr_to_rot(*np.asarray(b, float)[3:6]).T


def axis_tilt_deg(a, b, up=WORLD_UP) -> float:
    """How far pose a's installed-up direction leans from pose b's, independent of yaw and of the part's local axes.

    `up` is the world direction that is up in the target pose b (objects.yaml: -insert_axis). Comparing local z
    axes is wrong for parts whose local z lies flat when installed (the *_04 locks): a pure yaw turn would leak in.
    """
    u = np.asarray(up, float) / np.linalg.norm(up)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(u @ _world_delta(a, b) @ u)))))


def yaw_err_deg(a, b) -> float:
    """Turn of pose a relative to pose b about world z (independent of which local axis lies where)."""
    d = _world_delta(a, b)
    return abs(wrap_deg(math.degrees(math.atan2(d[1, 0], d[0, 0]))))


@dataclass
class GateLog:
    values: dict[str, dict] = field(default_factory=dict)
    touch_method: str = "unmeasured"

    def check(self, name: str, value: float, limit: float, ok: bool | None = None) -> bool:
        ok = (abs(value) <= limit) if ok is None else ok
        self.values[name] = {"value": round(float(value), 4), "limit": limit, "pass": bool(ok)}
        return ok

    @property
    def failed(self) -> list[str]:
        return [k for k, v in self.values.items() if not v["pass"]]

    def require(self, names) -> None:
        bad = [n for n in names if n in self.failed]
        if bad:
            raise GateFailed("precision gate: " + ", ".join(f"{n}={self.values[n]['value']}" for n in bad))


def seat_yaw_limit(part_yaw_tol_deg: float | None) -> float:
    """1 deg, or the part's own CAD yaw tolerance (objects.yaml) when that is tighter."""
    return SEAT_YAW_DEG if part_yaw_tol_deg is None else min(SEAT_YAW_DEG, float(part_yaw_tol_deg))


def seat_gates(log: GateLog, actual, cad, part_yaw_tol_deg: float | None = None, up=WORLD_UP) -> None:
    a, c = np.asarray(actual, float), np.asarray(cad, float)
    log.check("seat_xy_mm", math.hypot(a[0] - c[0], a[1] - c[1]), SEAT_XY_MM)
    log.check("seat_yaw_deg", yaw_err_deg(a, c), seat_yaw_limit(part_yaw_tol_deg))
    log.check("seat_z_mm", a[2] - c[2], SEAT_Z_MM)
    log.check("seat_tilt_deg", axis_tilt_deg(a, c, up), SEAT_TILT_DEG)
    log.require(("seat_xy_mm", "seat_yaw_deg", "seat_z_mm", "seat_tilt_deg"))


def spot_gates(log: GateLog, actual, expected) -> None:
    a, e = np.asarray(actual, float), np.asarray(expected, float)
    log.check("spot_xy_mm", math.hypot(a[0] - e[0], a[1] - e[1]), SPOT_XY_MM)
    log.check("spot_yaw_deg", yaw_err_deg(a, e), SPOT_YAW_DEG)
    log.check("spot_tilt_deg", axis_tilt_deg(a, e), SPOT_TILT_DEG)
    log.require(("spot_xy_mm", "spot_yaw_deg", "spot_tilt_deg"))


def grip_gates(log: GateLog, obs: dict, oid: str, planned_t_obj_tcp: np.ndarray, planned_r: float) -> None:
    """After the fingers close: the TCP sits on the planned CAD pad at the planned roll, holding the target."""
    obj = np.asarray(obs["sim"]["object_poses"][oid], float)
    tcp = np.asarray(obs["tcp_pose"], float)[:6]
    want = pose_to_matrix(obj) @ planned_t_obj_tcp
    log.check("grip_pad_mm", float(np.linalg.norm(want[:3, 3] - tcp[:3])), GRIP_PAD_MM)
    log.check("grip_roll_deg", wrap_deg(tcp[5] - planned_r), GRIP_ROLL_DEG)
    held = obs["sim"].get("gripper_obj", oid)
    log.check("grip_holds_target", 0.0, 0.0, ok=held == oid)
    log.require(("grip_pad_mm", "grip_roll_deg", "grip_holds_target"))


def others_moved(log: GateLog, start: dict, end: dict, movers: set[str]) -> None:
    """No object other than the manipulated part moved more than 1 mm / 2 deg over the episode."""
    worst_mm, worst_deg, who = 0.0, 0.0, ""
    for oid, p0 in start.items():
        if oid in movers:
            continue
        p0, p1 = np.asarray(p0, float), np.asarray(end[oid], float)
        d_mm = float(np.linalg.norm(p1[:3] - p0[:3]))
        r0, r1 = wpr_to_rot(*p0[3:6]), wpr_to_rot(*p1[3:6])
        d_deg = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r0.T @ r1) - 1) / 2))))
        if d_mm / OTHER_MOVE_MM + d_deg / OTHER_MOVE_DEG > worst_mm / OTHER_MOVE_MM + worst_deg / OTHER_MOVE_DEG:
            worst_mm, worst_deg, who = d_mm, d_deg, oid
    log.check("other_moved_mm", worst_mm, OTHER_MOVE_MM)
    log.check("other_moved_deg", worst_deg, OTHER_MOVE_DEG)
    if who:
        log.values["other_moved_mm"]["object"] = who
    log.require(("other_moved_mm", "other_moved_deg"))


@dataclass
class TouchMonitor:
    """Arm and fingers touch only the target.

    Measured when the env reports `sim.robot_touch` {uid: peak N of any robot link (arm or finger) on that object over
    the step}. Without it, the only signal is `sim.gripper_obj` (which object is in both pads); that proxy cannot see a
    single finger or an arm link brushing a neighbour, so it is accepted only when the caller allows it explicitly.
    """

    allow_proxy: bool
    worst_n: float = 0.0
    who: str = ""
    method: str = ""

    def observe(self, obs: dict, targets: set[str]) -> None:
        sim = obs.get("sim") or {}
        if "robot_touch" in sim:
            self.method = "robot_touch"
            for oid, f in sim["robot_touch"].items():
                if oid not in targets and float(f) > self.worst_n:
                    self.worst_n, self.who = float(f), oid
            return
        self.method = self.method or "proxy:gripper_obj"
        held = sim.get("gripper_obj")
        if held is not None and held not in targets:
            self.worst_n, self.who = max(self.worst_n, FINGER_TOUCH_N * 2), held

    def finish(self, log: GateLog, require: bool = True) -> None:
        """require=False (measure-only runs): the value is logged but the episode is not ended on it."""
        log.touch_method = self.method or "unmeasured"
        ok = self.worst_n <= FINGER_TOUCH_N and (self.method == "robot_touch" or self.allow_proxy)
        log.check("touch_non_target_N", self.worst_n, FINGER_TOUCH_N, ok=ok)
        log.values["touch_non_target_N"]["method"] = log.touch_method
        if self.who:
            log.values["touch_non_target_N"]["object"] = self.who
        if require:
            log.require(("touch_non_target_N",))
