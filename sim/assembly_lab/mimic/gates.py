"""Precision gates that every generated demo must pass before it is kept.
Authored by Shritej Gunda.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .force_guard import ForceLimits
from .frames import wpr_to_rot


@dataclass(frozen=True)
class GateLimits:
    """Gate limits from INTERFACES.md, with grip limits chosen where it gives no number."""
    xy_mm: float = 0.3
    yaw_deg: float = 1.0
    z_mm: float = 0.3
    tilt_deg: float = 1.0
    other_move_mm: float = 1.0
    other_turn_deg: float = 2.0
    grip_pos_mm: float = 0.5      # in-hand offset vs the planned CAD grip (INTERFACES: "on the pad")
    grip_roll_deg: float = 1.0    # TCP roll vs the planned roll


@dataclass
class GateReport:
    """Gate result: pass flag, every gate value and the gates that failed."""
    ok: bool
    values: dict = field(default_factory=dict)
    failed: list = field(default_factory=list)


def _rot(pose) -> np.ndarray:
    return wpr_to_rot(np.asarray(pose, dtype=np.float64)[3:6])


def pose_error(actual, expected) -> dict:
    """Return the xy, dz, tilt and yaw errors between two frames."""
    a, e = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    ra, re = _rot(a), _rot(e)
    tilt = np.degrees(np.arccos(np.clip(ra[:, 2] @ re[:, 2], -1.0, 1.0)))
    rel = re.T @ ra
    yaw = np.degrees(np.arctan2(rel[1, 0], rel[0, 0]))
    return {"xy": float(np.hypot(*(a[:2] - e[:2]))), "dz": float(a[2] - e[2]), "tilt": float(tilt),
            "yaw": float(yaw)}


def rotation_deg(a, b) -> float:
    """Return the rotation angle in deg between two poses."""
    rel = _rot(a).T @ _rot(b)
    return float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))))


def _check(rep: GateReport, name: str, value: float, limit: float) -> None:
    rep.values[name] = round(float(value), 4)
    if not np.isfinite(value) or abs(value) > limit:
        rep.failed.append(name)


WORLD_UP = (0.0, 0.0, 1.0)


def placed_error(actual, expected, up=WORLD_UP) -> dict:
    """Return the pose error of a placed part, measured in the world frame."""
    a, e = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    d = _rot(a) @ _rot(e).T
    u = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    tilt = np.degrees(np.arccos(np.clip(u @ d @ u, -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(d[1, 0], d[0, 0]))
    return {"xy": float(np.hypot(*(a[:2] - e[:2]))), "dz": float(a[2] - e[2]), "tilt": float(tilt),
            "yaw": float(yaw)}


def gate_final_pose(rep: GateReport, final_pose, expected_pose, lim: GateLimits, tilt_only: bool = False,
                    up=WORLD_UP) -> None:
    """Check a placed part's final pose against the gate limits."""
    e = placed_error(final_pose, expected_pose, up)
    _check(rep, "final_tilt_deg", e["tilt"], lim.tilt_deg)
    if tilt_only:
        return
    _check(rep, "final_xy_mm", e["xy"], lim.xy_mm)
    _check(rep, "final_yaw_deg", e["yaw"], lim.yaw_deg)
    _check(rep, "final_dz_mm", e["dz"], lim.z_mm)


def gate_forces(rep: GateReport, contact_trace: np.ndarray, limits: ForceLimits) -> None:
    """Check the peak contact forces against the force limits."""
    c = np.asarray(contact_trace, dtype=np.float64)
    peak = np.where(np.isfinite(c), np.abs(c), np.inf).max(axis=0)
    for name, v, lim in zip(("peak_axial_N", "peak_lateral_N", "peak_torque_Nm"), peak, limits.as_array()):
        _check(rep, name, v, lim)


def gate_others(rep: GateReport, pose_traces: dict[str, np.ndarray], target: str, lim: GateLimits) -> None:
    """Every non-target object: max displacement / turn from its first pose over the whole episode."""
    move, turn, worst = 0.0, 0.0, None
    for uid, trace in pose_traces.items():
        if uid == target:
            continue
        t = np.asarray(trace, dtype=np.float64)
        d = float(np.linalg.norm(t[:, :3] - t[0, :3], axis=1).max())
        a = max(rotation_deg(t[0], p) for p in [*t[:: max(1, len(t) // 50)], t[-1]])
        if d > move or a > turn:
            worst = uid
        move, turn = max(move, d), max(turn, a)
    rep.values["others_worst"] = worst
    _check(rep, "others_move_mm", move, lim.other_move_mm)
    _check(rep, "others_turn_deg", turn, lim.other_turn_deg)


def gate_touch(rep: GateReport, gripped: list, target: str) -> None:
    """The fingers held nothing but the target at every step."""
    wrong = sorted({g for g in gripped if g not in (None, target)})
    rep.values["fingers_touched"] = wrong
    if wrong:
        rep.failed.append("fingers_touched_non_target")


def gate_grip(rep: GateReport, tcp_in_obj, planned_tcp_in_obj, lim: GateLimits) -> None:
    """TCP expressed in the target's frame right after the grasp, against the planner's CAD grip (both 6-vectors)."""
    e = pose_error(tcp_in_obj, planned_tcp_in_obj)
    _check(rep, "grip_pos_mm", float(np.hypot(e["xy"], e["dz"])), lim.grip_pos_mm)
    _check(rep, "grip_roll_deg", e["yaw"], lim.grip_roll_deg)


def finish(rep: GateReport) -> GateReport:
    """Set the report's pass flag and return the report."""
    rep.ok = not rep.failed
    return rep
