"""Turn act_tcp rows into ER-4iA joint targets using the starter FK and damped least-squares IK.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from layout import UF0_ABOVE_BASE_M, fanuc_rotation, wpr_from_rotation

REPO = Path(__file__).resolve().parents[3]
STARTER = REPO / "sim" / "starter"
if str(STARTER) not in sys.path:
    sys.path.insert(0, str(STARTER))

import fk
import ik
import scene_config as C

TOOL_OFFSET = tuple(np.array(C.TOOL_AXIS) * C.TOOL_LENGTH)
# flange axes of the TCP frame's columns: tcp x = -flange z, tcp y = flange y (closing), tcp z = flange x (approach)
TCP_IN_FLANGE = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
LIMITS_RAD = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
# The official operator home (robot/cell/mock_cell.py HOME, J1..J6 deg, gripper open): every real rollout starts here
# (robot/cell/README section 5), so every sim episode does too. Tool axis straight down, flange at (0.300, 0, 0.533) m.
OPERATOR_HOME_DEG = {"J1": -0.0065, "J2": 3.2909, "J3": -11.8967, "J4": -0.0007, "J5": -74.8218, "J6": -0.0286}
HOME_RAD = np.deg2rad([OPERATOR_HOME_DEG[f"J{i}"] for i in range(1, 7)])
PARK_RAD = np.deg2rad([C.CAPTURE_POSE_DEG[f"J{i}"] for i in range(1, 7)])
WORKSPACE = {"x": (150.0, 650.0), "y": (-450.0, 450.0), "z": (-334.0, 300.0)}   # PLAN 3.1, mm UF0

ik.set_tool_axis(C.TOOL_AXIS)
ik.set_spread_axis(C.FINGER_SPREAD_AXIS)


class TcpError(ValueError):
    """A TCP row is outside the workspace box, malformed, or has no IK solution."""


def tcp_pose(q_rad: np.ndarray) -> list[float]:
    """Joint angles (rad, J1..J6) -> UF0 [x, y, z, w, p, r] mm / deg."""
    rot, p = fk.tool_pose(np.asarray(q_rad, float), TOOL_OFFSET)
    xyz = (np.asarray(p) - (0.0, 0.0, UF0_ABOVE_BASE_M)) * 1000.0
    return [*map(float, xyz), *wpr_from_rotation(rot @ TCP_IN_FLANGE)]


def check_row(row) -> np.ndarray:
    row = np.asarray(row, dtype=float)
    if row.shape != (7,) or not np.all(np.isfinite(row)):
        raise TcpError(f"act_tcp row must be 7 finite numbers, got {row!r}")
    for k, axis in enumerate("xyz"):
        lo, hi = WORKSPACE[axis]
        if not lo <= row[k] <= hi:
            raise TcpError(f"TCP {axis} = {row[k]:.1f} mm outside the workspace box [{lo}, {hi}]")
    return row


def solve(tcp6, seed_rad: np.ndarray | None = None, tol_m: float = 5e-5) -> np.ndarray:
    """UF0 [x y z w p r] -> joints (rad). Seeds: the current joints first (smooth paths), then the
    sdg seeds (facing the target, home, park). Raises TcpError when none converges."""
    x, y, z, w, p, r = (float(v) for v in tcp6)
    rot = fanuc_rotation(w, p, r)
    target = np.array([x, y, z]) / 1000.0 + (0.0, 0.0, UF0_ABOVE_BASE_M)
    facing = HOME_RAD.copy()
    facing[0] = math.atan2(y, x)
    seeds = ([np.asarray(seed_rad, float)] if seed_rad is not None else []) + [facing, HOME_RAD, PARK_RAD]
    for s in seeds:
        q, ok = ik.solve(s, target, rot @ (0, 0, 1), LIMITS_RAD, tool_offset=TOOL_OFFSET,
                         spread_dir=rot @ (0, 1, 0), iters=400, tol=tol_m)
        # The starter's solver also reports "ok" for a 5 mm miss after its last iteration: re-check here.
        if ok:
            pos_err, rot_err = _pose_error(q, target, rot)
            if pos_err < MAX_POS_ERR_M and rot_err < MAX_ROT_ERR_RAD:
                # unwrap against the CURRENT joints: a fallback seed (HOME J6 = 0) would pick the far turn
                return _unwrap_j6(q, seeds[0])
    raise TcpError(f"no IK solution for TCP {list(tcp6)}")


MAX_POS_ERR_M = 3e-4
MAX_ROT_ERR_RAD = math.radians(0.2)


def _pose_error(q: np.ndarray, target: np.ndarray, rot_des: np.ndarray) -> tuple[float, float]:
    rot, p = fk.tool_pose(q, TOOL_OFFSET)
    rel = (rot @ TCP_IN_FLANGE).T @ rot_des
    ang = math.acos(max(-1.0, min(1.0, (np.trace(rel) - 1) / 2)))
    return float(np.linalg.norm(np.asarray(p) - target)), ang


def _unwrap_j6(q: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """J6 has +-360 deg: keep the turn nearest the seed so a small r change never spins the wrist."""
    q = q.copy()
    lo, hi = LIMITS_RAD[5]
    for k in (-1, 1):
        alt = q[5] + k * 2 * math.pi
        if lo <= alt <= hi and abs(alt - seed[5]) < abs(q[5] - seed[5]):
            q[5] = alt
    return q


def _slerp_rot(r0: np.ndarray, r1: np.ndarray, t: float) -> np.ndarray:
    rel = r0.T @ r1
    angle = math.acos(max(-1.0, min(1.0, (np.trace(rel) - 1) / 2)))
    if angle < 1e-9:
        return r0
    axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]) / (2 * math.sin(angle))
    if abs(math.pi - angle) < 1e-6:     # 180 deg: axis from the symmetric part
        axis = np.sqrt(np.clip((np.diag(rel) + 1) / 2, 0, 1))
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = angle * t
    return r0 @ (np.eye(3) + math.sin(a) * kx + (1 - math.cos(a)) * kx @ kx)


def interpolate(tcp_from, tcp_to, n: int) -> list[list[float]]:
    """n poses from just after tcp_from to tcp_to: straight line in position, slerp in orientation."""
    a, b = np.asarray(tcp_from[:6], float), np.asarray(tcp_to[:6], float)
    r0, r1 = fanuc_rotation(*a[3:]), fanuc_rotation(*b[3:])
    out = []
    for k in range(1, n + 1):
        t = k / n
        out.append([*(a[:3] + (b[:3] - a[:3]) * t), *wpr_from_rotation(_slerp_rot(r0, r1, t))])
    return out
