"""UF0 TCP pose -> ER-4iA joint angles with the starter's damped least-squares IK. numpy only.

Shared by the renderer (`sdg/isaac/generate.py`) and the scene sampler, which only emits poses `reachable` accepts:
the renderer would otherwise stop a whole chunk at the first pose with no solution.
"""

from __future__ import annotations

import functools
import math
import sys
from pathlib import Path
from unittest import mock

import numpy as np

UF0_ABOVE_BASE_M = 0.330   # UF0 origin above the robot base frame
STARTER = Path(__file__).resolve().parents[1] / "sim" / "starter"
ISAAC_ONLY = ("pxr", "omni", "omni.usd", "isaacsim", "isaacsim.core", "isaacsim.core.utils",
              "isaacsim.core.utils.stage", "isaacsim.storage", "isaacsim.storage.native")


def fanuc_rotation(w: float, p: float, r: float) -> np.ndarray:
    """FANUC fixed-XYZ angles (deg): W about X, then P about Y, then R about Z."""
    return _rz(r) @ _ry(p) @ _rx(w)


def solve_tcp(ik, tcp: list[float], home: np.ndarray, park: np.ndarray, limits: np.ndarray,
              tool_offset: tuple) -> np.ndarray | None:
    """UF0 [x y z w p r] (mm/deg) -> joints, or None if no seed converges.

    Damped least squares is local, so it is seeded facing the target (J1 = atan2(y, x)) before home and park.
    """
    x, y, z, w, p, r = tcp
    rot = fanuc_rotation(w, p, r)
    target = np.array([x, y, z]) / 1000.0 + (0.0, 0.0, UF0_ABOVE_BASE_M)
    facing = home.copy()
    facing[0] = math.atan2(y, x)
    for seed in (facing, home, park):
        q, ok = ik.solve(seed, target, rot @ (0, 0, 1), limits, tool_offset=tool_offset,
                         spread_dir=rot @ (0, 1, 0), iters=400)
        if ok:
            return q
    return None


def reachable(tcp: list[float]) -> bool:
    """True if the render host's IK will solve this TCP (same solver, limits, seeds and tool length)."""
    ik, robot = _starter_robot()
    return solve_tcp(ik, tcp, **robot) is not None


@functools.cache
def _starter_robot():
    """The starter's IK and robot numbers outside Isaac. `cell.py` imports USD modules it does not need for the
    gripper geometry, so those are stubbed (only when absent: inside Isaac the real ones stay)."""
    for name in ISAAC_ONLY:
        sys.modules.setdefault(name, mock.MagicMock())
    if str(STARTER) not in sys.path:
        sys.path.insert(0, str(STARTER))
    import cell
    import ik
    import scene_config as C

    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    deg = lambda poses: np.deg2rad([poses[f"J{i}"] for i in range(1, 7)])
    return ik, {"home": deg(C.HOME_POSE_DEG), "park": deg(C.CAPTURE_POSE_DEG), "limits": deg(C.JOINT_LIMITS_DEG),
                "tool_offset": tuple(np.array(C.TOOL_AXIS) * cell.tip_offset())}


def _rx(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
