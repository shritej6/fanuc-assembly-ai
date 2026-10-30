"""Success checks on the sim ground truth (`obs["sim"]["object_poses"]`, mm / deg).

These are the scripted-demo success criteria. They read sim truth because the script is a teacher: the policy never
sees these values, and the real cell repeats the checks with cameras (docs/ROBOT_BRAIN.md section 1 step 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .gates import SEAT_TILT_DEG, SEAT_XY_MM, SEAT_YAW_DEG, SEAT_Z_MM, WORLD_UP, axis_tilt_deg, yaw_err_deg
from .geometry import matrix_to_pose, pose_to_matrix

GRASP_FOLLOW_FRAC = 0.8  # the object must rise at least 80 % of the check lift
GRASP_SLIP_XY_MM = 2.0  # and stay within 2 mm sideways of where it was relative to the fingers


@dataclass(frozen=True)
class SeatTolerance:
    z_mm: float = SEAT_Z_MM  # the precision gates (INTERFACES.md): also the retry decision
    xy_mm: float = SEAT_XY_MM
    tilt_deg: float = SEAT_TILT_DEG
    yaw_deg: float = SEAT_YAW_DEG


SEAT_TOL = SeatTolerance()
DEFECT_RADIUS_MM = 40.0  # a dropped defect must end up inside the defect area around the drop point


def object_pose(obs: dict, oid: str) -> np.ndarray:
    return np.asarray(obs["sim"]["object_poses"][oid], dtype=float)


def tcp_pose(obs: dict) -> np.ndarray:
    return np.asarray(obs["tcp_pose"], dtype=float)[:6]


def in_hand(obs: dict, oid: str) -> np.ndarray:
    """Object pose in the TCP frame (4x4, mm), measured from sim truth after the grasp."""
    return np.linalg.inv(pose_to_matrix(tcp_pose(obs))) @ pose_to_matrix(object_pose(obs, oid))


def expected_object(tcp, t_tcp_obj: np.ndarray) -> np.ndarray:
    return matrix_to_pose(pose_to_matrix(tcp) @ t_tcp_obj)


def grasp_ok(before: dict, after: dict, oid: str) -> tuple[bool, str]:
    lift = tcp_pose(after)[2] - tcp_pose(before)[2]
    rise = object_pose(after, oid)[2] - object_pose(before, oid)[2]
    rel_b = object_pose(before, oid)[:2] - tcp_pose(before)[:2]
    rel_a = object_pose(after, oid)[:2] - tcp_pose(after)[:2]
    slip = float(np.linalg.norm(rel_a - rel_b))
    if lift <= 0:
        return False, f"no check lift ({lift:.2f} mm)"
    if rise < GRASP_FOLLOW_FRAC * lift:
        return False, f"{oid} rose {rise:.2f} of {lift:.2f} mm: not in the fingers"
    if slip > GRASP_SLIP_XY_MM:
        return False, f"{oid} slipped {slip:.2f} mm in the fingers"
    return True, "ok"


def seat_error(actual, seated, up=WORLD_UP) -> dict[str, float]:
    a, s = np.asarray(actual, dtype=float), np.asarray(seated, dtype=float)
    return {"dx": a[0] - s[0], "dy": a[1] - s[1], "dz": a[2] - s[2],
            "xy": float(np.hypot(a[0] - s[0], a[1] - s[1])),
            "tilt": axis_tilt_deg(a, s, up),
            "yaw": yaw_err_deg(a, s)}


def seated_ok(actual, seated, tol: SeatTolerance = SEAT_TOL, up=WORLD_UP) -> tuple[bool, dict[str, float]]:
    e = seat_error(actual, seated, up)
    ok = abs(e["dz"]) <= tol.z_mm and e["xy"] <= tol.xy_mm and e["tilt"] <= tol.tilt_deg and abs(e["yaw"]) <= tol.yaw_deg
    return ok, e


def in_defect_area(actual, drop_tcp) -> tuple[bool, str]:
    d = math.hypot(actual[0] - drop_tcp[0], actual[1] - drop_tcp[1])
    return (d <= DEFECT_RADIUS_MM, f"{d:.1f} mm from the drop point")
