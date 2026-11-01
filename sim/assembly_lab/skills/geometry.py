"""Pose maths for the skills in mm and degrees.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import math

import numpy as np

Pose = np.ndarray  # shape (6,): x, y, z mm and w, p, r deg


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wpr_to_rot(w: float, p: float, r: float) -> np.ndarray:
    return _rz(math.radians(r)) @ _ry(math.radians(p)) @ _rx(math.radians(w))


def rot_to_wpr(rot: np.ndarray) -> tuple[float, float, float]:
    p = math.asin(max(-1.0, min(1.0, -rot[2, 0])))
    if abs(math.cos(p)) > 1e-9:
        w = math.atan2(rot[2, 1], rot[2, 2])
        r = math.atan2(rot[1, 0], rot[0, 0])
    else:  # gimbal lock: fold everything into r
        w = 0.0
        r = math.atan2(-rot[0, 1], rot[1, 1])
    return math.degrees(w), math.degrees(p), math.degrees(r)


def pose_to_matrix(pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    t = np.eye(4)
    t[:3, :3] = wpr_to_rot(*pose[3:6])
    t[:3, 3] = pose[:3]
    return t


def matrix_to_pose(t: np.ndarray) -> Pose:
    return np.array([*t[:3, 3], *rot_to_wpr(t[:3, :3])])


def to_matrix_m(pose) -> np.ndarray:
    """4x4 pose in metres, the Isaac Lab convention used only inside the HDF5 datagen info."""
    t = pose_to_matrix(pose)
    t[:3, 3] /= 1000.0
    return t


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def angle_between_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = rot_a.T @ rot_b
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rel) - 1.0) / 2.0))))


def tilt_deg(pose) -> float:
    """Angle between the pose's local Z axis and straight down (or straight up: the sign of Z is ignored)."""
    z = wpr_to_rot(*np.asarray(pose, dtype=float)[3:6])[:, 2]
    return math.degrees(math.acos(min(1.0, abs(z[2]))))


def axis_angle(axis, angle_deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    th = math.radians(angle_deg)
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * (k @ k)


def tilted(pose, lead_dir_deg: float, tilt: float, twist: float = 0.0) -> Pose:
    """Pose tilted about the TCP so the side facing `lead_dir_deg` (base-frame heading) goes down by `tilt` deg."""
    pose = np.asarray(pose, dtype=float)
    u = np.array([math.cos(math.radians(lead_dir_deg)), math.sin(math.radians(lead_dir_deg)), 0.0])
    axis = np.cross([0.0, 0.0, 1.0], u)  # rotating +tilt about z x u lowers the +u side
    rot = axis_angle([0, 0, 1], twist) @ axis_angle(axis, tilt) @ wpr_to_rot(*pose[3:6])
    return np.array([*pose[:3], *rot_to_wpr(rot)])


def interpolate(a, b, frac: float) -> Pose:
    """Straight line in position, shortest-arc slerp in orientation."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ra, rb = wpr_to_rot(*a[3:6]), wpr_to_rot(*b[3:6])
    rel = ra.T @ rb
    ang = math.acos(max(-1.0, min(1.0, (np.trace(rel) - 1.0) / 2.0)))
    if ang < 1e-9:
        rot = ra
    else:
        axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
        if np.linalg.norm(axis) < 1e-9:  # 180 deg: any perpendicular axis works
            axis = np.linalg.svd(rel - np.eye(3))[2][-1]
        rot = ra @ axis_angle(axis, math.degrees(ang) * frac)
    pos = a[:3] + (b[:3] - a[:3]) * frac
    wpr = np.array(rot_to_wpr(rot))
    # keep the angles continuous with `a` (avoid +-180 flips between rows)
    wpr = a[3:6] + np.array([wrap_deg(v) for v in (wpr - a[3:6])])
    return np.array([*pos, *wpr])


def rotation_distance_deg(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return angle_between_deg(wpr_to_rot(*a[3:6]), wpr_to_rot(*b[3:6]))
