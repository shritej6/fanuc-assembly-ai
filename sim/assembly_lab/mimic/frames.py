"""Vectorised pose conversions between sim units (m, wxyz quaternion) and API units (mm, FANUC W/P/R).
Authored by Shritej Gunda.
"""

from __future__ import annotations

import numpy as np

# Fingers straight down are w = -180. A plain atan2 flips between +180 and -180 on tiny noise, which wrecks the
# GR00T normalisation stats; w is therefore stored continuous in [W_LOW, W_LOW + 360). The deploy adapter
# wraps it back to (-180, 180] before act_tcp (the cell accepts either; same rotation).
W_LOW = -270.0


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz -> (N, 3, 3). Normalises first."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def rot_to_wpr(rot: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> (N, 3) degrees, w wrapped into [W_LOW, W_LOW + 360)."""
    rot = np.asarray(rot, dtype=np.float64)
    p = np.arcsin(np.clip(-rot[..., 2, 0], -1.0, 1.0))
    w = np.arctan2(rot[..., 2, 1], rot[..., 2, 2])
    r = np.arctan2(rot[..., 1, 0], rot[..., 0, 0])
    gimbal = np.abs(rot[..., 2, 0]) > 1.0 - 1e-9
    w = np.where(gimbal, 0.0, w)
    r = np.where(gimbal, np.arctan2(-rot[..., 0, 1], rot[..., 1, 1]), r)
    out = np.degrees(np.stack([w, p, r], -1))
    out[..., 0] = (out[..., 0] - W_LOW) % 360.0 + W_LOW
    return out


def wpr_to_rot(wpr: np.ndarray) -> np.ndarray:
    """Return rotation matrices for FANUC W/P/R angles in deg."""
    a = np.radians(np.asarray(wpr, dtype=np.float64))
    cw, sw, cp, sp, cr, sr = np.cos(a[..., 0]), np.sin(a[..., 0]), np.cos(a[..., 1]), np.sin(a[..., 1]), \
        np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([
        np.stack([cr * cp, cr * sp * sw - sr * cw, cr * sp * cw + sr * sw], -1),
        np.stack([sr * cp, sr * sp * sw + cr * cw, sr * sp * cw - cr * sw], -1),
        np.stack([-sp, cp * sw, cp * cw], -1),
    ], -2)


def unwrap_deg(a: np.ndarray) -> np.ndarray:
    """Continuous angle along time (axis 0), so r never jumps 360 inside an episode."""
    return np.degrees(np.unwrap(np.radians(np.asarray(a, dtype=np.float64)), axis=0))


def sim_to_api_pose(pos_m: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """(N, 3) m + (N, 4) wxyz -> (N, 6) [x, y, z mm, w, p, r deg], r unwrapped along the episode."""
    wpr = rot_to_wpr(quat_to_rot(quat_wxyz))
    wpr[:, 2] = unwrap_deg(wpr[:, 2])
    return np.concatenate([np.asarray(pos_m, dtype=np.float64) * 1000.0, wpr], axis=1)


def yaw_deg_of(quat_wxyz: np.ndarray) -> np.ndarray:
    """Heading of the frame's x axis in the UF0 xy plane (deg, (-180, 180])."""
    rot = quat_to_rot(quat_wxyz)
    return np.degrees(np.arctan2(rot[..., 1, 0], rot[..., 0, 0]))
