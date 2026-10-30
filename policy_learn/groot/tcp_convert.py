"""FANUC `[x, y, z, w, p, r]` <-> GR00T `eef_9d` = `[x, y, z, rot6d]` (pure numpy, no GR00T import).

Why: w sits at +-180 when the fingers point down, so raw WPR jumps by 360 deg between equal poses. Min-max
normalisation and a regression head cannot learn across that jump. GR00T's own EEF format (XYZ_ROT6D, as in its
DROID pre-training) is continuous, so the model sees rot6d and we convert back to WPR only at the client.

Conventions (identical to `policy_learn/deploy/guards.py`):
- WPR are FANUC fixed angles in degrees: R = Rz(r) @ Ry(p) @ Rx(w).
- rot6d = the first two ROWS of R, flattened (`gr00t/data/state_action/pose.py::_matrix_to_rot6d`).
- Translation stays in mm.
"""

from __future__ import annotations

import numpy as np


def wpr_to_matrix(wpr: np.ndarray) -> np.ndarray:
    """(..., 3) degrees -> (..., 3, 3)."""
    w, p, r = np.moveaxis(np.radians(np.asarray(wpr, dtype=np.float64)), -1, 0)
    cw, sw, cp, sp, cr, sr = np.cos(w), np.sin(w), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    m = np.empty(w.shape + (3, 3))
    m[..., 0, 0] = cr * cp
    m[..., 0, 1] = cr * sp * sw - sr * cw
    m[..., 0, 2] = cr * sp * cw + sr * sw
    m[..., 1, 0] = sr * cp
    m[..., 1, 1] = sr * sp * sw + cr * cw
    m[..., 1, 2] = sr * sp * cw - cr * sw
    m[..., 2, 0] = -sp
    m[..., 2, 1] = cp * sw
    m[..., 2, 2] = cp * cw
    return m


def matrix_to_wpr_pair(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both fixed-angle solutions (degrees) of (..., 3, 3): (w, p, r) and (w+180, 180-p, r+180), wrapped to +-180."""
    m = np.asarray(m, dtype=np.float64)
    p = np.arcsin(np.clip(-m[..., 2, 0], -1.0, 1.0))
    w = np.arctan2(m[..., 2, 1], m[..., 2, 2])
    r = np.arctan2(m[..., 1, 0], m[..., 0, 0])
    first = np.degrees(np.stack([w, p, r], axis=-1))
    second = first + np.array([180.0, 0.0, 180.0])
    second[..., 1] = 180.0 - first[..., 1]
    return _wrap180(first), _wrap180(second)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) by Gram-Schmidt on the two rows (same as GR00T's `_rot6d_to_matrix`)."""
    rows = np.asarray(rot6d, dtype=np.float64).reshape(np.shape(rot6d)[:-1] + (2, 3))
    a = rows[..., 0, :] / np.linalg.norm(rows[..., 0, :], axis=-1, keepdims=True)
    b = rows[..., 1, :] - np.sum(a * rows[..., 1, :], axis=-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], axis=-2)


def tcp_to_eef9d(tcp: np.ndarray) -> np.ndarray:
    """(..., 6) `[x, y, z, w, p, r]` (mm, deg) -> (..., 9) `[x, y, z, r00, r01, r02, r10, r11, r12]` float32."""
    tcp = np.asarray(tcp, dtype=np.float64)
    rot = wpr_to_matrix(tcp[..., 3:6])
    rot6d = rot[..., :2, :].reshape(tcp.shape[:-1] + (6,))
    return np.concatenate([tcp[..., :3], rot6d], axis=-1).astype(np.float32)


def eef9d_to_tcp(eef9d: np.ndarray, reference_wpr: np.ndarray) -> np.ndarray:
    """(T, 9) chunk -> (T, 6) `[x, y, z, w, p, r]`.

    Each row takes the WPR solution closest to the previous row (the first row: to `reference_wpr`, normally the
    current `tcp_pose[3:]`), and every angle is unwrapped by multiples of 360 to be continuous, so the arm never
    gets a +-360 spin on J6 or a w flip between -180 and +180. Whether r stays inside J6 is the guard's job.
    """
    eef9d = np.atleast_2d(np.asarray(eef9d, dtype=np.float64))
    first, second = matrix_to_wpr_pair(rot6d_to_matrix(eef9d[:, 3:9]))
    out = np.empty((len(eef9d), 6))
    out[:, :3] = eef9d[:, :3]
    prev = np.asarray(reference_wpr, dtype=np.float64)
    for i in range(len(eef9d)):
        candidates = [prev + _wrap180(sol[i] - prev) for sol in (first, second)]
        prev = min(candidates, key=lambda c: float(np.sum(np.abs(c - prev))))
        out[i, 3:] = prev
    return out


def _wrap180(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + 180.0) % 360.0 - 180.0
