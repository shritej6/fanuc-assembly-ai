"""Convert FANUC [x, y, z, w, p, r] to and from GR00T eef_9d (numpy only).
Authored by Shritej Gunda."""

from __future__ import annotations

import numpy as np


def wpr_to_matrix(wpr: np.ndarray) -> np.ndarray:
    """Return rotation matrices for w, p, r in degrees."""
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
    """Return both fixed-angle w, p, r solutions for rotation matrices."""
    m = np.asarray(m, dtype=np.float64)
    p = np.arcsin(np.clip(-m[..., 2, 0], -1.0, 1.0))
    w = np.arctan2(m[..., 2, 1], m[..., 2, 2])
    r = np.arctan2(m[..., 1, 0], m[..., 0, 0])
    first = np.degrees(np.stack([w, p, r], axis=-1))
    second = first + np.array([180.0, 0.0, 180.0])
    second[..., 1] = 180.0 - first[..., 1]
    return _wrap180(first), _wrap180(second)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Return rotation matrices from 6D rotations by Gram-Schmidt."""
    rows = np.asarray(rot6d, dtype=np.float64).reshape(np.shape(rot6d)[:-1] + (2, 3))
    a = rows[..., 0, :] / np.linalg.norm(rows[..., 0, :], axis=-1, keepdims=True)
    b = rows[..., 1, :] - np.sum(a * rows[..., 1, :], axis=-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], axis=-2)


def tcp_to_eef9d(tcp: np.ndarray) -> np.ndarray:
    """Return eef_9d rows for FANUC TCP poses."""
    tcp = np.asarray(tcp, dtype=np.float64)
    rot = wpr_to_matrix(tcp[..., 3:6])
    rot6d = rot[..., :2, :].reshape(tcp.shape[:-1] + (6,))
    return np.concatenate([tcp[..., :3], rot6d], axis=-1).astype(np.float32)


def eef9d_to_tcp(eef9d: np.ndarray, reference_wpr: np.ndarray) -> np.ndarray:
    """Return FANUC TCP rows for an eef_9d chunk."""
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
