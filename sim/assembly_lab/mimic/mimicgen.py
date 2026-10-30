"""MimicGen core (the algorithm of Isaac Lab Mimic / MimicGen, Mandlekar et al. 2023) over act_tcp rows in mm / deg.

Agent 1's AssemblyCellEnv is a single-env SimulationContext wrapper (reset(layout) / step_tcp(rows, dt)), not the
ManagerBasedRLMimicEnv that Isaac Lab's annotate_demos.py / generate_dataset.py need, so generation drives the env
directly with the same steps Isaac Lab Mimic uses:
  1. split each seed demo into subtask segments at its subtask term signals (obs/datagen_info/subtask_term_signals);
  2. per subtask, pick a source segment whose object_ref frame pose is nearest the new scene's (nearest_neighbor_object,
     k random among the nearest);
  3. transform the segment rigidly: T_new_frame @ inv(T_src_frame) @ T_row (the frame pose at the subtask start);
  4. interpolate from the current TCP target to the first transformed row, add per-row action noise;
  5. execute, then keep the demo only if it succeeds and passes every precision gate (gates.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

from .frames import rot_to_wpr, wpr_to_rot

EEF = "er4ia"


def pose_to_matrix(pose) -> np.ndarray:
    p = np.asarray(pose, dtype=np.float64)
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = wpr_to_rot(p[3:6]), p[:3]
    return t


def matrix_to_pose(t: np.ndarray) -> np.ndarray:
    return np.array([*t[:3, 3], *rot_to_wpr(t[:3, :3])])


@dataclass(frozen=True)
class SourceSegment:
    demo: str
    object_id: str
    product: int
    subtask: int
    rows: np.ndarray          # (n, 7) act_tcp rows, mm / deg / gripper
    frame_pose: np.ndarray    # (4, 4) mm: the subtask's object_ref frame at the segment start


def _boundaries(signals: list[np.ndarray], n: int) -> list[tuple[int, int]]:
    """Subtask i spans [end_{i-1}, end_i); end_i = first step its signal is on; the last runs to the episode end."""
    ends = []
    for sig in signals:
        on = np.flatnonzero(np.asarray(sig, dtype=bool))
        if not on.size:
            raise ValueError("a subtask term signal never turns on")
        ends.append(int(on[0]))
    if ends != sorted(ends):
        raise ValueError(f"subtask term signals out of order: {ends}")
    starts = [0, *ends]
    return list(zip(starts, [*ends, n]))


def load_segments(path: str, signals: list[str], refs: list[str]) -> list[list[SourceSegment]]:
    """Per subtask index, every source segment in a seed file (agent 3's HDF5 layout, skills/hdf5_io.py)."""
    out: list[list[SourceSegment]] = [[] for _ in refs]
    with h5py.File(path, "r") as f:
        for key in sorted(f["data"].keys()):
            d = f["data"][key]
            dg = d["obs/datagen_info"]
            rows = d["actions"][()].astype(np.float64)
            spans = _boundaries([dg[f"subtask_term_signals/{s}"][()] for s in signals], len(rows))
            oid = str(d.attrs.get("object_id", ""))
            for i, ((a, b), ref) in enumerate(zip(spans, refs)):
                frame = dg[f"object_pose/{ref}"][a].astype(np.float64)
                frame[:3, 3] *= 1000.0  # stored in metres (Mimic convention), rows are mm
                out[i].append(SourceSegment(key, oid, int(d.attrs.get("product", 0)), i, rows[a:b], frame))
    return out


def _frame_distance(a: np.ndarray, b: np.ndarray, mm_per_rad: float = 50.0) -> float:
    rel = a[:3, :3].T @ b[:3, :3]
    ang = np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]) + mm_per_rad * ang)


def select_source(cands: list[SourceSegment], new_frame: np.ndarray, object_id: str, k: int,
                  rng: np.random.Generator) -> SourceSegment:
    """nearest_neighbor_object among the same part's segments (else any), random among the k nearest."""
    same = [c for c in cands if c.object_id == object_id] or cands
    if not same:
        raise ValueError("no source segment for this subtask")
    order = sorted(same, key=lambda c: _frame_distance(c.frame_pose, new_frame))
    return order[int(rng.integers(0, min(k, len(order))))]


def transform_rows(rows: np.ndarray, src_frame: np.ndarray, new_frame: np.ndarray) -> np.ndarray:
    """Rigid object-centric transform of TCP rows; gripper column unchanged; r kept continuous with the source."""
    delta = new_frame @ np.linalg.inv(src_frame)
    out = rows.copy()
    for i, row in enumerate(rows):
        out[i, :6] = matrix_to_pose(delta @ pose_to_matrix(row[:6]))
    out[:, 5] = np.degrees(np.unwrap(np.radians(out[:, 5])))
    return out


def _slerp_r(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    rel = a.T @ b
    ang = np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    if ang < 1e-9:
        return a.copy()
    axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]) / (2 * np.sin(ang))
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    th = s * ang
    return a @ (np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * kx @ kx)


def interpolate(start_row: np.ndarray, end_row: np.ndarray, steps: int) -> np.ndarray:
    """`steps` rows strictly between start and end (linear position, slerp rotation, start gripper)."""
    if steps <= 0:
        return np.zeros((0, 7))
    ra, rb = wpr_to_rot(start_row[3:6]), wpr_to_rot(end_row[3:6])
    out = []
    for s in np.arange(1, steps + 1) / (steps + 1):
        pos = (1 - s) * start_row[:3] + s * end_row[:3]
        out.append([*pos, *rot_to_wpr(_slerp_r(ra, rb, s)), start_row[6]])
    rows = np.array(out)
    rows[:, 5] = np.degrees(np.unwrap(np.radians(np.r_[start_row[5], rows[:, 5]])))[1:]
    return rows


def add_noise(rows: np.ndarray, pos_mm: float, rot_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise on position (mm) and a small random rotation (deg) per row; gripper untouched."""
    out = rows.copy()
    if pos_mm <= 0 and rot_deg <= 0:
        return out
    for i in range(len(out)):
        out[i, :3] += pos_mm * rng.standard_normal(3)
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        ang = np.radians(rot_deg * rng.standard_normal())
        kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        noise = np.eye(3) + np.sin(ang) * kx + (1 - np.cos(ang)) * kx @ kx
        wpr = rot_to_wpr(noise @ wpr_to_rot(out[i, 3:6]))
        wpr[2] = out[i, 5] + ((wpr[2] - out[i, 5] + 180.0) % 360.0 - 180.0)
        out[i, 3:6] = wpr
    return out
