"""MimicGen core (the algorithm of Isaac Lab Mimic / MimicGen, Mandlekar et al. 2023) over act_tcp rows in mm / deg.

Source demos are agent 3's kept episodes, written by the Isaac Lab recorder (skills/isaac_backend.py) and annotated
with agent 3's metadata (marks, kept, case, seat_tries). Generation drives the same env through agent 3's
EpisodeRecorder (generate.py), with the steps Isaac Lab Mimic uses:
  1. split each source demo into subtask segments at its subtask marks (attrs `marks`: signal -> first step after);
  2. per subtask, pick a source segment whose reference frame pose is nearest the new scene's (nearest_neighbor_object,
     k random among the nearest);
  3. transform the segment rigidly: T_new_frame @ inv(T_src_frame) @ T_row (the frame pose at the subtask start). Held
     insert subtasks use the part's planned seated pose as the frame and absorb the in-hand difference
     (transform_held), so the part, not just the TCP, repeats the source path into the 0.5 mm clearance bore;
  4. connect from the current TCP target to the first transformed row, add per-row action noise;
  5. execute, then keep the demo only if it succeeds and passes every precision gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import h5py
import numpy as np

from .frames import rot_to_wpr, wpr_to_rot

EEF = "er4ia"
FRAME_NAMES = ("grasp_frame", "seat_frame", "spot_frame")


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
    frame_pose: np.ndarray    # (4, 4) mm: the subtask's reference frame at the segment start (see demo_segments)
    in_hand: np.ndarray | None = None  # (4, 4) mm: held part in the TCP frame after the grasp (CAD local frame)


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


def quat_xyzw_to_rot(q) -> np.ndarray:
    """Isaac Lab 3.x quaternion order (x, y, z, w), as the recorder writes obs/*_quat and states/*/root_pose."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def frame_mm(pos_m, quat_xyzw) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = quat_xyzw_to_rot(quat_xyzw), np.asarray(pos_m, dtype=np.float64) * 1000.0
    return t


def seat_object_frame(seat_frame: np.ndarray, grasp_frame: np.ndarray, part_start: np.ndarray) -> np.ndarray:
    """The part's planned seated pose (4x4 mm): seat_frame is the seated TCP for the planned grip, and the planned grip
    (part -> TCP) is inv(part_start) @ grasp_frame. Held insert subtasks are transformed in this frame, so the part
    repeats the source's path relative to its seat whatever grip pad or roll the new plan uses."""
    return seat_frame @ np.linalg.inv(grasp_frame) @ part_start


@dataclass(frozen=True)
class RecordedDemo:
    """One kept episode of agent 3's Isaac Lab recorder file (skills/isaac_backend.py + hdf5_io.annotate_...)."""
    key: str
    object_id: str
    product: int
    case: str
    marks: dict
    seat_tries: int
    actions: np.ndarray       # (T, 7) mm / deg / gripper
    tcp: np.ndarray           # (T, 6) obs/tcp_pose before each action
    frames: dict              # frame name -> (T, 4, 4) mm, UF0 (obs/<frame>_pos, _quat)
    part: np.ndarray          # (T + 1, 4, 4) mm, CAD local frame of the target before each action (+ after the last)
    line: int = 0             # 0-based line of this episode in <file>.meta.jsonl (agent 3's drop tags use it)
    carry_drift_mm: float = 0.0  # agent 3's in_hand_drift_mm["carry"]: worst slip of the held part in free carry
    pull_drift_mm: float = 0.0   # in_hand_drift_mm["pull"] (pull_out, newer metas): slip in the fingers during the pull


def _part_local_mm(d, oid: str, z_off_m: float, to_local) -> np.ndarray:
    """states/ are written after each step (measured: obs/tcp_pose[t] pairs with states[t - 1], in-hand constant to
    0.01 mm over a carry), initial_state before the first: stacked, row t is the pose observation t saw. World metres
    -> UF0 mm is a pure z shift (env/layout.world_m_to_uf0), measured per file from the grasp_frame marker."""
    rows = np.r_[d[f"initial_state/rigid_object/{oid}/root_pose"][()], d[f"states/rigid_object/{oid}/root_pose"][()]]
    return np.stack([to_local(oid, frame_mm(r[:3] - (0.0, 0.0, z_off_m), r[3:7])) for r in rows.astype(np.float64)])


def _truthy(v) -> bool:
    if isinstance(v, bytes):
        v = v.decode()
    return v.lower() in ("true", "1") if isinstance(v, str) else bool(v)


def _cad_local(uid: str, body_mm: np.ndarray) -> np.ndarray:
    from ..skills.frames import body_to_local
    from ..skills.geometry import matrix_to_pose as geo_pose
    from ..skills.geometry import pose_to_matrix as geo_matrix

    return geo_matrix(body_to_local(uid, geo_pose(body_mm)))


def read_recorded(path: str, to_local=None, kept_only: bool = True) -> list[RecordedDemo]:
    """Kept episodes of one recorder file. Agent 3's `kept` is the authority: the recorder's own `success` attr is
    overwritten at reset by the env's success termination term (measured false on 3 of 3 kept run10 demos). `to_local(uid, body_4x4_mm)` maps body
    poses into CAD local frames (default skills/frames.body_to_local, the frame every skill and gate uses)."""
    to_local = to_local or _cad_local
    out = []
    metas = _metas(path)
    with h5py.File(path, "r") as f:
        keys = [k for k in sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1])) if "actions" in f["data"][k]]
        for n, key in enumerate(keys):
            d = f["data"][key]
            a = dict(d.attrs)
            if "marks" not in a and n < len(metas):  # writer still open: agent 3 annotates attrs only at close
                a = _meta_attrs(metas[n], a, int(d["actions"].shape[0]))
            if "marks" not in a:
                continue
            if kept_only and not _truthy(a.get("kept", False)):
                continue
            oid, o = str(a["object_id"]), d["obs"]
            frames = {n: np.stack([frame_mm(p, q) for p, q in zip(o[f"{n}_pos"][()], o[f"{n}_quat"][()])])
                      for n in FRAME_NAMES if f"{n}_pos" in o}
            z_off = float(d["states/rigid_object/grasp_frame/root_pose"][0, 2] - o["grasp_frame_pos"][0, 2])
            tries = a.get("seat_tries", "[]")
            drift = json.loads(a.get("in_hand_drift_mm", "{}")) if isinstance(a.get("in_hand_drift_mm"), str) else {}
            out.append(RecordedDemo(key, oid, int(a.get("product", 0)), str(a.get("case", "")), json.loads(a["marks"]),
                                    len(json.loads(tries)) if isinstance(tries, str) else 0,
                                    d["actions"][()].astype(np.float64), o["tcp_pose"][()].astype(np.float64)[:, :6],
                                    frames, _part_local_mm(d, oid, z_off, to_local), n,
                                    float(drift.get("carry", 0.0)), float(drift.get("pull", 0.0))))
    return out


def _metas(path: str) -> list[dict]:
    """Agent 3's <file>.meta.jsonl: one line per non-empty recorder episode, in order (hdf5_io.annotate_...)."""
    from pathlib import Path

    p = Path(f"{path}.meta.jsonl")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def _meta_attrs(meta: dict, attrs: dict, steps: int) -> dict:
    """The attrs annotate_recorder_file would write, or none if the step counts disagree (never papered over)."""
    if int(meta.get("steps", -1)) != steps:
        return attrs
    out = dict(attrs)
    for k, v in meta.items():
        out[k] = v if isinstance(v, (int, float, str, bool)) else json.dumps(v)
    return out


def in_hand_at(demo: RecordedDemo, step: int) -> np.ndarray:
    return np.linalg.inv(pose_to_matrix(demo.tcp[step])) @ demo.part[step]


def demo_segments(demo: RecordedDemo, signals: list[str], refs: list[str], held_seat: list[bool]) -> list[SourceSegment]:
    """Split one demo at its subtask marks. Frame per subtask: the object_ref frame at the segment start, except held
    insert subtasks (held_seat), whose frame is the part's planned seated pose (seat_object_frame)."""
    n = len(demo.actions)
    sig = []
    for s in signals:
        v = np.zeros(n, dtype=bool)
        if s in demo.marks:
            v[min(int(demo.marks[s]), n - 1):] = True
        sig.append(v)
    spans = _boundaries(sig, n)
    hand = in_hand_at(demo, int(demo.marks["grasped"])) if "grasped" in demo.marks else None
    out = []
    for i, ((a, b), ref, held) in enumerate(zip(spans, refs, held_seat)):
        if held:
            frame = seat_object_frame(demo.frames["seat_frame"][a], demo.frames["grasp_frame"][0], demo.part[0])
        else:
            frame = demo.frames[ref][a]
        out.append(SourceSegment(demo.key, demo.object_id, demo.product, i, demo.actions[a:b], frame,
                                 hand if i > 0 else None))
    return out


MAX_CARRY_DRIFT_MM = 0.5  # agent 3 / main 2026-10-30: a source whose part slipped more in the fingers is not used


def drop_lines(path: str) -> set[int]:
    """<file>.drop.json: meta lines agent 3 tagged as bad sources (run10_drop_tags.jsonl, mapped per file)."""
    from pathlib import Path

    p = Path(f"{path}.drop.json")
    return set(json.loads(p.read_text())) if p.exists() else set()


def load_sources(paths, signals: list[str], refs: list[str], held_seat: list[bool],
                 cases: tuple[str, ...] | None = None, max_seat_tries: int | None = None,
                 to_local=None) -> tuple[list[list[SourceSegment]], dict]:
    """Per subtask index, every source segment of agent 3's kept recorder demos in `paths`; plus what was skipped."""
    out: list[list[SourceSegment]] = [[] for _ in refs]
    counts: dict = {"files": len(paths), "kept": 0, "used": 0, "skipped": {}}
    for path in paths:
        dropped = drop_lines(str(path))
        for demo in read_recorded(str(path), to_local):
            counts["kept"] += 1
            why = None
            if demo.line in dropped:
                why = "drop tag"
            elif demo.carry_drift_mm > MAX_CARRY_DRIFT_MM:
                why = "carry drift"
            elif demo.pull_drift_mm > MAX_CARRY_DRIFT_MM:  # agent 3 2026-10-30: the same limit for the pull itself
                why = "pull drift"
            elif cases is not None and demo.case not in cases:
                why = f"case {demo.case}"
            elif max_seat_tries is not None and demo.seat_tries > max_seat_tries:
                why = "seat retry"
            elif any(s not in demo.marks for s in signals):
                why = "missing marks"
            if why:
                counts["skipped"][why] = counts["skipped"].get(why, 0) + 1
                continue
            for i, seg in enumerate(demo_segments(demo, signals, refs, held_seat)):
                out[i].append(replace(seg, demo=f"{path}:{demo.key}"))
            counts["used"] += 1
    return out, counts


def transform_held(rows: np.ndarray, src_frame: np.ndarray, new_frame: np.ndarray, src_in_hand: np.ndarray,
                   new_in_hand: np.ndarray) -> np.ndarray:
    """Rows of a subtask that carries the part: the PART repeats the source part's path relative to the frame, so the
    TCP absorbs the difference of the two in-hand poses: T_new = D @ T_src @ H_src @ inv(H_new), D = F_new inv(F_src)."""
    delta = new_frame @ np.linalg.inv(src_frame)
    fix = src_in_hand @ np.linalg.inv(new_in_hand)
    out = rows.copy()
    for i, row in enumerate(rows):
        out[i, :6] = matrix_to_pose(delta @ pose_to_matrix(row[:6]) @ fix)
    out[:, 5] = np.degrees(np.unwrap(np.radians(out[:, 5])))
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
