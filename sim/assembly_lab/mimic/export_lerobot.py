"""Convert accepted Isaac Lab Mimic HDF5 demos into a GR00T N1.7 LeRobot v2.1 dataset.

An episode is exported only if (1) agent 3 kept it (attr `kept`; `success` for files without it) and (2) its whole contact trace passes the force
guard (physics/limits.yaml). Rejections are counted per reason in `meta/t12_export_report.json`, never dropped
silently. Per frame: state = joints J1..J6 (deg) + gripper + TCP [x, y, z mm, w, p, r deg] (UF0);
action = the absolute act_tcp row [x, y, z, w, p, r, gripper] the arm must reach by the next frame.
Frames whose obs/train_mask is False (rows that only stage a recovery case) are left out.

    python -m sim.assembly_lab.mimic.export_lerobot --inputs ~/t12_data/mimic/pick_insert/p4/*.hdf5 \
        --out ~/t12_data/lerobot/t12_p4_small --max-episodes 300
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from . import lerobot_writer as W
from .catalog import REPO, decode_plan_ids
from .force_guard import ForceLimits, check_trace, load_limits
from .frames import W_LOW, unwrap_deg
from .instructions import instruction
from .spec import load_spec

ROBOT_TYPE = "fanuc_er4ia_teamv"


@dataclass(frozen=True)
class EpisodeRef:
    file: str
    key: str


def _obs(spec_keys: dict) -> dict:
    return {**{k: v for k, v in spec_keys.items() if k != "images"}, **{f"img_{c}": n for c, n in
                                                                        spec_keys["images"].items()}}


def screen(path: str, limits: ForceLimits, contact_key: str, cams: tuple[str, ...] = ("table_cam", "side_cam",
                                                                                     "wrist_cam")) -> tuple[list, dict]:
    """Cheap pass: success attr, force guard, no dry-run stamp, all 3 cameras. Returns accepted refs + counts."""
    accepted = []
    counts = {"seen": 0, "not_success": 0, "force": 0, "no_contact_obs": 0, "dry_run": 0, "no_images": 0}
    with h5py.File(path, "r") as f:
        for key in sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1])):
            counts["seen"] += 1
            demo = f["data"][key]
            if "actions" not in demo:  # a reset that never ran a step
                counts["empty"] = counts.get("empty", 0) + 1
                continue
            if not accepted_attr(demo.attrs):
                counts["not_success"] += 1
                continue
            if bool(demo.attrs.get("KINEMATIC_DRY_RUN", False)):
                counts["dry_run"] += 1
                continue
            if any(c not in demo["obs"] for c in cams):
                counts["no_images"] += 1
                continue
            if contact_key not in demo["obs"]:
                counts["no_contact_obs"] += 1
                continue
            if not check_trace(demo["obs"][contact_key][()][:, :3], limits).ok:  # col 3 = robot_touch_N (env)
                counts["force"] += 1
                continue
            accepted.append(EpisodeRef(path, key))
    return accepted, counts


def _truthy(v) -> bool:
    if isinstance(v, bytes):
        v = v.decode()
    return v.lower() in ("true", "1") if isinstance(v, str) else bool(v)


def accepted_attr(attrs) -> bool:
    """Agent 3's `kept` decides when present (all gates, annotated at close); the Isaac Lab recorder's `success` attr
    is overwritten at reset by the env's success term, so it is used only for files without agent 3's metadata."""
    if "kept" in attrs:
        return _truthy(attrs["kept"])
    return _truthy(attrs.get("success", False))


def recorded_dt(path: str) -> float | None:
    """Seconds per recorded row from the recorder's env_args (dt x decimation), None if the file has no sim_args."""
    with h5py.File(path, "r") as f:
        args = json.loads(f["data"].attrs.get("env_args", "{}")).get("sim_args")
    return float(args["dt"]) * int(args["decimation"]) if args else None


def frame_indices(n_steps: int, sim_dt: float, fps: int) -> tuple[np.ndarray, np.ndarray]:
    """Observation step of each exported frame, and the action step that is that frame's label (end of interval)."""
    stride = round((1.0 / fps) / sim_dt)
    if stride < 1 or abs(stride * sim_dt - 1.0 / fps) > 1e-6:
        raise ValueError(f"1/fps = {1.0 / fps} s is not a whole multiple of sim_dt = {sim_dt} s")
    obs_idx = np.arange(0, n_steps, stride)
    act_idx = np.minimum(obs_idx + stride - 1, n_steps - 1)
    return obs_idx, act_idx


def _align_r(r_action: np.ndarray, r_state: np.ndarray) -> np.ndarray:
    """Shift the unwrapped action r by whole turns so it starts next to the state r (same wrist turn)."""
    turns = np.round((r_action[0] - r_state[0]) / 360.0)
    return r_action - 360.0 * turns


def _wrap_w(w: np.ndarray) -> np.ndarray:
    return (w - W_LOW) % 360.0 + W_LOW


def state_action(demo, obs_idx: np.ndarray, act_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """API units straight from the demo (obs/joints, obs/tcp_pose, actions = act_tcp rows). w continuous around
    -180 (frames.W_LOW), r unwrapped along the episode and on the same wrist turn in state and action."""
    o = demo["obs"]
    joints = o["joints"][()].astype(np.float64)
    tcp = o["tcp_pose"][()].astype(np.float64)[:, :6]
    act = demo["actions"][()].astype(np.float64)
    tcp[:, 3], act[:, 3] = _wrap_w(tcp[:, 3]), _wrap_w(act[:, 3])
    tcp[:, 5] = unwrap_deg(tcp[:, 5])
    tcp[:, 5] -= 360.0 * np.round(tcp[0, 5] / 360.0)  # first r into (-180, 180]
    act[:, 5] = _align_r(unwrap_deg(act[:, 5]), tcp[:, 5])
    state = np.concatenate([joints[:, :7], tcp], axis=1)[obs_idx]
    return state.astype(np.float32), act[act_idx, :7].astype(np.float32)


def train_rows(demo, obs_idx: np.ndarray) -> np.ndarray:
    """Frames to keep: obs/train_mask (False on rows that only stage a recovery case), all if absent."""
    if "train_mask" not in demo["obs"]:
        return np.ones(len(obs_idx), dtype=bool)
    return demo["obs"]["train_mask"][()].astype(bool)[obs_idx]


def _resize(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Whole frame to (width, height), never cropped (DATA_FORMAT.md: the image the robot policy gets)."""
    w, h = size
    if frames.shape[1:3] == (h, w):
        return frames.astype(np.uint8)
    return np.stack([np.asarray(Image.fromarray(fr).resize((w, h), Image.BILINEAR)) for fr in frames])


def convert_episode(job: dict) -> dict:
    """Worker: one HDF5 demo -> parquet + 3 mp4. Returns the episode record (meta + arrays for global stats)."""
    ref, ep, keys, root = EpisodeRef(*job["ref"]), job["episode_index"], job["keys"], Path(job["root"])
    with h5py.File(ref.file, "r") as f:
        demo = f["data"][ref.key]
        n = demo["actions"].shape[0]
        obs_idx, act_idx = frame_indices(n, job["sim_dt"], job["fps"])
        keep = train_rows(demo, obs_idx)
        obs_idx, act_idx = obs_idx[keep], act_idx[keep]
        state, action = state_action(demo, obs_idx, act_idx)
        ids, target = demo["obs"][keys["plan_ids"]][0], demo["obs"][keys["plan_target"]][0]
        stats = {}
        for cam in W.CAMERAS:
            frames = _resize(demo["obs"][keys[f"img_{cam}"]][obs_idx], job["size"])
            W.encode_mp4(frames, W.video_path(root, ep, cam), job["fps"])
            stats[f"observation.images.{cam}"] = W.image_stats(frames)
        peak = demo["obs"][job["contact_key"]][()].max(axis=0).tolist()
    return {"ref": job["ref"], "episode_index": ep, "state": state, "action": action, "stats": stats,
            "task": instruction(ids, target), "plan": decode_plan_ids(ids), "force_peak": peak}


def write_parquet(root: Path, rec: dict, task_index: int, start_index: int, fps: int) -> dict:
    t = len(rec["state"])
    ts = (np.arange(t) / fps).astype(np.float32)
    df = pd.DataFrame({
        "observation.state": list(rec["state"]), "action": list(rec["action"]), "timestamp": ts,
        "frame_index": np.arange(t, dtype=np.int64), "episode_index": np.full(t, rec["episode_index"], np.int64),
        "index": np.arange(start_index, start_index + t, dtype=np.int64),
        "task_index": np.full(t, task_index, np.int64), W.LANG_COLUMN: np.full(t, task_index, np.int64),
        "next.reward": np.zeros(t, np.float32), "next.done": np.arange(t) == t - 1,
    })
    path = W.parquet_path(root, rec["episode_index"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    stats = dict(rec["stats"])
    for col in ("observation.state", "action"):
        stats[col] = {**{k: v for k, v in W.array_stats(rec[col.split(".")[-1]]).items() if k in
                         ("mean", "std", "min", "max")}, "count": [t]}
    stats["timestamp"] = {**{k: v for k, v in W.array_stats(ts.reshape(-1, 1)).items()
                             if k in ("mean", "std", "min", "max")}, "count": [t]}
    return stats


def export(inputs: list[str], out: Path, sim_dt: float, fps: int, size: tuple[int, int], max_episodes: int | None,
           workers: int, product: int | None = None, limits_path: Path | None = None) -> dict:
    spec = load_spec()
    keys = _obs(spec.raw["obs_keys"])
    contact_key = spec.raw["force_guard"]["contact_obs"]
    limits = load_limits(limits_path or REPO / spec.raw["force_guard"]["limits_file"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: export into a fresh directory")
    refs, report = [], {"files": len(inputs), "limits": limits.__dict__, "sim_dt": sim_dt, "fps": fps}
    for path in inputs:
        dt = recorded_dt(path)
        if dt is not None and abs(dt - sim_dt) > 1e-9:
            raise ValueError(f"{path} records one row per {dt} s, not --sim-dt {sim_dt} s")
        acc, counts = screen(path, limits, contact_key)
        refs += acc
        for k, v in counts.items():
            report[k] = report.get(k, 0) + v
    if product is not None:
        refs = [r for r in refs if _product_of_ref(r, keys) == product]
    refs = refs[:max_episodes] if max_episodes else refs
    if not refs:
        raise RuntimeError(f"no episode passed success + force guard: {report}")
    jobs = [{"ref": (r.file, r.key), "episode_index": i, "keys": keys, "root": str(out), "sim_dt": sim_dt,
             "fps": fps, "size": size, "contact_key": contact_key} for i, r in enumerate(refs)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(convert_episode, jobs, chunksize=4))
    return _finish(out, records, fps, size, report)


def _product_of_ref(ref: EpisodeRef, keys: dict) -> int | None:
    with h5py.File(ref.file, "r") as f:
        return decode_plan_ids(f["data"][ref.key]["obs"][keys["plan_ids"]][0])["product"]


def _finish(out: Path, records: list[dict], fps: int, size: tuple[int, int], report: dict) -> dict:
    tasks, task_idx, episodes, index = [], {}, [], 0
    for rec in sorted(records, key=lambda r: r["episode_index"]):
        ti = task_idx.setdefault(rec["task"], len(tasks))
        if ti == len(tasks):
            tasks.append(rec["task"])
        stats = write_parquet(out, rec, ti, index, fps)
        index += len(rec["state"])
        episodes.append({"episode_index": rec["episode_index"], "task": rec["task"], "length": len(rec["state"]),
                         "stats": stats})
    all_state = np.concatenate([r["state"] for r in records])
    all_action = np.concatenate([r["action"] for r in records])
    all_ts = np.concatenate([np.arange(len(r["state"])) / fps for r in records]).astype(np.float32)
    W.write_meta(out, episodes, tasks, size, fps, ROBOT_TYPE, all_state, all_action, all_ts)
    with open(out / "meta/t12_episodes.jsonl", "w") as f:
        for rec in sorted(records, key=lambda r: r["episode_index"]):
            f.write(json.dumps({"episode_index": rec["episode_index"], "source": list(rec["ref"]),
                                "force_peak": rec["force_peak"], **rec["plan"]}) + "\n")
    report.update(exported=len(records), frames=int(index), tasks=len(tasks))
    (out / "meta/t12_export_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="generated Mimic HDF5 files (successful ones)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sim-dt", type=float, default=0.1, help="time per recorded row (skills/motion.py DT_S = env step)")
    ap.add_argument("--fps", type=int, default=5, help="exported frame rate = 1 / act dt on the cell")
    ap.add_argument("--width", type=int, default=640, help="frame width (DATA_FORMAT.md: 640)")
    ap.add_argument("--height", type=int, default=360, help="frame height (DATA_FORMAT.md: 360)")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--product", type=int, default=None, help="keep only this product (1..6)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limits", type=Path, default=None, help="override physics/limits.yaml (default from spec)")
    a = ap.parse_args()
    report = export(a.inputs, a.out.expanduser(), a.sim_dt, a.fps, (a.width, a.height), a.max_episodes, a.workers, a.product,
                    a.limits)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
