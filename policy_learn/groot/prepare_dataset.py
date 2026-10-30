"""Check Agent 4's LeRobot export and turn it into the GR00T training layout (DATA_FORMAT.md).

    python -m policy_learn.groot.prepare_dataset --src ~/t12_data/lerobot/<ds> [--src ...] --out ~/groot_data \
        [--val-fraction 0.02]

For every source dataset it writes `<out>/train/<name>` and `<out>/val/<name>` (held-out episodes, re-indexed):
- parquet: the source columns plus `observation.eef_9d` and `action.eef_9d` (tcp_convert.tcp_to_eef9d),
- videos: symlinks to the source mp4 files (the source is never modified),
- meta: info.json (+ the two new features), episodes.jsonl, tasks.jsonl, modality.json.
Any check failure stops the run with every problem listed; nothing is silently dropped.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from policy_learn.groot.tcp_convert import tcp_to_eef9d

CAMERAS = ("table", "side", "wrist")
STATE_DIM, ACTION_DIM = 13, 7  # [J1..J6, gripper, x, y, z, w, p, r] and act_tcp [x, y, z, w, p, r, gripper]
FPS = 5  # the demos' fixed control step: sim/assembly_lab/skills/motion.py DT_S = 0.2 s
WORKSPACE = ((150.0, 650.0), (-450.0, 450.0), (-334.0, 300.0))  # PLAN 3.1, mm
BOX_TOL_MM = 1.0
CHUNK = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

MODALITY = {
    "state": {
        "eef_9d": {"start": 0, "end": 9, "original_key": "observation.eef_9d"},
        "gripper": {"start": 6, "end": 7},
        "joints": {"start": 0, "end": 6},
    },
    "action": {
        "eef_9d": {"start": 0, "end": 9, "original_key": "action.eef_9d"},
        "gripper": {"start": 6, "end": 7},
    },
    "video": {cam: {"original_key": f"observation.images.{cam}"} for cam in CAMERAS},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}


class DatasetError(ValueError):
    """The source export does not match DATA_FORMAT.md."""


@dataclass
class Source:
    root: Path
    info: dict
    episodes: list[dict]
    tasks: dict[int, str]
    problems: list[str] = field(default_factory=list)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_source(root: Path) -> Source:
    meta = root / "meta"
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl"):
        if not (meta / name).is_file():
            raise DatasetError(f"{root}: missing meta/{name}")
    info = json.loads((meta / "info.json").read_text())
    tasks = {int(t["task_index"]): str(t["task"]) for t in _jsonl(meta / "tasks.jsonl")}
    src = Source(root, info, _jsonl(meta / "episodes.jsonl"), tasks)
    check_info(src)
    return src


def check_info(src: Source) -> None:
    info, feats = src.info, src.info.get("features", {})
    if info.get("fps") != FPS:
        src.problems.append(f"info.fps={info.get('fps')} but the demos run at {FPS} Hz (dt 0.2 s)")
    for key, dim in (("observation.state", STATE_DIM), ("action", ACTION_DIM)):
        shape = feats.get(key, {}).get("shape")
        if shape != [dim]:
            src.problems.append(f"feature {key} shape {shape}, expected [{dim}]")
    for cam in CAMERAS:
        if feats.get(f"observation.images.{cam}", {}).get("dtype") != "video":
            src.problems.append(f"feature observation.images.{cam} missing or not dtype video")
    for key in ("data_path", "video_path", "chunks_size"):
        if key not in info:
            src.problems.append(f"info.json has no {key}")
    if not src.episodes:
        src.problems.append("episodes.jsonl is empty")


def _path(src: Source, pattern_key: str, ep: int, video_key: str = "") -> Path:
    pattern = src.info[pattern_key]
    return src.root / pattern.format(episode_chunk=ep // src.info["chunks_size"], episode_index=ep, video_key=video_key)


def check_episode(src: Source, ep_meta: dict, df: pd.DataFrame) -> list[str]:
    ep = int(ep_meta["episode_index"])
    out = []
    if len(df) != int(ep_meta["length"]):
        out.append(f"ep {ep}: {len(df)} rows but episodes.jsonl length {ep_meta['length']}")
    for col in ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"):
        if col not in df.columns:
            return out + [f"ep {ep}: missing column {col}"]
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    action = np.stack(df["action"].to_numpy()).astype(np.float64)
    if state.shape[1:] != (STATE_DIM,) or action.shape[1:] != (ACTION_DIM,):
        return out + [f"ep {ep}: state {state.shape} / action {action.shape}"]
    if not (np.isfinite(state).all() and np.isfinite(action).all()):
        out.append(f"ep {ep}: NaN/inf in state or action")
    for axis, (lo, hi) in enumerate(WORKSPACE):
        vals = action[:, axis]
        if vals.min() < lo - BOX_TOL_MM or vals.max() > hi + BOX_TOL_MM:
            out.append(f"ep {ep}: act_tcp axis {'xyz'[axis]} {vals.min():.1f}..{vals.max():.1f} leaves {lo}..{hi}")
    for name, g in (("action", action[:, 6]), ("state", state[:, 6])):
        if g.min() < 0.0 or g.max() > 1.0:
            out.append(f"ep {ep}: {name} gripper outside 0..1")
    if (df["episode_index"] != ep).any() or not np.array_equal(df["frame_index"].to_numpy(), np.arange(len(df))):
        out.append(f"ep {ep}: episode_index / frame_index columns are not {ep} / 0..n-1")
    if not np.allclose(df["timestamp"].to_numpy(), np.arange(len(df)) / FPS, atol=1e-3):
        out.append(f"ep {ep}: timestamp is not frame_index / {FPS}")
    bad = sorted({int(t) for t in df["task_index"]} - {t for t, s in src.tasks.items() if s.strip()})
    if bad:
        out.append(f"ep {ep}: task_index {bad} has no non-empty instruction in tasks.jsonl")
    for cam in CAMERAS:
        if not _path(src, "video_path", ep, f"observation.images.{cam}").is_file():
            out.append(f"ep {ep}: missing video {cam}")
    return out


def add_eef_columns(df: pd.DataFrame) -> pd.DataFrame:
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    action = np.stack(df["action"].to_numpy()).astype(np.float64)
    df = df.copy()
    df["observation.eef_9d"] = list(tcp_to_eef9d(state[:, 7:13]))
    df["action.eef_9d"] = list(tcp_to_eef9d(action[:, 0:6]))
    df["observation.state"] = list(state.astype(np.float32))
    df["action"] = list(action.astype(np.float32))
    return df


def write_split(src: Source, frames: dict[int, pd.DataFrame], episodes: list[dict], dest: Path) -> int:
    """Write the given source episodes re-indexed 0..n-1 under `dest`; returns the frame count."""
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "meta").mkdir(parents=True)
    out_eps, total = [], 0
    for new_ep, ep_meta in enumerate(episodes):
        old_ep = int(ep_meta["episode_index"])
        df = add_eef_columns(frames[old_ep])
        df["episode_index"] = new_ep
        df["index"] = np.arange(total, total + len(df))
        parquet = dest / DATA_PATH.format(episode_chunk=new_ep // CHUNK, episode_index=new_ep)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet, index=False)
        for cam in CAMERAS:
            key = f"observation.images.{cam}"
            link = dest / VIDEO_PATH.format(episode_chunk=new_ep // CHUNK, episode_index=new_ep, video_key=key)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(_path(src, "video_path", old_ep, key).resolve())
        out_eps.append({**ep_meta, "episode_index": new_ep, "source_episode_index": old_ep})
        total += len(df)
    write_meta(src, dest, out_eps, total)
    return total


def write_meta(src: Source, dest: Path, episodes: list[dict], total: int) -> None:
    info = json.loads(json.dumps(src.info))
    info.update(
        codebase_version="v2.1", fps=FPS, chunks_size=CHUNK, data_path=DATA_PATH, video_path=VIDEO_PATH,
        total_episodes=len(episodes), total_frames=total, total_chunks=(len(episodes) - 1) // CHUNK + 1,
        total_videos=len(episodes) * len(CAMERAS), splits={"train": f"0:{len(episodes)}"},
    )
    for key in ("observation.eef_9d", "action.eef_9d"):
        names = ["x_mm", "y_mm", "z_mm", "r00", "r01", "r02", "r10", "r11", "r12"]
        info["features"][key] = {"dtype": "float32", "shape": [9], "names": names}
    meta = dest / "meta"
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    # All tasks are kept: the parquet task_index values are not renumbered.
    (meta / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": t, "task": src.tasks[t]}) + "\n" for t in sorted(src.tasks))
    )
    (meta / "modality.json").write_text(json.dumps(MODALITY, indent=2))


def split_episodes(episodes: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    n_val = 0 if val_fraction <= 0 else min(50, max(1, round(len(episodes) * val_fraction)))
    if n_val >= len(episodes):
        raise DatasetError(f"{len(episodes)} episodes cannot give {n_val} held-out episodes and a training set")
    order = list(range(len(episodes)))
    random.Random(seed).shuffle(order)
    val_ids = set(order[:n_val])
    train = [e for i, e in enumerate(episodes) if i not in val_ids]
    return train, [e for i, e in enumerate(episodes) if i in val_ids]


def prepare(src_root: Path, out: Path, val_fraction: float, seed: int) -> dict:
    src = load_source(src_root)
    frames: dict[int, pd.DataFrame] = {}
    for ep_meta in src.episodes:
        ep = int(ep_meta["episode_index"])
        parquet = _path(src, "data_path", ep)
        if not parquet.is_file():
            src.problems.append(f"ep {ep}: missing {parquet.relative_to(src.root)}")
            continue
        frames[ep] = pd.read_parquet(parquet)
        src.problems += check_episode(src, ep_meta, frames[ep])
    if src.problems:
        shown = "\n  ".join(src.problems[:50])
        raise DatasetError(f"{src_root}: {len(src.problems)} problem(s):\n  {shown}")
    train, val = split_episodes(src.episodes, val_fraction, seed)
    name = src_root.name
    report = {"source": str(src_root), "name": name, "train_episodes": len(train), "val_episodes": len(val)}
    report["train_frames"] = write_split(src, frames, train, out / "train" / name)
    if val:
        report["val_frames"] = write_split(src, frames, val, out / "val" / name)
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, action="append", required=True, help="LeRobot dataset root (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    names = [s.name for s in args.src]
    if len(set(names)) != len(names):
        raise DatasetError(f"source dataset folder names must be unique: {names}")
    reports = [prepare(s.expanduser(), args.out.expanduser(), args.val_fraction, args.seed) for s in args.src]
    args.out.expanduser().joinpath("prepare_report.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports))


if __name__ == "__main__":
    main()
