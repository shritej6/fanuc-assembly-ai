"""Turn the GR00T-prepared data into one LeRobot v2.1 dataset per split for the ACT / SmolVLA baselines.

    python -m policy_learn.baselines.lerobot_view --prepared ~/groot_data --out ~/baseline_data

Input: the output of `policy_learn.groot.prepare_dataset` (`<prepared>/train/<name>`, `<prepared>/val/<name>`), so
the baselines train on exactly GR00T's training episodes and hold out exactly GR00T's validation episodes.
Output: `<out>/train` and `<out>/val`, each ONE merged dataset (LeRobot 0.3.3's trainer takes a single dataset):
- `observation.state` float32[16] = [eef_9d (x, y, z mm + rot6d), gripper, J1..J6 deg], GR00T's three state keys
  in the same order as its modality config;
- `action` float32[10] = [eef_9d, gripper], GR00T's two action keys. Targets stay ABSOLUTE (GR00T regresses them
  relative to the current TCP; ACT and SmolVLA are trained on absolute targets, their standard setup);
- the three camera videos, symlinked to the source files (nothing is re-encoded);
- tasks merged across datasets (task_index renumbered), and `meta/episodes_stats.jsonl`, which LeRobot v2.1 needs.
  Camera entries there hold the ImageNet mean/std: ACT replaces camera stats with ImageNet's anyway
  (`dataset.use_imagenet_stats`, default true) and SmolVLA does not normalise images (VISUAL: IDENTITY).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from policy_learn.groot.prepare_dataset import CAMERAS, CHUNK, DATA_PATH, FPS, VIDEO_PATH

EEF_NAMES = ["x_mm", "y_mm", "z_mm", "r00", "r01", "r02", "r10", "r11", "r12"]
STATE_NAMES = [*EEF_NAMES, "gripper", "J1", "J2", "J3", "J4", "J5", "J6"]
ACTION_NAMES = [*EEF_NAMES, "gripper"]
IMAGENET = {"mean": [[[0.485]], [[0.456]], [[0.406]]], "std": [[[0.229]], [[0.224]], [[0.225]]],
            "min": [[[0.0]], [[0.0]], [[0.0]]], "max": [[[1.0]], [[1.0]], [[1.0]]]}


class ViewError(ValueError):
    """The prepared data is missing something the baselines need."""


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def to_baseline_columns(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Prepared parquet rows -> (state (n, 16), action (n, 10)) float32."""
    raw_state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)   # [J1..J6, gripper, x..r]
    raw_action = np.stack(df["action"].to_numpy()).astype(np.float32)             # [x..r, gripper]
    eef_state = np.stack(df["observation.eef_9d"].to_numpy()).astype(np.float32)
    eef_action = np.stack(df["action.eef_9d"].to_numpy()).astype(np.float32)
    state = np.concatenate([eef_state, raw_state[:, 6:7], raw_state[:, 0:6]], axis=1)
    action = np.concatenate([eef_action, raw_action[:, 6:7]], axis=1)
    return state, action


def _numeric_stats(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    return {"min": v.min(0).tolist(), "max": v.max(0).tolist(), "mean": v.mean(0).tolist(),
            "std": v.std(0).tolist(), "count": [len(v)]}


def episode_stats(df: pd.DataFrame) -> dict:
    stats = {key: _numeric_stats(np.stack(df[key].to_numpy()) if key in ("observation.state", "action")
                                 else df[key].to_numpy())
             for key in ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index",
                         "task_index")}
    for cam in CAMERAS:
        stats[f"observation.images.{cam}"] = {**IMAGENET, "count": [len(df)]}
    return stats


def build_split(sources: list[Path], dest: Path) -> dict:
    """Merge the prepared datasets in `sources` into one LeRobot v2.1 dataset at `dest`."""
    if not sources:
        raise ViewError(f"no prepared datasets for {dest.name}")
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "meta").mkdir(parents=True)
    tasks: dict[str, int] = {}
    episodes, ep_stats, total = [], [], 0
    video_features = None
    for src in sources:
        info = json.loads((src / "meta/info.json").read_text())
        if info.get("fps") != FPS:
            raise ViewError(f"{src}: fps {info.get('fps')} != {FPS}")
        video_features = video_features or {f"observation.images.{c}": info["features"][f"observation.images.{c}"]
                                             for c in CAMERAS}
        src_tasks = {int(t["task_index"]): t["task"] for t in _jsonl(src / "meta/tasks.jsonl")}
        for meta in _jsonl(src / "meta/episodes.jsonl"):
            old = int(meta["episode_index"])
            df = pd.read_parquet(src / DATA_PATH.format(episode_chunk=old // CHUNK, episode_index=old))
            new = len(episodes)
            state, action = to_baseline_columns(df)
            out = pd.DataFrame({
                "observation.state": list(state), "action": list(action),
                "timestamp": df["timestamp"].to_numpy(np.float32), "frame_index": df["frame_index"].to_numpy(),
                "episode_index": np.full(len(df), new), "index": np.arange(total, total + len(df)),
                "task_index": [tasks.setdefault(src_tasks[int(t)], len(tasks)) for t in df["task_index"]],
            })
            path = dest / DATA_PATH.format(episode_chunk=new // CHUNK, episode_index=new)
            path.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(path, index=False)
            for cam in CAMERAS:
                key = f"observation.images.{cam}"
                link = dest / VIDEO_PATH.format(episode_chunk=new // CHUNK, episode_index=new, video_key=key)
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to((src / VIDEO_PATH.format(episode_chunk=old // CHUNK, episode_index=old,
                                                         video_key=key)).resolve())
            episodes.append({"episode_index": new, "tasks": meta["tasks"], "length": len(out),
                             "source": f"{src.name}/{old}"})
            ep_stats.append({"episode_index": new, "stats": episode_stats(out)})
            total += len(out)
    _write_meta(dest, episodes, ep_stats, tasks, total, video_features)
    return {"split": dest.name, "datasets": [s.name for s in sources], "episodes": len(episodes), "frames": total,
            "tasks": len(tasks)}


def _write_meta(dest: Path, episodes, ep_stats, tasks, total, video_features) -> None:
    scalar = {"dtype": "float32", "shape": [1], "names": None}
    integer = {"dtype": "int64", "shape": [1], "names": None}
    info = {
        "codebase_version": "v2.1", "robot_type": "fanuc_er4ia_teamv", "fps": FPS, "chunks_size": CHUNK,
        "total_episodes": len(episodes), "total_frames": total, "total_tasks": len(tasks),
        "total_videos": len(episodes) * len(CAMERAS), "total_chunks": (len(episodes) - 1) // CHUNK + 1,
        "splits": {"train": f"0:{len(episodes)}"}, "data_path": DATA_PATH, "video_path": VIDEO_PATH,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [len(STATE_NAMES)], "names": STATE_NAMES},
            "action": {"dtype": "float32", "shape": [len(ACTION_NAMES)], "names": ACTION_NAMES},
            "timestamp": scalar, "frame_index": integer, "episode_index": integer, "index": integer,
            "task_index": integer, **video_features,
        },
    }
    meta = dest / "meta"
    (meta / "info.json").write_text(json.dumps(info, indent=2))
    (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (meta / "episodes_stats.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ep_stats))
    (meta / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n"
                                              for t, i in sorted(tasks.items(), key=lambda kv: kv[1])))


def build(prepared: Path, out: Path) -> list[dict]:
    reports = []
    for split in ("train", "val"):
        sources = sorted(d for d in (prepared / split).glob("*") if (d / "meta/info.json").is_file())
        if split == "val" and not sources:
            continue
        reports.append(build_split(sources, out / split))
    return reports


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", type=Path, required=True, help="prepare_dataset.py --out directory")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    reports = build(args.prepared.expanduser(), args.out.expanduser())
    args.out.expanduser().joinpath("view_report.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports))


if __name__ == "__main__":
    main()
