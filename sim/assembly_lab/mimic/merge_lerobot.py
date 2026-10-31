"""Merge LeRobot datasets written by export_lerobot into one (one per skill for training).

Each host and incremental export made its own small dataset; prepare_dataset holds out at least one episode per
source and cannot split a 1-episode source, so many small sources waste training data. Episodes are re-indexed in
input order, tasks are re-numbered, videos are hard-linked (copied across filesystems), global stats are recomputed
from the parquet rows and meta/t12_episodes.jsonl keeps each episode's provenance plus its source dataset.

    python -m sim.assembly_lab.mimic.merge_lerobot --src ~/t12_data/lerobot/n2_pick_insert_* --out ~/merged/pick_insert
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import lerobot_writer as W


class MergeError(RuntimeError):
    """Sources that cannot share one dataset (different fps, image size or robot)."""


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _layout(src: Path) -> tuple[int, tuple[int, int], str]:
    info = json.loads((src / "meta/info.json").read_text())
    h, w, _ = info["features"][f"observation.images.{W.CAMERAS[0]}"]["shape"]
    return info["fps"], (w, h), info["robot_type"]


def _link(a: Path, b: Path) -> None:
    b.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(a, b)
    except OSError:
        shutil.copy2(a, b)


def merge(sources: list[Path], out: Path) -> dict:
    layouts = {_layout(s) for s in sources}
    if len(layouts) != 1:
        raise MergeError(f"sources differ in (fps, size, robot): {sorted(layouts)}")
    fps, size, robot = layouts.pop()
    if out.exists():
        raise MergeError(f"{out} exists: refusing to merge into it")
    tasks, task_idx, episodes, prov = [], {}, [], []
    states, actions, stamps = [], [], []
    index = 0
    for src in sources:
        stats = {e["episode_index"]: e["stats"] for e in _jsonl(src / "meta/episodes_stats.jsonl")}
        t12 = {e["episode_index"]: e for e in _jsonl(src / "meta/t12_episodes.jsonl")}
        for ep in _jsonl(src / "meta/episodes.jsonl"):
            old, new, task = ep["episode_index"], len(episodes), ep["tasks"][0]
            ti = task_idx.setdefault(task, len(tasks))
            if ti == len(tasks):
                tasks.append(task)
            df = pd.read_parquet(W.parquet_path(src, old))
            t = len(df)
            df["episode_index"] = np.full(t, new, np.int64)
            df["index"] = np.arange(index, index + t, dtype=np.int64)
            df["task_index"] = df[W.LANG_COLUMN] = np.full(t, ti, np.int64)
            path = W.parquet_path(out, new)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            for cam in W.CAMERAS:
                _link(W.video_path(src, old, cam), W.video_path(out, new, cam))
            states.append(np.stack(df["observation.state"].to_numpy()))
            actions.append(np.stack(df["action"].to_numpy()))
            stamps.append(df["timestamp"].to_numpy(np.float32))
            episodes.append({"episode_index": new, "task": task, "length": t, "stats": stats[old]})
            prov.append({**t12[old], "episode_index": new, "src_dataset": src.name, "src_episode_index": old})
            index += t
    W.write_meta(out, episodes, tasks, size, fps, robot, np.concatenate(states), np.concatenate(actions),
                 np.concatenate(stamps))
    with open(out / "meta/t12_episodes.jsonl", "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in prov)
    report = {"sources": [s.name for s in sources], "episodes": len(episodes), "frames": index, "tasks": len(tasks)}
    (out / "meta/t12_merge_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(merge([s.expanduser() for s in a.src], a.out.expanduser())))


if __name__ == "__main__":
    main()
