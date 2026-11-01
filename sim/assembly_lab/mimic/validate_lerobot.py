"""Run a structural check on a LeRobot v2.1 export before it is used for training.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import lerobot_writer as W


class DatasetError(ValueError):
    """The dataset breaks the GR00T LeRobot format."""


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _video_frames(path: Path) -> int:
    """Decoded frame count. The Brev hosts have no ffprobe; there imageio-ffmpeg's ffmpeg decodes every frame."""
    if shutil.which("ffprobe") is None:
        import imageio_ffmpeg

        try:
            return int(imageio_ffmpeg.count_frames_and_secs(str(path))[0])
        except RuntimeError as e:
            raise DatasetError(f"ffmpeg cannot read {path}: {e}") from e
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise DatasetError(f"ffprobe cannot read {path}: {out.stderr[-300:]}")
    return int(out.stdout.strip())


def _check_meta(root: Path) -> tuple[dict, list[dict], dict]:
    meta = root / "meta"
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "stats.json"):
        if not (meta / name).exists():
            raise DatasetError(f"missing meta/{name}")
    info = json.loads((meta / "info.json").read_text())
    if info["codebase_version"] != W.CODEBASE_VERSION:
        raise DatasetError(f"codebase_version {info['codebase_version']} != {W.CODEBASE_VERSION} (GR00T needs v2)")
    modality = W.MODALITY
    stats = json.loads((meta / "stats.json").read_text())
    for feat in ("observation.state", "action"):
        if any(k not in stats.get(feat, {}) for k in W.STATS_KEYS):
            raise DatasetError(f"stats.json lacks {W.STATS_KEYS} for {feat}")
        fp = stats.get("__fingerprints__", {}).get(feat)
        if fp != W.stats_fingerprint(feat, info["features"][feat]):
            raise DatasetError(f"stats fingerprint for {feat} does not match info.json")
    for group in ("state", "action"):
        dim = info["features"]["observation.state" if group == "state" else "action"]["shape"][0]
        spans = sorted((v["start"], v["end"]) for v in modality[group].values())
        if spans[0][0] != 0 or spans[-1][1] != dim or any(a[1] != b[0] for a, b in itertools.pairwise(spans)):
            raise DatasetError(f"modality.json {group} slices {spans} do not tile 0..{dim}")
    for cam in modality["video"].values():
        if cam["original_key"] not in info["features"]:
            raise DatasetError(f"video key {cam['original_key']} not in info.json features")
    return info, _jsonl(meta / "episodes.jsonl"), modality


def _check_episode(root: Path, info: dict, ep: dict, tasks: dict, check_video: bool) -> int:
    i = ep["episode_index"]
    df = pd.read_parquet(W.parquet_path(root, i))
    if len(df) != ep["length"]:
        raise DatasetError(f"episode {i}: parquet has {len(df)} rows, episodes.jsonl says {ep['length']}")
    state, action = np.stack(df["observation.state"]), np.stack(df["action"])
    if state.shape[1] != len(W.STATE_NAMES) or action.shape[1] != len(W.ACTION_NAMES):
        raise DatasetError(f"episode {i}: state/action dims {state.shape[1]}/{action.shape[1]}")
    if not (np.isfinite(state).all() and np.isfinite(action).all()):
        raise DatasetError(f"episode {i}: non-finite state or action")
    if not ((action[:, 6] >= 0) & (action[:, 6] <= 1)).all():
        raise DatasetError(f"episode {i}: gripper action outside [0, 1]")
    lang = df[W.LANG_COLUMN].map(tasks)
    if lang.isna().any() or lang.iloc[0] != ep["tasks"][0]:
        raise DatasetError(f"episode {i}: language index does not resolve to its task")
    if check_video:
        for cam in W.CAMERAS:
            n = _video_frames(W.video_path(root, i, cam))
            if n != len(df):
                raise DatasetError(f"episode {i} {cam}: {n} video frames for {len(df)} rows")
    return len(df)


def validate(root: Path, check_video_every: int = 1) -> dict:
    """Check the dataset at root and return a summary."""
    info, episodes, _ = _check_meta(root)
    tasks = {t["task_index"]: t["task"] for t in _jsonl(root / "meta/tasks.jsonl")}
    if len(episodes) != info["total_episodes"]:
        raise DatasetError("episodes.jsonl count != info.total_episodes")
    frames = sum(_check_episode(root, info, ep, tasks, k % check_video_every == 0) for k, ep in enumerate(episodes))
    if frames != info["total_frames"]:
        raise DatasetError(f"{frames} frames != info.total_frames {info['total_frames']}")
    return {"episodes": len(episodes), "frames": frames, "tasks": len(tasks)}


def main() -> None:
    """Parse the command line and validate a dataset."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--video-every", type=int, default=1, help="ffprobe every k-th episode (1 = all)")
    a = ap.parse_args()
    out = validate(a.root.expanduser(), a.video_every)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
