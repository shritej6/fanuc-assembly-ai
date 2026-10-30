"""Write a LeRobot v2.1 dataset as policy_learn/groot/DATA_FORMAT.md (agent 5) asks:
meta/{info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl, stats.json} (no modality.json: agent 5's
prepare_dataset.py writes it),
data/chunk-XXX/episode_XXXXXX.parquet, videos/chunk-XXX/observation.images.<cam>/episode_XXXXXX.mp4.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000
CAMERAS = ("table", "side", "wrist")
STATE_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "gripper", "x", "y", "z", "w", "p", "r"]
ACTION_NAMES = ["x", "y", "z", "w", "p", "r", "gripper"]
LANG_COLUMN = "annotation.human.task_description"
STATS_KEYS = ("mean", "std", "min", "max", "q01", "q99")

# How the 13 / 7 vectors split (agent 5's prepare step writes the real meta/modality.json; used only to self-check).
MODALITY = {
    "state": {"joints": {"start": 0, "end": 6}, "gripper": {"start": 6, "end": 7},
              "tcp_xyz": {"start": 7, "end": 10}, "tcp_wpr": {"start": 10, "end": 13}},
    "action": {"tcp_xyz": {"start": 0, "end": 3}, "tcp_wpr": {"start": 3, "end": 6},
               "gripper": {"start": 6, "end": 7}},
    "video": {cam: {"original_key": f"observation.images.{cam}"} for cam in CAMERAS},
    "annotation": {"human.task_description": {}},
}


def chunk_of(episode_index: int) -> int:
    return episode_index // CHUNK_SIZE


def parquet_path(root: Path, ep: int) -> Path:
    return root / f"data/chunk-{chunk_of(ep):03d}/episode_{ep:06d}.parquet"


def video_path(root: Path, ep: int, cam: str) -> Path:
    return root / f"videos/chunk-{chunk_of(ep):03d}/observation.images.{cam}/episode_{ep:06d}.mp4"


def encode_mp4(frames: np.ndarray, path: Path, fps: int) -> None:
    """(T, H, W, 3) uint8 -> H.264 yuv420p mp4 (decoded by GR00T's torchcodec). Raises on any ffmpeg failure."""
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (T, H, W, 3) uint8, got {frames.shape} {frames.dtype}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH: install it (apt-get install ffmpeg) before exporting")
    path.parent.mkdir(parents=True, exist_ok=True)
    _t, h, w, _ = frames.shape
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
           "-crf", "18", "-g", "2", str(path)]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {proc.stderr.decode(errors='replace')[-500:]}")


def array_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "min": x.min(0).tolist(),
            "max": x.max(0).tolist(), "q01": np.quantile(x, 0.01, axis=0).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).tolist()}


def image_stats(frames: np.ndarray, max_frames: int = 32) -> dict:
    """LeRobot v2.1 per-episode image stats: per channel, values in [0, 1], shape (3, 1, 1)."""
    idx = np.linspace(0, len(frames) - 1, min(len(frames), max_frames)).round().astype(int)
    x = frames[idx].astype(np.float32).reshape(-1, 3) / 255.0
    stat = {"mean": x.mean(0), "std": x.std(0), "min": x.min(0), "max": x.max(0)}
    out = {k: v.reshape(3, 1, 1).tolist() for k, v in stat.items()}
    out["count"] = [len(idx)]
    return out


def stats_fingerprint(feature: str, meta: dict) -> str:
    """Same hash as Isaac-GR00T gr00t/data/stats.py::_compute_stats_fingerprint, so its cache sees fresh stats."""
    payload = {"feature": feature, "dtype": meta.get("dtype"), "shape": meta.get("shape")}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _video_feature(width: int, height: int, fps: int) -> dict:
    return {"dtype": "video", "shape": [height, width, 3], "names": ["height", "width", "channels"],
            "info": {"video.height": height, "video.width": width, "video.codec": "h264",
                     "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "video.fps": fps,
                     "video.channels": 3, "has_audio": False}}


def features(image_size: tuple[int, int], fps: int) -> dict:
    """image_size = (width, height)."""
    scalar = {"shape": [1], "names": None}
    feats = {
        "observation.state": {"dtype": "float32", "shape": [len(STATE_NAMES)], "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": [len(ACTION_NAMES)], "names": ACTION_NAMES},
        "timestamp": {"dtype": "float32", **scalar},
        "frame_index": {"dtype": "int64", **scalar},
        "episode_index": {"dtype": "int64", **scalar},
        "index": {"dtype": "int64", **scalar},
        "task_index": {"dtype": "int64", **scalar},
        LANG_COLUMN: {"dtype": "int64", **scalar},
        "next.reward": {"dtype": "float32", **scalar},
        "next.done": {"dtype": "bool", **scalar},
    }
    feats.update({f"observation.images.{cam}": _video_feature(*image_size, fps) for cam in CAMERAS})
    return feats


def write_meta(root: Path, episodes: list[dict], tasks: list[str], image_size: tuple[int, int], fps: int,
               robot_type: str,
               all_state: np.ndarray, all_action: np.ndarray, all_ts: np.ndarray) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    feats = features(image_size, fps)
    total_frames = int(sum(e["length"] for e in episodes))
    info = {"codebase_version": CODEBASE_VERSION, "robot_type": robot_type, "total_episodes": len(episodes),
            "total_frames": total_frames, "total_tasks": len(tasks), "total_videos": len(episodes) * len(CAMERAS),
            "total_chunks": chunk_of(max(len(episodes) - 1, 0)) + 1, "chunks_size": CHUNK_SIZE, "fps": fps,
            "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": feats}
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    with open(meta / "tasks.jsonl", "w") as f:
        f.writelines(json.dumps({"task_index": i, "task": t}) + "\n" for i, t in enumerate(tasks))
    with open(meta / "episodes.jsonl", "w") as f, open(meta / "episodes_stats.jsonl", "w") as fs:
        for e in episodes:
            f.write(json.dumps({"episode_index": e["episode_index"], "tasks": [e["task"]],
                                "length": e["length"]}) + "\n")
            fs.write(json.dumps({"episode_index": e["episode_index"], "stats": e["stats"]}) + "\n")
    stats = {"observation.state": array_stats(all_state), "action": array_stats(all_action),
             "timestamp": array_stats(all_ts.reshape(-1, 1))}
    stats["__fingerprints__"] = {k: stats_fingerprint(k, feats[k]) for k in ("observation.state", "action",
                                                                             "timestamp")}
    (meta / "stats.json").write_text(json.dumps(stats, indent=4))
