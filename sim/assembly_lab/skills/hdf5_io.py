"""Write recorded episodes as Isaac Lab HDF5 demos annotated for Mimic.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np

from .geometry import to_matrix_m, wpr_to_rot
from .motion import DT_S
from .runner import EpisodeRecorder

EEF = "er4ia"  # mimic/skills.yaml eef_name
CAMERAS = ("table", "side", "wrist")


def quat_wxyz(rot: np.ndarray) -> np.ndarray:
    tr = np.trace(rot)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(rot)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + rot[i, i] - rot[j, j] - rot[k, k]) * 2
        q = [0.0] * 4
        q[0] = (rot[k, j] - rot[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (rot[j, i] + rot[i, j]) / s
        q[1 + k] = (rot[k, i] + rot[i, k]) / s
    q = np.array(q)
    return q if q[0] >= 0 else -q


def root_pose_m(pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    return np.array([*(pose[:3] / 1000.0), *quat_wxyz(wpr_to_rot(*pose[3:6]))])


def joint_state(joints) -> np.ndarray:
    j = np.asarray(joints, dtype=float)
    return np.array([*np.radians(j[:6]), j[6]])


def _signals(rec: EpisodeRecorder, n: int) -> dict[str, np.ndarray]:
    out = {}
    for name, step in rec.marks.items():
        sig = np.zeros(n, dtype=bool)
        sig[min(step, n - 1):] = True
        out[name] = sig
    return out


def _m(poses_mm: np.ndarray) -> np.ndarray:
    return np.stack([to_matrix_m(p) for p in poses_mm])


def _pos_quat(mats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return mats[:, :3, 3].astype(np.float32), np.stack([quat_wxyz(t[:3, :3]) for t in mats]).astype(np.float32)


def _obs_arrays(rec: EpisodeRecorder) -> dict[str, np.ndarray]:
    obs = [f.obs for f in rec.frames]
    n = len(obs)
    joints = np.array([o["joints"] for o in obs], dtype=np.float64)
    tcp = np.array([np.asarray(o["tcp_pose"])[:6] for o in obs], dtype=np.float64)
    eef_pos, eef_quat = _pos_quat(_m(tcp))
    data = {
        "eef_pos": eef_pos, "eef_quat": eef_quat,
        "arm_joint_pos": np.radians(joints[:, :6]).astype(np.float32),
        "gripper_state": joints[:, 6:7].astype(np.float32),
        "joints": joints.astype(np.float32), "tcp_pose": tcp.astype(np.float32),
        "plan_ids": np.tile(np.asarray(rec.meta["plan_ids"], dtype=np.float32), (n, 1)),
        "plan_target": np.tile(np.asarray(rec.meta["plan_target"], dtype=np.float32), (n, 1)),
        "sim_contact_peak": np.asarray(rec.guard.trace[:n], dtype=np.float32),
        "instruction_index": np.array([f.instruction for f in rec.frames], dtype=np.int32),
        "train_mask": np.array([f.seg.learn for f in rec.frames], dtype=bool),
    }
    for name in rec.frame_defs:
        pos, quat = _pos_quat(np.stack([_scaled(rec.frame_pose(o, name)) for o in obs]))
        data[f"{name}_pos"], data[f"{name}_quat"] = pos, quat
    for cam in CAMERAS:
        if cam in (obs[0].get("images") or {}):
            data[f"{cam}_cam"] = np.stack([o["images"][cam] for o in obs]).astype(np.uint8)
    return data


def _scaled(t_mm: np.ndarray) -> np.ndarray:
    t = t_mm.copy()
    t[:3, 3] /= 1000.0
    return t


def write_episode(grp: h5py.Group, rec: EpisodeRecorder, attrs: dict) -> None:
    n = len(rec.frames)
    frames_obs = [f.obs for f in rec.frames]
    for k, v in _obs_arrays(rec).items():
        grp.create_dataset(f"obs/{k}", data=v, **({"compression": "gzip"} if v.ndim == 4 else {}))
    grp.create_dataset("actions", data=np.array([f.action for f in rec.frames], dtype=np.float32))
    dg = "obs/datagen_info"
    grp.create_dataset(f"{dg}/eef_pose/{EEF}", data=_m(np.array([o["tcp_pose"][:6] for o in frames_obs])))
    grp.create_dataset(f"{dg}/target_eef_pose/{EEF}", data=_m(np.array([f.action[:6] for f in rec.frames])))
    for name in rec.frame_defs:
        grp.create_dataset(f"{dg}/object_pose/{name}",
                           data=np.stack([_scaled(rec.frame_pose(o, name)) for o in frames_obs]))
    for name, sig in _signals(rec, n).items():
        grp.create_dataset(f"{dg}/subtask_term_signals/{name}", data=sig)
    body_key = "object_poses_body" if "object_poses_body" in frames_obs[0]["sim"] else "object_poses"
    for oid in sorted(frames_obs[0]["sim"][body_key]):
        poses = np.stack([root_pose_m(o["sim"][body_key][oid]) for o in frames_obs])
        grp.create_dataset(f"states/rigid_object/{oid}/root_pose", data=poses)
        grp.create_dataset(f"initial_state/rigid_object/{oid}/root_pose", data=poses[:1])
    joints = np.stack([joint_state(o["joints"]) for o in frames_obs])
    grp.create_dataset("states/articulation/robot/joint_position", data=joints)
    grp.create_dataset("initial_state/articulation/robot/joint_position", data=joints[:1])
    grp.attrs["num_samples"] = n
    grp.attrs["instruction"] = rec.instructions[0] if rec.instructions else ""
    grp.attrs["instructions"] = json.dumps(rec.instructions)
    grp.attrs["subtasks"] = json.dumps([{"name": s.name, "skill": s.skill, "object_ref": s.object_ref,
                                         "start": s.start, "end": s.end} for s in rec.subtasks])
    grp.attrs["marks"] = json.dumps(rec.marks)
    grp.attrs["events"] = json.dumps(rec.events)
    grp.attrs["peaks"] = json.dumps(rec.guard.episode_peaks())
    for k, v in (rec.meta | attrs).items():
        grp.attrs[k] = v if isinstance(v, (int, float, str, bool, np.bool_)) else json.dumps(v)


def env_args(env_name: str) -> dict:
    return {"env_name": env_name, "type": 2, "eef_name": EEF, "dt_s": DT_S,
            "action": "act_tcp [x, y, z, w, p, r, gripper] mm/deg, gripper 0 open 1 closed",
            "datagen_units": "metres, robot base frame UF0"}


def append_demo(path: Path, rec: EpisodeRecorder, attrs: dict, env_name: str) -> str:
    """Append one successful episode as data/demo_<k>; returns the demo key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        data = f.require_group("data")
        if "env_args" not in data.attrs:
            data.attrs["env_args"] = json.dumps(env_args(env_name))
        k = int(data.attrs.get("total_demos", 0))
        key = f"demo_{k}"
        write_episode(data.create_group(key), rec, attrs | {"success": True})
        data.attrs["total_demos"] = k + 1
        data.attrs["total"] = int(data.attrs.get("total", 0)) + len(rec.frames)
    return key


def annotate_recorder_file(path: Path) -> dict:
    """Add per-episode metadata to the recorder HDF5 and return a report of what was written."""
    meta_path = Path(str(path) + ".meta.jsonl")
    metas = [json.loads(line) for line in meta_path.read_text().splitlines()] if meta_path.exists() else []
    report = {"episodes": 0, "metas": len(metas), "step_mismatch": []}
    with h5py.File(path, "a") as f:
        all_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        keys = [k for k in all_keys if "actions" in f["data"][k]]  # resets that never ran a step export empty
        report["episodes"], report["empty_episodes"] = len(keys), len(all_keys) - len(keys)
        for key, meta in zip(keys, metas):
            grp = f["data"][key]
            n = int(grp["actions"].shape[0])
            for k, v in meta.items():
                if k == "train_mask":
                    continue
                grp.attrs[k] = v if isinstance(v, (int, float, str, bool)) else json.dumps(v)
            if n != meta["steps"]:
                report["step_mismatch"].append({"demo": key, "recorder": n, "agent3": meta["steps"]})
                continue
            if "plan_ids" in meta:
                grp.create_dataset("obs/plan_ids", data=np.tile(np.float32(meta["plan_ids"]), (n, 1)))
                grp.create_dataset("obs/plan_target", data=np.tile(np.float32(meta["plan_target"]), (n, 1)))
            grp.create_dataset("obs/train_mask", data=np.array(meta["train_mask"], dtype=bool))
    if report["episodes"] != report["metas"]:
        report["count_mismatch"] = True
    return report
