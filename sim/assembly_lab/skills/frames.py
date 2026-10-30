"""Sim body frames -> CAD installed local frames, and sim truth -> the planner's scene.

The env reports each rigid body in its STL frame (the mesh is spawned as-is). The planner, objects.yaml and every
gate use the CAD installed local frame (origin = centre of the lowest seating plane, z up when installed):
T_world_local = T_world_stl @ inv(T_local_from_stl). `LocalFrameEnv` rewrites `obs["sim"]["object_poses"]` into
local frames and keeps the raw body poses under `object_poses_body` (what Isaac Lab replays from the HDF5 states).
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from .geometry import matrix_to_pose, pose_to_matrix, wpr_to_rot

OBJECTS_YAML = Path(__file__).resolve().parents[1] / "objects" / "objects.yaml"
STANDING_TILT_DEG = 10.0  # local z within this of straight up: standing in its installed orientation


def type_id(uid: str) -> str:
    """'M4_02#2' -> 'M4_02' (the env's spare instances share the CAD id)."""
    return uid.split("#")[0]


@lru_cache(maxsize=1)
def catalog() -> dict:
    return yaml.safe_load(OBJECTS_YAML.read_text())["objects"]


def installed_up(uid: str) -> tuple[float, float, float]:
    """World direction that is up for this part when installed: -insert_axis from objects.yaml (world +z when the
    part has no insert axis, e.g. a base)."""
    axis = catalog()[type_id(uid)].get("insert_axis")
    if not axis:
        return (0.0, 0.0, 1.0)
    u = -np.asarray(axis, dtype=float)
    return tuple(float(v) for v in u / np.linalg.norm(u))


@lru_cache(maxsize=64)
def _stl_from_local(oid: str) -> tuple:
    t = np.array(catalog()[oid]["T_local_from_stl"], dtype=float)
    return tuple(map(tuple, np.linalg.inv(t)))


def body_to_local(uid: str, body_pose) -> np.ndarray:
    oid = type_id(uid)
    if oid not in catalog():
        return np.asarray(body_pose, dtype=float)  # unknown piece: no CAD frame
    return matrix_to_pose(pose_to_matrix(body_pose) @ np.array(_stl_from_local(oid)))


class LocalFrameEnv:
    """Wraps agent 1's env so every skill, check and gate sees CAD local frames."""

    def __init__(self, env):
        self.env = env

    def _convert(self, obs: dict) -> dict:
        sim = obs.get("sim") or {}
        body = sim.get("object_poses", {})
        sim["object_poses_body"] = body
        sim["object_poses"] = {u: list(body_to_local(u, p)) for u, p in body.items()}
        return obs

    def reset(self, layout: dict) -> dict:
        return self._convert(self.env.reset(layout))

    def reset_with(self, layout: dict, frames_fn) -> dict:
        """Reset with the env's frames callback (plan on the settled layout, frames in the recorder's initial state)."""
        return self._convert(self.env.reset(layout, frames_fn=frames_fn))

    def step_tcp(self, rows, dt: float) -> dict:
        return self._convert(self.env.step_tcp(rows, dt))

    def close(self) -> None:
        self.env.close()


def standing(pose) -> bool:
    z = wpr_to_rot(*np.asarray(pose, dtype=float)[3:6])[:, 2]
    return math.degrees(math.acos(max(-1.0, min(1.0, z[2])))) <= STANDING_TILT_DEG


def heading_deg(pose) -> float:
    x = wpr_to_rot(*np.asarray(pose, dtype=float)[3:6])[:, 0]
    return math.degrees(math.atan2(x[1], x[0]))


def scene_from_obs(obs: dict, installed: dict[str, str]) -> dict:
    """Planner scene (objects/scene.py) from sim truth: every object on the table, in its measured pose."""
    on_table = obs["sim"].get("on_table", {})
    objects = []
    for uid, pose in obs["sim"]["object_poses"].items():
        if on_table and not on_table.get(uid, True):
            continue
        oid = type_id(uid)
        if uid in installed:
            objects.append({"key": uid, "id": oid, "state": f"on:{installed[uid]}"})
            continue
        state = "installed" if standing(pose) else "fallen"
        objects.append({"key": uid, "id": oid, "x": float(pose[0]), "y": float(pose[1]), "yaw": heading_deg(pose),
                        "state": state})
    return {"objects": objects, "spares": [], "table_z_mm": -335.0}
