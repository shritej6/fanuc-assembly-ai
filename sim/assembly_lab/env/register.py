"""Register the cell tasks with Gym for the Isaac Lab Mimic scripts.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(HERE), str(REPO / "sim" / "starter"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

TASKS = {"PickInsert": "pick_insert", "PullOut": "pull_out", "PlaceFree": "place_free", "DefectDrop": "defect_drop"}
JAW_FORCE_N = 34.0


def task_id(name: str) -> str:
    return f"Isaac-T12-Cell-{name}-Mimic-v0"


def cell_env_cfg(skill: str):
    """CellMimicEnvCfg for one skill, configured from the T12_* environment variables (see module docstring)."""
    import layout as L
    import numpy as np
    import tcp
    from mimic_env import make_env_cfg
    from physics_cfg import CONTACT_YAML, load_contact

    contact, _ = load_contact(Path(os.environ.get("T12_CONTACT_YAML", CONTACT_YAML)), {})
    solver = L.LayoutSolver()
    pool_env = os.environ.get("T12_POOL", "").strip()
    pool = [u.strip() for u in pool_env.split(",") if u.strip()] if pool_env else L.default_pool(solver.geometry, 1)
    cams_env = os.environ.get("T12_CAMERAS")
    cameras = ("table", "side", "wrist") if cams_env is None else tuple(c for c in cams_env.split(",") if c)
    w, h = (int(v) for v in os.environ.get("T12_IMAGE", "640x360").lower().split("x"))
    home = tcp.OPERATOR_HOME_DEG
    jit = [float(v) for v in os.environ.get("T12_JITTER", "").split(",") if v.strip()]
    jitter = dict(zip(("cam_pos_mm", "cam_rot_deg", "light_frac"), jit))
    return make_env_cfg(skill, pool, contact, cameras=cameras, image_size=(w, h), jaw_force_n=JAW_FORCE_N,
                        home_rad=np.deg2rad([home[f"J{i}"] for i in range(1, 7)]),
                        sampler=os.environ.get("T12_SAMPLER") or None, jitter=jitter)


def _factory(skill: str):
    def make():
        return cell_env_cfg(skill)

    make.__name__ = f"cell_env_cfg_{skill}"
    return make


def register_tasks() -> None:
    """--external_callback target: registers the four cell tasks with gymnasium (idempotent)."""
    import gymnasium as gym

    for name, skill in TASKS.items():
        tid = task_id(name)
        if tid in gym.registry:
            continue
        gym.register(id=tid, entry_point="mimic_env:CellMimicEnv", disable_env_checker=True,
                     kwargs={"env_cfg_entry_point": _factory(skill)})
