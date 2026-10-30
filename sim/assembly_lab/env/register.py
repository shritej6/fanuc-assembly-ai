"""Gym registration of the cell for the official Isaac Lab Mimic scripts (annotate_demos.py, generate_dataset.py).

    isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
        --external_callback sim.assembly_lab.env.register.register_tasks \
        --task Isaac-T12-Cell-PickInsert-Mimic-v0 --input_file annotated.hdf5 --output_file generated.hdf5 ...

(the repo root must be on PYTHONPATH; isaaclab_run.sh sets it). Task ids: Isaac-T12-Cell-{PickInsert,PullOut,
PlaceFree,DefectDrop}-Mimic-v0, one per skill (the subtask signals and success term differ).

The env cfg is built when the script calls parse_env_cfg, from environment variables:
- T12_POOL: comma-separated object instances (default: all 30 + one spare copy of each part). It MUST equal the
  pool of the env that recorded the source demos, because reset_to restores rigid_object/<prim> states by name.
- T12_CAMERAS: comma-separated subset of table,side,wrist (default all three; empty string = none).
- T12_IMAGE: WxH (default 640x360). T12_SAMPLER: "module:function" for reset() without a layout (optional).
- T12_CONTACT_YAML: physics contact file (default sim/assembly_lab/physics/contact.yaml).
- T12_JITTER: "cam_pos_mm,cam_rot_deg,light_frac" per-episode camera / light jitter (default off), e.g. "2,0.3,0.15".
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
