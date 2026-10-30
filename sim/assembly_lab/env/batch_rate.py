"""Measure env-steps/s with N cloned cells in one Isaac process (the num_envs gate: is batching worth it?).

    GPU=0 CACHE_TAG=_ne REPO=~/t12_data/repo_ne isaaclab_run.sh sim/assembly_lab/env/batch_rate.py \
        --num_envs 8 --cameras table side wrist --out ~/t12_data/batch_rate/n8_cam.json

The cell env's own scene, solver and sensor settings (mimic_env.make_env_cfg: physics dt, decimation, contact sensors,
cameras at 640x360), product 4's pool (base + 4 sheet parts, what one generation worker spawns), the arm and jaws held
at home. Every physics step reads all three contact sensors and the per-contact buffers of every env, like
contact_metrics.accumulate does; every env step renders and fetches the cameras, like the observation manager does.
Not timed: the cell env's Python per step (IK, signals, gates), which batching does not multiply on the GPU side.
Writes {num_envs, cameras, env_steps, wall_s, env_steps_per_s, ms_per_env_step, build_s}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(HERE), str(REPO / "sim" / "starter"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=1)
ap.add_argument("--cameras", nargs="*", default=[])
ap.add_argument("--product", type=int, default=4)
ap.add_argument("--steps", type=int, default=20, help="timed env steps (0.1 s each)")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=bool(args.cameras), device=args.device).app

import mdp
import numpy as np
import tcp
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from mimic_env import make_env_cfg
from physics_cfg import CONTACT_YAML, load_contact


class HoldAction(ActionTerm):
    """Holds every env's arm and jaws at the default joint pose; reads every contact sensor each physics step."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros((env.num_envs, 7), device=self.device)
        self.sink = torch.zeros((), device=self.device)

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions

    def apply_actions(self) -> None:
        import warp as wp

        robot = self._asset
        robot.set_joint_position_target_index(target=robot.data.default_joint_pos.torch.clone(), full_data=True)
        s = self._env.scene
        rc, oc, ec = s["robot_contact"].data, s["object_contact"].data, s["object_env_contact"].data
        acc = rc.force_matrix_w.torch.sum() + rc.contact_pos_w.torch.nan_to_num().sum() + oc.net_forces_w.torch.sum()
        acc = acc + ec.friction_forces_w.torch.sum() + ec.force_matrix_w.torch.sum()
        view = s["object_env_contact"].contact_view
        fn, *_ = view.get_contact_data(dt=self._env.physics_dt)
        fr, *_ = view.get_friction_data(dt=self._env.physics_dt)
        self.sink = self.sink + acc + wp.to_torch(fn).sum() + wp.to_torch(fr).sum()


@configclass
class HoldActionCfg(ActionTermCfg):
    class_type: type = HoldAction
    asset_name: str = "robot"


def _episode_steps(env) -> torch.Tensor:
    return env.episode_length_buf.float().unsqueeze(-1)


def build_cfg():
    contact, _ = load_contact(CONTACT_YAML, {})
    pool = [f"F0{args.product}"] + [f"M{args.product}_0{k}" for k in range(1, 5)]
    home = tcp.OPERATOR_HOME_DEG
    cfg = make_env_cfg("pick_insert", pool, contact, cameras=tuple(args.cameras), jaw_force_n=34.0,
                       home_rad=np.deg2rad([home[f"J{i}"] for i in range(1, 7)]), device=args.device,
                       num_envs=args.num_envs)

    @configclass
    class ActionsCfg:
        hold = HoldActionCfg()

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        concatenate_terms: bool = False
        enable_corruption: bool = False

    @configclass
    class ObservationsCfg:
        policy = PolicyCfg()

    obs = ObservationsCfg()
    for cam in args.cameras:
        setattr(obs.policy, f"{cam}_cam", ObservationTermCfg(func=mdp.camera_rgb,
                                                             params={"sensor_cfg": SceneEntityCfg(cam)}))
    if not args.cameras:
        obs.policy.time = ObservationTermCfg(func=_episode_steps)
    cfg.actions, cfg.observations = ActionsCfg(), obs
    cfg.events = None
    cfg.terminations.success = None
    cfg.episode_length_s = 1.0e4
    return cfg


def main() -> None:
    t0 = time.time()
    env = ManagerBasedRLEnv(build_cfg())
    build_s = time.time() - t0
    env.reset()
    act = torch.zeros((env.num_envs, 7), device=env.device)
    for _ in range(args.warmup):
        env.step(act)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(args.steps):
        env.step(act)
    torch.cuda.synchronize()
    wall = time.time() - t1
    rep = {"num_envs": env.num_envs, "cameras": list(args.cameras), "physics_dt": env.physics_dt,
           "decimation": env.cfg.decimation, "env_steps": args.steps * env.num_envs, "wall_s": round(wall, 3),
           "env_steps_per_s": round(args.steps * env.num_envs / wall, 3),
           "ms_per_process_step": round(1000 * wall / args.steps, 1), "build_s": round(build_s, 1),
           "env_origins": env.scene.env_origins.cpu().tolist()[:4]}
    print(f"[batch_rate] {json.dumps(rep)}", flush=True)
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    env.close()
    app.close()


if __name__ == "__main__":
    main()
