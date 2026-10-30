"""Check the cell through the same path the official Mimic scripts use: register_tasks -> parse_env_cfg -> gym.make.

    T12_POOL=F04,M4_01 T12_CAMERAS=table,side,wrist isaaclab_run.sh sim/assembly_lab/env/mimic_check.py \
        --out ~/t12_data/mimic_check [--task Isaac-T12-Cell-PickInsert-Mimic-v0] [--device cuda:0]

Measures (report.json): Mimic API round trips (target_eef_pose_to_action <-> action_to_target_eef_pose), the
get_object_poses / get_subtask_term_signals keys over a scripted CAD-grasp pick, scene get_state -> reset_to
restoring every pose, and env steps per second.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--task", default="Isaac-T12-Cell-PickInsert-Mimic-v0")
ap.add_argument("--rate_s", type=float, default=10.0)
ap.add_argument("--list_colliders", action="store_true", help="print every non-robot collider prim and exit")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import layout as L
import numpy as np
import register
import sampler
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def frames_fn(env, part: str, base: str):
    def fn(poses: dict) -> dict:
        grasp = sampler.cad_grasp(env.solver, part, poses[part])
        rel = np.linalg.inv(poses[part]) @ grasp
        spot = L.translation(60.0, -80.0, 0.0) @ grasp
        return {"grasp_frame": grasp, "seat_frame": env.solver.seat_tf_for(part, base, poses) @ rel, "spot_frame": spot}
    return fn


def act(env, tcp6, grip):
    return torch.tensor([[*tcp6, grip]], dtype=torch.float32, device=env.device)


def main() -> None:
    args.out = args.out.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    register.register_tasks()
    t0 = time.time()
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=cfg).unwrapped
    if args.list_colliders:
        from pxr import UsdPhysics

        for p in env.scene.stage.Traverse():
            path = str(p.GetPath())
            if p.HasAPI(UsdPhysics.CollisionAPI) and not path.startswith(("/World/ER4iA", "/World/Objects")):
                print(f"[check] collider {path}", flush=True)
        env.close()
        return
    rep = {"task": args.task, "device": args.device, "pool": env.pool, "build_s": round(time.time() - t0, 2),
           "cameras": [c for c in ("table", "side", "wrist") if c in env.scene.sensors]}
    base = next(u for u in env.pool if u.startswith("F"))
    part = next(u for u in env.pool if u.startswith("M"))
    env.layout_override = {"objects": [{"id": base, "x": 300.0, "y": -60.0, "yaw": 0.0},
                                       {"id": part, "x": 300.0, "y": 40.0, "yaw": 30.0}], "instruction": "check"}
    env.frames_fn = frames_fn(env, part, base)
    env.reset()
    rep["target"], rep["base"] = env.task.target, env.task.base

    # Mimic API round trip at the current pose
    eef = env.get_robot_eef_pose("er4ia")
    a = env.target_eef_pose_to_action({"er4ia": eef[0]}, {"er4ia": torch.tensor([0.0])})
    back = env.action_to_target_eef_pose(a)["er4ia"].reshape(-1, 4, 4)[0]
    rep["roundtrip_pos_err_mm"] = float((back[:3, 3] - eef[0, :3, 3]).norm() * 1000)
    rep["roundtrip_rot_err"] = float((back[:3, :3] - eef[0, :3, :3]).abs().max())
    rep["action_shape"] = list(a.shape)
    rep["object_pose_keys"] = sorted(env.get_object_poses().keys())
    rep["signal_keys"] = sorted(env.get_subtask_term_signals().keys())
    state = env.scene.get_state(is_relative=False)
    start = env.object_poses_mm()

    grasp = L.pose6_from_tf(env.frames_mm()["grasp_frame"])
    above = list(grasp)
    above[2] += 60.0
    log = []
    for name, tcp6, grip, n in (("above", above, 0, 10), ("down", grasp, 0, 10), ("close", grasp, 1, 5),
                                ("lift", above, 1, 10)):
        for _ in range(n):
            env.step(act(env, tcp6, grip))
        sig = {k: bool(v.reshape(-1)[0]) for k, v in env.get_subtask_term_signals().items()}
        log.append({"phase": name, "tcp": [round(v, 2) for v in env.tcp_obs()], "signals": sig,
                    "touch": {u: round(v, 2) for u, v in env._metrics.chunk()["robot_touch"].items()}})
        print(f"[check] {name}: {log[-1]}", flush=True)
    rep["phases"] = log

    before = env.object_poses_mm()
    env.reset_to(state, env_ids=torch.tensor([0], device=env.device), is_relative=False)
    after = env.object_poses_mm()
    rep["reset_to_max_pos_err_mm"] = max(float(np.linalg.norm(after[u][:3, 3] - start[u][:3, 3])) for u in env.pool)
    rep["moved_before_reset_to_mm"] = float(np.linalg.norm(before[part][:3, 3] - after[part][:3, 3]))
    rep["target_after_reset_to"] = env.task.target

    hold = act(env, env.tcp_obs(), 0)
    t0, n = time.time(), 0
    while time.time() - t0 < args.rate_s:
        env.step(hold)
        n += 1
    hz = n / (time.time() - t0)
    rep["env_steps_per_s"] = round(hz, 2)
    rep["realtime_factor"] = round(hz * cfg.sim.dt * cfg.decimation, 3)
    (args.out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"[check] {json.dumps({k: v for k, v in rep.items() if k != 'phases'}, default=float)}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
