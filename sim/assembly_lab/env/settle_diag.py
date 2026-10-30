"""Trace why a reset with seated parts does not settle: per-physics-step motion of every seated part and its contacts.

    isaaclab_run.sh sim/assembly_lab/env/settle_diag.py --product 4 --installed 1 [--steps 400]

Prints, for each seated part, its displacement from the spawn pose and speed every 10 physics steps, plus the deepest
non-robot contact (separation, mm) on the first step, so penetration at spawn and the onset of motion are visible.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--product", type=int, default=4)
ap.add_argument("--installed", type=int, default=1)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--twist", type=float, default=0.0, help="N m about z on the first seated part from step 1 (+1 N down)")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import sampler
from cell_env import AssemblyCellEnv, CellEnvCfg


def main() -> None:
    p = args.product
    ids = [f"F0{p}"] + [f"M{p}_0{k}" for k in range(1, 5)]
    seated = [f"M{p}_0{j}" for j in range(1, args.installed + 1)]
    env = AssemblyCellEnv(CellEnvCfg(device=args.device, pool=ids, cameras=()))   # starts the app
    import mimic_env  # needs the app (imports Isaac Lab)

    mimic_env.SETTLE["cap_s"] = 0.12                         # return from reset after one window; this script steps physics
    lay = sampler.random_layout(np.random.default_rng(args.seed), ids, {u: f"F0{p}" for u in seated})
    env.reset({"objects": lay["objects"], "instruction": "diag"})
    e = env.env
    import isaaclab.utils.math as M
    import torch

    watch = [f"F0{p}"] + seated
    objs = {u: e.scene[e.names[e.pool.index(u)]] for u in watch}
    p0 = {u: o.data.root_link_pos_w.torch[0].clone() for u, o in objs.items()}
    q0 = {u: o.data.root_link_quat_w.torch[0].clone() for u, o in objs.items()}
    dt = e.physics_dt
    for k in range(1, args.steps + 1):
        e.scene.write_data_to_sim()
        e.sim.step(render=False)
        e.scene.update(dt=dt)
        if k == 1:
            for u in watch:
                print(f"[diag] step1 {u} world pos m {[round(v, 5) for v in objs[u].data.root_link_pos_w.torch[0].tolist()]}"
                      f" spawn UF0 mm {[round(v, 2) for v in e.placements[u].tf_uf0[:3, 3].tolist()]}", flush=True)
            for u in watch:
                raw = e._metrics.raw_env_contacts(e.names[e.pool.index(u)])
                seps = sorted(raw["normal"], key=lambda c: c["sep"])[:6]
                base = objs[f"F0{p}"].data
                R = M.matrix_from_quat(base.root_link_quat_w.torch[0:1])[0]
                loc = lambda q, R=R, base=base: [round(v * 1000, 2) for v in (R.T @ (torch.tensor(q, device=R.device)
                                                                      - base.root_link_pos_w.torch[0])).tolist()]
                print(f"[diag] step1 {u}: {len(raw['normal'])} contacts, deepest (filter, sep mm, fn N, point in base "
                      f"frame mm, normal) {[(c['filter'], c['sep'], c['fn'], loc(c['p']), c['n']) for c in seps]}",
                      flush=True)
        if args.twist and k == 1:
            ids1 = torch.tensor([0], device=e.device)
            objs[seated[0]].permanent_wrench_composer.set_forces_and_torques_index(
                forces=torch.tensor([[[0.0, 0.0, -1.0]]], device=e.device),
                torques=torch.tensor([[[0.0, 0.0, args.twist]]], device=e.device), body_ids=ids1, env_ids=ids1,
                is_global=True)
        if seated and k in (1, 25, 100, 200):
            raw = e._metrics.raw_env_contacts(e.names[e.pool.index(seated[0])])
            base = objs[f"F0{p}"].data
            R = M.matrix_from_quat(base.root_link_quat_w.torch[0:1])[0]
            on_base = [c for c in raw["normal"] if c["filter"] == 0]
            pts = [[round(v * 1000, 1) for v in (R.T @ (torch.tensor(c["p"], device=R.device)
                                                       - base.root_link_pos_w.torch[0])).tolist()] + [c["fn"]]
                   for c in on_base]
            print(f"[diag] k={k} {seated[0]} on base: {len(on_base)} normal contacts, {len(raw['friction'])} friction "
                  f"anchors; points (x, y, z mm in base frame, fn N): {pts}", flush=True)
        if k % 10 == 0 or k <= 3:
            row = []
            for u, o in objs.items():
                d = float((o.data.root_link_pos_w.torch[0] - p0[u]).norm()) * 1000
                a = math.degrees(float(M.quat_error_magnitude(o.data.root_link_quat_w.torch[0:1], q0[u][None])[0]))
                v = float(o.data.root_lin_vel_w.torch[0].norm()) * 1000
                w = math.degrees(float(o.data.root_ang_vel_w.torch[0].norm()))
                row.append(f"{u} d={d:.3f}mm a={a:.3f}deg v={v:.2f}mm/s w={w:.2f}deg/s")
            print(f"[diag] k={k} t={k * dt:.3f}s " + " | ".join(row), flush=True)
    env.close()


if __name__ == "__main__":
    main()
