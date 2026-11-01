"""Check whether a gripper-held part seats or jams when inserted at a sideways offset.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--products", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 3, 4, 5, 6])
ap.add_argument("--offsets", type=lambda s: [float(x) for x in s.split(",")], default=[0.0, 0.5, 0.6, 1.0, 2.0])
ap.add_argument("--free_base", action="store_true")
args = ap.parse_args()

import layout as L
import sampler
from cell_env import AssemblyCellEnv, CellEnvCfg

AXIAL_LIMIT_N = 15.0
CAD_CAPTURE_MM = 0.5
BASE_HOLD_N = 30.0


def hold_base(env, base: str, on: bool) -> None:
    import torch

    e = env.env
    ids = torch.tensor([0], device=e.device)
    f = torch.tensor([[[0.0, 0.0, -BASE_HOLD_N if on else 0.0]]], device=e.device)
    e.scene[e.names[e.pool.index(base)]].permanent_wrench_composer.set_forces_and_torques_index(
        forces=f, torques=torch.zeros_like(f), body_ids=ids, env_ids=ids, is_global=True)


def episode(env, base: str, part: str, offset: float) -> dict:
    def frames(poses):
        grasp = sampler.cad_grasp(env.solver, part, poses[part])
        rel = np.linalg.inv(poses[part]) @ grasp
        return {"grasp_frame": grasp, "seat_frame": env.solver.seat_tf_for(part, base, poses) @ rel,
                "spot_frame": L.translation(60.0, -80.0, 0.0) @ grasp}

    lay = {"objects": [{"id": base, "x": 330.0, "y": -90.0, "yaw": 0.0}, {"id": part, "x": 300.0, "y": 60.0, "yaw": 30.0}],
           "instruction": f"put {part} on {base}"}
    obs = env.reset(lay, frames_fn=frames)
    hold_base(env, base, not args.free_base)
    g = list(obs["sim"]["frames"]["grasp_frame"])
    up = lambda p, dz: [p[0], p[1], p[2] + dz, *p[3:6]]
    for row, dt in (([*up(g, 60), 0], 1.0), ([*g, 0], 1.0), ([*g, 1], 0.5), ([*up(g, 60), 1], 1.0)):
        obs = env.step_tcp([row], dt)
    poses = env.env.object_poses_mm()
    tcp_tf = L.translation(*obs["tcp_pose"][:3])
    tcp_tf[:3, :3] = L.fanuc_rotation(*obs["tcp_pose"][3:6])
    seat = list(L.pose6_from_tf(env.solver.seat_tf_for(part, base, poses) @ np.linalg.inv(poses[part]) @ tcp_tf))
    seat[0] += offset
    base0 = poses[base][:3, 3].copy()
    for row, dt in (([*up(seat, 60), 1], 1.0), ([*up(seat, 10.0), 1], 1.0)):
        env.step_tcp([row], dt)
    peak, stop = 0.0, "depth"
    for k in range(1, 43):
        c = env.step_tcp([[*up(seat, 10.0 - 0.25 * k), 1]], 0.1)["sim"]["contact_peak"][part]
        peak = max(peak, c["normal_N"])
        if c["normal_N"] >= AXIAL_LIMIT_N:
            stop = f"force guard at row {k}"
            break
    poses = env.env.object_poses_mm()
    d = np.linalg.inv(env.solver.seat_tf_for(part, base, poses)) @ poses[part]
    h = float(d[2, 3])
    r = {"part": part, "base": base, "offset_mm": offset, "base_held": not args.free_base,
         "height_above_seat_mm": round(h, 3), "xy_from_seat_mm": [round(float(d[0, 3]), 3), round(float(d[1, 3]), 3)],
         "peak_normal_N": round(peak, 2), "stopped_by": stop,
         "base_moved_mm": round(float(np.linalg.norm(poses[base][:3, 3] - base0)), 3),
         "result": "seated" if abs(h) < 0.3 else ("jammed" if h >= 2.0 else "partial")}
    r["ok"] = r["result"] == ("seated" if offset <= CAD_CAPTURE_MM else "jammed")
    hold_base(env, base, False)
    print(f"[held] {r}", flush=True)
    return r


def main() -> None:
    pool = [u for p in args.products for u in (f"F0{p}", f"M{p}_01")]
    env = AssemblyCellEnv(CellEnvCfg(device=args.device, pool=pool, cameras=()))
    rows = [episode(env, f"F0{p}", f"M{p}_01", off) for p in args.products for off in args.offsets]
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, default=float))
    ok = sum(r["ok"] for r in rows)
    print(f"[held] {ok}/{len(rows)} as CAD says -> {out}", flush=True)
    print(f"[held] {'PASS' if ok == len(rows) else 'FAIL'}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
