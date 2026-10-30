"""Record one scripted pick_insert demo through AssemblyCellEnv with the Isaac Lab recorder, for checking the official
Mimic pipeline (annotate_demos.py --auto, generate_dataset.py) end to end on the cell.

    T12_POOL=F04,M4_01 isaaclab_run.sh sim/assembly_lab/env/demo_check.py --out ~/t12_data/mimic_demo/pick_insert.hdf5

The script follows the CAD frames only (grasp -> lift -> above the CAD seat -> down -> open -> retreat); it is a
pipeline check, not a training demo (agent 3 owns those). Prints the subtask signals after each move.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import layout as L
import sampler
from cell_env import AssemblyCellEnv, CellEnvCfg


def link_contacts(env) -> dict:
    """{object: {robot link: N}} for every robot link touching an object now (which link touches what)."""
    sensor = env.env._metrics.robot
    f = sensor.data.force_matrix_w.torch[0].norm(dim=-1).cpu().numpy()        # (links, objects)
    out = {}
    for j, name in enumerate(env.env.names):
        hit = {sensor.body_names[i]: round(float(f[i, j]), 2) for i in range(f.shape[0]) if f[i, j] > 0.05}
        if hit:
            out[env.env.pool[j]] = hit
    robot = env.env.scene["robot"]
    ids = [list(robot.joint_names).index(n) for n in ("jaw_l", "jaw_r")]
    out["jaw_q_mm"] = [round(float(v) * 1000, 3) for v in robot.data.joint_pos.torch[0, ids]]
    out["jaw_effort_N"] = [round(float(v), 2) for v in robot.data.applied_torque.torch[0, ids]]
    import isaaclab.utils.math as M

    names = list(robot.body_names)
    pose = robot.data.body_link_pose_w.torch[0]
    j6 = pose[names.index("J6_link")]
    rot = M.matrix_from_quat(j6[3:7])
    loc = {b: (rot.T @ (pose[names.index(b), :3] - j6[:3])).cpu().numpy() * 1000 for b in ("JawL", "JawR")}
    out["jaws_in_j6_mm"] = {b: v.round(2).tolist() for b, v in loc.items()}
    return out


def run(env, plan, part):
    obs = None
    for name, tcp6, grip, dt in plan:
        rows = [[*p, grip] for p in tcp6] if isinstance(tcp6[0], list) else [[*tcp6, grip]]
        obs = env.step_tcp(rows, dt)
        s = obs["sim"]
        peak = {k: {a: round(b, 2) for a, b in v.items()} for k, v in s["contact_peak"].items()}
        touch = {k: round(v, 2) for k, v in s["robot_touch"].items()}
        print(f"[demo] {name}: peak={peak} signals={s['signals']} latched={s['signals_latched']} touch={touch} "
              f"part={s['object_poses'][part]} links={link_contacts(env)}", flush=True)
    return obs


def main() -> None:
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pool = [u for u in os.environ.get("T12_POOL", "F04,M4_01").split(",") if u]
    base, part = pool[0], pool[1]
    env = AssemblyCellEnv(CellEnvCfg(device=args.device, pool=pool, record_to=str(out)))

    def frames(poses):
        grasp = sampler.cad_grasp(env.solver, part, poses[part])
        rel = np.linalg.inv(poses[part]) @ grasp
        return {"grasp_frame": grasp, "seat_frame": env.solver.seat_tf_for(part, base, poses) @ rel,
                "spot_frame": L.translation(60.0, -80.0, 0.0) @ grasp}

    lay = {"objects": [{"id": base, "x": 330.0, "y": -90.0, "yaw": 0.0}, {"id": part, "x": 300.0, "y": 60.0, "yaw": 30.0}],
           "instruction": f"put {part} on {base}"}
    obs = env.reset(lay, frames_fn=frames)
    f = {k: list(v) for k, v in obs["sim"]["frames"].items()}
    up = lambda p, dz: [p[0], p[1], p[2] + dz, *p[3:6]]
    plan = [("above", up(f["grasp_frame"], 60), 0, 1.0), ("down", f["grasp_frame"], 0, 1.0),
            ("close", f["grasp_frame"], 1, 0.5), ("lift", up(f["grasp_frame"], 60), 1, 1.0)]
    obs = run(env, plan, part)
    # carry the part's MEASURED in-hand pose to the base's current CAD seat (the grasp shifts the part sub-mm and
    # the free base may have moved), as a vision-corrected skill would
    poses = env.env.object_poses_mm()
    tcp_tf = L.translation(*obs["tcp_pose"][:3])
    tcp_tf[:3, :3] = L.fanuc_rotation(*obs["tcp_pose"][3:6])
    seat = L.pose6_from_tf(env.solver.seat_tf_for(part, base, poses) @ np.linalg.inv(poses[part]) @ tcp_tf)
    plan = [("over_seat", up(seat, 60), 1, 1.0), ("approach", up(seat, 10.0), 1, 1.0),
            ("insert", [up(seat, 10.0 - 0.25 * k) for k in range(1, 41)], 1, 0.1), ("open", seat, 0, 0.5),
            ("retreat", up(seat, 60), 0, 1.0)]
    run(env, plan, part)
    print(f"[demo] success={env.env.episode_success()} -> {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
