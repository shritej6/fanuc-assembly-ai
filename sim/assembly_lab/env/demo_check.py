"""Record one scripted pick and insert demo through the cell env to check the recorder output.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--offset_mm", type=float, default=0.0, help="shift the insert target +x (UF0) to test jamming")
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
            ("close", f["grasp_frame"], 1, 0.5)]
    obs = run(env, plan, part)
    # the synchronised jaws centre the part on the first close; a second close on the now-centred part must not move it
    before = np.array(obs["sim"]["object_poses"][part][:3])
    tf0 = env.env.object_poses_mm()[part]
    obs = run(env, [("reopen", f["grasp_frame"], 0, 0.5), ("reclose", f["grasp_frame"], 1, 0.5)], part)
    after = np.array(obs["sim"]["object_poses"][part])
    pad = np.array([*yaml.safe_load(L.OBJECTS_YAML.read_text())["objects"][L.type_id(part)]["grips"][0]
                    ["pad_point_local_mm"], 1.0])
    pad_shift = float(np.linalg.norm((env.env.object_poses_mm()[part] @ pad - tf0 @ pad)[:3]))
    print(f"[demo] re-close moved the part's grip point (CAD pad point) {pad_shift:.3f} mm", flush=True)
    shift = float(np.linalg.norm(after[:3] - before))
    print(f"[demo] re-close on the centred part moved it {shift:.3f} mm (synchronised jaws: <= 0.02 mm); "
          f"dxyz {np.round(after[:3] - before, 4).tolist()} mm, wpr now {after[3:].tolist()}; "
          f"jaws {link_contacts(env)['jaw_q_mm']}", flush=True)
    obs = run(env, [("lift", up(f["grasp_frame"], 60), 1, 1.0)], part)
    # carry the part's MEASURED in-hand pose to the base's current CAD seat (the grasp shifts the part sub-mm and
    # the free base may have moved), as a vision-corrected skill would
    poses = env.env.object_poses_mm()
    tcp_tf = L.translation(*obs["tcp_pose"][:3])
    tcp_tf[:3, :3] = L.fanuc_rotation(*obs["tcp_pose"][3:6])
    seat = L.pose6_from_tf(env.solver.seat_tf_for(part, base, poses) @ np.linalg.inv(poses[part]) @ tcp_tf)
    seat[0] += args.offset_mm
    base0 = env.env.object_poses_mm()[base][:3, 3].copy()
    run(env, [("over_seat", up(seat, 60), 1, 1.0), ("approach", up(seat, 10.0), 1, 1.0)], part)
    t0 = time.time()
    for k in range(1, 43):                    # 0.25 mm per 0.1 s env step, one trace row each, 0.5 mm into the seat
        s = env.step_tcp([[*up(seat, 10.0 - 0.25 * k), 1]], 0.1)["sim"]
        poses = env.env.object_poses_mm()
        d = np.linalg.inv(env.solver.seat_tf_for(part, base, poses)) @ poses[part]
        c = s["contact_peak"][part]
        sub = s.get("substep_contact", {}).get(part)
        print(f"[demo] insert row {k}: tcp dz {10.0 - 0.25 * k:.2f} mm, part above seat {d[2, 3]:.3f} mm "
              f"xy {d[0, 3]:.3f} {d[1, 3]:.3f}, peak normal {c['normal_N']:.3f} lateral {c['lateral_N']:.3f} N "
              f"torque {c['torque_Nm']:.4f} N m" + (f", substep normal {[t[0] for t in sub]}" if sub else ""),
              flush=True)
    insert_hz = 42 / (time.time() - t0)
    poses = env.env.object_poses_mm()
    d = np.linalg.inv(env.solver.seat_tf_for(part, base, poses)) @ poses[part]
    print(f"[demo] offset {args.offset_mm} mm: part {d[2, 3]:.3f} mm above seat before release "
          f"({'SEATED' if abs(d[2, 3]) < 0.3 and np.hypot(d[0, 3], d[1, 3]) < 1.0 else 'NOT SEATED'}); insert {insert_hz:.2f} env steps/s; "
          f"base moved {np.round(poses[base][:3, 3] - base0, 3).tolist()} mm during the insert", flush=True)
    run(env, [("open", seat, 0, 0.5), ("retreat", up(seat, 60), 0, 1.0)], part)
    print(f"[demo] success={env.env.episode_success()} -> {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
