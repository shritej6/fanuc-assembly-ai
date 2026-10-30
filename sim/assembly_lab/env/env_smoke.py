"""Smoke test for AssemblyCellEnv: one product's base + parts loose, IK moves, all cameras, contact forces, step rate.

    (inside the Isaac Lab container, see ~/ISAACLAB_READY on fa-sim)
    python sim/assembly_lab/env/env_smoke.py --out ~/t12_data/env_smoke [--product 4] [--minimal]

--minimal spawns only the base and the housing (the first part), the fastest end-to-end check.
Writes <out>/<phase>_<camera>.png and <out>/report.json (measured numbers only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--product", type=int, default=4)
ap.add_argument("--minimal", action="store_true")
ap.add_argument("--disturb", action="store_true", help="also exercise the eval disturbance hooks after the lift")
ap.add_argument("--installed", type=int, default=0, help="the first N parts start installed on the base")
ap.add_argument("--full", action="store_true", help="spawn the whole default pool (30 + spares); unused ones park")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
args.out = args.out.expanduser().resolve()

import layout as L
from cell_env import AssemblyCellEnv, CellEnvCfg

REPO = HERE.parents[2]
GRASPS = json.loads((REPO / "cad/fingers/product_grasp_frames.json").read_text())
TIP_Z_MM = 86.1      # Team V tip plane in the gripper STEP frame


def grasp_tcp(geometry: dict, tid: str, part_tf_uf0: np.ndarray, lift_mm: float = 0.0) -> list[float]:
    """TCP [x y z w p r] for the CAD grasp of a standing part (cad/fingers/product_grasp_frames.json)."""
    key = next(k for k in GRASPS if k.startswith(tid + "_"))
    g = GRASPS[key]
    rot = np.array(g["rotation_gripper_to_world"], float)
    tcp_asm = np.eye(4)
    tcp_asm[:3, :3] = rot
    tcp_asm[:3, 3] = rot @ (0.0, 0.0, TIP_Z_MM) + np.array(g["translation_mm"])
    asm = part_tf_uf0 @ np.linalg.inv(np.array(geometry[tid]["installed_in_base_frame"]))
    pose = L.pose6_from_tf(L.translation(0, 0, lift_mm) @ asm @ tcp_asm)
    return pose


def save(out: Path, phase: str, obs: dict) -> None:
    from PIL import Image

    for cam, img in obs["images"].items():
        Image.fromarray(img).save(out / f"{phase}_{cam}.png")


def frames_for(env, poses: dict, part: str, base: str) -> dict:
    import sampler

    grasp = sampler.cad_grasp(env.solver, part, poses[part])
    rel = np.linalg.inv(poses[part]) @ grasp
    return {"grasp_frame": grasp, "seat_frame": env.solver.seat_tf_for(part, base, poses) @ rel,
            "spot_frame": grasp.copy()}


def wrist_camera_check(env) -> dict:
    """Camera prim world position vs the position the calibrated mount implies from FK (does it follow J6?)."""
    import fk
    import scene_config as C

    q = env.env.arm_q()
    rot, p = fk.fk(q)
    want = (np.asarray(p) + rot @ np.array(C.WRIST_CAM["eye"])) + (0.0, 0.0, C.TABLE_TOP_Z)
    cam = env.env.scene["wrist"]
    got = cam.data.pos_w.torch[0].cpu().numpy()
    import isaaclab.utils.math as M

    fwd = M.matrix_from_quat(cam.data.quat_w_world.torch[:1])[0, :, 0].cpu().numpy()
    want_fwd = rot @ (np.array(C.WRIST_CAM["look_at"]) - np.array(C.WRIST_CAM["eye"]))
    want_fwd /= np.linalg.norm(want_fwd)
    return {"forward_deg_err": float(np.degrees(np.arccos(np.clip(fwd @ want_fwd, -1, 1)))),
            "forward": fwd.round(3).tolist(),"fk_mm": (want * 1000).round(1).tolist(), "sensor_mm": (got * 1000).round(1).tolist(),
            "err_mm": float(np.linalg.norm(got - want) * 1000)}


def disturbance_check(env, ids: list[str], part: str, above: list[float]) -> dict:
    """force_release drops the held part; the props land where asked (UF0 mm of each prop's centre)."""
    env.inject_foreign(250.0, -220.0, 20.0)
    env.inject_intruder(470.0, 230.0)
    env.block_holes(ids[-1], ids[0])
    z0 = env.env.object_poses_mm()[part][2, 3]
    env.force_release()
    obs = env.step_tcp([[*above, 1.0]], 0.5)
    out = {"part_drop_mm": round(float(z0 - obs["sim"]["object_poses"][part][2]), 2),
           "grasped_after_release": obs["sim"]["signals"].get("grasped")}
    top = env.env._top()
    for name in ("Foreign", "Intruder", "Plug"):
        p = env.env.scene[f"extra_{name}"].data.root_pos_w.torch[0].cpu().numpy()
        out[name] = [round(float(v), 1) for v in L.world_m_to_uf0(L.translation(*p), top)[:3, 3]]
    env.clear_intruder()
    print(f"[smoke] disturbances: {out}", flush=True)
    return out


def main() -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    p = args.product
    ids = [f"F0{p}", f"M{p}_01"] if args.minimal else [f"F0{p}"] + [f"M{p}_0{k}" for k in range(1, 5)]
    t0 = time.time()
    env = AssemblyCellEnv(CellEnvCfg(device=args.device, pool=None if args.full else ids,
                                     disturbances=args.disturb))
    build_s = time.time() - t0
    lay = {"objects": [{"id": ids[0], "x": 300.0, "y": -60.0, "yaw": 0.0}], "instruction": "smoke test"}
    for k, tid in enumerate(ids[1:]):
        if k < args.installed:
            lay["objects"].append({"id": tid, "state": f"on:{ids[0]}"})
            continue
        lay["objects"].append({"id": tid, "x": 300.0 + 60.0 * (k % 2), "y": 40.0 + 60.0 * (k // 2), "yaw": 30.0})
    part = ids[1 + args.installed]
    t0 = time.time()
    obs = env.reset(lay, frames_fn=lambda poses: frames_for(env, poses, part, ids[0]))
    reset_s = time.time() - t0
    save(args.out, "00_reset", obs)
    step0 = {"settled": obs["sim"]["settled"], "settle_s": obs["sim"]["settle_s"], "why": obs["sim"]["settle_detail"],
             "contact": {u: {k: round(v, 3) for k, v in c.items()} for u, c in obs["sim"]["contact"].items()}}
    print(f"[smoke] after reset: {step0}", flush=True)
    cam_checks = {"00_reset": wrist_camera_check(env)}
    part_tf = env.placements[part].tf_uf0
    pick = grasp_tcp(env.solver.geometry, part, part_tf)
    above = grasp_tcp(env.solver.geometry, part, part_tf, lift_mm=60.0)
    plan = [("01_above", [*above, 0], 1.0), ("02_down", [*pick, 0], 1.0), ("03_close", [*pick, 1], 0.5),
            ("04_lift", [*above, 1], 1.0)]
    log = []
    for phase, row, dt in plan:
        t0 = time.time()
        obs = env.step_tcp([row], dt)
        wall = time.time() - t0
        save(args.out, phase, obs)
        cam_checks[phase] = wrist_camera_check(env)
        err = np.array(obs["tcp_pose"][:3]) - np.array(row[:3])
        s = obs["sim"]
        log.append({"phase": phase, "cmd": row, "tcp_pose": obs["tcp_pose"], "tcp_pos_err_mm": float(np.linalg.norm(err)),
                    "joints": obs["joints"].tolist(), "sim_s": dt, "wall_s": round(wall, 3),
                    "realtime_factor": round(dt / wall, 4), "signals": s["signals"], "latched": s["signals_latched"],
                    "robot_touch": s["robot_touch"], "jaw_n": s["jaw_n"], "contact": s["contact"],
                    "contact_peak": s["contact_peak"], "object_poses": s["object_poses"]})
        print(f"[smoke] {phase}: tcp err {log[-1]['tcp_pos_err_mm']:.3f} mm, wall {wall:.2f} s for {dt} s sim, "
              f"signals={s['signals']} touch={ {k: round(v, 2) for k, v in s['robot_touch'].items()} }", flush=True)
    lifted = obs["sim"]["object_poses"][part][2] - L.pose6_from_tf(part_tf)[2]
    held_torque = obs["sim"]["contact"][part]["torque_Nm"]
    held_peak = obs["sim"]["contact_peak"][part]["torque_Nm"]
    m = env.env._metrics
    k = env.env.pool.index(ids[0])
    base_fz = float(m.env_force[k, 2])
    print(f"[smoke] raw {ids[0]} contacts: {m.raw_env_contacts(env.env.names[k])}", flush=True)
    print(f"[smoke] torque check: held part in the air torque_Nm now {held_torque} (lift-chunk peak {held_peak}, "
          f"robot pinch estimate {obs['sim']['contact'][part]['torque_robot_est_Nm']:.4f}); resting {ids[0]} "
          f"non-robot contact force z {base_fz:.4f} N vs weight {env.masses[ids[0]] * 9.81:.4f} N", flush=True)
    disturb = disturbance_check(env, ids, part, above) if args.disturb else None
    hold = [*obs["tcp_pose"], 1.0]
    t0 = time.time()
    n, rest = 100, {"guard_max_Nm": 0.0, "raw_max_Nm": 0.0, "worst": None}
    for _ in range(n):
        s = env.step_tcp([hold], 0.1)["sim"]
        for u, c in s["contact_peak"].items():         # parts nobody touched this step (not the held part)
            if u == part or s["robot_touch"][u] > 0.0:
                continue
            if c["torque_raw_Nm"] > rest["raw_max_Nm"]:
                rest.update(raw_max_Nm=c["torque_raw_Nm"], worst=u)
            rest["guard_max_Nm"] = max(rest["guard_max_Nm"], c["torque_Nm"])
    step_hz = n / (time.time() - t0)
    print(f"[smoke] resting untouched parts over {n} env steps: torque_Nm max {rest['guard_max_Nm']:.5f} "
          f"(raw ungated max {rest['raw_max_Nm']:.4f} on {rest['worst']})", flush=True)
    report = {"torque_check": {"held_part_torque_Nm": held_torque, "held_part_lift_peak_Nm": held_peak,
                               "base_env_force_z_N": base_fz, "base_weight_N": env.masses[ids[0]] * 9.81,
                               "resting_untouched": rest},
              "after_reset": step0, "disturbances": disturb, "objects": ids, "pool_size": len(env.pool), "device": args.device, "contact_source": env.contact_source, "physics_dt": env.dt,
              "decimation": env.env.cfg.decimation, "build_s": round(build_s, 2), "reset_s": round(reset_s, 2),
              "env_steps_per_s_with_cameras": round(step_hz, 2),
              "physics_steps_per_s_with_cameras": round(step_hz * env.env.cfg.decimation, 1),
              "realtime_factor_with_cameras": round(step_hz * 0.1, 3), "part_lift_mm": round(lifted, 2),
              "wrist_camera": cam_checks, "masses_g": {u: round(m * 1000, 2) for u, m in env.masses.items()},
              "phases": log, "image_shapes": {c: list(i.shape) for c, i in obs["images"].items()}}
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"[smoke] {step_hz:.2f} env steps/s with 3 cameras (x{step_hz * 0.1:.3f} real time); part lifted "
          f"{lifted:.1f} mm; wrist cam {cam_checks['04_lift']}; report {args.out / 'report.json'}", flush=True)
    fails = ([f"a part held in the air reports torque_Nm {held_torque} (must be 0.0)"] if held_torque != 0.0 else []) + \
        ([f"an untouched resting part reports torque_Nm {rest['guard_max_Nm']} (must be <= 0.005)"]
         if rest["guard_max_Nm"] > 0.005 else [])
    print(f"[smoke] {'FAIL: ' + '; '.join(fails) if fails else 'PASS'}", flush=True)
    env.close()            # closing the app ends the process, so the verdict is printed first
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
