"""Check the batched cell env by running the same scripted pick, insert and pull-out in every cell.
Authored by Shritej Gunda.
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
ap.add_argument("--num_envs", type=int, default=1)
ap.add_argument("--cameras", nargs="*", default=[])
ap.add_argument("--varied", action="store_true")
ap.add_argument("--skill", choices=("pick_insert", "pull_out"), default="pick_insert")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

import layout as L
import sampler
from cell_env import AssemblyCellEnv, CellEnvCfg

from sim.assembly_lab.mimic import gates as G
from sim.assembly_lab.mimic.force_guard import load_limits

BASE, PART = "F04", "M4_01"
SPOT_XY = (300.0, 120.0)        # pull_out drop spot (free test-bed area), UF0 mm


def layout(k: int) -> dict:
    dx = float(k) if args.varied else 0.0
    part = ({"id": PART, "state": f"on:{BASE}"} if args.skill == "pull_out"
            else {"id": PART, "x": 300.0 + dx, "y": 60.0, "yaw": 30.0})
    return {"objects": [{"id": BASE, "x": 330.0 + dx, "y": -90.0, "yaw": 0.0}, part],
            "instruction": f"{args.skill} {PART}"}


def frames_fn(env, k: int):
    dx = float(k) if args.varied else 0.0

    def frames(poses):
        grasp = sampler.cad_grasp(env.solver, PART, poses[PART])
        rel = np.linalg.inv(poses[PART]) @ grasp
        spot = env.solver.table_tf(L.type_id(PART), SPOT_XY[0] + dx, SPOT_XY[1], 0.0, "installed") @ rel
        return {"grasp_frame": grasp, "seat_frame": env.solver.seat_tf_for(PART, BASE, poses) @ rel,
                "spot_frame": spot}
    return frames


def _tf(pose6) -> np.ndarray:
    tf = L.translation(*pose6[:3])
    tf[:3, :3] = L.fanuc_rotation(*pose6[3:6])
    return tf


def gates(rec: dict, part0, grasp0, goal_frame, base_trace: list, limits) -> dict:
    """mimic/gates.py on one env's phases: final part pose vs the pose the goal frame implies, forces, others, touch."""
    rep = G.GateReport(ok=False)
    expected = L.pose6_from_tf(goal_frame @ np.linalg.inv(grasp0) @ part0)
    G.gate_final_pose(rep, rec["phases"]["retreat"]["part_pose"], expected, G.GateLimits())
    trace = [[p["peak"]["normal_N"], p["peak"]["lateral_N"], p["peak"]["torque_Nm"]] for p in rec["phases"].values()]
    G.gate_forces(rep, np.array(trace), limits)
    G.gate_others(rep, {BASE: np.array(base_trace)}, PART, G.GateLimits())
    G.gate_touch(rep, [BASE if p["touch_base_N"] > 0.05 else None for p in rec["phases"].values()], PART)
    G.finish(rep)
    return {"ok": rep.ok, "failed": rep.failed, "values": rep.values}


def summary(obs: dict) -> dict:
    s = obs["sim"]
    err = np.asarray(obs["tcp_pose"][:3]) - np.asarray(s["tcp_cmd"][:3])
    return {"tcp_err_mm": round(float(np.linalg.norm(err)), 4), "jaw_n": [round(v, 3) for v in s["jaw_n"][PART]],
            "peak": {k: round(s["contact_peak"][PART][k], 4) for k in ("normal_N", "lateral_N", "torque_Nm")},
            "touch_base_N": round(s["robot_touch"][BASE], 4), "signals": s["signals"],
            "latched": s["signals_latched"], "part_pose": s["object_poses"][PART], "base_pose": s["object_poses"][BASE],
            "ik_error": s.get("ik_error")}


def per_env_reset(env, ids: list[int], phases, limits) -> dict:
    """Odd envs reset on their own (begin_reset) while the even envs keep stepping a hold, then replay the episode
    alone (even envs not listed: they hold). Their results should match the lockstep run."""
    odd, even = [k for k in ids if k % 2], [k for k in ids if not k % 2]
    env.begin_reset({k: (layout(k), frames_fn(env, k)) for k in odd})
    hold = {k: env.term.tcp_cmd[k].tolist() + [float(env.term.grip_cmd[k])] for k in even}
    ready, hold_steps = {}, 0
    while set(ready) != set(odd):
        env.step_tcp_many({k: ([hold[k]], 0.1) for k in even})
        hold_steps += 1
        ready.update(env.ready_resets())
    g = {k: list(ready[k]["sim"]["frames"]["grasp_frame"]) for k in odd}
    goal = "spot_frame" if args.skill == "pull_out" else "seat_frame"
    rec = {k: {"settled": ready[k]["sim"]["settled"], "settle_s": ready[k]["sim"]["settle_s"], "phases": {}} for k in odd}
    start = {k: (_tf(ready[k]["sim"]["object_poses"][PART]), _tf(g[k]), _tf(ready[k]["sim"]["frames"][goal]))
             for k in odd}
    base_trace = {k: [ready[k]["sim"]["object_poses"][BASE]] for k in odd}
    for name, rows, dt in phases(g, {k: list(ready[k]["sim"]["frames"][goal]) for k in odd}):
        out = env.step_tcp_many({k: (rows(k), dt) for k in odd})
        for k in odd:
            rec[k]["phases"][name] = summary(out[k])
            base_trace[k].append(rec[k]["phases"][name]["base_pose"])
    for k in odd:
        rec[k]["gates"] = gates(rec[k], *start[k], base_trace[k], limits)
    print(f"[batch_check] per-env reset of {odd} took {hold_steps} hold steps of the others; gates ok "
          f"{[rec[k]['gates']['ok'] for k in odd]}", flush=True)
    return {"hold_steps": hold_steps, "envs": rec}


def make_phases(g: dict, seat: dict) -> list:
    up = lambda p, dz: [p[0], p[1], p[2] + dz, *p[3:6]]
    return [("approach", lambda k: [[*up(g[k], 60), 0]], 1.0), ("descend", lambda k: [[*g[k], 0]], 1.0),
            ("close", lambda k: [[*g[k], 1]], 0.5), ("lift", lambda k: [[*up(g[k], 60), 1]], 1.0),
            ("carry", lambda k: [[*up(seat[k], 10), 1]], 1.0), ("hold_above", lambda k: [[*up(seat[k], 10), 1]], 1.0),
            ("insert", lambda k: [[*seat[k], 1]], 1.0), ("release", lambda k: [[*seat[k], 0]], 0.5),
            ("retreat", lambda k: [[*up(seat[k], 40), 0]], 1.0)]


def main() -> None:
    env = AssemblyCellEnv(CellEnvCfg(device=args.device, pool=[BASE, PART], cameras=tuple(args.cameras),
                                     num_envs=args.num_envs, skill=args.skill))
    ids = list(range(args.num_envs))
    t0 = time.time()
    obs = env.reset_many([layout(k) for k in ids], [frames_fn(env, k) for k in ids])
    reset_s = time.time() - t0
    rec = {k: {"settled": obs[k]["sim"]["settled"], "settle_s": obs[k]["sim"]["settle_s"], "phases": {}} for k in ids}
    g = {k: list(obs[k]["sim"]["frames"]["grasp_frame"]) for k in ids}
    goal_key = "spot_frame" if args.skill == "pull_out" else "seat_frame"
    seat = {k: list(obs[k]["sim"]["frames"][goal_key]) for k in ids}
    start = {k: (_tf(obs[k]["sim"]["object_poses"][PART]), _tf(g[k]),
                 _tf(seat[k])) for k in ids}
    base_trace = {k: [obs[k]["sim"]["object_poses"][BASE]] for k in ids}
    steps, t1 = 0, time.time()
    for name, rows, dt in make_phases(g, seat):
        out = env.step_tcp_many({k: (rows(k), dt) for k in ids})
        steps += max(1, round(dt / env.cfg.step_dt))
        for k in ids:
            rec[k]["phases"][name] = summary(out[k])
            base_trace[k].append(rec[k]["phases"][name]["base_pose"])
    wall = time.time() - t1
    limits = load_limits(L.REPO / "sim" / "assembly_lab" / "physics" / "limits.yaml")
    for k in ids:
        rec[k]["gates"] = gates(rec[k], *start[k], base_trace[k], limits)
    rerun = per_env_reset(env, ids, make_phases, limits) if args.num_envs > 1 else {}
    rep = {"num_envs": args.num_envs, "skill": args.skill, "cameras": list(args.cameras), "varied": args.varied,
           "per_env_reset": rerun,
           "reset_wall_s": round(reset_s, 2),
           "process_steps": steps, "wall_s": round(wall, 2), "env_steps_per_s": round(steps * args.num_envs / wall, 3),
           "envs": rec}
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, indent=1, default=float))
    print(f"[batch_check] N={args.num_envs} cams={args.cameras} reset {reset_s:.1f} s, {steps} steps in {wall:.1f} s -> "
          f"{rep['env_steps_per_s']} env-steps/s", flush=True)
    for name in rec[0]["phases"]:
        print(f"[batch_check] env0 {name}: {json.dumps(rec[0]['phases'][name])}", flush=True)
    print(f"[batch_check] gates ok per env: {[rec[k]['gates']['ok'] for k in ids]}; env0 {rec[0]['gates']}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
