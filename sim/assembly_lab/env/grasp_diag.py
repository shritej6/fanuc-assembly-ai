"""Grasp contact diagnostics and regression check: where the fingers touch a held part, how hard, how deep, and how
far the part slips through a lift, a carry and a pull out of a tight fit.

    isaaclab_run.sh sim/assembly_lab/env/grasp_diag.py --parts M1_02,M3_04 --pull F01:M1_01+M1_02+M1_03,F03:M3_01+M3_02

Per part: CAD grasp from the table, close, lift 60 mm, carry 150 mm and back (tcp_track_check.carry_test motion),
then an axial force ramp on the CoM (the grip's friction capacity). Per pull case: the stack seated on the base, CAD
grasp of the top part, close, pull 60 mm up. Every few env steps it logs the jaw positions, the jaw actuation and
transmitted forces, and per jaw body the contacts with the part from the PhysX contact view (normal force, deepest
penetration, point spread in the TCP frame). One [verdict] line checks LIMITS (slip, pivot, pinch, penetration).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--parts", type=lambda s: [x for x in s.split(",") if x], default=["M1_02"])
ap.add_argument("--every", type=int, default=2, help="log every n-th env step")
ap.add_argument("--pull", type=lambda s: s.split(","), default=[],
                help="pull-out cases BASE:PART+PART+...: parts seated on BASE in order, the last one is pulled out")
args = ap.parse_args()

import layout as L
import sampler
from cell_env import AssemblyCellEnv, CellEnvCfg
from grip_load import part_in_tcp, pivot_deg


def finger_contacts(env, part: str, obs_tcp) -> dict:
    """Per jaw body, contacts with the part from the robot contact sensor's PhysX view: normal force, deepest
    penetration (separation < 0) and where the points sit in the TCP frame (z along the tool, 0 = TCP; the NBR pad
    spans the last 3.7 mm above the fingertip, the PLA finger body everything above it)."""
    as_np = lambda a: a.numpy() if hasattr(a, "numpy") and not hasattr(a, "cpu") else a.cpu().numpy()
    nf = len(env.env.names)
    fi = env.env.pool.index(part)
    tcp_tf = L.translation(*obs_tcp[:3])
    tcp_tf[:3, :3] = L.fanuc_rotation(*obs_tcp[3:6])
    inv = np.linalg.inv(tcp_tf)
    out = {}
    for name in ("JawL", "JawR"):
        sensor = env.env.scene[f"robot_contact_{name}"]          # one single-body sensor per link (scene.py)
        forces, points, _normals, seps, counts, starts = (np.asarray(as_np(x)) for x in
                                                         sensor.contact_view.get_contact_data(dt=env.dt))
        counts, starts = counts.reshape(-1, nf), starts.reshape(-1, nf)
        fric, fcounts, fstarts = None, None, None
        if hasattr(sensor.contact_view, "get_friction_data"):
            fr, _, fc, fs = sensor.contact_view.get_friction_data(dt=env.dt)
            fric = np.asarray(as_np(fr)).reshape(-1, 3)
            fcounts, fstarts = np.asarray(as_np(fc)).reshape(-1, nf), np.asarray(as_np(fs)).reshape(-1, nf)
        bi = 0                                                   # env 0's (only) body
        n, s0 = int(counts[bi, fi]), int(starts[bi, fi])
        if n == 0:
            continue
        f = forces.reshape(-1)[s0:s0 + n]
        pts = (points.reshape(-1, 3)[s0:s0 + n] - (0.0, 0.0, env.env._top() + L.UF0_ABOVE_BASE_M)) * 1000.0
        loc = pts @ inv[:3, :3].T + inv[:3, 3]
        sep = seps.reshape(-1)[s0:s0 + n] * 1000.0
        out[name] = {"n": n, "normal_N": round(float(f.sum()), 2), "min_sep_mm": round(float(sep.min()), 3),
                     "tcp_xyz_min": np.round(loc.min(0), 2).tolist(), "tcp_xyz_max": np.round(loc.max(0), 2).tolist()}
        if fric is not None:
            m, f0 = int(fcounts[bi, fi]), int(fstarts[bi, fi])
            out[name]["friction_N_world"] = np.round(fric[f0:f0 + m].sum(0), 2).tolist()
    return out


def log_step(env, part: str, tag: str) -> None:
    robot = env.env.scene["robot"]
    j = env.term.jaw_ids
    q = robot.data.joint_pos.torch[0, j].cpu().numpy() * 1000.0
    view = robot.root_physx_view
    as_np = lambda a: a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a.numpy())
    pf = as_np(view.get_dof_projected_joint_forces())[0].reshape(-1)[j]
    act = as_np(view.get_dof_actuation_forces())[0].reshape(-1)[j]
    contacts = finger_contacts(env, part, env.env.tcp_obs())
    print(f"[diag] {tag}: jaws_mm={np.round(q, 3).tolist()} actuation_N={np.round(act, 2).tolist()} "
          f"transmitted_N={np.round(pf, 2).tolist()} contacts={contacts}", flush=True)
    if "_hold" in tag:
        pinch = min((contacts.get(n, {}).get("normal_N", 0.0) for n in ("JawL", "JawR")), default=0.0)
        sep = min((c["min_sep_mm"] for c in contacts.values()), default=0.0)
        worst("pinch_min_N", pinch, min)
        worst("penetration_max_mm", -sep, max)


VERDICT: dict[str, float] = {}
LIMITS = {"slip_max_mm": ("<=", 0.1), "pivot_max_deg": ("<=", 1.0), "pinch_min_N": (">=", 30.0),
          "penetration_max_mm": ("<=", 0.2)}


def worst(key: str, value: float, pick=max) -> None:
    VERDICT[key] = pick(VERDICT.get(key, value), value)


def run(env, part: str, rows: list, dt: float, tag: str) -> dict:
    """step_tcp, logging every args.every env steps."""
    import tcp
    import torch

    n = max(1, round(dt / env.cfg.step_dt))
    for row in rows:
        for k, pose in enumerate(tcp.interpolate(env.term.tcp_cmd[0], np.asarray(row, float), n)):
            action = torch.tensor([[*pose, row[6]]], dtype=torch.float32, device=env.env.device)
            env._obs, *_ = env.env.step(action)
            if k % args.every == 0 or k == n - 1:
                log_step(env, part, f"{part} {tag} {k}")
    return env._observe()


def diag(env, part: str) -> None:
    lay = {"objects": [{"id": part, "x": 300.0, "y": 60.0, "yaw": 30.0}], "instruction": f"pick {part}"}

    def frames(poses):
        grasp = sampler.cad_grasp(env.solver, part, poses[part])
        return {"grasp_frame": grasp, "seat_frame": grasp, "spot_frame": grasp}

    obs = env.reset(lay, frames_fn=frames)
    g = list(obs["sim"]["frames"]["grasp_frame"])
    up = lambda p_, dz, dy=0.0: [p_[0], p_[1] + dy, p_[2] + dz, *p_[3:6]]
    env.step_tcp([[*up(g, 60), 0]], 1.0)
    env.step_tcp([[*g, 0]], 1.0)
    obs = run(env, part, [[*g, 1]], 0.5, "close")
    ref = part_in_tcp(env, part, obs)
    for name, row in (("lift", up(g, 60)), ("carry", up(g, 60, -150.0)), ("back", up(g, 60))):
        obs = run(env, part, [[*row, 1]], 1.0, name)
        obs = run(env, part, [[*row, 1]], 0.5, name + "_hold")
        cur = part_in_tcp(env, part, obs)
        worst("slip_max_mm", float(np.linalg.norm(cur[:3, 3] - ref[:3, 3])))
        worst("pivot_max_deg", pivot_deg(ref, cur))
        print(f"[slip] {part} {name}: slip {np.linalg.norm(cur[:3, 3] - ref[:3, 3]):.3f} mm "
              f"(tcp-frame d={np.round(cur[:3, 3] - ref[:3, 3], 3).tolist()}) pivot {pivot_deg(ref, cur):.2f} deg",
              flush=True)
    axial_ramp(env, part, up(g, 60))


def jaw_dof_props(env) -> None:
    view = env.env.scene["robot"].root_physx_view
    j = env.term.jaw_ids
    for name in ("get_dof_friction_properties", "get_dof_friction_coefficients", "get_dof_armatures",
                 "get_dof_max_forces", "get_dof_stiffnesses", "get_dof_dampings"):
        if hasattr(view, name):
            v = getattr(view, name)()
            v = (v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v.numpy()))[0]
            print(f"[dof] {name}: {np.round(v.reshape(len(env.env.scene['robot'].joint_names), -1)[j], 5).tolist()}",
                  flush=True)


def axial_ramp(env, part: str, row: list) -> None:
    """Friction capacity of the grip: growing axial force (world z, both signs) on the part's CoM, 1 s each, with the
    finger contacts (normal and friction force) logged while the load acts."""
    from grip_load import push

    for f in (5.0, 10.0, 15.0, 20.0, 30.0):
        for sign in (1.0, -1.0):
            obs = env.step_tcp([[*row, 1]], 0.1)
            ref = part_in_tcp(env, part, obs)
            push(env, part, (0.0, 0.0, sign * f))
            obs = run(env, part, [[*row, 1]], 1.0, f"axial {sign * f:+.0f} N")
            push(env, part, (0.0, 0.0, 0.0))
            cur = part_in_tcp(env, part, obs)
            s_ = float(np.linalg.norm(cur[:3, 3] - ref[:3, 3]))
            print(f"[axial] {part} {sign * f:+.0f} N at CoM: shift {s_:.3f} mm pivot {pivot_deg(ref, cur):.2f} deg",
                  flush=True)
            if s_ > 1.0:
                return


def pull_diag(env, case: str) -> None:
    """Pull the top part of a seated stack out of its fit (fix_above): slide in the fingers, and how far closing the
    jaws moved the parts below it."""
    base, stack = case.split(":")
    stack = stack.split("+")
    part = stack[-1]
    lay = {"objects": [{"id": base, "x": 330.0, "y": -90.0, "yaw": 0.0}] +
           [{"id": p_, "state": f"on:{base}"} for p_ in stack], "instruction": f"pull {part}"}

    def frames(poses):
        grasp = sampler.cad_grasp(env.solver, part, poses[part])
        return {"grasp_frame": grasp, "seat_frame": grasp, "spot_frame": grasp}

    obs = env.reset(lay, frames_fn=frames)
    g = list(obs["sim"]["frames"]["grasp_frame"])
    up = lambda p_, dz: [p_[0], p_[1], p_[2] + dz, *p_[3:6]]
    env.step_tcp([[*up(g, 60), 0]], 1.0)
    env.step_tcp([[*g, 0]], 1.0)
    before = env.env.object_poses_mm()
    obs = run(env, part, [[*g, 1]], 0.5, "close")
    after = env.env.object_poses_mm()
    for p_ in [base, *stack[:-1]]:
        print(f"[pull] {case}: closing moved {p_} {np.linalg.norm(after[p_][:3, 3] - before[p_][:3, 3]):.3f} mm "
              f"{pivot_deg(before[p_], after[p_]):.3f} deg", flush=True)
    ref = part_in_tcp(env, part, obs)
    for name, dz in (("pull", 60.0),):
        obs = run(env, part, [[*up(g, dz), 1]], 1.0, name)
        obs = run(env, part, [[*up(g, dz), 1]], 0.5, name + "_hold")
        cur = part_in_tcp(env, part, obs)
        worst("slip_max_mm", float(np.linalg.norm(cur[:3, 3] - ref[:3, 3])))
        worst("pivot_max_deg", pivot_deg(ref, cur))
        print(f"[pull] {case} {name}: slide {np.linalg.norm(cur[:3, 3] - ref[:3, 3]):.3f} mm "
              f"(tcp-frame d={np.round(cur[:3, 3] - ref[:3, 3], 3).tolist()}) pivot {pivot_deg(ref, cur):.2f} deg; "
              f"base moved {np.linalg.norm(env.env.object_poses_mm()[base][:3, 3] - before[base][:3, 3]):.3f} mm",
              flush=True)


def main() -> None:
    pool = list(args.parts) + [u for c in args.pull for u in [c.split(":")[0], *c.split(":")[1].split("+")]]
    env = AssemblyCellEnv(CellEnvCfg(device=args.device, pool=list(dict.fromkeys(pool)), cameras=()))
    jaw_dof_props(env)
    for part in args.parts:
        diag(env, part)
    for case in args.pull:
        pull_diag(env, case)
    ok = all((VERDICT[k] <= v) if op == "<=" else (VERDICT[k] >= v) for k, (op, v) in LIMITS.items() if k in VERDICT)
    print(f"[verdict] {'PASS' if ok and set(LIMITS) <= set(VERDICT) else 'FAIL'} "
          f"{ {k: round(v, 3) for k, v in VERDICT.items()} } limits {LIMITS}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
