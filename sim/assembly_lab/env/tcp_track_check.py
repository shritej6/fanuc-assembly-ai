"""Stationary TCP tracking: joint error (q - q_cmd), TCP error (FK(q) - commanded) and per-joint drive torque.

    T12_POOL=... isaaclab_run.sh sim/assembly_lab/env/tcp_track_check.py [--products 1,4]

Per product: CAD grasp of M<p>_01, carry to 10 mm above its seat on F0<p>, hold 2 s, then report. Also the same
pose with empty open jaws (no part). Robot links ignore gravity, so any steady joint error is drive torque from
contacts or constraints: torque = stiffness * error. Also prints the robot link masses / inertias.
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
ap.add_argument("--products", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4])
ap.add_argument("--carry", type=lambda s: s.split(","), default=[])
args = ap.parse_args()

import layout as L
import sampler
import tcp
from cell_env import AssemblyCellEnv, CellEnvCfg

TOL_MM = 0.05
AXIAL_N = 15.0          # insertion force guard
LATERAL_N = 4.5         # lateral force guard
SLIP_MM = 0.1
PIVOT_DEG = 1.0
TILT_DEG = 3.0


def part_in_tcp(env, part: str, obs) -> np.ndarray:
    tcp_tf = L.translation(*obs["tcp_pose"][:3])
    tcp_tf[:3, :3] = L.fanuc_rotation(*obs["tcp_pose"][3:6])
    return np.linalg.inv(tcp_tf) @ env.env.object_poses_mm()[part]


def pivot_deg(a: np.ndarray, b: np.ndarray) -> float:
    r = a[:3, :3].T @ b[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def carry_test(env, part: str) -> tuple[float, float]:
    """Grasp a part from the table, lift 60 mm, carry 150 mm sideways and back: slip / pivot in the gripper."""
    lay = {"objects": [{"id": part, "x": 300.0, "y": 60.0, "yaw": 30.0}], "instruction": f"pick {part}"}

    def frames(poses):
        grasp = sampler.cad_grasp(env.solver, part, poses[part])
        return {"grasp_frame": grasp, "seat_frame": grasp, "spot_frame": grasp}

    obs = env.reset(lay, frames_fn=frames)
    g = list(obs["sim"]["frames"]["grasp_frame"])
    up = lambda p_, dz, dy=0.0: [p_[0], p_[1] + dy, p_[2] + dz, *p_[3:6]]
    env.step_tcp([[*up(g, 60), 0]], 1.0)
    env.step_tcp([[*g, 0]], 1.0)
    obs = env.step_tcp([[*g, 1]], 0.5)
    ref = part_in_tcp(env, part, obs)
    worst_s, worst_p = 0.0, 0.0
    for name, row in (("lift", up(g, 60)), ("carry", up(g, 60, -150.0)), ("back", up(g, 60))):
        obs = env.step_tcp([[*row, 1]], 1.0)
        obs = env.step_tcp([[*row, 1]], 0.5)
        cur = part_in_tcp(env, part, obs)
        s_, p_ = float(np.linalg.norm(cur[:3, 3] - ref[:3, 3])), pivot_deg(ref, cur)
        worst_s, worst_p = max(worst_s, s_), max(worst_p, p_)
        print(f"[carry] {part} {name}: slip {s_:.3f} mm pivot {p_:.2f} deg", flush=True)
    f = env.env.scene["robot"].root_view.get_dof_projected_joint_forces()
    f = np.asarray(f.numpy() if hasattr(f, "numpy") else f.cpu())[0].reshape(-1)[env.term.jaw_ids]
    sensor = env.env.scene["robot_contact"]
    pad = sensor.data.net_forces_w.torch[0].cpu().numpy()
    pads = [round(float(np.linalg.norm(pad[sensor.body_names.index(n)])), 2) for n in ("JawL", "JawR")]
    print(f"[carry] {part}: pad normal per finger N={pads} transmitted jaw force N={np.round(f, 2).tolist()} "
          f"-> {'PASS' if worst_s <= SLIP_MM and worst_p <= PIVOT_DEG else 'FAIL'}", flush=True)
    return worst_s, worst_p


def push(env, part: str, f) -> None:
    import torch

    e = env.env
    ids = torch.tensor([0], device=e.device)
    force = torch.tensor([[list(f)]], dtype=torch.float32, device=e.device)
    e.scene[e.names[e.pool.index(part)]].permanent_wrench_composer.set_forces_and_torques_index(
        forces=force, torques=torch.zeros_like(force), body_ids=ids, env_ids=ids, is_global=True)


def grip_tests(env, part: str, hold_row: list) -> float:
    """Pinch per finger, then slip of the part in the gripper under 15 N pushes and a 3 deg wrist tilt."""
    obs = env.step_tcp([hold_row], 0.5)
    q = env.env.scene["robot"].data
    tau = q.applied_torque.torch[0].cpu().numpy()[env.term.jaw_ids]
    jaw = q.joint_pos.torch[0].cpu().numpy()[env.term.jaw_ids] * 1000.0
    print(f"[grip] {part}: jaw drive force N={np.round(tau, 2).tolist()} jaws_mm={np.round(jaw, 3).tolist()}"
          f" asymmetry_mm={abs(jaw[0] + jaw[1]) / 2:.3f}", flush=True)
    robot = env.env.scene["robot"]
    view = robot.root_physx_view
    as_np = lambda a: a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a.numpy())
    proj = as_np(view.get_dof_projected_joint_forces())[0].reshape(-1)
    print(f"[grip] {part}: PhysX projected jaw joint forces N={np.round(proj[env.term.jaw_ids], 2).tolist()}", flush=True)
    sensor = env.env.scene["robot_contact"]
    fm = sensor.data.force_matrix_w.torch[0][:, env.env.names.index(env.env.names[env.env.pool.index(part)])].cpu().numpy()
    for name, f in zip(sensor.body_names, fm):
        if np.linalg.norm(f) > 0.01:
            print(f"[grip] {part}: normal force on {name} {np.round(f, 2).tolist()} N", flush=True)
    net = sensor.data.net_forces_w.torch[0].cpu().numpy()
    for name, f in zip(sensor.body_names, net):
        if np.linalg.norm(f) > 0.01:
            print(f"[grip] net normal force on {name} (all contacts) {np.round(f, 2).tolist()} N", flush=True)
    ref = part_in_tcp(env, part, obs)
    worst = 0.0
    tests = [(f"push {v}", v, hold_row) for v in ([0, 0, AXIAL_N], [0, 0, -AXIAL_N], [LATERAL_N, 0, 0],
                                                  [-LATERAL_N, 0, 0], [0, LATERAL_N, 0], [0, -LATERAL_N, 0])]
    for k in (3, 4):
        tilted = list(hold_row)
        tilted[k] += TILT_DEG
        tests.append((f"tilt {'wp'[k - 3]} +{TILT_DEG}", None, tilted))
    for name, f, row in tests:
        if f is not None:
            push(env, part, f)
        obs = env.step_tcp([row], 1.0)
        during = part_in_tcp(env, part, obs)
        if f is not None:
            push(env, part, (0.0, 0.0, 0.0))
        obs = env.step_tcp([hold_row], 1.0)
        after = part_in_tcp(env, part, obs)
        d_dur = float(np.linalg.norm(during[:3, 3] - ref[:3, 3]))
        d_aft = float(np.linalg.norm(after[:3, 3] - ref[:3, 3]))
        worst = max(worst, d_aft)
        print(f"[grip] {part} {name}: shift during {d_dur:.3f} mm, after {d_aft:.3f} mm, pivot after "
              f"{pivot_deg(ref, after):.2f} deg", flush=True)
    return worst


def report(env, tag: str) -> float:
    e, term = env.env, env.term
    robot = e.scene["robot"]
    q = robot.data.joint_pos.torch[0].cpu().double().numpy()
    q_arm = q[term.arm_ids]
    err = np.rad2deg(q_arm - np.asarray(term.q_cmd))
    fk = np.asarray(tcp.tcp_pose(q_arm))
    cmd_fk = np.asarray(tcp.tcp_pose(np.asarray(term.q_cmd)))
    d = fk[:3] - np.asarray(term.tcp_cmd[:3])
    jaw = q[term.jaw_ids] * 1000.0
    from scene import ARM_STIFFNESS
    print(f"[track] {tag}: dq_deg={np.round(err, 4).tolist()} torque_Nm={np.round(np.deg2rad(err) * ARM_STIFFNESS, 1).tolist()}"
          f" tcp_err_mm={np.round(d, 3).tolist()} |xy|={np.hypot(*d[:2]):.3f} fk(q_cmd)-cmd={np.round(cmd_fk[:3] - term.tcp_cmd[:3], 3).tolist()}"
          f" jaws_mm={np.round(jaw, 3).tolist()}", flush=True)
    return float(np.linalg.norm(d))


def masses(env) -> None:
    robot = env.env.scene["robot"]
    view = robot.root_physx_view
    as_np = lambda a: a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a.numpy())
    m = as_np(view.get_masses())[0].reshape(-1)
    inertia = as_np(view.get_inertias())[0].reshape(len(m), -1)
    for name, mi, ii in zip(robot.body_names, m, inertia):
        print(f"[mass] {name}: m={mi:.5f} kg I_diag={np.round(ii[[0, 4, 8]], 8).tolist()}", flush=True)


def episode(env, p: int) -> list[float]:
    base, part = f"F0{p}", f"M{p}_01"

    def frames(poses):
        grasp = sampler.cad_grasp(env.solver, part, poses[part])
        rel = np.linalg.inv(poses[part]) @ grasp
        return {"grasp_frame": grasp, "seat_frame": env.solver.seat_tf_for(part, base, poses) @ rel,
                "spot_frame": L.translation(60.0, -80.0, 0.0) @ grasp}

    lay = {"objects": [{"id": base, "x": 330.0, "y": -90.0, "yaw": 0.0}, {"id": part, "x": 300.0, "y": 60.0, "yaw": 30.0}],
           "instruction": f"put {part} on {base}"}
    obs = env.reset(lay, frames_fn=frames)
    g = list(obs["sim"]["frames"]["grasp_frame"])
    up = lambda p_, dz: [p_[0], p_[1], p_[2] + dz, *p_[3:6]]
    seat = list(obs["sim"]["frames"]["seat_frame"])
    out = []
    env.step_tcp([[*up(seat, 10.0), 0]], 1.0)
    env.step_tcp([[*up(seat, 10.0), 0]], 1.0)
    out.append(report(env, f"P{p} seat+10 empty open"))
    env.step_tcp([[*up(seat, 10.0), 1]], 1.0)
    env.step_tcp([[*up(seat, 10.0), 1]], 1.0)
    out.append(report(env, f"P{p} seat+10 empty closed"))
    for row, dt in (([*up(g, 60), 0], 1.0), ([*g, 0], 1.0), ([*g, 1], 0.5), ([*up(g, 60), 1], 1.0)):
        env.step_tcp([row], dt)
    env.step_tcp([[*up(g, 60), 1]], 1.0)
    out.append(report(env, f"P{p} lifted holding"))
    env.step_tcp([[*up(seat, 10.0), 1]], 1.0)
    env.step_tcp([[*up(seat, 10.0), 1]], 1.0)
    out.append(report(env, f"P{p} seat+10 holding"))
    slip = grip_tests(env, part, [*up(g, 60), 1])
    print(f"[grip] {part}: worst slip after load {slip:.3f} mm -> {'PASS' if slip <= SLIP_MM else 'FAIL'}", flush=True)
    return out


def main() -> None:
    pool = [u for p in args.products for u in (f"F0{p}", f"M{p}_01")] + args.carry
    env = AssemblyCellEnv(CellEnvCfg(device=args.device, pool=pool, cameras=()))
    masses(env)
    errs = [x for p in args.products for x in episode(env, p)]
    for part in args.carry:
        carry_test(env, part)
    print(f"[track] max tcp err {max(errs):.3f} mm -> {'PASS' if max(errs) <= TOL_MM else 'FAIL'}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
