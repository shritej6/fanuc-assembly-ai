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
from grip_load import closing_axis_world, grip_point_world_m, load_step, part_in_tcp, pivot_deg

TOL_MM = 0.05
AXIAL_N = 15.0          # insertion force guard
LATERAL_N = 4.5         # lateral force guard
SLIP_MM = 0.1
PIVOT_DEG = 1.0
TILT_DEG = 3.0
PUSH_SLIP_MM = 0.05     # no slip beyond this under the axial / lateral guard loads
TORSION_NM = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)


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
    sensor = env.env._metrics.robot
    pad = sensor.data.net_forces_w.torch[0].cpu().numpy()
    pads = [round(float(np.linalg.norm(pad[sensor.body_names.index(n)])), 2) for n in ("JawL", "JawR")]
    print(f"[carry] {part}: pad normal per finger N={pads} transmitted jaw force N={np.round(f, 2).tolist()} "
          f"-> {'PASS' if worst_s <= SLIP_MM and worst_p <= PIVOT_DEG else 'FAIL'}", flush=True)
    return worst_s, worst_p


def grip_tests(env, part: str, hold_row: list) -> float:
    """Pinch per finger, then slip of the part in the gripper: 15 N axial / 4.5 N lateral pushes at the CAD grip
    point (friction) and at the part's CoM (adds torque r x F about the grip), a 3 deg wrist tilt, and a torsion
    ramp about the closing axis (the grip's torque capacity)."""
    env.step_tcp([hold_row], 0.5)
    q = env.env.scene["robot"].data
    jaw = q.joint_pos.torch[0].cpu().numpy()[env.term.jaw_ids] * 1000.0
    robot = env.env.scene["robot"]
    as_np = lambda a: a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a.numpy())
    proj = as_np(robot.root_physx_view.get_dof_projected_joint_forces())[0].reshape(-1)[env.term.jaw_ids]
    sensor = env.env._metrics.robot
    fm = sensor.data.force_matrix_w.torch[0][:, env.env.pool.index(part)].cpu().numpy()
    pads = [round(float(np.linalg.norm(fm[sensor.body_names.index(n)])), 2) for n in ("JawL", "JawR")]
    print(f"[grip] {part}: pinch on the part per finger N={pads} transmitted jaw force N={np.round(proj, 2).tolist()}"
          f" jaws_mm={np.round(jaw, 3).tolist()} asymmetry_mm={abs(jaw[0] + jaw[1]) / 2:.3f}", flush=True)
    worst = 0.0
    loads = ([0, 0, AXIAL_N], [0, 0, -AXIAL_N], [LATERAL_N, 0, 0], [-LATERAL_N, 0, 0], [0, LATERAL_N, 0],
             [0, -LATERAL_N, 0])
    for v in loads:
        s_, _ = load_step(env, part, hold_row, f"push {v} at grip point", f=v, at_m=grip_point_world_m(env, part))
        worst = max(worst, s_)
    for k in (3, 4):
        tilted = list(hold_row)
        tilted[k] += TILT_DEG
        s_, _ = load_step(env, part, hold_row, f"tilt {'wp'[k - 3]} +{TILT_DEG}", row=tilted)
        worst = max(worst, s_)
    print(f"[grip] {part}: worst slip, grip-point loads + tilt {worst:.3f} mm -> "
          f"{'PASS' if worst <= PUSH_SLIP_MM else 'FAIL'} (<= {PUSH_SLIP_MM} mm)", flush=True)
    com_worst = 0.0
    for v in loads:
        s_, _ = load_step(env, part, hold_row, f"push {v} at CoM", f=v)
        com_worst = max(com_worst, s_)
    print(f"[grip] {part}: worst slip, CoM loads {com_worst:.3f} mm -> "
          f"{'PASS' if com_worst <= PUSH_SLIP_MM else 'FAIL'} (<= {PUSH_SLIP_MM} mm)", flush=True)
    axis = closing_axis_world(env)
    for t in TORSION_NM:
        _, piv = load_step(env, part, hold_row, f"torque {t} N m about the closing axis", f=(0.0, 0.0, 0.0),
                           torque=tuple(t * axis))
        if piv > PIVOT_DEG:
            break
    return worst


def report(env, tag: str) -> float:
    e, term = env.env, env.term
    robot = e.scene["robot"]
    q = robot.data.joint_pos.torch[0].cpu().double().numpy()
    q_arm = q[term.arm_ids]
    err = np.rad2deg(q_arm - np.asarray(term.q_cmd[0]))
    fk = np.asarray(tcp.tcp_pose(q_arm))
    cmd_fk = np.asarray(tcp.tcp_pose(np.asarray(term.q_cmd[0])))
    d = fk[:3] - np.asarray(term.tcp_cmd[0, :3])
    jaw = q[term.jaw_ids] * 1000.0
    from scene import ARM_STIFFNESS
    print(f"[track] {tag}: dq_deg={np.round(err, 4).tolist()} torque_Nm={np.round(np.deg2rad(err) * ARM_STIFFNESS, 1).tolist()}"
          f" tcp_err_mm={np.round(d, 3).tolist()} |xy|={np.hypot(*d[:2]):.3f} fk(q_cmd)-cmd={np.round(cmd_fk[:3] - term.tcp_cmd[0, :3], 3).tolist()}"
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
    grip_tests(env, part, [*up(g, 60), 1])
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
