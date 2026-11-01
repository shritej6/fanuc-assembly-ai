"""Compute loads on a part held in the gripper and its slip in the TCP frame.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import layout as L
import numpy as np


def part_in_tcp(env, part: str, obs) -> np.ndarray:
    tcp_tf = L.translation(*obs["tcp_pose"][:3])
    tcp_tf[:3, :3] = L.fanuc_rotation(*obs["tcp_pose"][3:6])
    return np.linalg.inv(tcp_tf) @ env.env.object_poses_mm()[part]


def pivot_deg(a: np.ndarray, b: np.ndarray) -> float:
    r = a[:3, :3].T @ b[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def push(env, part: str, f, at_m=None, torque=(0.0, 0.0, 0.0)) -> None:
    """Permanent world-frame force on the part at its CoM, or at world point at_m (m), plus a world torque."""
    import torch

    e = env.env
    ids = torch.tensor([0], device=e.device)
    force = torch.tensor([[list(f)]], dtype=torch.float32, device=e.device)
    pos = None if at_m is None else torch.tensor([[list(at_m)]], dtype=torch.float32, device=e.device)
    tq = torch.tensor([[list(torque)]], dtype=torch.float32, device=e.device)
    e.scene[e.names[e.pool.index(part)]].permanent_wrench_composer.set_forces_and_torques_index(
        forces=force, torques=tq, positions=pos, body_ids=ids, env_ids=ids, is_global=True)


def grip_point_world_m(env, part: str) -> np.ndarray:
    """The part's CAD pad point (objects.yaml grips[0].pad_point_local_mm, body frame) in world metres."""
    import yaml

    pad = yaml.safe_load(L.OBJECTS_YAML.read_text())["objects"][L.type_id(part)]["grips"][0]["pad_point_local_mm"]
    p = env.env.object_poses_mm()[part] @ np.array([*pad, 1.0])
    return L.uf0_to_world_m(L.translation(*p[:3]), env.env._top())[:3, 3]


def closing_axis_world(env) -> np.ndarray:
    robot = env.env.scene["robot"]
    bodies = robot.data.body_pos_w.torch[0].cpu().numpy()
    axis = bodies[robot.body_names.index("JawL")] - bodies[robot.body_names.index("JawR")]
    return axis / np.linalg.norm(axis)


def load_step(env, part: str, hold_row: list, name: str, row=None, **wrench) -> tuple[float, float]:
    """Apply one load for 1 s, release, hold 1 s: (shift, pivot) of the part in the TCP frame, from its pose just
    before this load (each load is judged on its own, not on the drift of the earlier ones)."""
    obs = env.step_tcp([hold_row], 0.1)
    ref = part_in_tcp(env, part, obs)
    if wrench:
        push(env, part, **wrench)
    obs = env.step_tcp([row or hold_row], 1.0)
    during = part_in_tcp(env, part, obs)
    if wrench:
        push(env, part, (0.0, 0.0, 0.0))
    obs = env.step_tcp([hold_row], 1.0)
    after = part_in_tcp(env, part, obs)
    d_dur, d_aft = (float(np.linalg.norm(x[:3, 3] - ref[:3, 3])) for x in (during, after))
    print(f"[grip] {part} {name}: shift during {d_dur:.3f} mm, after {d_aft:.3f} mm, pivot after "
          f"{pivot_deg(ref, after):.2f} deg", flush=True)
    return d_aft, pivot_deg(ref, after)
