"""Seating torque on each object from its NON-robot contacts (other objects, table, mat, floor), per contact point.

PhysX reports, for every sensor body i and filter shape j, a run of contact points (normal force magnitude, point,
normal) and a run of friction anchors (friction force vector, point), each run located by count[i, j] / start[i, j]
in a flat buffer (RigidContactView.get_contact_data / get_friction_data). The torque on body i about its centre of
mass is sum (p - com) x F over both runs; torque_Nm is its component along the body's insert axis (body +Z).

A part held in the air has no non-robot contact, so its torque is exactly 0 (the robot's pinch couple, which the old
estimate counted, is not a seating load). Contact points farther than `max_arm` from the COM cannot lie on the part
and are dropped (buffer artefacts). The guard value is gated by `robot_loaded`: a part the robot is not loading (not
touched, not touching a touched part) is at rest under its own weight and reads 0; the ungated value is kept as the
raw reading for logging. Works on numpy or torch arrays, so it is unit-tested without Isaac Sim.
"""

from __future__ import annotations

import numpy as np

# Force on the sensor body = SIGN * normal * normal_force (checked in sim: a resting part's sum is +m g up).
NORMAL_SIGN = 1.0


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def _runs(counts, starts):
    """(owner body index, flat buffer index) of every entry of the per-(i, j) runs."""
    n, m = counts.shape
    flat_c, flat_s = counts.reshape(-1), starts.reshape(-1)
    if _is_torch(counts):
        import torch

        pair = torch.repeat_interleave(torch.arange(n * m, device=counts.device), flat_c.long())
        first = torch.cumsum(flat_c.long(), 0) - flat_c.long()
        offs = torch.arange(pair.numel(), device=counts.device) - first[pair]
        return pair // m, flat_s.long()[pair] + offs
    pair = np.repeat(np.arange(n * m), flat_c.astype(int))
    first = np.cumsum(flat_c.astype(int)) - flat_c.astype(int)
    offs = np.arange(pair.size) - first[pair]
    return pair // m, flat_s.astype(int)[pair] + offs


def _scatter_sum(owner, values, n):
    if _is_torch(values):
        import torch

        out = torch.zeros((n, 3), dtype=values.dtype, device=values.device)
        return out.index_add_(0, owner, values)
    out = np.zeros((n, 3))
    np.add.at(out, owner, values)
    return out


def _cross(a, b):
    if _is_torch(a):
        import torch

        return torch.cross(a, b, dim=-1)
    return np.cross(a, b)


def _near(arm, max_arm):
    """(K, 1) 1.0 where the lever arm is a plausible point on the part, else 0.0."""
    if max_arm is None:
        return 1.0
    d = (arm * arm).sum(-1) ** 0.5
    return (d < max_arm)[:, None] * 1.0


def env_torque(normal_f, points, normals, n_counts, n_starts, friction, f_points, f_counts, f_starts, com, axis,
               max_arm=None):
    """Returns (torque about the insert axis |tau . axis| (N,), full torque (N, 3), contact force sum (N, 3)).

    normal_f (C,) or (C, 1), points / normals (C, 3), friction / f_points (F, 3), counts / starts (N, M),
    com / axis (N, 3) world frame (axis unit); contacts farther than max_arm (m) from the COM are dropped."""
    n = com.shape[0]
    owner, idx = _runs(n_counts, n_starts)
    fn = normal_f.reshape(-1)[idx]
    arm = points[idx] - com[owner]
    f_normal = NORMAL_SIGN * normals[idx] * fn[:, None] * _near(arm, max_arm)
    tau = _scatter_sum(owner, _cross(arm, f_normal), n)
    force = _scatter_sum(owner, f_normal, n)
    f_owner, f_idx = _runs(f_counts, f_starts)
    f_arm = f_points[f_idx] - com[f_owner]
    f_fric = friction[f_idx] * _near(f_arm, max_arm)
    tau = tau + _scatter_sum(f_owner, _cross(f_arm, f_fric), n)
    force = force + _scatter_sum(f_owner, f_fric, n)
    about = (tau * axis).sum(-1)
    return abs(about), tau, force


def robot_loaded(touch_n, pair_n, eps=1e-3):
    """(..., N) bool: the robot loads object i this physics step, i.e. a robot link touches it (touch_n[i] > eps) or it
    touches an object a robot link touches (pair_n[i, j] > eps, the contact force between objects i and j). Leading
    dims (envs) broadcast: touch_n (..., N), pair_n (..., N, N)."""
    touched = touch_n > eps
    return touched | ((pair_n > eps) & touched[..., None, :]).any(-1)


def guard_torque(raw_tz, loaded):
    """The torque the 0.10 N m guard sees: the raw non-robot torque where the robot loads the part, else 0."""
    return raw_tz * loaded
