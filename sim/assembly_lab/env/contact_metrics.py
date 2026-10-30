"""Per-object contact numbers from two Isaac Lab contact sensors, accumulated every physics step on the sim device.

- robot_contact: every robot link (arm links, gripper body, both finger bodies) against every pool object
  (force matrix + mean contact point per link-object pair).
- object_contact: net contact force on every object.

Per object:
- robot_touch_N: the largest force any single robot link puts on it (the "touched only the target" gate).
- jaw_n: (JawL, JawR) force magnitudes (grasp signal).
- contact: the force from everything except the robot (net minus all robot-link forces): normal_N = |vertical|
  (insertion is vertical), lateral_N = horizontal magnitude, torque_Nm = |torque of the robot-link forces about the
  object's centre of mass| (quasi-static: it balances what the base and table put on the part).
  torque_Nm: torque from the NON-robot contacts only (other objects, table, mat, floor; every contact point and
  friction anchor, contact_torque.py) about the object's insert axis (body +Z), the axis physics/limits.yaml caps;
  a part held in the air reads 0. It is gated to parts the robot loads this physics step (a robot link touches it,
  or it touches a part a robot link touches): an untouched part at rest reads 0 (measured by agent 3 in run7: the
  ungated value read 0.12-2.33 N m on a resting M1_04). torque_raw_Nm: the ungated value (logging only).
  torque3d_Nm: full magnitude of the gated torque. torque_robot_est_Nm: the old estimate (|torque of the robot-link
  forces about the COM|, logging only: the pinch couple fires in the air).
Peaks are kept per env step (reset by begin_step) and per episode chunk (reset by the thin layer).
"""

from __future__ import annotations

import contact_torque
import isaaclab.utils.math as M
import torch

MAX_ARM_M = 0.1      # a contact point farther than this from the object COM is a sensor artefact (no contact)


class ContactMetrics:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot_contact"]
        self.obj = env.scene["object_contact"]
        names = env.names
        links = list(self.robot.body_names)
        self.jaw_rows = [links.index("JawL"), links.index("JawR")]
        order = list(self.obj.body_names)
        self.net_index = torch.tensor([order.index(n) for n in names], device=env.device)
        n = len(names)
        self.envc = env.scene["object_env_contact"]
        env_order = list(self.envc.body_names)
        self.env_index = torch.tensor([env_order.index(n) for n in names], device=env.device)
        k = 8      # normal, lateral, torque_z (gated), touch, torque3d (gated), robot pinch est, coverage, raw torque_z
        self._step_peak = torch.zeros((n, k), device=env.device)
        self._chunk_peak = torch.zeros((n, k), device=env.device)
        self._now = torch.zeros((n, k), device=env.device)
        self.env_force = torch.zeros((n, 3), device=env.device)
        self._jaw = torch.zeros((n, 2), device=env.device)

    def _compute(self) -> None:
        f = self.robot.data.force_matrix_w.torch[0]                 # (B, N, 3) force on robot link b from object n
        p = self.robot.data.contact_pos_w.torch[0]                  # (B, N, 3)
        net = self.obj.data.net_forces_w.torch[0][self.net_index]   # (N, 3)
        com = torch.stack([self.env.scene[n].data.root_com_pos_w.torch[0] for n in self.env.names])   # (N, 3)
        quat = torch.stack([self.env.scene[n].data.root_link_quat_w.torch[0] for n in self.env.names])  # (N, 4) xyzw
        axis = M.matrix_from_quat(quat)[:, :, 2]                                                        # body +Z
        on_obj = -f                                                 # robot-link forces on the objects
        env_f = net - on_obj.sum(dim=0)
        arm = p - com.unsqueeze(0)
        ok = torch.isfinite(arm).all(dim=-1) & (arm.norm(dim=-1) < MAX_ARM_M) & (f.norm(dim=-1) > 0)
        tau = torch.where(ok.unsqueeze(-1), torch.cross(torch.nan_to_num(arm), on_obj, dim=-1), 0.0).sum(dim=0)
        touch = f.norm(dim=-1).max(dim=0).values
        raw_z, tau3, self.env_force = self._env_torque(com, axis)
        pair = self.envc.data.force_matrix_w.torch[0][self.env_index][:, : len(self.env.names)].norm(dim=-1)
        loaded = contact_torque.robot_loaded(touch, pair)
        tau_z = contact_torque.guard_torque(raw_z, loaded)
        tau3 = tau3 * loaded.unsqueeze(-1)
        # coverage: |sum of the per-contact non-robot forces| / |net non-robot force|. PhysX's per-contact buffer
        # under-reports (measured 0.47 for a resting base: 0.137 N of 0.290 N), so torque_Nm may be low by that much
        cover = torch.where(env_f.norm(dim=-1) > 0.02, self.env_force.norm(dim=-1) / env_f.norm(dim=-1).clamp(min=1e-6),
                            torch.ones_like(tau_z))
        self._now = torch.stack([env_f[:, 2].abs(), env_f[:, :2].norm(dim=-1), tau_z, touch, tau3.norm(dim=-1),
                                 tau.norm(dim=-1), cover, raw_z], dim=-1)
        self._jaw = f[self.jaw_rows].norm(dim=-1).T

    def _env_torque(self, com: torch.Tensor, axis: torch.Tensor):
        """Torque on every object from its non-robot contacts, in self.env.names order."""
        import warp as wp

        view, dt = self.envc.contact_view, self.env.physics_dt
        fn, pts, nrm, _, cnt, start = view.get_contact_data(dt=dt)
        fr, fpts, fcnt, fstart = view.get_friction_data(dt=dt)
        t = lambda a, d=3: wp.to_torch(a).reshape(-1, d) if d else wp.to_torch(a).to(torch.int64)
        order = self.env_index
        tz, tau, force = contact_torque.env_torque(
            t(fn, 1), t(pts), t(nrm), t(cnt, 0), t(start, 0), t(fr), t(fpts), t(fcnt, 0), t(fstart, 0),
            torch.empty((len(order), 3), device=com.device).index_copy_(0, order, com),
            torch.empty((len(order), 3), device=com.device).index_copy_(0, order, axis), max_arm=MAX_ARM_M)
        return tz[order], tau[order], force[order]

    def raw_env_contacts(self, name: str) -> dict:
        """Every non-robot contact point and friction anchor on object `name` now (diagnostics)."""
        import warp as wp

        view, dt = self.envc.contact_view, self.env.physics_dt
        fn, pts, nrm, sep, cnt, start = view.get_contact_data(dt=dt)
        fr, _fpts, fcnt, fstart = view.get_friction_data(dt=dt)
        i = list(self.envc.body_names).index(name)
        c, s0 = wp.to_torch(cnt).to(torch.int64)[i], wp.to_torch(start).to(torch.int64)[i]
        fc, fs0 = wp.to_torch(fcnt).to(torch.int64)[i], wp.to_torch(fstart).to(torch.int64)[i]
        T = lambda a, d: wp.to_torch(a).reshape(-1, d)
        out = {"normal": [], "friction": []}
        for j in range(c.numel()):
            for k in range(int(s0[j]), int(s0[j] + c[j])):
                out["normal"].append({"filter": j, "fn": round(float(T(fn, 1)[k, 0]), 4),
                                      "n": [round(float(v), 3) for v in T(nrm, 3)[k]],
                                      "p": [round(float(v), 4) for v in T(pts, 3)[k]],
                                      "sep": round(float(T(sep, 1)[k, 0]) * 1000, 3)})
            for k in range(int(fs0[j]), int(fs0[j] + fc[j])):
                out["friction"].append({"filter": j, "f": [round(float(v), 4) for v in T(fr, 3)[k]]})
        return out

    def accumulate(self) -> None:
        self._compute()
        self._step_peak = torch.maximum(self._step_peak, self._now)
        self._chunk_peak = torch.maximum(self._chunk_peak, self._now)

    def begin_step(self) -> None:
        self._step_peak.zero_()

    def reset_episode(self) -> None:
        self._step_peak.zero_()
        self._chunk_peak.zero_()

    def reset_chunk(self) -> None:
        self._chunk_peak.zero_()

    def peak_vector(self) -> torch.Tensor:
        """(1, 4): max over objects of [normal_N, lateral_N, torque_Nm (insert axis), robot_touch_N], last env step."""
        return self._step_peak[:, :4].max(dim=0).values.unsqueeze(0)

    def current(self) -> dict:
        self._compute()
        now, jaw = self._now.cpu().tolist(), self._jaw.cpu().tolist()
        return {"touch_n": {n: now[k][3] for k, n in enumerate(self.env.names)},
                "jaw_n": {n: tuple(jaw[k]) for k, n in enumerate(self.env.names)},
                "contact": {n: self._named(now[k]) for k, n in enumerate(self.env.names)}}

    @staticmethod
    def _named(row: list[float]) -> dict:
        return {"normal_N": row[0], "lateral_N": row[1], "torque_Nm": row[2], "torque3d_Nm": row[4],
                "torque_robot_est_Nm": row[5], "env_contact_coverage": row[6], "torque_raw_Nm": row[7]}

    def chunk(self) -> dict:
        peak = self._chunk_peak.cpu().tolist()
        return {"contact_peak": {n: self._named(peak[k]) for k, n in enumerate(self.env.names)},
                "robot_touch": {n: peak[k][3] for k, n in enumerate(self.env.names)}}
