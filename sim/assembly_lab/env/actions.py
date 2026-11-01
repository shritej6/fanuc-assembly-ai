"""act_tcp as an Isaac Lab action term: one action = one CellClient.act_tcp row [x, y, z, w, p, r, gripper]
(UF0 mm / deg, gripper 0 open .. 1 closed), reached over one env step (CellEnvCfg.step_dt, default 0.1 s, the
shortest dt the real cell accepts). Inside the step the TCP moves on a straight line (orientation slerp) from the
previous commanded row, with IK every `ik_every` physics steps and PhysX joint PD in between, like step_tcp.

Rows outside the PLAN 3.1 workspace box are clipped to it and counted in `clipped` (the real cell would refuse them;
scripted demos never send one, Mimic noise can).

Batched over the env's cells: one action row per env; q_cmd / tcp_cmd (E, 6), grip_cmd / clipped / force_open (E,).
IK runs per env (numpy); the jaw squeeze is vectorised. With one env an IK failure raises TcpError as before;
with several, the failing env holds its last joint command and the error is kept in `ik_errors[env]` for the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import tcp
import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class TcpAction(ActionTerm):
    cfg: TcpActionCfg

    def __init__(self, cfg: TcpActionCfg, env):
        super().__init__(cfg, env)
        names = list(self._asset.joint_names)
        self.arm_ids = [names.index(f"J{i}") for i in range(1, 7)]
        self.jaw_ids = [names.index("jaw_l"), names.index("jaw_r")]
        n = env.num_envs
        self._raw = torch.zeros((n, 7), device=self.device)
        self.q_cmd = np.tile(np.asarray(cfg.home_rad, float), (n, 1))
        self.tcp_cmd = np.tile(np.asarray(tcp.tcp_pose(self.q_cmd[0]), float), (n, 1))
        self.grip_cmd = np.zeros(n)
        self.travel = cfg.jaw_travel_m
        self.clipped = np.zeros(n, dtype=int)
        self.force_open = np.zeros(n, dtype=bool)   # env.force_release(): open until the policy itself commands open
        self.ik_errors: dict[int, str] = {}
        self._path = np.zeros((n, 0, 6))
        self._moving = np.zeros(n, dtype=bool)      # env has a path this env step (False after reset until an action)
        self.frozen = np.zeros(n, dtype=bool)       # env is settling after a per-env reset: hold q_cmd, no IK
        self._k = 0

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        rows = actions.detach().cpu().double().numpy()
        clipped = rows.copy()
        for k, axis in enumerate("xyz"):
            clipped[:, k] = np.clip(rows[:, k], *tcp.WORKSPACE[axis])
        self.clipped += ~np.isclose(clipped[:, :3], rows[:, :3]).all(axis=1)
        self.grip_cmd = np.clip(clipped[:, 6], 0.0, 1.0)
        self.force_open &= self.grip_cmd >= 0.5
        n_ik = max(1, self._env.cfg.decimation // self.cfg.ik_every)
        self._path = np.array([tcp.interpolate(self.tcp_cmd[e], clipped[e, :6], n_ik) for e in range(len(rows))])
        self.tcp_cmd = clipped[:, :6].copy()
        self._moving[:] = ~self.frozen
        self._k = 0
        self._env.begin_action_step()

    def _solve_ik(self) -> None:
        pose_k = min(self._k // self.cfg.ik_every, self._path.shape[1] - 1)
        for e in np.flatnonzero(self._moving):
            try:
                self.q_cmd[e] = tcp.solve(self._path[e, pose_k], self.q_cmd[e])
            except tcp.TcpError as err:
                if len(self.q_cmd) == 1:
                    raise
                self.ik_errors[int(e)] = str(err)
                self._moving[e] = False

    def apply_actions(self) -> None:
        if self._k % self.cfg.ik_every == 0 and self._path.shape[1]:
            self._solve_ik()
        self._k += 1
        tgt = self._asset.data.joint_pos_target.torch.clone()
        tgt[:, self.arm_ids] = torch.tensor(self.q_cmd, dtype=tgt.dtype, device=tgt.device)
        d_np = np.where(self.force_open, 0.0, self.grip_cmd * self.travel)
        d = torch.tensor(d_np, dtype=tgt.dtype, device=tgt.device)
        eff = self._asset.data.joint_effort_target.torch.clone()
        # SMC MHZL2: pneumatic (constant force) with rack-and-pinion synchronised jaws. The squeeze is a constant
        # feed-forward force per jaw; the position drives only centre the pair on its mean closing depth. A stiff
        # drive targeting mean + force / stiffness re-based on every penetrated position and ratcheted the
        # fingers ~1-1.8 mm into the part onto the jaw stop (grasp_diag.py), so the pinch collapsed to 3-10 N.
        q = self._asset.data.joint_pos.torch[:, self.jaw_ids]
        squeeze = torch.tensor((d_np >= 0.5 * self.travel) & (self.cfg.jaw_force_n > 0.0), device=tgt.device)
        depth = torch.where(squeeze, 0.5 * (q[:, 1] - q[:, 0]), d)
        force = torch.where(squeeze, torch.full_like(d, self.cfg.jaw_force_n), torch.zeros_like(d))
        tgt[:, self.jaw_ids[0]], tgt[:, self.jaw_ids[1]] = -depth, depth
        eff[:, self.jaw_ids[0]], eff[:, self.jaw_ids[1]] = -force, force
        self._asset.set_joint_effort_target_index(target=eff, full_data=True)
        self._asset.set_joint_position_target_index(target=tgt, full_data=True)
        self._env.accumulate_contacts()
        if self._k == self._env.cfg.decimation:      # last physics step before this env step renders
            self._env.sync_wrist_camera()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = np.arange(len(self.q_cmd)) if env_ids is None else np.asarray(
            env_ids.cpu() if isinstance(env_ids, torch.Tensor) else env_ids, dtype=int).reshape(-1)
        self._raw[torch.as_tensor(ids, device=self.device)] = 0.0
        q = self._asset.data.joint_pos.torch[torch.as_tensor(ids, device=self.device)].cpu().double().numpy()
        for row, e in zip(q, ids):
            self.q_cmd[e] = row[self.arm_ids]
            self.tcp_cmd[e] = tcp.tcp_pose(self.q_cmd[e])
            self.grip_cmd[e] = float(np.clip(-row[self.jaw_ids[0]] / self.travel, 0.0, 1.0))
            self.ik_errors.pop(int(e), None)
        self._moving[ids] = False
        self.force_open[ids] = False


@configclass
class TcpActionCfg(ActionTermCfg):
    class_type: type = TcpAction
    asset_name: str = "robot"
    home_rad: tuple = MISSING
    jaw_travel_m: float = MISSING
    jaw_force_n: float = 0.0   # > 0: closing squeezes with this constant force per jaw (see apply_actions)
    ik_every: int = 5          # physics steps between IK solves (10 ms at the 2 ms physics step)
