"""act_tcp as an Isaac Lab action term: one action = one CellClient.act_tcp row [x, y, z, w, p, r, gripper]
(UF0 mm / deg, gripper 0 open .. 1 closed), reached over one env step (CellEnvCfg.step_dt, default 0.1 s, the
shortest dt the real cell accepts). Inside the step the TCP moves on a straight line (orientation slerp) from the
previous commanded row, with IK every `ik_every` physics steps and PhysX joint PD in between, like step_tcp.

Rows outside the PLAN 3.1 workspace box are clipped to it and counted in `clipped` (the real cell would refuse them;
scripted demos never send one, Mimic noise can).
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
        if env.num_envs != 1:
            raise ValueError("TcpAction drives a single cell (num_envs = 1)")
        names = list(self._asset.joint_names)
        self.arm_ids = [names.index(f"J{i}") for i in range(1, 7)]
        self.jaw_ids = [names.index("jaw_l"), names.index("jaw_r")]
        self._raw = torch.zeros((1, 7), device=self.device)
        self.q_cmd = np.asarray(cfg.home_rad, float).copy()
        self.tcp_cmd = tcp.tcp_pose(self.q_cmd)
        self.grip_cmd = 0.0
        self.travel = cfg.jaw_travel_m
        self.clipped = 0
        self.force_open = False      # env.force_release(): jaws open until the policy itself commands open
        self._path: list[list[float]] = []
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
        row = actions[0].detach().cpu().double().numpy()
        clipped = row.copy()
        for k, axis in enumerate("xyz"):
            clipped[k] = np.clip(row[k], *tcp.WORKSPACE[axis])
        if not np.allclose(clipped[:3], row[:3]):
            self.clipped += 1
        self.grip_cmd = float(np.clip(clipped[6], 0.0, 1.0))
        if self.grip_cmd < 0.5:
            self.force_open = False
        n_ik = max(1, self._env.cfg.decimation // self.cfg.ik_every)
        self._path = tcp.interpolate(self.tcp_cmd, clipped[:6], n_ik)
        self.tcp_cmd = list(clipped[:6])
        self._k = 0
        self._env.begin_action_step()

    def apply_actions(self) -> None:
        if self._k % self.cfg.ik_every == 0 and self._path:
            pose = self._path[min(self._k // self.cfg.ik_every, len(self._path) - 1)]
            self.q_cmd = tcp.solve(pose, self.q_cmd)
        self._k += 1
        tgt = self._asset.data.joint_pos_target.torch.clone()
        tgt[0, self.arm_ids] = torch.tensor(self.q_cmd, dtype=tgt.dtype, device=tgt.device)
        d = 0.0 if self.force_open else self.grip_cmd * self.travel
        if self.cfg.jaw_lead_m > 0.0 and d >= 0.5 * self.travel:
            # software rack-and-pinion: both jaws target the same closing depth, mean(|q_l|, |q_r|) plus a lead that
            # squeezes with stiffness * lead = the per-finger force; an off-centre jaw pair feels stiffness * offset
            q = self._asset.data.joint_pos.torch[0, self.jaw_ids]
            m = 0.5 * (q[1] - q[0])
            tgt[0, self.jaw_ids[0]] = -torch.clamp(m + self.cfg.jaw_lead_m, max=d)
            tgt[0, self.jaw_ids[1]] = torch.clamp(m + self.cfg.jaw_lead_m, max=d)
        else:
            tgt[0, self.jaw_ids[0]], tgt[0, self.jaw_ids[1]] = -d, d
        self._asset.set_joint_position_target_index(target=tgt, full_data=True)
        self._env.accumulate_contacts()
        if self._k == self._env.cfg.decimation:      # last physics step before this env step renders
            self._env.sync_wrist_camera()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw.zero_()
        q = self._asset.data.joint_pos.torch[0].cpu().double().numpy()
        self.q_cmd = q[self.arm_ids].copy()
        self.tcp_cmd = tcp.tcp_pose(self.q_cmd)
        self.grip_cmd = float(np.clip(-q[self.jaw_ids[0]] / self.travel, 0.0, 1.0))
        self._path = []
        self.force_open = False


@configclass
class TcpActionCfg(ActionTermCfg):
    class_type: type = TcpAction
    asset_name: str = "robot"
    home_rad: tuple = MISSING
    jaw_travel_m: float = MISSING
    jaw_lead_m: float = 0.0    # > 0: synchronised closing (see apply_actions); jaw force / jaw drive stiffness
    ik_every: int = 5          # physics steps between IK solves (10 ms at the 2 ms physics step)
