"""The cell as an Isaac Lab ManagerBasedRLMimicEnv (Isaac Lab 3.0.0-beta2): what the official Isaac Lab Mimic tools
(annotate_demos.py, generate_dataset.py) and the recorder run against. `cell_env.AssemblyCellEnv` is a thin layer on
this same class for agent 3's recorder (reset(layout) / step_tcp).

Frames: every pose Mimic sees (eef, target eef, object frames) is a 4x4 in UF0 metres. Actions are act_tcp rows
[x, y, z, w, p, r, gripper] in UF0 mm / deg, one per env step (step_dt, default 0.1 s). EEF name: "er4ia".

Several cells (cfg.scene.num_envs > 1): every cell has its own episode state (EnvEpisode: task, layout, settle
result); every method that reads or writes one cell takes env_id (default 0), the Mimic hooks take env_ids, and the
observation / termination terms return one row per env. Poses are relative to each env's origin (UF0 of that cell).
A reset with layouts for several envs places them all, then settles them together (settle.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import contact_metrics
import layout as L
import mdp
import numpy as np
import tcp
import torch
import yaml
from extras import CellExtrasMixin, _axis_angle
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLMimicEnv
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass
from scene import FRAMES, JAW_STIFF_X, make_scene_cfg, prim_name
from settle import SettleTracker, settle
from task import SKILL_SIGNALS, TaskState

# reset settling (coordinator spec): at rest for hold_s, give up at cap_s
SETTLE = {"lin_m_s": 0.001, "ang_rad_s": math.radians(0.5), "dforce_n": 0.05, "hold_s": 0.5, "cap_s": 5.0,
          "window_s": 0.1}

EEF = "er4ia"
SKILLS_YAML = L.REPO / "sim" / "assembly_lab" / "mimic" / "skills.yaml"


@dataclass
class EnvEpisode:
    """One cell's episode state."""

    task: TaskState
    layout_override: dict | None = None
    frames_fn: object = None
    placements: dict = field(default_factory=dict)
    settled: bool = False
    settle_s: float = 0.0
    settle_detail: dict = field(default_factory=dict)
    instruction: str = ""


@configclass
class CellMimicEnvCfg(ManagerBasedRLEnvCfg, MimicEnvCfg):
    t12: dict = {}  # noqa: RUF012  configclass deep-copies defaults per instance; skill, pool, contact, sampler, settle_s, home_rad, ... (see make_env_cfg)


class CellMimicEnv(CellExtrasMixin, ManagerBasedRLMimicEnv):
    cfg: CellMimicEnvCfg

    def __init__(self, cfg: CellMimicEnvCfg, render_mode: str | None = None, **kwargs):
        t = cfg.t12
        self.solver = L.LayoutSolver()
        self.pool = list(t["pool"])
        self.names = [prim_name(u) for u in self.pool]
        self.eps = [EnvEpisode(TaskState(t["skill"], self.solver)) for _ in range(cfg.scene.num_envs)]
        self.rng = np.random.default_rng(t.get("seed", 0))
        self._sig_key = None
        self._sig_now: list[dict] = [{} for _ in self.eps]
        self._metrics = None
        self._wrist = None
        self._nominal = None
        self._restoring = False
        self._tracker = None
        self._settled_now: list = []
        super().__init__(cfg, render_mode, **kwargs)
        self._tracker = SettleTracker(self, SETTLE)
        self._metrics = contact_metrics.ContactMetrics(self)
        self._wrist = self._wrist_mount()

    # ------------------------------------------------------------------ env 0 shorthands (one-cell callers)
    task = property(lambda self: self.eps[0].task)
    placements = property(lambda self: self.eps[0].placements)
    settled = property(lambda self: self.eps[0].settled)
    settle_s = property(lambda self: self.eps[0].settle_s)
    settle_detail = property(lambda self: self.eps[0].settle_detail)
    instruction = property(lambda self: self.eps[0].instruction)

    @property
    def layout_override(self) -> dict | None:
        return self.eps[0].layout_override

    @layout_override.setter
    def layout_override(self, lay: dict | None) -> None:
        self.eps[0].layout_override = lay

    @property
    def frames_fn(self):
        return self.eps[0].frames_fn

    @frames_fn.setter
    def frames_fn(self, fn) -> None:
        self.eps[0].frames_fn = fn

    # ------------------------------------------------------------------ poses (UF0 mm 4x4 unless named _m)
    def _top(self) -> float:
        import scene_config as C

        return C.TABLE_TOP_Z

    def _origin_m(self, env_id: int = 0) -> np.ndarray:
        """World position of env `env_id`'s cell origin (m); (0, 0, 0) with one env."""
        return self.scene.env_origins[env_id].detach().cpu().double().numpy()

    def _pose_rows(self, names: list[str], env_id: int = 0) -> np.ndarray:
        """(K, 7) env-local xyz + quat xyzw of the named rigid objects in env `env_id`."""
        rows = torch.stack([self.scene[n].data.root_link_pose_w.torch[env_id] for n in names])
        rows = rows.detach().cpu().double().numpy()
        rows[:, :3] -= self._origin_m(env_id)
        return rows

    def _rows_to_mm(self, rows: np.ndarray) -> list[np.ndarray]:
        out = []
        for r in rows:
            tf = np.eye(4)
            tf[:3, :3], tf[:3, 3] = L.rotation_from_quat_xyzw(r[3:]), r[:3]
            out.append(L.world_m_to_uf0(tf, self._top()))
        return out

    def object_poses_mm(self, env_id: int = 0) -> dict[str, np.ndarray]:
        return dict(zip(self.pool, self._rows_to_mm(self._pose_rows(self.names, env_id))))

    def frames_mm(self, env_id: int = 0) -> dict[str, np.ndarray]:
        return dict(zip(FRAMES, self._rows_to_mm(self._pose_rows(list(FRAMES), env_id))))

    def _to_m(self, tf_mm: np.ndarray) -> torch.Tensor:
        m = np.array(tf_mm, float)
        m[:3, 3] /= 1000.0
        return torch.tensor(m, dtype=torch.float32, device=self.device)

    def _env_list(self, env_ids=None) -> list[int]:
        if env_ids is None:
            return list(range(self.num_envs))
        return [int(e) for e in (env_ids.tolist() if isinstance(env_ids, torch.Tensor) else env_ids)]

    def frame_matrix_m(self, name: str, env_ids=None) -> torch.Tensor:
        """(len(env_ids), 4, 4) frame pose, UF0 m."""
        return torch.stack([self._to_m(self.frames_mm(e)[name]) for e in self._env_list(env_ids)])

    def arm_q(self, env_id: int = 0) -> np.ndarray:
        term = self.action_manager.get_term("tcp")
        return self.scene["robot"].data.joint_pos.torch[env_id].cpu().double().numpy()[term.arm_ids]

    def eef_matrix_m(self, env_ids=None) -> torch.Tensor:
        """(len(env_ids), 4, 4) TCP pose from the simulated joints (FK), UF0 m."""
        out = []
        for e in self._env_list(env_ids):
            x, y, z, w, p, r = tcp.tcp_pose(self.arm_q(e))
            tf = L.translation(x, y, z)
            tf[:3, :3] = L.fanuc_rotation(w, p, r)
            out.append(self._to_m(tf))
        return torch.stack(out)

    def joints_obs(self, env_id: int = 0) -> list[float]:
        term = self.action_manager.get_term("tcp")
        q = self.scene["robot"].data.joint_pos.torch[env_id].cpu().double().numpy()
        grip = float(np.clip(-q[term.jaw_ids[0]] / term.travel, 0.0, 1.0))
        return [*np.rad2deg(q[term.arm_ids]), grip]

    def tcp_obs(self, env_id: int = 0) -> list[float]:
        return tcp.tcp_pose(self.arm_q(env_id))

    # ------------------------------------------------------------------ layout / frames
    def _write_poses(self, names: list[str], tfs_mm: list[np.ndarray], env_id: int = 0) -> None:
        origin = self._origin_m(env_id)
        ids = torch.tensor([env_id], device=self.device)
        for n, tf in zip(names, tfs_mm):
            w = L.uf0_to_world_m(tf, self._top())
            pose = torch.tensor([[*(w[:3, 3] + origin), *L.quat_xyzw(w[:3, :3])]], dtype=torch.float32,
                                device=self.device)
            obj = self.scene[n]
            obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            if n not in FRAMES:                       # frame markers are kinematic
                obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=self.device), env_ids=ids)

    def _place(self, lay: dict, env_id: int) -> None:
        """Every pool object of env `env_id` at its layout pose (parked if not named), arm home, gripper open."""
        floor_z = -(self._top() + L.UF0_ABOVE_BASE_M) * 1000.0
        ep = self.eps[env_id]
        ep.placements = {p.uid: p for p in self.solver.solve(lay, self.pool, floor_z)}
        drop = L.translation(0, 0, self.cfg.t12["drop_mm"])
        self._write_poses(self.names, [drop @ ep.placements[u].tf_uf0 for u in self.pool], env_id)
        self.park_extras(env_id)
        robot = self.scene["robot"]
        ids = torch.tensor([env_id], device=self.device)
        q = robot.data.default_joint_pos.torch[ids].clone()
        robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), env_ids=ids)
        robot.set_joint_position_target_index(target=q, env_ids=ids)
        self.action_manager.get_term("tcp").reset([env_id])
        ep.instruction = lay.get("instruction", "")

    def _settle_envs(self, env_ids: list[int]) -> None:
        for e, (ok, sim_s, detail) in settle(self, env_ids, SETTLE, self.cfg.t12["settle_s"]).items():
            ep = self.eps[e]
            ep.settled, ep.settle_s, ep.settle_detail = ok, sim_s, detail
        self.sync_wrist_camera()

    def apply_layout(self, lay: dict, env_id: int = 0) -> dict[str, np.ndarray]:
        """Place every pool object (parked if not named), arm home, gripper open, settle; returns body poses."""
        self._place(lay, env_id)
        self._settle_envs([env_id])
        return self.object_poses_mm(env_id)

    # ------------------------------------------------------------------ wrist camera
    def _wrist_mount(self):
        """(J6_link body index, 4x4 camera-in-J6 transform, USD camera convention) or None without a wrist camera.

        The RTX render does not move a camera prim parented under an articulation link when physics writes the link
        pose (measured: the wrist sensor stayed at its spawn pose while J6 moved 0.5 m), so the camera is re-posed
        from the simulated J6 pose before every render. The transform is (E, 4, 4): per-env jitter moves each mount."""
        if "wrist" not in self.scene.sensors:
            return None
        from pxr import UsdGeom
        from scene import CAMERA_PATHS, env_path

        prim = self.scene.stage.GetPrimAtPath(env_path(CAMERA_PATHS["wrist"], 0))   # every env's mount is the same
        local = np.array(UsdGeom.Xformable(prim).GetLocalTransformation(), dtype=float).T   # column-vector form
        body = list(self.scene["robot"].body_names).index(prim.GetParent().GetName())
        local_t = torch.tensor(local, dtype=torch.float32, device=self.device)
        return body, local_t.unsqueeze(0).repeat(self.num_envs, 1, 1)

    def sync_wrist_camera(self) -> None:
        if self._wrist is None:
            return
        import isaaclab.utils.math as M

        body, local = self._wrist
        pose = self.scene["robot"].data.body_link_pose_w.torch[:, body]            # (E, 7) pos + quat xyzw
        link = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1)
        link[:, :3, :3] = M.matrix_from_quat(pose[:, 3:7])
        link[:, :3, 3] = pose[:, :3]
        cam = link @ local
        self.scene["wrist"].set_world_poses(cam[:, :3, 3], M.quat_from_matrix(cam[:, :3, :3]), convention="opengl")

    def set_frames(self, frames_mm: dict[str, np.ndarray], env_id: int = 0) -> None:
        missing = [f for f in FRAMES if f not in frames_mm]
        if missing:
            raise ValueError(f"set_frames needs all of {FRAMES}; missing {missing}")
        self._write_poses(list(FRAMES), [np.asarray(frames_mm[f], float) for f in FRAMES], env_id)
        self.scene.write_data_to_sim()
        self.eps[env_id].task.infer(self.object_poses_mm(env_id), {f: np.asarray(frames_mm[f], float) for f in FRAMES})
        self._sig_key = None

    def sample_episode(self, env_ids=None) -> None:
        """Reset event: each env's layout (+ frames callback) when the thin layer gave one, else the configured
        sampler. Envs with a layout are placed first and settled together."""
        if self._restoring:
            return
        ids = self._env_list(env_ids)
        given = [e for e in ids if self.eps[e].layout_override is not None]
        for e in given:
            self._place(self.eps[e].layout_override, e)
        if given:
            self._settle_envs(given)
        for e in ids:
            ep = self.eps[e]
            if e in given:
                poses = self.object_poses_mm(e)
                frames = ep.frames_fn(poses) if ep.frames_fn else _placeholder_frames(poses, self.pool)
                ep.layout_override, ep.frames_fn = None, None
            else:
                frames = _load_sampler(self.cfg.t12.get("sampler"))(_CellView(self, e), self.rng)
            self.set_frames(frames, e)
        self.apply_jitter(ids)
        if self._metrics is not None:
            self._metrics.reset_episode(ids)

    def reachable(self, tf_mm: np.ndarray, env_id: int = 0) -> bool:
        try:
            tcp.solve(L.pose6_from_tf(tf_mm), self.arm_q(env_id))
            return True
        except tcp.TcpError:
            return False

    @property
    def skill(self) -> str:
        return self.task.skill

    # ------------------------------------------------------------------ contacts / signals
    def begin_action_step(self) -> None:
        if self._metrics is not None:
            self._metrics.begin_step()

    def accumulate_contacts(self) -> None:
        if self._metrics is not None:
            self._metrics.accumulate()
        if self._tracker is not None and self._tracker.active:
            self._settled_now += self._tracker.tick()

    # ------------------------------------------------------------------ per-env reset (others keep stepping)
    def begin_reset(self, env_id: int, lay: dict, frames_fn=None) -> None:
        """Place env `env_id`'s layout now; it holds home and settles during the next env steps (settle.SettleTracker)
        while the other envs keep running. finish_resets() completes it once settled."""
        self._place(lay, env_id)
        self.eps[env_id].frames_fn = frames_fn
        self.action_manager.get_term("tcp").frozen[env_id] = True
        self._tracker.start(env_id, self.cfg.t12["settle_s"])

    @property
    def settling(self) -> set[int]:
        return set(self._tracker.active) if self._tracker is not None else set()

    def finish_resets(self) -> list[int]:
        """Complete every env that settled (or hit the cap) during the last env step: frames, task, jitter, fresh
        metrics and episode counter, and a fresh recorder episode whose initial state is this settled scene."""
        done, self._settled_now = self._settled_now, []
        ids = []
        for e, ok, sim_s, detail in done:
            ep = self.eps[e]
            ep.settled, ep.settle_s, ep.settle_detail = ok, sim_s, detail
            poses = self.object_poses_mm(e)
            frames = ep.frames_fn(poses) if ep.frames_fn else _placeholder_frames(poses, self.pool)
            ep.frames_fn = None
            self.set_frames(frames, e)
            self.action_manager.get_term("tcp").frozen[e] = False
            ids.append(e)
        if ids:
            self.apply_jitter(ids)
            self._metrics.reset_episode(ids)
            self.episode_length_buf[ids] = 0
            self._sig_key = None
            if self.recorder_manager is not None:
                self.recorder_manager.reset(env_ids=ids)          # drop the settle steps
                self.recorder_manager.record_post_reset(ids)      # initial_state = the settled scene
        return ids

    def step_peak_vector(self) -> torch.Tensor:
        if self._metrics is None:      # the observation manager probes term shapes before __init__ finishes
            return torch.zeros((self.num_envs, 4), device=self.device)
        return self._metrics.peak_vector()

    def _signals(self, env_id: int = 0) -> dict[str, bool]:
        """Instantaneous signals of env `env_id`; all envs are evaluated together once per env step (and after a
        reset / set_frames), which also advances every env's latches."""
        key = int(self.common_step_counter), tuple(self.episode_length_buf.tolist())
        if key != self._sig_key and self._metrics is not None:
            grip = self.action_manager.get_term("tcp").grip_cmd
            for e, ep in enumerate(self.eps):
                if ep.task.target is None:
                    continue
                m = self._metrics.current(e)
                t = self.names[self.pool.index(ep.task.target)]
                self._sig_now[e] = ep.task.signals(self.object_poses_mm(e), self.frames_mm(e), m["jaw_n"][t],
                                                   m["touch_n"][t], float(grip[e]))
            self._sig_key = key
        return self._sig_now[env_id]

    def latched_signal(self, name: str, env_id: int = 0) -> bool:
        self._signals(env_id)
        return self.eps[env_id].task.latched.get(name, False)

    def episode_success(self, env_id: int = 0) -> bool:
        task = self.eps[env_id].task
        if task.target is None:
            return False
        now = self._signals(env_id)
        t = self.names[self.pool.index(task.target)]
        return task.success(now, self._metrics.current(env_id)["touch_n"][t])

    # ------------------------------------------------------------------ Isaac Lab Mimic API
    def get_robot_eef_pose(self, eef_name: str, env_ids=None) -> torch.Tensor:
        return self.eef_matrix_m(env_ids)

    def target_eef_pose_to_action(self, target_eef_pose_dict: dict, gripper_action_dict: dict,
                                  action_noise_dict: dict | None = None, env_id: int = 0) -> torch.Tensor:
        pose = target_eef_pose_dict[EEF].detach().cpu().double().numpy().reshape(4, 4).copy()
        noise = float((action_noise_dict or {}).get(EEF, 0.0) or 0.0)
        if noise > 0:
            pose[:3, 3] += np.random.normal(0.0, noise, 3)
            axis = np.random.normal(size=3)
            pose[:3, :3] = _axis_angle(axis / np.linalg.norm(axis), np.random.normal(0.0, noise * 1000.0 *
                                       math.radians(self.cfg.t12["rot_deg_per_mm"]))) @ pose[:3, :3]
        pose[:3, 3] *= 1000.0
        grip = float(gripper_action_dict[EEF].reshape(-1)[0])
        return torch.tensor([[*L.pose6_from_tf(pose), grip]], dtype=torch.float32, device=self.device)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        out = []
        for row in action.detach().cpu().double().numpy().reshape(-1, 7):
            tf = L.translation(*row[:3])
            tf[:3, :3] = L.fanuc_rotation(*row[3:6])
            out.append(self._to_m(tf))
        return {EEF: torch.stack(out)}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {EEF: actions[..., -1:]}

    def get_object_poses(self, env_ids=None) -> dict[str, torch.Tensor]:
        ids = self._env_list(env_ids)
        per_env = [{**self.frames_mm(e), **dict(zip(self.names, self.object_poses_mm(e).values()))} for e in ids]
        return {k: torch.stack([self._to_m(p[k]) for p in per_env]) for k in per_env[0]}

    def get_subtask_term_signals(self, env_ids=None) -> dict[str, torch.Tensor]:
        ids = self._env_list(env_ids)
        for e in ids:
            self._signals(e)
        return {s: torch.tensor([float(self.eps[e].task.latched.get(s, False)) for e in ids], device=self.device)
                for s in SKILL_SIGNALS[self.task.skill]}

    def reset_to(self, state, env_ids, seed=None, is_relative=False):
        self._restoring = True          # the reset event must not sample a layout that the state overwrites
        try:
            out = super().reset_to(state, env_ids, seed=seed, is_relative=is_relative)
        finally:
            self._restoring = False
        ids = self._env_list(env_ids)
        self.action_manager.get_term("tcp").reset(ids)
        self.sync_wrist_camera()
        self.apply_jitter(ids)
        for e in ids:
            self.eps[e].task.infer(self.object_poses_mm(e), self.frames_mm(e))
        self._sig_key = None
        if self._metrics is not None:
            self._metrics.reset_episode(ids)
        return out


class _CellView:
    """What an episode sampler sees of one cell (sampler.py signature: api.apply_layout / pool / solver / skill /
    reachable), bound to env `env_id`."""

    def __init__(self, env: CellMimicEnv, env_id: int):
        self.env, self.env_id = env, env_id
        self.pool, self.solver, self.skill = env.pool, env.solver, env.skill

    def apply_layout(self, lay: dict) -> dict[str, np.ndarray]:
        return self.env.apply_layout(lay, self.env_id)

    def reachable(self, tf_mm: np.ndarray) -> bool:
        return self.env.reachable(tf_mm, self.env_id)


def _placeholder_frames(poses: dict, pool: list[str]) -> dict[str, np.ndarray]:
    """No plan given: park all three frames on the first movable object (signals stay false)."""
    first = next((u for u in pool if u.startswith("M") and poses[u][2, 3] > -400), next(iter(poses)))
    return {f: poses[first].copy() for f in FRAMES}


def _load_sampler(spec: str | None):
    if not spec:
        import sampler

        return sampler.sample
    mod, fn = spec.split(":")
    import importlib

    return getattr(importlib.import_module(mod), fn)


def subtask_configs(skill: str) -> list[SubTaskConfig]:
    """agent 4's mimic/skills.yaml -> Isaac Lab Mimic SubTaskConfig list (noise mm -> m)."""
    spec = yaml.safe_load(SKILLS_YAML.read_text())
    noise = spec["noise_profiles"]
    sel = spec.get("selection", {})
    out = []
    for st in spec["skills"][skill]["subtasks"]:
        out.append(SubTaskConfig(
            object_ref=st["object_ref"], subtask_term_signal=st["term_signal"],
            subtask_term_offset_range=tuple(st["term_offset"]),
            selection_strategy=sel.get("strategy", "nearest_neighbor_object"),
            selection_strategy_kwargs={"nn_k": sel.get("nn_k", 3)},
            action_noise=float(noise[st["noise"]]["pos_mm"]) / 1000.0,
            num_interpolation_steps=int(st["interp_steps"]), num_fixed_steps=0,
            apply_noise_during_interpolation=False, description=st["name"]))
    return out


def make_env_cfg(skill: str, pool: list[str], contact: dict, *, cameras=("table", "side", "wrist"),
                 image_size=(640, 360), depth=(), step_dt: float = 0.1, episode_s: float = 60.0,
                 jaw_force_n: float, home_rad: np.ndarray, settle_s: float = 0.3, drop_mm: float = 0.2,
                 sampler: str | None = None, device: str = "cuda:0", seed: int = 0,
                 jitter: dict | None = None, extras: bool = False, num_envs: int = 1) -> CellMimicEnvCfg:
    import actions
    import isaaclab.sim as sim_utils
    import scene_config as C
    from isaaclab.managers import ObservationGroupCfg  # noqa: F401  (configclass import side effects)
    from isaaclab_physx.physics import PhysxCfg

    cfg = CellMimicEnvCfg()
    dt = float(contact["physics_dt"])
    cfg.decimation = max(1, round(step_dt / dt))
    cfg.episode_length_s = episode_s
    cfg.num_rerenders_on_reset = 1       # the reset observation shows the new layout, not the previous frame
    cfg.sim = sim_utils.SimulationCfg(
        dt=dt, device=device, render_interval=cfg.decimation,
        physics=PhysxCfg(solver_type=1, bounce_threshold_velocity=contact["bounce_threshold_velocity"],
                         friction_offset_threshold=contact["friction_offset_threshold"],
                         friction_correlation_distance=contact["friction_correlation_distance"]))
    cfg.scene = make_scene_cfg(pool, contact, jaw_force_n, home_rad, tuple(cameras), tuple(image_size), tuple(depth),
                               extras=extras, num_envs=num_envs)

    @configclass
    class ActionsCfg:
        tcp = actions.TcpActionCfg(home_rad=tuple(home_rad), jaw_travel_m=C.JAW_OPEN - C.JAW_CLOSED,
                                   jaw_lead_m=jaw_force_n / (JAW_STIFF_X * C.JAW_DRIVE_STIFF))

    @configclass
    class ObservationsCfg:
        policy = mdp.policy_group(tuple(cameras))
        subtask_terms = mdp.subtask_group(skill)

    cfg.actions, cfg.observations = ActionsCfg(), ObservationsCfg()
    cfg.events, cfg.terminations, cfg.rewards = mdp.EventsCfg(), mdp.TerminationsCfg(), mdp.RewardsCfg()
    cfg.commands = None
    spec = yaml.safe_load(SKILLS_YAML.read_text())
    cfg.subtask_configs = {EEF: subtask_configs(skill)}
    cfg.datagen_config.name = f"t12_{skill}"
    cfg.datagen_config.generation_select_src_per_subtask = True
    cfg.datagen_config.generation_interpolate_from_last_target_pose = True
    cfg.datagen_config.generation_keep_failed = False
    cfg.datagen_config.seed = seed
    cfg.t12 = {"skill": skill, "pool": list(pool), "settle_s": settle_s, "drop_mm": drop_mm, "sampler": sampler,
               "seed": seed, "rot_deg_per_mm": float(spec.get("rot_deg_per_mm", 1.0)), "jitter": dict(jitter or {})}
    return cfg
