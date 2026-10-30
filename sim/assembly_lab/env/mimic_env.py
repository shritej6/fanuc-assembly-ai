"""The cell as an Isaac Lab ManagerBasedRLMimicEnv (Isaac Lab 3.0.0-beta2): what the official Isaac Lab Mimic tools
(annotate_demos.py, generate_dataset.py) and the recorder run against. `cell_env.AssemblyCellEnv` is a thin layer on
this same class for agent 3's recorder (reset(layout) / step_tcp).

Frames: every pose Mimic sees (eef, target eef, object frames) is a 4x4 in UF0 metres. Actions are act_tcp rows
[x, y, z, w, p, r, gripper] in UF0 mm / deg, one per env step (step_dt, default 0.1 s). EEF name: "er4ia".
"""

from __future__ import annotations

import math

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
from task import SKILL_SIGNALS, TaskState

# reset settling (coordinator spec): at rest for hold_s, give up at cap_s
SETTLE = {"lin_m_s": 0.001, "ang_rad_s": math.radians(0.5), "dforce_n": 0.05, "hold_s": 0.5, "cap_s": 5.0,
          "window_s": 0.1}

EEF = "er4ia"
SKILLS_YAML = L.REPO / "sim" / "assembly_lab" / "mimic" / "skills.yaml"


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
        self.task = TaskState(t["skill"], self.solver)
        self.rng = np.random.default_rng(t.get("seed", 0))
        self.layout_override: dict | None = None
        self.frames_fn = None
        self.placements: dict = {}
        self.settled, self.settle_s, self.settle_detail = False, 0.0, {}
        self.instruction = ""
        self._sig_key = None
        self._sig_now: dict = {}
        self._metrics = None
        self._wrist = None
        self._nominal = None
        self._restoring = False
        super().__init__(cfg, render_mode, **kwargs)
        self._metrics = contact_metrics.ContactMetrics(self)
        self._wrist = self._wrist_mount()

    # ------------------------------------------------------------------ poses (UF0 mm 4x4 unless named _m)
    def _top(self) -> float:
        import scene_config as C

        return C.TABLE_TOP_Z

    def _pose_rows(self, names: list[str]) -> np.ndarray:
        rows = torch.stack([self.scene[n].data.root_link_pose_w.torch[0] for n in names])     # (K, 7) xyz + quat xyzw
        return rows.detach().cpu().double().numpy()

    def _rows_to_mm(self, rows: np.ndarray) -> list[np.ndarray]:
        out = []
        for r in rows:
            tf = np.eye(4)
            tf[:3, :3], tf[:3, 3] = L.rotation_from_quat_xyzw(r[3:]), r[:3]
            out.append(L.world_m_to_uf0(tf, self._top()))
        return out

    def object_poses_mm(self) -> dict[str, np.ndarray]:
        return dict(zip(self.pool, self._rows_to_mm(self._pose_rows(self.names))))

    def frames_mm(self) -> dict[str, np.ndarray]:
        return dict(zip(FRAMES, self._rows_to_mm(self._pose_rows(list(FRAMES)))))

    def _to_m(self, tf_mm: np.ndarray) -> torch.Tensor:
        m = np.array(tf_mm, float)
        m[:3, 3] /= 1000.0
        return torch.tensor(m, dtype=torch.float32, device=self.device)

    def frame_matrix_m(self, name: str) -> torch.Tensor:
        return self._to_m(self.frames_mm()[name]).unsqueeze(0)

    def arm_q(self) -> np.ndarray:
        term = self.action_manager.get_term("tcp")
        return self.scene["robot"].data.joint_pos.torch[0].cpu().double().numpy()[term.arm_ids]

    def eef_matrix_m(self) -> torch.Tensor:
        x, y, z, w, p, r = tcp.tcp_pose(self.arm_q())
        tf = L.translation(x, y, z)
        tf[:3, :3] = L.fanuc_rotation(w, p, r)
        return self._to_m(tf).unsqueeze(0)

    def joints_obs(self) -> list[float]:
        term = self.action_manager.get_term("tcp")
        q = self.scene["robot"].data.joint_pos.torch[0].cpu().double().numpy()
        grip = float(np.clip(-q[term.jaw_ids[0]] / term.travel, 0.0, 1.0))
        return [*np.rad2deg(q[term.arm_ids]), grip]

    def tcp_obs(self) -> list[float]:
        return tcp.tcp_pose(self.arm_q())

    # ------------------------------------------------------------------ layout / frames
    def _write_poses(self, names: list[str], tfs_mm: list[np.ndarray]) -> None:
        for n, tf in zip(names, tfs_mm):
            w = L.uf0_to_world_m(tf, self._top())
            pose = torch.tensor([[*w[:3, 3], *L.quat_xyzw(w[:3, :3])]], dtype=torch.float32, device=self.device)
            obj = self.scene[n]
            obj.write_root_pose_to_sim_index(root_pose=pose)
            if n not in FRAMES:                       # frame markers are kinematic
                obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=self.device))

    def apply_layout(self, lay: dict) -> dict[str, np.ndarray]:
        """Place every pool object (parked if not named), arm home, gripper open, settle; returns body poses."""
        floor_z = -(self._top() + L.UF0_ABOVE_BASE_M) * 1000.0
        self.placements = {p.uid: p for p in self.solver.solve(lay, self.pool, floor_z)}
        drop = L.translation(0, 0, self.cfg.t12["drop_mm"])
        self._write_poses(self.names, [drop @ self.placements[u].tf_uf0 for u in self.pool])
        self.park_extras()
        robot = self.scene["robot"]
        q = robot.data.default_joint_pos.torch.clone()
        robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), full_data=True)
        robot.set_joint_position_target_index(target=q, full_data=True)
        self.action_manager.get_term("tcp").reset()
        self.settled, self.settle_s = self._settle()
        self.instruction = lay.get("instruction", "")
        self.sync_wrist_camera()
        return self.object_poses_mm()

    # ------------------------------------------------------------------ wrist camera
    def _wrist_mount(self):
        """(J6_link body index, 4x4 camera-in-J6 transform, USD camera convention) or None without a wrist camera.

        The RTX render does not move a camera prim parented under an articulation link when physics writes the link
        pose (measured: the wrist sensor stayed at its spawn pose while J6 moved 0.5 m), so the camera is re-posed
        from the simulated J6 pose before every render."""
        if "wrist" not in self.scene.sensors:
            return None
        from pxr import UsdGeom
        from scene import CAMERA_PATHS

        prim = self.scene.stage.GetPrimAtPath(CAMERA_PATHS["wrist"])
        local = np.array(UsdGeom.Xformable(prim).GetLocalTransformation(), dtype=float).T   # column-vector form
        body = list(self.scene["robot"].body_names).index(prim.GetParent().GetName())
        return body, torch.tensor(local, dtype=torch.float32, device=self.device)

    def sync_wrist_camera(self) -> None:
        if self._wrist is None:
            return
        import isaaclab.utils.math as M

        body, local = self._wrist
        pose = self.scene["robot"].data.body_link_pose_w.torch[:, body]            # (1, 7) pos + quat xyzw
        link = torch.eye(4, device=self.device).unsqueeze(0)
        link[:, :3, :3] = M.matrix_from_quat(pose[:, 3:7])
        link[:, :3, 3] = pose[:, :3]
        cam = link @ local
        self.scene["wrist"].set_world_poses(cam[:, :3, 3], M.quat_from_matrix(cam[:, :3, :3]), convention="opengl")

    def _settle(self) -> tuple[bool, float]:
        """Step physics until every object is at rest for SETTLE["hold_s"]; give up at SETTLE["cap_s"] and say so.

        Motion is the displacement over a sliding SETTLE["window_s"] window (position change / window < 1 mm/s,
        rotation change / window < 0.5 deg/s), so a one-step solver velocity spike is not motion but sustained
        motion still fails; every object's net contact force must change < 0.05 N between physics steps. On failure
        settle_detail names the worst object and value for each quantity. Returns (settled, simulated seconds)."""
        import isaaclab.utils.math as M

        objs = [self.scene[n] for n in self.names]
        net = self.scene["object_contact"]
        dt = self.physics_dt
        need, cap = max(1, round(SETTLE["hold_s"] / dt)), round(SETTLE["cap_s"] / dt)
        w = max(1, round(SETTLE["window_s"] / dt))
        hist_p, hist_q = [], []
        prev_f, calm, best = None, 0, 0
        fails = {"lin": 0, "ang": 0, "dforce": 0}
        self.settle_detail = {}
        lin = ang = dfo = None
        for k in range(1, cap + 1):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=dt)
            hist_p.append(torch.stack([o.data.root_link_pos_w.torch[0] for o in objs]).clone())
            hist_q.append(torch.stack([o.data.root_link_quat_w.torch[0] for o in objs]).clone())
            f = net.data.net_forces_w.torch[0].clone()
            dfo = None if prev_f is None else (f - prev_f).norm(dim=-1)
            prev_f = f
            if len(hist_p) <= w or dfo is None:
                continue
            p0, q0 = hist_p.pop(0), hist_q.pop(0)
            lin = (hist_p[-1] - p0).norm(dim=-1) / (w * dt)
            ang = M.quat_error_magnitude(hist_q[-1], q0) / (w * dt)
            ok = {"lin": bool(lin.max() < SETTLE["lin_m_s"]), "ang": bool(ang.max() < SETTLE["ang_rad_s"]),
                  "dforce": bool(dfo.max() < SETTLE["dforce_n"])}
            for key, good in ok.items():
                fails[key] += 0 if good else 1
            calm = calm + 1 if all(ok.values()) else 0
            best = max(best, calm)
            if calm >= need and k * dt >= self.cfg.t12["settle_s"]:
                return True, k * dt
        if lin is None:                                          # cap shorter than one window: nothing measured
            self.settle_detail = {"longest_calm_s": 0.0, "failed_steps": fails}
            return False, cap * dt
        f_order = list(net.body_names)
        worst = lambda v: (self.pool[int(v.argmax())], float(v.max()))
        self.settle_detail = {
            "max_lin_mm_s": [worst(lin)[0], round(worst(lin)[1] * 1000, 3)],
            "max_ang_deg_s": [worst(ang)[0], round(math.degrees(worst(ang)[1]), 3)],
            "max_dforce_N": [self.pool[self.names.index(f_order[int(dfo.argmax())])], round(float(dfo.max()), 4)],
            "longest_calm_s": round(best * dt, 3), "failed_steps": fails}
        return False, cap * dt

    def set_frames(self, frames_mm: dict[str, np.ndarray]) -> None:
        missing = [f for f in FRAMES if f not in frames_mm]
        if missing:
            raise ValueError(f"set_frames needs all of {FRAMES}; missing {missing}")
        self._write_poses(list(FRAMES), [np.asarray(frames_mm[f], float) for f in FRAMES])
        self.scene.write_data_to_sim()
        self.task.infer(self.object_poses_mm(), {f: np.asarray(frames_mm[f], float) for f in FRAMES})
        self._sig_key = None

    def sample_episode(self) -> None:
        """Reset event: the thin layer's layout (+ frames callback) when given, else the configured sampler."""
        if self._restoring:
            return
        if self.layout_override is not None:
            poses = self.apply_layout(self.layout_override)
            frames = self.frames_fn(poses) if self.frames_fn else _placeholder_frames(poses, self.pool)
            self.layout_override, self.frames_fn = None, None
        else:
            frames = _load_sampler(self.cfg.t12.get("sampler"))(self, self.rng)
        self.set_frames(frames)
        self.apply_jitter()
        if self._metrics is not None:
            self._metrics.reset_episode()

    def reachable(self, tf_mm: np.ndarray) -> bool:
        try:
            tcp.solve(L.pose6_from_tf(tf_mm), self.arm_q())
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

    def step_peak_vector(self) -> torch.Tensor:
        if self._metrics is None:      # the observation manager probes term shapes before __init__ finishes
            return torch.zeros((1, 4), device=self.device)
        return self._metrics.peak_vector()

    def _signals(self) -> dict[str, bool]:
        key = int(self.common_step_counter), int(self.episode_length_buf[0])
        if key != self._sig_key and self.task.target is not None and self._metrics is not None:
            m = self._metrics.current()
            t = self.names[self.pool.index(self.task.target)]
            self._sig_now = self.task.signals(self.object_poses_mm(), self.frames_mm(), m["jaw_n"][t],
                                              m["touch_n"][t], self.action_manager.get_term("tcp").grip_cmd)
            self._sig_key = key
        return self._sig_now

    def latched_signal(self, name: str) -> bool:
        self._signals()
        return self.task.latched.get(name, False)

    def episode_success(self) -> bool:
        if self.task.target is None:
            return False
        now = self._signals()
        t = self.names[self.pool.index(self.task.target)]
        return self.task.success(now, self._metrics.current()["touch_n"][t])

    # ------------------------------------------------------------------ Isaac Lab Mimic API
    def get_robot_eef_pose(self, eef_name: str, env_ids=None) -> torch.Tensor:
        return self.eef_matrix_m()

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
        poses = {f: self._to_m(tf).unsqueeze(0) for f, tf in self.frames_mm().items()}
        poses.update({n: self._to_m(tf).unsqueeze(0) for n, tf in zip(self.names, self.object_poses_mm().values())})
        return poses

    def get_subtask_term_signals(self, env_ids=None) -> dict[str, torch.Tensor]:
        self._signals()
        return {s: torch.tensor([float(self.task.latched.get(s, False))], device=self.device)
                for s in SKILL_SIGNALS[self.task.skill]}

    def reset_to(self, state, env_ids, seed=None, is_relative=False):
        self._restoring = True          # the reset event must not sample a layout that the state overwrites
        try:
            out = super().reset_to(state, env_ids, seed=seed, is_relative=is_relative)
        finally:
            self._restoring = False
        self.action_manager.get_term("tcp").reset()
        self.sync_wrist_camera()
        self.apply_jitter()
        self.task.infer(self.object_poses_mm(), self.frames_mm())
        self._sig_key = None
        if self._metrics is not None:
            self._metrics.reset_episode()
        return out


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
                 jitter: dict | None = None, extras: bool = False) -> CellMimicEnvCfg:
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
                               extras=extras)

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
