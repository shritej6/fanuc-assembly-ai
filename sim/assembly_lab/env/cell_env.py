"""AssemblyCellEnv: the thin CellClient-style layer over the Isaac Lab Mimic env (mimic_env.CellMimicEnv).

    env = AssemblyCellEnv(CellEnvCfg(pool=[...], cameras=(...)))   # starts Isaac Sim via Isaac Lab's AppLauncher
    obs = env.reset(layout, frames_fn=None)      # layout: INTERFACES.md; frames_fn(poses) -> Mimic frames (optional)
    obs = env.step_tcp(rows, dt)                 # rows (N, 7) [x, y, z, w, p, r, gripper], exactly CellClient.act_tcp
    env.close()

Every row is played as round(dt / step_dt) env steps (straight-line sub-targets, step_dt = 0.1 s), so the Isaac Lab
recorder and Mimic see one act_tcp action per 0.1 s whatever dt the skill used.

frames_fn(poses) gets {uid: 4x4 UF0 mm} after the layout settled and returns {"grasp_frame", "seat_frame",
"spot_frame": 4x4 UF0 mm TCP poses}; they become kinematic markers in the scene (so the initial state the recorder
writes contains them) and drive the subtask signals. Without it the signals stay false.

Observation = CellClient.observe keys (images, optional depth, joints (7), tcp_pose, instruction) plus `sim`:
object_poses {uid: [x y z w p r]} (body = STL frame, UF0 mm / deg); contact {uid: {normal_N, lateral_N, torque_Nm
(about the object's insert axis, body +Z), torque3d_Nm (full magnitude, logging)}} now; contact_peak (same keys, max
over every physics step of this step_tcp call); robot_touch {uid: peak N between any robot link and that object during
this call}; signals (instantaneous) and signals_latched; frames; tcp_cmd; settled / settle_s / settle_detail (reset
steps physics until every object rests: < 1 mm/s, < 0.5 deg/s, contact forces changing < 0.05 N, for 0.5 s; cap 5 s,
settled False if the cap is hit). Layout x, y place each object's CAD local origin (objects/objects.yaml).
CellEnvCfg(disturbances=True) adds the eval hooks inject_foreign(x, y, yaw), block_holes(part, base),
force_release(), inject_intruder(x, y) / clear_intruder() (mimic_env.py).
Policies never read `sim` (INTERFACES.md).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(HERE), str(REPO / "sim" / "starter"), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import layout as L
import tcp
from physics_cfg import CONTACT_YAML, load_contact

JAW_FORCE_N = 34.0      # SMC MHZL2-16 external grip force per finger at 0.5 MPa (Team V measured_cell_geometry.json)
CAMERAS = ("table", "side", "wrist")


@dataclass
class CellEnvCfg:
    device: str = "cuda:0"
    headless: bool = True
    pool: list[str] | None = None          # object instances to spawn; None = all 30 + one spare copy of each part
    spare_copies: int = 1
    cameras: tuple[str, ...] = CAMERAS
    image_size: tuple[int, int] = (640, 360)   # (W, H), CellClient.observe default
    depth: tuple[str, ...] = ()               # e.g. ("table", "wrist"): uint16 mm at image_size
    skill: str = "pick_insert"                # which subtask signals / success term the env computes
    step_dt: float = 0.1                      # one act_tcp action per env step
    settle_s: float = 0.3
    jaw_force_n: float = JAW_FORCE_N
    home_deg: dict | None = None              # default tcp.OPERATOR_HOME_DEG (robot/cell/mock_cell.py HOME)
    drop_mm: float = 0.2                      # spawn this far above the rest pose
    sampler: str | None = None                # "module:function" for env.reset() without a layout (Mimic generation)
    record_to: str | None = None              # Isaac Lab recorder HDF5 path (ActionStateRecorderManagerCfg) or None
    contact_yaml: Path = CONTACT_YAML
    contact_overrides: dict = field(default_factory=dict)
    disturbances: bool = False                # eval props + hooks: inject_foreign, block_holes, force_release, inject_intruder
    jitter: dict = field(default_factory=dict)  # per-episode {cam_pos_mm, cam_rot_deg, light_frac}; {} = nominal views


class AssemblyCellEnv:
    app = None

    def __init__(self, cfg: CellEnvCfg | None = None, app=None):
        self.cfg = cfg or CellEnvCfg()
        if app is not None:
            AssemblyCellEnv.app = app
        if AssemblyCellEnv.app is None:
            from isaaclab.app import AppLauncher
            AssemblyCellEnv.app = AppLauncher(headless=self.cfg.headless, enable_cameras=bool(self.cfg.cameras),
                                              device=self.cfg.device).app
        from mimic_env import CellMimicEnv, make_env_cfg

        self.contact, self.contact_source = load_contact(Path(self.cfg.contact_yaml), self.cfg.contact_overrides)
        self.solver = L.LayoutSolver()
        self.pool = self.cfg.pool or L.default_pool(self.solver.geometry, self.cfg.spare_copies)
        home = self.cfg.home_deg or tcp.OPERATOR_HOME_DEG
        env_cfg = make_env_cfg(self.cfg.skill, self.pool, self.contact, cameras=self.cfg.cameras,
                               image_size=self.cfg.image_size, depth=self.cfg.depth, step_dt=self.cfg.step_dt,
                               jaw_force_n=self.cfg.jaw_force_n, settle_s=self.cfg.settle_s, drop_mm=self.cfg.drop_mm,
                               home_rad=np.deg2rad([home[f"J{i}"] for i in range(1, 7)]), sampler=self.cfg.sampler,
                               device=self.cfg.device, jitter=self.cfg.jitter,
                               extras=self.cfg.disturbances)
        env_cfg.terminations.time_out = None          # the recorder layer decides when an episode ends
        env_cfg.terminations.success = None
        if self.cfg.record_to:
            from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

            rec = ActionStateRecorderManagerCfg()
            rec.dataset_export_dir_path = str(Path(self.cfg.record_to).parent)
            rec.dataset_filename = Path(self.cfg.record_to).stem
            env_cfg.recorders = rec
        self.env = CellMimicEnv(env_cfg)
        self.term = self.env.action_manager.get_term("tcp")
        self.dt = self.env.physics_dt
        self.masses = {u: L.pla_mass_kg(REPO / self.solver.geometry[L.type_id(u)]["stl"])[0] for u in self.pool}
        self._obs = None
        if self.cfg.disturbances:     # only then do the hooks exist (eval/run_eval.py checks hasattr)
            for name in ("inject_foreign", "block_holes", "force_release", "inject_intruder", "clear_intruder"):
                setattr(self, name, getattr(self.env, name))

    @property
    def placements(self) -> dict:
        return self.env.placements

    # ------------------------------------------------------------------ API
    def reset(self, layout: dict, frames_fn=None) -> dict:
        self.env.layout_override, self.env.frames_fn = layout, frames_fn
        self._obs, _ = self.env.reset()
        self.env._metrics.reset_chunk()
        return self._observe()

    def set_frames(self, frames_mm: dict) -> None:
        """Set / move the Mimic frames after reset (prefer reset(frames_fn=...): the recorder's initial state is
        written at reset, so frames set later are not in it)."""
        self.env.set_frames(frames_mm)

    def step_tcp(self, rows, dt: float) -> dict:
        """Play act_tcp rows like the real cell: each row is an absolute TCP target reached over dt seconds."""
        import torch

        rows = np.atleast_2d(np.asarray(rows, dtype=float))
        if not 1 <= len(rows) <= 100 or not 0.1 - 1e-9 <= dt <= 1.0 + 1e-9:
            raise tcp.TcpError(f"chunk of {len(rows)} rows at dt={dt}: CellClient allows 1-100 rows, dt 0.1-1.0 s")
        rows = np.array([tcp.check_row(r) for r in rows])
        n = max(1, round(dt / self.cfg.step_dt))
        self.env._metrics.reset_chunk()
        for row in rows:
            for pose in tcp.interpolate(self.term.tcp_cmd, row, n):
                action = torch.tensor([[*pose, row[6]]], dtype=torch.float32, device=self.env.device)
                self._obs, *_ = self.env.step(action)
        return self._observe()

    def close(self) -> None:
        if self.cfg.record_to and self.env.recorder_manager is not None:
            self.env.recorder_manager.export_episodes()
        self.env.close()
        if AssemblyCellEnv.app is not None:
            AssemblyCellEnv.app.close()
            AssemblyCellEnv.app = None

    # ------------------------------------------------------------------ obs
    def _images(self) -> tuple[dict, dict]:
        images, depth = {}, {}
        for cam in self.cfg.cameras:
            out = self.env.scene[cam].data.output
            images[cam] = out["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8)
            if "distance_to_image_plane" in out:
                d = np.nan_to_num(out["distance_to_image_plane"].torch[0, ..., 0].cpu().numpy(), posinf=0.0)
                depth[cam] = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
        return images, depth

    def _observe(self) -> dict:
        e = self.env
        images, depth = self._images()
        by_name = dict(zip(e.names, e.pool))
        now, chunk = e._metrics.current(), e._metrics.chunk()
        signals = e._signals() if e.task.target is not None else {}
        obs = {"t": time.time(), "images": images, "joints": np.array(e.joints_obs(), dtype=np.float32),
               "tcp_pose": [round(v, 3) for v in e.tcp_obs()], "instruction": e.instruction,
               "sim": {"object_poses": {u: [round(v, 3) for v in L.pose6_from_tf(tf)]
                                        for u, tf in e.object_poses_mm().items()},
                       "contact": {by_name[n]: v for n, v in now["contact"].items()},
                       "contact_peak": {by_name[n]: v for n, v in chunk["contact_peak"].items()},
                       "robot_touch": {by_name[n]: v for n, v in chunk["robot_touch"].items()},
                       "jaw_n": {by_name[n]: v for n, v in now["jaw_n"].items()},
                       "signals": signals, "signals_latched": dict(e.task.latched),
                       "target": e.task.target, "base": e.task.base,
                       "frames": {f: L.pose6_from_tf(tf) for f, tf in e.frames_mm().items()},
                       "tcp_cmd": [float(v) for v in self.term.tcp_cmd], "clipped_rows": self.term.clipped,
                       "on_table": {u: p.role != "parked" for u, p in e.placements.items()},
                       "settled": bool(e.settled), "settle_s": round(float(e.settle_s), 3),
                       "settle_detail": e.settle_detail}}
        sub = e._metrics.substep_log()
        if sub is not None:
            obs["sim"]["substep_contact"] = {by_name[n]: v for n, v in sub.items()}
        if depth:
            obs["depth"] = depth
        return obs
