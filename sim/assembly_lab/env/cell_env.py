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

Several cells in one process (CellEnvCfg(num_envs=N)); the full contract is BATCH_API.md:

    obs = env.reset_many([layout_0, ..., layout_N-1], [frames_fn_0, ...])   # all cells at once, settled together
    env.begin_reset({k: (layout, frames_fn)})          # per-env reset: env k settles while the others keep stepping
    out = env.step_tcp_many({0: (rows, dt), 3: (rows, dt)})                # {env: obs} for the envs asked
    ready = env.ready_resets()                         # {k: reset obs} for envs whose reset finished
    key = env.end_episode(k, ok, attrs)                # recorder: mark + export env k's episode now -> "demo_<n>"

step_tcp_many plays every listed env's rows at once; an env that runs out of rows (or is not listed) holds its last
TCP command until the longest list is done. Hold steps are recorded into a live episode, so end_episode an env as
soon as its episode is over; whatever it records after that is dropped at its next reset. An IK failure in one env
does not stop the others: that env holds and its obs carries "ik_error" (with one env, TcpError is raised as before).
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
    num_envs: int = 1                         # cells stepped in lockstep in this process (reset_many / step_tcp_many)


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
                               extras=self.cfg.disturbances, num_envs=self.cfg.num_envs)
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
        self._ended: set[int] = set()     # envs whose recorder episode end_episode exported (later steps are idle)
        self._ready: dict[int, dict] = {}  # per-env resets that finished: {env: reset obs}
        if self.cfg.disturbances:     # only then do the hooks exist (eval/run_eval.py checks hasattr)
            for name in ("inject_foreign", "block_holes", "force_release", "inject_intruder", "clear_intruder"):
                setattr(self, name, getattr(self.env, name))

    @property
    def placements(self) -> dict:
        return self.env.placements

    # ------------------------------------------------------------------ API
    def reset(self, layout: dict, frames_fn=None) -> dict:
        if self.cfg.num_envs != 1:
            raise ValueError(f"reset() drives one cell; this env has {self.cfg.num_envs}: use reset_many")
        return self.reset_many([layout], [frames_fn])[0]

    def reset_many(self, layouts: list[dict], frames_fns: list | None = None) -> list[dict]:
        """Reset every cell together: env k gets layouts[k] (+ frames_fns[k]); all settle at once."""
        if len(layouts) != self.cfg.num_envs:
            raise ValueError(f"reset_many needs one layout per env ({self.cfg.num_envs}), got {len(layouts)}")
        fns = frames_fns or [None] * len(layouts)
        for ep, lay, fn in zip(self.env.eps, layouts, fns):
            ep.layout_override, ep.frames_fn = lay, fn
        rm = self.env.recorder_manager
        if self._ended and rm is not None:
            rm.reset(env_ids=sorted(self._ended))      # drop the idle steps recorded after end_episode
        self._ended.clear()
        self._obs, _ = self.env.reset()
        self.env._metrics.reset_chunk()
        return [self._observe(e) for e in range(self.cfg.num_envs)]

    def begin_reset(self, resets: dict) -> None:
        """{env: (layout, frames_fn)}: place those cells now; each settles during the following step_tcp_many calls
        (holding home) while the other cells keep running, then shows up in ready_resets(). Its recorder episode
        starts at the settled scene (the settle steps are not recorded)."""
        if self._obs is None:
            raise RuntimeError("begin_reset needs one reset_many() first: the observation and recorder buffers only "
                               "exist after Isaac Lab's env.reset()")
        busy = sorted(set(resets) & self.env.settling)
        if busy:
            raise ValueError(f"envs {busy} are already resetting")
        for k, (lay, fn) in resets.items():
            self.env.begin_reset(int(k), lay, fn)
            self._ready.pop(int(k), None)
            self._ended.discard(int(k))

    def ready_resets(self) -> dict[int, dict]:
        """{env: reset obs} of every per-env reset that finished since the last call (obs["sim"]["settled"] False if
        the settle cap was hit, as with reset())."""
        out, self._ready = self._ready, {}
        return out

    def reset_envs(self, resets: dict) -> dict[int, dict]:
        """begin_reset + hold every cell until those envs are ready. Only for when no other cell is mid-episode
        (the hold steps would be recorded into its demo); otherwise drive the resets with step_tcp_many."""
        self.begin_reset(resets)
        want, got = {int(k) for k in resets}, {}
        while not want <= set(got):
            self.step_tcp_many({})
            got.update(self.ready_resets())
        self._ready.update({k: v for k, v in got.items() if k not in want})
        return {k: got[k] for k in want}

    def end_episode(self, env_id: int, ok: bool, attrs: dict | None = None) -> str | None:
        """Mark env_id's recorder episode succeeded / failed, export it now and write `attrs` on its demo group.
        Returns the demo key written ("demo_<n>"), or None (no recorder, empty episode, or not exported by the
        recorder's export mode)."""
        import torch

        self._ended.add(int(env_id))
        rm = self.env.recorder_manager
        if rm is None:
            return None
        handler = rm._dataset_file_handler        # Isaac Lab has no public "last demo written"; the handler counts
        before = handler.demo_count
        ids = torch.tensor([env_id], device=self.env.device)
        rm.set_success_to_episodes(ids, torch.tensor([[bool(ok)]], dtype=torch.bool, device=self.env.device))
        rm.export_episodes(ids)
        if handler.demo_count == before:
            return None
        key = f"demo_{before}"
        for name, value in (attrs or {}).items():
            handler._hdf5_data_group[key].attrs[name] = value
        handler.flush()
        return key

    def rotate(self, next_path: str) -> str | None:
        """Close the recorder file and record on into next_path; returns the closed path. Safe while episodes are
        in progress: they live in memory until end_episode exports them (into whichever file is open then)."""
        rm = self.env.recorder_manager
        if rm is None:
            return None
        closed = rm._dataset_file_handler._hdf5_file_stream.filename
        rm._dataset_file_handler.close()
        rm._dataset_file_handler = rm.cfg.dataset_file_handler_class_type()
        rm._dataset_file_handler.create(str(next_path), env_name=getattr(self.env.cfg, "env_name", None))
        return closed

    def set_frames(self, frames_mm: dict) -> None:
        """Set / move the Mimic frames after reset (prefer reset(frames_fn=...): the recorder's initial state is
        written at reset, so frames set later are not in it)."""
        self.env.set_frames(frames_mm)

    def step_tcp(self, rows, dt: float) -> dict:
        """Play act_tcp rows like the real cell: each row is an absolute TCP target reached over dt seconds."""
        if self.cfg.num_envs != 1:
            raise ValueError(f"step_tcp drives one cell; this env has {self.cfg.num_envs}: use step_tcp_many")
        return self.step_tcp_many({0: (rows, dt)})[0]

    def _env_steps(self, env_id: int, rows, dt: float) -> list[list[float]]:
        """The per-env-step actions of one act_tcp chunk: round(dt / step_dt) straight-line sub-targets per row, each
        starting where the previous sub-target ended as TcpAction.tcp_cmd will hold it (the float32 action)."""
        rows = np.atleast_2d(np.asarray(rows, dtype=float))
        if not 1 <= len(rows) <= 100 or not 0.1 - 1e-9 <= dt <= 1.0 + 1e-9:
            raise tcp.TcpError(f"chunk of {len(rows)} rows at dt={dt}: CellClient allows 1-100 rows, dt 0.1-1.0 s")
        rows = np.array([tcp.check_row(r) for r in rows])
        n = max(1, round(dt / self.cfg.step_dt))
        prev, out = self.term.tcp_cmd[env_id], []
        for row in rows:
            for pose in tcp.interpolate(prev, row, n):
                out.append([*pose, row[6]])
                prev = np.asarray(pose, dtype=np.float32).astype(float)
        return out

    def step_tcp_many(self, requests: dict) -> dict[int, dict]:
        """{env_id: (rows, dt)} -> {env_id: obs}: every listed env plays its chunk; the others hold."""
        import torch

        busy = sorted({int(e) for e in requests} & self.env.settling)
        if busy:
            raise ValueError(f"envs {busy} are resetting: wait for ready_resets()")
        plans = {int(e): self._env_steps(int(e), rows, dt) for e, (rows, dt) in requests.items()}
        ids = sorted(plans)
        if ids:
            self.env._metrics.reset_chunk(ids)
        term = self.term
        for k in range(max([len(p) for p in plans.values()], default=1)):
            act = np.concatenate([term.tcp_cmd, term.grip_cmd[:, None]], axis=1)
            for e in ids:
                if k < len(plans[e]) and e not in term.ik_errors:
                    act[e] = plans[e][k]
            self._obs, *_ = self.env.step(torch.tensor(act, dtype=torch.float32, device=self.env.device))
        # resets complete only between calls, so a reset env records nothing until its lane sends its first chunk
        for e in self.env.finish_resets():
            self.env._metrics.reset_chunk([e])
            self._ready[e] = self._observe(e)
        return {e: self._observe(e) for e in ids}

    def close(self) -> None:
        if self.cfg.record_to and self.env.recorder_manager is not None:
            self.env.recorder_manager.export_episodes()
        self.env.close()
        if AssemblyCellEnv.app is not None:
            AssemblyCellEnv.app.close()
            AssemblyCellEnv.app = None

    # ------------------------------------------------------------------ obs
    def _images(self, env_id: int = 0) -> tuple[dict, dict]:
        images, depth = {}, {}
        for cam in self.cfg.cameras:
            out = self.env.scene[cam].data.output
            images[cam] = out["rgb"].torch[env_id, ..., :3].cpu().numpy().astype(np.uint8)
            if "distance_to_image_plane" in out:
                d = np.nan_to_num(out["distance_to_image_plane"].torch[env_id, ..., 0].cpu().numpy(), posinf=0.0)
                depth[cam] = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
        return images, depth

    def _observe(self, env_id: int = 0) -> dict:
        e, k = self.env, env_id
        ep = e.eps[k]
        images, depth = self._images(k)
        by_name = dict(zip(e.names, e.pool))
        now, chunk = e._metrics.current(k), e._metrics.chunk(k)
        signals = e._signals(k) if ep.task.target is not None else {}
        obs = {"t": time.time(), "images": images, "joints": np.array(e.joints_obs(k), dtype=np.float32),
               "tcp_pose": [round(v, 3) for v in e.tcp_obs(k)], "instruction": ep.instruction,
               "sim": {"object_poses": {u: [round(v, 3) for v in L.pose6_from_tf(tf)]
                                        for u, tf in e.object_poses_mm(k).items()},
                       "contact": {by_name[n]: v for n, v in now["contact"].items()},
                       "contact_peak": {by_name[n]: v for n, v in chunk["contact_peak"].items()},
                       "robot_touch": {by_name[n]: v for n, v in chunk["robot_touch"].items()},
                       "jaw_n": {by_name[n]: v for n, v in now["jaw_n"].items()},
                       "signals": signals, "signals_latched": dict(ep.task.latched),
                       "target": ep.task.target, "base": ep.task.base,
                       "frames": {f: L.pose6_from_tf(tf) for f, tf in e.frames_mm(k).items()},
                       "tcp_cmd": [float(v) for v in self.term.tcp_cmd[k]], "clipped_rows": int(self.term.clipped[k]),
                       "on_table": {u: p.role != "parked" for u, p in ep.placements.items()},
                       "settled": bool(ep.settled), "settle_s": round(float(ep.settle_s), 3),
                       "settle_detail": ep.settle_detail}}
        if k in self.term.ik_errors:
            obs["sim"]["ik_error"] = self.term.ik_errors[k]
        sub = e._metrics.substep_log(k)
        if sub is not None:
            obs["sim"]["substep_contact"] = {by_name[n]: v for n, v in sub.items()}
        if depth:
            obs["depth"] = depth
        return obs
