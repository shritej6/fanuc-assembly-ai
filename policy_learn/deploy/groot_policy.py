"""GR00T N1.7 behind the official policy interface, with every chunk passed through the arm guard.
Authored by Shritej Gunda."""
from __future__ import annotations

import logging
import os
from typing import ClassVar, Protocol

import numpy as np

from policy_learn.deploy.guards import ChunkGuard, ContactTarget, GuardLimits, GuardResult, matrix_to_wpr

log = logging.getLogger(__name__)
DT_S = 0.2  # GR00T training control step (policy_learn/groot/groot_client.py DT_S, 5 Hz)
EXEC_HORIZON = 16  # rows of each model chunk sent before GR00T is asked again (the full trained chunk)

# FK fallback when obs has no tcp_pose; chain from robot/cell/calibration/geometry.py, UTOOL1 293 mm along flange x.
_CHAIN = [((0.0, 0.0, 0.33), "z", +1), ((0.0, 0.0, 0.00), "y", +1), ((0.0, 0.0, 0.26), "y", -1),
          ((0.0, 0.0, 0.02), "x", -1), ((0.29, 0.0, 0.0), "y", -1), ((0.0, 0.0, 0.00), "x", -1)]
_TOOL_M, _UF0_Z_MM = 0.293, -330.0
# Flange to FANUC tool axes: tool z = flange x, tool x = flange z, tool y = -flange y (fingers down at HOME).
_FLANGE_TO_TOOL = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def tcp_from_joints(joints) -> np.ndarray:
    """Return the estimated TCP pose from joint angles, used only to hold."""
    rot, pos = np.eye(3), np.zeros(3)
    for (off, axis, sign), q in zip(_CHAIN, np.radians(np.asarray(joints, float)[:6])):
        pos = pos + rot @ np.array(off)
        rot = rot @ _rot(axis, sign * q)
    tip = pos + rot[:, 0] * _TOOL_M
    w, p, r = matrix_to_wpr(rot @ _FLANGE_TO_TOOL)
    return np.array([tip[0] * 1000.0, tip[1] * 1000.0, tip[2] * 1000.0 + _UF0_Z_MM, w, p, r])


class ChunkModel(Protocol):
    """Interface of a model that predicts one action chunk."""
    def predict(self, obs: dict) -> np.ndarray:
        """Return (N, 7) absolute act_tcp rows for one observation."""


class StubModel:
    """Plumbing test model that glides toward a hover pose; never for the real cell."""

    HOVER = np.array([450.0, 0.0, -200.0, -180.0, 0.0, 0.0])

    def __init__(self, horizon: int = 16, dt: float = DT_S, speed_mm_s: float = 40.0):
        self.horizon, self.step = horizon, speed_mm_s * dt

    def predict(self, obs: dict) -> np.ndarray:
        """Return a stub chunk toward the hover pose."""
        pose, rows = np.asarray(obs["tcp_pose"], float)[:6].copy(), []
        grip = 1.0 if float(obs["joints"][6]) >= 0.5 else 0.0
        for _ in range(self.horizon):
            d = self.HOVER - pose
            d[3:6] = np.clip(d[3:6], -1.0, 1.0)
            n = np.linalg.norm(d[:3])
            if n > self.step:
                d[:3] *= self.step / n
            pose = pose + d
            rows.append(np.concatenate([pose, [grip]]))
        return np.array(rows)


class GrootServerModel:
    """Model backed by a running GR00T policy server over ZMQ."""

    def __init__(self, host: str, port: int, client=None):
        if client is None:  # imported here: it needs pyzmq + msgpack, which the stub path does not
            from policy_learn.groot.groot_client import GrootTcpClient

            client = GrootTcpClient(host=host, port=port)
            if not client.ping():
                raise ConnectionError(f"GR00T policy server at {host}:{port} does not answer ping")
        self.client = client

    def reset(self) -> None:
        """Reset the server session."""
        self.client.reset()

    def predict(self, obs: dict) -> np.ndarray:
        """Return the server's chunk for one observation."""
        return np.asarray(self.client.act(obs), dtype=float)


def load_model(spec: str | None) -> ChunkModel:
    """Return the model for a spec: stub, groot://host:port, or None to read $GROOT_POLICY."""
    spec = spec or os.environ.get("GROOT_POLICY")
    if spec == "stub":
        log.warning("GrootPolicy is using the STUB model: plumbing test only, not for the real cell")
        return StubModel()
    if spec and spec.startswith("groot://"):
        host, _, port = spec[len("groot://"):].rpartition(":")
        return GrootServerModel(host or "127.0.0.1", int(port))
    raise ValueError(f"no GR00T policy given: pass 'groot://host:port' or 'stub' (or set GROOT_POLICY), got {spec!r}")


def nearest_turns(chunk, tcp_pose) -> np.ndarray:
    """Return the rows with w and r unwrapped to the angle nearest the previous row."""
    arr = np.array(chunk, dtype=float, copy=True)
    if arr.ndim != 2 or arr.shape[1] != 7 or not np.isfinite(arr).all():
        return arr
    prev = np.asarray(tcp_pose, float)[[3, 5]]
    for row in arr:
        row[[3, 5]] = prev = prev + (row[[3, 5]] - prev + 180.0) % 360.0 - 180.0
    return arr


class GrootPolicy:
    """The official policy interface used by the cell runner."""

    CAMERAS: ClassVar[list[str]] = ["table", "side", "wrist"]
    IMAGE_SIZE = (640, 360)
    IMAGE_FORMAT = "png"
    ACTION = "tcp"
    PREFETCH = False   # every chunk is guarded against the pose it starts from; prefetch would start from a stale pose
    DEPTH: ClassVar[list[str]] = []
    DT = DT_S

    def __init__(self, checkpoint: str | None = None, model: ChunkModel | None = None,
                 limits: GuardLimits | None = None, exec_horizon: int | None = None):
        """Create the policy; exec_horizon is how many rows of each chunk to send."""
        self.exec_horizon = int(exec_horizon or os.environ.get("GROOT_EXEC_HORIZON") or EXEC_HORIZON)
        if not 1 <= self.exec_horizon <= 100:
            raise ValueError(f"exec_horizon must be 1..100 rows, got {self.exec_horizon}")
        self.model = model or load_model(checkpoint)
        self.guard = ChunkGuard(limits)
        self.instruction: str | None = None
        self.guard_log: list[dict] = []  # kept across rollouts, for the run's report
        self.steps, self.rollout = 0, -1

    def reset(self) -> None:
        """Start a new rollout."""
        self.instruction, self.steps, self.rollout = None, 0, self.rollout + 1
        self.guard.set_target(None)
        if hasattr(self.model, "reset"):
            self.model.reset()

    def set_task(self, instruction: str | None, target: ContactTarget | None = None) -> None:
        """Set the planner step: the instruction for GR00T and the guard contact target."""
        self.instruction = instruction
        self.guard.set_target(target)

    def set_plan(self, skill: str, object_id: str, base_id: str | None, grip: int, case: str, target,
                 contact: ContactTarget | None = None) -> str:
        """Set the task using the same instruction format the demos were labelled with."""
        from sim.assembly_lab.mimic.catalog import encode_plan_ids
        from sim.assembly_lab.mimic.instructions import instruction

        text = instruction(encode_plan_ids(skill, object_id, base_id, grip, case), target)
        self.set_task(text, contact)
        return text

    def _instruction(self, obs: dict) -> str:
        trial = obs.get("trial") or {}
        return self.instruction or obs.get("instruction") or trial.get("instruction") or ""

    def act(self, obs: dict) -> np.ndarray:
        """Return guarded act_tcp rows for one observation."""
        self.steps += 1
        joints = np.asarray(obs["joints"], float)
        if obs.get("tcp_pose") is None:
            pose = tcp_from_joints(joints)
            return self._record(GuardResult(np.concatenate([pose, [1.0 if joints[6] >= 0.5 else 0.0]])[None], False,
                                            True, ("tcp_pose missing: holding at the FK estimate",)))
        tcp = np.asarray(obs["tcp_pose"], float)[:6]
        model_obs = {"images": obs.get("images", {}), "joints": joints, "tcp_pose": tcp,
                     "instruction": self._instruction(obs)}  # never obs["sim"]
        chunk = nearest_turns(self.model.predict(model_obs), tcp)
        if chunk.ndim == 2:
            chunk = chunk[:self.exec_horizon]  # rows past N are never sent, so they are not guarded either
        return self._record(self.guard.guard(chunk, tcp, float(joints[6]), self.DT))

    def _record(self, res: GuardResult) -> np.ndarray:
        if res.reasons:
            self.guard_log.append({"rollout": self.rollout, "step": self.steps, "held": res.held,
                                   "truncated_at": res.truncated_at, "reasons": list(res.reasons)})
            kind = "HOLD" if res.held else "truncate" if res.truncated_at is not None else "clamp"
            (log.info if kind == "clamp" else log.warning)("guard %s at act %d: %s", kind, self.steps,
                                                           "; ".join(res.reasons))
        return res.rows
