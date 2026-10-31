"""GR00T N1.7 as the official `policy.py` interface (ACTION = "tcp"), with every chunk passed through the real-arm guard.

    from policy_learn.deploy.groot_policy import GrootPolicy
    policy = GrootPolicy("groot://127.0.0.1:5555")   # or "stub" for a plumbing test; None reads $GROOT_POLICY
    # each planner (T09) step: the instruction is built by mimic/instructions.instruction, the function the demos
    # were labelled with, e.g. "pick M4_02 at (300, -50, 37) gripping X at roll 127, insert on F04 at (400, 50, 90)"
    policy.set_plan("pick_insert", "M4_02", "F04", grip=0, case="nominal",
                    target=[300, -50, 37, 127, 400, 50, 90],   # part x, y, yaw, roll r, base x, y, yaw
                    contact=ContactTarget(400.0, 50.0, -301.2, label="M4_02 on F04"))
    rows = policy.act(obs)                         # (N, 7) [x, y, z, w, p, r, gripper], already guarded

The model sees cameras + joints + TCP + the instruction, never obs["sim"]. The wire format is agent 5's
`policy_learn/groot/groot_client.py` (Isaac-GR00T PolicyServer, eef_9d + gripper, dt 0.2 s).
"""
from __future__ import annotations

import logging
import os
from typing import ClassVar, Protocol

import numpy as np

from policy_learn.deploy.guards import ChunkGuard, ContactTarget, GuardLimits, GuardResult, matrix_to_wpr

log = logging.getLogger(__name__)
DT_S = 0.2  # GR00T training control step (policy_learn/groot/groot_client.py DT_S, 5 Hz)
EXEC_HORIZON = 16  # rows of each model chunk sent before GR00T is asked again (the full trained chunk)

# FK fallback for a missing obs["tcp_pose"] (the mock reports none before the first tcp command). The chain is the
# official twin's, copied from robot/cell/calibration/geometry.py (read-only there); UTOOL1 is 293 mm along the
# flange x axis (robot/cell/calibration/FRAMES.md). UF0 z = base z - 330 mm.
_CHAIN = [((0.0, 0.0, 0.33), "z", +1), ((0.0, 0.0, 0.00), "y", +1), ((0.0, 0.0, 0.26), "y", -1),
          ((0.0, 0.0, 0.02), "x", -1), ((0.29, 0.0, 0.0), "y", -1), ((0.0, 0.0, 0.00), "x", -1)]
_TOOL_M, _UF0_Z_MM = 0.293, -330.0
# Flange axes -> FANUC tool axes: tool z = flange x (approach), tool x = flange z, tool y = -flange y. At the cell's
# HOME joints this gives fingers down (w, p, r = -180, 0, 0), matching the PLAN's "fingers down" convention.
_FLANGE_TO_TOOL = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def tcp_from_joints(joints) -> np.ndarray:
    """Estimated [x, y, z mm, w, p, r deg] in UF0 from J1..J6 (an estimate: used only to HOLD, never to move)."""
    rot, pos = np.eye(3), np.zeros(3)
    for (off, axis, sign), q in zip(_CHAIN, np.radians(np.asarray(joints, float)[:6])):
        pos = pos + rot @ np.array(off)
        rot = rot @ _rot(axis, sign * q)
    tip = pos + rot[:, 0] * _TOOL_M
    w, p, r = matrix_to_wpr(rot @ _FLANGE_TO_TOOL)
    return np.array([tip[0] * 1000.0, tip[1] * 1000.0, tip[2] * 1000.0 + _UF0_Z_MM, w, p, r])


class ChunkModel(Protocol):
    def predict(self, obs: dict) -> np.ndarray:
        """obs: images, joints, tcp_pose, instruction (never obs["sim"]). Return (N, 7) absolute act_tcp rows."""


class StubModel:
    """Plumbing test only: glides toward a hover pose at 40 mm/s, gripper unchanged. Never for the real cell."""

    HOVER = np.array([450.0, 0.0, -200.0, -180.0, 0.0, 0.0])

    def __init__(self, horizon: int = 16, dt: float = DT_S, speed_mm_s: float = 40.0):
        self.horizon, self.step = horizon, speed_mm_s * dt

    def predict(self, obs: dict) -> np.ndarray:
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
    """Agent 5's `GrootTcpClient` (policy_learn/groot/groot_client.py): ZMQ to a running GR00T policy server."""

    def __init__(self, host: str, port: int, client=None):
        if client is None:  # imported here: it needs pyzmq + msgpack, which the stub path does not
            from policy_learn.groot.groot_client import GrootTcpClient

            client = GrootTcpClient(host=host, port=port)
            if not client.ping():
                raise ConnectionError(f"GR00T policy server at {host}:{port} does not answer ping")
        self.client = client

    def reset(self) -> None:
        self.client.reset()

    def predict(self, obs: dict) -> np.ndarray:
        return np.asarray(self.client.act(obs), dtype=float)


def load_model(spec: str | None) -> ChunkModel:
    """spec: "stub", "groot://host:port", or None to read $GROOT_POLICY. Anything else is an error, not a fallback."""
    spec = spec or os.environ.get("GROOT_POLICY")
    if spec == "stub":
        log.warning("GrootPolicy is using the STUB model: plumbing test only, not for the real cell")
        return StubModel()
    if spec and spec.startswith("groot://"):
        host, _, port = spec[len("groot://"):].rpartition(":")
        return GrootServerModel(host or "127.0.0.1", int(port))
    raise ValueError(f"no GR00T policy given: pass 'groot://host:port' or 'stub' (or set GROOT_POLICY), got {spec!r}")


def nearest_turns(chunk, tcp_pose) -> np.ndarray:
    """Write each row's w and r as the equivalent angle numerically nearest the row before, starting at the reported
    pose: a model's 6D-rotation output decoded near r = +-180 flips sign between rows (179.8 vs -179.9) although the
    fingers barely turn. Same orientation either way, so the controller and the guard both see the real, small turn;
    the guard still refuses a genuine large one. Malformed chunks pass through untouched for the guard to refuse."""
    arr = np.array(chunk, dtype=float, copy=True)
    if arr.ndim != 2 or arr.shape[1] != 7 or not np.isfinite(arr).all():
        return arr
    prev = np.asarray(tcp_pose, float)[[3, 5]]
    for row in arr:
        row[[3, 5]] = prev = prev + (row[[3, 5]] - prev + 180.0) % 360.0 - 180.0
    return arr


class GrootPolicy:
    """The official policy interface. Class attributes are read by robot/cell/run.py and cell_client.run_trial."""

    CAMERAS: ClassVar[list[str]] = ["table", "side", "wrist"]
    IMAGE_SIZE = (640, 360)
    IMAGE_FORMAT = "png"
    ACTION = "tcp"
    PREFETCH = False   # every chunk is guarded against the pose it starts from; prefetch would start from a stale pose
    DEPTH: ClassVar[list[str]] = []
    DT = DT_S

    def __init__(self, checkpoint: str | None = None, model: ChunkModel | None = None,
                 limits: GuardLimits | None = None, exec_horizon: int | None = None):
        """exec_horizon: send only the first N rows of each chunk, then ask GR00T again from the new pose (None reads
        $GROOT_EXEC_HORIZON, default 16 = the whole chunk). The guard checks just those N rows."""
        self.exec_horizon = int(exec_horizon or os.environ.get("GROOT_EXEC_HORIZON") or EXEC_HORIZON)
        if not 1 <= self.exec_horizon <= 100:
            raise ValueError(f"exec_horizon must be 1..100 rows, got {self.exec_horizon}")
        self.model = model or load_model(checkpoint)
        self.guard = ChunkGuard(limits)
        self.instruction: str | None = None
        self.guard_log: list[dict] = []  # kept across rollouts, for the run's report
        self.steps, self.rollout = 0, -1

    def reset(self) -> None:
        self.instruction, self.steps, self.rollout = None, 0, self.rollout + 1
        self.guard.set_target(None)
        if hasattr(self.model, "reset"):
            self.model.reset()

    def set_task(self, instruction: str | None, target: ContactTarget | None = None) -> None:
        """The planner's (T09) current step: the instruction GR00T gets and the contact target the guard enforces."""
        self.instruction = instruction
        self.guard.set_target(target)

    def set_plan(self, skill: str, object_id: str, base_id: str | None, grip: int, case: str, target,
                 contact: ContactTarget | None = None) -> str:
        """set_task with the training instruction format (sim/assembly_lab/mimic/instructions.py, the same function
        agent 3's demos use), so the words GR00T sees on the arm match the words it was trained on."""
        from sim.assembly_lab.mimic.catalog import encode_plan_ids
        from sim.assembly_lab.mimic.instructions import instruction

        text = instruction(encode_plan_ids(skill, object_id, base_id, grip, case), target)
        self.set_task(text, contact)
        return text

    def _instruction(self, obs: dict) -> str:
        trial = obs.get("trial") or {}
        return self.instruction or obs.get("instruction") or trial.get("instruction") or ""

    def act(self, obs: dict) -> np.ndarray:
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
