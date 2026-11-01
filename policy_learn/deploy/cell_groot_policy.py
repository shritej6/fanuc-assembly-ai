"""GR00T N1.7 as the official `policy.py` interface (ACTION = "tcp"), with every chunk passed through the real-arm guard.

    from policy_learn.deploy.cell_groot_policy import GrootPolicy
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

from assembly.controller_ports import ExecutionTolerances
from assembly.fixed_inspection import FixedInspectionMove, next_inspection_leg
from policy_learn.deploy.attestation import (
    AttestationError,
    DeploymentAttestation,
    ReviewedDeployment,
    attestation_from_mapping,
    validate_attestation,
)
from policy_learn.deploy.cell_guards import (
    ChunkGuard,
    ContactTarget,
    GuardLimits,
    GuardResult,
    UnsafeHoldError,
    hold_row,
    matrix_to_wpr,
    observed_gripper_state,
)

log = logging.getLogger(__name__)
DT_S = 0.2  # GR00T training control step (policy_learn/groot/groot_client.py DT_S, 5 Hz)


class CellPolicySafetyError(RuntimeError):
    """A production-only invariant is missing, so no robot action may be produced."""


# FK fallback for a missing obs["tcp_pose"] (the mock reports none before the first tcp command). The chain is the
# official twin's, copied from robot/cell/calibration/geometry.py (read-only there); UTOOL1 is 293 mm along the
# flange x axis (robot/cell/calibration/FRAMES.md). UF0 z = base z - 330 mm.
_CHAIN = [
    ((0.0, 0.0, 0.33), "z", +1),
    ((0.0, 0.0, 0.00), "y", +1),
    ((0.0, 0.0, 0.26), "y", -1),
    ((0.0, 0.0, 0.02), "x", -1),
    ((0.29, 0.0, 0.0), "y", -1),
    ((0.0, 0.0, 0.00), "x", -1),
]
_TOOL_M, _UF0_Z_MM = 0.293, -330.0
# Flange axes -> FANUC tool axes: tool z = flange x (approach), tool x = flange z, tool y = -flange y. At the cell's
# HOME joints this gives fingers down (w, p, r = -180, 0, 0), matching the PLAN's "fingers down" convention.
_FLANGE_TO_TOOL = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


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
        self.injected_client = client is not None
        if client is None:  # imported here: it needs pyzmq + msgpack, which the stub path does not
            from policy_learn.groot.groot_client import GrootTcpClient

            client = GrootTcpClient(host=host, port=port, execution_horizon=1)
            if not client.ping():
                raise ConnectionError(f"GR00T policy server at {host}:{port} does not answer ping")
        self.client = client

    def reset(self) -> None:
        self.client.reset()

    def describe(self) -> dict:
        """Fetch provenance from the serving process; no client-side identity is synthesized."""
        return self.client.describe()

    def predict(self, obs: dict) -> np.ndarray:
        return np.asarray(self.client.act(obs), dtype=float)

    def close(self) -> None:
        """Release the owned GR00T transport."""
        self.client.close()


def load_model(spec: str | None) -> ChunkModel:
    """spec: "stub", "groot://host:port", or None to read $GROOT_POLICY. Anything else is an error, not a fallback."""
    spec = spec or os.environ.get("GROOT_POLICY")
    if spec == "stub":
        log.warning("GrootPolicy is using the STUB model: plumbing test only, not for the real cell")
        return StubModel()
    if spec and spec.startswith("groot://"):
        host, _, port = spec[len("groot://") :].rpartition(":")
        return GrootServerModel(host or "127.0.0.1", int(port))
    raise ValueError(f"no GR00T policy given: pass 'groot://host:port' or 'stub' (or set GROOT_POLICY), got {spec!r}")


def _attest_production_model(
    model: ChunkModel,
    limits: GuardLimits,
    approved: ReviewedDeployment | None,
) -> DeploymentAttestation:
    if isinstance(model, StubModel):
        raise CellPolicySafetyError("the GR00T stub is forbidden in CELL production")
    if not isinstance(model, GrootServerModel):
        raise CellPolicySafetyError("CELL production requires an attested GrootServerModel")
    if model.injected_client:
        raise CellPolicySafetyError("CELL production forbids injected GR00T clients")
    if not limits.production_ready:
        raise CellPolicySafetyError("CELL production requires MEASURED guard limits with a recorded source")
    if not isinstance(approved, ReviewedDeployment):
        raise CellPolicySafetyError("CELL production requires a registry-pinned deployment record")
    if approved.attestation.guard_calibration_sha256 != limits.calibration_sha256:
        raise CellPolicySafetyError("CELL guard calibration differs from the reviewed deployment")
    try:
        remote = attestation_from_mapping(model.describe())
        return validate_attestation(remote, expected=approved.attestation)
    except (AttestationError, ConnectionError, OSError, RuntimeError, TimeoutError) as error:
        raise CellPolicySafetyError(f"GR00T deployment attestation failed: {error}") from error


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


def _incremental_fixed_pose(
    current: np.ndarray,
    target: tuple[float, ...],
    limits: GuardLimits,
) -> np.ndarray:
    target_array = nearest_turns(np.asarray([(*target, 0.0)]), current)[0, :6]
    delta = target_array - current
    floor_zone = limits.z_mm[0] + limits.slow_zone_mm
    slow = current[2] <= floor_zone or target_array[2] <= floor_zone
    linear_step = (limits.slow_speed_mm_s if slow else limits.max_speed_mm_s) * DT_S
    angular_step = (limits.slow_rot_deg_s if slow else limits.max_rot_deg_s) * DT_S
    linear_ratio = float(np.linalg.norm(delta[:3])) / linear_step
    angular_ratio = float(np.abs(delta[3:6]).max()) / angular_step
    ratio = max(linear_ratio, angular_ratio, 1.0)
    return current + delta / ratio


class GrootPolicy:
    """The official policy interface. Class attributes are read by robot/cell/run.py and cell_client.run_trial."""

    CAMERAS: ClassVar[list[str]] = ["table", "side", "wrist"]
    IMAGE_SIZE = (640, 360)
    IMAGE_FORMAT = "png"
    ACTION = "tcp"
    PREFETCH = False  # every chunk is guarded against the pose it starts from; prefetch would start from a stale pose
    DEPTH: ClassVar[list[str]] = []
    DT = DT_S

    def __init__(
        self,
        checkpoint: str | None = None,
        model: ChunkModel | None = None,
        limits: GuardLimits | None = None,
        *,
        production: bool = False,
        saturday_cell: bool = False,
        approved_deployment: ReviewedDeployment | None = None,
    ):
        if production and saturday_cell:
            raise CellPolicySafetyError("production and saturday_cell modes are mutually exclusive")
        self.model = model or load_model(checkpoint)
        active_limits = limits or GuardLimits()
        self.deployment_attestation = (
            _attest_production_model(self.model, active_limits, approved_deployment) if production else None
        )
        if saturday_cell:
            self._validate_saturday_server()
        # Saturday keeps the one-row and T09 release fences while explicitly
        # deferring checkpoint attestation and measured-limit provenance.
        self.production = production or saturday_cell
        self.saturday_cell = saturday_cell
        self.guard = ChunkGuard(active_limits)
        self.instruction: str | None = None
        self.guard_log: list[dict] = []  # kept across rollouts, for the run's report
        self.last_guard_result: GuardResult | None = None
        self.release_authorized = False
        self._grasp_fence: tuple[tuple[float, ...], ExecutionTolerances] | None = None
        self.steps, self.rollout = 0, -1

    def _validate_saturday_server(self) -> None:
        if isinstance(self.model, StubModel):
            raise CellPolicySafetyError("the GR00T stub is forbidden in Saturday CELL mode")
        if not isinstance(self.model, GrootServerModel):
            raise CellPolicySafetyError("Saturday CELL mode requires a live GrootServerModel")
        if self.model.injected_client:
            raise CellPolicySafetyError("Saturday CELL mode forbids injected GR00T clients")

    def reset(self) -> None:
        self.instruction, self.steps, self.rollout = None, 0, self.rollout + 1
        self.last_guard_result = None
        self.release_authorized = False
        self._grasp_fence = None
        self.guard.set_target(None)
        if hasattr(self.model, "reset"):
            self.model.reset()

    def close(self) -> None:
        """Release an optional model transport without relying on GC teardown."""
        close = getattr(self.model, "close", None)
        if callable(close):
            close()

    def set_task(self, instruction: str | None, target: ContactTarget | None = None) -> None:
        """The planner's (T09) current step: the instruction GR00T gets and the contact target the guard enforces."""
        self.instruction = instruction
        self.guard.set_target(target)

    def set_grasp_fence(
        self,
        grasp_tcp: tuple[float, ...],
        tolerances: ExecutionTolerances,
    ) -> None:
        """Authorize deferring a moving close only at T09's reviewed grasp TCP."""
        grasp = tuple(float(value) for value in grasp_tcp)
        if len(grasp) != 6 or not np.isfinite(grasp).all():
            raise CellPolicySafetyError("grasp fence requires one finite six-axis TCP")
        self._grasp_fence = (grasp, tolerances)

    def authorize_release(self) -> None:
        """Authorize one observed model step to open the gripper after T09 verification."""
        self.release_authorized = True

    def set_plan(
        self,
        skill: str,
        object_id: str,
        base_id: str | None,
        grip: int,
        case: str,
        target,
        contact: ContactTarget | None = None,
    ) -> str:
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

    def _observed_state(self, obs: dict) -> tuple[np.ndarray, float, np.ndarray | None]:
        try:
            joints = np.asarray(obs["joints"], float)
        except (KeyError, TypeError, ValueError) as error:
            raise CellPolicySafetyError("CELL observation requires seven numeric joint values") from error
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise CellPolicySafetyError("CELL observation requires seven finite joint values")
        try:
            gripper = observed_gripper_state(joints[6], self.guard.limits)
        except UnsafeHoldError as error:
            raise CellPolicySafetyError(str(error)) from error
        if obs.get("tcp_pose") is None:
            return joints, gripper, None
        try:
            tcp = np.asarray(obs["tcp_pose"], float)
        except (TypeError, ValueError) as error:
            raise CellPolicySafetyError("CELL observation requires one numeric six-axis tcp_pose") from error
        if tcp.shape != (6,) or not np.isfinite(tcp).all():
            raise CellPolicySafetyError("CELL observation requires one finite six-axis tcp_pose")
        return joints, gripper, tcp

    def _production_refusal(
        self,
        chunk: np.ndarray,
        tcp: np.ndarray,
        gripper: float,
        release_authorized: bool,
    ) -> GuardResult | None:
        if not self.production:
            return None
        if chunk.shape != (1, 7):
            reason = "CELL production requires exactly one row per fresh observation"
        elif gripper == 1.0 and np.any(chunk[:, 6] < 0.5) and not release_authorized:
            reason = "gripper release is not authorized by the T09 verification gate"
        else:
            return None
        return GuardResult(hold_row(tcp, gripper, self.guard.limits), False, True, (reason,))

    def act(self, obs: dict) -> np.ndarray:
        self.steps += 1
        joints, gripper, tcp = self._observed_state(obs)
        if tcp is None:
            if self.production:
                raise CellPolicySafetyError("CELL production requires an observed tcp_pose; FK is not motion authority")
            pose = tcp_from_joints(joints)
            result = GuardResult(
                np.concatenate([pose, [gripper]])[None],
                False,
                True,
                ("tcp_pose missing: holding at the FK estimate",),
            )
            return self._record(result)
        model_obs = {
            "images": obs.get("images", {}),
            "joints": joints,
            "tcp_pose": tcp,
            "instruction": self._instruction(obs),
        }  # never obs["sim"]
        release_authorized = self.release_authorized
        self.release_authorized = False
        chunk = nearest_turns(self.model.predict(model_obs), tcp)
        chunk, close_deferred = self._defer_moving_close(chunk, tcp, gripper)
        refusal = self._production_refusal(chunk, tcp, gripper, release_authorized)
        if refusal is not None:
            return self._record(refusal)
        result = self.guard.guard(chunk, tcp, gripper, self.DT)
        if close_deferred and not result.held:
            result = GuardResult(
                result.rows,
                False,
                False,
                ("gripper close deferred until stationary at the reviewed T29 grasp TCP", *result.reasons),
            )
        return self._record(result)

    def _defer_moving_close(
        self,
        chunk: np.ndarray,
        tcp: np.ndarray,
        gripper: float,
    ) -> tuple[np.ndarray, bool]:
        """Split GR00T's near-grasp move+close before the unchanged ChunkGuard."""
        if self._grasp_fence is None or gripper != 0.0 or chunk.shape != (1, 7):
            return chunk, False
        row = chunk[0]
        if not np.isfinite(row).all() or abs(float(row[6]) - 1.0) > self.guard.limits.gripper_snap:
            return chunk, False
        grasp, tolerances = self._grasp_fence
        target = row[:6]
        xyz_error = float(np.linalg.norm(target[:3] - np.asarray(grasp[:3], dtype=float)))
        angle_error = float(np.abs((target[3:6] - grasp[3:6] + 180.0) % 360.0 - 180.0).max())
        current_xyz_error = float(np.linalg.norm(target[:3] - tcp[:3]))
        current_angle_error = float(np.abs((target[3:6] - tcp[3:6] + 180.0) % 360.0 - 180.0).max())
        if (
            xyz_error > tolerances.max_position_error_mm
            or angle_error > tolerances.max_orientation_error_deg
            or (
                current_xyz_error <= tolerances.max_position_error_mm
                and current_angle_error <= tolerances.max_orientation_error_deg
            )
        ):
            return chunk, False
        deferred = np.array(chunk, copy=True)
        deferred[0, 6] = 0.0
        return deferred, True

    def guard_fixed_inspection(
        self,
        obs: dict,
        move: FixedInspectionMove,
        tolerances: ExecutionTolerances,
    ) -> np.ndarray:
        """Guard one reviewed camera pose without invoking the learned model."""
        self.steps += 1
        _, gripper, tcp = self._observed_state(obs)
        if tcp is None:
            raise CellPolicySafetyError("fixed inspection requires an observed tcp_pose")
        if gripper != 0.0:
            raise CellPolicySafetyError(
                "fixed inspection requires an observed open gripper; policy refuses an unauthorized release"
            )
        self.guard.set_target(None)
        final_row = np.asarray([(*move.endpoint_tcp, 0.0)], dtype=float)
        reasons = self.guard.static_reasons(final_row, self.DT)
        if reasons:
            result = GuardResult(hold_row(tcp, gripper, self.guard.limits), False, True, reasons)
            return self._record(result)
        current_tcp = tuple(float(value) for value in tcp)
        leg_target = next_inspection_leg(current_tcp, move, tolerances)
        step = _incremental_fixed_pose(tcp, leg_target, self.guard.limits)
        row = np.concatenate([step, [0.0]])[None, :]
        return self._record(self.guard.guard(row, tcp, gripper, self.DT))

    def _record(self, res: GuardResult) -> np.ndarray:
        self.last_guard_result = res
        if res.reasons:
            self.guard_log.append(
                {"rollout": self.rollout, "step": self.steps, "held": res.held, "reasons": list(res.reasons)}
            )
            (log.warning if res.held else log.info)(
                "guard %s at act %d: %s", "HOLD" if res.held else "clamp", self.steps, "; ".join(res.reasons)
            )
        return res.rows
