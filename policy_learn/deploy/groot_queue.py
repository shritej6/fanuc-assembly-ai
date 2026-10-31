"""CELL queue for GR00T part motion and explicitly non-learned camera poses."""

from __future__ import annotations

import math
from numbers import Number

from assembly.controller_ports import ExecutionTolerances, endpoint_reached
from assembly.execution import EndpointSettleLimits, EndpointSettleTimeout, ExecutionError
from assembly.fixed_inspection import FixedInspectionMove
from assembly.motion import MotionPhase, Waypoint
from assembly.motion_intent import MotionIntent
from policy_learn.deploy.cell_executor import T09GrootExecutor
from robot.adapter import Action, Observation


class GrootGuardTrip(ExecutionError):
    """The learned-policy guard returned a hold row and a concrete reason."""


class GrootMotionQueue:
    """Preserve T09 lifecycle without accepting generic scripted MotionChunk data."""

    def __init__(
        self,
        executor: T09GrootExecutor,
        tolerances: ExecutionTolerances,
        settle_limits: EndpointSettleLimits,
    ) -> None:
        if not isinstance(settle_limits, EndpointSettleLimits):
            raise TypeError("CELL queue requires sourced EndpointSettleLimits")
        self._executor = executor
        self._tolerances = tolerances
        self._settle_limits = settle_limits
        self._last_authority_label: str | None = None
        self._last_authority_source: str | None = None
        self.clear()

    @property
    def active(self) -> bool:
        return self._intent is not None

    @property
    def instance_id(self) -> str | None:
        return None if self._intent is None else self._intent.instance_id

    @property
    def base_instance_id(self) -> str | None:
        return None if self._intent is None else self._intent.base_instance_id

    @property
    def pending_endpoint(self) -> Waypoint | None:
        return self._expected_endpoint

    @property
    def last_emitted_phase(self) -> MotionPhase | None:
        return self._last_emitted_phase

    @property
    def last_authority_label(self) -> str | None:
        return self._last_authority_label

    @property
    def last_authority_source(self) -> str | None:
        return self._last_authority_source

    def clear(self) -> None:
        self._intent: MotionIntent | FixedInspectionMove | None = None
        self._expected_endpoint: Waypoint | None = None
        self._expected_issued_at_s: float | None = None
        self._expected_deadline_s: float | None = None
        self._last_emitted_phase: MotionPhase | None = None

    def queue(self, intent: MotionIntent | FixedInspectionMove) -> None:
        if self.active or self._expected_endpoint is not None:
            raise ExecutionError("CELL queue already owns an unfinished motion intent")
        if not isinstance(intent, (MotionIntent, FixedInspectionMove)):
            raise ExecutionError(
                "CELL queue accepts MotionIntent only for semantic motion; "
                "fixed inspection requires FixedInspectionMove"
            )
        self._intent = intent

    def confirm_previous_endpoint(self, current: Waypoint, observed_at_s: float) -> Action | None:
        if self._expected_endpoint is None:
            return None
        now = self._time(observed_at_s)
        if self._expected_issued_at_s is None or self._expected_deadline_s is None:
            raise ExecutionError("CELL endpoint settlement is missing its original deadline")
        if now < self._expected_issued_at_s:
            raise ExecutionError("CELL endpoint settlement observation time moved backwards")
        expected = self._expected_endpoint
        if endpoint_reached(current, expected, self._tolerances):
            self._clear_endpoint()
            return None
        if now >= self._expected_deadline_s:
            detail = (
                "guarded CELL endpoint did not settle within "
                f"{self._expected_deadline_s - self._expected_issued_at_s:.3g} s at "
                f"{self._tolerances.max_position_error_mm} mm / "
                f"{self._tolerances.max_orientation_error_deg} deg; "
                f"source={self._settle_limits.source}"
            )
            self.clear()
            raise EndpointSettleTimeout(detail)
        pose_only = Waypoint(*current.as_action()[:6], expected.gripper)
        gripper = expected.gripper if endpoint_reached(pose_only, expected, self._tolerances) else current.gripper
        return [*current.as_action()[:6], gripper]

    def emit(
        self,
        *,
        observation: Observation,
        current: Waypoint,
        observed_at_s: float,
    ) -> Action:
        if self._expected_endpoint is not None:
            raise ExecutionError(
                "CELL cannot emit another GR00T row before the previous endpoint settles"
            )
        if self._intent is None:
            raise ExecutionError("CELL motion queue is empty")
        try:
            if isinstance(self._intent, FixedInspectionMove):
                self._last_authority_label = self._intent.label
                self._last_authority_source = self._intent.source
                result = self._executor.inspect(self._intent, observation=observation)
            else:
                self._last_authority_label = "groot"
                self._last_authority_source = None
                result = self._executor.act(self._intent, observation=observation, scene=None)
        except (OSError, RuntimeError, ValueError) as error:
            self.clear()
            detail = str(error).strip() or type(error).__name__
            raise ExecutionError(f"GR00T boundary failed: {detail}") from error
        self._last_emitted_phase = result.phase
        if result.blocked:
            reasons = "; ".join(result.reasons) or "unspecified learned-policy guard refusal"
            self.clear()
            raise GrootGuardTrip(reasons)
        if result.reached_endpoint:
            self._intent = None
            return result.action
        endpoint = Waypoint(*result.action)
        issued_at = self._time(observed_at_s)
        self._expected_endpoint = endpoint
        self._expected_issued_at_s = issued_at
        self._expected_deadline_s = issued_at + self._settle_limits.timeout_s(
            gripper_transition=endpoint.gripper != current.gripper,
        )
        return result.action

    def finish(self) -> None:
        if self.active or self._expected_endpoint is not None:
            raise ExecutionError("CELL queue cannot finish before its guarded endpoint")
        self.clear()

    def _clear_endpoint(self) -> None:
        self._expected_endpoint = None
        self._expected_issued_at_s = None
        self._expected_deadline_s = None

    @staticmethod
    def _time(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Number):
            raise ExecutionError("CELL endpoint settlement time must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ExecutionError("CELL endpoint settlement time must be finite")
        return result
