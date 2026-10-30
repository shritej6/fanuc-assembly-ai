"""CELL queue for GR00T part motion and explicitly non-learned camera poses."""

from __future__ import annotations

from assembly.controller_ports import ExecutionTolerances, endpoint_reached
from assembly.execution import ExecutionError
from assembly.fixed_inspection import FixedInspectionMove
from assembly.motion import MotionPhase, Waypoint
from assembly.motion_intent import MotionIntent
from policy_learn.deploy.cell_executor import T09GrootExecutor
from robot.adapter import Action, Observation


class GrootGuardTrip(ExecutionError):
    """The learned-policy guard returned a hold row and a concrete reason."""


class GrootMotionQueue:
    """Preserve T09 lifecycle without accepting generic scripted MotionChunk data."""

    def __init__(self, executor: T09GrootExecutor, tolerances: ExecutionTolerances) -> None:
        self._executor = executor
        self._tolerances = tolerances
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

    def confirm_previous_endpoint(self, current: Waypoint) -> None:
        if self._expected_endpoint is None:
            return
        if not endpoint_reached(current, self._expected_endpoint, self._tolerances):
            raise ExecutionError("observed TCP did not reach the previous guarded GR00T row")
        self._expected_endpoint = None

    def emit(
        self,
        *,
        observation: Observation,
        current: Waypoint,
        observed_at_s: float,
    ) -> Action:
        del current, observed_at_s
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
        self._expected_endpoint = Waypoint(*result.action)
        return result.action

    def finish(self) -> None:
        if self.active or self._expected_endpoint is not None:
            raise ExecutionError("CELL queue cannot finish before its guarded endpoint")
        self.clear()
