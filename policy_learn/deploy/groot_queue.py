"""CELL queue for GR00T part motion and explicitly non-learned camera poses."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Number
from time import monotonic

from assembly.controller_ports import (
    CollisionScene,
    ControllerPortError,
    ExecutionTolerances,
    PerceptionPort,
    PerceptionTolerances,
    call_port,
    endpoint_reached,
    pose_agrees,
    require_complete,
    require_fresh,
)
from assembly.execution import (
    EndpointSettleLimits,
    EndpointSettleTimeout,
    ExecutionError,
    GrootWatchdogLimits,
)
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
        *,
        perception: PerceptionPort,
        perception_tolerances: PerceptionTolerances,
        watchdog_limits: GrootWatchdogLimits,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if not isinstance(settle_limits, EndpointSettleLimits):
            raise TypeError("CELL queue requires sourced EndpointSettleLimits")
        if not isinstance(perception_tolerances, PerceptionTolerances):
            raise TypeError("CELL queue requires sourced PerceptionTolerances")
        if not isinstance(watchdog_limits, GrootWatchdogLimits):
            raise TypeError("CELL queue requires sourced GrootWatchdogLimits")
        if not callable(clock):
            raise TypeError("CELL queue watchdog clock must be callable")
        self._executor = executor
        self._tolerances = tolerances
        self._settle_limits = settle_limits
        self._perception = perception
        self._perception_tolerances = perception_tolerances
        self._watchdog_limits = watchdog_limits
        self._clock = clock
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
        self._intent_started_at_s: float | None = None
        self._intent_deadline_s: float | None = None
        self._expected_base_pose: tuple[float, float, float, float] | None = None
        self._last_observed_gripper: float | None = None
        self._last_validated_scene: CollisionScene | None = None
        self._rows_since_scene = 0
        self._last_clock_s: float | None = None
        self._last_emitted_phase: MotionPhase | None = None

    def queue(
        self,
        intent: MotionIntent | FixedInspectionMove,
        *,
        observation: Observation,
        current: Waypoint,
        observed_at_s: float,
        base_pose: tuple[float, float, float, float],
    ) -> None:
        if self.active or self._expected_endpoint is not None:
            raise ExecutionError("CELL queue already owns an unfinished motion intent")
        if not isinstance(intent, (MotionIntent, FixedInspectionMove)):
            raise ExecutionError(
                "CELL queue accepts MotionIntent only for semantic motion; "
                "fixed inspection requires FixedInspectionMove"
            )
        expected_base_pose = self._pose(base_pose)
        timeout_s = self._watchdog_limits.timeout_s(self._watchdog_skill(intent))
        started_at = self._clock_time()
        self._intent = intent
        self._expected_base_pose = expected_base_pose
        self._intent_started_at_s = started_at
        self._intent_deadline_s = started_at + timeout_s
        self._last_observed_gripper = current.gripper
        try:
            self._check_watchdog(
                observation,
                current,
                observed_at_s,
                force_scene=True,
            )
        except (ExecutionError, OSError, RuntimeError, ValueError):
            self.clear()
            raise

    def confirm_previous_endpoint(
        self,
        observation: Observation,
        current: Waypoint,
        observed_at_s: float,
    ) -> Action | None:
        if self._expected_endpoint is None:
            return None
        gripper_changed = current.gripper != self._last_observed_gripper
        try:
            self._check_watchdog(
                observation,
                current,
                observed_at_s,
                force_scene=gripper_changed or current.gripper == 1.0,
            )
        except GrootGuardTrip:
            self.clear()
            raise
        self._last_observed_gripper = current.gripper
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
        return [*expected.as_action()[:6], gripper]

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
            scene = self._check_watchdog(
                observation,
                current,
                observed_at_s,
            )
            if isinstance(self._intent, FixedInspectionMove):
                self._last_authority_label = self._intent.label
                self._last_authority_source = self._intent.source
                result = self._executor.inspect(self._intent, observation=observation)
            else:
                self._last_authority_label = "groot"
                self._last_authority_source = None
                result = self._executor.act(
                    self._intent,
                    observation=observation,
                    scene=scene,
                )
            self._check_deadline(after_model=True)
        except GrootGuardTrip:
            self.clear()
            raise
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
            self._clear_watchdog()
            return result.action
        endpoint = Waypoint(*result.action)
        issued_at = self._time(observed_at_s)
        self._expected_endpoint = endpoint
        self._expected_issued_at_s = issued_at
        self._expected_deadline_s = issued_at + self._settle_limits.timeout_s(
            gripper_transition=endpoint.gripper != current.gripper,
        )
        self._rows_since_scene += 1
        return result.action

    def finish(self) -> None:
        if self.active or self._expected_endpoint is not None:
            raise ExecutionError("CELL queue cannot finish before its guarded endpoint")
        self.clear()

    def _clear_endpoint(self) -> None:
        self._expected_endpoint = None
        self._expected_issued_at_s = None
        self._expected_deadline_s = None

    def _clear_watchdog(self) -> None:
        self._intent_started_at_s = None
        self._intent_deadline_s = None
        self._expected_base_pose = None
        self._last_observed_gripper = None
        self._last_validated_scene = None
        self._rows_since_scene = 0
        self._last_clock_s = None

    def _check_watchdog(
        self,
        observation: Observation,
        current: Waypoint,
        observed_at_s: float,
        *,
        force_scene: bool = False,
    ) -> CollisionScene:
        self._check_deadline()
        due = self._rows_since_scene >= self._watchdog_limits.scene_interval_rows
        observation_changed = (
            self._last_validated_scene is None
            or self._last_validated_scene.observed_at_s != self._time(observed_at_s)
        )
        if not force_scene and not due and not observation_changed:
            return self._last_validated_scene
        scene = self._scene(observation, observed_at_s)
        try:
            self._validate_scene(scene, current)
        except GrootGuardTrip:
            raise
        except (ControllerPortError, ExecutionError, ValueError) as error:
            detail = str(error).strip() or type(error).__name__
            raise GrootGuardTrip(f"CELL watchdog scene validation refused: {detail}") from error
        self._last_validated_scene = scene
        self._rows_since_scene = 0
        self._check_deadline()
        return scene

    def _check_deadline(self, *, after_model: bool = False) -> None:
        now = self._clock_time()
        if self._intent_started_at_s is None or self._intent_deadline_s is None:
            raise ExecutionError("CELL intent watchdog is missing its original deadline")
        if now < self._intent_deadline_s:
            return
        timeout_s = self._intent_deadline_s - self._intent_started_at_s
        suffix = "; late model row discarded" if after_model else ""
        raise GrootGuardTrip(
            f"GR00T intent exceeded its {timeout_s:.3g} s time budget{suffix}; "
            f"source={self._watchdog_limits.source}"
        )

    def _scene(self, observation: Observation, observed_at_s: float) -> CollisionScene:
        intent = self._intent
        if intent is None or intent.base_instance_id is None:
            raise GrootGuardTrip("CELL watchdog requires identity-bound part and base instances")
        try:
            scene = call_port(
                "CELL watchdog perception",
                self._perception.collision_scene,
                observation,
                intent.instance_id,
                intent.base_instance_id,
            )
            if not isinstance(scene, CollisionScene):
                raise ControllerPortError("watchdog perception must return CollisionScene")
            require_fresh(scene.observed_at_s, observed_at_s, "watchdog collision scene")
            require_complete(scene.complete, "watchdog collision scene")
            return scene
        except (ControllerPortError, OSError, RuntimeError, ValueError) as error:
            detail = str(error).strip() or type(error).__name__
            raise GrootGuardTrip(f"CELL watchdog perception refused: {detail}") from error

    def _validate_scene(self, scene: CollisionScene, current: Waypoint) -> None:
        intent = self._intent
        expected_base = self._expected_base_pose
        if intent is None or intent.base_instance_id is None or expected_base is None:
            raise GrootGuardTrip("CELL watchdog lacks identity-bound base evidence")
        if scene.moving_instance_id != intent.instance_id or scene.base_instance_id != intent.base_instance_id:
            raise GrootGuardTrip("CELL watchdog scene is bound to the wrong physical instances")
        if type(scene.intrusion_detected) is not bool or scene.intrusion_detected:
            raise GrootGuardTrip("CELL watchdog reports a work-area intrusion")
        if type(scene.moving_attached) is not bool:
            raise GrootGuardTrip("CELL watchdog attachment evidence must be boolean")
        if current.gripper == 1.0 and not scene.moving_attached:
            raise GrootGuardTrip("moving-part attachment is lost while the gripper is closed")
        if current.gripper == 0.0 and scene.moving_attached:
            raise GrootGuardTrip("moving-part attachment disagrees with the open gripper")
        if not pose_agrees(scene.base_pose, expected_base, self._perception_tolerances):
            raise GrootGuardTrip("active product base moved outside perception tolerance")
        tilt_deg = self._time(scene.base_tilt_deg)
        if tilt_deg < 0.0:
            raise GrootGuardTrip("active product base tilt evidence must be nonnegative")
        if tilt_deg > self._watchdog_limits.base_tilt_limit_deg:
            raise GrootGuardTrip("active product base tilt exceeds the watchdog limit")

    @staticmethod
    def _watchdog_skill(intent: MotionIntent | FixedInspectionMove) -> str:
        if isinstance(intent, FixedInspectionMove):
            return "fixed_inspection"
        return intent.skill.value

    def _clock_time(self) -> float:
        now = self._time(self._clock())
        if self._last_clock_s is not None and now < self._last_clock_s:
            raise ExecutionError("CELL intent watchdog clock moved backwards")
        self._last_clock_s = now
        return now

    @classmethod
    def _pose(cls, value: object) -> tuple[float, float, float, float]:
        if not isinstance(value, tuple) or len(value) != 4:
            raise ExecutionError("CELL watchdog base pose must contain x, y, z, and yaw")
        result = tuple(cls._time(item) for item in value)
        return result  # type: ignore[return-value]

    @staticmethod
    def _time(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Number):
            raise ExecutionError("CELL endpoint settlement time must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ExecutionError("CELL endpoint settlement time must be finite")
        return result
