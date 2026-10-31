"""One-row T09-to-GR00T execution with independent live-scene safety validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from assembly.controller_ports import (
    CollisionScene,
    ExecutionTolerances,
    endpoint_reached,
    observation_time,
    require_complete,
    require_fresh,
)
from assembly.fixed_inspection import FixedInspectionMove
from assembly.motion import MotionChunk, MotionPhase, Waypoint
from assembly.motion_intent import MotionIntent, SemanticSkill, semantic_instruction
from assembly.safety import MotionOwner, SafetySupervisor, SafetyViolation
from policy_learn.deploy.guards import ContactTarget, GuardResult, UnsafeHoldError, observed_gripper_state


class CellExecutionError(RuntimeError):
    """A learned motion cannot be proven safe for the current observation."""


_T18_EXECUTABLE_SKILLS = frozenset(
    {
        SemanticSkill.PICK_INSERT_HOLD,
        SemanticSkill.RELEASE_RETREAT,
        SemanticSkill.PULL_OUT,
        SemanticSkill.DEFECT_DROP,
    }
)


class GuardedPolicy(Protocol):
    last_guard_result: GuardResult | None

    def reset(self) -> None: ...

    def close(self) -> None: ...

    def set_task(self, instruction: str, target: ContactTarget | None = None) -> None: ...

    def authorize_release(self) -> None: ...

    def act(self, observation: dict) -> np.ndarray: ...

    def guard_fixed_inspection(
        self,
        observation: dict,
        move: FixedInspectionMove,
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class GrootExecutionResult:
    action: list[float]
    phase: MotionPhase
    reached_endpoint: bool
    blocked: bool
    reasons: tuple[str, ...]


def _contact(intent: MotionIntent) -> ContactTarget | None:
    value = intent.contact
    if value is None:
        return None
    return ContactTarget(
        value.x_mm,
        value.y_mm,
        value.z_contact_mm,
        radius_mm=value.radius_mm,
        label=value.label,
    )


def _phase(intent: MotionIntent, current: Waypoint, target: Waypoint) -> MotionPhase:
    if current.gripper == 0.0 and target.gripper == 1.0:
        return MotionPhase.GRIP
    if current.gripper == 1.0 and target.gripper == 0.0:
        return MotionPhase.RELEASE
    return {
        SemanticSkill.INSPECT_VIEW: MotionPhase.INSPECT,
        SemanticSkill.PICK_INSERT_HOLD: MotionPhase.VERTICAL_INSERT,
        SemanticSkill.RELEASE_RETREAT: MotionPhase.RETREAT,
        SemanticSkill.PULL_OUT: MotionPhase.LIFT,
        SemanticSkill.PLACE_FREE: MotionPhase.CARRY_YAW,
        SemanticSkill.DEFECT_DROP: MotionPhase.CARRY_YAW,
    }[intent.skill]


def _skill_phase(intent: MotionIntent) -> MotionPhase:
    return {
        SemanticSkill.INSPECT_VIEW: MotionPhase.INSPECT,
        SemanticSkill.PICK_INSERT_HOLD: MotionPhase.VERTICAL_INSERT,
        SemanticSkill.RELEASE_RETREAT: MotionPhase.RETREAT,
        SemanticSkill.PULL_OUT: MotionPhase.LIFT,
        SemanticSkill.PLACE_FREE: MotionPhase.CARRY_YAW,
        SemanticSkill.DEFECT_DROP: MotionPhase.CARRY_YAW,
    }[intent.skill]


def _waypoint(row: np.ndarray) -> Waypoint:
    if row.shape != (7,) or not np.isfinite(row).all():
        raise CellExecutionError("GR00T guard did not return one finite seven-axis row")
    if float(row[6]) not in (0.0, 1.0):
        raise CellExecutionError("GR00T guard returned a non-binary gripper command")
    return Waypoint(*(float(value) for value in row))


def _observed_waypoint(observation: dict) -> Waypoint:
    try:
        pose = np.asarray(observation["tcp_pose"], dtype=float)
        joints = np.asarray(observation["joints"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise CellExecutionError("observation must contain numeric tcp_pose and joints") from error
    if pose.shape != (6,) or not np.isfinite(pose).all() or joints.shape != (7,) or not np.isfinite(joints).all():
        raise CellExecutionError("observation must contain one finite six-axis TCP and seven joints")
    try:
        gripper = observed_gripper_state(joints[6])
    except UnsafeHoldError as error:
        raise CellExecutionError(str(error)) from error
    return Waypoint(*(float(value) for value in pose), gripper)


def _tcp(waypoint: Waypoint) -> tuple[float, float, float, float, float, float]:
    return (waypoint.x, waypoint.y, waypoint.z, waypoint.w, waypoint.p, waypoint.r)


def _pose_reached(actual: Waypoint, pose: tuple[float, ...], grip: float, tolerances: ExecutionTolerances) -> bool:
    return endpoint_reached(actual, Waypoint(*pose, grip), tolerances)


class T09GrootExecutor:
    """Execute one semantic intent at a time; never fall back to scripted TCP motion."""

    def __init__(
        self,
        *,
        policy: GuardedPolicy,
        safety: SafetySupervisor | None,
        tolerances: ExecutionTolerances,
        require_scene_safety: bool = True,
    ) -> None:
        if require_scene_safety and safety is None:
            raise ValueError("scene safety requires a SafetySupervisor")
        self._policy = policy
        self._safety = safety
        self._tolerances = tolerances
        self._require_scene_safety = require_scene_safety

    def reset(self) -> None:
        """Start a fresh rollout in both the T09 and served-policy layers."""
        self._policy.reset()

    def close(self) -> None:
        """Release the guarded policy's network transport deterministically."""
        self._policy.close()

    def act(
        self,
        intent: MotionIntent,
        *,
        observation: dict,
        scene: CollisionScene | None,
    ) -> GrootExecutionResult:
        if intent.skill not in _T18_EXECUTABLE_SKILLS:
            raise CellExecutionError(f"{intent.skill.value} is absent from the T18 dataset")
        current = _observed_waypoint(observation)
        if self._require_scene_safety:
            if scene is None:
                raise CellExecutionError("live collision scene is required")
            self._validate_scene(intent, observation, current, scene)
        expected = Waypoint(*intent.endpoint)
        if endpoint_reached(current, expected, self._tolerances):
            return GrootExecutionResult(current.as_action(), _skill_phase(intent), True, False, ())
        self._policy.set_task(semantic_instruction(intent), _contact(intent))
        release_ready = (
            intent.release_authorized
            and intent.release_tcp is not None
            and _pose_reached(current, intent.release_tcp, 1.0, self._tolerances)
        )
        if release_ready:
            self._policy.authorize_release()
        rows = np.asarray(self._policy.act(observation), dtype=float)
        result = self._policy.last_guard_result
        if result is None or rows.shape != (1, 7):
            raise CellExecutionError("GR00T policy did not expose one guarded execution row")
        target = _waypoint(rows[0])
        phase = _phase(intent, current, target)
        if result.held:
            return GrootExecutionResult(target.as_action(), phase, False, True, result.reasons)
        self._validate_release_fence(current, target, release_ready)
        self._validate_gripper_transition(intent, current, target, release_ready)
        if self._require_scene_safety:
            assert scene is not None
            self._validate_motion(intent, current, target, phase, scene)
        return GrootExecutionResult(target.as_action(), phase, False, False, result.reasons)

    def inspect(self, move: FixedInspectionMove, *, observation: dict) -> GrootExecutionResult:
        """Execute a reviewed non-learned camera pose through the policy guard."""
        current = _observed_waypoint(observation)
        if current.gripper != 0.0:
            raise CellExecutionError(
                "fixed inspection requires an observed open gripper; CELL refuses an unauthorized release"
            )
        expected = Waypoint(*move.endpoint_tcp, 0.0)
        if endpoint_reached(current, expected, self._tolerances):
            return GrootExecutionResult(current.as_action(), MotionPhase.INSPECT, True, False, ())
        rows = np.asarray(
            self._policy.guard_fixed_inspection(observation, move),
            dtype=float,
        )
        result = self._policy.last_guard_result
        if result is None or rows.shape != (1, 7):
            raise CellExecutionError("fixed inspection guard did not expose one execution row")
        target = _waypoint(rows[0])
        if target.gripper != 0.0:
            raise CellExecutionError("fixed inspection requires an open gripper")
        return GrootExecutionResult(
            target.as_action(),
            MotionPhase.INSPECT,
            False,
            result.held,
            result.reasons,
        )

    def _validate_release_fence(
        self,
        current: Waypoint,
        target: Waypoint,
        release_ready: bool,
    ) -> None:
        if not release_ready:
            return
        stationary = _pose_reached(target, _tcp(current), target.gripper, self._tolerances)
        if target.gripper != 0.0 or not stationary:
            raise CellExecutionError("the first row at an authorized release TCP must open while stationary")

    def _validate_gripper_transition(
        self,
        intent: MotionIntent,
        current: Waypoint,
        target: Waypoint,
        release_ready: bool,
    ) -> None:
        if current.gripper == target.gripper:
            return
        stationary = _pose_reached(target, _tcp(current), target.gripper, self._tolerances)
        if current.gripper == 0.0 and target.gripper == 1.0:
            at_grasp = _pose_reached(current, intent.grasp_tcp, 0.0, self._tolerances)
            if not at_grasp or not stationary:
                raise CellExecutionError("GR00T may close only while stationary at the reviewed T29 grasp TCP")
            return
        if current.gripper == 1.0 and target.gripper == 0.0 and release_ready and stationary:
            return
        raise CellExecutionError("GR00T may open only while stationary at the authorized release TCP")

    @staticmethod
    def _validate_scene(
        intent: MotionIntent,
        observation: dict,
        current: Waypoint,
        scene: CollisionScene,
    ) -> None:
        try:
            require_fresh(scene.observed_at_s, observation_time(observation), "collision scene")
            require_complete(scene.complete, "collision scene")
        except ValueError as error:
            raise CellExecutionError(str(error)) from error
        if scene.intrusion_detected:
            raise CellExecutionError("work-area intrusion detected")
        if scene.moving_instance_id != intent.instance_id:
            raise CellExecutionError("collision scene is bound to the wrong moving instance")
        if intent.base_instance_id is not None and scene.base_instance_id != intent.base_instance_id:
            raise CellExecutionError("collision scene is bound to the wrong base instance")
        if scene.moving_attached is not (current.gripper == 1.0):
            raise CellExecutionError("collision scene attachment disagrees with the observed gripper")

    def _validate_motion(
        self,
        intent: MotionIntent,
        current: Waypoint,
        target: Waypoint,
        phase: MotionPhase,
        scene: CollisionScene,
    ) -> None:
        if self._safety is None:  # constructor rejects this in scene-safety mode
            raise CellExecutionError("live-scene SafetySupervisor is unavailable")
        owner = MotionOwner(intent.instance_id, intent.object_id, current.gripper == 1.0)
        try:
            self._safety.validate(
                (MotionChunk(phase, (target,)),),
                start=current,
                fixtures=scene.fixtures,
                parts=scene.parts,
                owner=owner,
            )
        except SafetyViolation as error:
            raise CellExecutionError(f"live-scene safety refused GR00T row: {error}") from error
