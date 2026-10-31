"""Per-callback guarded motion queue with fresh scene revalidation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

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
from assembly.motion import MotionChunk, MotionPhase, Waypoint
from assembly.safety import AABB, MotionOwner, SafetySupervisor
from robot.adapter import Action, Observation


class ExecutionError(RuntimeError):
    """Observed execution diverged from the previously guarded plan."""


class CellTrialHeld(ExecutionError):
    """CELL intentionally emitted no action and held only the active trial."""


class EndpointSettleTimeout(ExecutionError):
    """A guarded CELL row did not settle before its sourced deadline."""


@dataclass(frozen=True, slots=True)
class EndpointSettleLimits:
    """Explicit CELL endpoint-settlement deadlines; never motion tolerances."""

    tcp_timeout_s: float
    gripper_timeout_s: float
    status: Literal["PRELIMINARY", "MEASURED", "MOCK"]
    source: str

    def __post_init__(self) -> None:
        values = (self.tcp_timeout_s, self.gripper_timeout_s)
        if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("endpoint settle timeouts must be finite and positive")
        if self.status not in ("PRELIMINARY", "MEASURED", "MOCK"):
            raise ValueError("endpoint settle status must be PRELIMINARY, MEASURED, or MOCK")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("endpoint settle limits require a recorded source")

    def timeout_s(self, *, gripper_transition: bool) -> float:
        return self.gripper_timeout_s if gripper_transition else self.tcp_timeout_s


@dataclass(frozen=True, slots=True)
class GrootWatchdogLimits:
    """Sourced intent deadlines and recurring scene-check policy."""

    pick_insert_s: float
    release_retreat_s: float
    pull_out_s: float
    defect_drop_s: float
    inspection_s: float
    scene_interval_rows: int
    base_tilt_limit_deg: float
    status: Literal["PRELIMINARY", "MEASURED", "MOCK"]
    source: str

    def __post_init__(self) -> None:
        durations = (
            self.pick_insert_s,
            self.release_retreat_s,
            self.pull_out_s,
            self.defect_drop_s,
            self.inspection_s,
        )
        if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0.0 for value in durations):
            raise ValueError("GR00T watchdog durations must be finite and positive")
        if isinstance(self.scene_interval_rows, bool) or not isinstance(self.scene_interval_rows, int):
            raise TypeError("GR00T watchdog scene interval must be an integer")
        if self.scene_interval_rows <= 0:
            raise ValueError("GR00T watchdog scene interval must be positive")
        if isinstance(self.base_tilt_limit_deg, bool) or not math.isfinite(self.base_tilt_limit_deg):
            raise ValueError("GR00T watchdog base tilt limit must be finite")
        if not 0.0 <= self.base_tilt_limit_deg <= 180.0:
            raise ValueError("GR00T watchdog base tilt limit must be between 0 and at most 180 degrees")
        if self.status not in ("PRELIMINARY", "MEASURED", "MOCK"):
            raise ValueError("GR00T watchdog status must be PRELIMINARY, MEASURED, or MOCK")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("GR00T watchdog limits require a recorded source")

    def timeout_s(self, skill: str) -> float:
        try:
            return {
                "pick_insert_hold": self.pick_insert_s,
                "release_retreat": self.release_retreat_s,
                "pull_out": self.pull_out_s,
                "defect_drop": self.defect_drop_s,
                "fixed_inspection": self.inspection_s,
            }[skill]
        except KeyError as error:
            raise ValueError(f"no watchdog deadline is defined for skill {skill!r}") from error


class WorkAreaIntrusionError(ExecutionError):
    """A fresh scene reports an unexpected work-area intrusion."""


class DroppedPartError(ExecutionError):
    """The gripper is closed but perception reports no attached moving part."""


class GuardedMotionQueue:
    """Validate the full plan, then revalidate each chunk from actual TCP."""

    def __init__(
        self,
        *,
        perception: PerceptionPort,
        safety: SafetySupervisor,
        tolerances: ExecutionTolerances,
        perception_tolerances: PerceptionTolerances,
    ) -> None:
        self._perception = perception
        self._safety = safety
        self._tolerances = tolerances
        self._perception_tolerances = perception_tolerances
        self.clear()

    @property
    def active(self) -> bool:
        return bool(self._chunks)

    @property
    def instance_id(self) -> str | None:
        return self._instance_id

    @property
    def last_emitted_phase(self) -> MotionPhase | None:
        return self._last_emitted_phase

    @property
    def base_instance_id(self) -> str | None:
        return self._base_instance_id

    @property
    def pending_endpoint(self) -> Waypoint | None:
        return self._expected_endpoint

    def clear(self) -> None:
        self._chunks: list[MotionChunk] = []
        self._instance_id: str | None = None
        self._object_id: str | None = None
        self._base_instance_id: str | None = None
        self._base_pose: tuple[float, float, float, float] | None = None
        self._expected_endpoint: Waypoint | None = None
        self._last_emitted_phase: MotionPhase | None = None
        self._attachment_override: bool | None = None

    def confirm_previous_endpoint(self, current: Waypoint) -> None:
        if self._expected_endpoint is None:
            return
        if not endpoint_reached(current, self._expected_endpoint, self._tolerances):
            raise ExecutionError("observed TCP did not reach the previously guarded endpoint")
        self._expected_endpoint = None

    def queue(
        self,
        plan: Sequence[MotionChunk],
        *,
        observation: Observation,
        current: Waypoint,
        observed_at_s: float,
        instance_id: str,
        object_id: str,
        base_instance_id: str,
        base_pose: tuple[float, float, float, float],
    ) -> None:
        scene = self._scene(observation, instance_id, base_instance_id, observed_at_s)
        self._validate_scene_binding(scene, current, instance_id, base_instance_id, base_pose)
        self._chunks = list(self._guard_plan(plan, current, scene, instance_id, object_id))
        self._instance_id = instance_id
        self._object_id = object_id
        self._base_instance_id = base_instance_id
        self._base_pose = base_pose

    def queue_detached_release(
        self,
        plan: Sequence[MotionChunk],
        *,
        observation: Observation,
        current: Waypoint,
        observed_at_s: float,
        instance_id: str,
        object_id: str,
        base_instance_id: str,
        base_pose: tuple[float, float, float, float],
    ) -> None:
        scene = self._scene(observation, instance_id, base_instance_id, observed_at_s)
        self._validate_scene_binding(scene, current, instance_id, base_instance_id, base_pose, expected_attached=False)
        self._chunks = list(self._guard_plan(plan, current, scene, instance_id, object_id, attached=False))
        self._instance_id = instance_id
        self._object_id = object_id
        self._base_instance_id = base_instance_id
        self._base_pose = base_pose
        self._attachment_override = False

    def emit(
        self,
        *,
        observation: Observation,
        current: Waypoint,
        observed_at_s: float,
    ) -> Action:
        if not self._chunks or None in (self._instance_id, self._object_id, self._base_instance_id, self._base_pose):
            raise ExecutionError("motion queue is empty")
        scene = self._scene(observation, self._instance_id, self._base_instance_id, observed_at_s)
        self._validate_scene_binding(
            scene,
            current,
            self._instance_id,
            self._base_instance_id,
            self._base_pose,
            expected_attached=self._attachment_override,
        )
        chunk = self._chunks[0]
        attached = current.gripper == 1.0 if self._attachment_override is None else self._attachment_override
        owner = MotionOwner(self._instance_id, self._object_id, attached)
        guarded = self._safety.validate(
            (chunk,),
            start=current,
            fixtures=scene.fixtures,
            parts=scene.parts,
            owner=owner,
        )
        self._chunks.pop(0)
        self._last_emitted_phase = chunk.phase
        self._expected_endpoint = guarded[0].waypoints[-1]
        return guarded[0].as_action()

    def finish(self) -> None:
        if self._chunks or self._expected_endpoint is not None:
            raise ExecutionError("motion queue cannot finish before its endpoint is confirmed")
        self.clear()

    def _scene(
        self,
        observation: Observation,
        instance_id: str,
        base_instance_id: str,
        observed_at_s: float,
    ) -> CollisionScene:
        scene = call_port(
            "collision perception",
            self._perception.collision_scene,
            observation,
            instance_id,
            base_instance_id,
        )
        if not isinstance(scene, CollisionScene):
            raise ControllerPortError("collision port must return CollisionScene")
        require_fresh(scene.observed_at_s, observed_at_s, "collision scene")
        require_complete(scene.complete, "collision scene")
        if not isinstance(scene.intrusion_detected, bool):
            raise ControllerPortError("collision scene intrusion_detected must be a boolean")
        self._require_aabbs(scene.fixtures, "collision scene fixtures")
        self._require_aabbs(scene.parts, "collision scene parts")
        if scene.intrusion_detected:
            raise WorkAreaIntrusionError("work-area intrusion detected")
        return scene

    @staticmethod
    def _require_aabbs(value: object, label: str) -> None:
        if not isinstance(value, tuple) or any(not isinstance(item, AABB) for item in value):
            raise ControllerPortError(f"{label} must be a tuple of AABB values")

    def _validate_scene_binding(
        self,
        scene: CollisionScene,
        current: Waypoint,
        instance_id: str,
        base_instance_id: str,
        base_pose: tuple[float, float, float, float],
        expected_attached: bool | None = None,
    ) -> None:
        if scene.moving_instance_id != instance_id or scene.base_instance_id != base_instance_id:
            raise ExecutionError("collision scene is bound to the wrong physical instances")
        expected = current.gripper == 1.0 if expected_attached is None else expected_attached
        if expected and scene.moving_attached is False:
            raise DroppedPartError("moving-part attachment is lost while the gripper is closed")
        if scene.moving_attached is not expected:
            raise ExecutionError("moving-part attachment does not match the observed gripper state")
        if not pose_agrees(scene.base_pose, base_pose, self._perception_tolerances):
            raise ExecutionError("active product base moved outside measured perception tolerance")

    def _guard_plan(
        self,
        plan: Sequence[MotionChunk],
        current: Waypoint,
        scene: CollisionScene,
        instance_id: str,
        object_id: str,
        attached: bool | None = None,
    ) -> tuple[MotionChunk, ...]:
        guarded: list[MotionChunk] = []
        cursor = current
        for chunk in plan:
            owner = MotionOwner(instance_id, object_id, cursor.gripper == 1.0 if attached is None else attached)
            safe = self._safety.validate(
                (chunk,), start=cursor, fixtures=scene.fixtures, parts=scene.parts, owner=owner
            )[0]
            guarded.append(safe)
            cursor = safe.waypoints[-1]
        return tuple(guarded)
