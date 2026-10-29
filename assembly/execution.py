"""Per-callback guarded motion queue with fresh scene revalidation."""

from __future__ import annotations

from collections.abc import Sequence

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
        self._validate_scene_binding(scene, current, self._instance_id, self._base_instance_id, self._base_pose)
        chunk = self._chunks[0]
        owner = MotionOwner(self._instance_id, self._object_id, current.gripper == 1.0)
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
        self._require_aabbs(scene.fixtures, "collision scene fixtures")
        self._require_aabbs(scene.parts, "collision scene parts")
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
    ) -> None:
        if scene.moving_instance_id != instance_id or scene.base_instance_id != base_instance_id:
            raise ExecutionError("collision scene is bound to the wrong physical instances")
        if scene.moving_attached is not (current.gripper == 1.0):
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
    ) -> tuple[MotionChunk, ...]:
        guarded: list[MotionChunk] = []
        cursor = current
        for chunk in plan:
            owner = MotionOwner(instance_id, object_id, cursor.gripper == 1.0)
            safe = self._safety.validate(
                (chunk,), start=cursor, fixtures=scene.fixtures, parts=scene.parts, owner=owner
            )[0]
            guarded.append(safe)
            cursor = safe.waypoints[-1]
        return tuple(guarded)
