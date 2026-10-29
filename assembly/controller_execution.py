"""Guarded T09 motion queue execution and boundary handling."""

from __future__ import annotations

from collections.abc import Sequence

from assembly.controller_ports import ControllerPortError, oriented_pose_values
from assembly.execution import ExecutionError
from assembly.inventory import ObjectCondition
from assembly.motion import MotionChunk, MotionPhase, OrientedPartPose, Waypoint
from assembly.safety import SafetyViolation
from robot.adapter import Action, Observation


class ExecutionControllerMixin:
    """Emit only safety-checked chunks owned by live inventory instances."""

    def _emit_motion_chunk(
        self,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> Action:
        instance_id, item = self._motion_inventory_context(current)
        if item is None:
            return current.as_action()
        if not self._moving_item_is_usable(instance_id, item.condition):
            self._abort(f"moving instance {item.instance_id!r} is no longer verified")
            return current.as_action()
        return self._emit_guarded_action(observation, current, observed_at, instance_id)

    def _motion_inventory_context(self, current: Waypoint):
        instance_id = self._motion_queue.instance_id
        if instance_id is None:
            raise RuntimeError("queued motion has no physical instance owner")
        base_instance_id = self._motion_queue.base_instance_id
        if base_instance_id is None:
            raise RuntimeError("queued motion has no physical base reference")
        base = self.inventory.item(base_instance_id)
        if base.condition is not ObjectCondition.VERIFIED:
            self._abort("motion reference base must remain verified")
            return instance_id, None
        return instance_id, self.inventory.item(instance_id)

    def _moving_item_is_usable(self, instance_id: str, condition: ObjectCondition) -> bool:
        if condition is ObjectCondition.VERIFIED:
            return True
        pending_reject = self._pending_reject
        pending_pick = self._pending_pre_pick
        rejecting = pending_reject is not None and pending_reject.prepared.part.instance_id == instance_id
        inspecting = (
            pending_pick is not None
            and self._forced_reject_condition is not None
            and pending_pick.part.instance_id == instance_id
        )
        return (rejecting or inspecting) and condition in {ObjectCondition.FALLEN, ObjectCondition.DEFECTIVE}

    def _emit_guarded_action(
        self,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
        instance_id: str,
    ) -> Action:
        try:
            action = self._motion_queue.emit(
                observation=observation,
                current=current,
                observed_at_s=observed_at,
            )
        except (ControllerPortError, ExecutionError, SafetyViolation) as error:
            self._abort(str(error))
            return current.as_action()
        self._track_emitted_phase(instance_id)
        return action

    def _track_emitted_phase(self, instance_id: str) -> None:
        if self._motion_queue.last_emitted_phase is MotionPhase.GRIP:
            self._part_held = True
            self._held_instance_id = instance_id
        if self._motion_queue.last_emitted_phase is MotionPhase.RELEASE:
            self._part_held = False
            self._held_instance_id = None
            self._release_dispatched = True

    def _set_guarded_motion(
        self,
        plan: Sequence[MotionChunk],
        observation: Observation,
        current: Waypoint,
        observed_at: float,
        instance_id: str,
        object_id: str,
        base_instance_id: str,
        base_pose: OrientedPartPose,
    ) -> bool:
        try:
            self._motion_queue.queue(
                plan,
                observation=observation,
                current=current,
                observed_at_s=observed_at,
                instance_id=instance_id,
                object_id=object_id,
                base_instance_id=base_instance_id,
                base_pose=oriented_pose_values(base_pose),
            )
        except (ControllerPortError, ExecutionError, SafetyViolation) as error:
            self._abort(str(error))
            return False
        return True
