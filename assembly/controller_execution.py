"""Guarded controller motion queue execution and boundary handling.
Authored by Shritej Gunda.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from assembly.controller_ports import ControllerEvent, ControllerPortError, oriented_pose_values
from assembly.execution import DroppedPartError, ExecutionError, WorkAreaIntrusionError
from assembly.inventory import ObjectCondition
from assembly.motion import (
    MotionChunk,
    MotionError,
    MotionPhase,
    OrientedPartPose,
    Waypoint,
    compile_reinsert,
    compile_transfer_held,
)
from assembly.planner import PlannerError
from assembly.safety import CollisionViolation, SafetyViolation
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
        except WorkAreaIntrusionError as error:
            self._freeze_for_intrusion(str(error), self._motion_queue.instance_id)
            return current.as_action()
        except DroppedPartError as error:
            return self._queue_drop_release(observation, current, observed_at, instance_id, str(error))
        except CollisionViolation:
            self._hold(self._crowded_dependency_reason())
            return current.as_action()
        except (ControllerPortError, ExecutionError, SafetyViolation) as error:
            self._abort(str(error))
            return current.as_action()
        self._intrusion_attempts = 0
        self._track_emitted_phase(instance_id)
        return action

    def _queue_drop_release(self, observation, current, observed_at, instance_id, reason):
        prepared = self._drop_recovery_plan(instance_id, reason)
        if prepared is None:
            return current.as_action()
        release = Waypoint(current.x, current.y, current.z, current.w, current.p, current.r, 0.0)
        self._motion_queue.clear()
        self._prepared = None
        self._pending_reject = None
        self._pending_insertion_lift = None
        self._awaiting_post = False
        self._pending_drop_release = instance_id
        return self._emit_detached_release(
            prepared, release, observation, current, observed_at, instance_id
        )

    def _drop_recovery_plan(self, instance_id, reason):
        prepared = self._prepared
        if prepared is None and self._pending_reject is not None:
            prepared = self._pending_reject.prepared
        if prepared is None:
            self._hold(f"{reason}; drop recovery lacks a safe active plan")
            return None
        attempts = self._drop_attempts_by_instance.get(instance_id, 0) + 1
        self._drop_attempts_by_instance[instance_id] = attempts
        if attempts > 1:
            self._hold(f"{reason}; part dropped again after one re-pick")
            return None
        return prepared

    def _emit_detached_release(self, prepared, release, observation, current, observed_at, instance_id):
        try:
            self._motion_queue.queue_detached_release(
                (MotionChunk(MotionPhase.RELEASE, (release,)),),
                observation=observation,
                current=current,
                observed_at_s=observed_at,
                instance_id=instance_id,
                object_id=prepared.part.object_id,
                base_instance_id=prepared.base.instance_id,
                base_pose=oriented_pose_values(prepared.base_pose),
            )
            action = self._motion_queue.emit(
                observation=observation,
                current=current,
                observed_at_s=observed_at,
            )
        except (ControllerPortError, ExecutionError, SafetyViolation) as error:
            self._pending_drop_release = None
            self._hold(f"drop recovery release failed: {error}")
            return current.as_action()
        self._track_emitted_phase(instance_id)
        return action

    def _commit_drop_release(self) -> None:
        instance_id = self._pending_drop_release
        if instance_id is None:
            raise RuntimeError("drop release has no physical instance")
        self._motion_queue.finish()
        self._pending_drop_release = None
        self._part_held = False
        self._held_instance_id = None
        item = self.inventory.item(instance_id)
        if instance_id not in self._coarse_xy_by_instance or item.condition is ObjectCondition.NOT_OBSERVED:
            self._hold(f"dropped part {instance_id!r} was not found in the fresh post-drop scan")
            return
        if item.condition is ObjectCondition.VERIFIED:
            detail = "fresh scan upright; re-pick"
        elif item.condition in {ObjectCondition.FALLEN, ObjectCondition.DEFECTIVE}:
            detail = "fresh scan fallen or defective; reject then spare"
        else:
            self._hold(f"dropped part {instance_id!r} has unresolved post-drop condition")
            return
        self._events.append(
            ControllerEvent("dropped", self.state_machine.current_product, item.object_id, detail)
        )

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
        except WorkAreaIntrusionError as error:
            self._freeze_for_intrusion(str(error), instance_id)
            return False
        except CollisionViolation:
            self._hold(self._crowded_dependency_reason())
            return False
        except (ControllerPortError, ExecutionError, SafetyViolation) as error:
            self._abort(str(error))
            return False
        self._intrusion_attempts = 0
        return True

    def _freeze_for_intrusion(self, reason: str, instance_id: str | None) -> None:
        if self._intrusion_attempts:
            self._hold(f"{reason} persists after rescan")
            return
        self._intrusion_attempts = 1
        self._intrusion_waiting_for_rescan = True
        self._pending_intrusion_replan = True
        self._motion_queue.clear()
        self._events.append(
            ControllerEvent("intrusion", self.state_machine.current_product, instance_id, "freeze and rescan")
        )

    @staticmethod
    def _crowded_dependency_reason() -> str:
        return (
            "motion blocked by collision; automatic crowded-part relocation requires "
            "typed blocker identity and a calibrated empty staging target"
        )

    def _replan_after_intrusion(self, observation, current, observed_at) -> bool:
        self._pending_intrusion_replan = False
        blocked_state = self._intrusion_replan_blocked_state()
        if blocked_state is not None:
            self._hold(f"intrusion during {blocked_state} requires operator reconciliation")
            return False
        if current.gripper == 0.0:
            if self._release_dispatched:
                self._hold("intrusion after release requires operator reconciliation")
                return False
            self._prepared = None
            self._pending_pre_pick = None
            self._pending_reject = None
            self._forced_reject_condition = None
            self._awaiting_post = False
            return False
        prepared = self._prepared
        if prepared is None and self._pending_reject is not None:
            prepared = self._pending_reject.prepared
        if prepared is None:
            self._hold("intrusion replan cannot identify the held physical part")
            return False
        try:
            refreshed, chunks = self._held_intrusion_plan(prepared, current, observed_at)
        except (MotionError, PlannerError, ValueError) as error:
            self._hold(f"intrusion replan failed: {error}")
            return False
        return self._queue_intrusion_plan(refreshed, chunks, observation, current, observed_at)

    def _intrusion_replan_blocked_state(self) -> str | None:
        states = (
            (self._pending_install is not None, "verified release"),
            (self._pending_insertion_lift is not None, "insertion recovery lift"),
        )
        return next((name for active, name in states if active), None)

    def _resume_physical_blocker(self) -> str | None:
        states = (
            (self._pending_install is not None, "verified release"),
            (self._pending_insertion_lift is not None, "insertion recovery lift"),
            (self._pending_reject is not None, "reject transfer"),
            (self._pending_drop_release is not None, "drop release"),
            (self._pending_pre_pick is not None, "inspection stop"),
            (self._awaiting_post, "insertion verification"),
            (self._release_dispatched, "uncommitted release"),
            (self._part_held, "held part"),
        )
        return next((name for active, name in states if active), None)

    def _held_intrusion_plan(self, prepared, current, observed_at):
        if self._pending_reject is not None:
            base_pose = self._poses.base_for(prepared.base, observed_at_s=observed_at)
            target = self._recovery_targets.reject_target_for(prepared.part)
            refreshed = replace(prepared, base_pose=base_pose, target=target)
            chunks = compile_transfer_held(
                current,
                target,
                gripper_roll_deg=prepared.record.gripper_roll_deg,
                config=self._motion_config,
                max_position_error_mm=self._execution_tolerances.max_position_error_mm,
                max_orientation_error_deg=self._execution_tolerances.max_orientation_error_deg,
            )
            self._pending_reject = replace(self._pending_reject, prepared=refreshed, target=target)
            return refreshed, chunks
        candidate = self._planner.next_candidate(
            self.inventory,
            observed_at_s=observed_at,
            attempt=1,
        )
        if candidate is None or candidate.part.instance_id != prepared.part.instance_id:
            raise PlannerError("fresh plan does not own the held physical part")
        refreshed = replace(
            prepared,
            base=candidate.base,
            base_pose=candidate.base_pose,
            target=candidate.target,
            verification_target=candidate.verification_target,
        )
        self._prepared = refreshed
        chunks = compile_reinsert(
            current,
            refreshed.target,
            gripper_roll_deg=refreshed.record.gripper_roll_deg,
            config=self._motion_config,
        )
        return refreshed, chunks

    def _queue_intrusion_plan(self, prepared, chunks, observation, current, observed_at):
        return self._set_guarded_motion(
            chunks,
            observation,
            current,
            observed_at,
            prepared.part.instance_id,
            prepared.part.object_id,
            prepared.base.instance_id,
            prepared.base_pose,
        )
