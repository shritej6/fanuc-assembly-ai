"""CELL-only T09 controller whose executable rows come exclusively from GR00T."""

from __future__ import annotations

import math
from collections.abc import Sequence

from assembly.catalog import TRUSTED_GRASP_SOURCE
from assembly.controller import AssemblyController
from assembly.controller_ports import ControllerEvent
from assembly.execution import ExecutionError
from assembly.fixed_inspection import FixedInspectionMove
from assembly.motion import MotionChunk, MotionPhase, OrientedPartPose, Waypoint
from assembly.motion_intent import ContactEnvelope, MotionIntent, MotionIntentError, SemanticSkill
from assembly.planner import PreparedPlanStep
from policy_learn.deploy.cell_executor import T09GrootExecutor
from policy_learn.deploy.groot_queue import GrootGuardTrip, GrootMotionQueue
from robot.adapter import Action, Observation

_RETRY_LIFT_MM = 3.0  # docs/ROBOT_BRAIN.md@2a751da, section 3, "Won't seat".


class GrootAssemblyController(AssemblyController):
    """Keep T09 planning/recovery, replacing its scripted queue only in CELL."""

    def __init__(self, *, groot_executor: T09GrootExecutor, **kwargs) -> None:
        self._groot_executor = groot_executor
        super().__init__(**kwargs)

    def reset(self) -> None:
        super().reset()
        self._groot_executor.reset()
        self._motion_queue = GrootMotionQueue(self._groot_executor, self._execution_tolerances)

    def _queue_drop_release(
        self,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
        instance_id: str,
        reason: str,
    ) -> Action:
        """Fail closed until recovery has a semantic, identity-bound release skill."""
        del observation, observed_at, instance_id, reason
        self._hold("dropped-part release needs operator reconciliation in Saturday CELL mode")
        return current.as_action()

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
        # Deliberately discard deterministic coordinates. They remain useful to
        # the mock controller, but they are never CELL motion authority.
        del plan, observation, instance_id, object_id, base_instance_id, base_pose
        try:
            intent = self._intent_from_state(current, observed_at)
            self._motion_queue.queue(intent)
        except (ExecutionError, MotionIntentError, RuntimeError, ValueError) as error:
            self._hold(f"CELL semantic intent refused: {error}")
            return False
        return True

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
        except GrootGuardTrip as error:
            self._hold(f"GR00T guard trip: {error}")
            return current.as_action()
        except ExecutionError as error:
            self._hold(f"GR00T execution refused: {error}")
            return current.as_action()
        self._track_emitted_phase(instance_id)
        return action

    def _intent_from_state(
        self,
        current: Waypoint,
        observed_at: float,
    ) -> MotionIntent | FixedInspectionMove:
        if self._pending_pre_pick is not None:
            return self._inspection_intent(current, observed_at)
        if self._pending_install is not None:
            return self._release_intent(self._pending_install)
        if self._pending_insertion_lift is not None:
            return self._pull_out_intent(current)
        if self._pending_reject is not None:
            return self._defect_intent(self._pending_reject.prepared)
        if self._prepared is not None and self._awaiting_post:
            return self._pick_insert_intent(self._prepared)
        raise RuntimeError("T09 has no semantic state for the requested CELL motion")

    def _inspection_intent(
        self,
        current: Waypoint,
        observed_at: float,
    ) -> FixedInspectionMove:
        del observed_at
        pending = self._pending_pre_pick
        if pending is None:
            raise RuntimeError("inspection intent has no pending part")
        coarse = self._coarse_xy_by_instance[pending.part.instance_id]
        endpoint = (
            coarse[0], coarse[1], self._inspection_stop.tcp_z_mm,
            self._inspection_stop.tool_w_deg, self._inspection_stop.tool_p_deg, current.r,
        )
        return FixedInspectionMove(
            instance_id=pending.part.instance_id,
            base_instance_id=pending.base.instance_id,
            endpoint_tcp=endpoint,
            safe_z_mm=self._motion_config.safe_z_mm,
            source=(
                f"{self._inspection_stop.source}; safe Z from "
                f"{self._motion_config.safe_z_source}; observed inventory slot XY "
                f"for {pending.part.instance_id}"
            ),
        )

    def _release_intent(self, prepared: PreparedPlanStep) -> MotionIntent:
        release = self._command_tcp(prepared.target, prepared)
        endpoint = (*release[:2], release[2] + self._motion_config.retreat_mm, *release[3:])
        return self._intent(
            prepared, SemanticSkill.RELEASE_RETREAT, endpoint, 0.0, "nominal",
            release_tcp=release, release_authorized=True,
        )

    def _pull_out_intent(self, current: Waypoint) -> MotionIntent:
        pending = self._pending_insertion_lift
        if pending is None:
            raise RuntimeError("pull-out intent has no pending insertion")
        prepared = pending.prepared
        endpoint = (current.x, current.y, current.z + _RETRY_LIFT_MM, current.w, current.p, current.r)
        return self._intent(
            prepared, SemanticSkill.PULL_OUT, endpoint, 1.0, "wont_seat", grasp=prepared.target,
        )

    def _defect_intent(self, prepared: PreparedPlanStep) -> MotionIntent:
        endpoint = self._command_tcp(prepared.target, prepared)
        return self._intent(
            prepared, SemanticSkill.DEFECT_DROP, endpoint, 0.0, "defect",
            release_tcp=endpoint, release_authorized=True,
        )

    def _pick_insert_intent(self, prepared: PreparedPlanStep) -> MotionIntent:
        endpoint = self._command_tcp(prepared.target, prepared)
        contact = ContactEnvelope(
            endpoint[0], endpoint[1], endpoint[2],
            max(self._execution_tolerances.max_position_error_mm, 1e-9),
            f"{prepared.part.object_id} on {prepared.base.object_id}",
            "T09 execution tolerance around calibrated insertion TCP",
        )
        return self._intent(
            prepared, SemanticSkill.PICK_INSERT_HOLD, endpoint, 1.0, "nominal", contact=contact,
        )

    def _intent(
        self,
        prepared: PreparedPlanStep,
        skill: SemanticSkill,
        endpoint: tuple[float, float, float, float, float, float],
        final_gripper: float,
        case: str,
        *,
        grasp: OrientedPartPose | None = None,
        contact: ContactEnvelope | None = None,
        release_tcp: tuple[float, float, float, float, float, float] | None = None,
        release_authorized: bool = False,
    ) -> MotionIntent:
        active_grasp = grasp or prepared.grasp
        object_pose, origin_z, grasp_tcp, wrist_roll = self._grasp_fields(active_grasp, prepared)
        goal = prepared.verification_target or prepared.target
        return MotionIntent(
            skill=skill,
            instance_id=prepared.part.instance_id,
            object_id=prepared.part.object_id,
            base_instance_id=prepared.base.instance_id,
            base_object_id=prepared.base.object_id,
            grip_index=0,
            case=case,
            object_pose=object_pose,
            object_origin_z_mm=origin_z,
            wrist_roll_deg=wrist_roll,
            goal_pose=(goal.tcp.x, goal.tcp.y, goal.tcp.r),
            grasp_tcp=grasp_tcp,
            release_tcp=release_tcp,
            endpoint_tcp=endpoint,
            final_gripper=final_gripper,
            pose_source=f"{active_grasp.tcp.source} + T29 preferred grasp",
            grasp_source_sha256=TRUSTED_GRASP_SOURCE.sha256,
            contact=contact,
            release_authorized=release_authorized,
        )

    @staticmethod
    def _grasp_fields(
        grasp: OrientedPartPose,
        prepared: PreparedPlanStep,
    ) -> tuple[
        tuple[float, float, float],
        float,
        tuple[float, float, float, float, float, float],
        float,
    ]:
        preferred = prepared.recipe_step.preferred_grasp
        yaw = grasp.tcp.r
        angle = math.radians(yaw)
        local_x, local_y, local_z = preferred.tcp_local_mm
        origin_x = grasp.tcp.x - math.cos(angle) * local_x + math.sin(angle) * local_y
        origin_y = grasp.tcp.y - math.sin(angle) * local_x - math.cos(angle) * local_y
        origin_z = grasp.tcp.z - local_z
        wrist_roll = yaw + preferred.turn_deg
        command = (grasp.tcp.x, grasp.tcp.y, grasp.tcp.z, -180.0, 0.0, wrist_roll)
        return (origin_x, origin_y, yaw), origin_z, command, wrist_roll

    @staticmethod
    def _command_tcp(pose: OrientedPartPose, prepared: PreparedPlanStep) -> tuple[float, ...]:
        return (
            pose.tcp.x,
            pose.tcp.y,
            pose.tcp.z,
            -180.0,
            0.0,
            pose.tcp.r + prepared.recipe_step.preferred_grasp.turn_deg,
        )

    def _track_emitted_phase(self, instance_id: str) -> None:
        super()._track_emitted_phase(instance_id)
        phase = self._motion_queue.last_emitted_phase
        if phase is MotionPhase.INSPECT:
            detail = (
                f"{self._motion_queue.last_authority_label}: "
                f"{self._motion_queue.last_authority_source}"
            )
            self._events.append(
                ControllerEvent("inspection_motion", self.state_machine.current_product, None, detail)
            )
        elif phase is MotionPhase.VERTICAL_INSERT:
            self._events.append(
                ControllerEvent("groot", self.state_machine.current_product, None, "guarded semantic row")
            )
