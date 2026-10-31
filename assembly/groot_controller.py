"""CELL-only T09 controller whose executable rows come exclusively from GR00T."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum

from assembly.catalog import TRUSTED_GRASP_SOURCE
from assembly.contact_geometry import contact_envelope_radius_mm, contact_geometry_source
from assembly.controller import AssemblyController
from assembly.controller_ports import ControllerEvent, oriented_pose_values
from assembly.execution import (
    CellTrialHeld,
    EndpointSettleLimits,
    ExecutionError,
    GrootWatchdogLimits,
)
from assembly.fixed_inspection import FixedInspectionMove, InspectionPoseRegistry
from assembly.motion import MotionChunk, MotionPhase, OrientedPartPose, Waypoint
from assembly.motion_intent import ContactEnvelope, MotionIntent, MotionIntentError, SemanticSkill
from assembly.planner import PreparedPlanStep
from assembly.recovery import PullOutTarget
from policy_learn.deploy.cell_executor import T09GrootExecutor
from policy_learn.deploy.groot_queue import GrootGuardTrip, GrootMotionQueue
from robot.adapter import Action, Observation


class GrootControllerAuthority(str, Enum):
    """Reviewed runtime provenance carried by the concrete CELL controller."""

    REAL_CELL = "real_cell"
    OFFICIAL_MOCK_PROOF = "official_mock_proof"


class GrootAssemblyController(AssemblyController):
    """Keep T09 planning/recovery, replacing its scripted queue only in CELL."""

    def __init__(
        self,
        *,
        groot_executor: T09GrootExecutor,
        inspection_poses: InspectionPoseRegistry,
        endpoint_settle_limits: EndpointSettleLimits,
        watchdog_limits: GrootWatchdogLimits,
        runtime_authority: GrootControllerAuthority,
        **kwargs,
    ) -> None:
        if not isinstance(inspection_poses, InspectionPoseRegistry):
            raise TypeError("CELL inspection poses must use InspectionPoseRegistry")
        if not isinstance(runtime_authority, GrootControllerAuthority):
            raise TypeError("CELL controller requires a typed GrootControllerAuthority")
        if not isinstance(endpoint_settle_limits, EndpointSettleLimits):
            raise TypeError("CELL controller requires sourced EndpointSettleLimits")
        if not isinstance(watchdog_limits, GrootWatchdogLimits):
            raise TypeError("CELL controller requires sourced GrootWatchdogLimits")
        self._groot_executor = groot_executor
        self._inspection_poses = inspection_poses
        self._endpoint_settle_limits = endpoint_settle_limits
        self._watchdog_limits = watchdog_limits
        self._runtime_authority = runtime_authority
        super().__init__(**kwargs)

    def reset(self) -> None:
        super().reset()
        self._groot_executor.reset()
        self._motion_queue = GrootMotionQueue(
            self._groot_executor,
            self._execution_tolerances,
            self._endpoint_settle_limits,
            perception=self._perception,
            perception_tolerances=self._perception_tolerances,
            watchdog_limits=self._watchdog_limits,
        )

    def close(self) -> None:
        """Release the sole CELL motion transport."""
        self._groot_executor.close()

    @property
    def groot_executor(self) -> T09GrootExecutor:
        """Expose immutable motion-authority identity to the CELL factory gate."""
        return self._groot_executor

    @property
    def runtime_authority(self) -> GrootControllerAuthority:
        """Identify real-cell versus reviewed software-proof dependencies."""
        return self._runtime_authority

    def _hold_last_safe(self, reason: str) -> Action:
        """Refuse CELL output when the current TCP is not measured."""
        self._hold(reason)
        raise CellTrialHeld(
            f"CELL refuses action without a measured current TCP: {reason}"
        )

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
        del plan, object_id
        try:
            intent = self._intent_from_state(current, observed_at)
            if intent.instance_id != instance_id or intent.base_instance_id != base_instance_id:
                raise ExecutionError("CELL intent identity does not match the planned physical instances")
            self._motion_queue.queue(
                intent,
                observation=observation,
                current=current,
                observed_at_s=observed_at,
                base_pose=oriented_pose_values(base_pose),
            )
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

    def _confirm_previous_endpoint(
        self,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> Action | None:
        try:
            return self._motion_queue.confirm_previous_endpoint(
                observation,
                current,
                observed_at,
            )
        except GrootGuardTrip as error:
            self._hold(f"GR00T watchdog trip: {error}")
            return current.as_action()
        except ExecutionError as error:
            self._hold(f"GR00T watchdog refused: {error}")
            return current.as_action()

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
        del current, observed_at
        pending = self._pending_pre_pick
        if pending is None:
            raise RuntimeError("inspection intent has no pending part")
        coarse = self._coarse_xy_by_instance[pending.part.instance_id]
        slot = self._inspection_poses.resolve(coarse)
        return FixedInspectionMove(
            instance_id=pending.part.instance_id,
            base_instance_id=pending.base.instance_id,
            slot_id=slot.slot_id,
            endpoint_tcp=slot.endpoint_tcp,
            safe_z_mm=slot.safe_z_mm,
            source=slot.source,
        )

    def _release_intent(self, prepared: PreparedPlanStep) -> MotionIntent:
        release = self._command_tcp(prepared.target, prepared)
        endpoint = (*release[:2], release[2] + self._motion_config.retreat_mm, *release[3:])
        return self._intent(
            prepared, SemanticSkill.RELEASE_RETREAT, endpoint, 0.0, "nominal",
            goal_pose=self._base_goal(prepared),
            release_tcp=release, release_authorized=True,
        )

    def _pull_out_intent(self, current: Waypoint) -> MotionIntent:
        del current
        pending = self._pending_insertion_lift
        if pending is None:
            raise RuntimeError("pull-out intent has no pending insertion")
        prepared = pending.prepared
        target = self._pull_out_target(prepared)
        release = self._command_tcp(target.pose, prepared)
        endpoint = (*release[:2], target.clear_z_mm, *release[3:])
        return self._intent(
            prepared, SemanticSkill.PULL_OUT, endpoint, 0.0, "fix_above",
            goal_pose=(target.pose.tcp.x, target.pose.tcp.y, target.pose.tcp.r),
            grasp=prepared.target,
            release_tcp=release,
            release_authorized=True,
            pose_source_suffix=f"pull-out target: {target.source}",
        )

    def _defect_intent(self, prepared: PreparedPlanStep) -> MotionIntent:
        endpoint = self._command_tcp(prepared.target, prepared)
        return self._intent(
            prepared, SemanticSkill.DEFECT_DROP, endpoint, 0.0, "defect",
            goal_pose=(prepared.target.tcp.x, prepared.target.tcp.y, 0.0),
            release_tcp=endpoint, release_authorized=True,
        )

    def _pick_insert_intent(self, prepared: PreparedPlanStep) -> MotionIntent:
        endpoint = self._command_tcp(prepared.target, prepared)
        part_origin, _, _, _ = self._grasp_fields(prepared.target, prepared)
        radius_mm = contact_envelope_radius_mm(
            (endpoint[0], endpoint[1]),
            (
                (prepared.part.object_id, part_origin[:2], part_origin[2]),
                (
                    prepared.base.object_id,
                    (prepared.base_pose.tcp.x, prepared.base_pose.tcp.y),
                    prepared.base_pose.tcp.r,
                ),
            ),
        )
        contact = ContactEnvelope(
            endpoint[0], endpoint[1], endpoint[2],
            radius_mm,
            f"{prepared.part.object_id} on {prepared.base.object_id}",
            f"CAD part/base footprints from {contact_geometry_source()}",
        )
        return self._intent(
            prepared, SemanticSkill.PICK_INSERT_HOLD, endpoint, 1.0, "nominal",
            goal_pose=self._base_goal(prepared), contact=contact,
        )

    def _intent(
        self,
        prepared: PreparedPlanStep,
        skill: SemanticSkill,
        endpoint: tuple[float, float, float, float, float, float],
        final_gripper: float,
        case: str,
        *,
        goal_pose: tuple[float, float, float],
        grasp: OrientedPartPose | None = None,
        contact: ContactEnvelope | None = None,
        release_tcp: tuple[float, float, float, float, float, float] | None = None,
        release_authorized: bool = False,
        pose_source_suffix: str | None = None,
    ) -> MotionIntent:
        active_grasp = grasp or prepared.grasp
        object_pose, origin_z, grasp_tcp, wrist_roll = self._grasp_fields(active_grasp, prepared)
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
            goal_pose=goal_pose,
            grasp_tcp=grasp_tcp,
            release_tcp=release_tcp,
            endpoint_tcp=endpoint,
            final_gripper=final_gripper,
            pose_source=self._intent_pose_source(active_grasp, pose_source_suffix),
            grasp_source_sha256=TRUSTED_GRASP_SOURCE.sha256,
            contact=contact,
            release_authorized=release_authorized,
        )

    @staticmethod
    def _intent_pose_source(grasp: OrientedPartPose, suffix: str | None) -> str:
        source = f"{grasp.tcp.source} + T29 preferred grasp"
        return source if suffix is None else f"{source}; {suffix}"

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
        wrist_roll = (yaw + preferred.turn_deg) % 360.0
        command = (grasp.tcp.x, grasp.tcp.y, grasp.tcp.z, -180.0, 0.0, wrist_roll)
        return (origin_x, origin_y, yaw), origin_z, command, wrist_roll

    @staticmethod
    def _base_goal(prepared: PreparedPlanStep) -> tuple[float, float, float]:
        base = prepared.base_pose.tcp
        return (base.x, base.y, base.r)

    def _pull_out_target(self, prepared: PreparedPlanStep) -> PullOutTarget:
        provider = getattr(self._recovery_targets, "pull_out_target_for", None)
        if not callable(provider):
            raise MotionIntentError("pull-out requires a calibrated empty test-bed target")
        target = provider(prepared.record.product, prepared.part)
        if not isinstance(target, PullOutTarget):
            raise MotionIntentError("pull-out target must include a calibrated pose and clear Z")
        return target

    @staticmethod
    def _command_tcp(pose: OrientedPartPose, prepared: PreparedPlanStep) -> tuple[float, ...]:
        return (
            pose.tcp.x,
            pose.tcp.y,
            pose.tcp.z,
            -180.0,
            0.0,
            (pose.tcp.r + prepared.recipe_step.preferred_grasp.turn_deg) % 360.0,
        )

    def _prepare_after_insertion_lift(
        self,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> bool:
        if current.gripper == 1.0:
            return super()._prepare_after_insertion_lift(observation, current, observed_at)
        self._motion_queue.finish()
        self._pending_insertion_lift = None
        self._part_held = False
        self._held_instance_id = None
        self._release_dispatched = True
        self._hold("GR00T pull-out set-down requires fresh operator reconciliation before retry")
        return False

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
