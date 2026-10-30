"""Recovery flow mixed into the cell controller to keep orchestration focused."""

from __future__ import annotations

from dataclasses import dataclass, replace

from assembly.controller_ports import (
    ControllerEvent,
    EvidencePair,
    call_port,
)
from assembly.motion import (
    MotionChunk,
    MotionError,
    MotionPhase,
    Waypoint,
    compile_inspection_stop,
    compile_reinsert,
)
from assembly.planner import PlanCandidate, PlannerError, PreparedPlanStep
from assembly.verification import (
    VerificationDecision,
    VerificationOutcome,
    VerificationReason,
    verify_pre_pick,
)
from robot.adapter import Observation

# Source: docs/ROBOT_BRAIN.md@2a751da, section 3, "Won't seat".
_RETRY_LIFT_MM = 3.0


@dataclass(frozen=True, slots=True)
class PendingInsertionLift:
    """Insertion failure waiting for fresh evidence at the lifted endpoint."""

    prepared: PreparedPlanStep
    final_inspection: bool
    failure_reason: VerificationReason


class RecoveryControllerMixin:
    """Execute the sourced one-retry insertion recovery."""

    def _recover(
        self,
        prepared: PreparedPlanStep,
        decision: VerificationDecision,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> bool:
        retries = self._insertion_retries_by_instance.get(prepared.part.instance_id, 0)
        retryable = decision.reason is VerificationReason.NOT_SEATED
        final_inspection = not retryable or retries >= 1
        queued = self._queue_insertion_lift(
            prepared, final_inspection, decision.reason, observation, current, observed_at
        )
        if queued and retryable and not final_inspection:
            self._insertion_retries_by_instance[prepared.part.instance_id] = 1
        return not queued

    def _queue_insertion_lift(
        self, prepared, final_inspection, failure_reason, observation, current, observed_at
    ) -> bool:
        if current.gripper != 1.0:
            self._hold("insertion recovery requires the failed part to remain held")
            return False
        endpoint = Waypoint(
            current.x, current.y, current.z + _RETRY_LIFT_MM,
            current.w, current.p, current.r, current.gripper,
        )
        inspect = compile_inspection_stop(
            endpoint,
            (endpoint.x, endpoint.y),
            inspection=self._inspection_stop,
            motion=self._motion_config,
        )
        plan = (
            MotionChunk(MotionPhase.LIFT, (endpoint,)),
            *self._keep_part_held(inspect),
        )
        self._pending_insertion_lift = PendingInsertionLift(
            prepared, final_inspection, failure_reason
        )
        queued = self._set_guarded_motion(
            plan, observation, current, observed_at,
            prepared.part.instance_id, prepared.part.object_id,
            prepared.base.instance_id, prepared.base_pose,
        )
        if not queued and not self._pending_intrusion_replan:
            self._pending_insertion_lift = None
        return queued

    @staticmethod
    def _keep_part_held(plan: tuple[MotionChunk, ...]) -> tuple[MotionChunk, ...]:
        return tuple(
            MotionChunk(
                chunk.phase,
                tuple(replace(waypoint, gripper=1.0) for waypoint in chunk.waypoints),
            )
            for chunk in plan
        )

    def _prepare_after_insertion_lift(
        self, observation: Observation, current: Waypoint, observed_at: float
    ) -> bool:
        pending = self._pending_insertion_lift
        if pending is None:
            raise RuntimeError("insertion lift has no pending recovery")
        self._motion_queue.finish()
        evidence, decision = self._post_lift_evidence(pending, observation, observed_at)
        self._record_verification_advisories(pending.prepared.part, decision)
        if decision.outcome is VerificationOutcome.HOLD_ANDON:
            self._hold(self._inspection_hold_reason(pending.prepared.part.object_id, decision.reason))
            return False
        if pending.final_inspection and decision.outcome is VerificationOutcome.ACCEPT:
            self._hold(self._final_inspection_dependency(pending))
            return False
        grasp = self._foundationpose_grasp(pending.prepared.part, evidence, observed_at)
        candidate = self._fresh_retry_candidate(pending.prepared, observed_at)
        if decision.outcome is VerificationOutcome.REJECT_TO_SPARE:
            return self._reject_failed_held(candidate, grasp, decision, observation, current, observed_at)
        return self._queue_fresh_reinsert(candidate, grasp, observation, current, observed_at)

    def _post_lift_evidence(self, pending, observation, observed_at):
        part = pending.prepared.part
        evidence = call_port("post-lift perception", self._perception.pre_pick, observation, part)
        self._require_fresh_evidence(evidence, EvidencePair, observed_at, "post-lift evidence")
        decision = verify_pre_pick(
            part.object_id,
            evidence.detection,
            evidence.inspection,
            expected_instance_id=part.instance_id,
            thresholds=self._verification_thresholds,
        )
        return evidence, decision

    def _fresh_retry_candidate(self, prepared, observed_at) -> PlanCandidate:
        candidate = self._planner.next_candidate(
            self.inventory, observed_at_s=observed_at, attempt=2
        )
        if candidate is None or candidate.part.instance_id != prepared.part.instance_id:
            raise PlannerError("fresh retry plan does not own the held physical part")
        return candidate

    def _queue_fresh_reinsert(self, candidate, grasp, observation, current, observed_at) -> bool:
        prepared = self._planner.finalize_candidate(candidate, grasp)
        try:
            plan = compile_reinsert(
                current,
                prepared.target,
                gripper_roll_deg=prepared.record.gripper_roll_deg,
                config=self._motion_config,
            )
        except MotionError as error:
            self._hold(str(error))
            return False
        self._pending_insertion_lift = None
        self._prepared = prepared
        self._awaiting_post = True
        queued = self._queue_retry_plan(plan, prepared, observation, current, observed_at)
        if queued:
            self._records.append(prepared.record)
            self._events.append(
                ControllerEvent("retry", prepared.record.product, prepared.record.object_id, "1")
            )
        return queued

    def _queue_retry_plan(self, plan, prepared, observation, current, observed_at) -> bool:
        return self._set_guarded_motion(
            plan, observation, current, observed_at,
            prepared.part.instance_id, prepared.part.object_id,
            prepared.base.instance_id, prepared.base_pose,
        )

    def _reject_failed_held(self, candidate, grasp, decision, observation, current, observed_at) -> bool:
        target = self._recovery_targets.reject_target_for(candidate.part)
        prepared = self._planner.finalize_reject_candidate(candidate, grasp, target)
        self._pending_insertion_lift = None
        condition = self._rejection_condition(decision.reason)
        return self._queue_held_rejected(
            prepared, condition, observation, current, observed_at
        )

    @staticmethod
    def _final_inspection_dependency(pending: PendingInsertionLift) -> str:
        object_id = pending.prepared.part.object_id
        if pending.failure_reason is not VerificationReason.NOT_SEATED:
            return (
                f"post-place {pending.failure_reason.value} for {object_id} conflicts with fresh "
                "post-lift part inspection; calibrated alternate-view evidence is required"
            )
        return (
            f"second insertion failure for {object_id}; the part passed fresh post-lift inspection, "
            "but a typed calibrated hole-clearance verdict is required before further motion"
        )
