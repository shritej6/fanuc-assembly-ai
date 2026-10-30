"""Inspection-gated pick, reject, and insert preparation for T09."""

from __future__ import annotations

from dataclasses import dataclass

from assembly.catalog import CatalogError
from assembly.controller_ports import (
    ControllerEvent,
    ControllerPortError,
    EvidencePair,
    RecoveryTransferEvidence,
    call_port,
    require_fresh,
)
from assembly.inventory import (
    InventoryItem,
    InventoryShortageError,
    InventoryZone,
    ObjectCondition,
)
from assembly.motion import (
    MotionError,
    OrientedPartPose,
    Waypoint,
    compile_inspection_stop,
    compile_pick_to_insert,
    compile_pick_to_reject,
    compile_transfer_held,
)
from assembly.planner import PlanCandidate, PlannerError, PreparedPlanStep
from assembly.verification import (
    VerificationOutcome,
    VerificationReason,
    verify_pre_pick,
    verify_recovery_transfer,
)
from robot.adapter import Observation


@dataclass(frozen=True, slots=True)
class PendingReject:
    """A physical reject awaiting independent transfer verification."""

    prepared: PreparedPlanStep
    target: OrientedPartPose
    condition: ObjectCondition


class InspectionControllerMixin:
    """Require a wrist inspection before any grasp is executable."""

    def _prepare_motion(self, observation: Observation, current: Waypoint, observed_at: float) -> bool:
        forced = self._environment_reject_candidate(observed_at)
        if forced is not None:
            candidate, condition = forced
            return self._queue_inspection(candidate, condition, observation, current, observed_at)
        try:
            candidate = self._planner.next_candidate(
                self.inventory,
                observed_at_s=observed_at,
                attempt=1,
            )
        except PlannerError as error:
            self._abort(str(error))
            return False
        if candidate is None:
            return self._verify_finished_product(observation, observed_at)
        return self._queue_inspection(candidate, None, observation, current, observed_at)

    def _queue_inspection(
        self,
        candidate: PlanCandidate,
        forced_condition: ObjectCondition | None,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> bool:
        coarse_xy = self._coarse_xy_by_instance.get(candidate.part.instance_id)
        if coarse_xy is None:
            self._hold(f"coarse perception center unavailable for {candidate.part.instance_id!r}")
            return False
        try:
            plan = compile_inspection_stop(
                current,
                coarse_xy,
                inspection=self._inspection_stop,
                motion=self._motion_config,
            )
        except MotionError as error:
            self._hold(str(error))
            return False
        self._pending_pre_pick = candidate
        self._forced_reject_condition = forced_condition
        queued = self._queue_candidate_inspection(plan, candidate, observation, current, observed_at)
        if not queued:
            self._pending_pre_pick = None
            self._forced_reject_condition = None
        return queued

    def _queue_candidate_inspection(self, plan, candidate, observation, current, observed_at) -> bool:
        return self._set_guarded_motion(
            plan,
            observation,
            current,
            observed_at,
            candidate.part.instance_id,
            candidate.part.object_id,
            candidate.base.instance_id,
            candidate.base_pose,
        )

    def _environment_reject_candidate(
        self,
        observed_at: float,
    ) -> tuple[PlanCandidate, ObjectCondition] | None:
        for zone in (InventoryZone.TABLE, InventoryZone.RESERVED):
            for item in self.inventory.items(zone):
                condition = self._environment_reject_condition(item)
                if condition is not None:
                    return self._candidate_environment_reject(item, observed_at), condition
        return None

    def _environment_reject_condition(self, item: InventoryItem) -> ObjectCondition | None:
        if not item.object_id.startswith("M"):
            return None
        if item.condition is ObjectCondition.FALLEN:
            return ObjectCondition.FALLEN
        if item.condition is ObjectCondition.DEFECTIVE:
            return ObjectCondition.DEFECTIVE
        if item.object_id.startswith("M") and int(item.object_id[1]) not in self._products:
            return ObjectCondition.DEFECTIVE
        return None

    def _surface_claimed_conditions(self) -> bool:
        unusable = {ObjectCondition.FALLEN, ObjectCondition.DEFECTIVE}
        for zone in (InventoryZone.INSTALLED, InventoryZone.STAGED):
            for item in self.inventory.items(zone):
                if item.condition in unusable:
                    self._hold(f"claimed {zone.value} part {item.instance_id!r} requires calibrated recovery")
                    return False
        pending = self._pending_pre_pick
        if pending is not None:
            condition = self._environment_reject_condition(self.inventory.item(pending.part.instance_id))
            if condition is not None:
                self._forced_reject_condition = condition
        return True

    def _candidate_environment_reject(self, part: InventoryItem, observed_at: float) -> PlanCandidate:
        owner = int(part.object_id[1])
        product = self._catalog.product(owner)
        recipe_step = next(step for step in product.steps if step.object_id == part.object_id)
        base = self._reference_base()
        base_pose = self._poses.base_for(base, observed_at_s=observed_at)
        target = self._recovery_targets.reject_target_for(part)
        return PlanCandidate(product, recipe_step, base, part, base_pose, target, None, 1, observed_at)

    def _reference_base(self) -> InventoryItem:
        if self.inventory.active_product is not None:
            return self.inventory.active_reservation.base
        current = self.state_machine.current_product
        if current is None:
            raise PlannerError("reject inspection requires an active product request")
        base_type = self._catalog.product(current).base_type
        for item in self.inventory.items(InventoryZone.TABLE):
            if self._is_compatible_reference_base(item, base_type):
                return item
        raise PlannerError("reject inspection requires a verified compatible base reference")

    def _is_compatible_reference_base(self, item: InventoryItem, base_type: str) -> bool:
        if item.condition is not ObjectCondition.VERIFIED:
            return False
        try:
            return self._catalog.base_type_for_object(item.object_id) == base_type
        except CatalogError:
            return False

    def _prepare_after_inspection(
        self,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> bool:
        candidate = self._pending_pre_pick
        if candidate is None:
            raise RuntimeError("inspection stop has no pending pre-pick")
        self._motion_queue.finish()
        self._pending_pre_pick = None
        forced_condition = self._forced_reject_condition
        self._forced_reject_condition = None
        return self._handle_pre_pick(candidate, forced_condition, observation, current, observed_at)

    def _handle_pre_pick(
        self,
        candidate: PlanCandidate,
        forced_condition: ObjectCondition | None,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> bool:
        evidence = call_port("pre-pick perception", self._perception.pre_pick, observation, candidate.part)
        self._require_fresh_evidence(evidence, EvidencePair, observed_at, "pre-pick evidence")
        decision = verify_pre_pick(
            candidate.part.object_id,
            evidence.detection,
            evidence.inspection,
            expected_instance_id=candidate.part.instance_id,
            thresholds=self._verification_thresholds,
        )
        if decision.outcome is VerificationOutcome.HOLD_ANDON:
            self._hold(self._alternate_view_dependency(candidate.part.object_id))
            return False
        grasp = self._foundationpose_grasp(candidate.part, evidence, observed_at)
        if decision.outcome is VerificationOutcome.REJECT_TO_SPARE or forced_condition is not None:
            target = self._recovery_targets.reject_target_for(candidate.part)
            prepared = self._planner.finalize_reject_candidate(candidate, grasp, target)
            return self._handle_reject_decision(prepared, forced_condition, decision, observation, current, observed_at)
        prepared = self._planner.finalize_candidate(candidate, grasp)
        return self._queue_prepared(prepared, observation, current, observed_at)

    @staticmethod
    def _alternate_view_dependency(object_id: str) -> str:
        return (
            f"pre-pick checks disagree for {object_id}; alternate close-up requires "
            "published calibrated wrist-view targets and a safety-approved motion plan"
        )

    def _handle_reject_decision(
        self,
        prepared,
        forced_condition,
        decision,
        observation,
        current,
        observed_at,
    ) -> bool:
        if decision.instance_id != prepared.part.instance_id:
            self._hold("pre-pick evidence does not match the reserved physical instance")
            return False
        condition = forced_condition or self._rejection_condition(decision.reason)
        return self._queue_rejected(prepared, condition, observation, current, observed_at)

    def _foundationpose_grasp(
        self,
        part: InventoryItem,
        evidence: EvidencePair,
        observed_at: float,
    ) -> OrientedPartPose:
        try:
            grasp = self._poses.grasp_from_foundationpose(
                part,
                evidence.detection,
                observed_at_s=observed_at,
            )
        except Exception as error:
            raise PlannerError(f"FoundationPose grasp is unavailable: {error}") from error
        return grasp

    @staticmethod
    def _rejection_condition(reason: VerificationReason) -> ObjectCondition:
        if reason is VerificationReason.UNSAFE_ORIENTATION:
            return ObjectCondition.FALLEN
        return ObjectCondition.DEFECTIVE

    def _queue_rejected(
        self,
        prepared: PreparedPlanStep,
        condition: ObjectCondition,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> bool:
        target = prepared.target
        plan = compile_pick_to_reject(
            current,
            prepared.grasp,
            target,
            gripper_roll_deg=prepared.record.gripper_roll_deg,
            config=self._motion_config,
        )
        self._pending_reject = PendingReject(prepared, target, condition)
        queued = self._queue_reject_plan(plan, prepared, observation, current, observed_at)
        if not queued:
            self._pending_reject = None
        return queued

    def _queue_held_rejected(
        self,
        prepared: PreparedPlanStep,
        condition: ObjectCondition,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> bool:
        try:
            plan = compile_transfer_held(
                current,
                prepared.target,
                gripper_roll_deg=prepared.record.gripper_roll_deg,
                config=self._motion_config,
                max_position_error_mm=self._execution_tolerances.max_position_error_mm,
                max_orientation_error_deg=self._execution_tolerances.max_orientation_error_deg,
            )
        except MotionError as error:
            self._hold(str(error))
            return False
        self._pending_reject = PendingReject(prepared, prepared.target, condition)
        queued = self._queue_reject_plan(plan, prepared, observation, current, observed_at)
        if not queued:
            self._pending_reject = None
        return queued

    def _queue_reject_plan(self, plan, prepared, observation, current, observed_at) -> bool:
        return self._set_guarded_motion(
            plan,
            observation,
            current,
            observed_at,
            prepared.part.instance_id,
            prepared.part.object_id,
            prepared.base.instance_id,
            prepared.base_pose,
        )

    def _verify_reject_transfer(self, observation: Observation, current: Waypoint, observed_at: float) -> None:
        pending = self._pending_reject
        if pending is None:
            raise RuntimeError("reject transfer has no pending item")
        evidence = self._reject_transfer_evidence(pending, observation, observed_at)
        decision = verify_recovery_transfer(
            pending.prepared.part.object_id,
            evidence.detection,
            evidence.inspection,
            expected_instance_id=pending.prepared.part.instance_id,
            thresholds=self._verification_thresholds,
            pose_agrees=(evidence.target_zone is InventoryZone.REJECT and evidence.pose_agrees),
            allow_rejected=True,
        )
        if decision.outcome is not VerificationOutcome.ACCEPT or current.gripper != 0.0:
            self._hold(f"reject transfer was not independently verified for {pending.prepared.part.instance_id!r}")
            return
        self._commit_rejected(pending)

    def _reject_transfer_evidence(self, pending, observation, observed_at):
        part = pending.prepared.part
        evidence = call_port(
            "reject transfer perception",
            self._perception.post_transfer,
            observation,
            part,
            InventoryZone.REJECT,
        )
        self._require_fresh_evidence(evidence, RecoveryTransferEvidence, observed_at, "reject transfer evidence")
        return evidence

    def _commit_rejected(self, pending: PendingReject) -> None:
        part = pending.prepared.part
        was_reserved = part.zone is InventoryZone.RESERVED
        self._motion_queue.finish()
        self.inventory.reject(part.instance_id, condition=pending.condition)
        self._insertion_retries_by_instance.pop(part.instance_id, None)
        self._pending_reject = None
        self._held_instance_id = None
        self._events.append(
            ControllerEvent("rejected", pending.prepared.record.product, part.object_id, part.instance_id)
        )
        if was_reserved:
            self._replace_rejected_reserved_part(pending)

    def _replace_rejected_reserved_part(self, pending: PendingReject) -> None:
        part = pending.prepared.part
        try:
            replacement = self.inventory.replace_with_spare(part.instance_id)
        except InventoryShortageError as error:
            self._pending_spare_instance = part.instance_id
            self._suspend(str(error))
            return
        kind = "spare" if replacement.source_zone is InventoryZone.SPARE else "replacement"
        self._events.append(ControllerEvent(kind, pending.prepared.record.product, replacement.object_id, replacement.instance_id))

    def _queue_prepared(
        self,
        prepared: PreparedPlanStep,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> bool:
        try:
            plan = compile_pick_to_insert(
                current,
                prepared.grasp,
                prepared.target,
                gripper_roll_deg=prepared.record.gripper_roll_deg,
                config=self._motion_config,
            )
        except MotionError as error:
            self._abort(str(error))
            return False
        self._prepared = prepared
        self._awaiting_post = True
        queued = self._queue_insert_plan(plan, prepared, observation, current, observed_at)
        if queued:
            self._records.append(prepared.record)
            self._events.append(
                ControllerEvent("motion", prepared.record.product, prepared.record.object_id, "guarded")
            )
        return queued

    def _queue_insert_plan(self, plan, prepared, observation, current, observed_at) -> bool:
        return self._set_guarded_motion(
            plan,
            observation,
            current,
            observed_at,
            prepared.part.instance_id,
            prepared.part.object_id,
            prepared.base.instance_id,
            prepared.base_pose,
        )

    @staticmethod
    def _require_fresh_evidence(evidence: object, expected_type: type, observed_at: float, label: str) -> None:
        if type(evidence) is not expected_type:
            raise ControllerPortError(f"{label} must be exactly {expected_type.__name__}")
        require_fresh(evidence.observed_at_s, observed_at, label)
