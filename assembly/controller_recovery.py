"""Recovery flow mixed into the cell controller to keep orchestration focused."""

from __future__ import annotations

from assembly.controller_ports import (
    ControllerEvent,
    EvidencePair,
    PlacementEvidence,
    RecoveryTransferEvidence,
    call_port,
    oriented_pose_values,
    pose_agrees,
)
from assembly.inventory import InventoryError, InventoryZone, ObjectCondition
from assembly.motion import MotionError, Waypoint, compile_release_retreat
from assembly.planner import PlannerError, PreparedPlanStep
from assembly.recovery import RecoveryError, RecoveryStageKind, RecoveryStagePlan, compile_retry_recovery
from assembly.verification import (
    RecoveryAction,
    RecoveryDecision,
    VerificationOutcome,
    decide_recovery,
    verify_post_place,
    verify_pre_pick,
    verify_recovery_transfer,
)
from robot.adapter import Observation


class RecoveryControllerMixin:
    """Execute bounded retries and staged spare rebuilds."""

    def _installed_prefix(self, product: int) -> tuple[str, ...]:
        recipe = self._catalog.product(product)
        installed = {item.object_id for item in self.inventory.items(InventoryZone.INSTALLED)}
        return tuple(step.object_id for step in recipe.steps if step.object_id in installed)

    def _recover(
        self,
        prepared: PreparedPlanStep,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> bool:
        spare = any(
            item.object_id == prepared.record.object_id
            and item.zone is InventoryZone.SPARE
            and item.condition is ObjectCondition.VERIFIED
            for item in self.inventory.items()
        )
        recovery = decide_recovery(
            prepared.record.object_id,
            self._failed_attempts,
            spare,
            installed_object_ids=self._installed_prefix(prepared.record.product),
        )
        if recovery.action is RecoveryAction.RETRY_INSERTION:
            return self._queue_retry(prepared, recovery, observation, current, observed_at)
        if recovery.action is RecoveryAction.DISMANTLE_REBUILD:
            try:
                self._recovery.begin(self.inventory, prepared, recovery)
            except RecoveryError as error:
                self._hold(str(error))
                return True
            self._prepared = None
            self._events.append(
                ControllerEvent("rebuild", prepared.record.product, prepared.record.object_id, "started")
            )
            return False
        self._hold("verified spare unavailable after two retries")
        return True

    def _queue_retry(
        self,
        prepared: PreparedPlanStep,
        recovery: RecoveryDecision,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> bool:
        try:
            retry = compile_retry_recovery(prepared, recovery, current, self._motion_config)
        except (MotionError, RecoveryError) as error:
            self._hold(str(error))
            return True
        self._failed_attempts = retry.retry_number
        self._prepared = retry.prepared
        self._awaiting_post = True
        self._records.append(retry.prepared.record)
        detail = str(retry.retry_number)
        self._events.append(ControllerEvent("retry", prepared.record.product, prepared.record.object_id, detail))
        self._set_guarded_motion(
            retry.chunks,
            observation,
            current,
            observed_at,
            prepared.part.instance_id,
            prepared.part.object_id,
            prepared.base.instance_id,
            prepared.base_pose,
        )
        return False

    def _prepare_recovery_stage(
        self,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> bool:
        try:
            planned = self._recovery.prepare_stage(
                inventory=self.inventory,
                catalog=self._catalog,
                poses=self._poses,
                current=current,
                config=self._motion_config,
                observed_at_s=observed_at,
            )
        except (MotionError, PlannerError, RecoveryError) as error:
            self._hold(str(error))
            return False
        if planned is None:
            self._finish_rebuild()
            return False
        if not self._recovery_pick_passes(planned, observation, observed_at):
            return False
        self._pending_recovery_stage = planned
        return self._set_guarded_motion(
            planned.chunks,
            observation,
            current,
            observed_at,
            planned.part.instance_id,
            planned.part.object_id,
            self.inventory.active_reservation.base.instance_id,
            planned.base_pose,
        )

    def _recovery_pick_passes(
        self,
        planned: RecoveryStagePlan,
        observation: Observation,
        observed_at: float,
    ) -> bool:
        if planned.stage.kind is RecoveryStageKind.REJECT_FAILED:
            return True
        part = planned.part
        evidence = call_port("recovery pre-pick perception", self._perception.pre_pick, observation, part)
        self._require_fresh_evidence(evidence, EvidencePair, observed_at, "recovery pre-pick evidence")
        decision = verify_pre_pick(
            part.object_id,
            evidence.detection,
            evidence.inspection,
            expected_instance_id=part.instance_id,
            thresholds=self._verification_thresholds,
        )
        grasp = planned.grasp
        agrees = grasp is not None and pose_agrees(
            evidence.detection.get("pose"),
            oriented_pose_values(grasp),
            self._perception_tolerances,
        )
        if decision.outcome is VerificationOutcome.ACCEPT and agrees:
            return True
        self._hold(f"recovery pre-pick rejected {part.instance_id!r}")
        return False

    def _verify_recovery_stage(self, observation: Observation, current: Waypoint, observed_at: float) -> bool:
        planned = self._pending_recovery_stage
        if planned is None:
            raise RuntimeError("recovery verification has no pending stage")
        if not self._recovery_stage_passes(planned, observation, current, observed_at):
            self._hold(f"recovery stage {planned.stage.kind.value} was not independently verified")
            return False
        if planned.release_after_verification:
            return self._queue_recovery_release(planned, observation, current, observed_at)
        self._commit_recovery_stage(planned)
        return True

    def _commit_recovery_stage(self, planned: RecoveryStagePlan) -> None:
        base = self.inventory.active_reservation.base
        if base.zone is not InventoryZone.RESERVED or base.condition is not ObjectCondition.VERIFIED:
            self._hold("active product base is no longer verified at recovery commit")
            return
        if self.inventory.item(planned.part.instance_id).condition is not ObjectCondition.VERIFIED:
            self._hold(f"recovery instance {planned.part.instance_id!r} is no longer verified")
            return
        try:
            spare = self._recovery.commit(self.inventory, planned.stage)
        except (InventoryError, RecoveryError) as error:
            self._hold(str(error))
            return
        self._motion_queue.finish()
        self._release_dispatched = False
        self._part_held = False
        self._pending_recovery_stage = None
        self._pending_recovery_release = None
        detail = planned.stage.instance_id if spare is None else spare.instance_id
        self._events.append(
            ControllerEvent(planned.stage.kind.value, self.state_machine.current_product, planned.stage.object_id, detail)
        )

    def _recovery_stage_passes(
        self,
        planned: RecoveryStagePlan,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> bool:
        if planned.stage.kind is RecoveryStageKind.REBUILD_INSTALLED:
            evidence = call_port("recovery placement perception", self._perception.post_place, observation, planned.part)
            self._require_fresh_evidence(
                evidence, PlacementEvidence, observed_at, "recovery placement evidence"
            )
            return current.gripper == 1.0 and self._recovery_placement_passes(planned, evidence)
        zone = InventoryZone.REJECT if planned.stage.kind is RecoveryStageKind.REJECT_FAILED else InventoryZone.STAGED
        evidence = call_port(
            "recovery transfer perception", self._perception.post_transfer, observation, planned.part, zone
        )
        self._require_fresh_evidence(
            evidence, RecoveryTransferEvidence, observed_at, "recovery transfer evidence"
        )
        expected_gripper = 0.0 if zone is InventoryZone.REJECT else 1.0
        return current.gripper == expected_gripper and self._recovery_transfer_passes(planned, evidence, zone)

    def _recovery_transfer_passes(
        self,
        planned: RecoveryStagePlan,
        evidence: RecoveryTransferEvidence,
        zone: InventoryZone,
    ) -> bool:
        if evidence.target_zone is not zone:
            return False
        part = planned.part
        expected = oriented_pose_values(planned.target)
        target_agrees = evidence.pose_agrees and pose_agrees(
            evidence.detection.get("pose"), expected, self._perception_tolerances
        )
        decision = verify_recovery_transfer(
            part.object_id,
            evidence.detection,
            evidence.inspection,
            expected_instance_id=part.instance_id,
            thresholds=self._verification_thresholds,
            pose_agrees=target_agrees,
            allow_rejected=zone is InventoryZone.REJECT,
        )
        return decision.outcome is VerificationOutcome.ACCEPT

    def _recovery_placement_passes(
        self,
        planned: RecoveryStagePlan,
        evidence: PlacementEvidence,
    ) -> bool:
        part = planned.part
        expected = oriented_pose_values(planned.target)
        target_agrees = evidence.pose_agrees and pose_agrees(
            evidence.detection.get("pose"), expected, self._perception_tolerances
        )
        decision = verify_post_place(
            part.object_id,
            evidence.detection,
            evidence.inspection,
            expected_instance_id=part.instance_id,
            thresholds=self._verification_thresholds,
            seated=evidence.seated,
            pose_agrees=target_agrees,
        )
        return decision.outcome is VerificationOutcome.ACCEPT

    def _queue_recovery_release(
        self,
        planned: RecoveryStagePlan,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> bool:
        try:
            chunks = compile_release_retreat(
                current,
                planned.target,
                gripper_roll_deg=planned.gripper_roll_deg,
                config=self._motion_config,
            )
        except MotionError as error:
            self._hold(str(error))
            return False
        queued = self._set_guarded_motion(
            chunks,
            observation,
            current,
            observed_at,
            planned.part.instance_id,
            planned.part.object_id,
            self.inventory.active_reservation.base.instance_id,
            planned.base_pose,
        )
        if not queued:
            return False
        self._pending_recovery_stage = None
        self._pending_recovery_release = planned
        return True

    def _commit_recovery_release(self) -> None:
        planned = self._pending_recovery_release
        if planned is None:
            raise RuntimeError("recovery release has no pending stage")
        self._commit_recovery_stage(planned)

    def _finish_rebuild(self) -> None:
        self._failed_attempts = 0
        self._events.append(ControllerEvent("rebuild", self.state_machine.current_product, None, "verified complete"))
