"""Post-placement verification and inventory commit flow."""

from __future__ import annotations

from assembly.controller_ports import (
    ControllerEvent,
    PlacementEvidence,
    call_port,
    oriented_pose_values,
    pose_agrees,
)
from assembly.inventory import InventoryShortageError, InventoryZone, ObjectCondition
from assembly.motion import Waypoint, compile_release_retreat
from assembly.planner import PreparedPlanStep
from assembly.verification import (
    VerificationDecision,
    VerificationOutcome,
    VerificationReason,
    verify_post_place,
)
from robot.adapter import Observation


class PlacementControllerMixin:
    """Keep a placed part held until two independent checks accept it."""

    def _placement_decision(self, prepared: PreparedPlanStep, evidence: PlacementEvidence) -> VerificationDecision:
        if prepared.verification_target is None:
            raise RuntimeError("post-place verification has no canonical object target")
        expected = oriented_pose_values(prepared.verification_target)
        agrees = evidence.pose_agrees and pose_agrees(
            evidence.detection.get("pose"), expected, self._perception_tolerances
        )
        return verify_post_place(
            prepared.record.object_id,
            evidence.detection,
            evidence.inspection,
            expected_instance_id=prepared.part.instance_id,
            thresholds=self._verification_thresholds,
            seated=evidence.seated,
            pose_agrees=agrees,
        )

    def _verify_placement(self, observation: Observation, current: Waypoint, observed_at: float) -> bool:
        prepared = self._prepared
        if prepared is None:
            raise RuntimeError("post-place verification has no active plan step")
        evidence = call_port("post-place perception", self._perception.post_place, observation, prepared.part)
        self._require_fresh_evidence(evidence, PlacementEvidence, observed_at, "post-place evidence")
        decision = self._placement_decision(prepared, evidence)
        self._awaiting_post = False
        if decision.outcome is VerificationOutcome.ACCEPT:
            return self._queue_verified_release(prepared, observation, current, observed_at)
        if decision.outcome is VerificationOutcome.HOLD_ANDON:
            self._hold(f"post-place checks disagree for {prepared.record.object_id}")
            return True
        return self._recover(prepared, decision, observation, current, observed_at)

    def _queue_verified_release(
        self,
        prepared: PreparedPlanStep,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> bool:
        plan = compile_release_retreat(
            current,
            prepared.target,
            gripper_roll_deg=prepared.record.gripper_roll_deg,
            config=self._motion_config,
            max_position_error_mm=self._execution_tolerances.max_position_error_mm,
            max_orientation_error_deg=self._execution_tolerances.max_orientation_error_deg,
        )
        self._pending_install = prepared
        self._prepared = None
        self._set_guarded_motion(
            plan,
            observation,
            current,
            observed_at,
            prepared.part.instance_id,
            prepared.part.object_id,
            prepared.base.instance_id,
            prepared.base_pose,
        )
        return False

    def _replace_rejected(self, prepared: PreparedPlanStep, decision: VerificationDecision) -> None:
        condition = (
            ObjectCondition.FALLEN
            if decision.reason is VerificationReason.UNSAFE_ORIENTATION
            else ObjectCondition.DEFECTIVE
        )
        try:
            spare = self.inventory.replace_with_spare(prepared.part.instance_id, condition=condition)
        except InventoryShortageError as error:
            self._pending_spare_instance = prepared.part.instance_id
            self._suspend(str(error))
            return
        self._events.append(ControllerEvent("spare", prepared.record.product, spare.object_id, spare.instance_id))
        self._prepared = None

    def _commit_install(self) -> None:
        prepared = self._pending_install
        if prepared is None:
            raise RuntimeError("install commit has no verified plan step")
        base = self.inventory.active_reservation.base
        if base.zone is not InventoryZone.RESERVED or base.condition is not ObjectCondition.VERIFIED:
            self._hold("active product base is no longer verified at placement commit")
            return
        if self.inventory.item(prepared.part.instance_id).condition is not ObjectCondition.VERIFIED:
            self._hold(f"placed instance {prepared.part.instance_id!r} is no longer verified")
            return
        self.inventory.install(prepared.part.instance_id)
        self._insertion_retries_by_instance.pop(prepared.part.instance_id, None)
        self._events.append(
            ControllerEvent("installed", prepared.record.product, prepared.record.object_id, "verified")
        )
        self._pending_install = None
        self._motion_queue.finish()
        self._release_dispatched = False
        self._part_held = False
