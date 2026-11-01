"""Post-placement verification and inventory commit flow.
Authored by Shritej Gunda.
"""

from __future__ import annotations

from enum import Enum

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
from assembly.state_machine import PackMLState
from assembly.verification import (
    VerificationDecision,
    VerificationOutcome,
    VerificationReason,
    verify_post_place,
)
from robot.adapter import Observation


class PlacementOperatorDecision(str, Enum):
    """Independent operator authority after an advisory placement hold."""

    ACCEPT = "accept"
    REINSERT = "reinsert"


class PlacementOperatorDecisionError(ValueError):
    """Raised when an operator decision is not bound to a held placement."""


class PlacementControllerMixin:
    """Keep a placed part held until two independent checks accept it."""

    def resolve_post_place(self, decision: PlacementOperatorDecision) -> None:
        """Bind a typed operator choice; the next fresh observation executes it."""
        if not isinstance(decision, PlacementOperatorDecision):
            raise PlacementOperatorDecisionError("post-place decision must be a PlacementOperatorDecision")
        if self.state_machine.state is not PackMLState.HELD:
            raise PlacementOperatorDecisionError("post-place decision requires Held state")
        if self._prepared is None or not self._awaiting_post or not self._part_held:
            raise PlacementOperatorDecisionError("post-place decision requires the current part held for verification")
        if self._pending_post_place_operator_decision is not None:
            raise PlacementOperatorDecisionError("a post-place operator decision is already pending")
        self._pending_post_place_operator_decision = decision
        self._resume_acknowledged = True

    def _operator_post_place_resume_blocker(self) -> str | None:
        conflicts = (
            (self._pending_install is not None, "verified release"),
            (self._pending_insertion_lift is not None, "insertion recovery lift"),
            (self._pending_reject is not None, "reject transfer"),
            (self._pending_drop_release is not None, "drop release"),
            (self._pending_pre_pick is not None, "inspection stop"),
            (self._release_dispatched, "uncommitted release"),
        )
        blocker = next((name for active, name in conflicts if active), None)
        if blocker is not None:
            return blocker
        if self._prepared is None or not self._awaiting_post:
            return "post-place verification"
        if not self._part_held:
            return "post-place part is not held"
        return None

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
        self._record_verification_advisories(prepared.part, decision)
        decision = self._apply_post_place_operator_decision(prepared, evidence, decision)
        if decision.outcome is VerificationOutcome.ACCEPT:
            self._awaiting_post = False
            return self._queue_verified_release(prepared, observation, current, observed_at)
        if decision.outcome is VerificationOutcome.HOLD_ANDON:
            self._awaiting_post = True
            if decision.reason is VerificationReason.PATCHCORE_WARNING:
                self._hold(
                    f"PatchCore advisory warning for {prepared.record.object_id}; operator decision is required"
                )
            else:
                self._hold(f"post-place checks disagree for {prepared.record.object_id}")
            return True
        self._awaiting_post = False
        return self._recover(prepared, decision, observation, current, observed_at)

    def _apply_post_place_operator_decision(
        self,
        prepared: PreparedPlanStep,
        evidence: PlacementEvidence,
        decision: VerificationDecision,
    ) -> VerificationDecision:
        operator = self._pending_post_place_operator_decision
        if operator is None:
            return decision
        self._pending_post_place_operator_decision = None
        if not self._operator_can_resolve_advisory(prepared, evidence, decision):
            return decision
        accepting = operator is PlacementOperatorDecision.ACCEPT
        outcome = VerificationOutcome.ACCEPT if accepting else VerificationOutcome.RECOVER
        reason = VerificationReason.VERIFIED if accepting else VerificationReason.NOT_SEATED
        self._events.append(
            ControllerEvent("operator", prepared.record.product, prepared.record.object_id, f"post-place {operator.value}")
        )
        return VerificationDecision(outcome, reason, prepared.part.instance_id, prepared.part.object_id, decision.advisories)

    @staticmethod
    def _operator_can_resolve_advisory(
        prepared: PreparedPlanStep,
        evidence: PlacementEvidence,
        decision: VerificationDecision,
    ) -> bool:
        if decision.outcome is not VerificationOutcome.HOLD_ANDON:
            return False
        expected_instance = prepared.part.instance_id
        expected_object = prepared.part.object_id
        return (
            evidence.inspection.get("verdict") == "uncertain"
            and evidence.inspection.get("action") == "ask_operator"
            and evidence.detection.get("instance_id") == expected_instance
            and evidence.inspection.get("instance_id") == expected_instance
            and evidence.detection.get("object_id") == expected_object
            and evidence.inspection.get("object_id") == expected_object
        )

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
