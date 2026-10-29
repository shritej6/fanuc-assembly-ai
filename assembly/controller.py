"""PackML orchestration from perception evidence to guarded TCP chunks."""

from __future__ import annotations

from collections.abc import Sequence

from assembly.catalog import Catalog
from assembly.controller_placement import PlacementControllerMixin
from assembly.controller_ports import (
    ControllerEvent,
    ControllerPortError,
    EvidencePair,
    ExecutionTolerances,
    PerceptionPort,
    PerceptionTolerances,
    call_port,
    current_waypoint,
    observation_time,
    oriented_pose_values,
    pose_agrees,
    reconcile_inventory,
    require_fresh,
)
from assembly.controller_recovery import RecoveryControllerMixin
from assembly.execution import ExecutionError, GuardedMotionQueue
from assembly.inventory import (
    Inventory,
    InventoryError,
    InventoryShortageError,
    InventoryZone,
    ObjectCondition,
    ProductReservation,
)
from assembly.motion import (
    MotionChunk,
    MotionConfig,
    MotionError,
    MotionPhase,
    OrientedPartPose,
    Waypoint,
    compile_pick_to_insert,
)
from assembly.planner import CalibratedPoseProvider, PlannerError, PlanStepRecord, PreparedPlanStep, RecipePlanner
from assembly.recovery import RecoveryCoordinator, RecoveryStagePlan, RecoveryTargetProvider
from assembly.safety import SafetySupervisor, SafetyViolation
from assembly.state_machine import PackMLState, PackMLStateMachine, ReservationReleaseError
from assembly.verification import VerificationInputError, VerificationOutcome, VerificationThresholds, verify_pre_pick
from robot.adapter import Action, Observation


class ControllerInputError(ValueError):
    """Observation or perception-port data is unsafe to execute."""


class AssemblyController(PlacementControllerMixin, RecoveryControllerMixin):
    """Execute one reserved product at a time with fresh verification gates."""

    def __init__(
        self,
        *,
        catalog: Catalog,
        products: Sequence[int],
        perception: PerceptionPort,
        poses: CalibratedPoseProvider,
        motion_config: MotionConfig,
        safety: SafetySupervisor,
        verification_thresholds: VerificationThresholds,
        execution_tolerances: ExecutionTolerances,
        perception_tolerances: PerceptionTolerances,
        recovery_targets: RecoveryTargetProvider,
    ) -> None:
        normalized = tuple(products)
        if not normalized or len(set(normalized)) != len(normalized):
            raise ControllerInputError("products must be a non-empty unique sequence")
        for product in normalized:
            catalog.product(product)
        self._catalog = catalog
        self._products = normalized
        self._perception = perception
        self._poses = poses
        self._motion_config = motion_config
        self._safety = safety
        self._verification_thresholds = verification_thresholds
        self._execution_tolerances = execution_tolerances
        self._perception_tolerances = perception_tolerances
        self._recovery_targets = recovery_targets
        self.reset()

    def reset(self) -> None:
        self.inventory = Inventory(self._catalog)
        self.state_machine = PackMLStateMachine(reservation_releaser=self.inventory)
        self._planner = RecipePlanner(self._catalog, self._poses)
        self._product_index = 0
        self._completed: list[int] = []
        self._events: list[ControllerEvent] = []
        self._records: list[PlanStepRecord] = []
        self._motion_queue = GuardedMotionQueue(
            perception=self._perception,
            safety=self._safety,
            tolerances=self._execution_tolerances,
            perception_tolerances=self._perception_tolerances,
        )
        self._prepared: PreparedPlanStep | None = None
        self._pending_install: PreparedPlanStep | None = None
        self._recovery = RecoveryCoordinator(self._recovery_targets)
        self._pending_recovery_stage: RecoveryStagePlan | None = None
        self._pending_recovery_release: RecoveryStagePlan | None = None
        self._awaiting_post = False
        self._failed_attempts = 0
        self._pending_spare_instance: str | None = None
        self._announced_spares: set[str] = set()
        self._finished = False
        self._last_safe_waypoint: Waypoint | None = None
        self._release_dispatched = False
        self._part_held = False

    @property
    def completed_products(self) -> tuple[int, ...]:
        return tuple(self._completed)

    @property
    def events(self) -> tuple[ControllerEvent, ...]:
        return tuple(self._events)

    @property
    def plan_records(self) -> tuple[PlanStepRecord, ...]:
        return tuple(self._records)

    @property
    def finished(self) -> bool:
        return self._finished

    def act(self, observation: Observation) -> Action:
        try:
            current = current_waypoint(observation)
        except ControllerPortError as error:
            return self._hold_last_safe(str(error))
        self._last_safe_waypoint = current
        if current.gripper == 1.0:
            self._part_held = True
        try:
            observed_at = observation_time(observation)
            return self._act_current(observation, current, observed_at)
        except (ControllerPortError, InventoryError, MotionError, VerificationInputError) as error:
            self._hold(str(error))
            return current.as_action()

    def _act_current(self, observation: Observation, current: Waypoint, observed_at: float) -> Action:
        snapshot = call_port("inventory scan", self._perception.scan, observation)
        reconcile_inventory(self.inventory, snapshot, observed_at)
        try:
            self._motion_queue.confirm_previous_endpoint(current)
        except ExecutionError as error:
            self._abort(str(error))
            return current.as_action()
        return self._progress(observation, current, observed_at)

    def _progress(self, observation: Observation, current: Waypoint, observed_at: float) -> Action:
        for _ in range(len(self._products) + 2):
            terminal = {PackMLState.HELD, PackMLState.ABORTED}
            if self._finished or self.state_machine.state in terminal:
                return current.as_action()
            if not self._ensure_product_started():
                return current.as_action()
            if self._motion_queue.active:
                return self._emit_motion_chunk(observation, current, observed_at)
            if self._pending_install is not None:
                self._commit_install()
                continue
            if self._pending_recovery_release is not None:
                self._commit_recovery_release()
                continue
            if self._pending_recovery_stage is not None:
                if not self._verify_recovery_stage(observation, current, observed_at):
                    return current.as_action()
                continue
            if self._recovery.active:
                if self._prepare_recovery_stage(observation, current, observed_at):
                    return self._emit_motion_chunk(observation, current, observed_at)
                continue
            if self._awaiting_post:
                if self._verify_placement(observation, current, observed_at):
                    return current.as_action()
                continue
            if self._prepare_motion(observation, current, observed_at):
                return self._emit_motion_chunk(observation, current, observed_at)
        raise RuntimeError("controller failed to make bounded progress")

    def _hold_last_safe(self, reason: str) -> Action:
        hold = self._motion_queue.pending_endpoint or self._last_safe_waypoint
        if hold is None:
            raise ControllerInputError(f"no trusted TCP is available for safe hold: {reason}")
        self._hold(reason)
        return hold.as_action()

    def _emit_motion_chunk(
        self,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> Action:
        instance_id = self._motion_queue.instance_id
        if instance_id is None:
            raise RuntimeError("queued motion has no physical instance owner")
        base = self.inventory.active_reservation.base
        if base.zone is not InventoryZone.RESERVED or base.condition is not ObjectCondition.VERIFIED:
            self._abort("active product base must remain reserved and verified during motion")
            return current.as_action()
        item = self.inventory.item(instance_id)
        if item.condition is not ObjectCondition.VERIFIED:
            self._abort(f"moving instance {item.instance_id!r} is no longer verified")
            return current.as_action()
        try:
            action = self._motion_queue.emit(
                observation=observation,
                current=current,
                observed_at_s=observed_at,
            )
            if self._motion_queue.last_emitted_phase is MotionPhase.GRIP:
                self._part_held = True
            if self._motion_queue.last_emitted_phase is MotionPhase.RELEASE:
                self._part_held = False
                self._release_dispatched = True
            return action
        except (ControllerPortError, ExecutionError, SafetyViolation) as error:
            self._abort(str(error))
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

    def _ensure_product_started(self) -> bool:
        state = self.state_machine.state
        if state is PackMLState.COMPLETE:
            if not self._advance_product():
                return False
            state = self.state_machine.state
        if state is PackMLState.IDLE:
            product = self._products[self._product_index]
            self.state_machine.start(product)
            self.state_machine.transition(PackMLState.EXECUTE)
        if self.state_machine.state is PackMLState.SUSPENDED:
            return self._retry_reservation()
        try:
            reservation = self._planner.reserve_product(self.inventory, self._products[self._product_index])
        except InventoryShortageError as error:
            self._suspend(str(error))
            return False
        self._record_spare_sources(reservation)
        return True

    def _retry_reservation(self) -> bool:
        try:
            if self._pending_spare_instance is None:
                reservation = self._planner.reserve_product(self.inventory, self._products[self._product_index])
                self._record_spare_sources(reservation)
            else:
                spare = self.inventory.replace_with_spare(self._pending_spare_instance)
                self._pending_spare_instance = None
                product = self.state_machine.current_product
                self._events.append(ControllerEvent("spare", product, spare.object_id, spare.instance_id))
        except InventoryShortageError:
            return False
        self.state_machine.resume()
        event = ControllerEvent("resume", self.state_machine.current_product, None, "inventory restored")
        self._events.append(event)
        return True

    def _record_spare_sources(self, reservation: ProductReservation) -> None:
        for part in reservation.parts:
            if part.source_zone is not InventoryZone.SPARE or part.instance_id in self._announced_spares:
                continue
            self._announced_spares.add(part.instance_id)
            self._events.append(ControllerEvent("spare", reservation.product, part.object_id, part.instance_id))

    def _advance_product(self) -> bool:
        if self._product_index + 1 >= len(self._products):
            self._finished = True
            return False
        self.state_machine.reset("next product")
        self.state_machine.transition(PackMLState.IDLE)
        self._product_index += 1
        return True

    def _prepare_motion(self, observation: Observation, current: Waypoint, observed_at: float) -> bool:
        try:
            prepared = self._planner.next_step(
                self.inventory,
                observed_at_s=observed_at,
                attempt=self._failed_attempts + 1,
            )
        except PlannerError as error:
            self._abort(str(error))
            return False
        if prepared is None:
            self._finish_product()
            return False
        if not self._pre_pick_allows(prepared, observation, observed_at):
            return False
        return self._queue_prepared(prepared, observation, current, observed_at)

    def _pre_pick_allows(
        self,
        prepared: PreparedPlanStep,
        observation: Observation,
        observed_at: float,
    ) -> bool:
        evidence = call_port("pre-pick perception", self._perception.pre_pick, observation, prepared.part)
        self._require_fresh_evidence(evidence, EvidencePair, observed_at, "pre-pick evidence")
        decision = verify_pre_pick(
            prepared.record.object_id,
            evidence.detection,
            evidence.inspection,
            expected_instance_id=prepared.part.instance_id,
            thresholds=self._verification_thresholds,
        )
        expected = oriented_pose_values(prepared.grasp)
        agrees = pose_agrees(evidence.detection.get("pose"), expected, self._perception_tolerances)
        if decision.outcome is VerificationOutcome.HOLD_ANDON or not agrees:
            self._hold(f"pre-pick checks disagree for {prepared.record.object_id}")
            return False
        if decision.outcome is VerificationOutcome.REJECT_TO_SPARE:
            if decision.instance_id != prepared.part.instance_id:
                self._hold("pre-pick evidence does not match the reserved physical instance")
                return False
            self._replace_rejected(prepared, decision)
            return False
        return True

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
        queued = self._set_guarded_motion(
            plan,
            observation,
            current,
            observed_at,
            prepared.part.instance_id,
            prepared.part.object_id,
            prepared.base.instance_id,
            prepared.base_pose,
        )
        if queued:
            self._records.append(prepared.record)
            event = ControllerEvent("motion", prepared.record.product, prepared.record.object_id, "guarded")
            self._events.append(event)
        return queued

    @staticmethod
    def _require_fresh_evidence(
        evidence: object,
        expected_type: type[EvidencePair],
        observed_at: float,
        label: str,
    ) -> None:
        if type(evidence) is not expected_type:
            raise ControllerPortError(f"{label} must be exactly {expected_type.__name__}")
        require_fresh(evidence.observed_at_s, observed_at, label)

    def _finish_product(self) -> None:
        product = self.state_machine.current_product
        if product is None:
            raise RuntimeError("completed PackML cycle has no product owner")
        self.inventory.complete_product(product)
        self.state_machine.complete()
        self.state_machine.transition(PackMLState.COMPLETE)
        self._completed.append(product)
        self._events.append(ControllerEvent("complete", product, None, "four verified steps"))

    def _suspend(self, reason: str) -> None:
        if self.state_machine.state is PackMLState.EXECUTE:
            self.state_machine.suspend(reason)
        event = ControllerEvent("andon", self.state_machine.current_product, None, reason)
        self._events.append(event)

    def _hold(self, reason: str) -> None:
        if "operator reconciliation" not in reason:
            if self._part_held:
                reason = f"{reason}; held part requires operator reconciliation"
            elif self._release_dispatched or self._has_physical_progress():
                reason = f"{reason}; physical progress requires operator reconciliation"
        self._motion_queue.clear()
        if self.state_machine.state is PackMLState.EXECUTE:
            self.state_machine.hold(reason)
        event = ControllerEvent("andon", self.state_machine.current_product, None, reason)
        self._events.append(event)

    def _abort(self, reason: str) -> None:
        if self._part_held:
            self._hold(f"{reason}; held part requires operator reconciliation")
            return
        if self._release_dispatched or self._has_physical_progress():
            self._hold(f"{reason}; installed or staged parts require operator reconciliation")
            return
        self._motion_queue.clear()
        try:
            self.state_machine.abort(reason)
        except ReservationReleaseError as error:
            self._hold(f"{reason}; {error}; staged parts require operator reconciliation")
            return
        self.state_machine.transition(PackMLState.ABORTED)
        event = ControllerEvent("abort", self.state_machine.current_product, None, reason)
        self._events.append(event)

    def _has_physical_progress(self) -> bool:
        product = self.state_machine.current_product
        if product is None:
            return False
        physical = {InventoryZone.INSTALLED, InventoryZone.STAGED}
        return any(item.reserved_for == product and item.zone in physical for item in self.inventory.items())
