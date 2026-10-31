"""PackML orchestration from perception evidence to guarded TCP chunks."""

from __future__ import annotations

from collections.abc import Mapping

from assembly.catalog import Catalog
from assembly.controller_completion import CompletionControllerMixin, validate_final_product_verifier
from assembly.controller_execution import ExecutionControllerMixin
from assembly.controller_inspection import InspectionControllerMixin, PendingReject
from assembly.controller_placement import PlacementControllerMixin
from assembly.controller_ports import (
    ControllerEvent,
    ControllerPortError,
    ExecutionTolerances,
    FinalProductVerifier,
    PerceptionPort,
    PerceptionTolerances,
    call_port,
    current_waypoint,
    observation_time,
    reconcile_inventory,
)
from assembly.controller_recovery import PendingInsertionLift, RecoveryControllerMixin
from assembly.controller_startup import StartupHoldConfig, validated_startup_hold
from assembly.execution import EndpointSettleTimeout, ExecutionError, GuardedMotionQueue
from assembly.instruction import InstructionError, parse_requested_products
from assembly.inventory import (
    InvalidInventoryObjectError,
    Inventory,
    InventoryError,
    InventoryShortageError,
    InventoryZone,
    ProductReservation,
)
from assembly.motion import (
    InspectionStopConfig,
    MotionConfig,
    MotionError,
    Waypoint,
)
from assembly.planner import (
    CalibratedPoseProvider,
    PlannerError,
    PlanStepRecord,
    PreparedPlanStep,
    RecipePlanner,
)
from assembly.recovery import RecoveryTargetProvider
from assembly.safety import SafetySupervisor
from assembly.state_machine import PackMLState, PackMLStateMachine, ReservationReleaseError, StateMachineError
from assembly.verification import (
    VerificationInputError,
    VerificationThresholds,
)
from robot.adapter import Action, Observation


class ControllerInputError(ValueError):
    """Observation or perception-port data is unsafe to execute."""


class AssemblyController(
    CompletionControllerMixin,
    ExecutionControllerMixin,
    InspectionControllerMixin,
    PlacementControllerMixin,
    RecoveryControllerMixin,
):
    """Execute one reserved product at a time with fresh verification gates."""

    def __init__(
        self,
        *,
        catalog: Catalog,
        perception: PerceptionPort,
        poses: CalibratedPoseProvider,
        motion_config: MotionConfig,
        inspection_stop: InspectionStopConfig,
        startup_hold: StartupHoldConfig | None,
        safety: SafetySupervisor,
        verification_thresholds: VerificationThresholds,
        execution_tolerances: ExecutionTolerances,
        perception_tolerances: PerceptionTolerances,
        recovery_targets: RecoveryTargetProvider,
        final_verifier: FinalProductVerifier,
        allow_mock_final_verification: bool = False,
        allow_missing_startup_hold: bool = False,
    ) -> None:
        self._catalog = catalog
        self._perception = perception
        self._poses = poses
        self._motion_config = motion_config
        self._inspection_stop = inspection_stop
        if type(allow_missing_startup_hold) is not bool:
            raise ControllerInputError("allow_missing_startup_hold must be a boolean")
        if startup_hold is None:
            if not allow_missing_startup_hold:
                raise ControllerInputError("startup hold is required unless the controller owns missing-TCP handling")
            self._startup_hold = None
        else:
            self._startup_hold = validated_startup_hold(startup_hold)
        self._safety = safety
        self._verification_thresholds = verification_thresholds
        self._execution_tolerances = execution_tolerances
        self._perception_tolerances = perception_tolerances
        self._recovery_targets = recovery_targets
        self._final_verifier = final_verifier
        if type(allow_mock_final_verification) is not bool:
            raise ControllerInputError("allow_mock_final_verification must be a boolean")
        self._final_motion_source = validate_final_product_verifier(
            final_verifier,
            allow_mock=allow_mock_final_verification,
            controller_owned=self._owns_final_view_motion(final_verifier),
        )
        self.reset()

    def _owns_final_view_motion(self, verifier: object) -> bool:
        """Generic controllers cannot authorize physical final-view motion."""
        del verifier
        return False

    def reset(self) -> None:
        self._products: tuple[int, ...] = ()
        self._schedule: tuple[int, ...] = ()
        self._deferred_missing: set[int] = set()
        self._instruction: str | None = None
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
        self._reset_pending_state()

    def _reset_pending_state(self) -> None:
        self._prepared: PreparedPlanStep | None = None
        self._pending_pre_pick = None
        self._forced_reject_condition = None
        self._pending_reject: PendingReject | None = None
        self._pending_install = None
        self._pending_insertion_lift: PendingInsertionLift | None = None
        self._insertion_retries_by_instance: dict[str, int] = {}
        self._pending_drop_release: str | None = None
        self._drop_attempts_by_instance: dict[str, int] = {}
        self._awaiting_post = False
        self._pending_spare_instance: str | None = None
        self._announced_spares: set[str] = set()
        self._finished = False
        self._last_safe_waypoint: Waypoint | None = None
        self._release_dispatched = False
        self._part_held = False
        self._held_instance_id: str | None = None
        self._resume_acknowledged = False
        self._pending_post_place_operator_decision = None
        self._pending_recovery_transfer_operator_decision = None
        self._coarse_xy_by_instance: dict[str, tuple[float, float]] = {}
        self._intrusion_attempts = 0
        self._intrusion_waiting_for_rescan = False
        self._pending_intrusion_replan = False

    @property
    def completed_products(self) -> tuple[int, ...]:
        return tuple(self._completed)

    @property
    def requested_products(self) -> tuple[int, ...]:
        return self._products

    @property
    def events(self) -> tuple[ControllerEvent, ...]:
        return tuple(self._events)

    @property
    def plan_records(self) -> tuple[PlanStepRecord, ...]:
        return tuple(self._records)

    @property
    def finished(self) -> bool:
        return self._finished

    def acknowledge_resume(self) -> None:
        """Record an operator acknowledgement; the next fresh scan performs resume."""
        if self.state_machine.state is not PackMLState.HELD:
            raise ControllerInputError("operator resume acknowledgement requires Held state")
        self._resume_acknowledged = True

    def act(self, observation: Observation) -> Action:
        try:
            current = current_waypoint(observation)
        except ControllerPortError as error:
            return self._hold_last_safe(str(error))
        self._last_safe_waypoint = current
        self._part_held = current.gripper == 1.0
        try:
            observed_at = observation_time(observation)
            self._bind_instruction(observation)
            return self._act_current(observation, current, observed_at)
        except InvalidInventoryObjectError as error:
            reason = (
                f"foreign object cannot be picked without validated catalog identity and sourced FoundationPose grasp: "
                f"{error.object_id!r}"
            )
            self._hold(reason)
            return current.as_action()
        except (
            ControllerPortError,
            InstructionError,
            InventoryError,
            MotionError,
            PlannerError,
            RuntimeError,
            StateMachineError,
            VerificationInputError,
        ) as error:
            self._hold(str(error))
            return current.as_action()

    def _bind_instruction(self, observation: Observation) -> None:
        trial = observation.get("trial")
        if not isinstance(trial, Mapping):
            raise InstructionError("observation.trial must be a mapping")
        instruction = trial.get("instruction")
        if self._instruction is None:
            products = parse_requested_products(instruction, self._catalog)
            self._instruction = instruction
            self._products = products
            self._schedule = products
            return
        if instruction != self._instruction:
            raise InstructionError("trial.instruction changed before controller reset")

    def _act_current(self, observation: Observation, current: Waypoint, observed_at: float) -> Action:
        snapshot = call_port("inventory scan", self._perception.scan, observation)
        reconcile_inventory(
            self.inventory,
            snapshot,
            observed_at,
            held_instance_id=self._reconciliation_exempt_instance(),
        )
        self._coarse_xy_by_instance = {
            item.instance_id: item.coarse_xy_mm
            for item in snapshot.items
            if item.coarse_xy_mm is not None
        }
        if self._intrusion_waiting_for_rescan:
            self._intrusion_waiting_for_rescan = False
        if not self._surface_claimed_conditions():
            return current.as_action()
        self._resume_after_scan()
        try:
            settle_action = self._confirm_previous_endpoint(observation, current, observed_at)
        except EndpointSettleTimeout as error:
            self._hold(str(error))
            return current.as_action()
        except ExecutionError as error:
            self._abort(str(error))
            return current.as_action()
        if settle_action is not None:
            return settle_action
        return self._progress(observation, current, observed_at)

    def _confirm_previous_endpoint(
        self,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> Action | None:
        del observation, observed_at
        self._motion_queue.confirm_previous_endpoint(current)
        return None

    def _resume_after_scan(self) -> None:
        if self.state_machine.state is not PackMLState.HELD or not self._resume_acknowledged:
            return
        if self._pending_post_place_operator_decision is not None:
            blocker = self._operator_post_place_resume_blocker()
        elif self._pending_recovery_transfer_operator_decision is not None:
            blocker = self._operator_recovery_transfer_resume_blocker()
        else:
            blocker = self._resume_physical_blocker()
        if blocker is not None:
            self._resume_acknowledged = False
            reason = f"operator resume blocked: unresolved {blocker} requires physical reconciliation"
            self._events.append(
                ControllerEvent("andon", self.state_machine.current_product, None, reason)
            )
            return
        if self.state_machine.current_product is None:
            self.state_machine.reset("operator corrected trial instruction")
            self.state_machine.transition(PackMLState.IDLE)
        else:
            self.state_machine.resume()
        self._resume_acknowledged = False
        self._events.append(
            ControllerEvent("resume", self.state_machine.current_product, None, "operator acknowledged after rescan")
        )

    def _progress(self, observation: Observation, current: Waypoint, observed_at: float) -> Action:
        for _ in range(len(self._products) + 4):
            handled, action = self._progress_cycle(observation, current, observed_at)
            if action is not None:
                return action
            if handled:
                continue
        raise RuntimeError("controller failed to make bounded progress")

    def _progress_cycle(
        self,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> tuple[bool, Action | None]:
        terminal = {PackMLState.HELD, PackMLState.ABORTED}
        if self._finished or self.state_machine.state in terminal:
            return True, current.as_action()
        if self._intrusion_waiting_for_rescan:
            return True, current.as_action()
        if not self._ensure_product_started():
            return True, current.as_action()
        if self._motion_queue.active:
            return True, self._emit_motion_chunk(observation, current, observed_at)
        handled, action = self._progress_pending(observation, current, observed_at)
        if handled:
            return handled, action
        return self._progress_ready(observation, current, observed_at)

    def _progress_pending(
        self,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> tuple[bool, Action | None]:
        if self._pending_intrusion_replan:
            queued = self._replan_after_intrusion(observation, current, observed_at)
            action = self._emit_motion_chunk(observation, current, observed_at) if queued else None
            return True, action
        if self._pending_reject is not None:
            self._verify_reject_transfer(observation, current, observed_at)
            return True, None
        if self._pending_drop_release is not None:
            self._commit_drop_release()
            return True, None
        if self._pending_pre_pick is not None:
            queued = self._prepare_after_inspection(observation, current, observed_at)
            action = self._emit_motion_chunk(observation, current, observed_at) if queued else None
            return True, action
        if self._pending_install is not None:
            self._commit_install()
            return True, None
        return False, None

    def _progress_ready(
        self,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> tuple[bool, Action | None]:
        if self.inventory.active_product is None:
            handled, action = self._queue_environment_cleanup(observation, current, observed_at)
            if handled:
                return True, action
            if not self._ensure_reservation():
                return True, current.as_action()
        if self._pending_insertion_lift is not None:
            queued = self._prepare_after_insertion_lift(observation, current, observed_at)
            action = self._emit_motion_chunk(observation, current, observed_at) if queued else None
            return True, action
        if self._awaiting_post:
            verified = self._verify_placement(observation, current, observed_at)
            return True, current.as_action() if verified else None
        queued = self._prepare_motion(observation, current, observed_at)
        action = self._emit_motion_chunk(observation, current, observed_at) if queued else None
        return True, action

    def _queue_environment_cleanup(
        self,
        observation: Observation,
        current: Waypoint,
        observed_at: float,
    ) -> tuple[bool, Action | None]:
        forced = self._environment_reject_candidate(observed_at)
        if forced is None:
            return False, None
        candidate, condition = forced
        queued = self._queue_inspection(candidate, condition, observation, current, observed_at)
        if not queued:
            return True, None
        return True, self._emit_motion_chunk(observation, current, observed_at)

    def _hold_last_safe(self, reason: str) -> Action:
        hold = self._motion_queue.pending_endpoint or self._last_safe_waypoint or self._startup_hold
        if hold is None:
            raise ControllerInputError("controller has no measured TCP or reviewed startup hold")
        self._hold(reason)
        return hold.as_action()

    def _ensure_product_started(self) -> bool:
        state = self.state_machine.state
        if state is PackMLState.COMPLETE:
            if not self._advance_product():
                return False
            state = self.state_machine.state
        if state is PackMLState.IDLE:
            product = self._schedule[self._product_index]
            self.state_machine.start(product)
            self.state_machine.transition(PackMLState.EXECUTE)
        if self.state_machine.state is PackMLState.SUSPENDED:
            return self._retry_reservation()
        return True

    def _ensure_reservation(self) -> bool:
        if self.inventory.active_product is not None:
            return True
        try:
            reservation = self._planner.reserve_product(self.inventory, self._schedule[self._product_index])
        except InventoryShortageError as error:
            if self._defer_missing_product(error):
                return False
            self._suspend(str(error))
            return False
        spare = next((part for part in reservation.parts if part.source_zone is InventoryZone.SPARE), None)
        if spare is not None and self._defer_current_product(
            spare.object_id,
            f"table part {spare.object_id} missing; defer before using spare {spare.instance_id}",
        ):
            return False
        self._record_spare_sources(reservation)
        return True

    def _retry_reservation(self) -> bool:
        try:
            if self._pending_spare_instance is None:
                reservation = self._planner.reserve_product(self.inventory, self._schedule[self._product_index])
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
        if self._product_index + 1 >= len(self._schedule):
            self._finished = True
            return False
        self.state_machine.reset("next product")
        self.state_machine.transition(PackMLState.IDLE)
        self._product_index += 1
        return True

    def _defer_missing_product(self, error: InventoryShortageError) -> bool:
        return self._defer_current_product(error.required_object_id, str(error))

    def _defer_current_product(self, object_id: str | None, detail: str) -> bool:
        product = self._schedule[self._product_index]
        has_other_work = self._product_index + 1 < len(self._schedule)
        if product in self._deferred_missing or not has_other_work:
            return False
        remaining = self._schedule[self._product_index + 1 :]
        self._schedule = (*self._schedule[: self._product_index], *remaining, product)
        self._deferred_missing.add(product)
        self.state_machine.stop("defer missing product while other requested products remain")
        self.state_machine.transition(PackMLState.STOPPED)
        self.state_machine.reset("deferred product")
        self.state_machine.transition(PackMLState.IDLE)
        self._events.append(ControllerEvent("deferred", product, object_id, detail))
        return True

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
        self._resume_acknowledged = False
        if self.state_machine.state in {PackMLState.IDLE, PackMLState.EXECUTE, PackMLState.SUSPENDED}:
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
