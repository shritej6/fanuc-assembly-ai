"""Staged, inventory-consistent dismantle and rebuild orchestration.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from enum import StrEnum
from numbers import Number
from typing import Protocol

from assembly.catalog import Catalog, ProductRecipe, RecipeStep
from assembly.inventory import Inventory, InventoryItem, InventoryZone, ObjectCondition
from assembly.motion import (
    MotionChunk,
    MotionConfig,
    OrientedPartPose,
    Waypoint,
    compile_pick_to_insert,
    compile_reinsert,
    compile_transfer_held,
)
from assembly.planner import CalibratedPoseProvider, PreparedPlanStep
from assembly.verification import RecoveryDecision, RecoveryOperation


class RecoveryError(RuntimeError):
    """Recovery cannot be planned without calibrated physical state."""


@dataclass(frozen=True, slots=True)
class PullOutTarget:
    """Calibrated set-down pose and absolute retreat plane for learned pull-out."""

    pose: OrientedPartPose
    clear_z_mm: float
    source: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.clear_z_mm, bool)
            or not isinstance(self.clear_z_mm, Number)
            or not math.isfinite(float(self.clear_z_mm))
        ):
            raise RecoveryError("pull-out clear_z_mm must be finite")
        if self.clear_z_mm <= self.pose.tcp.z:
            raise RecoveryError("pull-out clear_z_mm must be above the set-down pose")
        if not isinstance(self.source, str) or not self.source.strip():
            raise RecoveryError("pull-out target requires a recorded source")


class RecoveryStageKind(StrEnum):
    REJECT_FAILED = "reject_failed"
    STAGE_INSTALLED = "stage_installed"
    REBUILD_INSTALLED = "rebuild_installed"
    INSERT_SPARE = "insert_spare"


@dataclass(frozen=True, slots=True)
class RecoveryStage:
    kind: RecoveryStageKind
    instance_id: str
    object_id: str
    staging_slot: int | None = None


@dataclass(frozen=True, slots=True)
class RecoveryStagePlan:
    stage: RecoveryStage
    part: InventoryItem
    chunks: tuple[MotionChunk, ...]
    grasp: OrientedPartPose | None
    target: OrientedPartPose
    verification_target: OrientedPartPose | None
    base_pose: OrientedPartPose
    gripper_roll_deg: float
    release_after_verification: bool


@dataclass(frozen=True, slots=True)
class RetryRecoveryPlan:
    prepared: PreparedPlanStep
    chunks: tuple[MotionChunk, ...]
    retry_number: int


@dataclass(frozen=True, slots=True)
class _RecoveryCompileContext:
    inventory: Inventory
    catalog: Catalog
    poses: CalibratedPoseProvider
    current: Waypoint
    config: MotionConfig
    max_position_error_mm: float
    max_orientation_error_deg: float
    observed_at_s: float


class RecoveryTargetProvider(Protocol):
    """Cell-calibrated staging and reject targets; no defaults are safe."""

    def pull_out_target_for(
        self,
        product: int,
        part: InventoryItem,
    ) -> PullOutTarget: ...

    def staging_target_for(
        self,
        product: int,
        part: InventoryItem,
        slot: int,
    ) -> OrientedPartPose: ...

    def staging_verification_target_for(
        self,
        product: int,
        part: InventoryItem,
        slot: int,
    ) -> OrientedPartPose: ...

    def reject_target_for(self, part: InventoryItem) -> OrientedPartPose: ...


class RecoverySession:
    """Advance one physically verified recovery stage at a time."""

    def __init__(self, stages: tuple[RecoveryStage, ...]) -> None:
        if not stages:
            raise RecoveryError("recovery session requires at least one stage")
        self._pending = deque(stages)
        self._completed: list[RecoveryStage] = []
        self.replacement_instance_id: str | None = None

    @property
    def current(self) -> RecoveryStage:
        if not self._pending:
            raise RecoveryError("recovery session is complete")
        return self._pending[0]

    @property
    def completed(self) -> tuple[RecoveryStage, ...]:
        return tuple(self._completed)

    @property
    def done(self) -> bool:
        return not self._pending

    def advance(self, stage: RecoveryStage) -> None:
        if self.current != stage:
            raise RecoveryError("only the current verified recovery stage may advance")
        self._completed.append(self._pending.popleft())


class RecoveryCoordinator:
    """Compile and commit one calibrated dismantle/rebuild stage at a time."""

    def __init__(self, targets: RecoveryTargetProvider) -> None:
        self._targets = targets
        self._session: RecoverySession | None = None
        self._failed: PreparedPlanStep | None = None

    @property
    def active(self) -> bool:
        return self._session is not None

    @property
    def current_stage(self) -> RecoveryStage:
        session, _ = self._state()
        return session.current

    def begin(
        self,
        inventory: Inventory,
        prepared: PreparedPlanStep,
        decision: RecoveryDecision,
    ) -> None:
        if self._session is not None:
            raise RecoveryError("a recovery session is already active")
        self._session = build_recovery_session(inventory, prepared, decision)
        self._failed = prepared

    def prepare_stage(
        self,
        *,
        inventory: Inventory,
        catalog: Catalog,
        poses: CalibratedPoseProvider,
        current: Waypoint,
        config: MotionConfig,
        max_position_error_mm: float,
        max_orientation_error_deg: float,
        observed_at_s: float,
        grasp: OrientedPartPose | None = None,
    ) -> RecoveryStagePlan | None:
        session, failed = self._state()
        stage = session.current
        if stage.kind is RecoveryStageKind.INSERT_SPARE:
            self._finish_session(session, stage)
            return None
        part = inventory.item(stage.instance_id)
        context = _RecoveryCompileContext(
            inventory,
            catalog,
            poses,
            current,
            config,
            max_position_error_mm,
            max_orientation_error_deg,
            observed_at_s,
        )
        try:
            return self._compile_stage(stage, part, failed, context, grasp)
        except RecoveryError:
            raise
        except Exception as error:
            raise RecoveryError(f"recovery calibration or motion is invalid: {error}") from error

    def commit(self, inventory: Inventory, stage: RecoveryStage) -> InventoryItem | None:
        session, _ = self._state()
        if stage.kind is RecoveryStageKind.REJECT_FAILED:
            inventory.reject(stage.instance_id, condition=ObjectCondition.DEFECTIVE)
            spare = inventory.replace_with_spare(stage.instance_id)
            session.replacement_instance_id = spare.instance_id
            session.advance(stage)
            return spare
        if stage.kind is RecoveryStageKind.STAGE_INSTALLED:
            inventory.stage_for_rebuild(stage.instance_id)
        elif stage.kind is RecoveryStageKind.REBUILD_INSTALLED:
            inventory.reinstall(stage.instance_id)
        else:
            raise RecoveryError(f"unsupported recovery stage {stage.kind}")
        session.advance(stage)
        return None

    def _state(self) -> tuple[RecoverySession, PreparedPlanStep]:
        if self._session is None or self._failed is None:
            raise RecoveryError("no recovery session is active")
        return self._session, self._failed

    def _finish_session(self, session: RecoverySession, stage: RecoveryStage) -> None:
        if session.replacement_instance_id is None:
            raise RecoveryError("recovery finished without a reserved exact spare")
        session.advance(stage)
        self._session = None
        self._failed = None

    def _compile_stage(
        self,
        stage: RecoveryStage,
        part: InventoryItem,
        failed: PreparedPlanStep,
        context: _RecoveryCompileContext,
        grasp: OrientedPartPose | None,
    ) -> RecoveryStagePlan:
        recipe = context.catalog.product(failed.record.product)
        recipe_step = next(step for step in recipe.steps if step.object_id == part.object_id)
        base_pose = context.poses.base_for(
            context.inventory.active_reservation.base,
            observed_at_s=context.observed_at_s,
        )
        if stage.kind is RecoveryStageKind.REJECT_FAILED:
            return self._compile_reject_stage(
                stage,
                part,
                recipe_step,
                base_pose,
                context.current,
                context.config,
                context.max_position_error_mm,
                context.max_orientation_error_deg,
            )
        if grasp is None:
            raise RecoveryError("recovery re-pick requires a sourced FoundationPose grasp")
        return self._compile_pick_stage(
            stage,
            part,
            recipe,
            recipe_step,
            context,
            base_pose,
            grasp,
        )

    def _compile_reject_stage(
        self,
        stage: RecoveryStage,
        part: InventoryItem,
        recipe_step: RecipeStep,
        base_pose: OrientedPartPose,
        current: Waypoint,
        config: MotionConfig,
        max_position_error_mm: float,
        max_orientation_error_deg: float,
    ) -> RecoveryStagePlan:
        target = self._targets.reject_target_for(part)
        chunks = compile_transfer_held(
            current,
            target,
            gripper_roll_deg=recipe_step.preferred_grasp.turn_deg,
            config=config,
            max_position_error_mm=max_position_error_mm,
            max_orientation_error_deg=max_orientation_error_deg,
        )
        return RecoveryStagePlan(
            stage,
            part,
            chunks,
            None,
            target,
            None,
            base_pose,
            recipe_step.preferred_grasp.turn_deg,
            False,
        )

    def _compile_pick_stage(
        self,
        stage: RecoveryStage,
        part: InventoryItem,
        recipe: ProductRecipe,
        recipe_step: RecipeStep,
        context: _RecoveryCompileContext,
        base_pose: OrientedPartPose,
        grasp: OrientedPartPose,
    ) -> RecoveryStagePlan:
        target, verification_target = self._stage_targets(
            stage,
            part,
            recipe,
            recipe_step,
            context.inventory,
            context.poses,
            context.observed_at_s,
        )
        chunks = compile_pick_to_insert(
            context.current,
            grasp,
            target,
            gripper_roll_deg=recipe_step.preferred_grasp.turn_deg,
            config=context.config,
        )
        return RecoveryStagePlan(
            stage,
            part,
            chunks,
            grasp,
            target,
            verification_target,
            base_pose,
            recipe_step.preferred_grasp.turn_deg,
            True,
        )

    def _stage_targets(
        self,
        stage: RecoveryStage,
        part: InventoryItem,
        recipe: ProductRecipe,
        recipe_step: RecipeStep,
        inventory: Inventory,
        poses: CalibratedPoseProvider,
        observed_at_s: float,
    ) -> tuple[OrientedPartPose, OrientedPartPose]:
        if stage.staging_slot is None:
            raise RecoveryError("staged recovery stage has no calibrated slot")
        staging = self._targets.staging_target_for(recipe.product, part, stage.staging_slot)
        if stage.kind is RecoveryStageKind.STAGE_INSTALLED:
            verification = self._targets.staging_verification_target_for(
                recipe.product,
                part,
                stage.staging_slot,
            )
            return staging, verification
        target = poses.target_for(
            recipe,
            inventory.active_reservation.base,
            recipe_step,
            observed_at_s=observed_at_s,
        )
        verification = poses.verification_target_for(
            recipe,
            inventory.active_reservation.base,
            recipe_step,
            observed_at_s=observed_at_s,
        )
        return target, verification


def compile_retry_recovery(
    prepared: PreparedPlanStep,
    decision: RecoveryDecision,
    current: Waypoint,
    config: MotionConfig,
) -> RetryRecoveryPlan:
    expected = (RecoveryOperation.LIFT, RecoveryOperation.RE_YAW, RecoveryOperation.REINSERT)
    if decision.operations != expected or decision.retry_number is None:
        raise RecoveryError("recovery policy returned an invalid retry sequence")
    retry_record = replace(
        prepared.record,
        attempt=decision.retry_number + 1,
        step_id=f"{prepared.record.step_id}-retry-{decision.retry_number}",
    )
    retry_prepared = replace(prepared, record=retry_record)
    chunks = compile_reinsert(
        current,
        prepared.target,
        gripper_roll_deg=prepared.record.gripper_roll_deg,
        config=config,
    )
    return RetryRecoveryPlan(retry_prepared, chunks, decision.retry_number)


def build_recovery_session(
    inventory: Inventory,
    prepared: PreparedPlanStep,
    decision: RecoveryDecision,
) -> RecoverySession:
    """Resolve decision object IDs to the active product's physical instances."""
    if decision.replacement_object_id != prepared.part.object_id:
        raise RecoveryError("recovery replacement must match the failed exact part")
    installed = _installed_instances(inventory, decision)
    staged = tuple(enumerate(installed, start=1))
    stages = [RecoveryStage(RecoveryStageKind.REJECT_FAILED, prepared.part.instance_id, prepared.part.object_id)]
    stages.extend(
        RecoveryStage(RecoveryStageKind.STAGE_INSTALLED, item.instance_id, item.object_id, slot)
        for slot, item in staged
    )
    stages.extend(
        RecoveryStage(RecoveryStageKind.REBUILD_INSTALLED, item.instance_id, item.object_id, slot)
        for slot, item in reversed(staged)
    )
    stages.append(RecoveryStage(RecoveryStageKind.INSERT_SPARE, "pending-spare", prepared.part.object_id))
    return RecoverySession(tuple(stages))


def _installed_instances(inventory: Inventory, decision: RecoveryDecision) -> tuple[InventoryItem, ...]:
    by_object = {item.object_id: item for item in inventory.items(InventoryZone.INSTALLED)}
    try:
        return tuple(by_object[object_id] for object_id in decision.dismantle_order)
    except KeyError as error:
        raise RecoveryError(f"installed recovery part {error.args[0]!r} is missing") from error
