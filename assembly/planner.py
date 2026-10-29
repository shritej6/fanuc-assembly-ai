"""One-product recipe planning backed by calibrated perception poses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Number
from typing import Any, Protocol

from assembly.catalog import Catalog, ProductRecipe, RecipeStep
from assembly.inventory import (
    Inventory,
    InventoryItem,
    InventoryZone,
    NoActiveReservationError,
    ObjectCondition,
    ProductReservation,
)
from assembly.motion import AbsolutePoseUF0, OrientedPartPose, RestingState


class PlannerError(Exception):
    """Base class for recipe-planning failures."""


class PlannerStateError(PlannerError):
    """Inventory state cannot produce the requested plan step."""


class PlanOrderError(PlannerError):
    """Installed inventory is not a valid prefix of the recipe graph."""


class PlannerPoseError(PlannerError):
    """A pose is not calibrated, installed, finite absolute UF0 data."""


class CalibratedPoseProvider(Protocol):
    """Runtime boundary joining perception/calibration to recipe identity."""

    def grasp_for(self, part: InventoryItem, *, observed_at_s: float) -> OrientedPartPose: ...

    def base_for(self, base: InventoryItem, *, observed_at_s: float) -> OrientedPartPose: ...

    def target_for(
        self,
        product: ProductRecipe,
        base: InventoryItem,
        step: RecipeStep,
        *,
        observed_at_s: float,
    ) -> OrientedPartPose: ...


@dataclass(frozen=True, slots=True)
class ContractPoseUF0:
    x: float
    y: float
    z: float
    yaw: float

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw}


@dataclass(frozen=True, slots=True)
class PlanStepRecord:
    step_id: str
    product: int
    instance_id: str
    object_id: str
    grasp: ContractPoseUF0
    target: ContractPoseUF0
    gripper_roll_deg: float
    attempt: int
    op: str = "insert"

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "op": self.op,
            "product": self.product,
            "instance_id": self.instance_id,
            "object_id": self.object_id,
            "grasp": self.grasp.as_dict(),
            "target": self.target.as_dict(),
            "gripper_roll_deg": self.gripper_roll_deg,
            "attempt": self.attempt,
        }


@dataclass(frozen=True, slots=True)
class PreparedPlanStep:
    record: PlanStepRecord
    recipe_step: RecipeStep
    base: InventoryItem
    part: InventoryItem
    base_pose: OrientedPartPose
    grasp: OrientedPartPose
    target: OrientedPartPose


def _contract_pose(pose: OrientedPartPose) -> ContractPoseUF0:
    tcp = pose.tcp
    return ContractPoseUF0(tcp.x, tcp.y, tcp.z, tcp.r)


def _validate_pose(label: str, pose: object) -> OrientedPartPose:
    if not isinstance(pose, OrientedPartPose):
        raise PlannerPoseError(f"{label} pose provider must return OrientedPartPose")
    if not isinstance(pose.tcp, AbsolutePoseUF0):
        raise PlannerPoseError(f"{label} TCP must be an AbsolutePoseUF0")
    if (
        pose.resting_state is not RestingState.INSTALLED
        or not isinstance(pose.up_face, str)
        or not pose.up_face.strip()
    ):
        raise PlannerPoseError(f"{label} pose must identify the installed up-face")
    if pose.tcp.frame != "UF0" or pose.tcp.source != "calibrated":
        raise PlannerPoseError(f"{label} pose must be calibrated absolute UF0")
    for field in ("x", "y", "z", "w", "p", "r"):
        value = getattr(pose.tcp, field)
        if isinstance(value, bool) or not isinstance(value, Number) or not math.isfinite(float(value)):
            raise PlannerPoseError(f"{label}.{field} must be finite")
    return pose


def _next_recipe_step(
    inventory: Inventory,
    reservation: ProductReservation,
    recipe: ProductRecipe,
) -> tuple[RecipeStep, InventoryItem] | None:
    first_open: tuple[RecipeStep, InventoryItem] | None = None
    for recipe_step in recipe.steps:
        reserved = reservation.part(recipe_step.object_id)
        current = inventory.item(reserved.instance_id)
        if current.zone is InventoryZone.INSTALLED:
            if first_open is not None:
                raise PlanOrderError("installed parts must form a contiguous recipe prefix")
            continue
        if current.zone is not InventoryZone.RESERVED or current.reserved_for != recipe.product:
            raise PlannerStateError(f"{current.instance_id!r} is not reserved for product {recipe.product}")
        if first_open is None:
            first_open = (recipe_step, current)
    return first_open


def _live_base(inventory: Inventory, reservation: ProductReservation) -> InventoryItem:
    base = inventory.item(reservation.base.instance_id)
    valid = (
        base.object_id == reservation.base.object_id
        and base.zone is InventoryZone.RESERVED
        and base.condition is ObjectCondition.VERIFIED
        and base.reserved_for == reservation.product
    )
    if not valid:
        raise PlannerStateError("active product base must remain reserved and verified")
    return base


def _prepared(
    product: ProductRecipe,
    base: InventoryItem,
    recipe_step: RecipeStep,
    part: InventoryItem,
    poses: CalibratedPoseProvider,
    attempt: int,
    observed_at_s: float,
) -> PreparedPlanStep:
    try:
        grasp_result = poses.grasp_for(part, observed_at_s=observed_at_s)
        base_result = poses.base_for(base, observed_at_s=observed_at_s)
        target_result = poses.target_for(product, base, recipe_step, observed_at_s=observed_at_s)
    except PlannerError:
        raise
    except Exception as error:
        raise PlannerPoseError(f"calibrated pose snapshot is unavailable: {error}") from error
    grasp = _validate_pose("grasp", grasp_result)
    base_pose = _validate_pose("base", base_result)
    target = _validate_pose("target", target_result)
    if grasp.up_face != target.up_face:
        raise PlannerPoseError("grasp and target installed up-face must match")
    record = PlanStepRecord(
        step_id=f"product-{product.product}-step-{recipe_step.step:02d}-attempt-{attempt}",
        product=product.product,
        instance_id=part.instance_id,
        object_id=part.object_id,
        grasp=_contract_pose(grasp),
        target=_contract_pose(target),
        gripper_roll_deg=recipe_step.gripper_roll_deg,
        attempt=attempt,
    )
    return PreparedPlanStep(record, recipe_step, base, part, base_pose, grasp, target)


class RecipePlanner:
    """Emit the next exact recipe step for the single active reservation."""

    def __init__(self, catalog: Catalog, poses: CalibratedPoseProvider) -> None:
        self._catalog = catalog
        self._poses = poses

    def reserve_product(self, inventory: Inventory, product: int) -> ProductReservation:
        self._catalog.product(product)
        return inventory.reserve_product(product)

    def next_step(
        self,
        inventory: Inventory,
        *,
        observed_at_s: float,
        attempt: int = 1,
    ) -> PreparedPlanStep | None:
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise PlannerStateError("attempt must be a positive integer")
        if isinstance(observed_at_s, bool) or not isinstance(observed_at_s, Number) or not math.isfinite(observed_at_s):
            raise PlannerStateError("observed_at_s must be a finite number")
        try:
            reservation = inventory.active_reservation
        except NoActiveReservationError as error:
            raise PlannerStateError("an active reservation is required") from error
        recipe = self._catalog.product(reservation.product)
        base = _live_base(inventory, reservation)
        next_item = _next_recipe_step(inventory, reservation, recipe)
        if next_item is None:
            return None
        recipe_step, part = next_item
        return _prepared(recipe, base, recipe_step, part, self._poses, attempt, float(observed_at_s))
