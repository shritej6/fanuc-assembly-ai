"""Seeded prompt, layout and fault generation for the scenario fleet.
Authored by Shritej Gunda."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import combinations
from pathlib import Path

from assembly.catalog import BaseType, Catalog
from assembly.inventory import InventoryZone, ObjectCondition

TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "contracts" / "taxonomy.json"


class FleetInputError(ValueError):
    """Raised when a fleet request is incomplete or cannot make a valid scenario."""


class FaultKind(StrEnum):
    """Kinds of fault a scenario can inject."""
    CLEAN = "clean"
    DEFECT = "defect"
    FALLEN = "fallen"
    LOOKALIKE = "lookalike"
    FOREIGN = "foreign"
    SPARE_SHORTAGE = "spare_shortage"


@dataclass(frozen=True, slots=True)
class FleetBounds:
    """Mock geometry supplied by the caller, never real cell calibration."""

    x_mm: tuple[float, float]
    y_mm: tuple[float, float]
    part_z_mm: float
    base_z_mm: float
    tcp_local_mm: tuple[float, float, float]
    reject_pose: tuple[float, float, float, float]
    staging_x_mm: float
    staging_y_mm: tuple[float, float]
    home_pose: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        _validate_interval(self.x_mm, "x_mm")
        _validate_interval(self.y_mm, "y_mm")
        _validate_interval(self.staging_y_mm, "staging_y_mm")
        _finite_values((self.part_z_mm, self.base_z_mm, self.staging_x_mm), "fleet bounds")
        _finite_values(self.tcp_local_mm, "tcp_local_mm", count=3)
        _finite_values(self.reject_pose, "reject_pose", count=4)
        _finite_values(self.home_pose, "home_pose", count=6)


@dataclass(frozen=True, slots=True)
class ScenarioObject:
    """One object placed in a generated scenario."""
    instance_id: str
    object_id: str
    owner_product: int
    base_type: BaseType | None
    zone: InventoryZone
    condition: ObjectCondition
    pose: tuple[float, float, float, float]
    grasp_approved: bool = True
    hazardous: bool = False


@dataclass(frozen=True, slots=True)
class Scenario:
    """One generated scenario: prompt, objects and injected faults."""
    scenario_id: str
    products: tuple[int, ...]
    instruction: str
    fault: FaultKind
    objects: tuple[ScenarioObject, ...]
    bounds: FleetBounds
    required_missing_object_id: str | None = None

    @property
    def hazardous_objects(self) -> tuple[ScenarioObject, ...]:
        """Return the objects that are hazards in this scenario."""
        return tuple(item for item in self.objects if item.hazardous)

    @property
    def layout_signature(self) -> tuple[tuple[str, tuple[float, float, float, float]], ...]:
        """Return a key that identifies this layout."""
        return tuple(sorted((item.instance_id, item.pose) for item in self.objects))


def parse_instruction_products(instruction: object, catalog: Catalog) -> tuple[int, ...]:
    """Return the product ids named in the current trial instruction."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise FleetInputError("trial instruction must be a non-empty string")
    match = re.fullmatch(r"\s*assemble\s+products?\s+(.+?)\.?\s*", instruction, re.IGNORECASE)
    if match is None:
        raise FleetInputError("trial instruction must request products explicitly")
    products = tuple(int(value) for value in re.findall(r"\d+", match.group(1)))
    if len(products) != 3 or len(set(products)) != 3:
        raise FleetInputError("trial instruction must request exactly three unique products")
    known = {recipe.product for recipe in catalog.products}
    if any(product not in known for product in products):
        raise FleetInputError("trial instruction contains an unknown product")
    return products


class ScenarioGenerator:
    """Generate reproducible scenarios from a seed."""

    def __init__(self, catalog: Catalog, *, bounds: FleetBounds, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise FleetInputError("seed must be an integer")
        self._catalog = catalog
        self._bounds = bounds
        self._rng = random.Random(seed)
        self._product_ids = tuple(recipe.product for recipe in catalog.products)
        if len(self._product_ids) < 3:
            raise FleetInputError("catalog must contain at least three products")
        self._base_ids = _base_ids_by_type(catalog)
        self._part_products, self._lookalikes = _taxonomy_parts()

    def generate(self, count: int) -> tuple[Scenario, ...]:
        """Return `count` generated scenarios."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise FleetInputError("scenario count must be a positive integer")
        faults = self._fault_schedule(count)
        products = self._product_schedule(count, faults)
        return tuple(self._scenario(index, fault, selected) for index, (fault, selected) in enumerate(zip(faults, products)))

    def _fault_schedule(self, count: int) -> tuple[FaultKind, ...]:
        faults: list[FaultKind] = []
        choices = list(FaultKind)
        while len(faults) < count:
            self._rng.shuffle(choices)
            faults.extend(choices)
        return tuple(faults[:count])

    def _scenario(self, index: int, fault: FaultKind, products: tuple[int, ...]) -> Scenario:
        objects = self._nominal_objects(index, products)
        objects, missing = self._inject(index, products, objects, fault)
        joined = ", ".join(str(product) for product in products[:-1])
        instruction = f"Assemble products {joined}, and {products[-1]}."
        return Scenario(f"fleet-{index:04d}", products, instruction, fault, tuple(objects), self._bounds, missing)

    def _product_schedule(self, count: int, faults: tuple[FaultKind, ...]) -> tuple[tuple[int, ...], ...]:
        combinations_ = list(combinations(self._product_ids, 3))
        scheduled: list[tuple[int, ...]] = []
        while len(scheduled) < count:
            cycle = list(combinations_)
            self._rng.shuffle(cycle)
            scheduled.extend(cycle)
        scheduled = scheduled[:count]
        self._repair_lookalike_assignments(scheduled, faults, combinations_)
        return tuple(self._shuffled_prompt_order(products) for products in scheduled)

    def _repair_lookalike_assignments(
        self,
        scheduled: list[tuple[int, ...]],
        faults: tuple[FaultKind, ...],
        combinations_: list[tuple[int, ...]],
    ) -> None:
        for index, (products, fault) in enumerate(zip(scheduled, faults)):
            if fault is not FaultKind.LOOKALIKE or self._trap_candidates(products, fault):
                continue
            replacement = next(
                (
                    other
                    for other in range(len(scheduled))
                    if other != index
                    and faults[other] is not FaultKind.LOOKALIKE
                    and self._trap_candidates(scheduled[other], FaultKind.LOOKALIKE)
                ),
                None,
            )
            if replacement is None:
                scheduled[index] = self._rng.choice(
                    [combo for combo in combinations_ if self._trap_candidates(combo, FaultKind.LOOKALIKE)]
                )
            else:
                scheduled[index], scheduled[replacement] = scheduled[replacement], scheduled[index]

    def _shuffled_prompt_order(self, products: tuple[int, ...]) -> tuple[int, ...]:
        ordered = list(products)
        self._rng.shuffle(ordered)
        return tuple(ordered)

    def _nominal_objects(self, index: int, products: tuple[int, ...]) -> list[ScenarioObject]:
        objects: list[ScenarioObject] = []
        for product in products:
            recipe = self._catalog.product(product)
            base_id = self._rng.choice(self._base_ids[recipe.base_type])
            objects.append(self._object(index, f"base-{product}", base_id, product, recipe.base_type, True))
            for step in recipe.steps:
                objects.append(self._object(index, f"part-{product}-{step.step}", step.object_id, product, None, False))
        return objects

    def _object(
        self,
        index: int,
        suffix: str,
        object_id: str,
        product: int,
        base_type: BaseType | None,
        is_base: bool,
    ) -> ScenarioObject:
        z = self._bounds.base_z_mm if is_base else self._bounds.part_z_mm
        pose = (*self._random_xy(), z, self._rng.uniform(-180.0, 180.0))
        return ScenarioObject(f"s{index}-{suffix}", object_id, product, base_type, InventoryZone.TABLE, ObjectCondition.VERIFIED, pose)

    def _inject(
        self,
        index: int,
        products: tuple[int, ...],
        objects: list[ScenarioObject],
        fault: FaultKind,
    ) -> tuple[list[ScenarioObject], str | None]:
        if fault is FaultKind.CLEAN:
            return objects, None
        if fault in {FaultKind.LOOKALIKE, FaultKind.FOREIGN}:
            trap = self._rng.choice(self._trap_candidates(products, fault))
            owner = self._part_products[trap]
            objects.append(replace(self._object(index, f"trap-{fault.value}", trap, owner, None, False), hazardous=True))
            return objects, None
        target_index = self._rng.choice([i for i, item in enumerate(objects) if item.base_type is None])
        target = objects[target_index]
        if fault is FaultKind.SPARE_SHORTAGE:
            del objects[target_index]
            return objects, target.object_id
        condition = ObjectCondition.DEFECTIVE if fault is FaultKind.DEFECT else ObjectCondition.FALLEN
        grasp_approved = fault is not FaultKind.FALLEN or bool(self._rng.getrandbits(1))
        objects[target_index] = replace(target, condition=condition, grasp_approved=grasp_approved, hazardous=True)
        spare = replace(
            self._object(index, f"spare-{target.object_id}", target.object_id, target.owner_product, None, False),
            zone=InventoryZone.SPARE,
        )
        objects.append(spare)
        return objects, None

    def _trap_candidates(self, products: tuple[int, ...], fault: FaultKind) -> tuple[str, ...]:
        selected = {step.object_id for product in products for step in self._catalog.product(product).steps}
        outside = {part for part, product in self._part_products.items() if product not in products}
        lookalikes = {candidate for part in selected for candidate in self._lookalikes.get(part, ())}
        candidates = outside & lookalikes if fault is FaultKind.LOOKALIKE else outside - lookalikes
        return tuple(sorted(candidates))

    def _random_xy(self) -> tuple[float, float]:
        return (self._rng.uniform(*self._bounds.x_mm), self._rng.uniform(*self._bounds.y_mm))


def _base_ids_by_type(catalog: Catalog) -> dict[BaseType, tuple[str, ...]]:
    grouped: dict[BaseType, list[str]] = {"A": [], "B": [], "C": []}
    for object_id, base_type in catalog._base_types_by_object:
        grouped[base_type].append(object_id)
    if any(not values for values in grouped.values()):
        raise FleetInputError("catalog must provide at least one base for every base type")
    return {key: tuple(values) for key, values in grouped.items()}


def _taxonomy_parts() -> tuple[dict[str, int], dict[str, tuple[str, ...]]]:
    try:
        data = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
        objects = data["objects"]
        products = {item["object_id"]: item["product"] for item in objects if item["object_id"].startswith("M")}
        lookalikes = {item["object_id"]: tuple(item["lookalikes"]) for item in objects if item["object_id"].startswith("M")}
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise FleetInputError(f"cannot load validated taxonomy: {error}") from error
    return products, lookalikes


def _validate_interval(value: object, field: str) -> None:
    numbers = _finite_values(value, field, count=2)
    if numbers[0] >= numbers[1]:
        raise FleetInputError(f"{field} lower bound must be below upper bound")


def _finite_values(value: object, field: str, *, count: int | None = None) -> tuple[float, ...]:
    if not isinstance(value, tuple) or (count is not None and len(value) != count):
        raise FleetInputError(f"{field} must be a tuple with {count} finite values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
        raise FleetInputError(f"{field} must contain finite numbers")
    return tuple(float(item) for item in value)
