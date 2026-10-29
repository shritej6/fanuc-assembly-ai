"""Deterministic Kanban and defect-Pareto view models.

T09 inventory is the sole stock authority. Controller events can trigger a
card or contribute a validated defect occurrence, but they never mutate or
decrement stock. The immutable output is intentionally chart-renderer agnostic.
"""

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from assembly.catalog import Catalog
from assembly.controller_ports import ControllerEvent
from assembly.inventory import Inventory, InventoryZone, ObjectCondition

_KNOWN_EVENT_KINDS = frozenset(
    {
        "abort",
        "andon",
        "complete",
        "defect",
        "installed",
        "motion",
        "rebuild",
        "rebuild_installed",
        "rejected",
        "reject_failed",
        "replacement",
        "resume",
        "retry",
        "spare",
        "stage_installed",
    }
)
_OBJECT_REQUIRED_KINDS = frozenset(
    {
        "defect",
        "installed",
        "motion",
        "rebuild_installed",
        "rejected",
        "reject_failed",
        "replacement",
        "retry",
        "spare",
        "stage_installed",
    }
)
_OBJECT_OPTIONAL_KINDS = frozenset({"rebuild"})
_OBJECT_FORBIDDEN_KINDS = _KNOWN_EVENT_KINDS - _OBJECT_REQUIRED_KINDS - _OBJECT_OPTIONAL_KINDS
_PRODUCT_REQUIRED_KINDS = _OBJECT_REQUIRED_KINDS | _OBJECT_OPTIONAL_KINDS | frozenset({"complete", "resume"})
_SPARE_USE_EVENT_KINDS = frozenset({"reject_failed", "spare"})


class KanbanInputError(ValueError):
    """An inventory, event, or policy boundary is malformed."""


@dataclass(frozen=True, slots=True)
class FailureRateEvidence:
    """Measured counts behind a rate that must remain PRELIMINARY."""

    failures: int
    opportunities: int
    source: str
    rate: float = field(init=False)
    classification: str = field(init=False, default="PRELIMINARY")

    def __post_init__(self) -> None:
        failures = _count(self.failures, "failures")
        opportunities = _count(self.opportunities, "opportunities")
        if opportunities == 0:
            raise KanbanInputError("opportunities must be positive")
        if failures > opportunities:
            raise KanbanInputError("failures cannot exceed opportunities")
        _nonempty(self.source, "source")
        object.__setattr__(self, "rate", failures / opportunities)


@dataclass(frozen=True, slots=True)
class StockPolicy:
    """Operator-supplied replenishment thresholds for one catalog part."""

    object_id: str
    reorder_point: int
    target_stock: int
    failure_rate: FailureRateEvidence

    def __post_init__(self) -> None:
        _nonempty(self.object_id, "object_id")
        reorder_point = _count(self.reorder_point, "reorder_point")
        target_stock = _count(self.target_stock, "target_stock")
        if target_stock <= reorder_point:
            raise KanbanInputError("target_stock must exceed reorder_point")
        if type(self.failure_rate) is not FailureRateEvidence:
            raise KanbanInputError("failure_rate must be measured FailureRateEvidence")


class ReorderTrigger(StrEnum):
    SPARE_USE = "spare_use"
    LOW_STOCK = "low_stock"
    SPARE_USE_AND_LOW_STOCK = "spare_use_and_low_stock"


class DefectCause(StrEnum):
    CRACK_AT_LAYER_LINE = "crack_at_layer_line"
    JAM = "jam"
    SLIP = "slip"
    EXTRACTION_FAILURE = "extraction_failure"
    FALLEN = "fallen"
    BROKEN_FEATURE = "broken_feature"
    MISSING_FEATURE = "missing_feature"
    PRINT_DEFECT = "print_defect"
    MISSING_LAYER = "missing_layer"
    WRONG_PART = "wrong_part"
    FOREIGN_OBJECT = "foreign_object"
    INSPECTION_REJECT = "inspection_reject"
    INSERTION_RETRY = "insertion_retry"
    INSERTION_FAILURE = "insertion_failure"


@dataclass(frozen=True, slots=True)
class ReorderCard:
    """A chart-ready replenishment request derived from live spare stock."""

    object_id: str
    available_spares: int
    reorder_point: int
    target_stock: int
    order_quantity: int
    observed_spare_uses: int
    trigger: ReorderTrigger
    failure_rate: FailureRateEvidence


@dataclass(frozen=True, slots=True)
class ParetoRow:
    """One ranked part/cause occurrence count and its measured shares."""

    object_id: str
    cause: DefectCause
    count: int
    share: float
    cumulative_share: float


@dataclass(frozen=True, slots=True)
class DefectPareto:
    """Ranked defect occurrences; rows are empty when no defects were logged."""

    total_defects: int
    rows: tuple[ParetoRow, ...]


@dataclass(frozen=True, slots=True)
class OperationsView:
    """Immutable data model shared with the separately owned chart layer."""

    source_event_count: int
    cards: tuple[ReorderCard, ...]
    pareto: DefectPareto


def build_operations_view(
    catalog: Catalog,
    inventory: Inventory,
    events: tuple[ControllerEvent, ...],
    policies: tuple[StockPolicy, ...],
) -> OperationsView:
    """Build cards and Pareto rows from validated T09 state without mutation."""

    _validate_view_boundary(catalog, inventory, events, policies)
    validated = _validate_policies(catalog, policies)
    validated_events = _validate_events(catalog, events)
    _require_spare_policies(validated_events, validated)
    cards = _build_cards(inventory, validated_events, validated)
    pareto = _build_pareto(inventory, validated_events)
    return OperationsView(len(validated_events), cards, pareto)


def _validate_view_boundary(catalog: object, inventory: object, events: object, policies: object) -> None:
    if type(catalog) is not Catalog:
        raise KanbanInputError("catalog must be a validated Catalog")
    if type(inventory) is not Inventory:
        raise KanbanInputError("inventory must be a validated Inventory")
    if not isinstance(events, tuple):
        raise KanbanInputError("events must be an immutable tuple")
    if not isinstance(policies, tuple):
        raise KanbanInputError("policies must be an immutable tuple")


def _validate_policies(catalog: Catalog, policies: tuple[StockPolicy, ...]) -> tuple[StockPolicy, ...]:
    part_ids = {step.object_id for product in catalog.products for step in product.steps}
    seen: set[str] = set()
    for policy in policies:
        if type(policy) is not StockPolicy:
            raise KanbanInputError("policies must contain only StockPolicy values")
        if policy.object_id not in part_ids:
            raise KanbanInputError(f"stock policy object {policy.object_id!r} is not a catalog part")
        if policy.object_id in seen:
            raise KanbanInputError(f"duplicate stock policy for {policy.object_id!r}")
        seen.add(policy.object_id)
    return policies


def _validate_events(catalog: Catalog, events: object) -> tuple[ControllerEvent, ...]:
    if not isinstance(events, tuple):
        raise KanbanInputError("events must be an immutable tuple")
    owners = {step.object_id: product.product for product in catalog.products for step in product.steps}
    product_ids = {product.product for product in catalog.products}
    for event in events:
        _validate_event(event, owners, product_ids)
    return events


def _validate_event(event: object, owners: dict[str, int], product_ids: set[int]) -> None:
    if type(event) is not ControllerEvent:
        raise KanbanInputError("events must contain only ControllerEvent values")
    kind = _nonempty(event.kind, "event.kind")
    if kind not in _KNOWN_EVENT_KINDS:
        raise KanbanInputError(f"event kind {kind!r} is not registered")
    _validate_event_product(kind, event.product, product_ids)
    _validate_event_object(kind, event.object_id, event.product, owners)
    detail = _nonempty(event.detail, "event.detail")
    if kind == "defect":
        try:
            DefectCause(detail)
        except ValueError as error:
            raise KanbanInputError(f"defect cause {detail!r} is not registered") from error
    if kind == "retry" and detail not in {"1", "2"}:
        raise KanbanInputError("retry detail must be bounded retry number '1' or '2'")


def _validate_event_product(kind: str, product: object, product_ids: set[int]) -> None:
    if product is None:
        if kind in _PRODUCT_REQUIRED_KINDS:
            raise KanbanInputError(f"event kind {kind!r} requires a catalog product")
        return
    if isinstance(product, bool) or not isinstance(product, int) or product not in product_ids:
        raise KanbanInputError(f"event product {product!r} is not in the catalog")


def _validate_event_object(kind: str, object_id: object, product: int | None, owners: dict[str, int]) -> None:
    if object_id is None:
        if kind in _OBJECT_REQUIRED_KINDS:
            raise KanbanInputError(f"event kind {kind!r} requires a catalog part")
        return
    if kind in _OBJECT_FORBIDDEN_KINDS:
        raise KanbanInputError(f"event kind {kind!r} forbids a catalog part")
    if not isinstance(object_id, str) or object_id not in owners:
        raise KanbanInputError(f"event object {object_id!r} is not a catalog part")
    if product is None or owners[object_id] != product:
        raise KanbanInputError(f"event object {object_id!r} does not belong to product {product!r}")


def _require_spare_policies(
    events: tuple[ControllerEvent, ...],
    policies: tuple[StockPolicy, ...],
) -> None:
    covered = {policy.object_id for policy in policies}
    consumed = {event.object_id for event in events if event.kind in _SPARE_USE_EVENT_KINDS}
    missing = sorted(object_id for object_id in consumed if object_id not in covered)
    if missing:
        raise KanbanInputError(f"spare use has no stock policy: {', '.join(missing)}")


def _build_pareto(inventory: Inventory, events: tuple[ControllerEvent, ...]) -> DefectPareto:
    counts = Counter(
        (item.object_id, _rejected_cause(item.condition)) for item in inventory.items(InventoryZone.REJECT)
    )
    for event in events:
        cause = _cause_for_event(event)
        if cause is not None and event.object_id is not None:
            _refine_inventory_reject(counts, event.object_id, cause)
            counts[(event.object_id, cause)] += 1
    total = sum(counts.values())
    if total == 0:
        return DefectPareto(0, ())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1].value))
    rows: list[ParetoRow] = []
    cumulative = 0
    for (object_id, cause), count in ordered:
        cumulative += count
        rows.append(ParetoRow(object_id, cause, count, count / total, cumulative / total))
    return DefectPareto(total, tuple(rows))


def _rejected_cause(condition: ObjectCondition) -> DefectCause:
    if condition is ObjectCondition.FALLEN:
        return DefectCause.FALLEN
    return DefectCause.INSPECTION_REJECT


def _refine_inventory_reject(
    counts: Counter[tuple[str, DefectCause]],
    object_id: str,
    cause: DefectCause,
) -> None:
    key = (object_id, DefectCause.INSPECTION_REJECT)
    if cause is not DefectCause.INSERTION_FAILURE or counts.get(key, 0) == 0:
        return
    counts[key] -= 1
    if counts[key] == 0:
        del counts[key]


def _cause_for_event(event: ControllerEvent) -> DefectCause | None:
    if event.kind == "defect":
        return DefectCause(event.detail)
    if event.kind == "retry":
        return DefectCause.INSERTION_RETRY
    if event.kind == "reject_failed":
        return DefectCause.INSERTION_FAILURE
    return None


def _build_cards(
    inventory: Inventory,
    events: tuple[ControllerEvent, ...],
    policies: tuple[StockPolicy, ...],
) -> tuple[ReorderCard, ...]:
    available = Counter(
        item.object_id
        for item in inventory.items()
        if item.zone is InventoryZone.SPARE and item.condition is ObjectCondition.VERIFIED
    )
    uses = Counter(event.object_id for event in events if event.kind in _SPARE_USE_EVENT_KINDS)
    cards: list[ReorderCard] = []
    for policy in sorted(policies, key=lambda item: item.object_id):
        count = available[policy.object_id]
        used = uses[policy.object_id]
        low = count <= policy.reorder_point
        if not low and (not used or count >= policy.target_stock):
            continue
        trigger = _reorder_trigger(used, low)
        cards.append(
            ReorderCard(
                policy.object_id,
                count,
                policy.reorder_point,
                policy.target_stock,
                policy.target_stock - count,
                used,
                trigger,
                policy.failure_rate,
            )
        )
    return tuple(cards)


def _reorder_trigger(observed_uses: int, low_stock: bool) -> ReorderTrigger:
    if observed_uses and low_stock:
        return ReorderTrigger.SPARE_USE_AND_LOW_STOCK
    if observed_uses:
        return ReorderTrigger.SPARE_USE
    return ReorderTrigger.LOW_STOCK


def _count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise KanbanInputError(f"{field} must be a nonnegative integer")
    return value


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KanbanInputError(f"{field} must be a non-empty string")
    return value
