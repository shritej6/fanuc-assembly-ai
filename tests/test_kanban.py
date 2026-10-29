"""Behavioral tests for deterministic replenishment and defect analytics."""

from dataclasses import asdict

import pytest

from assembly.catalog import load_catalog
from assembly.controller_ports import ControllerEvent
from assembly.inventory import Inventory, InventoryZone, ObjectCondition
from assembly.kanban import (
    DefectCause,
    FailureRateEvidence,
    KanbanInputError,
    ParetoRow,
    ReorderCard,
    ReorderTrigger,
    StockPolicy,
    build_operations_view,
)


def _rate(*, failures: int = 1, opportunities: int = 10) -> FailureRateEvidence:
    return FailureRateEvidence(failures, opportunities, "cycle-test-U05")


def _spares(object_id: str, count: int) -> Inventory:
    inventory = Inventory(load_catalog())
    for index in range(count):
        inventory.add(f"spare-{object_id}-{index}", object_id, zone=InventoryZone.SPARE)
    return inventory


def test_spare_use_raises_card_from_only_verified_spare_stock() -> None:
    inventory = Inventory(load_catalog())
    inventory.add("available-1", "M1_01", zone=InventoryZone.SPARE)
    inventory.add("available-2", "M1_01", zone=InventoryZone.SPARE)
    inventory.add(
        "fallen-spare",
        "M1_01",
        zone=InventoryZone.SPARE,
        condition=ObjectCondition.FALLEN,
    )
    inventory.add("table-part", "M1_01", zone=InventoryZone.TABLE)
    rate = FailureRateEvidence(failures=2, opportunities=25, source="cycle-test-U05")
    policy = StockPolicy("M1_01", reorder_point=2, target_stock=4, failure_rate=rate)
    events = (ControllerEvent("spare", 1, "M1_01", "used-spare"),)

    view = build_operations_view(load_catalog(), inventory, events, (policy,))

    assert view.source_event_count == 1
    assert view.cards == (
        ReorderCard(
            object_id="M1_01",
            available_spares=2,
            reorder_point=2,
            target_stock=4,
            order_quantity=2,
            observed_spare_uses=1,
            trigger=ReorderTrigger.SPARE_USE_AND_LOW_STOCK,
            failure_rate=rate,
        ),
    )
    assert view.pareto.total_defects == 0
    assert view.pareto.rows == ()


def test_replayed_spare_event_never_double_decrements_authoritative_inventory() -> None:
    inventory = _spares("M1_01", 2)
    rate = _rate()
    policy = StockPolicy("M1_01", reorder_point=2, target_stock=4, failure_rate=rate)
    replayed = ControllerEvent("spare", 1, "M1_01", "same-instance")

    view = build_operations_view(load_catalog(), inventory, (replayed, replayed), (policy,))

    assert view.cards == (
        ReorderCard("M1_01", 2, 2, 4, 2, 2, ReorderTrigger.SPARE_USE_AND_LOW_STOCK, rate),
    )
    assert len(inventory.items(InventoryZone.SPARE)) == 2


def test_spare_use_replenishes_to_target_before_low_stock_point() -> None:
    inventory = _spares("M2_02", 3)
    policy = StockPolicy("M2_02", reorder_point=1, target_stock=4, failure_rate=_rate())

    view = build_operations_view(
        load_catalog(),
        inventory,
        (ControllerEvent("spare", 2, "M2_02", "used-spare"),),
        (policy,),
    )

    assert len(view.cards) == 1
    assert view.cards[0].trigger is ReorderTrigger.SPARE_USE
    assert view.cards[0].order_quantity == 1


def test_low_stock_without_spare_event_still_raises_reorder_card() -> None:
    inventory = _spares("M3_03", 1)
    policy = StockPolicy("M3_03", reorder_point=1, target_stock=3, failure_rate=_rate())

    view = build_operations_view(load_catalog(), inventory, (), (policy,))

    assert len(view.cards) == 1
    assert view.cards[0].trigger is ReorderTrigger.LOW_STOCK
    assert view.cards[0].observed_spare_uses == 0
    assert view.cards[0].order_quantity == 2


def test_surplus_stock_does_not_raise_card_even_after_recorded_use() -> None:
    inventory = _spares("M4_04", 5)
    policy = StockPolicy("M4_04", reorder_point=1, target_stock=4, failure_rate=_rate())
    events = (ControllerEvent("spare", 4, "M4_04", "used-spare"),)

    view = build_operations_view(load_catalog(), inventory, events, (policy,))

    assert view.cards == ()


def test_reorder_cards_are_stably_sorted_by_object_id() -> None:
    inventory = Inventory(load_catalog())
    policies = (
        StockPolicy("M6_04", 0, 2, _rate()),
        StockPolicy("M1_01", 0, 2, _rate()),
    )

    view = build_operations_view(load_catalog(), inventory, (), policies)

    assert [card.object_id for card in view.cards] == ["M1_01", "M6_04"]


def test_failure_rate_is_derived_from_observed_counts_and_labeled_preliminary() -> None:
    evidence = FailureRateEvidence(2, 25, "cycle-test-U05")

    assert evidence.rate == 0.08
    assert evidence.classification == "PRELIMINARY"
    assert asdict(evidence) == {
        "failures": 2,
        "opportunities": 25,
        "source": "cycle-test-U05",
        "rate": 0.08,
        "classification": "PRELIMINARY",
    }


@pytest.mark.parametrize(
    ("failures", "opportunities", "source"),
    [
        (-1, 10, "cycle-test"),
        (1, 0, "cycle-test"),
        (11, 10, "cycle-test"),
        (True, 10, "cycle-test"),
        (1, True, "cycle-test"),
        (1, 10, ""),
        (1, 10, "   "),
    ],
)
def test_failure_rate_rejects_unmeasured_or_impossible_evidence(
    failures: object,
    opportunities: object,
    source: object,
) -> None:
    with pytest.raises(KanbanInputError):
        FailureRateEvidence(failures, opportunities, source)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("object_id", "reorder_point", "target_stock", "failure_rate"),
    [
        ("", 1, 2, _rate()),
        ("M1_01", -1, 2, _rate()),
        ("M1_01", True, 2, _rate()),
        ("M1_01", 2, 2, _rate()),
        ("M1_01", 2, 1, _rate()),
        ("M1_01", 1, True, _rate()),
        ("M1_01", 1, 2, object()),
    ],
)
def test_stock_policy_rejects_malformed_thresholds_or_rate_evidence(
    object_id: object,
    reorder_point: object,
    target_stock: object,
    failure_rate: object,
) -> None:
    with pytest.raises(KanbanInputError):
        StockPolicy(  # type: ignore[arg-type]
            object_id,
            reorder_point,
            target_stock,
            failure_rate,
        )


@pytest.mark.parametrize(
    "policies",
    [
        (StockPolicy("M1_01", 0, 2, _rate()), StockPolicy("M1_01", 1, 3, _rate())),
        (StockPolicy("not-in-catalog", 0, 2, _rate()),),
        (StockPolicy("F01", 0, 2, _rate()),),
    ],
)
def test_view_rejects_duplicate_unknown_or_fixture_stock_policies(
    policies: tuple[StockPolicy, ...],
) -> None:
    with pytest.raises(KanbanInputError):
        build_operations_view(load_catalog(), Inventory(load_catalog()), (), policies)


def test_pareto_counts_explicit_defects_and_t09_insertion_failures() -> None:
    events = (
        ControllerEvent("defect", 1, "M1_01", "jam"),
        ControllerEvent("retry", 2, "M2_02", "1"),
        ControllerEvent("defect", 1, "M1_01", "crack_at_layer_line"),
        ControllerEvent("reject_failed", 2, "M2_02", "rejected-instance"),
        ControllerEvent("defect", 1, "M1_01", "jam"),
        ControllerEvent("spare", 1, "M1_01", "replacement-instance"),
    )

    inventory = _spares("M1_01", 2)
    policy = StockPolicy("M1_01", 0, 2, _rate())
    view = build_operations_view(load_catalog(), inventory, events, (policy,))

    assert view.pareto.total_defects == 5
    assert view.pareto.rows == (
        ParetoRow("M1_01", DefectCause.JAM, 2, 0.4, 0.4),
        ParetoRow("M1_01", DefectCause.CRACK_AT_LAYER_LINE, 1, 0.2, 0.6),
        ParetoRow("M2_02", DefectCause.INSERTION_FAILURE, 1, 0.2, 0.8),
        ParetoRow("M2_02", DefectCause.INSERTION_RETRY, 1, 0.2, 1.0),
    )


def test_andon_free_text_is_not_misreported_as_a_defect_cause() -> None:
    events = (ControllerEvent("andon", 1, None, "crack_at_layer_line"),)

    view = build_operations_view(load_catalog(), Inventory(load_catalog()), events, ())

    assert view.pareto.total_defects == 0
    assert view.pareto.rows == ()


@pytest.mark.parametrize(
    "events",
    [
        [ControllerEvent("complete", 1, None, "four verified steps")],
        (object(),),
        (ControllerEvent("", 1, None, "detail"),),
        (ControllerEvent("future_kind", 1, None, "detail"),),
        (ControllerEvent("spare", 1, None, "replacement"),),
        (ControllerEvent("retry", 1, "F01", "1"),),
        (ControllerEvent("spare", 7, "M1_01", "replacement"),),
        (ControllerEvent("spare", True, "M1_01", "replacement"),),
        (ControllerEvent("defect", None, "M1_01", "jam"),),
        (ControllerEvent("spare", 1, 42, "replacement"),),
        (ControllerEvent("spare", 1, "M1_01", ""),),
        (ControllerEvent("spare", 1, "M1_01", 42),),
        (ControllerEvent("defect", 1, "M1_01", "unregistered_cause"),),
        (ControllerEvent("retry", 1, "M1_01", "3"),),
        (ControllerEvent("retry", 1, "M1_01", "first"),),
    ],
)
def test_event_boundary_rejects_malformed_or_unregistered_records(events: object) -> None:
    with pytest.raises(KanbanInputError):
        build_operations_view(  # type: ignore[arg-type]
            load_catalog(),
            Inventory(load_catalog()),
            events,
            (),
        )


def test_non_defect_t09_events_remain_valid_and_do_not_inflate_pareto() -> None:
    events = (
        ControllerEvent("motion", 1, "M1_01", "guarded"),
        ControllerEvent("installed", 1, "M1_01", "verified"),
        ControllerEvent("complete", 1, None, "four verified steps"),
        ControllerEvent("resume", 1, None, "inventory restored"),
        ControllerEvent("rebuild", 1, None, "verified complete"),
        ControllerEvent("stage_installed", 1, "M1_01", "instance"),
        ControllerEvent("rebuild_installed", 1, "M1_01", "instance"),
        ControllerEvent("abort", 1, None, "guard failure"),
    )

    view = build_operations_view(load_catalog(), Inventory(load_catalog()), events, ())

    assert view.source_event_count == 8
    assert view.pareto.total_defects == 0


def test_spare_use_requires_a_stock_policy_for_the_consumed_part() -> None:
    events = (ControllerEvent("spare", 5, "M5_04", "replacement"),)

    with pytest.raises(KanbanInputError, match="M5_04"):
        build_operations_view(load_catalog(), Inventory(load_catalog()), events, ())


@pytest.mark.parametrize(
    ("catalog", "inventory", "events", "policies"),
    [
        (object(), Inventory(load_catalog()), (), ()),
        (load_catalog(), object(), (), ()),
        (load_catalog(), Inventory(load_catalog()), (), []),
    ],
)
def test_view_boundary_requires_typed_catalog_inventory_and_immutable_policies(
    catalog: object,
    inventory: object,
    events: object,
    policies: object,
) -> None:
    with pytest.raises(KanbanInputError):
        build_operations_view(  # type: ignore[arg-type]
            catalog,
            inventory,
            events,
            policies,
        )
