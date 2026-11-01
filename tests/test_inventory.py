"""Behavioral tests for observed live inventory and recipe reservation.
Authored by Shritej Gunda.
"""

import pytest

from assembly.catalog import load_catalog


def test_add_tracks_immutable_observed_item_in_exact_domain_enums() -> None:
    from assembly.inventory import Inventory, InventoryZone, ObjectCondition

    inventory = Inventory(load_catalog())
    added = inventory.add(
        "base-c-1",
        "F04",
        zone=InventoryZone.TABLE,
        condition=ObjectCondition.VERIFIED,
    )

    assert {zone.value for zone in InventoryZone} == {
        "table",
        "spare",
        "reserved",
        "installed",
        "staged",
        "output",
        "reject",
    }
    assert {condition.value for condition in ObjectCondition} == {
        "verified",
        "fallen",
        "defective",
        "uncertain",
        "not_observed",
    }
    assert added == inventory.item("base-c-1")
    assert added.instance_id == "base-c-1"
    assert added.object_id == "F04"
    assert added.zone is InventoryZone.TABLE
    assert added.condition is ObjectCondition.VERIFIED
    assert inventory.items(InventoryZone.TABLE) == (added,)


def _product_four_inventory(*, omit: str | None = None):
    from assembly.inventory import Inventory, InventoryZone, ObjectCondition

    inventory = Inventory(load_catalog())
    inventory.add("base-a", "F01")
    inventory.add("base-c", "F05")
    for object_id in ("M4_01", "M4_02", "M4_03", "M4_04"):
        if object_id != omit:
            inventory.add(f"table-{object_id}", object_id)
    inventory.add(
        "spare-M4_02",
        "M4_02",
        zone=InventoryZone.SPARE,
        condition=ObjectCondition.VERIFIED,
    )
    return inventory


def test_reserve_product_uses_compatible_base_and_exact_recipe_order() -> None:
    from assembly.inventory import InventoryZone

    inventory = _product_four_inventory()
    reservation = inventory.reserve_product(4)

    assert reservation.product == 4
    assert reservation.base.instance_id == "base-c"
    assert reservation.base.object_id == "F05"
    assert [(part.object_id, part.instance_id) for part in reservation.parts] == [
        ("M4_01", "table-M4_01"),
        ("M4_02", "table-M4_02"),
        ("M4_03", "table-M4_03"),
        ("M4_04", "table-M4_04"),
    ]
    assert all(item.zone is InventoryZone.RESERVED for item in reservation.items)
    assert all(item.reserved_for == 4 for item in reservation.items)
    assert inventory.item("base-a").zone is InventoryZone.TABLE
    assert inventory.active_product == 4


def test_unusable_table_part_is_skipped_for_verified_spare() -> None:
    from assembly.inventory import InventoryZone, ObjectCondition

    inventory = _product_four_inventory()
    inventory.set_condition("table-M4_02", ObjectCondition.FALLEN)

    reservation = inventory.reserve_product(4)

    assert reservation.part("M4_02").instance_id == "spare-M4_02"
    assert reservation.part("M4_02").source_zone is InventoryZone.SPARE
    assert inventory.item("table-M4_02").zone is InventoryZone.TABLE
    assert inventory.item("table-M4_02").condition is ObjectCondition.FALLEN


@pytest.mark.parametrize(
    "condition_name",
    ["fallen", "defective", "uncertain"],
)
def test_unusable_objects_are_never_reservable(condition_name: str) -> None:
    from assembly.inventory import InventoryShortageError, ObjectCondition

    inventory = _product_four_inventory()
    inventory.set_condition("table-M4_03", ObjectCondition(condition_name))

    with pytest.raises(InventoryShortageError) as caught:
        inventory.reserve_product(4)

    assert caught.value.required_object_id == "M4_03"
    assert inventory.item("table-M4_03").reserved_for is None


def test_wrong_lookalike_never_substitutes_for_missing_exact_recipe_part() -> None:
    from assembly.inventory import InventoryShortageError, InventoryZone

    inventory = _product_four_inventory(omit="M4_04")
    inventory.add("wrong-lock", "M5_04", zone=InventoryZone.SPARE)

    with pytest.raises(InventoryShortageError) as caught:
        inventory.reserve_product(4)

    assert caught.value.required_object_id == "M4_04"
    assert inventory.item("wrong-lock").zone is InventoryZone.SPARE


def test_shortage_is_typed_andon_signal_and_reservation_is_atomic() -> None:
    from assembly.inventory import InventoryShortageError, InventoryZone

    inventory = _product_four_inventory(omit="M4_03")

    with pytest.raises(InventoryShortageError) as caught:
        inventory.reserve_product(4)

    shortage = caught.value
    assert shortage.product == 4
    assert shortage.required_object_id == "M4_03"
    assert shortage.required_base_type is None
    assert shortage.recommended_state == "Suspended"
    assert shortage.andon_required is True
    assert inventory.active_product is None
    assert inventory.items(InventoryZone.RESERVED) == ()


def test_missing_compatible_base_reports_required_type_without_reserving_parts() -> None:
    from assembly.inventory import Inventory, InventoryShortageError, InventoryZone

    inventory = Inventory(load_catalog())
    inventory.add("base-a", "F01")
    for object_id in ("M4_01", "M4_02", "M4_03", "M4_04"):
        inventory.add(f"table-{object_id}", object_id)

    with pytest.raises(InventoryShortageError) as caught:
        inventory.reserve_product(4)

    assert caught.value.required_base_type == "C"
    assert caught.value.required_object_id is None
    assert inventory.items(InventoryZone.RESERVED) == ()


def test_only_one_product_can_own_reservations_at_a_time() -> None:
    from assembly.inventory import InventoryOwnershipError

    inventory = _product_four_inventory()
    inventory.reserve_product(4)

    with pytest.raises(InventoryOwnershipError) as caught:
        inventory.reserve_product(5)

    assert caught.value.owner == 4
    assert caught.value.requested == 5


def test_release_restores_each_reserved_item_to_its_source_zone() -> None:
    from assembly.inventory import InventoryZone, ObjectCondition

    inventory = _product_four_inventory()
    inventory.set_condition("table-M4_02", ObjectCondition.DEFECTIVE)
    inventory.reserve_product(4)

    released = inventory.release_reservations(4)

    assert set(released) == {
        "base-c",
        "table-M4_01",
        "spare-M4_02",
        "table-M4_03",
        "table-M4_04",
    }
    assert inventory.item("base-c").zone is InventoryZone.TABLE
    assert inventory.item("spare-M4_02").zone is InventoryZone.SPARE
    assert inventory.active_product is None


def test_install_and_reject_update_live_zones_without_releasing_installed_part() -> None:
    from assembly.inventory import InventoryZone, ObjectCondition

    inventory = _product_four_inventory()
    inventory.reserve_product(4)

    installed = inventory.install("table-M4_01")
    rejected = inventory.reject("table-M4_02", condition=ObjectCondition.DEFECTIVE)

    assert installed.zone is InventoryZone.INSTALLED
    assert inventory.item("table-M4_01").zone is InventoryZone.INSTALLED
    assert rejected.zone is InventoryZone.REJECT
    assert rejected.condition is ObjectCondition.DEFECTIVE
    assert inventory.active_reservation.part("M4_01").zone is InventoryZone.INSTALLED
    assert inventory.active_reservation.part("M4_02").zone is InventoryZone.REJECT
    inventory.release_reservations(4)


def test_replace_rejected_reservation_with_verified_exact_spare() -> None:
    from assembly.inventory import InventoryZone, ObjectCondition

    inventory = _product_four_inventory()
    inventory.reserve_product(4)

    replacement = inventory.replace_with_spare(
        "table-M4_02",
        condition=ObjectCondition.DEFECTIVE,
    )

    assert inventory.item("table-M4_02").zone is InventoryZone.REJECT
    assert replacement.instance_id == "spare-M4_02"
    assert replacement.object_id == "M4_02"
    assert replacement.zone is InventoryZone.RESERVED
    assert replacement.source_zone is InventoryZone.SPARE
    assert inventory.active_reservation.part("M4_02") == replacement


def test_replace_rejected_part_uses_verified_table_instance_before_spare() -> None:
    from assembly.inventory import InventoryZone, ObjectCondition

    inventory = _product_four_inventory()
    inventory.reserve_product(4)
    inventory.add("second-table-M4_02", "M4_02", zone=InventoryZone.TABLE)

    replacement = inventory.replace_with_spare("table-M4_02", condition=ObjectCondition.DEFECTIVE)

    assert replacement.instance_id == "second-table-M4_02"
    assert replacement.source_zone is InventoryZone.TABLE
    assert inventory.item("spare-M4_02").zone is InventoryZone.SPARE


def test_replacement_shortage_rejects_bad_part_and_can_resume_after_restock() -> None:
    from assembly.inventory import InventoryShortageError, InventoryZone, ObjectCondition

    inventory = _product_four_inventory()
    inventory.set_condition("spare-M4_02", ObjectCondition.UNCERTAIN)
    inventory.reserve_product(4)

    with pytest.raises(InventoryShortageError) as caught:
        inventory.replace_with_spare("table-M4_02", condition=ObjectCondition.FALLEN)

    assert caught.value.required_object_id == "M4_02"
    assert inventory.item("table-M4_02").zone is InventoryZone.REJECT
    assert inventory.active_reservation.part("M4_02").zone is InventoryZone.REJECT
    inventory.add("restocked-M4_02", "M4_02", zone=InventoryZone.SPARE)
    replacement = inventory.replace_with_spare("table-M4_02")
    assert replacement.instance_id == "restocked-M4_02"


def test_duplicate_unknown_and_illegal_initial_objects_fail_closed() -> None:
    from assembly.inventory import (
        DuplicateInstanceError,
        InvalidInventoryMoveError,
        InvalidInventoryObjectError,
        Inventory,
        InventoryZone,
        UnknownInventoryItemError,
    )

    inventory = Inventory(load_catalog())
    inventory.add("seen-1", "M1_01")

    with pytest.raises(DuplicateInstanceError):
        inventory.add("seen-1", "M1_02")
    with pytest.raises(InvalidInventoryObjectError):
        inventory.add("unknown-object", "sim_ground_truth_part")
    with pytest.raises(InvalidInventoryMoveError):
        inventory.add("pre-reserved", "M1_02", zone=InventoryZone.RESERVED)
    with pytest.raises(UnknownInventoryItemError):
        inventory.item("not-seen")

    assert inventory.item("seen-1").object_id == "M1_01"


def test_completion_consumes_base_and_prevents_same_family_reuse() -> None:
    from assembly.inventory import Inventory, InventoryZone

    inventory = Inventory(load_catalog())
    inventory.add("base-4", "F04")
    inventory.add("base-5", "F05")
    inventory.add("base-6", "F06")
    for product in (4, 5, 6):
        for step in range(1, 5):
            inventory.add(f"p{product}-{step}", f"M{product}_{step:02d}")

    first = inventory.reserve_product(4)
    for part in first.parts:
        inventory.install(part.instance_id)
    completed = inventory.complete_product(4)

    assert completed.base.instance_id == "base-4"
    assert inventory.item("base-4").zone is InventoryZone.OUTPUT
    assert inventory.active_product is None
    second = inventory.reserve_product(5)
    assert second.base.instance_id == "base-5"
    for part in second.parts:
        inventory.install(part.instance_id)
    inventory.complete_product(5)
    third = inventory.reserve_product(6)
    assert third.base.instance_id == "base-6"
    assert inventory.item("base-4").zone is InventoryZone.OUTPUT
    assert inventory.item("base-5").zone is InventoryZone.OUTPUT


def test_completion_requires_every_recipe_part_installed() -> None:
    from assembly.inventory import IncompleteProductError, InventoryZone

    inventory = _product_four_inventory()
    reservation = inventory.reserve_product(4)
    inventory.install(reservation.parts[0].instance_id)

    with pytest.raises(IncompleteProductError) as caught:
        inventory.complete_product(4)

    assert caught.value.product == 4
    assert inventory.active_product == 4
    assert inventory.item("base-c").zone is InventoryZone.RESERVED


def test_live_observation_updates_condition_without_losing_claimed_zone() -> None:
    from assembly.inventory import InventoryZone, ObjectCondition

    inventory = _product_four_inventory()
    reservation = inventory.reserve_product(4)

    updated = inventory.reconcile_observation(
        reservation.base.instance_id,
        reservation.base.object_id,
        zone=InventoryZone.TABLE,
        condition=ObjectCondition.FALLEN,
    )

    assert updated.zone is InventoryZone.RESERVED
    assert updated.condition is ObjectCondition.FALLEN
    assert updated.reserved_for == 4
    assert inventory.active_reservation.base == updated


def test_live_observation_can_move_unclaimed_item_between_table_and_spare() -> None:
    from assembly.inventory import InventoryZone, ObjectCondition

    inventory = _product_four_inventory()
    updated = inventory.reconcile_observation(
        "table-M4_01",
        "M4_01",
        zone=InventoryZone.SPARE,
        condition=ObjectCondition.UNCERTAIN,
    )

    assert updated.zone is InventoryZone.SPARE
    assert updated.condition is ObjectCondition.UNCERTAIN


def test_live_observation_rejects_identity_change_and_marks_absence() -> None:
    from assembly.inventory import (
        InvalidInventoryObservationError,
        ObjectCondition,
    )

    inventory = _product_four_inventory()
    inventory.reserve_product(4)

    with pytest.raises(InvalidInventoryObservationError, match="changed identity"):
        inventory.reconcile_observation("table-M4_01", "M5_01")

    missing = inventory.mark_unobserved("base-c")
    assert missing.condition is ObjectCondition.NOT_OBSERVED
    assert inventory.active_reservation.base == missing


def test_recovery_staging_tracks_physical_state_and_blocks_reset_release() -> None:
    from assembly.inventory import InvalidInventoryMoveError, InventoryZone

    inventory = _product_four_inventory()
    reservation = inventory.reserve_product(4)
    installed = inventory.install(reservation.parts[0].instance_id)

    staged = inventory.stage_for_rebuild(installed.instance_id)

    assert staged.zone is InventoryZone.STAGED
    with pytest.raises(InvalidInventoryMoveError, match="operator reconciliation"):
        inventory.release_reservations(4)
    assert inventory.reinstall(staged.instance_id).zone is InventoryZone.INSTALLED
