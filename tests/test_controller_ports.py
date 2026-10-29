"""Controller perception boundary regressions."""

import pytest

from assembly.catalog import load_catalog
from assembly.controller_ports import (
    ControllerPortError,
    InventorySnapshot,
    ObservedInventoryItem,
    PerceptionTolerances,
    current_waypoint,
    pose_agrees,
    reconcile_inventory,
)
from assembly.inventory import Inventory, InventoryZone, ObjectCondition


class ArrayLike:
    """Minimal NumPy-style container: indexable, but not a Sequence."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> float:
        return self._values[index]


class ZeroDimArray:
    """NumPy-style scalar array whose shape has no sequence length."""

    def __len__(self) -> int:
        raise TypeError("len() of unsized object")

    def __getitem__(self, index: int) -> float:
        raise IndexError(index)


def _reserved_inventory() -> Inventory:
    inventory = Inventory(load_catalog())
    inventory.add("base", "F01")
    for step in range(1, 5):
        inventory.add(f"part-{step}", f"M1_{step:02d}")
    inventory.reserve_product(1)
    return inventory


def _visible_except(*instance_ids: str) -> tuple[ObservedInventoryItem, ...]:
    omitted = set(instance_ids)
    items = [ObservedInventoryItem("base", "F01")]
    items.extend(ObservedInventoryItem(f"part-{step}", f"M1_{step:02d}") for step in range(1, 5))
    return tuple(item for item in items if item.instance_id not in omitted)


def test_complete_scan_does_not_invalidate_occluded_installed_part() -> None:
    inventory = _reserved_inventory()
    inventory.install("part-1")
    snapshot = InventorySnapshot(1.0, True, _visible_except("part-1"))

    reconcile_inventory(inventory, snapshot, 1.0)

    assert inventory.item("part-1").condition is ObjectCondition.VERIFIED


def test_complete_scan_does_not_invalidate_part_known_to_be_in_gripper() -> None:
    inventory = _reserved_inventory()
    snapshot = InventorySnapshot(1.0, True, _visible_except("part-1"))

    reconcile_inventory(inventory, snapshot, 1.0, held_instance_id="part-1")

    assert inventory.item("part-1").condition is ObjectCondition.VERIFIED


def test_claimed_fallen_part_keeps_ownership_for_physical_reject() -> None:
    inventory = _reserved_inventory()
    observed = tuple(
        ObservedInventoryItem(item.instance_id, item.object_id, condition=ObjectCondition.FALLEN)
        if item.instance_id == "part-1"
        else item
        for item in _visible_except()
    )

    reconcile_inventory(inventory, InventorySnapshot(1.0, True, observed), 1.0)

    part = inventory.item("part-1")
    assert part.zone is InventoryZone.RESERVED
    assert part.reserved_for == 1
    assert part.condition is ObjectCondition.FALLEN


def test_current_waypoint_accepts_official_client_array_shape() -> None:
    observation = {
        "tcp_pose": ArrayLike([200.0, 0.0, -250.0, -180.0, 0.0, 0.0]),
        "joints": ArrayLike([0.0] * 7),
    }

    waypoint = current_waypoint(observation)

    assert waypoint.xyz == (200.0, 0.0, -250.0)
    assert waypoint.gripper == 0.0


def test_pose_agreement_uses_measured_distance_and_wrapped_yaw() -> None:
    actual = {"x": 100.2, "y": 50.0, "z": -300.0, "yaw": -179.9}
    expected = (100.0, 50.0, -300.0, 180.0)

    assert pose_agrees(actual, expected, PerceptionTolerances(0.25, 0.2))
    assert not pose_agrees(actual, expected, PerceptionTolerances(0.1, 0.2))


def test_current_waypoint_normalizes_unsized_array_to_port_error() -> None:
    with pytest.raises(ControllerPortError, match="six absolute UF0 values"):
        current_waypoint({"tcp_pose": ZeroDimArray(), "joints": [0.0] * 7})


@pytest.mark.parametrize("value", [[0.0] * 5, [0.0] * 7, "not-a-pose", {"x": 0.0}])
def test_current_waypoint_rejects_malformed_tcp_shapes(value: object) -> None:
    with pytest.raises(ControllerPortError):
        current_waypoint({"tcp_pose": value, "joints": [0.0] * 7})
