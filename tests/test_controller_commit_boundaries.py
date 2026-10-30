"""Physical-to-inventory commit boundary regressions."""

from collections import Counter
from dataclasses import replace

from assembly.inventory import InventoryZone, ObjectCondition
from assembly.state_machine import PackMLState
from robot.adapter import CellAdapter
from tests.test_controller import ScriptedPerception, _controller
from tests.test_controller_fail_closed import _apply, _drive_until, _product_one_items


def _drive_commit(controller, perception, observation) -> None:
    perception.observed = tuple(item for item in perception.observed if item.instance_id != "p1-1")
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    for _ in range(3):
        _apply(observation, adapter.act(observation))
        if controller.state_machine.state is PackMLState.HELD:
            break


def test_missing_part_after_release_is_not_committed_as_verified() -> None:
    perception = ScriptedPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    observation = _drive_until(controller, lambda: controller._release_dispatched)
    _drive_commit(controller, perception, observation)

    item = controller.inventory.item("p1-1")
    assert controller.state_machine.state is PackMLState.HELD
    assert item.zone is InventoryZone.RESERVED
    assert item.condition is ObjectCondition.NOT_OBSERVED
    assert not any(event.kind == "installed" for event in controller.events)
    assert "operator reconciliation" in controller.events[-1].detail


def test_fallen_base_after_release_holds_before_inventory_commit() -> None:
    perception = ScriptedPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    observation = _drive_until(controller, lambda: controller._release_dispatched)
    perception.observed = tuple(
        replace(item, condition=ObjectCondition.FALLEN) if item.instance_id == "base-1" else item
        for item in perception.observed
    )
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    for _ in range(3):
        _apply(observation, adapter.act(observation))
        if controller.state_machine.state is PackMLState.HELD:
            break

    assert controller.state_machine.state is PackMLState.HELD
    assert controller.inventory.active_reservation.product == 1
    assert not any(event.kind == "installed" for event in controller.events)
    assert "operator reconciliation" in controller.events[-1].detail


def test_occluded_installed_part_uses_verified_assembly_state() -> None:
    perception = ScriptedPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)

    def installed() -> bool:
        return any(event.kind == "installed" for event in controller.events)

    observation = _drive_until(controller, installed)
    perception.observed = tuple(item for item in perception.observed if item.instance_id != "p1-1")

    CellAdapter(controller, action_mode="tcp", dt=0.1).act(observation)

    assert controller.state_machine.state is PackMLState.EXECUTE
    assert controller.inventory.active_reservation.product == 1
    assert controller.inventory.item("p1-1").condition is ObjectCondition.VERIFIED
    assert controller.completed_products == ()
