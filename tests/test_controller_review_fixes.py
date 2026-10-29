"""Regression coverage for PR #9 controller review findings."""

from collections import Counter
from dataclasses import replace

import pytest

from assembly.catalog import load_catalog
from assembly.controller_ports import EvidencePair, ObservedInventoryItem
from assembly.inventory import InventoryZone, ObjectCondition
from assembly.planner import RecipePlanner
from assembly.safety import AABB
from assembly.state_machine import PackMLState
from robot.adapter import CellAdapter
from tests.test_controller import (
    CalibratedScenarioPoses,
    ScriptedPerception,
    _controller,
    _drive,
    _observation,
    _observed_products,
    _observed_three_products,
)
from tests.test_controller_fail_closed import UnavailableScanPerception, _drive_until, _product_one_items


def _apply(observation: dict[str, object], action) -> None:
    rows = action if action and isinstance(action[0], list) else [action]
    observation["tcp_pose"] = rows[-1][:6]
    observation["joints"] = [0.0] * 6 + [rows[-1][6]]
    observation["t"] = float(observation["t"]) + 1.0


def test_first_observation_binds_requested_products_in_prompt_order() -> None:
    perception = ScriptedPerception(_observed_products((2, 6)), set(), Counter(), set())
    controller = _controller(perception)

    CellAdapter(controller, action_mode="tcp", dt=0.1).act(_observation((2, 6)))

    assert controller.state_machine.current_product == 2


def test_invalid_trial_instruction_holds_before_any_motion() -> None:
    perception = ScriptedPerception(_observed_products((2,)), set(), Counter(), set())
    controller = _controller(perception)
    observation = _observation((2,))
    observation["trial"] = {"instruction": "assemble the usual one"}

    output = CellAdapter(controller, action_mode="tcp", dt=0.1).act(observation)

    assert output == observation["tcp_pose"] + [0.0]
    assert controller.state_machine.state is PackMLState.HELD
    assert controller.plan_records == ()


def test_malformed_first_observation_uses_sourced_startup_hold_without_escape() -> None:
    perception = ScriptedPerception(_observed_products((1,)), set(), Counter(), set())
    controller = _controller(perception)
    observation = _observation((1,))
    observation["tcp_pose"] = None

    output = CellAdapter(controller, action_mode="tcp", dt=0.1).act(observation)

    assert output == [200.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0]
    assert controller.state_machine.state is PackMLState.HELD
    assert "observation tcp_pose" in controller.events[-1].detail


def test_pre_pick_runs_only_after_configured_inspection_stop_reaches_coarse_xy() -> None:
    perception = ScriptedPerception(_observed_products((1,)), set(), Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))

    inspect_action = adapter.act(observation)
    inspect_rows = inspect_action if isinstance(inspect_action[0], list) else [inspect_action]
    assert perception.pre_calls["p1-1"] == 0
    assert inspect_rows[-1][:6] == [240.0, -125.0, -168.0, -180.0, 0.0, 0.0]

    _apply(observation, inspect_action)
    adapter.act(observation)

    assert perception.pre_calls["p1-1"] == 1


def test_fallen_part_is_physically_transferred_before_inventory_reject_commit() -> None:
    perception = ScriptedPerception(_observed_products((1,)), {"p1-1"}, Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))

    _apply(observation, adapter.act(observation))
    adapter.act(observation)

    assert controller.inventory.item("p1-1").zone is InventoryZone.RESERVED
    assert perception.transfer_calls[("p1-1", InventoryZone.REJECT)] == 0


def test_unrequested_table_part_is_inspected_for_reject_before_requested_recipe() -> None:
    clutter = ObservedInventoryItem("clutter-2-1", "M2_01", coarse_xy_mm=(270.0, -125.0))
    perception = ScriptedPerception((*_observed_products((1,)), clutter), set(), Counter(), set())
    controller = _controller(perception)

    action = CellAdapter(controller, action_mode="tcp", dt=0.1).act(_observation((1,)))
    rows = action if isinstance(action[0], list) else [action]

    assert rows[-1][:3] == [270.0, -125.0, -168.0]
    assert controller.inventory.item("clutter-2-1").zone is InventoryZone.TABLE


def test_scan_flagged_fallen_part_can_move_only_on_physical_reject_path() -> None:
    observed = tuple(
        replace(item, condition=ObjectCondition.FALLEN) if item.instance_id == "p1-1" else item
        for item in _observed_products((1,))
    )
    spare = ObservedInventoryItem(
        "p1-1-spare",
        "M1_01",
        zone=InventoryZone.SPARE,
        coarse_xy_mm=(260.0, -125.0),
    )
    controller = _controller(ScriptedPerception((*observed, spare), set(), Counter(), set()))

    action = CellAdapter(controller, action_mode="tcp", dt=0.1).act(_observation((1,)))

    assert action
    assert controller.state_machine.state is PackMLState.EXECUTE
    assert controller.inventory.item("p1-1").condition is ObjectCondition.FALLEN


def test_reserved_part_becoming_fallen_is_physically_rejected_before_spare() -> None:
    observed = (*_observed_products((1,)), _spare("p1-1-spare", "M1_01"))
    perception = ScriptedPerception(observed, set(), Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))

    _apply(observation, adapter.act(observation))
    perception.observed = tuple(
        replace(item, condition=ObjectCondition.FALLEN) if item.instance_id == "p1-1" else item
        for item in perception.observed
    )
    for _ in range(100):
        _apply(observation, adapter.act(observation))
        if controller.inventory.item("p1-1").zone is InventoryZone.REJECT:
            break

    assert perception.transfer_calls[("p1-1", InventoryZone.REJECT)] == 1
    assert controller.inventory.item("p1-1").zone is InventoryZone.REJECT
    assert controller.inventory.item("p1-1-spare").zone is InventoryZone.RESERVED


def test_installed_scan_defect_holds_for_calibrated_recovery_instead_of_loose_pick() -> None:
    perception = ScriptedPerception(_observed_products((1,)), set(), Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))
    for _ in range(100):
        _apply(observation, adapter.act(observation))
        if controller.inventory.item("p1-1").zone is InventoryZone.INSTALLED:
            break
    perception.observed = tuple(
        replace(item, condition=ObjectCondition.DEFECTIVE) if item.instance_id == "p1-1" else item
        for item in perception.observed
    )

    adapter.act(observation)

    assert controller.state_machine.state is PackMLState.HELD
    assert "requires calibrated recovery" in controller.events[-1].detail
    assert perception.transfer_calls[("p1-1", InventoryZone.REJECT)] == 0


def test_fallen_part_is_rejected_before_shortage_holds_when_no_spare_exists() -> None:
    observed = tuple(
        replace(item, condition=ObjectCondition.FALLEN) if item.instance_id == "p1-1" else item
        for item in _observed_products((1,))
    )
    controller = _controller(ScriptedPerception(observed, set(), Counter(), set()))

    action = CellAdapter(controller, action_mode="tcp", dt=0.1).act(_observation((1,)))

    assert action
    assert controller.state_machine.state is PackMLState.EXECUTE
    assert controller.inventory.item("p1-1").zone is InventoryZone.TABLE


def test_operator_acknowledgement_resumes_held_cycle_only_after_a_fresh_scan() -> None:
    perception = ScriptedPerception(_observed_products((1,)), set(), Counter(), {"p1-1"})
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))

    _apply(observation, adapter.act(observation))
    adapter.act(observation)
    assert controller.state_machine.state is PackMLState.HELD
    scans_before_ack = perception.scan_calls

    controller.acknowledge_resume()
    observation["t"] = 2.0
    resumed = adapter.act(observation)

    assert perception.scan_calls == scans_before_ack + 1
    assert controller.state_machine.state is PackMLState.EXECUTE
    assert resumed != observation["tcp_pose"] + [0.0]


def test_safe_z_inspection_route_avoids_obstacle_on_old_diagonal_path() -> None:
    observed = tuple(
        item for item in _observed_three_products() if item.object_id.startswith("M1") or item.object_id == "F01"
    )
    obstacle = AABB("fixture", (205.0, -130.0, -310.0), (245.0, 5.0, -240.0))
    controller = _controller(ScriptedPerception(observed, set(), Counter(), set(), obstacles=(obstacle,)))

    output = CellAdapter(controller, action_mode="tcp", dt=0.1).act(_observation((1,)))

    assert output != [200.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0]
    assert controller.state_machine.state is PackMLState.EXECUTE


def test_collision_on_safe_z_rise_aborts_before_waypoint_is_returned() -> None:
    observed = tuple(
        item for item in _observed_three_products() if item.object_id.startswith("M1") or item.object_id == "F01"
    )
    obstacle = AABB("fixture", (195.0, -5.0, -230.0), (205.0, 5.0, -190.0))
    controller = _controller(ScriptedPerception(observed, set(), Counter(), set(), obstacles=(obstacle,)))

    output = CellAdapter(controller, action_mode="tcp", dt=0.1).act(_observation((1,)))

    assert output == [200.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0]
    assert controller.state_machine.state is PackMLState.ABORTED
    assert controller.events[-1].kind == "abort"


def test_andon_restock_replaces_pending_rejected_instance_and_resumes() -> None:
    observed = tuple(
        item
        for item in _observed_three_products()
        if (item.object_id.startswith("M1") or item.object_id == "F01") and item.instance_id != "p1-2-spare"
    )
    perception = ScriptedPerception(observed, {"p1-1"}, Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))

    for _ in range(100):
        _apply(observation, adapter.act(observation))
        if controller.state_machine.state is PackMLState.SUSPENDED:
            break
    assert controller.state_machine.state is PackMLState.SUSPENDED

    perception.observed += (
        ObservedInventoryItem(
            "p1-1-restock",
            "M1_01",
            zone=InventoryZone.SPARE,
            coarse_xy_mm=(240.0, -125.0),
        ),
    )
    _drive(controller, products=(1,))

    assert controller.inventory.item("p1-1-restock").zone is InventoryZone.INSTALLED
    assert any(event.kind == "resume" for event in controller.events)


class MissingApprovedGraspPoses:
    """Delegate all poses except the required sourced wrist grasp."""

    def __init__(self) -> None:
        from tests.test_controller import CalibratedScenarioPoses

        self._delegate = CalibratedScenarioPoses()

    def grasp_from_foundationpose(self, part, detection, *, observed_at_s: float):
        raise ValueError("no approved grasp for observed resting side")

    def __getattr__(self, name):
        return getattr(self._delegate, name)


class OffsetTcpPoses(CalibratedScenarioPoses):
    """Apply a sourced object-to-TCP offset after FoundationPose."""

    def grasp_from_foundationpose(self, part, detection, *, observed_at_s: float):
        grasp = super().grasp_from_foundationpose(part, detection, observed_at_s=observed_at_s)
        return replace(grasp, tcp=replace(grasp.tcp, x=grasp.tcp.x + 5.0))


class RecoveryAuditPerception(ScriptedPerception):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.recovery_pre_pick_poses: list[tuple[float, ...]] = []

    def pre_pick(self, observation, part):
        if part.zone in {InventoryZone.INSTALLED, InventoryZone.STAGED}:
            self.recovery_pre_pick_poses.append(tuple(observation["tcp_pose"]))
        return super().pre_pick(observation, part)


class FoundationPoseOnlyRecoveryPoses(CalibratedScenarioPoses):
    def grasp_for(self, part, *, observed_at_s: float):
        raise AssertionError(f"legacy grasp_for called for {part.instance_id} at {observed_at_s}")


def test_missing_approved_grasp_holds_andon_without_commanding_pick() -> None:
    perception = ScriptedPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    poses = MissingApprovedGraspPoses()
    controller._poses = poses
    controller._planner = RecipePlanner(load_catalog(), poses)

    _drive_until(controller, lambda: controller.state_machine.state is PackMLState.HELD)

    assert "no approved grasp" in controller.events[-1].detail
    assert controller.plan_records == ()


def test_foundationpose_object_to_tcp_offset_is_not_compared_to_object_origin() -> None:
    perception = ScriptedPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    poses = OffsetTcpPoses()
    controller._poses = poses
    controller._planner = RecipePlanner(load_catalog(), poses)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))

    _apply(observation, adapter.act(observation))
    output = adapter.act(observation)

    assert controller.state_machine.state is PackMLState.EXECUTE
    assert controller.plan_records[0].grasp.x == 245.0
    assert output != observation["tcp_pose"] + [0.0]


def test_recovery_repicks_only_after_inspection_stop_with_foundationpose_grasp() -> None:
    observed = tuple(
        item for item in _observed_three_products() if item.object_id.startswith("M1") or item.object_id == "F01"
    )
    perception = RecoveryAuditPerception(observed, set(), Counter({"p1-2": 3}), set())
    controller = _controller(perception)
    poses = FoundationPoseOnlyRecoveryPoses()
    controller._poses = poses
    controller._planner = RecipePlanner(load_catalog(), poses)

    _drive(controller, products=(1,), limit=1200)

    assert perception.recovery_pre_pick_poses
    assert all(pose[2:5] == (-168.0, -180.0, 0.0) for pose in perception.recovery_pre_pick_poses)


def test_runtime_failure_while_suspended_preserves_hold_request() -> None:
    observed = tuple(
        item
        for item in _observed_three_products(include_missing_spare=False)
        if item.object_id.startswith("M4") or item.object_id == "F04"
    )
    perception = UnavailableScanPerception(observed, set(), Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((4,))

    adapter.act(observation)
    assert controller.state_machine.state is PackMLState.SUSPENDED
    perception.unavailable = True
    observation["t"] = 1.0
    adapter.act(observation)

    assert controller.state_machine.state is PackMLState.HELD
    assert "camera inference unavailable" in controller.events[-1].detail


def test_controller_runtime_error_holds_last_safe_pose_without_crashing_client() -> None:
    perception = ScriptedPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    controller._progress = _raise_runtime
    observation = _observation((1,))

    output = CellAdapter(controller, action_mode="tcp", dt=0.1).act(observation)

    assert output == observation["tcp_pose"] + [0.0]
    assert controller.state_machine.state is PackMLState.HELD
    assert "controller invariant failed" in controller.events[-1].detail


def test_unknown_foreign_item_without_approved_identity_or_grasp_holds() -> None:
    foreign = ObservedInventoryItem("foreign-1", "FOREIGN", coarse_xy_mm=(300.0, -100.0))
    perception = ScriptedPerception((*_product_one_items(), foreign), set(), Counter(), set())
    controller = _controller(perception)

    output = CellAdapter(controller, action_mode="tcp", dt=0.1).act(_observation((1,)))

    assert controller.state_machine.state is PackMLState.HELD
    assert "foreign object cannot be picked" in controller.events[-1].detail
    assert "sourced FoundationPose grasp" in controller.events[-1].detail
    assert perception.pre_calls == Counter()
    assert output == _observation((1,))["tcp_pose"] + [0.0]


class VerdictRejectPerception(ScriptedPerception):
    verdict = "broken"

    def pre_pick(self, observation, part):
        evidence = super().pre_pick(observation, part)
        inspection = dict(evidence.inspection)
        inspection.update(verdict=self.verdict, action="reject")
        return EvidencePair(evidence.detection, inspection, evidence.observed_at_s)


class WrongRejectZonePerception(ScriptedPerception):
    def post_transfer(self, observation, part, target_zone):
        evidence = super().post_transfer(observation, part, target_zone)
        if target_zone is InventoryZone.REJECT:
            return replace(evidence, target_zone=InventoryZone.STAGED)
        return evidence


def test_reject_transfer_wrong_zone_holds_without_inventory_commit() -> None:
    observed = tuple(
        item for item in _observed_three_products() if item.object_id.startswith("M1") or item.object_id == "F01"
    )
    perception = WrongRejectZonePerception(observed, {"p1-2"}, Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))

    for _ in range(500):
        _apply(observation, adapter.act(observation))
        if controller.state_machine.state is PackMLState.HELD:
            break

    assert controller.state_machine.state is PackMLState.HELD
    assert controller.inventory.item("p1-2").zone is not InventoryZone.REJECT


@pytest.mark.parametrize("verdict", ["broken", "wrong_part", "uncertain"])
def test_explicit_reject_verdict_is_physically_transferred_before_replacement(verdict: str) -> None:
    observed = (*_observed_products((1,)), _spare("p1-1-spare", "M1_01"))
    perception = VerdictRejectPerception(observed, set(), Counter(), set())
    perception.verdict = verdict
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))

    for _ in range(100):
        _apply(observation, adapter.act(observation))
        if controller.inventory.item("p1-1").zone is InventoryZone.REJECT:
            break

    assert perception.transfer_calls[("p1-1", InventoryZone.REJECT)] == 1
    assert controller.inventory.item("p1-1").zone is InventoryZone.REJECT
    assert any(event.kind == "rejected" and event.detail == "p1-1" for event in controller.events)


def _spare(instance_id: str, object_id: str) -> ObservedInventoryItem:
    return ObservedInventoryItem(instance_id, object_id, InventoryZone.SPARE, coarse_xy_mm=(240.0, -125.0))


def _raise_runtime(*_args):
    raise RuntimeError("controller invariant failed")
