"""Regression tests for fail-closed runtime and perception boundaries."""

from collections import Counter
from dataclasses import replace

import pytest

from assembly.catalog import load_catalog
from assembly.controller_ports import EvidencePair, ExecutionTolerances, PerceptionTolerances
from assembly.inventory import InventoryZone, ObjectCondition
from assembly.planner import RecipePlanner
from assembly.state_machine import PackMLState
from robot.adapter import CellAdapter
from tests.test_controller import (
    CalibratedScenarioPoses,
    ScriptedPerception,
    _controller,
    _detection,
    _inspection,
    _observation,
    _observed_three_products,
)


def _product_one_items() -> tuple:
    return tuple(
        item for item in _observed_three_products() if item.object_id.startswith("M1") or item.object_id == "F01"
    )


def _apply(observation: dict[str, object], result) -> None:
    rows = result if result and isinstance(result[0], list) else [result]
    observation["tcp_pose"] = rows[-1][:6]
    observation["joints"] = [0.0] * 6 + [rows[-1][6]]
    observation["t"] = float(observation["t"]) + 1.0


def _drive_until(controller, stop, *, limit: int = 600) -> dict[str, object]:
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation()
    for _ in range(limit):
        _apply(observation, adapter.act(observation))
        if stop():
            break
    return observation


class MutableScanPerception(ScriptedPerception):
    stale_scan = False

    def scan(self, observation):
        snapshot = super().scan(observation)
        return replace(snapshot, observed_at_s=snapshot.observed_at_s - float(self.stale_scan))


class IdentityMutationPerception(ScriptedPerception):
    mutate_identity = False

    def scan(self, observation):
        snapshot = super().scan(observation)
        if not self.mutate_identity:
            return snapshot
        items = tuple(
            replace(item, object_id="M1_04") if item.instance_id == "p1-1" else item for item in snapshot.items
        )
        return replace(snapshot, items=items)


class SlightPoseNoisePerception(ScriptedPerception):
    def pre_pick(self, observation, part):
        evidence = super().pre_pick(observation, part)
        detection = dict(evidence.detection)
        detection["pose"] = {**detection["pose"], "x": detection["pose"]["x"] + 0.001}
        return EvidencePair(detection, evidence.inspection, evidence.observed_at_s)


class MovingBasePerception(ScriptedPerception):
    def collision_scene(self, observation, moving_instance_id, base_instance_id):
        scene = super().collision_scene(observation, moving_instance_id, base_instance_id)
        if self.collision_calls <= 2:
            return scene
        moved = (scene.base_pose[0] + 1.0, *scene.base_pose[1:])
        return replace(scene, base_pose=moved)


class SlippedGraspPerception(ScriptedPerception):
    def collision_scene(self, observation, moving_instance_id, base_instance_id):
        scene = super().collision_scene(observation, moving_instance_id, base_instance_id)
        if float(observation["joints"][6]) == 1.0:
            return replace(scene, moving_attached=False)
        return scene


class RecoveryUntrustedPosePerception(ScriptedPerception):
    def pre_pick(self, observation, part):
        evidence = super().pre_pick(observation, part)
        if part.zone not in {InventoryZone.INSTALLED, InventoryZone.STAGED}:
            return evidence
        detection = dict(evidence.detection)
        detection["pose_source"] = "detector"
        return EvidencePair(detection, evidence.inspection, evidence.observed_at_s)


class WrongPostPlaceTypePerception(ScriptedPerception):
    def post_place(self, observation, part):
        evidence = super().post_place(observation, part)
        return EvidencePair(evidence.detection, evidence.inspection, evidence.observed_at_s)


class WrongTransferTypePerception(ScriptedPerception):
    def post_transfer(self, observation, part, target_zone):
        evidence = super().post_transfer(observation, part, target_zone)
        return EvidencePair(evidence.detection, evidence.inspection, evidence.observed_at_s)


class StaleRecoveryPoses(CalibratedScenarioPoses):
    fail = False

    def base_for(self, base, *, observed_at_s: float):
        if self.fail:
            raise ValueError("base measurement is stale")
        return super().base_for(base, observed_at_s=observed_at_s)


class UnavailableScanPerception(ScriptedPerception):
    unavailable = False

    def scan(self, observation):
        if self.unavailable:
            raise RuntimeError("camera inference unavailable")
        return super().scan(observation)


class MalformedRecoveryTargets:
    def reject_target_for(self, part):
        return None

    def staging_target_for(self, product, part, slot):
        return None


class OutOfBoundsTargetPoses(CalibratedScenarioPoses):
    def target_for(self, product, base, step, *, observed_at_s: float):
        pose = super().target_for(product, base, step, observed_at_s=observed_at_s)
        return replace(pose, tcp=replace(pose.tcp, x=700.0))



class MalformedScanPerception(ScriptedPerception):
    malformed = False

    def scan(self, observation):
        snapshot = super().scan(observation)
        return replace(snapshot, items=None) if self.malformed else snapshot


class MalformedScenePerception(ScriptedPerception):
    malformed_field = "fixtures"

    def collision_scene(self, observation, moving_instance_id, base_instance_id):
        scene = super().collision_scene(observation, moving_instance_id, base_instance_id)
        return replace(scene, **{self.malformed_field: None})


class UnsizedArray:
    def __len__(self):
        raise TypeError("len() of unsized object")

    def __getitem__(self, index):
        raise IndexError(index)


def test_stale_complete_scan_holds_andon_without_raising() -> None:
    perception = MutableScanPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation()
    _apply(observation, adapter.act(observation))
    perception.stale_scan = True

    held = adapter.act(observation)

    assert controller.state_machine.state is PackMLState.HELD
    assert controller.events[-1].kind == "andon"
    assert held == observation["tcp_pose"] + [observation["joints"][-1]]


def test_port_runtime_failure_holds_andon_without_raising() -> None:
    perception = UnavailableScanPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation()
    _apply(observation, adapter.act(observation))
    perception.unavailable = True

    held = adapter.act(observation)

    assert controller.state_machine.state is PackMLState.HELD
    assert "camera inference unavailable" in controller.events[-1].detail
    assert held[-1] == observation["joints"][-1]


def test_malformed_inventory_collection_holds_andon_without_raising() -> None:
    perception = MalformedScanPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation()
    _apply(observation, adapter.act(observation))
    perception.malformed = True

    held = adapter.act(observation)

    assert controller.state_machine.state is PackMLState.HELD
    assert "items must be a tuple" in controller.events[-1].detail
    assert held[-1] == 0.0


@pytest.mark.parametrize("field", ["fixtures", "parts"])
def test_malformed_collision_collection_aborts_without_raising(field: str) -> None:
    perception = MalformedScenePerception(_product_one_items(), set(), Counter(), set())
    perception.malformed_field = field
    controller = _controller(perception)

    held = CellAdapter(controller, action_mode="tcp", dt=0.1).act(_observation())

    assert controller.state_machine.state is PackMLState.ABORTED
    assert f"{field} must be a tuple" in controller.events[-1].detail
    assert held[-1] == 0.0


def test_unsized_tcp_array_holds_last_safe_waypoint_without_raising() -> None:
    perception = ScriptedPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation()
    _apply(observation, adapter.act(observation))
    observation["tcp_pose"] = UnsizedArray()

    held = adapter.act(observation)

    assert controller.state_machine.state is PackMLState.HELD
    assert "six absolute UF0 values" in controller.events[-1].detail
    assert held[-1] == observation["joints"][-1]


def test_malformed_tcp_after_release_reissues_open_gripper_endpoint() -> None:
    perception = ScriptedPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    observation = _drive_until(controller, lambda: controller._release_dispatched)
    observation["tcp_pose"] = UnsizedArray()

    held = CellAdapter(controller, action_mode="tcp", dt=0.1).act(observation)

    assert controller.state_machine.state is PackMLState.HELD
    assert held[-1] == 0.0
    assert "operator reconciliation" in controller.events[-1].detail


def test_inventory_identity_mutation_holds_andon_without_raising() -> None:
    perception = IdentityMutationPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation()
    _apply(observation, adapter.act(observation))
    perception.mutate_identity = True

    held = adapter.act(observation)

    assert controller.state_machine.state is PackMLState.HELD
    assert controller.events[-1].kind == "andon"
    assert held == observation["tcp_pose"] + [observation["joints"][-1]]


def test_small_independent_pose_noise_uses_measured_tolerance() -> None:
    perception = SlightPoseNoisePerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception, perception_tolerances=PerceptionTolerances(0.01, 0.01))

    result = CellAdapter(controller, action_mode="tcp", dt=0.1).act(_observation())

    assert result
    assert controller.state_machine.state is PackMLState.EXECUTE


def test_moved_base_aborts_before_the_next_chunk() -> None:
    perception = MovingBasePerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation()
    _apply(observation, adapter.act(observation))

    adapter.act(observation)

    assert controller.state_machine.state is PackMLState.ABORTED
    assert "base moved" in controller.events[-1].detail


def test_lost_attachment_holds_with_gripped_part_ownership() -> None:
    perception = SlippedGraspPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation()
    for _ in range(20):
        _apply(observation, adapter.act(observation))
        if controller.state_machine.state is PackMLState.HELD:
            break

    assert controller.state_machine.state is PackMLState.HELD
    assert controller.inventory.active_reservation.product == 1
    assert "attachment" in controller.events[-1].detail


def test_abort_with_staged_part_becomes_operator_hold() -> None:
    perception = ScriptedPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    CellAdapter(controller, action_mode="tcp", dt=0.1).act(_observation())
    controller._motion_queue.clear()
    controller.inventory.install("p1-1")
    controller.inventory.stage_for_rebuild("p1-1")

    controller._abort("collision during rebuild")

    assert controller.state_machine.state is PackMLState.HELD
    assert controller.inventory.item("p1-1").zone is InventoryZone.STAGED
    assert "operator reconciliation" in controller.events[-1].detail


def test_recovery_pick_without_foundationpose_holds_before_dismantle_motion() -> None:
    perception = RecoveryUntrustedPosePerception(_product_one_items(), set(), Counter({"p1-2": 3}), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation()
    for _ in range(500):
        _apply(observation, adapter.act(observation))
        if controller.state_machine.state is PackMLState.HELD:
            break

    assert controller.state_machine.state is PackMLState.HELD
    assert "recovery pre-pick rejected" in controller.events[-1].detail


def test_base_failure_after_first_install_preserves_product_ownership() -> None:
    perception = ScriptedPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    observation = _drive_until(controller, lambda: any(event.kind == "installed" for event in controller.events))
    perception.observed = tuple(
        replace(item, condition=ObjectCondition.FALLEN) if item.instance_id == "base-1" else item
        for item in perception.observed
    )

    CellAdapter(controller, action_mode="tcp", dt=0.1).act(observation)

    assert controller.state_machine.state is PackMLState.HELD
    assert controller.inventory.active_reservation.product == 1
    assert "must remain verified" in controller.events[-1].detail
    assert "operator reconciliation" in controller.events[-1].detail


def test_same_class_wrong_physical_instance_holds_before_motion() -> None:
    class WrongInstancePerception(ScriptedPerception):
        def pre_pick(self, observation, part):
            return EvidencePair(
                _detection("different-instance", part.object_id),
                _inspection("different-instance", part.object_id),
                float(observation["t"]),
            )

    controller = _controller(WrongInstancePerception(_product_one_items(), set(), Counter(), set()))
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation()
    _apply(observation, adapter.act(observation))
    output = adapter.act(observation)

    assert output == observation["tcp_pose"] + [0.0]
    assert controller.state_machine.state is PackMLState.HELD
    assert controller.plan_records == ()


def test_wrong_post_place_evidence_subtype_holds_without_attribute_error() -> None:
    perception = WrongPostPlaceTypePerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)

    _drive_until(controller, lambda: controller.state_machine.state is PackMLState.HELD)

    assert controller.state_machine.state is PackMLState.HELD
    assert "exactly PlacementEvidence" in controller.events[-1].detail


def test_wrong_transfer_evidence_after_release_preserves_reservation() -> None:
    perception = WrongTransferTypePerception(_product_one_items(), set(), Counter({"p1-2": 3}), set())
    controller = _controller(perception)

    _drive_until(controller, lambda: controller.state_machine.state is PackMLState.HELD)

    assert controller.state_machine.state is PackMLState.HELD
    assert controller.inventory.active_reservation.product == 1
    assert "exactly RecoveryTransferEvidence" in controller.events[-1].detail
    assert "operator reconciliation" in controller.events[-1].detail


def test_stale_recovery_calibration_holds_without_raising() -> None:
    poses = StaleRecoveryPoses()
    perception = ScriptedPerception(_product_one_items(), set(), Counter({"p1-2": 3}), set())
    controller = _controller(perception)
    controller._poses = poses
    controller._planner = RecipePlanner(load_catalog(), poses)

    def stop() -> bool:
        if any(event.kind == "rebuild" for event in controller.events):
            poses.fail = True
        return controller.state_machine.state is PackMLState.HELD

    _drive_until(controller, stop)

    assert controller.state_machine.state is PackMLState.HELD
    assert "base measurement is stale" in controller.events[-1].detail


def test_malformed_recovery_target_holds_without_attribute_error() -> None:
    perception = ScriptedPerception(_product_one_items(), set(), Counter({"p1-2": 3}), set())
    controller = _controller(perception)
    controller._recovery._targets = MalformedRecoveryTargets()

    _drive_until(controller, lambda: controller.state_machine.state is PackMLState.HELD)

    assert controller.state_machine.state is PackMLState.HELD
    assert "OrientedPartPose" in controller.events[-1].detail


def test_official_action_bounds_failure_is_caught_by_controller() -> None:
    poses = OutOfBoundsTargetPoses()
    perception = ScriptedPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    controller._poses = poses
    controller._planner = RecipePlanner(load_catalog(), poses)

    _drive_until(controller, lambda: controller.state_machine.state is PackMLState.ABORTED)

    assert controller.state_machine.state is PackMLState.ABORTED
    assert "official UF0 action guard" in controller.events[-1].detail


def test_verified_endpoint_drift_releases_without_roll_pitch_correction() -> None:
    perception = ScriptedPerception(_product_one_items(), set(), Counter(), set())
    tolerances = ExecutionTolerances(0.5, 0.5)
    controller = _controller(perception, execution_tolerances=tolerances)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation()
    for _ in range(200):
        _apply(observation, adapter.act(observation))
        if controller._awaiting_post and not controller._motion_queue.active:
            observation["tcp_pose"][3] += 0.1
            release = adapter.act(observation)
            break
    else:
        pytest.fail("controller never reached provisional placement")

    assert release[-1][-1] == 0.0
    assert release[-1][3] == pytest.approx(-179.9)
    assert controller.state_machine.state is PackMLState.EXECUTE


def test_release_endpoint_failure_holds_and_keeps_active_reservation() -> None:
    perception = ScriptedPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    observation = _drive_until(controller, lambda: controller._release_dispatched)
    observation["tcp_pose"][0] += 10.0

    CellAdapter(controller, action_mode="tcp", dt=0.1).act(observation)

    assert controller.state_machine.state is PackMLState.HELD
    assert controller.inventory.active_reservation.product == 1
    assert "operator reconciliation" in controller.events[-1].detail


def test_endpoint_failure_while_gripping_preserves_reservation() -> None:
    perception = ScriptedPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    observation = _drive_until(controller, lambda: controller._part_held)
    observation["tcp_pose"][0] += 10.0

    CellAdapter(controller, action_mode="tcp", dt=0.1).act(observation)

    assert controller.state_machine.state is PackMLState.HELD
    assert controller.inventory.active_reservation.product == 1
    assert controller.inventory.item("p1-1").zone is InventoryZone.RESERVED
    assert "held part requires operator reconciliation" in controller.events[-1].detail
