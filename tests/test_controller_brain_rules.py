"""Regression coverage for the operator-approved robot-brain case rules."""

from collections import Counter
from dataclasses import replace

import pytest

from assembly.controller_ports import ControllerPortError, ObservedInventoryItem
from assembly.inventory import InventoryZone, ObjectCondition
from assembly.state_machine import PackMLState
from robot.adapter import CellAdapter
from tests.test_controller import (
    CalibratedScenarioPoses,
    ScriptedPerception,
    _controller,
    _observation,
    _observed_products,
)


def _advance(observation, result) -> None:
    rows = result if result and isinstance(result[0], list) else [result]
    observation["tcp_pose"] = rows[-1][:6]
    observation["joints"] = [0.0] * 6 + [rows[-1][6]]
    observation["t"] = float(observation["t"]) + 1.0


def _run_until(controller, observation, predicate, *, limit=1000) -> None:
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    for _ in range(limit):
        _advance(observation, adapter.act(observation))
        if predicate():
            return
    raise AssertionError("controller did not reach the expected brain-rule state")


def _run_until_collect(controller, observation, predicate, *, limit=1000) -> list[list[float]]:
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    emitted: list[list[float]] = []
    for _ in range(limit):
        result = adapter.act(observation)
        rows = result if result and isinstance(result[0], list) else [result]
        emitted.extend(rows)
        _advance(observation, result)
        if predicate():
            return emitted
    raise AssertionError("controller did not reach the expected brain-rule state")


class PersistentIntrusionPerception(ScriptedPerception):
    def collision_scene(self, observation, moving_instance_id, base_instance_id):
        scene = super().collision_scene(observation, moving_instance_id, base_instance_id)
        return replace(scene, intrusion_detected=True)


class OneIntrusionPerception(ScriptedPerception):
    intrusion_remaining = 1

    def collision_scene(self, observation, moving_instance_id, base_instance_id):
        scene = super().collision_scene(observation, moving_instance_id, base_instance_id)
        if self.intrusion_remaining:
            self.intrusion_remaining -= 1
            return replace(scene, intrusion_detected=True)
        return scene


class OneHeldIntrusionPerception(ScriptedPerception):
    intrusion_seen = False

    def collision_scene(self, observation, moving_instance_id, base_instance_id):
        scene = super().collision_scene(observation, moving_instance_id, base_instance_id)
        if not self.intrusion_seen and float(observation["joints"][6]) == 1.0:
            self.intrusion_seen = True
            return replace(scene, intrusion_detected=True)
        return scene


class OneDropPerception(ScriptedPerception):
    dropped = False
    drop_active = False

    def collision_scene(self, observation, moving_instance_id, base_instance_id):
        scene = super().collision_scene(observation, moving_instance_id, base_instance_id)
        closed = float(observation["joints"][6]) == 1.0
        if closed and (not self.dropped or self.drop_active):
            self.dropped = True
            self.drop_active = True
            return replace(scene, moving_attached=False)
        if not closed:
            self.drop_active = False
        return scene


class FallenDropPerception(OneDropPerception):
    def scan(self, observation):
        snapshot = super().scan(observation)
        if not self.drop_active or float(observation["joints"][6]) != 0.0:
            return snapshot
        items = tuple(
            replace(item, condition=ObjectCondition.FALLEN) if item.instance_id == "p1-1" else item
            for item in snapshot.items
        )
        return replace(snapshot, items=items)


class MissingDropPerception(OneDropPerception):
    def scan(self, observation):
        snapshot = super().scan(observation)
        if not self.drop_active or float(observation["joints"][6]) != 0.0:
            return snapshot
        return replace(snapshot, items=tuple(item for item in snapshot.items if item.instance_id != "p1-1"))


class RetryAuditPerception(ScriptedPerception):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.failed_post_z: list[float] = []
        self.retry_pre_z: list[float] = []
        self.retry_pre_scan_calls: list[int] = []

    def post_place(self, observation, part):
        if self.failed_insertions[part.instance_id]:
            self.failed_post_z.append(float(observation["tcp_pose"][2]))
        return super().post_place(observation, part)

    def pre_pick(self, observation, part):
        if part.instance_id == "p1-2" and self.pre_calls[part.instance_id]:
            self.retry_pre_z.append(float(observation["tcp_pose"][2]))
            self.retry_pre_scan_calls.append(self.scan_calls)
        return super().pre_pick(observation, part)

class DefectiveAfterRetryPerception(RetryAuditPerception):
    def pre_pick(self, observation, part):
        evidence = super().pre_pick(observation, part)
        if part.instance_id != "p1-2" or self.pre_calls[part.instance_id] != 3:
            return evidence
        inspection = dict(evidence.inspection)
        inspection.update({"verdict": "broken", "action": "reject"})
        return replace(evidence, inspection=inspection)


class DamagedPlacementPerception(RetryAuditPerception):
    @staticmethod
    def _broken(evidence):
        inspection = dict(evidence.inspection)
        inspection.update({"verdict": "broken", "action": "reject"})
        return replace(evidence, inspection=inspection)

    def post_place(self, observation, part):
        evidence = super().post_place(observation, part)
        return self._broken(evidence) if part.instance_id == "p1-2" else evidence

    def pre_pick(self, observation, part):
        evidence = super().pre_pick(observation, part)
        if part.instance_id == "p1-2" and self.pre_calls[part.instance_id] == 2:
            return self._broken(evidence)
        return evidence


class RetryLiftIntrusionPerception(RetryAuditPerception):
    retry_intrusion_seen = False

    def collision_scene(self, observation, moving_instance_id, base_instance_id):
        scene = super().collision_scene(observation, moving_instance_id, base_instance_id)
        if self.post_calls["p1-2"] and not self.retry_intrusion_seen:
            self.retry_intrusion_seen = True
            return replace(scene, intrusion_detected=True)
        return scene


class VerifiedReleaseIntrusionPerception(ScriptedPerception):
    release_intrusion_seen = False

    def collision_scene(self, observation, moving_instance_id, base_instance_id):
        scene = super().collision_scene(observation, moving_instance_id, base_instance_id)
        if self.post_calls["p1-1"] and not self.release_intrusion_seen:
            self.release_intrusion_seen = True
            return replace(scene, intrusion_detected=True)
        return scene


class UnavailableFinalProductVerifier:
    reviewed_motion_source = "T30 safety-approved physical final-view workflow"
    mock_only = False

    def verify(self, observation, product, base_instance_id, installed_instance_ids):
        del observation, product, base_instance_id, installed_instance_ids
        raise RuntimeError("published calibrated final-view motion is unavailable")


class CountingScenarioPoses(CalibratedScenarioPoses):
    def __init__(self) -> None:
        self.foundationpose_calls: Counter = Counter()

    def grasp_from_foundationpose(self, part, detection, *, observed_at_s: float):
        self.foundationpose_calls[part.instance_id] += 1
        return super().grasp_from_foundationpose(
            part, detection, observed_at_s=observed_at_s
        )


def test_uncertain_pick_holds_until_calibrated_alternate_view_motion_exists() -> None:
    observed = _observed_products((1,))
    perception = ScriptedPerception(observed, set(), Counter(), {"p1-1"})
    controller = _controller(perception)
    observation = _observation((1,))

    _run_until(controller, observation, lambda: controller.state_machine.state is PackMLState.HELD)

    assert perception.pre_calls["p1-1"] == 1
    assert "published calibrated wrist-view targets" in controller.events[-1].detail
    assert "safety-approved motion plan" in controller.events[-1].detail
    assert controller.plan_records == ()


def test_missing_product_is_deferred_while_other_product_is_built() -> None:
    observed = tuple(item for item in _observed_products((1, 3)) if item.instance_id != "p1-1")
    perception = ScriptedPerception(observed, set(), Counter(), set())
    controller = _controller(perception)
    observation = _observation((1, 3))

    _run_until(controller, observation, lambda: controller.completed_products == (3,))

    assert controller.completed_products == (3,)
    assert any(event.kind == "deferred" and event.product == 1 for event in controller.events)
    assert controller.inventory.item("p3-4").zone is InventoryZone.INSTALLED


def test_spare_is_not_consumed_before_other_product_final_verification() -> None:
    observed = tuple(item for item in _observed_products((1, 3)) if item.instance_id != "p1-1")
    observed = (
        *observed,
        ObservedInventoryItem(
            "p1-1-spare",
            "M1_01",
            zone=InventoryZone.SPARE,
            coarse_xy_mm=(240.0, -125.0),
        ),
    )
    perception = ScriptedPerception(observed, set(), Counter(), set())
    controller = _controller(perception)
    observation = _observation((1, 3))

    _run_until(controller, observation, lambda: controller.completed_products == (3,))

    assert any(event.kind == "deferred" and event.product == 1 for event in controller.events)
    complete_index = next(
        index for index, event in enumerate(controller.events) if event.kind == "complete" and event.product == 3
    )
    spare_index = next(
        index for index, event in enumerate(controller.events) if event.kind == "spare" and event.detail == "p1-1-spare"
    )
    assert complete_index < spare_index


def test_finished_product_holds_when_physical_final_view_workflow_is_unavailable() -> None:
    perception = ScriptedPerception(_observed_products((1,)), set(), Counter(), set())

    with pytest.raises(ControllerPortError, match="controller-owned physical view motion"):
        _controller(perception, final_verifier=UnavailableFinalProductVerifier())


def test_insert_failure_lifts_three_mm_then_remeasures_and_retries_once() -> None:
    perception = RetryAuditPerception(_observed_products((1,)), set(), Counter({"p1-2": 1}), set())
    controller = _controller(perception)
    observation = _observation((1,))

    emitted = _run_until_collect(
        controller,
        observation,
        lambda: any(event.kind == "retry" for event in controller.events),
    )

    assert perception.failed_post_z
    lift_z = perception.failed_post_z[0] + 3.0
    lift_index = next(index for index, row in enumerate(emitted) if row[2] == lift_z)
    assert perception.retry_pre_z[0] == -168.0
    assert perception.retry_pre_scan_calls[0] > 0
    assert {row[6] for row in emitted[lift_index:]} == {1.0}
    assert perception.pre_calls["p1-2"] == 2

    _run_until(controller, observation, lambda: controller.finished)

    retries = [event for event in controller.events if event.kind == "retry" and event.object_id == "M1_02"]
    assert len(retries) == 1
    assert perception.post_calls["p1-2"] == 2


def test_second_failed_insert_rejects_defect_before_no_spare_andon() -> None:
    perception = DefectiveAfterRetryPerception(
        _observed_products((1,)), set(), Counter({"p1-2": 2}), set()
    )
    controller = _controller(perception)
    observation = _observation((1,))

    _run_until(controller, observation, lambda: controller.state_machine.state is PackMLState.SUSPENDED)

    kinds = [event.kind for event in controller.events]
    assert kinds.count("retry") == 1
    assert kinds.index("rejected") < len(kinds) - 1
    assert controller.events[-1].kind == "andon"
    assert controller.inventory.item("p1-2").zone is InventoryZone.REJECT
    assert observation["joints"][6] == 0.0


def test_post_place_damage_rejects_without_consuming_insertion_retry() -> None:
    perception = DamagedPlacementPerception(_observed_products((1,)), set(), Counter(), set())
    controller = _controller(perception)
    observation = _observation((1,))

    _run_until(controller, observation, lambda: controller.state_machine.state is PackMLState.SUSPENDED)

    assert not any(event.kind == "retry" for event in controller.events)
    assert controller.inventory.item("p1-2").zone is InventoryZone.REJECT
    assert observation["joints"][6] == 0.0


def test_second_failed_insert_holds_when_part_is_good_but_holes_are_unresolved() -> None:
    perception = RetryAuditPerception(_observed_products((1,)), set(), Counter({"p1-2": 2}), set())
    poses = CountingScenarioPoses()
    controller = _controller(perception, poses=poses)
    observation = _observation((1,))

    _run_until(controller, observation, lambda: controller.state_machine.state is PackMLState.HELD)

    assert perception.pre_calls["p1-2"] == 3
    assert sum(event.kind == "retry" for event in controller.events) == 1
    assert poses.foundationpose_calls["p1-2"] == 2
    assert "typed calibrated hole-clearance verdict" in controller.events[-1].detail
    assert observation["joints"][6] == 1.0


def test_intrusion_during_retry_lift_fails_closed_after_fresh_scan() -> None:
    perception = RetryLiftIntrusionPerception(
        _observed_products((1,)), set(), Counter({"p1-2": 1}), set()
    )
    controller = _controller(perception)
    observation = _observation((1,))

    _run_until(controller, observation, lambda: controller.state_machine.state is PackMLState.HELD)

    assert perception.retry_intrusion_seen is True
    assert perception.scan_calls > 1
    assert "intrusion during insertion recovery lift" in controller.events[-1].detail


def test_resume_cannot_commit_release_interrupted_by_intrusion() -> None:
    perception = VerifiedReleaseIntrusionPerception(
        _observed_products((1,)), set(), Counter(), set()
    )
    controller = _controller(perception)
    observation = _observation((1,))

    _run_until(controller, observation, lambda: controller.state_machine.state is PackMLState.HELD)
    held_target = [*observation["tcp_pose"], observation["joints"][6]]
    controller.acknowledge_resume()
    result = CellAdapter(controller, action_mode="tcp", dt=0.1).act(observation)
    rows = result if result and isinstance(result[0], list) else [result]

    assert rows[-1] == held_target
    assert controller.state_machine.state is PackMLState.HELD
    assert controller.inventory.item("p1-1").zone is InventoryZone.RESERVED
    assert not any(event.kind == "installed" for event in controller.events)
    assert "unresolved verified release" in controller.events[-1].detail


def test_resume_cannot_reinsert_after_retry_lift_intrusion() -> None:
    perception = RetryLiftIntrusionPerception(
        _observed_products((1,)), set(), Counter({"p1-2": 1}), set()
    )
    controller = _controller(perception)
    observation = _observation((1,))

    _run_until(controller, observation, lambda: controller.state_machine.state is PackMLState.HELD)
    held_target = [*observation["tcp_pose"], observation["joints"][6]]
    controller.acknowledge_resume()
    result = CellAdapter(controller, action_mode="tcp", dt=0.1).act(observation)
    rows = result if result and isinstance(result[0], list) else [result]

    assert rows[-1] == held_target
    assert controller.state_machine.state is PackMLState.HELD
    assert not any(event.kind == "retry" for event in controller.events)
    assert perception.pre_calls["p1-2"] == 1
    assert "unresolved insertion recovery lift" in controller.events[-1].detail


def test_persistent_intrusion_freezes_once_then_holds_andon() -> None:
    perception = PersistentIntrusionPerception(_observed_products((1,)), set(), Counter(), set())
    controller = _controller(perception)
    observation = _observation((1,))
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)

    first = adapter.act(observation)
    _advance(observation, first)
    assert controller.state_machine.state is PackMLState.EXECUTE
    assert controller.events[-1].kind == "intrusion"

    adapter.act(observation)
    assert controller.state_machine.state is PackMLState.HELD
    assert "persists after rescan" in controller.events[-1].detail


def test_resolved_intrusion_discards_stale_queue_and_replans_after_fresh_scan() -> None:
    perception = OneIntrusionPerception(_observed_products((1,)), set(), Counter(), set())
    controller = _controller(perception)
    observation = _observation((1,))

    _run_until(controller, observation, lambda: controller.finished)

    assert perception.scan_calls > 1
    assert any(event.kind == "intrusion" for event in controller.events)
    assert perception.pre_calls["p1-1"] == 1


def test_intrusion_while_holding_rebuilds_remaining_plan_from_fresh_state() -> None:
    perception = OneHeldIntrusionPerception(_observed_products((1,)), set(), Counter(), set())
    controller = _controller(perception)
    observation = _observation((1,))

    _run_until(controller, observation, lambda: controller.finished)

    assert perception.intrusion_seen is True
    assert any(event.kind == "intrusion" for event in controller.events)
    assert controller.completed_products == (1,)


def test_upright_drop_opens_safely_and_replans_pick() -> None:
    perception = OneDropPerception(_observed_products((1,)), set(), Counter(), set())
    controller = _controller(perception)
    observation = _observation((1,))

    _run_until(controller, observation, lambda: controller.finished)

    assert perception.dropped is True
    assert any(event.kind == "dropped" and "re-pick" in event.detail for event in controller.events)
    assert perception.pre_calls["p1-1"] >= 2


def test_fallen_drop_is_rejected_and_replaced_after_fresh_scan() -> None:
    observed = (
        *_observed_products((1,)),
        ObservedInventoryItem(
            "p1-1-spare",
            "M1_01",
            zone=InventoryZone.SPARE,
            coarse_xy_mm=(240.0, -125.0),
        ),
    )
    perception = FallenDropPerception(observed, set(), Counter(), set())
    controller = _controller(perception)
    observation = _observation((1,))

    _run_until(controller, observation, lambda: controller.finished)

    assert controller.inventory.item("p1-1").zone is InventoryZone.REJECT
    assert controller.inventory.item("p1-1-spare").zone is InventoryZone.INSTALLED
    assert any(event.kind == "dropped" and "reject then spare" in event.detail for event in controller.events)


def test_unobserved_drop_holds_andon_after_fresh_scan() -> None:
    perception = MissingDropPerception(_observed_products((1,)), set(), Counter(), set())
    controller = _controller(perception)
    observation = _observation((1,))

    _run_until(controller, observation, lambda: controller.state_machine.state is PackMLState.HELD)

    assert "not found in the fresh post-drop scan" in controller.events[-1].detail
