"""Controller regressions for advisory FoundationPose and PatchCore evidence."""

from collections import Counter
from dataclasses import replace

import pytest

from assembly.calibrated_poses import CalibratedPoseError
from assembly.catalog import load_catalog
from assembly.controller_ports import EvidencePair
from assembly.inventory import ObjectCondition
from assembly.planner import RecipePlanner
from assembly.state_machine import PackMLState
from robot.adapter import CellAdapter
from tests.test_controller import (
    CalibratedScenarioPoses,
    ScriptedPerception,
    _controller,
    _observation,
)
from tests.test_controller_brain_rules import RetryAuditPerception, _run_until
from tests.test_controller_fail_closed import _product_one_items
from tests.test_controller_review_fixes import _apply


class UnavailableFoundationPoseGrasp:
    def __init__(self) -> None:
        self._delegate = CalibratedScenarioPoses()

    def grasp_from_foundationpose(self, part, detection, *, observed_at_s: float):
        raise CalibratedPoseError("no approved grasp for observed resting side")

    def __getattr__(self, name):
        return getattr(self._delegate, name)


class OffsetTcpPoses(CalibratedScenarioPoses):
    def grasp_from_foundationpose(self, part, detection, *, observed_at_s: float):
        grasp = super().grasp_from_foundationpose(part, detection, observed_at_s=observed_at_s)
        return replace(grasp, tcp=replace(grasp.tcp, x=grasp.tcp.x + 5.0))


class ExplicitFlippedPerception(ScriptedPerception):
    def pre_pick(self, observation, part):
        evidence = super().pre_pick(observation, part)
        inspection = dict(evidence.inspection)
        inspection.update(verdict="flipped", action="reject")
        return EvidencePair(evidence.detection, inspection, evidence.observed_at_s)


class InvalidFoundationPoseEvidence(ScriptedPerception):
    invalidity = "low_fit"

    def pre_pick(self, observation, part):
        evidence = super().pre_pick(observation, part)
        detection = dict(evidence.detection)
        detection["pose"] = {"x": 999.0, "y": -125.0, "z": -300.0, "yaw": 0.0}
        inspection = dict(evidence.inspection)
        if self.invalidity == "missing":
            detection.update(pose=None, pose_source="none", pose_fit_score=None)
        elif self.invalidity == "low_fit":
            detection["pose_fit_score"] = 0.1
        else:
            inspection["checks_agree"] = False
        return EvidencePair(detection, inspection, evidence.observed_at_s)


class PatchCoreAfterInsertionLiftPerception(RetryAuditPerception):
    def pre_pick(self, observation, part):
        evidence = super().pre_pick(observation, part)
        if part.instance_id != "p1-2" or self.pre_calls[part.instance_id] != 2:
            return evidence
        inspection = dict(evidence.inspection)
        inspection.update(verdict="uncertain", action="ask_operator")
        payload = dict(inspection["evidence"])
        payload["patchcore_score"] = 0.95
        inspection["evidence"] = payload
        return replace(evidence, inspection=inspection)


def test_missing_foundationpose_grasp_warns_and_uses_calibrated_fallback() -> None:
    perception = ScriptedPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    poses = UnavailableFoundationPoseGrasp()
    controller._poses = poses
    controller._planner = RecipePlanner(load_catalog(), poses)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))

    _apply(observation, adapter.act(observation))
    output = adapter.act(observation)

    assert controller.state_machine.state is PackMLState.EXECUTE
    assert controller.plan_records[0].grasp.x == 240.0
    assert output != observation["tcp_pose"] + [0.0]
    assert any(
        event.kind == "warning"
        and event.object_id == "M1_01"
        and "FoundationPose advisory" in event.detail
        for event in controller.events
    )


@pytest.mark.parametrize("invalidity", ["missing", "low_fit", "disagreement"])
def test_invalid_foundationpose_evidence_uses_calibrated_grasp(invalidity: str) -> None:
    perception = InvalidFoundationPoseEvidence(_product_one_items(), set(), Counter(), set())
    perception.invalidity = invalidity
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))

    _apply(observation, adapter.act(observation))
    adapter.act(observation)

    assert controller.state_machine.state is PackMLState.EXECUTE
    assert controller.plan_records[0].grasp.x == 240.0
    assert any(event.kind == "warning" and "using calibrated grasp" in event.detail for event in controller.events)


def test_fallen_part_does_not_use_upright_calibrated_grasp_fallback() -> None:
    observed = tuple(
        replace(item, condition=ObjectCondition.FALLEN) if item.instance_id == "p1-1" else item
        for item in _product_one_items()
    )
    perception = ScriptedPerception(observed, set(), Counter(), set())
    controller = _controller(perception)
    poses = UnavailableFoundationPoseGrasp()
    controller._poses = poses
    controller._planner = RecipePlanner(load_catalog(), poses)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))

    _apply(observation, adapter.act(observation))
    adapter.act(observation)

    assert "no approved grasp" in controller.events[-1].detail
    assert controller.plan_records == ()


def test_explicit_flipped_verdict_does_not_use_upright_calibrated_grasp_fallback() -> None:
    perception = ExplicitFlippedPerception(_product_one_items(), set(), Counter(), set())
    controller = _controller(perception)
    poses = UnavailableFoundationPoseGrasp()
    controller._poses = poses
    controller._planner = RecipePlanner(load_catalog(), poses)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))

    _apply(observation, adapter.act(observation))
    adapter.act(observation)

    assert controller.state_machine.state is PackMLState.HELD
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


def test_post_lift_patchcore_warning_holds_for_operator_without_reject() -> None:
    perception = PatchCoreAfterInsertionLiftPerception(
        _product_one_items(), set(), Counter({"p1-2": 1}), set()
    )
    controller = _controller(perception)
    observation = _observation((1,))

    _run_until(controller, observation, lambda: controller.state_machine.state is PackMLState.HELD)

    assert "PatchCore advisory warning" in controller.events[-1].detail
    assert "operator decision is required" in controller.events[-1].detail
    assert not any(event.kind == "rejected" and event.detail == "p1-2" for event in controller.events)
