"""CELL controller keeps frames and GR00T motion authority separate.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from assembly.catalog import load_catalog
from assembly.controller_ports import CollisionScene
from assembly.controller_recovery import PendingInsertionLift
from assembly.execution import CellTrialHeld
from assembly.fixed_inspection import (
    FixedInspectionMove,
    InspectionPoseRegistry,
    InspectionSlotPose,
)
from assembly.groot_controller import GrootAssemblyController
from assembly.inventory import Inventory, InventoryItem, InventoryZone, ObjectCondition
from assembly.motion import AbsolutePoseUF0, MotionChunk, MotionPhase, OrientedPartPose, RestingState, Waypoint
from assembly.motion_intent import MotionIntent, semantic_instruction
from assembly.planner import ContractPoseUF0, PlanCandidate, PlanStepRecord, PreparedPlanStep, RecipePlanner
from assembly.recovery import PullOutTarget
from assembly.state_machine import PackMLState
from assembly.verification import VerificationReason
from policy_learn.deploy.cell_executor import GrootExecutionResult
from policy_learn.deploy.cell_guards import ChunkGuard, ContactTarget
from robot.adapter import CellAdapter
from robot.cell.t09_mock_runtime import build_groot_mock_controller

OBJECT_ORIGIN = (300.0, -50.0, -300.0)
OBJECT_YAW_DEG = 37.0
T18_GOLDENS = json.loads(
    (Path(__file__).parent / "fixtures" / "t18_instruction_goldens.json").read_text()
)


def _pose(x: float, y: float, z: float, yaw: float) -> OrientedPartPose:
    return OrientedPartPose(
        AbsolutePoseUF0(x, y, z, -180.0, 0.0, yaw, "UF0", "calibrated"),
        RestingState.INSTALLED,
        "installed:+z",
    )


class _RearGuardPoses:
    def base_for(self, base, *, observed_at_s: float) -> OrientedPartPose:
        del base, observed_at_s
        return _pose(400.0, 80.0, -320.0, 90.0)

    def target_for(self, product, base, step, *, observed_at_s: float) -> OrientedPartPose:
        del product, base, step, observed_at_s
        return _pose(420.0, 80.0, -280.0, 90.0)

    def verification_target_for(self, product, base, step, *, observed_at_s: float) -> OrientedPartPose:
        return self.target_for(product, base, step, observed_at_s=observed_at_s)


def _rear_guard_candidate() -> PlanCandidate:
    inventory = Inventory(load_catalog())
    inventory.add("base-1", "F01")
    for step in range(1, 5):
        inventory.add(f"part-{step}", f"M1_{step:02d}")
    inventory.reserve_product(1)
    for step in range(1, 4):
        inventory.install(f"part-{step}")
    candidate = RecipePlanner(load_catalog(), _RearGuardPoses()).next_candidate(
        inventory,
        observed_at_s=10.0,
    )
    assert candidate is not None
    return candidate


def _reviewed_rear_guard_grasp(candidate: PlanCandidate) -> OrientedPartPose:
    local_x, local_y, local_z = candidate.recipe_step.preferred_grasp.tcp_local_mm
    yaw = math.radians(OBJECT_YAW_DEG)
    x = OBJECT_ORIGIN[0] + math.cos(yaw) * local_x - math.sin(yaw) * local_y
    y = OBJECT_ORIGIN[1] + math.sin(yaw) * local_x + math.cos(yaw) * local_y
    return _pose(x, y, OBJECT_ORIGIN[2] + local_z, OBJECT_YAW_DEG)


def _grasp_from_object(step, object_pose) -> OrientedPartPose:
    x, y, yaw_deg = (float(value) for value in object_pose)
    local_x, local_y, local_z = step.preferred_grasp.tcp_local_mm
    yaw = math.radians(yaw_deg)
    tcp_x = x + math.cos(yaw) * local_x - math.sin(yaw) * local_y
    tcp_y = y + math.sin(yaw) * local_x + math.cos(yaw) * local_y
    return _pose(tcp_x, tcp_y, -300.0 + local_z, yaw_deg)


def _record_pose(pose: OrientedPartPose) -> ContractPoseUF0:
    return ContractPoseUF0(pose.tcp.x, pose.tcp.y, pose.tcp.z, pose.tcp.r)


def _prepared_from_golden(row: dict) -> PreparedPlanStep:
    product_id = int(row["object_id"][1])
    product = load_catalog().product(product_id)
    step = next(item for item in product.steps if item.object_id == row["object_id"])
    base_id = row["base_id"] or f"F{product_id:02d}"
    base = InventoryItem("golden-base", base_id, InventoryZone.RESERVED, ObjectCondition.VERIFIED, product_id)
    part = InventoryItem("golden-part", row["object_id"], InventoryZone.RESERVED, ObjectCondition.VERIFIED, product_id)
    grasp = _grasp_from_object(step, row["object_pose"])
    goal_x, goal_y, goal_yaw = (float(value) for value in row["goal_pose"])
    base_pose = _pose(goal_x, goal_y, -320.0, goal_yaw)
    if row["skill"] == "pull_out":
        target = _grasp_from_object(step, row["object_pose"])
    elif row["skill"] == "defect_drop":
        target = _pose(goal_x, goal_y, -290.0, 73.0)
    else:
        target = _pose(goal_x + 11.0, goal_y - 7.0, -290.0, goal_yaw + 73.0)
    record = PlanStepRecord(
        "golden-step", product_id, part.instance_id, part.object_id,
        _record_pose(grasp), _record_pose(target), step.preferred_grasp.turn_deg, 1,
    )
    return PreparedPlanStep(record, step, base, part, base_pose, grasp, target, None)


class _GoldenRecoveryTargets:
    def __init__(self, row: dict) -> None:
        self._goal = tuple(float(value) for value in row["goal_pose"])

    def pull_out_target_for(self, product: int, part: InventoryItem) -> PullOutTarget:
        assert product == int(part.object_id[1])
        return PullOutTarget(
            _pose(self._goal[0], self._goal[1], -300.0, self._goal[2]),
            -180.0,
            "recorded T18 test-bed clearance fixture",
        )


def _intent_from_golden(controller: GrootAssemblyController, row: dict) -> MotionIntent:
    prepared = _prepared_from_golden(row)
    if row["skill"] == "pick_insert":
        return controller._pick_insert_intent(prepared)
    if row["skill"] == "defect_drop":
        return controller._defect_intent(prepared)
    controller._recovery_targets = _GoldenRecoveryTargets(row)
    controller._pending_insertion_lift = PendingInsertionLift(
        prepared, False, VerificationReason.NOT_SEATED
    )
    grasp = GrootAssemblyController._command_tcp(prepared.target, prepared)
    return controller._pull_out_intent(Waypoint(*grasp, 1.0))


@dataclass
class _Executor:
    result: GrootExecutionResult | None = None

    def __post_init__(self) -> None:
        self.resets = 0
        self.intents: list[MotionIntent] = []
        self.inspections: list[FixedInspectionMove] = []
        self.scenes: list[CollisionScene] = []

    def reset(self) -> None:
        self.resets += 1

    def act(self, intent, *, observation, scene):
        del observation
        assert isinstance(scene, CollisionScene)
        self.scenes.append(scene)
        self.intents.append(intent)
        if self.result is None:
            raise AssertionError("executor should not run while only queueing")
        return self.result

    def inspect(self, move, *, observation):
        del observation
        self.inspections.append(move)
        if self.result is None:
            raise AssertionError("executor should not run while only queueing")
        return self.result


def _inspection_slots() -> InspectionPoseRegistry:
    return InspectionPoseRegistry(
        (
            InspectionSlotPose(
                slot_id="slot-a",
                selection_bounds_xy_mm=(240.0, 260.0, -110.0, -90.0),
                endpoint_tcp=(321.0, -45.0, -168.0, -180.0, 0.0, 77.0),
                safe_z_mm=-180.0,
                source="reviewed CSI slot-pose test fixture",
            ),
        )
    )


def _controller(
    executor: _Executor,
    inspection_poses: InspectionPoseRegistry | None = None,
) -> GrootAssemblyController:
    controller = build_groot_mock_controller(
        executor,
        inspection_poses=inspection_poses or _inspection_slots(),
    )
    assert isinstance(controller, GrootAssemblyController)
    return controller


def test_cell_refuses_startup_hold_without_measured_tcp() -> None:
    executor = _Executor()
    controller = _controller(executor)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)

    with pytest.raises(CellTrialHeld, match="measured current TCP"):
        adapter.act({"t": 10.0, "tcp_pose": None, "joints": [0.0] * 7})

    assert controller.state_machine.state is PackMLState.HELD
    assert controller.events[-1].kind == "andon"
    assert executor.intents == []
    assert executor.inspections == []


def test_cell_reset_clears_perception_trial_state() -> None:
    controller = _controller(_Executor())
    controller._perception._instruction = "stale trial"

    controller.reset()

    assert controller._perception._instruction is None


def _queue_rear_guard_inspection(controller: GrootAssemblyController, plan: tuple[MotionChunk, ...]) -> None:
    candidate = _rear_guard_candidate()
    candidate = replace(candidate, target=replace(candidate.target, up_face="mock-installed:+z"))
    controller._pending_pre_pick = candidate
    controller._coarse_xy_by_instance[candidate.part.instance_id] = (250.0, -100.0)
    current = Waypoint(400.0, 0.0, -200.0, -180.0, 0.0, 15.0, 0.0)
    observation = {
        "t": 10.0,
        "tcp_pose": current.as_action()[:6],
        "joints": [0.0] * 7,
        "trial": {"instruction": "assemble product 1"},
    }
    queued = controller._set_guarded_motion(
        plan,
        observation,
        current,
        10.0,
        candidate.part.instance_id,
        candidate.part.object_id,
        candidate.base.instance_id,
        controller._poses.base_for(candidate.base, observed_at_s=10.0),
    )
    assert queued is True


def test_rear_guard_t29_inverse_recovers_origin_and_90_degree_wrist_turn() -> None:
    candidate = _rear_guard_candidate()
    grasp = _reviewed_rear_guard_grasp(candidate)
    prepared: PreparedPlanStep = RecipePlanner.finalize_candidate(candidate, grasp)

    object_pose, origin_z, grasp_tcp, wrist_roll = GrootAssemblyController._grasp_fields(grasp, prepared)

    assert candidate.recipe_step.preferred_grasp.tcp_local_mm == (0.0, 1.8, 27.9)
    assert candidate.recipe_step.preferred_grasp.turn_deg == 90.0
    assert object_pose == pytest.approx((*OBJECT_ORIGIN[:2], OBJECT_YAW_DEG), abs=1e-6)
    assert origin_z == pytest.approx(OBJECT_ORIGIN[2], abs=1e-6)
    assert grasp_tcp[:3] == pytest.approx((grasp.tcp.x, grasp.tcp.y, grasp.tcp.z), abs=1e-6)
    assert wrist_roll == pytest.approx(OBJECT_YAW_DEG + 90.0)
    assert grasp_tcp[5] == pytest.approx(wrist_roll)


def test_t38_neutral_sim_roll_maps_to_real_fanuc_r_without_changing_tool_cone() -> None:
    candidate = _rear_guard_candidate()
    grasp = _grasp_from_object(candidate.recipe_step, (300.0, -50.0, 90.0))
    prepared = RecipePlanner.finalize_candidate(candidate, grasp)

    _object_pose, _origin_z, command, wrist_roll = GrootAssemblyController._grasp_fields(
        grasp,
        prepared,
    )

    assert candidate.recipe_step.preferred_grasp.turn_deg == 90.0
    assert wrist_roll == pytest.approx(180.0)
    assert command[3:] == pytest.approx((-180.0, 0.0, 180.0))


def test_pick_insert_floor_envelope_covers_physical_base_footprint() -> None:
    candidate = _rear_guard_candidate()
    prepared = RecipePlanner.finalize_candidate(candidate, _reviewed_rear_guard_grasp(candidate))
    intent = _controller(_Executor())._pick_insert_intent(prepared)
    contact = intent.contact
    assert contact is not None
    assert math.hypot(
        intent.endpoint_tcp[0] - prepared.base_pose.tcp.x,
        intent.endpoint_tcp[1] - prepared.base_pose.tcp.y,
    ) == pytest.approx(20.0)
    assert contact.radius_mm == pytest.approx(57.828)
    assert load_catalog().geometry_source.sha256 in contact.source

    guard = ChunkGuard()
    guard.set_target(
        ContactTarget(
            contact.x_mm,
            contact.y_mm,
            contact.z_contact_mm,
            radius_mm=contact.radius_mm,
            label=contact.label,
        )
    )
    start = np.asarray(
        [[contact.x_mm + 0.5, contact.y_mm, contact.z_contact_mm + 0.5, -180.0, 0.0, 90.0, 1.0]],
        dtype=float,
    )
    below_floor = start.copy()
    below_floor[0, 2] = contact.z_contact_mm - 5.2

    result = guard.guard(below_floor, start[0, :6], 1.0, 0.1)

    assert result.held is True
    assert any("contact floor" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "row",
    T18_GOLDENS,
    ids=lambda row: f"{row['skill']}-{row['object_id']}-{row['task_index']}",
)
def test_controller_instruction_matches_recorded_t18_task_byte_for_byte(row: dict) -> None:
    intent = _intent_from_golden(_controller(_Executor()), row)

    assert semantic_instruction(intent) == row["expected"]
    assert intent.object_pose == pytest.approx(row["object_pose"])
    assert intent.goal_pose == pytest.approx(row["goal_pose"])
    assert intent.case == row["case"]
    assert intent.object_pose[2] != intent.wrist_roll_deg
    if row["skill"] == "pull_out":
        assert intent.release_authorized is True
        assert intent.release_tcp is not None
        assert intent.endpoint_tcp[2] == -180.0
        assert intent.final_gripper == 0.0


def test_product_five_on_interchangeable_f04_keeps_physical_goal_but_uses_f05_language() -> None:
    row = next(
        item
        for item in T18_GOLDENS
        if item["skill"] == "pick_insert" and item["object_id"] == "M5_01"
    )
    physical_f04 = {**row, "base_id": "F04"}

    controller = _controller(_Executor())
    prepared = _prepared_from_golden(physical_f04)
    intent = controller._pick_insert_intent(prepared)
    release = controller._release_intent(prepared)

    assert intent.base_object_id == "F04"
    assert intent.goal_pose == pytest.approx(row["goal_pose"])
    assert semantic_instruction(intent) == row["expected"]
    assert semantic_instruction(release) == row["expected"]


def test_pull_out_without_calibrated_target_fails_closed() -> None:
    row = next(item for item in T18_GOLDENS if item["skill"] == "pull_out")
    prepared = _prepared_from_golden(row)
    executor = _Executor()
    controller = _controller(executor)
    controller._recovery_targets = object()
    controller._pending_insertion_lift = PendingInsertionLift(
        prepared, False, VerificationReason.NOT_SEATED
    )
    current = Waypoint(*GrootAssemblyController._command_tcp(prepared.target, prepared), 1.0)

    queued = controller._set_guarded_motion(
        (), {}, current, 10.0, prepared.part.instance_id, prepared.part.object_id,
        prepared.base.instance_id, prepared.base_pose,
    )

    assert queued is False
    assert controller.state_machine.state is PackMLState.HELD
    assert "calibrated empty test-bed target" in controller.state_machine.current_record.reason
    assert executor.intents == []


def test_completed_open_pull_out_clears_stale_held_identity_and_requires_reconciliation() -> None:
    row = next(item for item in T18_GOLDENS if item["skill"] == "pull_out")
    prepared = _prepared_from_golden(row)
    controller = _controller(_Executor())
    controller._recovery_targets = _GoldenRecoveryTargets(row)
    controller._pending_insertion_lift = PendingInsertionLift(
        prepared, False, VerificationReason.NOT_SEATED
    )
    controller._part_held = True
    controller._held_instance_id = prepared.part.instance_id
    controller.state_machine.start(prepared.record.product)
    controller.state_machine.transition(PackMLState.EXECUTE)
    intent = controller._pull_out_intent(
        Waypoint(*GrootAssemblyController._command_tcp(prepared.target, prepared), 1.0)
    )
    endpoint = Waypoint(*intent.endpoint_tcp, 0.0)

    proceed = controller._prepare_after_insertion_lift({}, endpoint, 10.0)

    assert proceed is False
    assert controller._pending_insertion_lift is None
    assert controller._part_held is False
    assert controller._held_instance_id is None
    assert controller._release_dispatched is True
    assert controller.state_machine.state is PackMLState.HELD
    assert controller.state_machine.current_record.reason == (
        "GR00T pull-out set-down requires fresh operator reconciliation before retry"
    )


def test_scripted_motion_coordinates_never_enter_the_cell_semantic_queue() -> None:
    executor = _Executor()
    controller = _controller(executor)
    absurd = Waypoint(999.0, 998.0, 997.0, 180.0, 45.0, 90.0, 1.0)

    _queue_rear_guard_inspection(controller, (MotionChunk(MotionPhase.APPROACH, (absurd,)),))

    intent = controller._motion_queue._intent
    assert isinstance(intent, FixedInspectionMove)
    assert intent.slot_id == "slot-a"
    assert intent.endpoint_tcp == (321.0, -45.0, -168.0, -180.0, 0.0, 77.0)
    assert intent.label == "non_learned_inspection"
    assert intent.source == "reviewed CSI slot-pose test fixture"
    assert 999.0 not in intent.endpoint_tcp
    assert executor.intents == []


def test_missing_reviewed_inspection_slot_holds_and_never_queues_motion() -> None:
    executor = _Executor()
    missing = InspectionPoseRegistry(
        (
            InspectionSlotPose(
                slot_id="other-slot",
                selection_bounds_xy_mm=(400.0, 420.0, 200.0, 220.0),
                endpoint_tcp=(410.0, 210.0, -168.0, -180.0, 0.0, 0.0),
                safe_z_mm=-180.0,
                source="reviewed CSI slot-pose test fixture",
            ),
        )
    )
    controller = _controller(executor, missing)
    candidate = _rear_guard_candidate()
    controller._pending_pre_pick = candidate
    controller._coarse_xy_by_instance[candidate.part.instance_id] = (250.0, -100.0)
    current = Waypoint(400.0, 0.0, -200.0, -180.0, 0.0, 15.0, 0.0)

    queued = controller._set_guarded_motion(
        (), {}, current, 10.0, candidate.part.instance_id, candidate.part.object_id,
        candidate.base.instance_id, candidate.base_pose,
    )

    assert queued is False
    assert controller.state_machine.state is PackMLState.HELD
    assert "no reviewed inspection slot" in controller.state_machine.current_record.reason
    assert executor.inspections == []


def test_guard_trip_enters_packml_held_with_andon_reason() -> None:
    hold = [400.0, 0.0, -200.0, -180.0, 0.0, 15.0, 0.0]
    executor = _Executor(
        GrootExecutionResult(hold, MotionPhase.INSPECT, False, True, ("outside workspace",))
    )
    controller = _controller(executor)
    _queue_rear_guard_inspection(controller, ())
    current = Waypoint(*hold)

    action = controller._emit_guarded_action(
        {"t": 10.0, "tcp_pose": hold[:6], "joints": [0.0] * 7},
        current,
        10.0,
        "part-4",
    )

    assert action == hold
    assert controller.state_machine.state is PackMLState.HELD
    assert "GR00T guard trip: outside workspace" in controller.state_machine.current_record.reason
    assert controller.events[-1].kind == "andon"


def test_guarded_camera_move_is_operator_visible_as_non_learned() -> None:
    action = [250.0, -100.0, -168.0, -180.0, 0.0, 15.0, 0.0]
    executor = _Executor(
        GrootExecutionResult(action, MotionPhase.INSPECT, False, False, ())
    )
    controller = _controller(executor)
    _queue_rear_guard_inspection(controller, ())
    current = Waypoint(400.0, 0.0, -200.0, -180.0, 0.0, 15.0, 0.0)

    emitted = controller._emit_guarded_action(
        {"t": 10.0, "tcp_pose": current.as_action()[:6], "joints": [0.0] * 7},
        current,
        10.0,
        "part-4",
    )

    assert emitted == action
    assert executor.intents == []
    assert executor.inspections[0].label == "non_learned_inspection"
    assert controller.events[-1].kind == "inspection_motion"
    assert "non_learned_inspection" in controller.events[-1].detail
