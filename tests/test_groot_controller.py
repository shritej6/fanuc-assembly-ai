"""T09 CELL controller keeps T29 frames and GR00T motion authority separate."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pytest

from assembly.catalog import load_catalog
from assembly.fixed_inspection import FixedInspectionMove
from assembly.groot_controller import GrootAssemblyController
from assembly.inventory import Inventory
from assembly.motion import AbsolutePoseUF0, MotionChunk, MotionPhase, OrientedPartPose, RestingState, Waypoint
from assembly.motion_intent import MotionIntent
from assembly.planner import PlanCandidate, PreparedPlanStep, RecipePlanner
from assembly.state_machine import PackMLState
from policy_learn.deploy.cell_executor import GrootExecutionResult
from robot.cell.t09_mock_runtime import build_groot_mock_controller

OBJECT_ORIGIN = (300.0, -50.0, -300.0)
OBJECT_YAW_DEG = 37.0


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


@dataclass
class _Executor:
    result: GrootExecutionResult | None = None

    def __post_init__(self) -> None:
        self.resets = 0
        self.intents: list[MotionIntent] = []
        self.inspections: list[FixedInspectionMove] = []

    def reset(self) -> None:
        self.resets += 1

    def act(self, intent, *, observation, scene):
        del observation
        assert scene is None
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


def _controller(executor: _Executor) -> GrootAssemblyController:
    controller = build_groot_mock_controller(executor)
    assert isinstance(controller, GrootAssemblyController)
    return controller


def _queue_rear_guard_inspection(controller: GrootAssemblyController, plan: tuple[MotionChunk, ...]) -> None:
    candidate = _rear_guard_candidate()
    candidate = replace(candidate, target=replace(candidate.target, up_face="mock-installed:+z"))
    controller._pending_pre_pick = candidate
    controller._coarse_xy_by_instance[candidate.part.instance_id] = (250.0, -100.0)
    current = Waypoint(400.0, 0.0, -200.0, -180.0, 0.0, 15.0, 0.0)
    queued = controller._set_guarded_motion(
        plan,
        {"t": 10.0},
        current,
        10.0,
        candidate.part.instance_id,
        candidate.part.object_id,
        candidate.base.instance_id,
        candidate.base_pose,
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


def test_scripted_motion_coordinates_never_enter_the_cell_semantic_queue() -> None:
    executor = _Executor()
    controller = _controller(executor)
    absurd = Waypoint(999.0, 998.0, 997.0, 180.0, 45.0, 90.0, 1.0)

    _queue_rear_guard_inspection(controller, (MotionChunk(MotionPhase.APPROACH, (absurd,)),))

    intent = controller._motion_queue._intent
    assert isinstance(intent, FixedInspectionMove)
    assert intent.endpoint_tcp[:3] == (250.0, -100.0, -168.0)
    assert intent.label == "non_learned_inspection"
    assert "measured CSI wrist inspection stop" in intent.source
    assert 999.0 not in intent.endpoint_tcp
    assert executor.intents == []


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
