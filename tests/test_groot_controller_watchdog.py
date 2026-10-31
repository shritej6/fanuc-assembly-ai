"""Controller-level Held/Andon behavior for CELL settlement and watchdog trips."""

from dataclasses import replace

import pytest

from assembly.fixed_inspection import FixedInspectionMove
from assembly.groot_controller import GrootAssemblyController
from assembly.motion import MotionPhase, Waypoint
from assembly.planner import RecipePlanner
from assembly.state_machine import PackMLState
from policy_learn.deploy.cell_executor import GrootExecutionResult
from tests.test_groot_controller import (
    _controller,
    _Executor,
    _rear_guard_candidate,
    _reviewed_rear_guard_grasp,
)


def _observation(current: Waypoint, *, t: float, grip: float | None = None) -> dict:
    return {
        "t": t,
        "tcp_pose": current.as_action()[:6],
        "joints": [0.0] * 6 + [current.gripper if grip is None else grip],
        "trial": {"instruction": "assemble product 1"},
    }


def _reserve_product_one(controller, observation: dict) -> None:
    controller._bind_instruction(observation)
    controller.inventory.add("base-1", "F01")
    for step in range(1, 5):
        controller.inventory.add(f"p1-{step}", f"M1_{step:02d}")
    controller.inventory.reserve_product(1)
    controller.state_machine.start(1)
    controller.state_machine.transition(PackMLState.EXECUTE)


def test_endpoint_settle_timeout_holds_with_andon_and_keeps_reservation() -> None:
    current = Waypoint(300.0, -45.0, -168.0, -180.0, 0.0, 77.0, 0.0)
    target = [321.0, -45.0, -168.0, -180.0, 0.0, 77.0, 0.0]
    executor = _Executor(GrootExecutionResult(target, MotionPhase.INSPECT, False, False, ()))
    controller = _controller(executor)
    observation = _observation(current, t=10.0)
    _reserve_product_one(controller, observation)
    controller._motion_queue.queue(
        FixedInspectionMove(
            "p1-1", "base-1", "slot-a", tuple(target[:6]), -180.0,
            "reviewed CSI slot-pose test fixture",
        ),
        observation=observation, current=current, observed_at_s=10.0,
        base_pose=(420.0, 80.0, -320.0, 90.0),
    )
    controller._emit_guarded_action(observation, current, 10.0, "p1-1")

    action = controller.act(_observation(current, t=13.0))

    assert action == current.as_action()
    assert controller.state_machine.state is PackMLState.HELD
    assert controller.inventory.active_product == 1
    assert controller.events[-1].kind == "andon"
    assert "did not settle within 3 s" in controller.events[-1].detail
    assert len(executor.inspections) == 1


def test_periodic_drop_evidence_holds_before_next_model_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _rear_guard_candidate()
    prepared = RecipePlanner.finalize_candidate(candidate, _reviewed_rear_guard_grasp(candidate))
    current = Waypoint(*GrootAssemblyController._command_tcp(prepared.grasp, prepared), 0.0)
    closed = Waypoint(*current.as_action()[:6], 1.0)
    executor = _Executor(GrootExecutionResult(closed.as_action(), MotionPhase.GRIP, False, False, ()))
    controller = _controller(executor)
    intent = controller._pick_insert_intent(prepared)
    observation = _observation(current, t=10.0)
    controller._motion_queue.queue(
        intent, observation=observation, current=current, observed_at_s=10.0,
        base_pose=(420.0, 80.0, -320.0, 90.0),
    )
    controller._emit_guarded_action(observation, current, 10.0, prepared.part.instance_id)
    original_scene = controller._perception.collision_scene

    def dropped_scene(observation, moving_instance_id, base_instance_id):
        return replace(
            original_scene(observation, moving_instance_id, base_instance_id),
            moving_attached=False,
        )

    monkeypatch.setattr(controller._perception, "collision_scene", dropped_scene)
    action = controller._confirm_previous_endpoint(
        _observation(closed, t=10.1), closed, 10.1
    )

    assert action == closed.as_action()
    assert controller.state_machine.state is PackMLState.HELD
    assert controller.events[-1].kind == "andon"
    assert "attachment is lost" in controller.events[-1].detail
    assert len(executor.intents) == 1


def test_watchdog_clock_failure_holds_active_trial_instead_of_aborting() -> None:
    current = Waypoint(300.0, -45.0, -168.0, -180.0, 0.0, 77.0, 0.0)
    target = [321.0, -45.0, -168.0, -180.0, 0.0, 77.0, 0.0]
    executor = _Executor(GrootExecutionResult(target, MotionPhase.INSPECT, False, False, ()))
    controller = _controller(executor)
    observation = _observation(current, t=10.0)
    _reserve_product_one(controller, observation)
    controller._motion_queue._clock = lambda: 10.0
    controller._motion_queue.queue(
        FixedInspectionMove(
            "p1-1", "base-1", "slot-a", tuple(target[:6]), -180.0,
            "reviewed CSI slot-pose test fixture",
        ),
        observation=observation, current=current, observed_at_s=10.0,
        base_pose=(420.0, 80.0, -320.0, 90.0),
    )
    controller._emit_guarded_action(observation, current, 10.0, "p1-1")
    controller._motion_queue._last_clock_s = 11.0

    action = controller._confirm_previous_endpoint(observation, current, 10.0)

    assert action == current.as_action()
    assert controller.state_machine.state is PackMLState.HELD
    assert controller.inventory.active_product == 1
    assert controller.events[-1].kind == "andon"
    assert "clock moved backwards" in controller.events[-1].detail


def test_fixed_inspection_leg_progress_does_not_reset_time_budget() -> None:
    current = Waypoint(300.0, -45.0, -168.0, -180.0, 0.0, 77.0, 0.0)
    target = [321.0, -45.0, -168.0, -180.0, 0.0, 77.0, 0.0]
    executor = _Executor(GrootExecutionResult(target, MotionPhase.INSPECT, False, False, ()))
    controller = _controller(executor)
    observation = _observation(current, t=10.0)
    _reserve_product_one(controller, observation)
    clock = [0.0]
    controller._motion_queue._clock = lambda: clock[0]
    controller._motion_queue.queue(
        FixedInspectionMove(
            "p1-1", "base-1", "slot-a", tuple(target[:6]), -180.0,
            "reviewed CSI slot-pose test fixture",
        ),
        observation=observation, current=current, observed_at_s=10.0,
        base_pose=(420.0, 80.0, -320.0, 90.0),
    )
    clock[0] = 1.0
    assert controller._emit_guarded_action(
        observation, current, 10.0, "p1-1"
    ) == target
    reached = Waypoint(*target)
    assert controller._confirm_previous_endpoint(
        _observation(reached, t=10.1), reached, 10.1
    ) is None
    clock[0] = 31.0

    action = controller._emit_guarded_action(
        _observation(reached, t=10.2), reached, 10.2, "p1-1"
    )

    assert action == reached.as_action()
    assert controller.state_machine.state is PackMLState.HELD
    assert controller.inventory.active_product == 1
    assert controller.events[-1].kind == "andon"
    assert "30 s time budget" in controller.events[-1].detail
    assert len(executor.inspections) == 1
