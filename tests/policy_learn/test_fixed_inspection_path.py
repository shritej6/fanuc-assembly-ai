"""Fixed inspection poses are guarded camera moves, never learned part motion."""

from __future__ import annotations

import numpy as np
import pytest

from assembly.controller_ports import ExecutionTolerances
from assembly.execution import EndpointSettleLimits
from assembly.fixed_inspection import FixedInspectionMove
from assembly.motion import MotionPhase, Waypoint
from policy_learn.deploy.cell_executor import CellExecutionError, T09GrootExecutor
from policy_learn.deploy.groot_policy import CellPolicySafetyError, GrootPolicy
from policy_learn.deploy.groot_queue import GrootMotionQueue
from policy_learn.deploy.guards import GuardLimits

SETTLE = EndpointSettleLimits(2.0, 3.0, "PRELIMINARY", "fixed inspection test fixture")


class _ForbiddenModel:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, observation: dict) -> np.ndarray:
        del observation
        self.calls += 1
        raise AssertionError("fixed inspection must not invoke the learned model")


def _observation(*, tcp: tuple[float, ...], gripper: float = 0.0) -> dict:
    return {
        "t": 1.0,
        "tcp_pose": list(tcp),
        "joints": [0.0] * 6 + [gripper],
        "images": {},
        "depth": {},
        "trial": {"instruction": "assemble product 1"},
    }


def _move(
    endpoint: tuple[float, ...],
    *,
    safe_z_mm: float = -200.0,
) -> FixedInspectionMove:
    return FixedInspectionMove(
        instance_id="part-1",
        base_instance_id="base-1",
        slot_id="slot-a",
        endpoint_tcp=endpoint,
        safe_z_mm=safe_z_mm,
        source="reviewed CSI inspection stop; observed inventory slot XY",
    )


def _executor(model: _ForbiddenModel) -> T09GrootExecutor:
    policy = GrootPolicy(model=model, limits=GuardLimits())
    return T09GrootExecutor(
        policy=policy,
        safety=None,
        tolerances=ExecutionTolerances(0.3, 1.0),
        require_scene_safety=False,
    )


def test_fixed_inspection_refuses_closed_gripper_without_release_authority() -> None:
    model = _ForbiddenModel()
    executor = _executor(model)
    current = (300.0, 0.0, -200.0, -180.0, 0.0, 0.0)
    move = _move((310.0, 0.0, -200.0, -180.0, 0.0, 0.0))

    with pytest.raises(CellExecutionError, match="open gripper"):
        executor.inspect(move, observation=_observation(tcp=current, gripper=1.0))

    assert model.calls == 0


def test_policy_fixed_inspection_refuses_closed_gripper_direct_call() -> None:
    model = _ForbiddenModel()
    policy = GrootPolicy(model=model, limits=GuardLimits())
    current = (300.0, 0.0, -200.0, -180.0, 0.0, 0.0)

    with pytest.raises(CellPolicySafetyError, match="open gripper"):
        policy.guard_fixed_inspection(
            _observation(tcp=current, gripper=1.0),
            _move((310.0, 0.0, -200.0, -180.0, 0.0, 0.0)),
        )

    assert model.calls == 0


def test_fixed_inspection_outside_workspace_is_a_typed_guard_hold() -> None:
    model = _ForbiddenModel()
    executor = _executor(model)
    current = (300.0, 0.0, -200.0, -180.0, 0.0, 0.0)

    result = executor.inspect(
        _move((700.0, 0.0, -200.0, -180.0, 0.0, 0.0)),
        observation=_observation(tcp=current),
    )

    assert result.action == [*current, 0.0]
    assert result.blocked is True
    assert any("outside workspace" in reason for reason in result.reasons)
    assert model.calls == 0


def test_cell_queue_dispatches_fixed_inspection_separately_from_semantic_motion() -> None:
    model = _ForbiddenModel()
    executor = _executor(model)
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    queue = GrootMotionQueue(executor, ExecutionTolerances(0.3, 1.0), SETTLE)
    queue.queue(_move((310.0, 0.0, -200.0, -180.0, 0.0, 0.0)))

    action = queue.emit(
        observation=_observation(tcp=tuple(current.as_action()[:6])),
        current=current,
        observed_at_s=1.0,
    )

    assert action == [310.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0]
    assert queue.base_instance_id == "base-1"
    assert queue.last_emitted_phase is MotionPhase.INSPECT
    assert model.calls == 0


def test_fixed_inspection_rises_to_sourced_safe_z_before_lateral_motion() -> None:
    model = _ForbiddenModel()
    executor = _executor(model)
    current = (300.0, 0.0, -250.0, -180.0, 0.0, 0.0)

    result = executor.inspect(
        _move((500.0, 100.0, -168.0, -180.0, 0.0, 0.0), safe_z_mm=-180.0),
        observation=_observation(tcp=current),
    )

    assert result.blocked is False
    assert result.action[0:2] == [300.0, 0.0]
    assert current[2] < result.action[2] <= -180.0
    assert model.calls == 0


def test_fixed_inspection_far_lateral_leg_is_incremental_not_guard_refused() -> None:
    model = _ForbiddenModel()
    executor = _executor(model)
    current = (300.0, 0.0, -180.0, -180.0, 0.0, 0.0)

    result = executor.inspect(
        _move((500.0, 0.0, -180.0, -180.0, 0.0, 0.0), safe_z_mm=-180.0),
        observation=_observation(tcp=current),
    )

    assert result.blocked is False
    assert 300.0 < result.action[0] <= 330.0
    assert result.action[1:3] == [0.0, -180.0]
    assert model.calls == 0
