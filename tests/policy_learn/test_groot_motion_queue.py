"""CELL semantic queue: GR00T rows are the only motion authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from assembly.catalog import TRUSTED_GRASP_SOURCE
from assembly.controller_ports import ExecutionTolerances
from assembly.execution import ExecutionError
from assembly.motion import MotionPhase, Waypoint
from assembly.motion_intent import ContactEnvelope, MotionIntent, SemanticSkill
from policy_learn.deploy.cell_executor import GrootExecutionResult
from policy_learn.deploy.groot_queue import GrootGuardTrip, GrootMotionQueue


def _intent() -> MotionIntent:
    return MotionIntent(
        skill=SemanticSkill.PICK_INSERT_HOLD,
        instance_id="part-1",
        object_id="M1_02",
        base_instance_id="base-1",
        base_object_id="F01",
        grip_index=0,
        case="nominal",
        object_pose=(300.0, -50.0, 30.0),
        object_origin_z_mm=-281.4,
        wrist_roll_deg=120.0,
        goal_pose=(400.0, 20.0, 0.0),
        grasp_tcp=(299.1, -48.441154, -270.0, -180.0, 0.0, 120.0),
        release_tcp=None,
        endpoint_tcp=(402.0, 20.0, -300.0, -180.0, 0.0, 0.0),
        final_gripper=1.0,
        pose_source="T30 measurement + T29 grip 0",
        grasp_source_sha256=TRUSTED_GRASP_SOURCE.sha256,
        contact=ContactEnvelope(400.0, 20.0, -300.0, 18.0, "M1_02 on F01", "reviewed envelope"),
        release_authorized=False,
    )


@dataclass
class _Executor:
    result: GrootExecutionResult

    def __post_init__(self) -> None:
        self.intents: list[MotionIntent] = []

    def act(self, intent, *, observation, scene):
        del observation
        assert scene is None
        self.intents.append(intent)
        return self.result


class _UnavailableExecutor:
    def act(self, intent, *, observation, scene):
        del intent, observation, scene
        raise TimeoutError("fake GR00T server timed out")


def _observation(tcp: tuple[float, ...]) -> dict:
    return {"t": 1.0, "tcp_pose": list(tcp), "joints": [0.0] * 7}


def test_queue_emits_only_the_executor_row_and_confirms_it_from_fresh_observation() -> None:
    action = (301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    executor = _Executor(GrootExecutionResult(list(action), MotionPhase.APPROACH, False, False, ()))
    queue = GrootMotionQueue(executor, ExecutionTolerances(0.3, 1.0))
    queue.queue(_intent())

    emitted = queue.emit(
        observation=_observation((300.0, 0.0, -200.0, -180.0, 0.0, 0.0)),
        current=Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0),
        observed_at_s=1.0,
    )

    assert emitted == list(action)
    assert executor.intents == [_intent()]
    queue.confirm_previous_endpoint(Waypoint(*action))


def test_guard_trip_is_typed_and_never_becomes_a_motion_row() -> None:
    hold = [300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0]
    result = GrootExecutionResult(hold, MotionPhase.APPROACH, False, True, ("outside workspace",))
    queue = GrootMotionQueue(_Executor(result), ExecutionTolerances(0.3, 1.0))
    queue.queue(_intent())

    with pytest.raises(GrootGuardTrip, match="outside workspace"):
        queue.emit(
            observation=_observation(tuple(hold[:6])),
            current=Waypoint(*hold),
            observed_at_s=1.0,
        )

    assert queue.active is False


def test_transport_failure_becomes_typed_hold_path_and_clears_authority() -> None:
    hold = [300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0]
    queue = GrootMotionQueue(_UnavailableExecutor(), ExecutionTolerances(0.3, 1.0))
    queue.queue(_intent())

    with pytest.raises(ExecutionError, match="GR00T boundary failed.*timed out"):
        queue.emit(
            observation=_observation(tuple(hold[:6])),
            current=Waypoint(*hold),
            observed_at_s=1.0,
        )

    assert queue.active is False


def test_scripted_motion_chunks_are_never_accepted_as_cell_authority() -> None:
    action = [300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0]
    result = GrootExecutionResult(action, MotionPhase.APPROACH, False, False, ())
    queue = GrootMotionQueue(_Executor(result), ExecutionTolerances(0.3, 1.0))

    with pytest.raises(ExecutionError, match="MotionIntent only"):
        queue.queue(cast(MotionIntent, ("scripted", "waypoints")))
