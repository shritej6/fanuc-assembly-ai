"""CELL semantic queue: GR00T rows are the only motion authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from assembly.catalog import TRUSTED_GRASP_SOURCE
from assembly.controller_ports import CollisionScene, ExecutionTolerances, PerceptionTolerances
from assembly.execution import (
    EndpointSettleLimits,
    EndpointSettleTimeout,
    ExecutionError,
    GrootWatchdogLimits,
)
from assembly.motion import MotionPhase, Waypoint
from assembly.motion_intent import ContactEnvelope, MotionIntent, SemanticSkill
from policy_learn.deploy.cell_executor import GrootExecutionResult
from policy_learn.deploy.groot_queue import GrootGuardTrip, GrootMotionQueue

SETTLE = EndpointSettleLimits(2.0, 3.0, "PRELIMINARY", "Saturday queue regression fixture")
WATCHDOG = GrootWatchdogLimits(
    90.0,
    30.0,
    60.0,
    60.0,
    30.0,
    5,
    3.0,
    "MOCK",
    "Saturday queue regression fixture",
)
BASE_POSE = (400.0, 20.0, -330.0, 0.0)


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
        self.scenes: list[CollisionScene] = []

    def act(self, intent, *, observation, scene):
        del observation
        assert isinstance(scene, CollisionScene)
        self.scenes.append(scene)
        self.intents.append(intent)
        return self.result


class _UnavailableExecutor:
    def act(self, intent, *, observation, scene):
        del intent, observation, scene
        raise TimeoutError("fake GR00T server timed out")


class _Perception:
    def collision_scene(self, observation, moving_instance_id, base_instance_id):
        return CollisionScene(
            float(observation["t"]),
            True,
            base_instance_id,
            BASE_POSE,
            0.0,
            moving_instance_id,
            float(observation["joints"][6]) == 1.0,
            False,
        )


def _observation(tcp: tuple[float, ...]) -> dict:
    return {"t": 1.0, "tcp_pose": list(tcp), "joints": [0.0] * 7}


def _queue(executor) -> GrootMotionQueue:
    return GrootMotionQueue(
        executor,
        ExecutionTolerances(0.3, 1.0),
        SETTLE,
        perception=_Perception(),
        perception_tolerances=PerceptionTolerances(1.8, 3.0),
        watchdog_limits=WATCHDOG,
        clock=lambda: 0.0,
    )


def _start(queue: GrootMotionQueue, intent: MotionIntent | object = None) -> None:
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    queue.queue(
        _intent() if intent is None else intent,
        observation=_observation(tuple(current.as_action()[:6])),
        current=current,
        observed_at_s=1.0,
        base_pose=BASE_POSE,
    )


def _confirm(queue: GrootMotionQueue, current: Waypoint, observed_at_s: float) -> list[float] | None:
    observation = _observation(tuple(current.as_action()[:6]))
    observation["t"] = observed_at_s
    observation["joints"][-1] = current.gripper
    return queue.confirm_previous_endpoint(observation, current, observed_at_s)


def test_queue_emits_only_the_executor_row_and_confirms_it_from_fresh_observation() -> None:
    action = (301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    executor = _Executor(GrootExecutionResult(list(action), MotionPhase.APPROACH, False, False, ()))
    queue = _queue(executor)
    _start(queue)

    emitted = queue.emit(
        observation=_observation((300.0, 0.0, -200.0, -180.0, 0.0, 0.0)),
        current=Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0),
        observed_at_s=1.0,
    )

    assert emitted == list(action)
    assert executor.intents == [_intent()]
    _confirm(queue, Waypoint(*action), 1.1)


def test_guard_trip_is_typed_and_never_becomes_a_motion_row() -> None:
    hold = [300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0]
    result = GrootExecutionResult(hold, MotionPhase.APPROACH, False, True, ("outside workspace",))
    queue = _queue(_Executor(result))
    _start(queue)

    with pytest.raises(GrootGuardTrip, match="outside workspace"):
        queue.emit(
            observation=_observation(tuple(hold[:6])),
            current=Waypoint(*hold),
            observed_at_s=1.0,
        )

    assert queue.active is False


def test_transport_failure_becomes_typed_hold_path_and_clears_authority() -> None:
    hold = [300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0]
    queue = _queue(_UnavailableExecutor())
    _start(queue)

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
    queue = _queue(_Executor(result))

    with pytest.raises(ExecutionError, match="MotionIntent only"):
        _start(queue, cast(MotionIntent, ("scripted", "waypoints")))


def test_endpoint_confirmation_polls_without_loosening_position_or_angle_gate() -> None:
    action = (301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    executor = _Executor(GrootExecutionResult(list(action), MotionPhase.APPROACH, False, False, ()))
    queue = _queue(executor)
    _start(queue)
    queue.emit(
        observation=_observation((300.0, 0.0, -200.0, -180.0, 0.0, 0.0)),
        current=Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0),
        observed_at_s=1.0,
    )

    position_pending = Waypoint(300.69, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    assert _confirm(queue, position_pending, 1.5) == list(action)
    angle_pending = Waypoint(301.0, 0.0, -200.0, 178.99, 0.0, 0.0, 0.0)
    assert _confirm(queue, angle_pending, 2.0) == list(action)

    settled = Waypoint(300.71, 0.0, -200.0, 179.01, 0.0, 0.0, 0.0)
    assert _confirm(queue, settled, 2.5) is None
    assert queue.pending_endpoint is None
    assert executor.intents == [_intent()]


def test_endpoint_settlement_reissues_original_guarded_endpoint() -> None:
    action = (301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    queue = _queue(
        _Executor(GrootExecutionResult(list(action), MotionPhase.APPROACH, False, False, ()))
    )
    _start(queue)
    start = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    queue.emit(
        observation=_observation(tuple(start.as_action()[:6])),
        current=start,
        observed_at_s=1.0,
    )
    lagged = Waypoint(300.25, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)

    assert _confirm(queue, lagged, 1.5) == list(action)
    assert queue.pending_endpoint == Waypoint(*action)


def test_gripper_settlement_reissues_only_gripper_at_measured_tcp() -> None:
    action = (300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 1.0)
    queue = _queue(
        _Executor(GrootExecutionResult(list(action), MotionPhase.GRIP, False, False, ()))
    )
    _start(queue)
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    queue.emit(observation=_observation(tuple(current.as_action()[:6])), current=current, observed_at_s=1.0)

    assert _confirm(queue, current, 3.9) == [*current.as_action()[:6], 1.0]


def test_endpoint_settle_deadline_never_resets_and_times_out_typed() -> None:
    action = (301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    queue = _queue(
        _Executor(GrootExecutionResult(list(action), MotionPhase.APPROACH, False, False, ()))
    )
    _start(queue)
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    queue.emit(observation=_observation(tuple(current.as_action()[:6])), current=current, observed_at_s=1.0)

    assert _confirm(queue, current, 2.0) == list(action)
    assert _confirm(queue, current, 2.9) == list(action)
    with pytest.raises(EndpointSettleTimeout, match="0.3 mm / 1.0 deg"):
        _confirm(queue, current, 3.0)

    assert queue.active is False
    assert queue.pending_endpoint is None


def test_queue_refuses_another_model_row_while_endpoint_is_unsettled() -> None:
    action = (301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    executor = _Executor(
        GrootExecutionResult(list(action), MotionPhase.APPROACH, False, False, ())
    )
    queue = _queue(executor)
    _start(queue)
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    observation = _observation(tuple(current.as_action()[:6]))

    queue.emit(observation=observation, current=current, observed_at_s=1.0)

    with pytest.raises(ExecutionError, match="previous endpoint.*settle"):
        queue.emit(observation=observation, current=current, observed_at_s=1.1)

    assert executor.intents == [_intent()]
