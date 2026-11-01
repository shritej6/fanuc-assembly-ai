"""Intent-wide CELL watchdog and recurring physical-scene evidence.
Authored by Shritej Gunda.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from assembly.catalog import TRUSTED_GRASP_SOURCE
from assembly.controller_ports import (
    CollisionScene,
    ExecutionTolerances,
    PerceptionTolerances,
)
from assembly.execution import EndpointSettleLimits, GrootWatchdogLimits
from assembly.motion import MotionPhase, Waypoint
from assembly.motion_intent import ContactEnvelope, MotionIntent, SemanticSkill
from policy_learn.deploy.cell_executor import GrootExecutionResult
from policy_learn.deploy.groot_queue import GrootGuardTrip, GrootMotionQueue

BASE_POSE = (400.0, 20.0, -330.0, 0.0)
SETTLE = EndpointSettleLimits(2.0, 3.0, "MOCK", "watchdog test settle fixture")
WATCHDOG = GrootWatchdogLimits(
    pick_insert_s=90.0,
    release_retreat_s=30.0,
    pull_out_s=60.0,
    defect_drop_s=60.0,
    inspection_s=30.0,
    scene_interval_rows=2,
    base_tilt_limit_deg=3.0,
    status="PRELIMINARY",
    source="Engineering minimum; pending CSI measurement",
)

SPLIT_WATCHDOG = replace(
    WATCHDOG,
    pick_s=60.0,
    insert_s=90.0,
    scene_interval_s=1.0,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _Perception:
    def __init__(self, scenes: list[CollisionScene] | None = None) -> None:
        self.scenes = scenes or [_scene()]
        self.calls = 0

    def collision_scene(self, observation, moving_instance_id, base_instance_id):
        scene = self.scenes[min(self.calls, len(self.scenes) - 1)]
        self.calls += 1
        return replace(
            scene,
            observed_at_s=float(observation["t"]),
            moving_instance_id=moving_instance_id,
            base_instance_id=base_instance_id,
        )


class _ExactPerception(_Perception):
    def collision_scene(self, observation, moving_instance_id, base_instance_id):
        del observation, moving_instance_id, base_instance_id
        scene = self.scenes[min(self.calls, len(self.scenes) - 1)]
        self.calls += 1
        return scene


@dataclass
class _Executor:
    results: list[GrootExecutionResult]
    clock: _Clock
    delay_s: float = 0.0

    def __post_init__(self) -> None:
        self.calls = 0
        self.scenes: list[CollisionScene | None] = []

    def act(self, intent, *, observation, scene):
        del intent, observation
        self.scenes.append(scene)
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        self.clock.advance(self.delay_s)
        return result


def _intent() -> MotionIntent:
    return MotionIntent(
        SemanticSkill.PICK_INSERT_HOLD,
        "part-1",
        "M1_02",
        "base-1",
        "F01",
        0,
        "nominal",
        (300.0, -50.0, 30.0),
        -281.4,
        120.0,
        (400.0, 20.0, 0.0),
        (299.1, -48.441154, -270.0, -180.0, 0.0, 120.0),
        None,
        (402.0, 20.0, -300.0, -180.0, 0.0, 0.0),
        1.0,
        "T30 measurement + T29 grip 0",
        TRUSTED_GRASP_SOURCE.sha256,
        ContactEnvelope(400.0, 20.0, -300.0, 18.0, "M1_02 on F01", "reviewed envelope"),
        False,
    )


def _scene(**changes) -> CollisionScene:
    values = {
        "observed_at_s": 1.0,
        "complete": True,
        "base_instance_id": "base-1",
        "base_pose": BASE_POSE,
        "base_tilt_deg": 0.0,
        "moving_instance_id": "part-1",
        "moving_attached": False,
        "intrusion_detected": False,
    }
    values.update(changes)
    return CollisionScene(**values)


def _observation(t: float, current: Waypoint) -> dict:
    return {
        "t": t,
        "tcp_pose": current.as_action()[:6],
        "joints": [0.0] * 6 + [current.gripper],
    }


def _queue(
    executor: _Executor,
    perception: _Perception,
    clock: _Clock,
    limits: GrootWatchdogLimits = WATCHDOG,
) -> GrootMotionQueue:
    return GrootMotionQueue(
        executor,
        ExecutionTolerances(0.3, 1.0),
        SETTLE,
        perception=perception,
        perception_tolerances=PerceptionTolerances(1.8, 3.0),
        watchdog_limits=limits,
        clock=clock,
    )


def _start(queue: GrootMotionQueue, current: Waypoint, *, t: float = 1.0) -> None:
    queue.queue(
        _intent(),
        observation=_observation(t, current),
        current=current,
        observed_at_s=t,
        base_pose=BASE_POSE,
    )


def test_closed_gripper_without_attached_part_blocks_before_model_call() -> None:
    clock = _Clock()
    executor = _Executor([_result(301.0, grip=1.0)], clock)
    queue = _queue(executor, _Perception([_scene(moving_attached=False)]), clock)
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 1.0)

    with pytest.raises(GrootGuardTrip, match="attachment.*lost"):
        _start(queue, current)

    assert executor.calls == 0
    assert queue.active is False


def test_model_row_finishing_after_intent_deadline_is_discarded() -> None:
    clock = _Clock()
    executor = _Executor([_result(301.0)], clock, delay_s=91.0)
    queue = _queue(executor, _Perception(), clock)
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    _start(queue, current)

    with pytest.raises(GrootGuardTrip, match="90.*time budget.*discarded"):
        queue.emit(
            observation=_observation(1.0, current),
            current=current,
            observed_at_s=1.0,
        )

    assert executor.calls == 1
    assert queue.active is False
    assert queue.pending_endpoint is None


def test_pick_deadline_expires_at_sixty_seconds_before_gripper_close() -> None:
    clock = _Clock()
    executor = _Executor([_result(301.0)], clock)
    queue = _queue(executor, _Perception(), clock, SPLIT_WATCHDOG)
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    _start(queue, current)
    clock.advance(60.0)

    with pytest.raises(GrootGuardTrip, match="pick.*60.*time budget"):
        queue.emit(
            observation=_observation(1.0, current),
            current=current,
            observed_at_s=1.0,
        )

    assert executor.calls == 0
    assert queue.active is False


def test_first_observed_close_starts_one_nonresetting_insert_deadline() -> None:
    clock = _Clock()
    executor = _Executor(
        [_result(301.0, grip=1.0), _result(302.0, grip=1.0), _result(303.0, grip=1.0)],
        clock,
    )
    perception = _Perception([_scene(), _scene(moving_attached=True)])
    queue = _queue(executor, perception, clock, SPLIT_WATCHDOG)
    open_gripper = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    _start(queue, open_gripper)
    clock.advance(10.0)
    current = Waypoint(*open_gripper.as_action()[:6], 1.0)

    current = _emit_and_confirm(queue, current, 1.0)
    clock.advance(44.0)
    current = _emit_and_confirm(queue, current, 1.0)
    clock.advance(46.0)

    with pytest.raises(GrootGuardTrip, match="insert.*90.*time budget"):
        queue.emit(
            observation=_observation(1.0, current),
            current=current,
            observed_at_s=1.0,
        )

    assert executor.calls == 2
    assert queue.active is False


def test_intent_deadline_keeps_running_during_endpoint_settlement() -> None:
    clock = _Clock()
    executor = _Executor([_result(301.0)], clock)
    queue = _queue(executor, _Perception(), clock)
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    _start(queue, current)
    queue.emit(
        observation=_observation(1.0, current),
        current=current,
        observed_at_s=1.0,
    )

    clock.advance(89.0)
    assert queue.confirm_previous_endpoint(
        _observation(1.5, current), current, 1.5
    ) == [301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0]
    clock.advance(2.0)

    with pytest.raises(GrootGuardTrip, match="90.*time budget"):
        queue.confirm_previous_endpoint(_observation(1.6, current), current, 1.6)

    assert queue.active is False


def test_scene_rechecks_every_n_rows_and_after_observed_gripper_close() -> None:
    clock = _Clock()
    results = [_result(301.0), _result(302.0), _result(302.0, grip=1.0)]
    executor = _Executor(results, clock)
    perception = _Perception(
        [*[_scene() for _ in range(8)], _scene(moving_attached=True)]
    )
    queue = _queue(executor, perception, clock)
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    _start(queue, current)

    current = _emit_and_confirm(queue, current, 1.0)
    assert perception.calls == 3
    current = _emit_and_confirm(queue, current, 2.0)
    assert perception.calls == 6
    current = _emit_and_confirm(queue, current, 3.0)

    assert current.gripper == 1.0
    assert perception.calls == 9
    assert executor.calls == 3


def test_scene_rechecks_after_one_wall_clock_second_with_same_observation_time() -> None:
    clock = _Clock()
    limits = replace(SPLIT_WATCHDOG, scene_interval_rows=100)
    executor = _Executor([_result(301.0)], clock)
    perception = _ExactPerception([_scene(observed_at_s=1.0)])
    queue = _queue(executor, perception, clock, limits)
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    _start(queue, current)
    assert perception.calls == 1

    action = queue.emit(
        observation=_observation(1.0, current),
        current=current,
        observed_at_s=1.0,
    )
    reached = Waypoint(*action)
    calls_after_emit = perception.calls
    clock.advance(1.01)
    queue.confirm_previous_endpoint(_observation(1.0, reached), reached, 1.0)

    assert perception.calls == calls_after_emit + 1


def test_scene_is_revalidated_after_blocking_model_response_before_row_acceptance() -> None:
    clock = _Clock()
    moved = _scene(base_pose=(403.0, 20.0, -330.0, 0.0))
    executor = _Executor([_result(301.0)], clock, delay_s=0.5)
    perception = _ExactPerception([_scene(), moved])
    queue = _queue(executor, perception, clock, SPLIT_WATCHDOG)
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    _start(queue, current)

    with pytest.raises(GrootGuardTrip, match="base moved"):
        queue.emit(
            observation=_observation(1.0, current),
            current=current,
            observed_at_s=1.0,
        )

    assert executor.calls == 1
    assert perception.calls == 2
    assert queue.active is False
    assert queue.pending_endpoint is None


def test_last_validated_scene_is_supplied_between_periodic_refreshes() -> None:
    clock = _Clock()
    executor = _Executor([_result(301.0), _result(302.0)], clock)
    perception = _Perception()
    queue = _queue(executor, perception, clock)
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    _start(queue, current)

    queue.emit(observation=_observation(1.0, current), current=current, observed_at_s=1.0)

    assert perception.calls == 2
    assert executor.scenes == [perception.scenes[0]]


def test_closed_gripper_rechecks_attachment_during_each_settle_poll() -> None:
    clock = _Clock()
    executor = _Executor([_result(302.0, grip=1.0)], clock)
    perception = _Perception(
        [
            _scene(),
            _scene(),
            _scene(moving_attached=True),
            _scene(moving_attached=False),
        ]
    )
    queue = _queue(executor, perception, clock)
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    _start(queue, current)
    queue.emit(observation=_observation(1.0, current), current=current, observed_at_s=1.0)
    settling = Waypoint(301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 1.0)

    assert queue.confirm_previous_endpoint(
        _observation(1.1, settling), settling, 1.1
    ) == [302.0, 0.0, -200.0, -180.0, 0.0, 0.0, 1.0]
    with pytest.raises(GrootGuardTrip, match="attachment.*lost"):
        queue.confirm_previous_endpoint(_observation(1.2, settling), settling, 1.2)

    assert perception.calls == 4
    assert executor.calls == 1


def test_moved_or_tipped_base_blocks_before_another_model_row() -> None:
    from assembly.cell_runtime import SATURDAY_GROOT_WATCHDOG_LIMITS

    clock = _Clock()
    executor = _Executor([_result(301.0)], clock)
    moved = _scene(base_pose=BASE_POSE, base_tilt_deg=4.0)
    perception = _Perception([_scene(), _scene(), moved])
    queue = _queue(
        executor,
        perception,
        clock,
        replace(SATURDAY_GROOT_WATCHDOG_LIMITS, scene_interval_rows=1),
    )
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)
    _start(queue, current)
    action = queue.emit(
        observation=_observation(1.0, current),
        current=current,
        observed_at_s=1.0,
    )
    reached = Waypoint(*action)

    with pytest.raises(
        GrootGuardTrip,
        match="active product base tilt exceeds the watchdog limit",
    ):
        queue.confirm_previous_endpoint(
            _observation(1.1, reached),
            reached,
            1.1,
        )

    assert executor.calls == 1
    assert queue.active is False


def test_watchdog_tilt_limit_cannot_exceed_physical_evidence_domain() -> None:
    with pytest.raises(ValueError, match="at most 180"):
        replace(WATCHDOG, base_tilt_limit_deg=180.1)


@pytest.mark.parametrize("field", ("pick_s", "insert_s", "scene_interval_s"))
def test_split_watchdog_durations_must_be_positive_and_finite(field: str) -> None:
    with pytest.raises(ValueError, match="split watchdog durations"):
        replace(WATCHDOG, **{field: 0.0})


@pytest.mark.parametrize(
    ("scene", "observed_at_s", "reason"),
    (
        (_scene(complete=False), 1.0, "complete"),
        (_scene(observed_at_s=1.0), 2.0, "current observation"),
        (_scene(moving_instance_id="wrong-part"), 1.0, "wrong physical instances"),
        (_scene(base_instance_id="wrong-base"), 1.0, "wrong physical instances"),
        (_scene(intrusion_detected=True), 1.0, "intrusion"),
        (_scene(moving_attached=True), 1.0, "open gripper"),
    ),
)
def test_invalid_scene_evidence_blocks_before_model_call(
    scene: CollisionScene,
    observed_at_s: float,
    reason: str,
) -> None:
    clock = _Clock()
    executor = _Executor([_result(301.0)], clock)
    queue = _queue(executor, _ExactPerception([scene]), clock)
    current = Waypoint(300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)

    with pytest.raises(GrootGuardTrip, match=reason):
        _start(queue, current, t=observed_at_s)

    assert executor.calls == 0
    assert queue.active is False


def _result(x: float, *, grip: float = 0.0) -> GrootExecutionResult:
    action = [x, 0.0, -200.0, -180.0, 0.0, 0.0, grip]
    phase = MotionPhase.GRIP if grip == 1.0 else MotionPhase.APPROACH
    return GrootExecutionResult(action, phase, False, False, ())


def _emit_and_confirm(
    queue: GrootMotionQueue,
    current: Waypoint,
    observed_at_s: float,
) -> Waypoint:
    action = queue.emit(
        observation=_observation(observed_at_s, current),
        current=current,
        observed_at_s=observed_at_s,
    )
    reached = Waypoint(*action)
    observation = _observation(observed_at_s + 0.1, reached)
    queue.confirm_previous_endpoint(observation, reached, observed_at_s + 0.1)
    return reached
