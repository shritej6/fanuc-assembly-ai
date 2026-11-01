"""Positive software proof for guarded to-GR00T execution.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import numpy as np

from assembly.final_verification import ControllerOwnedFinalVerifier, FinalViewIdentityEvidence
from assembly.final_views import FinalViewPose, FinalViewRegistry
from assembly.motion_intent import MotionIntent, SemanticSkill
from assembly.state_machine import PackMLState
from policy_learn.deploy.cell_executor import T09GrootExecutor
from policy_learn.deploy.cell_groot_policy import GrootPolicy
from robot.cell.t09_mock_runtime import MockPerception, build_groot_mock_controller


def _reached(actual: np.ndarray, expected: tuple[float, ...]) -> bool:
    return bool(np.max(np.abs(actual - np.asarray(expected, dtype=float))) <= 1e-9)


def _increment(current: np.ndarray, target: tuple[float, ...]) -> np.ndarray:
    desired = np.asarray(target, dtype=float)
    row = current.copy()
    delta_xyz = desired[:3] - current[:3]
    distance = float(np.linalg.norm(delta_xyz))
    if distance:
        row[:3] += delta_xyz * min(1.0, 1.0 / distance)
    angle_delta = (desired[3:6] - current[3:6] + 180.0) % 360.0 - 180.0
    row[3:6] += np.clip(angle_delta, -1.0, 1.0)
    return row


class _IntentAwareSafeModel:
    """Fake learned model: proposes one small row; it does not own safety."""

    def __init__(self) -> None:
        self.intent: MotionIntent | None = None
        self.skills: list[SemanticSkill] = []

    def predict(self, observation: dict) -> np.ndarray:
        assert self.intent is not None
        current = np.asarray(observation["tcp_pose"], dtype=float)
        gripper = float(observation["joints"][6])
        target, next_gripper = self._next_target(current, gripper)
        row = _increment(current, target)
        row = np.concatenate((row, [next_gripper]))
        return row[None]

    def _next_target(self, current: np.ndarray, gripper: float) -> tuple[tuple[float, ...], float]:
        assert self.intent is not None
        intent = self.intent
        if intent.skill is SemanticSkill.PICK_INSERT_HOLD:
            if gripper == 0.0 and not _reached(current, intent.grasp_tcp):
                return intent.grasp_tcp, 0.0
            if gripper == 0.0:
                return tuple(current), 1.0
            return intent.endpoint_tcp, 1.0
        if intent.skill is SemanticSkill.RELEASE_RETREAT:
            assert intent.release_tcp is not None
            if gripper == 1.0 and not _reached(current, intent.release_tcp):
                return intent.release_tcp, 1.0
            if gripper == 1.0:
                return tuple(current), 0.0
            return intent.endpoint_tcp, 0.0
        return intent.endpoint_tcp, gripper


class _RecordingExecutor(T09GrootExecutor):
    def __init__(self, model: _IntentAwareSafeModel, policy: GrootPolicy) -> None:
        super().__init__(policy=policy, safety=None, tolerances=_zero_tolerances(), require_scene_safety=False)
        self._model = model
        self.results = []

    def act(self, intent, *, observation, scene):
        self._model.intent = intent
        self._model.skills.append(intent.skill)
        result = super().act(intent, observation=observation, scene=scene)
        self.results.append(result)
        return result


def _zero_tolerances():
    from assembly.controller_ports import ExecutionTolerances

    return ExecutionTolerances(0.0, 0.0)


def _observation() -> dict:
    return {
        "t": 1.0,
        "tcp_pose": [400.0, 0.0, -200.0, -180.0, 0.0, 0.0],
        "joints": [0.0] * 7,
        "images": {},
        "trial": {"instruction": "assemble product 1"},
    }


def _apply(action: list[float], observation: dict) -> None:
    observation["tcp_pose"] = action[:6]
    observation["joints"][6] = action[6]
    observation["t"] += 0.2


class _FinalIdentityBackend:
    def __init__(self) -> None:
        self.calls = []

    def capture(self, observation, view, base_instance_id, installed_instance_ids):
        del base_instance_id
        self.calls.append((view.view_id, tuple(observation["tcp_pose"]), float(observation["t"])))
        return FinalViewIdentityEvidence(
            float(observation["t"]),
            True,
            view.view_id,
            installed_instance_ids,
            view.expected_instance_roles,
            True,
        )


def _controller_owned_final_verifier():
    roles = ("M1_01", "M1_02", "M1_03", "M1_04")
    source = "mock proof of controller-owned guarded final views"
    views = tuple(
        FinalViewPose(
            1,
            "F01",
            f"mock-B2-view-{index}",
            (420.0, y_mm, -168.0, -180.0, 0.0, 90.0),
            -180.0,
            roles,
            source,
            (420.0, 80.0),
        )
        for index, y_mm in enumerate((70.0, 90.0), start=1)
    )
    registry = FinalViewRegistry(views)
    backend = _FinalIdentityBackend()
    verifier = ControllerOwnedFinalVerifier(
        registry,
        backend,
        selection_tolerance_mm=0.0,
    )
    return verifier, registry, backend


def test_valid_guarded_fake_groot_completes_one_product_without_untrained_motion() -> None:
    model = _IntentAwareSafeModel()
    policy = GrootPolicy(model=model)
    executor = _RecordingExecutor(model, policy)
    controller = build_groot_mock_controller(executor)
    observation = _observation()

    for step in range(20_000):
        action = controller.act(observation)
        _apply(action, observation)
        if step == 0:
            perception = controller._perception
            assert isinstance(perception, MockPerception)
            perception.fallen_once.clear()
            perception.failed_insertions.clear()
        if controller.completed_products:
            break

    skills = tuple(sorted({skill.value for skill in model.skills}))
    print(
        f"CELL_SUCCESS completed={controller.completed_products} "
        f"guarded_rows={len(executor.results)} steps={step + 1} skills={skills}"
    )
    assert controller.completed_products == (1,)
    assert SemanticSkill.PICK_INSERT_HOLD in model.skills
    assert SemanticSkill.RELEASE_RETREAT in model.skills
    assert SemanticSkill.INSPECT_VIEW not in model.skills
    assert SemanticSkill.PLACE_FREE not in model.skills
    assert executor.results
    assert all(not result.blocked for result in executor.results)
    clamp_reasons = tuple(
        reason
        for result in executor.results
        for reason in result.reasons
    )
    assert all(
        "raised to the contact floor" in reason or "step clamped from" in reason
        for reason in clamp_reasons
    )
    assert policy.last_guard_result is not None
    assert all(entry["held"] is False for entry in policy.guard_log)


def test_controller_owned_final_views_complete_after_two_guarded_camera_poses() -> None:
    model = _IntentAwareSafeModel()
    policy = GrootPolicy(model=model)
    executor = _RecordingExecutor(model, policy)
    verifier, registry, backend = _controller_owned_final_verifier()
    controller = build_groot_mock_controller(
        executor,
        final_verifier=verifier,
        final_view_registry=registry,
    )
    observation = _observation()

    for _step in range(20_000):
        action = controller.act(observation)
        _apply(action, observation)
        if _step == 0:
            controller._perception.fallen_once.clear()
            controller._perception.failed_insertions.clear()
        if controller.completed_products:
            break

    inspection_events = tuple(event for event in controller.events if event.kind == "inspection_motion")
    assert controller.completed_products == (1,)
    assert len(inspection_events) >= 2
    assert all("non_learned_inspection" in event.detail for event in inspection_events[-2:])
    assert tuple(call[0] for call in backend.calls) == ("mock-B2-view-1", "mock-B2-view-2")
    assert backend.calls[0][2] < backend.calls[1][2]
    endpoints = {view.view_id: np.asarray(view.endpoint_tcp_uf0) for view in registry.views}
    assert all(
        np.max(np.abs(np.asarray(tcp) - endpoints[view_id])) <= 0.3
        for view_id, tcp, _observed_at in backend.calls
    )


def test_final_capture_holds_if_tcp_drifts_after_the_view_settles() -> None:
    model = _IntentAwareSafeModel()
    policy = GrootPolicy(model=model)
    executor = _RecordingExecutor(model, policy)
    verifier, registry, backend = _controller_owned_final_verifier()
    controller = build_groot_mock_controller(
        executor,
        final_verifier=verifier,
        final_view_registry=registry,
    )
    observation = _observation()

    for step in range(20_000):
        action = controller.act(observation)
        _apply(action, observation)
        if step == 0:
            controller._perception.fallen_once.clear()
            controller._perception.failed_insertions.clear()
        pending = controller._pending_final_verification
        if pending is not None and pending.awaiting_capture_index == 0 and not controller._motion_queue.active:
            observation["tcp_pose"][0] += 1.0
            break
    else:
        raise AssertionError("first final-view endpoint never settled")

    controller.act(observation)

    assert backend.calls == []
    assert controller.completed_products == ()
    assert controller.state_machine.state is PackMLState.HELD
    assert "reviewed final-view endpoint" in controller.events[-1].detail
