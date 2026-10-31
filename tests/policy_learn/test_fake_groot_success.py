"""Positive software proof for guarded T09-to-GR00T execution."""

from __future__ import annotations

import numpy as np

from assembly.motion_intent import MotionIntent, SemanticSkill
from policy_learn.deploy.cell_executor import T09GrootExecutor
from policy_learn.deploy.groot_policy import GrootPolicy
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
