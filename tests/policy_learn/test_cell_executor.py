"""A learned CELL row remains subordinate to T09 identity, release, and scene gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from assembly.catalog import TRUSTED_GRASP_SOURCE
from assembly.controller_ports import CollisionScene, ExecutionTolerances
from assembly.motion import MotionPhase
from assembly.motion_intent import ContactEnvelope, MotionIntent, SemanticSkill
from assembly.safety import SafetySupervisor
from policy_learn.deploy.cell_executor import CellExecutionError, T09GrootExecutor
from policy_learn.deploy.groot_policy import GrootPolicy
from policy_learn.deploy.guards import GuardLimits, GuardResult


def _intent(**changes) -> MotionIntent:
    values = {
        "skill": SemanticSkill.PICK_INSERT_HOLD,
        "instance_id": "part-1",
        "object_id": "M1_02",
        "base_instance_id": "base-1",
        "base_object_id": "F01",
        "grip_index": 0,
        "case": "nominal",
        "object_pose": (300.0, -50.0, 30.0),
        "object_origin_z_mm": -281.4,
        "wrist_roll_deg": 120.0,
        "goal_pose": (400.0, 20.0, 0.0),
        "grasp_tcp": (299.1, -48.441154, -270.0, -180.0, 0.0, 120.0),
        "release_tcp": None,
        "endpoint_tcp": (402.0, 20.0, -300.0, -180.0, 0.0, 0.0),
        "final_gripper": 1.0,
        "pose_source": "T30 FoundationPose + T29 grip 0",
        "grasp_source_sha256": TRUSTED_GRASP_SOURCE.sha256,
        "contact": ContactEnvelope(
            400.0,
            20.0,
            -300.0,
            18.0,
            "M1_02 on F01",
            "reviewed measured contact envelope",
        ),
        "release_authorized": False,
    }
    values.update(changes)
    return MotionIntent(**values)


def _observation(*, grip: float = 0.0, tcp=None) -> dict:
    return {
        "t": 7.0,
        "joints": [0.0] * 6 + [grip],
        "tcp_pose": tcp or [300.0, 0.0, -200.0, -180.0, 0.0, 0.0],
        "images": {},
        "depth": {},
        "trial": {"instruction": "assemble products 1, 2, and 3"},
    }


def _scene(**changes) -> CollisionScene:
    values = {
        "observed_at_s": 7.0,
        "complete": True,
        "base_instance_id": "base-1",
        "base_pose": (400.0, 20.0, -330.0, 0.0),
        "base_tilt_deg": 0.0,
        "moving_instance_id": "part-1",
        "moving_attached": False,
        "intrusion_detected": False,
        "fixtures": (),
        "parts": (),
    }
    values.update(changes)
    return CollisionScene(**values)


@dataclass
class _Policy:
    row: tuple[float, ...]
    held: bool = False

    def __post_init__(self) -> None:
        self.task = None
        self.grasp_fence = None
        self.act_calls = 0
        self.release_calls = 0
        self.closed = False
        self.last_guard_result = None

    def set_task(self, instruction, target) -> None:
        self.task = (instruction, target)

    def set_grasp_fence(self, grasp_tcp, tolerances) -> None:
        self.grasp_fence = (grasp_tcp, tolerances)

    def reset(self) -> None:
        self.task = None

    def authorize_release(self) -> None:
        self.release_calls += 1

    def close(self) -> None:
        self.closed = True

    def act(self, observation):
        del observation
        self.act_calls += 1
        rows = np.asarray([self.row], dtype=float)
        reasons = ("guard refused",) if self.held else ()
        self.last_guard_result = GuardResult(rows, not self.held, self.held, reasons)
        return rows


class _Safety:
    def __init__(self) -> None:
        self.calls = []

    def validate(self, plan, *, start, fixtures, parts, owner):
        self.calls.append((plan, start, fixtures, parts, owner))
        return tuple(plan)


class _FixedModel:
    def __init__(self, row: tuple[float, ...]) -> None:
        self._row = row

    def predict(self, observation: dict) -> np.ndarray:
        del observation
        return np.asarray([self._row], dtype=float)


def _executor(policy: _Policy, safety: _Safety | None = None) -> T09GrootExecutor:
    return T09GrootExecutor(
        policy=policy,
        safety=safety or _Safety(),
        tolerances=ExecutionTolerances(0.3, 1.0),
    )


def test_executor_closes_its_guarded_policy_transport() -> None:
    policy = _Policy((300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0))

    _executor(policy).close()

    assert policy.closed is True


def test_cell_executor_clamps_small_model_floor_error_without_holding() -> None:
    model = _FixedModel((267.2, -105.8, -329.4, -180.0, 0.0, 0.0, 0.0))
    policy = GrootPolicy(model=model, limits=GuardLimits())
    executor = T09GrootExecutor(
        policy=policy,
        safety=None,
        tolerances=ExecutionTolerances(0.3, 1.0),
        require_scene_safety=False,
    )
    intent = _intent(
        endpoint_tcp=(267.2, -105.8, -326.7, -180.0, 0.0, 0.0),
        contact=ContactEnvelope(
            267.2,
            -105.8,
            -326.7,
            18.0,
            "grasp M1_01",
            "checkpoint-3000 eval 2026-10-31",
        ),
    )
    observation = _observation(tcp=[267.2, -105.8, -323.1, -180.0, 0.0, 0.0])

    result = executor.act(intent, observation=observation, scene=None)

    assert result.blocked is False
    assert result.action[2] == pytest.approx(-325.1)
    assert any("raised to the contact floor" in reason and "M1_01" in reason for reason in result.reasons)
    assert any("slow" in reason and "clamped" in reason for reason in result.reasons)


def test_learned_row_uses_raw_object_instruction_then_live_scene_safety() -> None:
    policy = _Policy((301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0))
    safety = _Safety()
    result = _executor(policy, safety).act(
        _intent(),
        observation=_observation(),
        scene=_scene(),
    )

    assert "M1_02 at (300, -50, 30)" in policy.task[0]
    assert "302" not in policy.task[0]
    assert policy.grasp_fence[0] == _intent().grasp_tcp
    assert result.action == [301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0]
    assert result.blocked is False
    assert len(safety.calls) == 1
    assert safety.calls[0][-1].instance_id == "part-1"


def test_guard_hold_never_reaches_the_secondary_motion_gate() -> None:
    policy = _Policy((300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0), held=True)
    safety = _Safety()
    result = _executor(policy, safety).act(
        _intent(),
        observation=_observation(),
        scene=_scene(),
    )

    assert result.blocked is True
    assert result.reasons == ("guard refused",)
    assert safety.calls == []


def test_explicit_saturday_mode_keeps_groot_guard_without_claiming_scene_safety() -> None:
    policy = _Policy((301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0))
    executor = T09GrootExecutor(
        policy=policy,
        safety=None,
        tolerances=ExecutionTolerances(0.3, 1.0),
        require_scene_safety=False,
    )

    result = executor.act(_intent(), observation=_observation(), scene=None)

    assert result.action == [301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0]
    assert result.blocked is False


def test_scene_safety_remains_fail_closed_by_default() -> None:
    with pytest.raises(ValueError, match="SafetySupervisor"):
        T09GrootExecutor(
            policy=_Policy((301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)),
            safety=None,
            tolerances=ExecutionTolerances(0.3, 1.0),
        )


def test_scene_safety_can_only_bind_once_before_executor_activation() -> None:
    executor = T09GrootExecutor(
        policy=_Policy((301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)),
        safety=None,
        tolerances=ExecutionTolerances(0.3, 1.0),
        require_scene_safety=False,
    )
    safety = object.__new__(SafetySupervisor)

    executor.bind_scene_safety(safety)

    assert executor._safety is safety
    assert executor._require_scene_safety is True
    with pytest.raises(ValueError, match="already configured"):
        executor.bind_scene_safety(object.__new__(SafetySupervisor))


def test_scene_safety_cannot_bind_after_executor_reset() -> None:
    executor = T09GrootExecutor(
        policy=_Policy((301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0)),
        safety=None,
        tolerances=ExecutionTolerances(0.3, 1.0),
        require_scene_safety=False,
    )
    executor.reset()

    with pytest.raises(ValueError, match="after executor activation"):
        executor.bind_scene_safety(object.__new__(SafetySupervisor))


@pytest.mark.parametrize("skill", [SemanticSkill.INSPECT_VIEW, SemanticSkill.PLACE_FREE])
def test_untrained_semantic_skills_are_rejected_before_policy_inference(skill: SemanticSkill) -> None:
    policy = _Policy((301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0))
    executor = T09GrootExecutor(
        policy=policy,
        safety=None,
        tolerances=ExecutionTolerances(0.3, 1.0),
        require_scene_safety=False,
    )

    changes = {"skill": skill}
    if skill is SemanticSkill.PLACE_FREE:
        changes.update(
            final_gripper=0.0,
            release_tcp=(300.0, 0.0, -200.0, -180.0, 0.0, 0.0),
            release_authorized=True,
            contact=None,
        )

    with pytest.raises(CellExecutionError, match="absent from the T18 dataset"):
        executor.act(_intent(**changes), observation=_observation(), scene=None)

    assert policy.act_calls == 0
    assert policy.task is None


def test_release_authority_is_forwarded_only_for_verified_release_intent() -> None:
    policy = _Policy((300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0))
    release = _intent(
        skill=SemanticSkill.RELEASE_RETREAT,
        final_gripper=0.0,
        release_authorized=True,
        release_tcp=(300.0, 0.0, -200.0, -180.0, 0.0, 0.0),
        endpoint_tcp=(300.0, 0.0, -200.0, -180.0, 0.0, 0.0),
    )
    result = _executor(policy).act(
        release,
        observation=_observation(grip=1.0),
        scene=_scene(moving_attached=True),
    )

    assert policy.release_calls == 1
    assert result.reached_endpoint is False
    assert result.phase is MotionPhase.RELEASE

    confirmed = _executor(policy).act(
        release,
        observation=_observation(grip=0.0),
        scene=_scene(moving_attached=False),
    )
    assert confirmed.reached_endpoint is True
    assert policy.act_calls == 1


@pytest.mark.parametrize(
    "scene",
    [
        _scene(observed_at_s=6.0),
        _scene(complete=False),
        _scene(moving_instance_id="other-part"),
        _scene(base_instance_id="other-base"),
        _scene(intrusion_detected=True),
    ],
)
def test_stale_partial_or_wrong_scene_blocks_before_model(scene: CollisionScene) -> None:
    policy = _Policy((301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0))
    with pytest.raises(CellExecutionError):
        _executor(policy).act(
            _intent(),
            observation=_observation(),
            scene=scene,
        )


def test_close_is_allowed_only_while_stationary_at_t29_grasp_tcp() -> None:
    policy = _Policy((300.0, 0.0, -200.0, -180.0, 0.0, 0.0, 1.0))
    with pytest.raises(CellExecutionError, match="T29 grasp TCP"):
        _executor(policy).act(_intent(), observation=_observation(), scene=_scene())


def test_moving_close_at_t29_grasp_is_split_into_move_then_stationary_close() -> None:
    grasp = (299.1, -48.441154, -270.0, -180.0, 0.0, 120.0)
    current = [grasp[0], grasp[1], grasp[2] + 2.7, grasp[3], grasp[4], grasp[5]]
    policy = _Policy((*grasp, 1.0))

    result = _executor(policy).act(
        _intent(grasp_tcp=grasp),
        observation=_observation(tcp=current),
        scene=_scene(),
    )

    assert result.action == [*grasp, 0.0]
    assert result.blocked is False
    assert result.reached_endpoint is False


def test_release_row_cannot_open_while_retreating() -> None:
    policy = _Policy((301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 0.0))
    release = _intent(
        skill=SemanticSkill.RELEASE_RETREAT,
        final_gripper=0.0,
        release_authorized=True,
        release_tcp=(300.0, 0.0, -200.0, -180.0, 0.0, 0.0),
    )
    with pytest.raises(CellExecutionError, match="stationary"):
        _executor(policy).act(
            release,
            observation=_observation(grip=1.0),
            scene=_scene(moving_attached=True),
        )


def test_release_row_cannot_retreat_while_still_closed() -> None:
    policy = _Policy((301.0, 0.0, -200.0, -180.0, 0.0, 0.0, 1.0))
    release = _intent(
        skill=SemanticSkill.RELEASE_RETREAT,
        final_gripper=0.0,
        release_authorized=True,
        release_tcp=(300.0, 0.0, -200.0, -180.0, 0.0, 0.0),
    )
    with pytest.raises(CellExecutionError, match="first row"):
        _executor(policy).act(
            release,
            observation=_observation(grip=1.0),
            scene=_scene(moving_attached=True),
        )
