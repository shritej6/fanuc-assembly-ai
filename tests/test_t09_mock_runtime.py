"""Opt-in official-mock wiring for the T09 AssemblyController."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

from robot import guard_tcp_action

CELL_DIR = Path(__file__).resolve().parents[1] / "robot" / "cell"


@pytest.fixture
def policy_type(monkeypatch):
    monkeypatch.syspath_prepend(str(CELL_DIR))
    sys.modules.pop("policy", None)
    from policy import Policy

    return Policy


def _observation() -> dict[str, object]:
    return {
        "t": 0.0,
        "tcp_pose": None,
        "joints": [0.0] * 7,
        "images": {},
        "trial": {"instruction": "assemble products 1, 3, and 4"},
    }


def _rows(action) -> list[list[float]]:
    return action if isinstance(action[0], list) else [action]


def _apply(observation: dict[str, object], action) -> None:
    last = _rows(action)[-1]
    observation["tcp_pose"] = last[:6]
    observation["joints"] = [0.0] * 6 + [last[6]]
    observation["t"] = float(observation["t"]) + 1.0


def _drive(policy, *, limit: int = 200) -> list[list[float]]:
    observation = _observation()
    bootstrap = policy.act(observation)
    actions = _rows(bootstrap)
    _apply(observation, bootstrap)
    for _ in range(limit):
        if policy.controller.finished:
            return actions
        action = policy.act(observation)
        actions.extend(_rows(action))
        _apply(observation, action)
    raise AssertionError(f"mock controller did not finish within {limit} callbacks")


def test_t09_mock_checkpoint_is_explicit_and_forbidden_on_cell(policy_type) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        policy_type("t09-mock", runtime="cell")

    policy = policy_type("t09-mock", runtime="mock")

    assert policy.mock_only is True
    assert policy.ACTION == "tcp"
    assert policy_type.ACTION == "joints"


def test_t09_mock_bootstraps_executes_faults_and_archives(policy_type) -> None:
    policy = policy_type("t09-mock", runtime="mock")

    actions = _drive(policy)

    assert all(guard_tcp_action(action, dt=policy.DT) == action for action in actions)
    assert policy.controller.completed_products == (1, 3, 4)
    assert policy.controller.finished is True
    assert any(event.kind == "spare" and event.object_id == "M1_02" for event in policy.controller.events)
    assert any(event.kind == "retry" and event.object_id == "M3_03" for event in policy.controller.events)

    policy.reset()

    assert policy.controller.archived_runs[-1].completed_products == (1, 3, 4)
    assert policy.controller.archived_runs[-1].finished is True
    assert policy.controller.events == ()


def test_default_policy_remains_joint_hold_in_mock_runtime(policy_type) -> None:
    joints = [-0.5, 3.0, -11.0, 0.0, -74.0, 0.0, 0.0]
    policy = policy_type(runtime="mock")

    output = policy.act({"joints": joints, "tcp_pose": None})

    assert output == joints
    assert output is not joints
    assert policy.mock_only is False
    assert policy.ACTION == "joints"


def test_cell_project_locks_catalog_schema_dependency() -> None:
    project = tomllib.loads((CELL_DIR / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]
    lock = (CELL_DIR / "uv.lock").read_text()

    assert "jsonschema>=4.18" in dependencies
    assert 'name = "jsonschema"' in lock
    assert '{ name = "jsonschema", specifier = ">=4.18" }' in lock
