"""Regression tests for fail-closed joint and TCP action guards."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from robot import ActionBoundsError, ActionShapeError, ActionValueError, guard_joint_action, guard_tcp_action

ROOT = Path(__file__).resolve().parents[1]
JOINT_LIMITS = ((-170, 170), (-110, 120), (-69, 205), (-190, 190), (-120, 120), (-360, 360))
SAFE_JOINT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
SAFE_TCP = [400.0, 0.0, 0.0, -180.0, 0.0, 0.0, 0.5]


def _official_joint_limits() -> tuple[tuple[int, int], ...]:
    tree = ast.parse((ROOT / "robot" / "cell" / "mock_cell.py").read_text())
    assignment = next(
        node for node in tree.body if isinstance(node, ast.Assign) and any(ast.unparse(t) == "LIMITS" for t in node.targets)
    )
    return tuple(tuple(pair) for pair in ast.literal_eval(assignment.value))


def test_joint_limits_match_the_official_mock_cell_contract() -> None:
    assert JOINT_LIMITS == _official_joint_limits()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("guard,row", [(guard_joint_action, SAFE_JOINT), (guard_tcp_action, SAFE_TCP)])
def test_guards_accept_numpy_chunks_without_mutating_shape(guard, row: list[float], dtype) -> None:
    action = np.asarray([row, row], dtype=dtype)

    guarded = guard(action, dt=0.1)

    assert guarded == [row, row]
    assert action.shape == (2, 7)


@pytest.mark.parametrize(
    ("action", "error"),
    [
        ([], ActionShapeError),
        ([[0.0] * 6], ActionShapeError),
        ([SAFE_JOINT] * 101, ActionShapeError),
        ([math.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ActionValueError),
    ],
)
def test_joint_guard_rejects_bad_shapes_and_nonfinite_values(action, error) -> None:
    with pytest.raises(error):
        guard_joint_action(action, dt=0.1)


@pytest.mark.parametrize("index,limits", list(enumerate(JOINT_LIMITS)))
def test_joint_guard_accepts_official_inclusive_limits(index: int, limits: tuple[int, int]) -> None:
    for value in limits:
        row = SAFE_JOINT.copy()
        row[index] = value
        assert guard_joint_action(row, dt=0.1)[index] == value


@pytest.mark.parametrize("index,limits", list(enumerate(JOINT_LIMITS)))
def test_joint_guard_rejects_epsilon_outside_official_limits(index: int, limits: tuple[int, int]) -> None:
    field = f"J{index + 1}"
    for value in (math.nextafter(limits[0], -math.inf), math.nextafter(limits[1], math.inf)):
        row = SAFE_JOINT.copy()
        row[index] = value
        with pytest.raises(ActionBoundsError) as raised:
            guard_joint_action(row, dt=0.1)
        assert raised.value.field == field


@pytest.mark.parametrize("value", [math.nextafter(0.0, -math.inf), math.nextafter(1.0, math.inf)])
def test_joint_guard_rejects_gripper_outside_contract(value: float) -> None:
    row = SAFE_JOINT.copy()
    row[6] = value
    with pytest.raises(ActionBoundsError) as raised:
        guard_joint_action(row, dt=0.1)
    assert raised.value.field == "gripper"
