"""Adapter callback behavior and robot-action safety tests."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
from copy import deepcopy
from pathlib import Path

import pytest

from robot import (
    ActionBoundsError,
    ActionShapeError,
    ActionValueError,
    CellAdapter,
    JointPositionHoldController,
    MockAdapter,
    SimAdapter,
    guard_tcp_action,
)

SAFE_TCP_ROW = [400.0, 0.0, 0.0, -180.0, 0.0, 0.0, 0.5]
SAFE_JOINT_ROW = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
CELL_DIR = Path(__file__).resolve().parents[1] / "robot" / "cell"
POLICY_DELEGATION_SCRIPT = textwrap.dedent(
    """
    import json

    from policy import Policy
    from robot import CellAdapter

    class SpyAdapter:
        def __init__(self):
            self.reset_calls = 0
            self.observations = []

        def reset(self):
            self.reset_calls += 1

        def act(self, observation):
            self.observations.append(observation)
            return [9.0] * 7

    policy = Policy("unused-checkpoint")
    assert Policy.ACTION == "joints"
    assert isinstance(policy.adapter, CellAdapter)
    spy = SpyAdapter()
    policy.adapter = spy
    observation = {"sentinel": "observation"}
    policy.reset()
    action = policy.act(observation)
    print(json.dumps({
        "action": action,
        "reset_calls": spy.reset_calls,
        "delegated_same_object": spy.observations == [observation],
    }))
    """
)
DEFAULT_POLICY_SCRIPT = textwrap.dedent(
    """
    import json

    from policy import Policy

    joints = [-0.0065, 3.2909, -11.8967, -0.0007, -74.8218, -0.0286, 0.0]
    action = Policy().act({"joints": joints, "tcp_pose": None})
    print(json.dumps({"action": action, "copied": action is not joints}))
    """
)
TCP_POLICY_GUARD_SCRIPT = textwrap.dedent(
    """
    import json

    from policy import Policy
    from robot import ActionBoundsError

    class UnsafeController:
        def reset(self):
            pass

        def act(self, observation):
            return [149.0, 0.0, 0.0, -180.0, 0.0, 0.0, 0.0]

    Policy.ACTION = "tcp"
    policy = Policy()
    policy.adapter._controller = UnsafeController()
    try:
        policy.act({"joints": [0.0] * 7, "tcp_pose": None})
    except ActionBoundsError as error:
        print(json.dumps({"field": error.field, "value": error.value}))
    else:
        raise AssertionError("unsafe TCP action passed through Policy")
    """
)
POLICY_CURRENT_DT_SCRIPT = textwrap.dedent(
    """
    import json

    from policy import Policy
    from robot import ActionBoundsError

    policy = Policy()
    Policy.DT = 1.1
    try:
        policy.act({"joints": [0.0] * 7, "tcp_pose": None})
    except ActionBoundsError as error:
        print(json.dumps({"field": error.field, "value": error.value}))
    else:
        raise AssertionError("Policy reused stale DT")
    """
)


class RecordingController:
    def __init__(self, action):
        self.action = action
        self.observations: list[dict] = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, observation: dict):
        self.observations.append(observation)
        return self.action


def official_observation() -> dict:
    return {
        "t": 1.25,
        "status": "running",
        "trial": {"trial_id": "trial-1", "instruction": "assemble"},
        "queued": 0,
        "joints": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0],
        "tcp_pose": None,
        "format": "png",
        "images": {"table": "rgb"},
        "depth": {"table": "depth"},
    }


def privileged_sim_observation() -> dict:
    observation = official_observation()
    observation["trial"] = {
        "trial_id": "trial-1",
        "index": 2,
        "layout": "L3",
        "status": "running",
        "instruction": "assemble",
        "scored": False,
        "ground_truth": {"object_poses": [[1.0, 2.0, 3.0]]},
    }
    observation["images"] = {
        "table": ["table-rgb"],
        "side": ["side-rgb"],
        "wrist": ["wrist-rgb"],
        "segmentation": [42],
    }
    observation["depth"] = {"table": [1000], "wrist": [350], "ground_truth_depth": [1]}
    return observation


@pytest.mark.parametrize("adapter_type", [SimAdapter, MockAdapter, CellAdapter])
def test_all_adapters_delegate_callbacks(adapter_type) -> None:
    observation = official_observation()
    action = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]
    controller = RecordingController(action)
    adapter = adapter_type(controller, action_mode="joints")

    adapter.reset()
    result = adapter.act(observation)

    assert controller.reset_calls == 1
    assert controller.observations == [observation]
    assert result == action


def test_sim_adapter_filters_privileged_ground_truth() -> None:
    observation = official_observation()
    observation |= {
        "ground_truth": {"object_poses": [[0.0, 0.0, 0.0]]},
        "segmentation_ids": [42],
        "sim_private_state": {"perfect_grasp": True},
    }
    original = deepcopy(observation)
    controller = RecordingController(observation["joints"])
    adapter = SimAdapter(controller, action_mode="joints")

    result = adapter.act(observation)

    assert set(controller.observations[0]) == {
        "t",
        "status",
        "trial",
        "queued",
        "joints",
        "tcp_pose",
        "format",
        "images",
        "depth",
    }
    assert result == observation["joints"]
    assert observation == original


def test_sim_adapter_projects_nested_observation_mappings() -> None:
    observation = privileged_sim_observation()
    original = deepcopy(observation)
    controller = RecordingController(observation["joints"])

    SimAdapter(controller, action_mode="joints").act(observation)

    projected = controller.observations[0]
    assert projected["trial"] == {
        key: observation["trial"][key] for key in ("trial_id", "index", "layout", "status", "instruction", "scored")
    }
    assert projected["images"] == {key: observation["images"][key] for key in ("table", "side", "wrist")}
    assert projected["depth"] == {key: observation["depth"][key] for key in ("table", "wrist")}
    assert projected["trial"] is not observation["trial"]
    assert projected["images"] is not observation["images"]
    assert projected["depth"] is not observation["depth"]
    assert projected["images"]["table"] is observation["images"]["table"]
    assert projected["depth"]["wrist"] is observation["depth"]["wrist"]
    assert observation == original


def test_joint_hold_controller_uses_observed_position_when_tcp_is_unknown() -> None:
    observation = official_observation()
    controller = JointPositionHoldController()

    controller.reset()
    result = controller.act(observation)

    assert result == observation["joints"]
    assert result is not observation["joints"]
    assert observation["tcp_pose"] is None


def test_joint_hold_accepts_decoded_array_like_joint_vector() -> None:
    class ArrayLike:
        def __iter__(self):
            return iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0])

    result = JointPositionHoldController().act({"joints": ArrayLike(), "tcp_pose": None})

    assert result == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]


def tcp_row(index: int, value: object) -> list:
    row = SAFE_TCP_ROW.copy()
    row[index] = value
    return row


@pytest.mark.parametrize(
    ("index", "value"),
    [
        (0, 150.0),
        (0, 650.0),
        (1, -450.0),
        (1, 450.0),
        (2, -334.0),
        (2, 300.0),
    ],
    ids=["x-min", "x-max", "y-min", "y-max", "z-min", "z-max"],
)
def test_tcp_workspace_faces_are_inclusive(index: int, value: float) -> None:
    row = tcp_row(index, value)

    assert guard_tcp_action(row, dt=0.1) == row


@pytest.mark.parametrize(
    ("index", "field", "value"),
    [
        (0, "x", math.nextafter(150.0, -math.inf)),
        (0, "x", math.nextafter(650.0, math.inf)),
        (1, "y", math.nextafter(-450.0, -math.inf)),
        (1, "y", math.nextafter(450.0, math.inf)),
        (2, "z", math.nextafter(-334.0, -math.inf)),
        (2, "z", math.nextafter(300.0, math.inf)),
    ],
    ids=["x-below", "x-above", "y-below", "y-above", "z-below", "z-above"],
)
def test_tcp_workspace_epsilon_outside_is_rejected(index: int, field: str, value: float) -> None:
    with pytest.raises(ActionBoundsError, match=rf"row 0.*'{field}'.*UF0") as raised:
        guard_tcp_action(tcp_row(index, value), dt=0.1)

    assert raised.value.field == field
    assert raised.value.row_index == 0
    assert raised.value.value == value


@pytest.mark.parametrize(
    ("action", "expected_row_index"),
    [
        ([], None),
        (SAFE_TCP_ROW[:6], None),
        ([*SAFE_TCP_ROW, 1.0], None),
        ([SAFE_TCP_ROW[:6]], 0),
        ([SAFE_TCP_ROW.copy() for _ in range(101)], None),
        ([SAFE_TCP_ROW.copy(), 1.0], 1),
        (set(range(7)), None),
    ],
    ids=["empty", "short-row", "long-row", "short-chunk-row", "101-rows", "mixed-chunk", "unordered-row"],
)
def test_malformed_tcp_shapes_are_rejected(action, expected_row_index: int | None) -> None:
    with pytest.raises(ActionShapeError, match="one row of 7 values or a chunk of 1..100 rows") as raised:
        guard_tcp_action(action, dt=0.1)

    assert raised.value.field == "shape"
    assert raised.value.row_index == expected_row_index


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "400", True])
def test_tcp_values_must_be_finite_numbers(value: object) -> None:
    with pytest.raises(ActionValueError, match="row 0.*'x'.*finite numeric") as raised:
        guard_tcp_action(tcp_row(0, value), dt=0.1)

    assert raised.value.field == "x"
    assert raised.value.row_index == 0
    assert raised.value.value == value or math.isnan(raised.value.value)


@pytest.mark.parametrize(
    ("index", "field", "value"),
    [
        (3, "w", math.nextafter(-177.0, math.inf)),
        (3, "w", math.nextafter(177.0, -math.inf)),
        (4, "p", math.nextafter(-3.0, -math.inf)),
        (4, "p", math.nextafter(3.0, math.inf)),
        (5, "r", math.nextafter(-360.0, -math.inf)),
        (5, "r", math.nextafter(360.0, math.inf)),
        (6, "gripper", math.nextafter(0.0, -math.inf)),
        (6, "gripper", math.nextafter(1.0, math.inf)),
    ],
)
def test_tcp_tool_down_and_gripper_outside_bounds_are_rejected(index: int, field: str, value: float) -> None:
    with pytest.raises(ActionBoundsError, match=rf"row 0.*'{field}'") as raised:
        guard_tcp_action(tcp_row(index, value), dt=0.1)

    assert raised.value.field == field
    assert raised.value.row_index == 0


@pytest.mark.parametrize(
    ("index", "value"),
    [
        (3, -183.0),
        (3, -177.0),
        (3, 177.0),
        (3, 183.0),
        (4, -3.0),
        (4, 3.0),
        (5, -360.0),
        (5, 360.0),
        (6, 0.0),
        (6, 1.0),
    ],
)
def test_tcp_tool_down_and_gripper_bounds_are_inclusive(index: int, value: float) -> None:
    row = tcp_row(index, value)

    assert guard_tcp_action(row, dt=0.1) == row


@pytest.mark.parametrize("dt", [0.1, 1.0])
def test_tcp_dt_bounds_are_inclusive(dt: float) -> None:
    assert guard_tcp_action(SAFE_TCP_ROW, dt=dt) == SAFE_TCP_ROW


@pytest.mark.parametrize(
    "dt",
    [math.nextafter(0.1, -math.inf), math.nextafter(1.0, math.inf)],
    ids=["below-min", "above-max"],
)
def test_tcp_dt_outside_bounds_is_rejected(dt: float) -> None:
    with pytest.raises(ActionBoundsError, match="'dt'.*seconds") as raised:
        guard_tcp_action(SAFE_TCP_ROW, dt=dt)

    assert raised.value.field == "dt"
    assert raised.value.row_index is None


@pytest.mark.parametrize("dt", [math.nan, math.inf, -math.inf, "0.1", True])
def test_tcp_dt_must_be_a_finite_number(dt: object) -> None:
    with pytest.raises(ActionValueError, match="'dt'.*finite numeric") as raised:
        guard_tcp_action(SAFE_TCP_ROW, dt=dt)

    assert raised.value.field == "dt"
    assert raised.value.row_index is None


def test_tcp_guard_preserves_single_and_chunk_shapes_without_mutation() -> None:
    single = SAFE_TCP_ROW.copy()
    chunk = [SAFE_TCP_ROW.copy(), tcp_row(0, 500.0)]
    single_before, chunk_before = deepcopy(single), deepcopy(chunk)

    guarded_single = guard_tcp_action(single, dt=0.1)
    guarded_chunk = guard_tcp_action(chunk, dt=0.1)

    assert guarded_single == single_before
    assert guarded_chunk == chunk_before
    assert guarded_single is not single
    assert guarded_chunk is not chunk
    assert guarded_chunk[0] is not chunk[0]
    assert single == single_before
    assert chunk == chunk_before


def test_tcp_guard_accepts_maximum_chunk() -> None:
    chunk = [SAFE_TCP_ROW.copy() for _ in range(100)]

    assert guard_tcp_action(chunk, dt=1.0) == chunk


def test_tcp_adapter_guards_controller_output_with_configured_dt() -> None:
    controller = RecordingController(tcp_row(0, 149.0))
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)

    with pytest.raises(ActionBoundsError, match="'x'"):
        adapter.act(official_observation())


def test_joint_adapter_guards_controller_output() -> None:
    controller = RecordingController([0.0, 0.0, 0.0, 0.0, 999.0, 0.0, 7.0])
    adapter = CellAdapter(controller, action_mode="joints", dt=0.1)

    with pytest.raises(ActionBoundsError, match="'J5'"):
        adapter.act(official_observation())


def test_adapter_resolves_current_dt_on_every_call() -> None:
    current_dt = {"value": 0.1}
    adapter = CellAdapter(
        RecordingController(SAFE_JOINT_ROW),
        action_mode="joints",
        dt=lambda: current_dt["value"],
    )

    assert adapter.act(official_observation()) == SAFE_JOINT_ROW
    current_dt["value"] = 1.1
    with pytest.raises(ActionBoundsError, match="'dt'"):
        adapter.act(official_observation())


def run_policy_script(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=CELL_DIR,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_policy_imports_from_cell_directory_and_delegates_without_pythonpath() -> None:
    result = run_policy_script(POLICY_DELEGATION_SCRIPT)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "action": [9.0] * 7,
        "reset_calls": 1,
        "delegated_same_object": True,
    }


def test_default_policy_holds_current_joints_when_tcp_pose_is_none() -> None:
    result = run_policy_script(DEFAULT_POLICY_SCRIPT)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "action": [-0.0065, 3.2909, -11.8967, -0.0007, -74.8218, -0.0286, 0.0],
        "copied": True,
    }


def test_policy_rejects_out_of_box_tcp_controller_output() -> None:
    result = run_policy_script(TCP_POLICY_GUARD_SCRIPT)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"field": "x", "value": 149.0}


def test_policy_uses_current_dt_instead_of_constructor_snapshot() -> None:
    result = run_policy_script(POLICY_CURRENT_DT_SCRIPT)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"field": "dt", "value": 1.1}
