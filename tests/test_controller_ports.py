"""Boundary tests for official-client observation and pose containers."""

import pytest

from assembly.controller_ports import (
    ControllerPortError,
    PerceptionTolerances,
    current_waypoint,
    pose_agrees,
)


class ArrayLike:
    """Minimal NumPy-style container: indexable, but not a Sequence."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> float:
        return self._values[index]


class ZeroDimArray:
    """NumPy-style scalar array whose shape has no sequence length."""

    def __len__(self) -> int:
        raise TypeError("len() of unsized object")

    def __getitem__(self, index: int) -> float:
        raise IndexError(index)


def test_current_waypoint_accepts_official_client_array_shape() -> None:
    observation = {
        "tcp_pose": ArrayLike([200.0, 0.0, -250.0, -180.0, 0.0, 0.0]),
        "joints": ArrayLike([0.0] * 7),
    }

    waypoint = current_waypoint(observation)

    assert waypoint.xyz == (200.0, 0.0, -250.0)
    assert waypoint.gripper == 0.0


def test_pose_agreement_uses_measured_distance_and_wrapped_yaw() -> None:
    actual = {"x": 100.2, "y": 50.0, "z": -300.0, "yaw": -179.9}
    expected = (100.0, 50.0, -300.0, 180.0)

    assert pose_agrees(actual, expected, PerceptionTolerances(0.25, 0.2))
    assert not pose_agrees(actual, expected, PerceptionTolerances(0.1, 0.2))


def test_current_waypoint_normalizes_unsized_array_to_port_error() -> None:
    with pytest.raises(ControllerPortError, match="six absolute UF0 values"):
        current_waypoint({"tcp_pose": ZeroDimArray(), "joints": [0.0] * 7})


@pytest.mark.parametrize("value", [[0.0] * 5, [0.0] * 7, "not-a-pose", {"x": 0.0}])
def test_current_waypoint_rejects_malformed_tcp_shapes(value: object) -> None:
    with pytest.raises(ControllerPortError):
        current_waypoint({"tcp_pose": value, "joints": [0.0] * 7})
