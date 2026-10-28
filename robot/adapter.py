"""Passive callback adapters shared by simulation, mock, and cell runtimes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Literal, Protocol, TypeAlias

ActionMode: TypeAlias = Literal["joints", "tcp"]
Action: TypeAlias = Sequence[float] | Sequence[Sequence[float]]
Observation: TypeAlias = Mapping[str, object]

OFFICIAL_OBSERVATION_FIELDS = frozenset(
    {
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
)
TCP_FIELDS = ("x", "y", "z", "w", "p", "r", "gripper")
TCP_LIMITS = (
    (150.0, 650.0, "mm in UF0"),
    (-450.0, 450.0, "mm in UF0"),
    (-334.0, 300.0, "mm in UF0"),
    (-360.0, 360.0, "degrees"),
    (-360.0, 360.0, "degrees"),
    (-360.0, 360.0, "degrees"),
    (0.0, 1.0, "state"),
)
DT_LIMITS = (0.1, 1.0)
MAX_ACTION_CHUNK = 100


class ActionValidationError(ValueError):
    """Base error carrying the rejected field, row, and value."""

    def __init__(self, message: str, *, field: str, row_index: int | None, value: object) -> None:
        super().__init__(message)
        self.field = field
        self.row_index = row_index
        self.value = value


class ActionShapeError(ActionValidationError):
    """The action is neither one seven-value row nor a valid chunk."""


class ActionValueError(ActionValidationError):
    """An action field is not a finite numeric value."""


class ActionBoundsError(ActionValidationError):
    """A finite action field is outside its inclusive safety bound."""


def _shape_error(value: object, *, row_index: int | None = None) -> ActionShapeError:
    row_context = "" if row_index is None else f"; row {row_index} is malformed"
    return ActionShapeError(
        f"tcp action must be one row of 7 values or a chunk of 1..100 rows{row_context}",
        field="shape",
        row_index=row_index,
        value=value,
    )


def _as_items(value: object) -> list | None:
    if isinstance(value, (str, bytes, bytearray, set, frozenset, Mapping)):
        return None
    try:
        return list(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def _tcp_rows(action: object) -> tuple[list[list], bool]:
    outer = _as_items(action)
    if not outer:
        raise _shape_error(action)
    possible_rows = [_as_items(item) for item in outer]
    is_chunk = any(row is not None for row in possible_rows)
    if not is_chunk:
        if len(outer) != len(TCP_FIELDS):
            raise _shape_error(action)
        return [outer], False
    if len(outer) > MAX_ACTION_CHUNK:
        raise _shape_error(action)
    for row_index, row in enumerate(possible_rows):
        if row is None or len(row) != len(TCP_FIELDS):
            raise _shape_error(outer[row_index], row_index=row_index)
    return [row for row in possible_rows if row is not None], True


def _finite_numeric(value: object, *, field: str, row_index: int | None) -> float:
    if isinstance(value, bool) or not isinstance(value, Number):
        numeric = None
    else:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            numeric = None
    if numeric is None or not math.isfinite(numeric):
        location = "tcp action" if row_index is None else f"tcp action row {row_index}"
        raise ActionValueError(
            f"{location} field '{field}' must be a finite numeric value, got {value!r}",
            field=field,
            row_index=row_index,
            value=value,
        )
    return numeric


def _check_bounds(value: float, *, field: str, row_index: int | None, limits: tuple[float, float], unit: str) -> None:
    lower, upper = limits
    if lower <= value <= upper:
        return
    location = "tcp action" if row_index is None else f"tcp action row {row_index}"
    frame = " UF0" if field in {"x", "y", "z"} else ""
    raise ActionBoundsError(
        f"{location} field '{field}'={value!r} outside inclusive{frame} bounds [{lower}, {upper}] {unit}",
        field=field,
        row_index=row_index,
        value=value,
    )


def guard_tcp_action(action: object, *, dt: object) -> list[float] | list[list[float]]:
    """Validate an absolute UTOOL1 target or chunk and return a shape-equivalent copy."""
    safe_dt = _finite_numeric(dt, field="dt", row_index=None)
    _check_bounds(safe_dt, field="dt", row_index=None, limits=DT_LIMITS, unit="seconds")
    rows, is_chunk = _tcp_rows(action)
    safe_rows: list[list[float]] = []
    for row_index, row in enumerate(rows):
        safe_row: list[float] = []
        for field, value, (lower, upper, unit) in zip(TCP_FIELDS, row, TCP_LIMITS, strict=True):
            numeric = _finite_numeric(value, field=field, row_index=row_index)
            _check_bounds(numeric, field=field, row_index=row_index, limits=(lower, upper), unit=unit)
            safe_row.append(numeric)
        safe_rows.append(safe_row)
    return safe_rows if is_chunk else safe_rows[0]


class Controller(Protocol):
    """Callback contract implemented by controllers and recipe state machines."""

    def reset(self) -> None: ...

    def act(self, observation: Observation) -> Action: ...


class RobotAdapter(Protocol):
    """Runtime-neutral callback interface consumed by policy runners."""

    def reset(self) -> None: ...

    def act(self, observation: Observation) -> Action: ...


class _ControllerAdapter:
    """Delegate callbacks without owning or duplicating runtime transport."""

    def __init__(self, controller: Controller, *, action_mode: ActionMode, dt: float = 0.1) -> None:
        if action_mode not in ("joints", "tcp"):
            raise ValueError(f"unsupported action mode: {action_mode!r}")
        self._controller = controller
        self.action_mode = action_mode
        self.dt = dt

    def reset(self) -> None:
        self._controller.reset()

    def act(self, observation: Observation) -> Action:
        action = self._controller.act(self._controller_observation(observation))
        if self.action_mode == "tcp":
            return guard_tcp_action(action, dt=self.dt)
        return action

    def _controller_observation(self, observation: Observation) -> Observation:
        return observation


class SimAdapter(_ControllerAdapter):
    """Project simulator observations onto fields available from the cell."""

    def _controller_observation(self, observation: Observation) -> Observation:
        return {key: value for key, value in observation.items() if key in OFFICIAL_OBSERVATION_FIELDS}


class MockAdapter(_ControllerAdapter):
    """Delegate observations produced by the official local mock."""


class CellAdapter(_ControllerAdapter):
    """Delegate decoded cell observations without creating a CellClient."""


class JointPositionHoldController:
    """Safely hold the joint position reported in the current observation."""

    def reset(self) -> None:
        """The stateless hold controller has no rollout state to clear."""

    def act(self, observation: Observation) -> Action:
        joints = _as_items(observation["joints"])
        if joints is None or len(joints) != len(TCP_FIELDS):
            raise ValueError("observation field 'joints' must contain J1..J6 and gripper")
        return joints
