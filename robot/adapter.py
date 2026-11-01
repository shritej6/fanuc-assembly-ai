"""Passive callback adapters shared by the simulation, mock and cell runtimes.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
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
OFFICIAL_TRIAL_FIELDS = frozenset({"trial_id", "index", "layout", "status", "instruction", "scored"})
RGB_CAMERA_FIELDS = frozenset({"table", "side", "wrist"})
DEPTH_CAMERA_FIELDS = frozenset({"table", "wrist"})
NESTED_OBSERVATION_FIELDS = {
    "trial": OFFICIAL_TRIAL_FIELDS,
    "images": RGB_CAMERA_FIELDS,
    "depth": DEPTH_CAMERA_FIELDS,
}
TCP_FIELDS = ("x", "y", "z", "w", "p", "r", "gripper")
JOINT_FIELDS = ("J1", "J2", "J3", "J4", "J5", "J6", "gripper")
TCP_LIMITS = (
    (150.0, 650.0, "mm in UF0"),
    (-450.0, 450.0, "mm in UF0"),
    (-334.0, 300.0, "mm in UF0"),
    (-360.0, 360.0, "degrees"),
    (-360.0, 360.0, "degrees"),
    (-360.0, 360.0, "degrees"),
    (0.0, 1.0, "state"),
)
# Matches LIMITS in the official robot/cell/mock_cell.py.
JOINT_LIMITS = (
    (-170.0, 170.0, "degrees"),
    (-110.0, 120.0, "degrees"),
    (-69.0, 205.0, "degrees"),
    (-190.0, 190.0, "degrees"),
    (-120.0, 120.0, "degrees"),
    (-360.0, 360.0, "degrees"),
    (0.0, 1.0, "state"),
)
# The insert trajectory tilts about 3 degrees with a +/-2 degree correction,
# so enforce one fingers-down cone rather than separate W/P boxes.
TOOL_DOWN_TOLERANCE_DEG = 6.0
DT_LIMITS = (0.1, 1.0)
MAX_ACTION_CHUNK = 100
DtSource: TypeAlias = object | Callable[[], object]


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


def _shape_error(value: object, *, action_name: str, row_index: int | None = None) -> ActionShapeError:
    row_context = "" if row_index is None else f"; row {row_index} is malformed"
    return ActionShapeError(
        f"{action_name} action must be one row of 7 values or a chunk of 1..100 rows{row_context}",
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


def _action_rows(action: object, *, action_name: str) -> tuple[list[list], bool]:
    outer = _as_items(action)
    if not outer:
        raise _shape_error(action, action_name=action_name)
    possible_rows = [_as_items(item) for item in outer]
    is_chunk = any(row is not None for row in possible_rows)
    if not is_chunk:
        if len(outer) != len(TCP_FIELDS):
            raise _shape_error(action, action_name=action_name)
        return [outer], False
    if len(outer) > MAX_ACTION_CHUNK:
        raise _shape_error(action, action_name=action_name)
    for row_index, row in enumerate(possible_rows):
        if row is None or len(row) != len(TCP_FIELDS):
            raise _shape_error(outer[row_index], action_name=action_name, row_index=row_index)
    return [row for row in possible_rows if row is not None], True


def _finite_numeric(value: object, *, action_name: str, field: str, row_index: int | None) -> float:
    if isinstance(value, bool) or not isinstance(value, Number):
        numeric = None
    else:
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            numeric = None
    if numeric is None or not math.isfinite(numeric):
        location = f"{action_name} action" if row_index is None else f"{action_name} action row {row_index}"
        raise ActionValueError(
            f"{location} field '{field}' must be a finite numeric value, got {value!r}",
            field=field,
            row_index=row_index,
            value=value,
        )
    return numeric


def _check_bounds(
    value: float,
    *,
    action_name: str,
    field: str,
    row_index: int | None,
    limits: tuple[float, float],
    unit: str,
) -> None:
    lower, upper = limits
    if lower <= value <= upper:
        return
    location = f"{action_name} action" if row_index is None else f"{action_name} action row {row_index}"
    frame = " UF0" if field in {"x", "y", "z"} else ""
    raise ActionBoundsError(
        f"{location} field '{field}'={value!r} outside inclusive{frame} bounds [{lower}, {upper}] {unit}",
        field=field,
        row_index=row_index,
        value=value,
    )


def _guard_bounded_action(
    action: object,
    *,
    action_name: str,
    dt: object,
    fields: tuple[str, ...],
    limits: tuple[tuple[float, float, str], ...],
) -> tuple[list[list[float]], bool]:
    safe_dt = _finite_numeric(dt, action_name=action_name, field="dt", row_index=None)
    _check_bounds(
        safe_dt,
        action_name=action_name,
        field="dt",
        row_index=None,
        limits=DT_LIMITS,
        unit="seconds",
    )
    rows, is_chunk = _action_rows(action, action_name=action_name)
    safe_rows: list[list[float]] = []
    for row_index, row in enumerate(rows):
        safe_row: list[float] = []
        for field, value, (lower, upper, unit) in zip(fields, row, limits, strict=True):
            numeric = _finite_numeric(value, action_name=action_name, field=field, row_index=row_index)
            _check_bounds(
                numeric,
                action_name=action_name,
                field=field,
                row_index=row_index,
                limits=(lower, upper),
                unit=unit,
            )
            safe_row.append(numeric)
        safe_rows.append(safe_row)
    return safe_rows, is_chunk


def _check_tool_down(row: list[float], *, row_index: int) -> None:
    w, p = row[3:5]
    cosine = -math.cos(math.radians(p)) * math.cos(math.radians(w))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    if tilt <= TOOL_DOWN_TOLERANCE_DEG or math.isclose(
        tilt,
        TOOL_DOWN_TOLERANCE_DEG,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return
    raise ActionBoundsError(
        f"tcp action row {row_index} field 'tilt'={tilt!r} violates fingers-down cone "
        f"{TOOL_DOWN_TOLERANCE_DEG} degrees",
        field="tilt",
        row_index=row_index,
        value=tilt,
    )


def guard_tcp_action(action: object, *, dt: object) -> list[float] | list[list[float]]:
    """Validate an absolute fingers-down UTOOL1 target or chunk."""
    safe_rows, is_chunk = _guard_bounded_action(
        action,
        action_name="tcp",
        dt=dt,
        fields=TCP_FIELDS,
        limits=TCP_LIMITS,
    )
    for row_index, row in enumerate(safe_rows):
        _check_tool_down(row, row_index=row_index)
    return safe_rows if is_chunk else safe_rows[0]


def guard_joint_action(action: object, *, dt: object) -> list[float] | list[list[float]]:
    """Validate an absolute joint target or chunk against official limits."""
    safe_rows, is_chunk = _guard_bounded_action(
        action,
        action_name="joint",
        dt=dt,
        fields=JOINT_FIELDS,
        limits=JOINT_LIMITS,
    )
    return safe_rows if is_chunk else safe_rows[0]


def _project_public_mapping(field: str, value: object, allowed_fields: frozenset[str]) -> object:
    if field == "trial" and value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"sim observation field '{field}' must be a mapping")
    return {key: item for key, item in value.items() if key in allowed_fields}


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

    def __init__(self, controller: Controller, *, action_mode: ActionMode, dt: DtSource = 0.1) -> None:
        if action_mode not in ("joints", "tcp"):
            raise ValueError(f"unsupported action mode: {action_mode!r}")
        self._controller = controller
        self.action_mode = action_mode
        self._dt_source = dt

    def reset(self) -> None:
        self._controller.reset()

    def act(self, observation: Observation) -> Action:
        action = self._controller.act(self._controller_observation(observation))
        current_dt = self._dt_source() if callable(self._dt_source) else self._dt_source
        if self.action_mode == "tcp":
            return guard_tcp_action(action, dt=current_dt)
        return guard_joint_action(action, dt=current_dt)

    def _controller_observation(self, observation: Observation) -> Observation:
        return observation


class SimAdapter(_ControllerAdapter):
    """Project simulator observations onto fields available from the cell."""

    def _controller_observation(self, observation: Observation) -> Observation:
        projected = {key: value for key, value in observation.items() if key in OFFICIAL_OBSERVATION_FIELDS}
        for field, allowed_fields in NESTED_OBSERVATION_FIELDS.items():
            if field in projected:
                projected[field] = _project_public_mapping(field, projected[field], allowed_fields)
        return projected


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
