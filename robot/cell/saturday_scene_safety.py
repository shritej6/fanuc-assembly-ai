"""PRELIMINARY TCP-point scene checks for the supervised Saturday cell run.

This supervisor deliberately does not claim full robot IK, robot-link collision,
or held-part sweep coverage.  Those production capabilities remain deferred.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import yaml

from assembly.motion import MotionChunk, Waypoint
from assembly.safety import (
    AABB,
    CartesianSpeedViolation,
    CollisionCapabilities,
    CollisionViolation,
    EmptyPlanViolation,
    MotionOwner,
    OfficialActionGuardViolation,
    ReachabilityCapabilities,
    SafetyConfigurationViolation,
    SafetySupervisor,
    SafetyViolation,
    TableClearanceViolation,
)
from policy_learn.deploy.cell_groot_policy import DT_S
from policy_learn.deploy.cell_guards import GuardLimits
from robot.adapter import ActionValidationError, guard_tcp_action

CELL_LAYOUT_PATH = Path(__file__).resolve().parents[2] / "sdg" / "configs" / "cell_layout.yaml"


class LayoutWorkspaceViolation(SafetyViolation):
    """A TCP point is outside every operator zone in the sourced layout."""


@dataclass(frozen=True, slots=True)
class _LayoutBounds:
    x_mm: tuple[float, float]
    y_mm: tuple[float, float]
    table_z_mm: float


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SafetyConfigurationViolation(f"cell layout {label} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise SafetyConfigurationViolation(f"cell layout {label} must be finite")
    return number


def _interval(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise SafetyConfigurationViolation(f"cell layout {label} must be a two-value interval")
    interval = (_finite(value[0], label), _finite(value[1], label))
    if interval[0] >= interval[1]:
        raise SafetyConfigurationViolation(f"cell layout {label} must increase")
    return interval


def _load_layout() -> _LayoutBounds:
    try:
        payload = yaml.safe_load(CELL_LAYOUT_PATH.read_text(encoding="utf-8"))
        work = payload["work_area"]
        reject = payload["reject_zone"]
        slots = payload["spare_store"]["slots"]
        work_x, work_y = _interval(work["x"], "work_area.x"), _interval(work["y"], "work_area.y")
        reject_x = _interval(reject["x"], "reject_zone.x")
        reject_y = _interval(reject["y"], "reject_zone.y")
        slot_x = [_finite(slot["x"], "spare_store.slot.x") for slot in slots]
        slot_y = [_finite(slot["y"], "spare_store.slot.y") for slot in slots]
        table_z = _finite(payload["table_z"], "table_z")
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        raise SafetyConfigurationViolation(f"cannot load checks-only cell layout: {error}") from error
    if not slot_x or not slot_y:
        raise SafetyConfigurationViolation("cell layout spare store must contain slots")
    return _LayoutBounds(
        (min(*work_x, *reject_x, *slot_x), max(*work_x, *reject_x, *slot_x)),
        (min(*work_y, *reject_y, *slot_y), max(*work_y, *reject_y, *slot_y)),
        table_z,
    )


def _one_row_chunks(plan: Sequence[MotionChunk]) -> tuple[MotionChunk, ...]:
    immutable = tuple(plan)
    if not immutable:
        raise EmptyPlanViolation("checks-only scene plan must be non-empty")
    for index, chunk in enumerate(immutable):
        if not isinstance(chunk, MotionChunk) or len(chunk.waypoints) != 1:
            raise SafetyConfigurationViolation(
                f"checks-only scene chunk {index} must contain exactly one waypoint"
            )
    return immutable


def _official_guard(
    chunks: tuple[MotionChunk, ...],
    start: Waypoint,
    dt_s: float,
) -> None:
    try:
        guard_tcp_action(start.as_action(), dt=dt_s)
        for chunk in chunks:
            guard_tcp_action(chunk.as_action(), dt=dt_s)
    except ActionValidationError as error:
        raise OfficialActionGuardViolation(
            f"official UF0 action guard rejected checks-only motion: {error}"
        ) from error


def _layout_check(points: tuple[Waypoint, ...], bounds: _LayoutBounds) -> None:
    for index, point in enumerate(points):
        inside_xy = (
            bounds.x_mm[0] <= point.x <= bounds.x_mm[1]
            and bounds.y_mm[0] <= point.y <= bounds.y_mm[1]
        )
        if not inside_xy:
            raise LayoutWorkspaceViolation(
                f"TCP point {index} ({point.x}, {point.y}) is outside layout workspace "
                f"x={bounds.x_mm}, y={bounds.y_mm}"
            )
        if point.z < bounds.table_z_mm:
            raise TableClearanceViolation(
                f"TCP point {index} z={point.z} is below table floor {bounds.table_z_mm}"
            )


def _wrapped_w_delta(after: float, before: float) -> float:
    return (after - before + 180.0) % 360.0 - 180.0


def _step_check(
    points: tuple[Waypoint, ...],
    bounds: _LayoutBounds,
    limits: GuardLimits,
    dt_s: float,
) -> None:
    for index, (before, after) in enumerate(pairwise(points)):
        slow = min(before.z, after.z) <= bounds.table_z_mm + limits.slow_zone_mm
        linear_limit = (limits.slow_speed_mm_s if slow else limits.max_speed_mm_s) * dt_s
        angular_limit = (limits.slow_rot_deg_s if slow else limits.max_rot_deg_s) * dt_s
        linear = math.dist(before.xyz, after.xyz)
        angular = max(
            abs(_wrapped_w_delta(after.w, before.w)),
            abs(after.p - before.p),
            abs(after.r - before.r),
        )
        if linear > linear_limit or angular > angular_limit:
            zone = "slow-zone " if slow else ""
            raise CartesianSpeedViolation(
                f"PRELIMINARY {zone}row {index} step {linear:.3f} mm / {angular:.3f} deg "
                f"exceeds {linear_limit:.3f} mm / {angular_limit:.3f} deg"
            )


def _validated_box(value: object) -> AABB:
    if not isinstance(value, AABB) or not isinstance(value.name, str) or not value.name:
        raise SafetyConfigurationViolation("scene obstacle must be a named AABB")
    if len(value.minimum) != 3 or len(value.maximum) != 3:
        raise SafetyConfigurationViolation(f"scene obstacle {value.name!r} must have three axes")
    for low, high in zip(value.minimum, value.maximum, strict=True):
        if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in (low, high)):
            raise SafetyConfigurationViolation(f"scene obstacle {value.name!r} bounds must be numeric")
        if not math.isfinite(float(low)) or not math.isfinite(float(high)) or low > high:
            raise SafetyConfigurationViolation(f"scene obstacle {value.name!r} bounds are invalid")
    return value


def _point_collisions(points: tuple[Waypoint, ...], obstacles: tuple[AABB, ...]) -> None:
    checked = tuple(_validated_box(obstacle) for obstacle in obstacles)
    for index, point in enumerate(points):
        for obstacle in checked:
            coordinates = point.xyz
            inside = all(
                obstacle.minimum[axis] <= coordinates[axis] <= obstacle.maximum[axis]
                for axis in range(3)
            )
            if inside:
                raise CollisionViolation(f"TCP point {index} intersects {obstacle.name}")


class SaturdaySceneSafetySupervisor(SafetySupervisor):
    """Checks-only scene supervisor; never physical-motion authority by itself."""

    status = "PRELIMINARY"
    checks_only = True
    reachability_capabilities = ReachabilityCapabilities(full_robot_kinematics=False)
    collision_capabilities = CollisionCapabilities(
        robot_links=False,
        held_part_geometry=False,
    )

    def __init__(self) -> None:
        # SafetySupervisor.__init__ correctly refuses these incomplete
        # capabilities.  This subclass supplies a separate checks-only
        # validate implementation and is never represented as production-ready.
        self._dt_s = DT_S
        self._layout = _load_layout()
        self._limits = GuardLimits(
            max_tilt_deg=6.0,
            status="PRELIMINARY",
            source="supervised Saturday checks-only limits; pending CSI measurement",
        )

    @property
    def workspace_xy_mm(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self._layout.x_mm, self._layout.y_mm

    @property
    def table_z_mm(self) -> float:
        return self._layout.table_z_mm

    @property
    def layout_source(self) -> str:
        return str(CELL_LAYOUT_PATH)

    def validate(
        self,
        plan: Sequence[MotionChunk],
        *,
        start: Waypoint,
        fixtures: Sequence[AABB],
        parts: Sequence[AABB],
        owner: MotionOwner,
    ) -> tuple[MotionChunk, ...]:
        """Return the immutable plan after the checks-only gates pass."""
        del owner
        chunks = _one_row_chunks(plan)
        _official_guard(chunks, start, self._dt_s)
        points = (start, *(chunk.waypoints[0] for chunk in chunks))
        _layout_check(points, self._layout)
        _step_check(points, self._layout, self._limits, self._dt_s)
        _point_collisions(points[1:], (*fixtures, *parts))
        return chunks
