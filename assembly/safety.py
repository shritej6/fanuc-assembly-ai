"""Fail-closed supervision for absolute UF0 motion chunks.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Number
from typing import Protocol

from assembly.motion import MotionChunk, Waypoint
from robot.adapter import ActionValidationError, guard_tcp_action


class SafetyViolation(RuntimeError):
    """Base error for motion rejected by a safety gate."""


class ToolOrientationViolation(SafetyViolation):
    """UTOOL1 is not in the configured downward orientation."""


class TableClearanceViolation(SafetyViolation):
    """A fingertip waypoint violates configured table clearance."""


class UnreachableWaypointViolation(SafetyViolation):
    """The injected checker reports a waypoint unreachable."""


class ReachabilityUnavailable(SafetyViolation):
    """No trustworthy reachability decision is available."""


class CollisionViolation(SafetyViolation):
    """A swept TCP segment intersects supplied scene geometry."""


class CollisionUnavailable(SafetyViolation):
    """No trustworthy collision decision is available."""


class SafetyConfigurationViolation(SafetyViolation):
    """Safety configuration or obstacle geometry is invalid."""


class OfficialActionGuardViolation(SafetyViolation):
    """The official UF0 action guard rejected a planned waypoint."""


class EmptyPlanViolation(SafetyViolation):
    """No executable waypoint was supplied."""


class CartesianSpeedViolation(SafetyViolation):
    """A waypoint segment exceeds the configured per-step TCP speed."""


@dataclass(frozen=True, slots=True)
class ReachabilityCapabilities:
    """Declared kinematic coverage of an injected reachability checker."""

    full_robot_kinematics: bool


@dataclass(frozen=True, slots=True)
class CollisionCapabilities:
    """Declared geometry coverage of an injected collision checker."""

    robot_links: bool
    held_part_geometry: bool


@dataclass(frozen=True, slots=True)
class MotionOwner:
    """Physical object whose geometry moves with the commanded tool."""

    instance_id: str
    object_id: str
    attached: bool


class ReachabilityChecker(Protocol):
    """Injected kinematic boundary used identically in tests and runtimes."""

    capabilities: ReachabilityCapabilities

    def is_reachable(self, pose: tuple[float, float, float, float, float, float]) -> bool: ...


class CollisionChecker(Protocol):
    """Injected swept-segment collision boundary."""

    capabilities: CollisionCapabilities

    def first_collision(
        self,
        start: Waypoint,
        end: Waypoint,
        obstacles: Sequence[AABB],
        sweep_radius_mm: float,
        owner: MotionOwner,
    ) -> str | None: ...


@dataclass(frozen=True)
class SafetyConfig:
    """Caller-supplied cell safety parameters."""

    dt_s: float
    table_z_mm: float
    min_table_clearance_mm: float
    expected_w_deg: float
    expected_p_deg: float
    orientation_tolerance_deg: float
    sweep_radius_mm: float
    max_cartesian_speed_mm_s: float
    max_cartesian_speed_source: str


@dataclass(frozen=True)
class AABB:
    """Named fixture or part bounds in absolute UF0 millimetres."""

    name: str
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]


def _pose(waypoint: Waypoint) -> tuple[float, float, float, float, float, float]:
    return (waypoint.x, waypoint.y, waypoint.z, waypoint.w, waypoint.p, waypoint.r)


def _angle_error(actual: float, expected: float) -> float:
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


def _segment_hits_box(start: Waypoint, end: Waypoint, box: AABB, radius: float) -> bool:
    lower = tuple(value - radius for value in box.minimum)
    upper = tuple(value + radius for value in box.maximum)
    t_min, t_max = 0.0, 1.0
    for origin, destination, low, high in zip(start.xyz, end.xyz, lower, upper, strict=True):
        delta = destination - origin
        if delta == 0.0:
            if origin < low or origin > high:
                return False
            continue
        near, far = sorted(((low - origin) / delta, (high - origin) / delta))
        t_min, t_max = max(t_min, near), min(t_max, far)
        if t_min > t_max:
            return False
    return True


class AABBCollisionChecker:
    """Mock/test-only swept-TCP check; it does not model robot links."""

    capabilities = CollisionCapabilities(robot_links=False, held_part_geometry=False)

    def first_collision(
        self,
        start: Waypoint,
        end: Waypoint,
        obstacles: Sequence[AABB],
        sweep_radius_mm: float,
        owner: MotionOwner,
    ) -> str | None:
        del owner
        for obstacle in obstacles:
            if _segment_hits_box(start, end, obstacle, sweep_radius_mm):
                return obstacle.name
        return None


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Number):
        raise SafetyConfigurationViolation(f"safety {field} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise SafetyConfigurationViolation(f"safety {field} must be finite") from error
    if not math.isfinite(numeric):
        raise SafetyConfigurationViolation(f"safety {field} must be finite")
    return numeric


def _validate_config(config: SafetyConfig) -> None:
    dt_s = _finite(config.dt_s, "dt_s")
    _finite(config.table_z_mm, "table_z_mm")
    expected_w = _finite(config.expected_w_deg, "expected_w_deg")
    expected_p = _finite(config.expected_p_deg, "expected_p_deg")
    clearance = _finite(config.min_table_clearance_mm, "min_table_clearance_mm")
    tolerance = _finite(config.orientation_tolerance_deg, "orientation_tolerance_deg")
    radius = _finite(config.sweep_radius_mm, "sweep_radius_mm")
    speed = _finite(config.max_cartesian_speed_mm_s, "max_cartesian_speed_mm_s")
    if not isinstance(config.max_cartesian_speed_source, str) or not config.max_cartesian_speed_source.strip():
        raise SafetyConfigurationViolation("safety max_cartesian_speed_mm_s requires a recorded source")
    if dt_s <= 0.0:
        raise SafetyConfigurationViolation("safety dt_s must be positive")
    if clearance < 0.0 or radius < 0.0:
        raise SafetyConfigurationViolation("safety clearances must be nonnegative")
    if speed <= 0.0:
        raise SafetyConfigurationViolation("safety max_cartesian_speed_mm_s must be positive")
    if _angle_error(expected_w, -180.0) > 1e-9 or _angle_error(expected_p, 0.0) > 1e-9:
        raise SafetyConfigurationViolation("safety orientation must remain official UTOOL1 tool-down")
    if not 0.0 <= tolerance < 90.0:
        raise SafetyConfigurationViolation("safety orientation tolerance must be in [0, 90)")


def _validate_obstacle(obstacle: object) -> AABB:
    if not isinstance(obstacle, AABB) or not isinstance(obstacle.name, str) or not obstacle.name:
        raise SafetyConfigurationViolation("obstacle must be a named AABB")
    if not _valid_bounds_shape(obstacle.minimum) or not _valid_bounds_shape(obstacle.maximum):
        raise SafetyConfigurationViolation(f"obstacle {obstacle.name!r} must have three-axis bounds")
    for axis, (low, high) in enumerate(zip(obstacle.minimum, obstacle.maximum, strict=True)):
        safe_low = _finite(low, f"obstacle {obstacle.name!r} axis {axis} minimum")
        safe_high = _finite(high, f"obstacle {obstacle.name!r} axis {axis} maximum")
        if safe_low > safe_high:
            raise SafetyConfigurationViolation(f"obstacle {obstacle.name!r} has inverted bounds")
    return obstacle


def _valid_bounds_shape(bounds: object) -> bool:
    return isinstance(bounds, tuple) and len(bounds) == 3


def _reachability_capabilities(checker: object) -> ReachabilityCapabilities | None:
    capabilities = getattr(checker, "capabilities", None)
    return capabilities if isinstance(capabilities, ReachabilityCapabilities) else None


def _collision_capabilities(checker: object) -> CollisionCapabilities | None:
    capabilities = getattr(checker, "capabilities", None)
    return capabilities if isinstance(capabilities, CollisionCapabilities) else None


def _validate_owner(owner: object) -> MotionOwner:
    if not isinstance(owner, MotionOwner):
        raise SafetyConfigurationViolation("motion owner context is required")
    if not owner.instance_id or not owner.object_id or not isinstance(owner.attached, bool):
        raise SafetyConfigurationViolation("motion owner context is invalid")
    return owner


class SafetySupervisor:
    """Run adapter, orientation, table, and reachability gates before motion."""

    def __init__(
        self,
        *,
        config: SafetyConfig,
        reachability: ReachabilityChecker | None,
        collision: CollisionChecker | None,
        allow_tcp_sweep_only_for_mock: bool = False,
    ) -> None:
        if collision is None:
            raise SafetyConfigurationViolation("an explicit collision checker is required")
        reachability_caps = _reachability_capabilities(reachability)
        collision_caps = _collision_capabilities(collision)
        if reachability is not None and reachability_caps is None:
            raise SafetyConfigurationViolation("reachability checker must declare capabilities")
        if collision_caps is None:
            raise SafetyConfigurationViolation("collision checker must declare capabilities")
        if not allow_tcp_sweep_only_for_mock:
            self._require_production_capabilities(reachability_caps, collision_caps)
        self._config = config
        self._reachability = reachability
        self._collision = collision

    @staticmethod
    def _require_production_capabilities(
        reachability: ReachabilityCapabilities | None,
        collision: CollisionCapabilities,
    ) -> None:
        if reachability is None or reachability.full_robot_kinematics is not True:
            raise SafetyConfigurationViolation("production reachability must model full robot kinematics")
        if collision.robot_links is not True or collision.held_part_geometry is not True:
            raise SafetyConfigurationViolation("production collision checking must model robot links and held parts")

    def _guard(self, plan: Sequence[MotionChunk], start: Waypoint) -> None:
        try:
            guard_tcp_action(start.as_action(), dt=self._config.dt_s)
            for chunk in plan:
                guard_tcp_action(chunk.as_action(), dt=self._config.dt_s)
        except ActionValidationError as error:
            raise OfficialActionGuardViolation(f"official UF0 action guard rejected motion: {error}") from error

    def _check_waypoint(self, waypoint: Waypoint, index: int) -> None:
        config = self._config
        if _angle_error(waypoint.w, config.expected_w_deg) > config.orientation_tolerance_deg:
            raise ToolOrientationViolation(f"waypoint {index} has unsafe tool orientation W={waypoint.w!r}")
        if _angle_error(waypoint.p, config.expected_p_deg) > config.orientation_tolerance_deg:
            raise ToolOrientationViolation(f"waypoint {index} has unsafe tool orientation P={waypoint.p!r}")
        if waypoint.z < config.table_z_mm + config.min_table_clearance_mm:
            raise TableClearanceViolation(f"waypoint {index} violates table clearance at z={waypoint.z!r}")
        if self._reachability is None:
            raise ReachabilityUnavailable("reachability checker is unavailable")
        try:
            reachable = self._reachability.is_reachable(_pose(waypoint))
        except Exception as error:
            raise ReachabilityUnavailable(f"reachability checker failed at waypoint {index}") from error
        if reachable is not True:
            raise UnreachableWaypointViolation(f"waypoint {index} is unreachable")

    def _check_segment(
        self,
        start: Waypoint,
        end: Waypoint,
        obstacles: Sequence[AABB],
        owner: MotionOwner,
        index: int,
    ) -> None:
        speed = math.dist(start.xyz, end.xyz) / self._config.dt_s
        if speed > self._config.max_cartesian_speed_mm_s:
            raise CartesianSpeedViolation(f"segment {index} speed {speed!r} exceeds configured limit")
        try:
            collision = self._collision.first_collision(start, end, obstacles, self._config.sweep_radius_mm, owner)
        except Exception as error:
            raise CollisionUnavailable(f"collision checker failed for segment {index}") from error
        if collision is None:
            return
        if not isinstance(collision, str) or not collision:
            raise CollisionUnavailable(f"collision checker returned an invalid result for segment {index}")
        raise CollisionViolation(f"segment {index} swept collision with {collision}")

    def validate(
        self,
        plan: Sequence[MotionChunk],
        *,
        start: Waypoint,
        fixtures: Sequence[AABB],
        parts: Sequence[AABB],
        owner: MotionOwner,
    ) -> tuple[MotionChunk, ...]:
        """Return an immutable plan only after all configured gates pass."""
        immutable = tuple(plan)
        if not immutable:
            raise EmptyPlanViolation("motion plan must not be empty")
        _validate_config(self._config)
        self._guard(immutable, start)
        motion_owner = _validate_owner(owner)
        obstacles = tuple(_validate_obstacle(obstacle) for obstacle in (*fixtures, *parts))
        waypoints = (start, *(point for chunk in immutable for point in chunk.waypoints))
        for index, waypoint in enumerate(waypoints):
            self._check_waypoint(waypoint, index)
        for index, (before, after) in enumerate(pairwise(waypoints)):
            self._check_segment(before, after, obstacles, motion_owner, index)
        return immutable
