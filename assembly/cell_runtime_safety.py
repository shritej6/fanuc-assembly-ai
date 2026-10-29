"""Fail-closed production reachability and collision-provider composition."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from typing import Protocol

from assembly.motion import Waypoint
from assembly.safety import AABB, CollisionCapabilities, MotionOwner, ReachabilityCapabilities


class ProductionSafetyConfigurationError(ValueError):
    """A production provider lacks a recorded source or required coverage."""


class ProductionSafetyUnavailable(RuntimeError):
    """A sourced production provider could not make a trustworthy decision."""


class CADGeometryProvider(Protocol):
    """Sourced swept geometry supplied by robot CAD, part STLs, and T29 frames."""

    source: str
    models_robot_links: bool
    models_held_parts: bool

    def swept_robot_links(self, start: Waypoint, end: Waypoint) -> tuple[AABB, ...]: ...

    def swept_held_part(self, start: Waypoint, end: Waypoint, owner: MotionOwner) -> AABB: ...


class RobotLinkSweepSource(Protocol):
    source: str

    def swept_bounds(self, start: Waypoint, end: Waypoint) -> tuple[AABB, ...]: ...


class HeldPartSweepSource(Protocol):
    source: str

    def swept_bounds(self, start: Waypoint, end: Waypoint, owner: MotionOwner) -> AABB: ...


def _recorded_source(source: object) -> str:
    if not isinstance(source, str) or not source.strip():
        raise ProductionSafetyConfigurationError("production safety provider must record its source")
    return source.strip()


def _finite_pose(pose: object) -> list[float]:
    if not isinstance(pose, tuple) or len(pose) != 6:
        raise ProductionSafetyUnavailable("reachability pose must contain six values")
    values: list[float] = []
    for value in pose:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ProductionSafetyUnavailable("reachability pose must contain finite numeric values")
        values.append(float(value))
    return values


class SourcedCADGeometry:
    """Join independently sourced robot-link and held-part CAD sweep providers."""

    models_robot_links = True
    models_held_parts = True

    def __init__(self, robot_links: RobotLinkSweepSource, held_parts: HeldPartSweepSource) -> None:
        robot_source = _recorded_source(getattr(robot_links, "source", None))
        held_source = _recorded_source(getattr(held_parts, "source", None))
        self.source = f"robot={robot_source}; held={held_source}"
        self._robot_links = robot_links
        self._held_parts = held_parts

    def swept_robot_links(self, start: Waypoint, end: Waypoint) -> tuple[AABB, ...]:
        bounds = self._robot_links.swept_bounds(start, end)
        if not isinstance(bounds, tuple) or any(not isinstance(item, AABB) for item in bounds):
            raise ProductionSafetyUnavailable("robot-link CAD provider returned invalid bounds")
        return bounds

    def swept_held_part(self, start: Waypoint, end: Waypoint, owner: MotionOwner) -> AABB:
        bounds = self._held_parts.swept_bounds(start, end, owner)
        if not isinstance(bounds, AABB):
            raise ProductionSafetyUnavailable("held-part STL provider returned invalid bounds")
        return bounds


class SDGReachabilityChecker:
    """Adapter for the sourced full ER-4iA IK in ``sdg.wrist_ik``."""

    capabilities = ReachabilityCapabilities(full_robot_kinematics=True)

    def __init__(self, reachable: Callable[[list[float]], object], *, source: str) -> None:
        if not callable(reachable):
            raise ProductionSafetyConfigurationError("reachability solver must be callable")
        self._reachable = reachable
        self.source = _recorded_source(source)

    @classmethod
    def from_sdg(cls) -> SDGReachabilityChecker:
        try:
            module = importlib.import_module("sdg.wrist_ik")
        except (ImportError, OSError) as error:
            raise ProductionSafetyConfigurationError("sdg.wrist_ik is required for cell reachability") from error
        source = getattr(module, "__file__", None)
        reachable = getattr(module, "reachable", None)
        return cls(reachable, source=f"{_recorded_source(source)}:reachable")

    def is_reachable(self, pose: tuple[float, float, float, float, float, float]) -> bool:
        try:
            result = self._reachable(_finite_pose(pose))
        except ProductionSafetyUnavailable:
            raise
        except Exception as error:
            raise ProductionSafetyUnavailable(f"full IK failed using {self.source}") from error
        if not isinstance(result, bool):
            raise ProductionSafetyUnavailable("full IK must return a boolean decision")
        return result


def _overlaps(left: AABB, right: AABB) -> bool:
    return all(
        left.minimum[axis] <= right.maximum[axis] and right.minimum[axis] <= left.maximum[axis]
        for axis in range(3)
    )


def _tool_sweep(start: Waypoint, end: Waypoint, radius: float) -> AABB:
    if not math.isfinite(radius) or radius < 0.0:
        raise ProductionSafetyUnavailable("tool sweep radius must be finite and nonnegative")
    minimum = tuple(min(a, b) - radius for a, b in zip(start.xyz, end.xyz, strict=True))
    maximum = tuple(max(a, b) + radius for a, b in zip(start.xyz, end.xyz, strict=True))
    return AABB("tool-sweep", minimum, maximum)


def _require_cad_bounds(body: object) -> AABB:
    if not isinstance(body, AABB) or not body.name:
        raise ProductionSafetyUnavailable("CAD bounds must be a named AABB")
    for low, high in zip(body.minimum, body.maximum, strict=True):
        values = (low, high)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ProductionSafetyUnavailable(f"CAD bounds for {body.name!r} must be numeric")
        if not all(math.isfinite(float(value)) for value in values) or low > high:
            raise ProductionSafetyUnavailable(f"CAD bounds for {body.name!r} are inverted or nonfinite")
    return body


def _collision(
    bodies: Sequence[AABB],
    obstacles: Sequence[AABB],
    owner: MotionOwner,
    prefix: str | None,
) -> str | None:
    for body in bodies:
        body = _require_cad_bounds(body)
        label = body.name if prefix is None else f"{prefix}:{body.name}"
        for obstacle in obstacles:
            if obstacle.name != owner.instance_id and _overlaps(body, obstacle):
                return f"{label} with {obstacle.name}"
    return None


class CADCollisionChecker:
    """Conservative collision gate backed by sourced swept robot and STL geometry."""

    capabilities = CollisionCapabilities(robot_links=True, held_part_geometry=True)

    def __init__(self, geometry: CADGeometryProvider) -> None:
        self.source = _recorded_source(getattr(geometry, "source", None))
        if getattr(geometry, "models_robot_links", None) is not True or getattr(
            geometry, "models_held_parts", None
        ) is not True:
            raise ProductionSafetyConfigurationError("CAD geometry must model robot links and held parts")
        self._geometry = geometry

    def first_collision(
        self,
        start: Waypoint,
        end: Waypoint,
        obstacles: Sequence[AABB],
        sweep_radius_mm: float,
        owner: MotionOwner,
    ) -> str | None:
        try:
            links = self._geometry.swept_robot_links(start, end)
            if not links:
                raise ProductionSafetyUnavailable("robot-link CAD sweep returned no geometry")
            collision = _collision(links, obstacles, owner, "robot-link")
            if collision is not None:
                return collision
            if owner.attached:
                held = self._geometry.swept_held_part(start, end, owner)
                collision = _collision((held,), obstacles, owner, None)
                if collision is not None:
                    return collision
            return _collision((_tool_sweep(start, end, sweep_radius_mm),), obstacles, owner, None)
        except ProductionSafetyUnavailable:
            raise
        except Exception as error:
            raise ProductionSafetyUnavailable(f"CAD collision query failed using {self.source}") from error
