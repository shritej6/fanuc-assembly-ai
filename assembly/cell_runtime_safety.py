"""Preflight reachability and swept-geometry adapters with explicit provenance.

Importing a simulator helper is not evidence that it matches the physical cell. These
adapters validate shape and declared source evidence, but intentionally do not advertise
the production capability flags required by :class:`SafetySupervisor`. A future sealed
loader must bind the transitive code, CAD, joint limits, tool, and physical validation.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from assembly.motion import Waypoint
from assembly.safety import AABB, CollisionCapabilities, MotionOwner, ReachabilityCapabilities


class ProductionSafetyConfigurationError(ValueError):
    """A production provider lacks reviewed immutable provenance."""


class ProductionSafetyUnavailable(RuntimeError):
    """A reviewed provider could not return a trustworthy decision."""


class SafetySourceKind(StrEnum):
    FULL_IK_JOINT_LIMITS = "full_ik_joint_limits"
    ROBOT_LINK_SWEEP = "robot_link_sweep"
    HELD_PART_SWEEP = "held_part_sweep"


@dataclass(frozen=True, slots=True)
class ReviewedSafetySource:
    """Immutable evidence tying one provider to a reviewed implementation/input set."""

    kind: SafetySourceKind
    source: str
    sha256: str
    status: str = "REVIEWED"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SafetySourceKind):
            raise ProductionSafetyConfigurationError("safety source kind must be typed")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ProductionSafetyConfigurationError("production safety source must be recorded")
        digest = self.sha256
        valid = isinstance(digest, str) and len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
        if not valid:
            raise ProductionSafetyConfigurationError("production safety source requires a lowercase SHA-256")
        if self.status != "REVIEWED":
            raise ProductionSafetyConfigurationError("production safety source must have REVIEWED status")


class RobotLinkSweepSource(Protocol):
    provenance: ReviewedSafetySource

    def swept_bounds(self, start: Waypoint, end: Waypoint) -> tuple[AABB, ...]: ...


class HeldPartSweepSource(Protocol):
    provenance: ReviewedSafetySource

    def swept_bounds(self, start: Waypoint, end: Waypoint, owner: MotionOwner) -> AABB: ...


def _require_source(value: object, kind: SafetySourceKind) -> ReviewedSafetySource:
    if not isinstance(value, ReviewedSafetySource) or value.kind is not kind:
        raise ProductionSafetyConfigurationError(f"provider requires reviewed {kind.value} provenance")
    return value


def _finite_pose(pose: object) -> list[float]:
    if not isinstance(pose, tuple) or len(pose) != 6:
        raise ProductionSafetyUnavailable("reachability pose must contain six values")
    values: list[float] = []
    for value in pose:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ProductionSafetyUnavailable("reachability pose must contain finite numeric values")
        values.append(float(value))
    return values


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ProductionSafetyConfigurationError(f"cannot hash reachability source {path}") from error


class SDGReachabilityChecker:
    """SDG preflight adapter; simulator IK is not physical reachability authority."""

    capabilities = ReachabilityCapabilities(full_robot_kinematics=False)

    def __init__(self, reachable: Callable[[list[float]], object], provenance: ReviewedSafetySource) -> None:
        if not callable(reachable):
            raise ProductionSafetyConfigurationError("reachability solver must be callable")
        self.provenance = _require_source(provenance, SafetySourceKind.FULL_IK_JOINT_LIMITS)
        self.source = self.provenance.source
        self._reachable = reachable

    @classmethod
    def from_sdg(cls, provenance: ReviewedSafetySource) -> SDGReachabilityChecker:
        """Load the solver only when its exact implementation matches reviewed evidence."""
        reviewed = _require_source(provenance, SafetySourceKind.FULL_IK_JOINT_LIMITS)
        try:
            module = importlib.import_module("sdg.wrist_ik")
        except (ImportError, OSError) as error:
            raise ProductionSafetyConfigurationError("sdg.wrist_ik is required for cell reachability") from error
        source = Path(getattr(module, "__file__", ""))
        if not source.is_file() or _file_sha256(source) != reviewed.sha256:
            raise ProductionSafetyConfigurationError("sdg.wrist_ik does not match its reviewed SHA-256")
        return cls(getattr(module, "reachable", None), reviewed)

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
        left.minimum[axis] <= right.maximum[axis] and right.minimum[axis] <= left.maximum[axis] for axis in range(3)
    )


def _validate_bounds(body: object) -> AABB:
    if not isinstance(body, AABB) or not body.name:
        raise ProductionSafetyUnavailable("swept CAD bounds must be a named AABB")
    for low, high in zip(body.minimum, body.maximum, strict=True):
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (low, high)):
            raise ProductionSafetyUnavailable(f"CAD bounds for {body.name!r} must be numeric")
        if not all(math.isfinite(float(value)) for value in (low, high)) or low > high:
            raise ProductionSafetyUnavailable(f"CAD bounds for {body.name!r} are inverted or nonfinite")
    return body


def _first_collision(
    bodies: Sequence[AABB], obstacles: Sequence[AABB], owner: MotionOwner, prefix: str | None
) -> str | None:
    for body in bodies:
        checked = _validate_bounds(body)
        label = checked.name if prefix is None else f"{prefix}:{checked.name}"
        for obstacle in obstacles:
            checked_obstacle = _validate_bounds(obstacle)
            if checked_obstacle.name != owner.instance_id and _overlaps(checked, checked_obstacle):
                return f"{label} with {checked_obstacle.name}"
    return None


class SourcedCADGeometry:
    """Join declared robot-link and held-part sweeps for preflight diagnostics."""

    models_robot_links = False
    models_held_parts = False

    def __init__(self, robot_links: RobotLinkSweepSource, held_parts: HeldPartSweepSource) -> None:
        robot = _require_source(getattr(robot_links, "provenance", None), SafetySourceKind.ROBOT_LINK_SWEEP)
        held = _require_source(getattr(held_parts, "provenance", None), SafetySourceKind.HELD_PART_SWEEP)
        self.source = f"robot={robot.source}@{robot.sha256}; held={held.source}@{held.sha256}"
        self._robot_links = robot_links
        self._held_parts = held_parts

    def swept_robot_links(self, start: Waypoint, end: Waypoint) -> tuple[AABB, ...]:
        bounds = self._robot_links.swept_bounds(start, end)
        if not isinstance(bounds, tuple) or not bounds:
            raise ProductionSafetyUnavailable("robot-link CAD sweep returned no geometry")
        return tuple(_validate_bounds(item) for item in bounds)

    def swept_held_part(self, start: Waypoint, end: Waypoint, owner: MotionOwner) -> AABB:
        return _validate_bounds(self._held_parts.swept_bounds(start, end, owner))


class CADCollisionChecker:
    """Preflight collision checker; it cannot authorize physical motion."""

    capabilities = CollisionCapabilities(robot_links=False, held_part_geometry=False)

    def __init__(self, geometry: SourcedCADGeometry) -> None:
        if not isinstance(geometry, SourcedCADGeometry):
            raise ProductionSafetyConfigurationError("collision geometry must be a SourcedCADGeometry")
        self.source = geometry.source
        self._geometry = geometry

    def first_collision(
        self,
        start: Waypoint,
        end: Waypoint,
        obstacles: Sequence[AABB],
        sweep_radius_mm: float,
        owner: MotionOwner,
    ) -> str | None:
        del sweep_radius_mm  # reviewed link/held sweeps already contain their physical envelopes
        try:
            collision = _first_collision(self._geometry.swept_robot_links(start, end), obstacles, owner, "robot-link")
            if collision is not None:
                return collision
            if owner.attached:
                held = self._geometry.swept_held_part(start, end, owner)
                return _first_collision((held,), obstacles, owner, None)
            return None
        except ProductionSafetyUnavailable:
            raise
        except Exception as error:
            raise ProductionSafetyUnavailable(f"CAD collision query failed using {self.source}") from error
