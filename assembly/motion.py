"""Deterministic motion compilation from calibrated absolute UF0 TCP poses."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from numbers import Number

from robot.adapter import MAX_ACTION_CHUNK


class MotionError(ValueError):
    """Base error for requests that cannot safely become motion."""


class MotionInputError(MotionError):
    """A pose is untrusted, fallen, nonfinite, or needs reorientation."""


class MotionConfigError(MotionError):
    """Caller-supplied motion limits are invalid."""


class MotionPhase(StrEnum):
    """Required pick-and-insert phase order."""

    APPROACH = "approach"
    DESCEND = "descend"
    GRIP = "grip"
    LIFT = "lift"
    CARRY_YAW = "carry_yaw"
    HOVER = "hover"
    VERTICAL_INSERT = "vertical_insert"
    RELEASE = "release"
    RETREAT = "retreat"


class RestingState(StrEnum):
    """Observed physical presentation of a part."""

    INSTALLED = "installed"
    FALLEN = "fallen"


@dataclass(frozen=True)
class AbsolutePoseUF0:
    """Already calibrated absolute UTOOL1 pose in controller frame UF0."""

    x: float
    y: float
    z: float
    w: float
    p: float
    r: float
    frame: str
    source: str


@dataclass(frozen=True)
class OrientedPartPose:
    """A calibrated TCP pose plus the observed part orientation identity."""

    tcp: AbsolutePoseUF0
    resting_state: RestingState
    up_face: str


@dataclass(frozen=True)
class MotionConfig:
    """Caller-supplied clearances and bounds; no cell distances are guessed."""

    approach_mm: float
    lift_mm: float
    hover_mm: float
    retreat_mm: float
    max_linear_step_mm: float
    max_yaw_step_deg: float
    max_waypoints_per_chunk: int


@dataclass(frozen=True)
class Waypoint:
    """One absolute T05-compatible TCP action row."""

    x: float
    y: float
    z: float
    w: float
    p: float
    r: float
    gripper: float

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def as_action(self) -> list[float]:
        return [self.x, self.y, self.z, self.w, self.p, self.r, self.gripper]


@dataclass(frozen=True)
class MotionChunk:
    """A short, phase-labelled chunk accepted by the official callback."""

    phase: MotionPhase
    waypoints: tuple[Waypoint, ...]

    def as_action(self) -> list[list[float]]:
        return [waypoint.as_action() for waypoint in self.waypoints]


def _finite(value: object, *, field: str, error_type: type[MotionError]) -> float:
    if isinstance(value, bool) or not isinstance(value, Number):
        raise error_type(f"{field} must be a finite numeric value")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise error_type(f"{field} must be a finite numeric value") from error
    if not math.isfinite(numeric):
        raise error_type(f"{field} must be a finite numeric value")
    return numeric


def _validate_pose(label: str, part_pose: object) -> None:
    if not isinstance(part_pose, OrientedPartPose):
        raise MotionInputError(f"{label} pose must be an OrientedPartPose")
    pose = part_pose.tcp
    if not isinstance(pose, AbsolutePoseUF0):
        raise MotionInputError(f"{label} TCP must be an AbsolutePoseUF0")
    if part_pose.resting_state != RestingState.INSTALLED:
        raise MotionInputError(f"{label} pose must be installed, got {part_pose.resting_state!r}")
    if pose.frame != "UF0":
        raise MotionInputError(f"{label} pose must be in absolute UF0, got {pose.frame!r}")
    if pose.source != "calibrated":
        raise MotionInputError(f"{label} pose source must be calibrated, got {pose.source!r}")
    for field in ("x", "y", "z", "w", "p", "r"):
        _finite(getattr(pose, field), field=f"{label}.{field}", error_type=MotionInputError)
    if not isinstance(part_pose.up_face, str) or not part_pose.up_face.strip():
        raise MotionInputError(f"{label} installed up-face must be identified")


def _validate_config(config: MotionConfig) -> None:
    for field in ("approach_mm", "lift_mm", "hover_mm", "retreat_mm"):
        if _finite(getattr(config, field), field=field, error_type=MotionConfigError) < 0.0:
            raise MotionConfigError(f"motion clearance {field} must be nonnegative")
    for field in ("max_linear_step_mm", "max_yaw_step_deg"):
        if _finite(getattr(config, field), field=field, error_type=MotionConfigError) <= 0.0:
            raise MotionConfigError(f"{field} must be finite and positive")
    size = config.max_waypoints_per_chunk
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_ACTION_CHUNK:
        raise MotionConfigError(f"max_waypoints_per_chunk must be an integer in 1..{MAX_ACTION_CHUNK}")


def _validate_request(grasp: OrientedPartPose, target: OrientedPartPose, config: MotionConfig) -> None:
    _validate_pose("grasp", grasp)
    _validate_pose("target", target)
    _validate_config(config)
    if grasp.up_face != target.up_face:
        raise MotionInputError("grasp and target installed up-face must match")


def _preserve_roll_pitch(reference: AbsolutePoseUF0, target: AbsolutePoseUF0) -> AbsolutePoseUF0:
    """Keep the observed installed up-face; only target yaw may change."""
    return replace(target, w=reference.w, p=reference.p)


def _apply_gripper_roll(pose: AbsolutePoseUF0, gripper_roll_deg: float) -> AbsolutePoseUF0:
    roll = _finite(gripper_roll_deg, field="gripper_roll_deg", error_type=MotionInputError)
    return replace(pose, r=pose.r + roll)


def _short_yaw_poses(grasp: AbsolutePoseUF0, target: AbsolutePoseUF0) -> tuple[AbsolutePoseUF0, AbsolutePoseUF0]:
    grasp_r = (grasp.r + 180.0) % 360.0 - 180.0
    target_r = (target.r + 180.0) % 360.0 - 180.0
    delta = (target_r - grasp_r + 180.0) % 360.0 - 180.0
    return replace(grasp, r=grasp_r), replace(target, r=grasp_r + delta)


def _closest_yaw(reference_r: float, target: AbsolutePoseUF0) -> AbsolutePoseUF0:
    delta = (target.r - reference_r + 180.0) % 360.0 - 180.0
    return replace(target, r=reference_r + delta)


def _waypoint(pose: AbsolutePoseUF0, *, gripper: float, z_offset: float = 0.0) -> Waypoint:
    return Waypoint(pose.x, pose.y, pose.z + z_offset, pose.w, pose.p, pose.r, gripper)


def _steps(start: Waypoint, end: Waypoint, config: MotionConfig) -> int:
    linear = math.dist(start.xyz, end.xyz) / config.max_linear_step_mm
    yaw = abs(end.r - start.r) / config.max_yaw_step_deg
    return max(1, math.ceil(max(linear, yaw)))


def _segment(start: Waypoint, end: Waypoint, config: MotionConfig) -> tuple[Waypoint, ...]:
    count = _steps(start, end, config)
    fields = ("x", "y", "z", "w", "p", "r")
    points: list[Waypoint] = []
    for index in range(1, count + 1):
        fraction = index / count
        values = {
            field: getattr(start, field) + (getattr(end, field) - getattr(start, field)) * fraction for field in fields
        }
        points.append(Waypoint(**values, gripper=end.gripper))
    return tuple(points)


def _chunk(phase: MotionPhase, points: tuple[Waypoint, ...], size: int) -> tuple[MotionChunk, ...]:
    return tuple(MotionChunk(phase, points[index : index + size]) for index in range(0, len(points), size))


def _anchors(grasp: AbsolutePoseUF0, target: AbsolutePoseUF0, config: MotionConfig) -> dict[str, Waypoint]:
    aligned_lift = replace(grasp, r=target.r)
    return {
        "approach": _waypoint(grasp, gripper=0.0, z_offset=config.approach_mm),
        "grasp_open": _waypoint(grasp, gripper=0.0),
        "grasp_closed": _waypoint(grasp, gripper=1.0),
        "lift": _waypoint(grasp, gripper=1.0, z_offset=config.lift_mm),
        "aligned_lift": _waypoint(aligned_lift, gripper=1.0, z_offset=config.lift_mm),
        "hover": _waypoint(target, gripper=1.0, z_offset=config.hover_mm),
        "target_closed": _waypoint(target, gripper=1.0),
        "target_open": _waypoint(target, gripper=0.0),
        "retreat": _waypoint(target, gripper=0.0, z_offset=config.retreat_mm),
    }


def _phase_points(
    anchors: dict[str, Waypoint], config: MotionConfig
) -> tuple[tuple[MotionPhase, tuple[Waypoint, ...]], ...]:
    return (
        (MotionPhase.APPROACH, (anchors["approach"],)),
        (MotionPhase.DESCEND, _segment(anchors["approach"], anchors["grasp_open"], config)),
        (MotionPhase.GRIP, (anchors["grasp_closed"],)),
        (MotionPhase.LIFT, _segment(anchors["grasp_closed"], anchors["lift"], config)),
        (MotionPhase.CARRY_YAW, _segment(anchors["lift"], anchors["aligned_lift"], config)),
        (MotionPhase.HOVER, _segment(anchors["aligned_lift"], anchors["hover"], config)),
        (MotionPhase.VERTICAL_INSERT, _segment(anchors["hover"], anchors["target_closed"], config)),
        (MotionPhase.RELEASE, (anchors["target_open"],)),
        (MotionPhase.RETREAT, _segment(anchors["target_open"], anchors["retreat"], config)),
    )


def _pick_phase_points(
    current: Waypoint,
    anchors: dict[str, Waypoint],
    config: MotionConfig,
) -> tuple[tuple[MotionPhase, tuple[Waypoint, ...]], ...]:
    if current.gripper != 0.0:
        raise MotionInputError("a new pick must start with an open gripper")
    phases = _phase_points(anchors, config)
    approach = _segment(current, anchors["approach"], config)
    return ((MotionPhase.APPROACH, approach), *phases[1:])


def compile_pick_insert(
    current: Waypoint,
    grasp: OrientedPartPose,
    target: OrientedPartPose,
    *,
    gripper_roll_deg: float,
    config: MotionConfig,
) -> tuple[MotionChunk, ...]:
    """Compile calibrated absolute poses into bounded phase-labelled chunks."""
    _validate_request(grasp, target, config)
    grasp_tcp = _apply_gripper_roll(grasp.tcp, gripper_roll_deg)
    target_tcp = _apply_gripper_roll(target.tcp, gripper_roll_deg)
    target_tcp = _preserve_roll_pitch(grasp_tcp, target_tcp)
    grasp_tcp, target_tcp = _short_yaw_poses(grasp_tcp, target_tcp)
    anchors = _anchors(grasp_tcp, target_tcp, config)
    chunks: list[MotionChunk] = []
    for phase, points in _pick_phase_points(current, anchors, config):
        chunks.extend(_chunk(phase, points, config.max_waypoints_per_chunk))
    return tuple(chunks)


def compile_pick_to_insert(
    current: Waypoint,
    grasp: OrientedPartPose,
    target: OrientedPartPose,
    *,
    gripper_roll_deg: float,
    config: MotionConfig,
) -> tuple[MotionChunk, ...]:
    """Pick and insert while retaining the part for provisional verification."""
    chunks = compile_pick_insert(
        current,
        grasp,
        target,
        gripper_roll_deg=gripper_roll_deg,
        config=config,
    )
    return tuple(chunk for chunk in chunks if chunk.phase not in {MotionPhase.RELEASE, MotionPhase.RETREAT})


def compile_release_retreat(
    current: Waypoint,
    target: OrientedPartPose,
    *,
    gripper_roll_deg: float,
    config: MotionConfig,
) -> tuple[MotionChunk, ...]:
    """Release a provisionally verified insertion and retreat vertically."""
    _validate_pose("target", target)
    _validate_config(config)
    if current.gripper != 1.0:
        raise MotionInputError("release requires a provisionally held part")
    _apply_gripper_roll(target.tcp, gripper_roll_deg)
    start = AbsolutePoseUF0(current.x, current.y, current.z, current.w, current.p, current.r, "UF0", "calibrated")
    anchors = _anchors(start, start, config)
    phases = _phase_points(anchors, config)
    selected = {MotionPhase.RELEASE, MotionPhase.RETREAT}
    return tuple(
        chunk
        for phase, points in phases
        if phase in selected
        for chunk in _chunk(phase, points, config.max_waypoints_per_chunk)
    )


def compile_reinsert(
    current: Waypoint,
    target: OrientedPartPose,
    *,
    gripper_roll_deg: float,
    config: MotionConfig,
) -> tuple[MotionChunk, ...]:
    """Lift a still-held part, yaw-align, and reinsert without releasing it."""
    _validate_pose("target", target)
    _validate_config(config)
    if current.gripper != 1.0:
        raise MotionInputError("reinsert requires the gripper to remain closed")
    target_tcp = _apply_gripper_roll(target.tcp, gripper_roll_deg)
    start = AbsolutePoseUF0(current.x, current.y, current.z, current.w, current.p, current.r, "UF0", "calibrated")
    _validate_pose("current", OrientedPartPose(start, RestingState.INSTALLED, target.up_face))
    target_tcp = _preserve_roll_pitch(start, target_tcp)
    start, target_tcp = _short_yaw_poses(start, target_tcp)
    anchors = _anchors(start, target_tcp, config)
    selected = {MotionPhase.LIFT, MotionPhase.CARRY_YAW, MotionPhase.HOVER, MotionPhase.VERTICAL_INSERT}
    phases = _phase_points(anchors, config)
    return tuple(
        chunk
        for phase, points in phases
        if phase in selected
        for chunk in _chunk(phase, points, config.max_waypoints_per_chunk)
    )


def compile_transfer_held(
    current: Waypoint,
    target: OrientedPartPose,
    *,
    gripper_roll_deg: float,
    config: MotionConfig,
) -> tuple[MotionChunk, ...]:
    """Move a still-held rejected part to a calibrated drop target."""
    reinsert = compile_reinsert(
        current,
        target,
        gripper_roll_deg=gripper_roll_deg,
        config=config,
    )
    release = compile_release_retreat(
        reinsert[-1].waypoints[-1],
        target,
        gripper_roll_deg=gripper_roll_deg,
        config=config,
    )
    return (*reinsert, *release)
