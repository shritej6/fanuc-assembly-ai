"""Deterministic motion compilation from calibrated absolute UF0 TCP poses."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from numbers import Number

from robot.adapter import MAX_ACTION_CHUNK, TCP_LIMITS


class MotionError(ValueError):
    """Base error for requests that cannot safely become motion."""


class MotionInputError(MotionError):
    """A pose is untrusted, fallen, nonfinite, or needs reorientation."""


class MotionConfigError(MotionError):
    """Caller-supplied motion limits are invalid."""


class MotionPhase(StrEnum):
    """Required pick-and-insert phase order."""

    INSPECT = "inspect"
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
    safe_z_mm: float
    safe_z_source: str


@dataclass(frozen=True)
class InspectionStopConfig:
    """Measured wrist inspection pose; values and provenance are deployment inputs."""

    tcp_z_mm: float
    tool_w_deg: float
    tool_p_deg: float
    source: str


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


def _validate_pose(
    label: str,
    part_pose: object,
    *,
    allowed_states: frozenset[RestingState] = frozenset({RestingState.INSTALLED}),
) -> None:
    if not isinstance(part_pose, OrientedPartPose):
        raise MotionInputError(f"{label} pose must be an OrientedPartPose")
    pose = part_pose.tcp
    if not isinstance(pose, AbsolutePoseUF0):
        raise MotionInputError(f"{label} TCP must be an AbsolutePoseUF0")
    if part_pose.resting_state not in allowed_states:
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
    _finite(config.safe_z_mm, field="safe_z_mm", error_type=MotionConfigError)
    if not isinstance(config.safe_z_source, str) or not config.safe_z_source.strip():
        raise MotionConfigError("safe_z_mm requires a recorded source")
    for field in ("approach_mm", "lift_mm", "hover_mm", "retreat_mm"):
        if _finite(getattr(config, field), field=field, error_type=MotionConfigError) < 0.0:
            raise MotionConfigError(f"motion clearance {field} must be nonnegative")
    for field in ("max_linear_step_mm", "max_yaw_step_deg"):
        if _finite(getattr(config, field), field=field, error_type=MotionConfigError) <= 0.0:
            raise MotionConfigError(f"{field} must be finite and positive")
    size = config.max_waypoints_per_chunk
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_ACTION_CHUNK:
        raise MotionConfigError(f"max_waypoints_per_chunk must be an integer in 1..{MAX_ACTION_CHUNK}")


def _validate_inspection_config(config: InspectionStopConfig) -> None:
    for field in ("tcp_z_mm", "tool_w_deg", "tool_p_deg"):
        _finite(getattr(config, field), field=field, error_type=MotionConfigError)
    if abs(abs(config.tool_w_deg) - 180.0) > 1e-9 or abs(config.tool_p_deg) > 1e-9:
        raise MotionConfigError("inspection stop must keep UTOOL1 straight down")
    if not isinstance(config.source, str) or not config.source.strip():
        raise MotionConfigError("inspection stop requires a recorded measurement source")


def _validate_request(grasp: OrientedPartPose, target: OrientedPartPose, config: MotionConfig) -> None:
    _validate_pose("grasp", grasp)
    _validate_pose("target", target)
    _validate_config(config)
    if grasp.up_face != target.up_face:
        raise MotionInputError("grasp and target installed up-face must match")
    _validate_safe_z(grasp, target, config)


def _validate_safe_z(grasp: OrientedPartPose, target: OrientedPartPose, config: MotionConfig) -> None:
    required = max(grasp.tcp.z + config.lift_mm, target.tcp.z + config.hover_mm)
    if config.safe_z_mm < required:
        raise MotionConfigError("safe_z_mm must be on or above all pick-transfer clearances")


def _preserve_roll_pitch(reference: AbsolutePoseUF0, target: AbsolutePoseUF0) -> AbsolutePoseUF0:
    """Keep the observed installed up-face; only target yaw may change."""
    return replace(target, w=reference.w, p=reference.p)


def _apply_gripper_roll(pose: AbsolutePoseUF0, gripper_roll_deg: float) -> AbsolutePoseUF0:
    roll = _finite(gripper_roll_deg, field="gripper_roll_deg", error_type=MotionInputError)
    return replace(pose, r=pose.r + roll)


def _validate_release_endpoint(
    current: Waypoint,
    expected: AbsolutePoseUF0,
    *,
    max_position_error_mm: float,
    max_orientation_error_deg: float,
) -> None:
    position_limit = _finite(
        max_position_error_mm,
        field="max_position_error_mm",
        error_type=MotionInputError,
    )
    orientation_limit = _finite(
        max_orientation_error_deg,
        field="max_orientation_error_deg",
        error_type=MotionInputError,
    )
    if position_limit < 0.0 or orientation_limit < 0.0:
        raise MotionInputError("release endpoint tolerances must be nonnegative")
    position_error = math.dist(current.xyz, (expected.x, expected.y, expected.z))
    angle_errors = (
        abs((actual - target + 180.0) % 360.0 - 180.0)
        for actual, target in zip((current.w, current.p, current.r), (expected.w, expected.p, expected.r))
    )
    if position_error > position_limit or max(angle_errors) > orientation_limit:
        raise MotionInputError("release requires the verified fingertip TCP endpoint")


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


def _nearest_angle(reference: float, target: float) -> float:
    delta = (target - reference + 180.0) % 360.0 - 180.0
    return reference if abs(delta) < 1e-12 else reference + delta


def _bounded_angle_pair(reference: float, target: float, field: str) -> tuple[float, float]:
    low, high, _unit = TCP_LIMITS[{"w": 3, "p": 4, "r": 5}[field]]
    if not low <= reference <= high:
        raise MotionInputError(f"start {field} is outside official TCP bounds")
    nearest = _nearest_angle(reference, target)
    if low <= nearest <= high:
        return reference, nearest
    shift = -360.0 if nearest > high else 360.0
    shifted = (reference + shift, nearest + shift)
    if all(low <= angle <= high for angle in shifted):
        return shifted
    raise MotionInputError(f"shortest {field} path cannot stay inside official TCP bounds")


def _bounded_segment_angles(start: Waypoint, end: Waypoint) -> tuple[Waypoint, Waypoint]:
    pairs = {field: _bounded_angle_pair(getattr(start, field), getattr(end, field), field) for field in "wpr"}
    bounded_start = replace(start, **{field: pair[0] for field, pair in pairs.items()})
    bounded_end = replace(end, **{field: pair[1] for field, pair in pairs.items()})
    return bounded_start, bounded_end


def _segment(start: Waypoint, end: Waypoint, config: MotionConfig) -> tuple[Waypoint, ...]:
    start, end = _bounded_segment_angles(start, end)
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


def compile_inspection_stop(
    current: Waypoint,
    coarse_xy_mm: tuple[float, float],
    *,
    inspection: InspectionStopConfig,
    motion: MotionConfig,
) -> tuple[MotionChunk, ...]:
    """Move to the measured wrist viewpoint using only coarse table XY."""
    _validate_config(motion)
    _validate_inspection_config(inspection)
    if not isinstance(coarse_xy_mm, tuple) or len(coarse_xy_mm) != 2:
        raise MotionInputError("coarse perception center must be an x/y tuple")
    x = _finite(coarse_xy_mm[0], field="coarse_x_mm", error_type=MotionInputError)
    y = _finite(coarse_xy_mm[1], field="coarse_y_mm", error_type=MotionInputError)
    endpoint = Waypoint(
        x,
        y,
        inspection.tcp_z_mm,
        inspection.tool_w_deg,
        inspection.tool_p_deg,
        current.r,
        0.0,
    )
    transit_z = max(current.z, motion.safe_z_mm)
    rise = _segment(current, replace(current, z=transit_z), motion)
    transit_target = replace(endpoint, z=transit_z)
    transit = _segment(rise[-1], transit_target, motion)
    descend = _segment(transit[-1], endpoint, motion)
    return _chunk(MotionPhase.INSPECT, (*rise, *transit, *descend), motion.max_waypoints_per_chunk)


def _anchors(grasp: AbsolutePoseUF0, target: AbsolutePoseUF0, config: MotionConfig) -> dict[str, Waypoint]:
    aligned_safe = replace(grasp, z=config.safe_z_mm, r=target.r)
    return {
        "approach": _waypoint(grasp, gripper=0.0, z_offset=config.approach_mm),
        "grasp_open": _waypoint(grasp, gripper=0.0),
        "grasp_closed": _waypoint(grasp, gripper=1.0),
        "lift": _waypoint(grasp, gripper=1.0, z_offset=config.lift_mm),
        "safe_pick": _waypoint(replace(grasp, z=config.safe_z_mm), gripper=1.0),
        "aligned_safe": _waypoint(aligned_safe, gripper=1.0),
        "safe_target": _waypoint(replace(target, z=config.safe_z_mm), gripper=1.0),
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
        (
            MotionPhase.LIFT,
            (*_segment(anchors["grasp_closed"], anchors["lift"], config), *_segment(anchors["lift"], anchors["safe_pick"], config)),
        ),
        (MotionPhase.CARRY_YAW, _segment(anchors["safe_pick"], anchors["aligned_safe"], config)),
        (
            MotionPhase.HOVER,
            (*_segment(anchors["aligned_safe"], anchors["safe_target"], config), *_segment(anchors["safe_target"], anchors["hover"], config)),
        ),
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
    transit_z = max(current.z, config.safe_z_mm)
    current_safe = replace(current, z=transit_z)
    approach_safe = replace(anchors["approach"], z=transit_z)
    rise = _segment(current, current_safe, config)
    transit = _segment(rise[-1], approach_safe, config)
    descend = _segment(transit[-1], anchors["approach"], config)
    approach = (*rise, *transit, *descend)
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


def compile_pick_to_reject(
    current: Waypoint,
    grasp: OrientedPartPose,
    reject: OrientedPartPose,
    *,
    gripper_roll_deg: float,
    config: MotionConfig,
) -> tuple[MotionChunk, ...]:
    """Pick an approved resting side without reorientation and drop it in reject."""
    _validate_pose("grasp", grasp, allowed_states=frozenset(RestingState))
    _validate_pose("reject", reject)
    _validate_config(config)
    _validate_safe_z(grasp, reject, config)
    grasp_tcp = _apply_gripper_roll(grasp.tcp, gripper_roll_deg)
    reject_tcp = _apply_gripper_roll(reject.tcp, gripper_roll_deg)
    reject_tcp = _preserve_roll_pitch(grasp_tcp, reject_tcp)
    grasp_tcp, reject_tcp = _short_yaw_poses(grasp_tcp, reject_tcp)
    anchors = _anchors(grasp_tcp, reject_tcp, config)
    chunks: list[MotionChunk] = []
    for phase, points in _pick_phase_points(current, anchors, config):
        chunks.extend(_chunk(phase, points, config.max_waypoints_per_chunk))
    return tuple(chunks)


def compile_release_retreat(
    current: Waypoint,
    target: OrientedPartPose,
    *,
    gripper_roll_deg: float,
    config: MotionConfig,
    max_position_error_mm: float,
    max_orientation_error_deg: float,
) -> tuple[MotionChunk, ...]:
    """Release a provisionally verified insertion and retreat vertically."""
    _validate_pose("target", target)
    _validate_config(config)
    if current.gripper != 1.0:
        raise MotionInputError("release requires a provisionally held part")
    expected = _apply_gripper_roll(target.tcp, gripper_roll_deg)
    _validate_release_endpoint(
        current,
        expected,
        max_position_error_mm=max_position_error_mm,
        max_orientation_error_deg=max_orientation_error_deg,
    )
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
    max_position_error_mm: float,
    max_orientation_error_deg: float,
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
        max_position_error_mm=max_position_error_mm,
        max_orientation_error_deg=max_orientation_error_deg,
    )
    return (*reinsert, *release)
