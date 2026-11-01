"""Fail-closed conversion of measured poses into recipe motion targets.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from numbers import Number
from pathlib import Path

from assembly.catalog import Catalog, ProductRecipe, RecipeStep, UnknownBaseError
from assembly.grasp_source import PreferredGrasp
from assembly.inventory import InventoryItem
from assembly.motion import AbsolutePoseUF0, OrientedPartPose, RestingState


class CalibratedPoseError(ValueError):
    """A required measurement or approved installed orientation is unsafe."""


@dataclass(frozen=True, slots=True)
class ApprovedInstalledOrientation:
    """Reviewed tool orientation for one part in its installed pose."""

    w_deg: float
    p_deg: float
    yaw_offset_deg: float
    up_face: str


@dataclass(frozen=True, slots=True)
class FoundationPoseFrameBoundary:
    """Recorded guarantee that raw STL poses were normalized before yaw-only use."""

    frame: str
    source: str

    def __post_init__(self) -> None:
        if self.frame != "installed_local":
            raise CalibratedPoseError("FoundationPose frame must be canonical installed_local")
        if not isinstance(self.source, str) or not self.source.strip():
            raise CalibratedPoseError("FoundationPose installed_local frame requires a recorded source")


Matrix4 = tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True, slots=True)
class SourcedGraspFrame:
    """One grasp-frame mesh transform and its authorized turn/TCP pair."""

    object_id: str
    local_from_stl: Matrix4
    preferred_grasp: PreferredGrasp
    source: str


class GraspFrameRegistry:
    """Immutable, hash-verified grasp frames for catalog parts."""

    def __init__(self, frames: Mapping[str, SourcedGraspFrame]) -> None:
        if not isinstance(frames, Mapping) or not frames:
            raise CalibratedPoseError("grasp frame registry must be a non-empty mapping")
        self._frames = dict(frames)

    @classmethod
    def from_catalog(cls, catalog: Catalog) -> GraspFrameRegistry:
        source = catalog.grasp_source
        path = Path(__file__).resolve().parents[1] / source.snapshot
        try:
            contents = path.read_bytes()
        except OSError as error:
            raise CalibratedPoseError(f"cannot read pinned grasp source {path}: {error}") from error
        if sha256(contents).hexdigest() != source.sha256:
            raise CalibratedPoseError("pinned grasp source sha256 mismatch")
        try:
            document = json.loads(contents)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise CalibratedPoseError("pinned grasp source is not valid JSON") from error
        return cls(_frames_from_document(catalog, document))

    def frame_for(self, object_id: str) -> SourcedGraspFrame:
        try:
            return self._frames[object_id]
        except KeyError as error:
            raise CalibratedPoseError(f"sourced grasp frame for {object_id!r} is missing") from error


class MeasuredPoseRegistry:
    """Replaceable snapshot of perception-calibrated absolute UF0 poses."""

    def __init__(
        self,
        poses: Mapping[str, OrientedPartPose],
        *,
        observed_at_s: float,
        complete: bool,
    ) -> None:
        self.replace_snapshot(poses, observed_at_s=observed_at_s, complete=complete)

    def replace_snapshot(
        self,
        poses: Mapping[str, OrientedPartPose],
        *,
        observed_at_s: float,
        complete: bool,
    ) -> None:
        if not isinstance(poses, Mapping):
            raise CalibratedPoseError("measured pose snapshot must be a mapping")
        observed_at = _finite(observed_at_s, "measured pose snapshot observed_at_s")
        if not isinstance(complete, bool):
            raise CalibratedPoseError("measured pose snapshot complete must be a boolean")
        self._poses = dict(poses)
        self._observed_at_s = observed_at
        self._complete = complete

    def pose_for(self, instance_id: str, *, expected_at_s: float) -> OrientedPartPose:
        self.require_current(expected_at_s)
        try:
            return self._poses[instance_id]
        except KeyError as error:
            raise CalibratedPoseError(f"measurement for {instance_id!r} is missing") from error

    def require_current(self, expected_at_s: float) -> None:
        expected = _finite(expected_at_s, "expected pose observation time")
        if not self._complete:
            raise CalibratedPoseError("measured pose snapshot must be explicitly complete")
        if self._observed_at_s != expected:
            raise CalibratedPoseError("measured pose snapshot does not belong to the current observation")


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Number):
        raise CalibratedPoseError(f"{field} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise CalibratedPoseError(f"{field} must be finite")
    return number


def _matrix4(value: object, field: str) -> Matrix4:
    if not isinstance(value, list) or len(value) != 4:
        raise CalibratedPoseError(f"{field} must be a 4x4 rigid transform")
    rows = tuple(_matrix_row(row, field) for row in value)
    if any(abs(actual - expected) > 1e-6 for actual, expected in zip(rows[3], (0.0, 0.0, 0.0, 1.0))):
        raise CalibratedPoseError(f"{field} must be a homogeneous rigid transform")
    rotation = tuple(row[:3] for row in rows[:3])
    if not _proper_rotation(rotation):
        raise CalibratedPoseError(f"{field} rotation must be orthonormal with determinant +1")
    return rows


def _matrix_row(value: object, field: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise CalibratedPoseError(f"{field} must be a 4x4 rigid transform")
    return tuple(_finite(item, field) for item in value)  # type: ignore[return-value]


def _proper_rotation(rotation: tuple[tuple[float, ...], ...]) -> bool:
    for left in range(3):
        for right in range(3):
            dot = sum(rotation[left][axis] * rotation[right][axis] for axis in range(3))
            if abs(dot - (1.0 if left == right else 0.0)) > 1e-6:
                return False
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    return abs(determinant - 1.0) <= 1e-6


def _frames_from_document(catalog: Catalog, document: object) -> dict[str, SourcedGraspFrame]:
    if not isinstance(document, dict) or not isinstance(document.get("objects"), dict):
        raise CalibratedPoseError("pinned grasp source has no objects mapping")
    objects = document["objects"]
    source = f"{catalog.grasp_source.file}@{catalog.grasp_source.commit}"
    return {
        step.object_id: _frame_from_entry(step, objects.get(step.object_id), source)
        for product in catalog.products
        for step in product.steps
    }


def _frame_from_entry(step: RecipeStep, entry: object, source: str) -> SourcedGraspFrame:
    if not isinstance(entry, dict):
        raise CalibratedPoseError(f"pinned grasp entry for {step.object_id!r} is missing")
    return SourcedGraspFrame(
        step.object_id,
        _matrix4(entry.get("T_local_from_stl"), f"{step.object_id}.T_local_from_stl"),
        step.preferred_grasp,
        source,
    )


def _angle_error(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _validate_pose(label: str, pose: object) -> OrientedPartPose:
    if not isinstance(pose, OrientedPartPose):
        raise CalibratedPoseError(f"{label} must be an OrientedPartPose")
    if not isinstance(pose.tcp, AbsolutePoseUF0):
        raise CalibratedPoseError(f"{label} TCP must be an AbsolutePoseUF0")
    if pose.resting_state is not RestingState.INSTALLED:
        raise CalibratedPoseError(f"{label} must be in its installed orientation")
    if pose.tcp.frame != "UF0":
        raise CalibratedPoseError(f"{label} must be absolute UF0")
    if pose.tcp.source != "calibrated":
        raise CalibratedPoseError(f"{label} source must be calibrated")
    for field in ("x", "y", "z", "w", "p", "r"):
        _finite(getattr(pose.tcp, field), f"{label}.{field}")
    if not isinstance(pose.up_face, str) or not pose.up_face.strip():
        raise CalibratedPoseError(f"{label} up-face must be identified")
    return pose


def _validate_rule(object_id: str, rule: object) -> ApprovedInstalledOrientation:
    if not isinstance(rule, ApprovedInstalledOrientation):
        raise CalibratedPoseError(f"approved orientation for {object_id} is missing")
    for field in ("w_deg", "p_deg", "yaw_offset_deg"):
        _finite(getattr(rule, field), f"{object_id}.{field}")
    if not isinstance(rule.up_face, str) or not rule.up_face.strip():
        raise CalibratedPoseError(f"{object_id}.up_face must be identified")
    return rule


def _validate_tolerance(value: object) -> float:
    tolerance = _finite(value, "orientation_tolerance_deg")
    if not 0.0 <= tolerance < 90.0:
        raise CalibratedPoseError("orientation_tolerance_deg must be in [0, 90)")
    return tolerance


def _rigid_inverse(matrix: Matrix4) -> Matrix4:
    rotation = tuple(tuple(matrix[row][column] for row in range(3)) for column in range(3))
    translation = tuple(matrix[row][3] for row in range(3))
    inverse_translation = tuple(-sum(rotation[row][axis] * translation[axis] for axis in range(3)) for row in range(3))
    return tuple(
        (*rotation[row], inverse_translation[row]) if row < 3 else (0.0, 0.0, 0.0, 1.0)
        for row in range(4)
    )  # type: ignore[return-value]


def _matmul(left: Matrix4, right: Matrix4) -> Matrix4:
    return tuple(
        tuple(sum(left[row][axis] * right[axis][column] for axis in range(4)) for column in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


def _pose_matrix(pose: AbsolutePoseUF0) -> Matrix4:
    angle = math.radians(pose.r)
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        (cosine, -sine, 0.0, pose.x),
        (sine, cosine, 0.0, pose.y),
        (0.0, 0.0, 1.0, pose.z),
        (0.0, 0.0, 0.0, 1.0),
    )


def _step_matrix(step: RecipeStep) -> Matrix4:
    dx, dy, dz = (_finite(value, "insert_offset_mm") for value in step.insert_offset_mm)
    rotation = step.installed_rotation_matrix
    return tuple(
        (*rotation[row], (dx, dy, dz)[row]) if row < 3 else (0.0, 0.0, 0.0, 1.0)
        for row in range(4)
    )  # type: ignore[return-value]


def _transform_point(matrix: Matrix4, point: tuple[float, float, float]) -> tuple[float, float, float]:
    homogeneous = (*point, 1.0)
    return tuple(sum(matrix[row][axis] * homogeneous[axis] for axis in range(4)) for row in range(3))  # type: ignore[return-value]


def _yaw_from_matrix(matrix: Matrix4) -> float:
    x_axis_xy = math.hypot(matrix[0][0], matrix[1][0])
    if x_axis_xy < 1e-6:
        raise CalibratedPoseError("installed part frame has no well-defined table yaw")
    return (math.degrees(math.atan2(matrix[1][0], matrix[0][0])) + 180.0) % 360.0 - 180.0


def _detection_pose(detection: Mapping[str, object]) -> tuple[float, float, float, float]:
    pose = detection.get("pose")
    if not isinstance(pose, Mapping):
        raise CalibratedPoseError("FoundationPose detection must contain an absolute UF0 pose")
    return tuple(_finite(pose.get(field), f"FoundationPose pose.{field}") for field in ("x", "y", "z", "yaw"))  # type: ignore[return-value]


class MeasuredRecipePoses:
    """Implement the planner pose protocol from measured poses and CAD offsets."""

    def __init__(
        self,
        *,
        catalog: Catalog,
        measurements: MeasuredPoseRegistry,
        installed_orientations: Mapping[str, ApprovedInstalledOrientation],
        orientation_tolerance_deg: float,
        foundationpose_boundary: FoundationPoseFrameBoundary,
        grasp_frames: GraspFrameRegistry | None = None,
    ) -> None:
        self._catalog = catalog
        self._measurements = measurements
        self._orientations = dict(installed_orientations)
        self._orientation_tolerance = _validate_tolerance(orientation_tolerance_deg)
        if not isinstance(foundationpose_boundary, FoundationPoseFrameBoundary):
            raise CalibratedPoseError("FoundationPose installed_local frame boundary is required")
        self._foundationpose_boundary = foundationpose_boundary
        self._grasp_frames = grasp_frames or GraspFrameRegistry.from_catalog(catalog)

    def grasp_for(self, part: InventoryItem, *, observed_at_s: float) -> OrientedPartPose:
        measured = _validate_pose(
            "grasp measurement",
            self._measurements.pose_for(part.instance_id, expected_at_s=observed_at_s),
        )
        approved = self._approved(part.object_id)
        if measured.up_face != approved.up_face:
            raise CalibratedPoseError(f"{part.instance_id!r} measured up-face does not match approved up-face")
        self._check_roll_pitch(part.instance_id, measured.tcp, approved)
        return measured

    def grasp_from_foundationpose(
        self,
        part: InventoryItem,
        detection: dict[str, object],
        *,
        observed_at_s: float,
    ) -> OrientedPartPose:
        """Convert a canonical installed-local object pose to its fingertip TCP."""
        _finite(observed_at_s, "FoundationPose observation time")
        if not isinstance(detection, Mapping):
            raise CalibratedPoseError("FoundationPose detection must be a mapping")
        if detection.get("instance_id") != part.instance_id or detection.get("object_id") != part.object_id:
            raise CalibratedPoseError("FoundationPose identity does not match the reserved part")
        if detection.get("pose_source") != "foundationpose":
            raise CalibratedPoseError("grasp pose must come from FoundationPose")
        if detection.get("resting_side") != "upright":
            raise CalibratedPoseError("fallen part requires a sourced full 6D grasp")
        x, y, z, yaw = _detection_pose(detection)
        frame = self._grasp_frames.frame_for(part.object_id)
        approved = self._approved(part.object_id)
        world_from_local = _pose_matrix(AbsolutePoseUF0(x, y, z, 0.0, 0.0, yaw, "UF0", "calibrated"))
        tcp_x, tcp_y, tcp_z = _transform_point(world_from_local, frame.preferred_grasp.tcp_local_mm)
        tcp = AbsolutePoseUF0(tcp_x, tcp_y, tcp_z, approved.w_deg, approved.p_deg, yaw, "UF0", "calibrated")
        return OrientedPartPose(tcp, RestingState.INSTALLED, approved.up_face)

    def target_for(
        self,
        product: ProductRecipe,
        base: InventoryItem,
        step: RecipeStep,
        *,
        observed_at_s: float,
    ) -> OrientedPartPose:
        self._validate_recipe_request(product, base, step)
        base_pose = self.base_for(base, observed_at_s=observed_at_s)
        approved = self._approved(step.object_id)
        frame = self._grasp_frames.frame_for(step.object_id)
        return self._target(base_pose.tcp, step, approved, frame)

    def verification_target_for(
        self,
        product: ProductRecipe,
        base: InventoryItem,
        step: RecipeStep,
        *,
        observed_at_s: float,
    ) -> OrientedPartPose:
        """Return the canonical part origin expected from camera pose estimation."""
        self._validate_recipe_request(product, base, step)
        base_pose = self.base_for(base, observed_at_s=observed_at_s)
        approved = self._approved(step.object_id)
        frame = self._grasp_frames.frame_for(step.object_id)
        world_from_local = self._world_from_local(base_pose.tcp, step, approved, frame)
        x, y, z = _transform_point(world_from_local, (0.0, 0.0, 0.0))
        yaw = _yaw_from_matrix(world_from_local)
        pose = AbsolutePoseUF0(x, y, z, approved.w_deg, approved.p_deg, yaw, "UF0", "calibrated")
        return OrientedPartPose(pose, RestingState.INSTALLED, approved.up_face)

    def base_for(self, base: InventoryItem, *, observed_at_s: float) -> OrientedPartPose:
        """Return the current measured base pose used to derive insertion targets."""
        return _validate_pose(
            "base measurement",
            self._measurements.pose_for(base.instance_id, expected_at_s=observed_at_s),
        )

    def _approved(self, object_id: str) -> ApprovedInstalledOrientation:
        return _validate_rule(object_id, self._orientations.get(object_id))

    def _check_roll_pitch(
        self,
        instance_id: str,
        pose: AbsolutePoseUF0,
        approved: ApprovedInstalledOrientation,
    ) -> None:
        tolerance = self._orientation_tolerance
        if _angle_error(pose.w, approved.w_deg) > tolerance or _angle_error(pose.p, approved.p_deg) > tolerance:
            raise CalibratedPoseError(f"{instance_id!r} measured roll/pitch does not match installed orientation")

    def _validate_recipe_request(
        self,
        product: ProductRecipe,
        base: InventoryItem,
        step: RecipeStep,
    ) -> None:
        try:
            base_type = self._catalog.base_type_for_object(base.object_id)
        except UnknownBaseError as error:
            raise CalibratedPoseError(f"unknown base {base.object_id!r}") from error
        if base_type != product.base_type:
            raise CalibratedPoseError(f"base type {base_type} is incompatible with product {product.product}")
        if step not in product.steps:
            raise CalibratedPoseError(f"recipe step {step.object_id!r} does not belong to product {product.product}")

    @staticmethod
    def _target(
        base: AbsolutePoseUF0,
        step: RecipeStep,
        approved: ApprovedInstalledOrientation,
        frame: SourcedGraspFrame,
    ) -> OrientedPartPose:
        world_from_local = MeasuredRecipePoses._world_from_local(base, step, approved, frame)
        x, y, z = _transform_point(world_from_local, frame.preferred_grasp.tcp_local_mm)
        yaw = _yaw_from_matrix(world_from_local)
        tcp = AbsolutePoseUF0(x, y, z, approved.w_deg, approved.p_deg, yaw, "UF0", "calibrated")
        return OrientedPartPose(tcp, RestingState.INSTALLED, approved.up_face)

    @staticmethod
    def _world_from_local(
        base: AbsolutePoseUF0,
        step: RecipeStep,
        approved: ApprovedInstalledOrientation,
        frame: SourcedGraspFrame,
    ) -> Matrix4:
        if abs(approved.yaw_offset_deg) > 1e-9:
            raise CalibratedPoseError("catalog transform owns installed yaw; approved yaw offset must be zero")
        world_from_stl = _matmul(_pose_matrix(base), _step_matrix(step))
        return _matmul(world_from_stl, _rigid_inverse(frame.local_from_stl))
