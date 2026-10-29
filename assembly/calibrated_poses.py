"""Fail-closed conversion of measured poses into recipe motion targets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Number

from assembly.catalog import Catalog, ProductRecipe, RecipeStep, UnknownBaseError
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


class MeasuredRecipePoses:
    """Implement the planner pose protocol from measured poses and CAD offsets."""

    def __init__(
        self,
        *,
        catalog: Catalog,
        measurements: MeasuredPoseRegistry,
        installed_orientations: Mapping[str, ApprovedInstalledOrientation],
        orientation_tolerance_deg: float,
    ) -> None:
        self._catalog = catalog
        self._measurements = measurements
        self._orientations = dict(installed_orientations)
        self._orientation_tolerance = _validate_tolerance(orientation_tolerance_deg)

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
        return self._target(base_pose.tcp, step.insert_offset_mm, approved)

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
        offset: tuple[float, float, float],
        approved: ApprovedInstalledOrientation,
    ) -> OrientedPartPose:
        dx, dy, dz = (_finite(value, "insert_offset_mm") for value in offset)
        radians = math.radians(base.r)
        x = base.x + math.cos(radians) * dx - math.sin(radians) * dy
        y = base.y + math.sin(radians) * dx + math.cos(radians) * dy
        yaw = (base.r + approved.yaw_offset_deg + 180.0) % 360.0 - 180.0
        tcp = AbsolutePoseUF0(x, y, base.z + dz, approved.w_deg, approved.p_deg, yaw, "UF0", "calibrated")
        return OrientedPartPose(tcp, RestingState.INSTALLED, approved.up_face)
