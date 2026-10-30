"""Semantic motion requests shared by T09 and learned real-cell executors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from numbers import Number

from assembly.catalog import TRUSTED_GRASP_SOURCE, load_catalog
from assembly.grasp_source import PreferredGrasp


class MotionIntentError(ValueError):
    """A semantic request is incomplete or would weaken a controller invariant."""


class SemanticSkill(StrEnum):
    """Physical operations whose trajectories are owned by the selected executor."""

    INSPECT_VIEW = "inspect_view"
    PICK_INSERT_HOLD = "pick_insert_hold"
    RELEASE_RETREAT = "release_retreat"
    PULL_OUT = "pull_out"
    PLACE_FREE = "place_free"
    DEFECT_DROP = "defect_drop"


def _rounded(value: float) -> str:
    return str(round(float(value)))


def semantic_instruction(intent: MotionIntent) -> str:
    """Stable language contract pinned by the deployment dataset manifest."""
    part = ", ".join(_rounded(value) for value in intent.object_pose)
    goal = ", ".join(_rounded(value) for value in intent.goal_pose)
    grip = f"grip {intent.grip_index} at roll {_rounded(intent.wrist_roll_deg)}"
    if intent.skill is SemanticSkill.INSPECT_VIEW:
        text = f"inspect {intent.object_id} at ({part}) from verification view ({goal})"
    elif intent.skill is SemanticSkill.PICK_INSERT_HOLD:
        text = (
            f"pick {intent.object_id} at ({part}) with {grip}, insert on {intent.base_object_id} "
            f"at ({goal}), keep holding"
        )
    elif intent.skill is SemanticSkill.RELEASE_RETREAT:
        text = f"release {intent.object_id} on {intent.base_object_id}, retreat to ({goal})"
    elif intent.skill is SemanticSkill.PULL_OUT:
        text = f"pull out {intent.object_id} at ({part}) with {grip}, set down at ({goal})"
    elif intent.skill is SemanticSkill.PLACE_FREE:
        text = f"pick {intent.object_id} at ({part}) with {grip}, set down at ({goal})"
    else:
        text = f"pick {intent.object_id} at ({part}) with {grip}, drop in defect area at ({goal})"
    return text if intent.case == "nominal" else f"{text} [{intent.case}]"


def _finite_tuple(value: object, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, tuple) or len(value) != size:
        raise MotionIntentError(f"{label} must contain exactly {size} values")
    if any(isinstance(item, bool) or not isinstance(item, Number) for item in value):
        raise MotionIntentError(f"{label} must contain numeric values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise MotionIntentError(f"{label} must contain finite values")
    return result


def _angle_error(actual: float, expected: float) -> float:
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


@lru_cache(maxsize=1)
def _reviewed_grasps() -> dict[str, PreferredGrasp]:
    catalog = load_catalog()
    return {step.object_id: step.preferred_grasp for product in catalog.products for step in product.steps}


@dataclass(frozen=True, slots=True)
class ContactEnvelope:
    """Sourced TCP contact target used by the learned-policy floor guard."""

    x_mm: float
    y_mm: float
    z_contact_mm: float
    radius_mm: float
    label: str
    source: str

    def __post_init__(self) -> None:
        _finite_tuple((self.x_mm, self.y_mm, self.z_contact_mm, self.radius_mm), 4, "contact target")
        if self.radius_mm <= 0.0:
            raise MotionIntentError("contact target radius must be positive")
        if not isinstance(self.label, str) or not self.label.strip():
            raise MotionIntentError("contact target requires a label")
        if not isinstance(self.source, str) or not self.source.strip():
            raise MotionIntentError("contact target requires a recorded source")


@dataclass(frozen=True, slots=True)
class MotionIntent:
    """One identity-bound operation; coordinates never conflate part origin and fingertip TCP."""

    skill: SemanticSkill
    instance_id: str
    object_id: str
    base_instance_id: str | None
    base_object_id: str | None
    grip_index: int
    case: str
    object_pose: tuple[float, float, float]
    object_origin_z_mm: float
    wrist_roll_deg: float
    goal_pose: tuple[float, float, float]
    grasp_tcp: tuple[float, float, float, float, float, float]
    release_tcp: tuple[float, float, float, float, float, float] | None
    endpoint_tcp: tuple[float, float, float, float, float, float]
    final_gripper: float
    pose_source: str
    grasp_source_sha256: str
    contact: ContactEnvelope | None
    release_authorized: bool

    def __post_init__(self) -> None:
        self._validate_identity()
        self._validate_grasp()
        self._validate_release()
        self._validate_pick_insert()

    def _validate_identity(self) -> None:
        if not isinstance(self.skill, SemanticSkill):
            raise MotionIntentError("skill must be a SemanticSkill")
        for label, value in (("instance_id", self.instance_id), ("object_id", self.object_id), ("case", self.case)):
            if not isinstance(value, str) or not value.strip():
                raise MotionIntentError(f"{label} must be non-empty")
        for label, value in (("base_instance_id", self.base_instance_id), ("base_object_id", self.base_object_id)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise MotionIntentError(f"{label} must be None or non-empty")
        if (self.base_instance_id is None) is not (self.base_object_id is None):
            raise MotionIntentError("base instance and object identifiers must be supplied together")
        _finite_tuple(self.object_pose, 3, "object_pose")
        _finite_tuple((self.object_origin_z_mm,), 1, "object_origin_z_mm")
        _finite_tuple(self.goal_pose, 3, "goal_pose")
        _finite_tuple(self.grasp_tcp, 6, "grasp_tcp")
        if self.release_tcp is not None:
            _finite_tuple(self.release_tcp, 6, "release_tcp")
        _finite_tuple(self.endpoint_tcp, 6, "endpoint_tcp")
        _finite_tuple((self.wrist_roll_deg,), 1, "wrist_roll_deg")
        if self.final_gripper not in (0.0, 1.0):
            raise MotionIntentError("final_gripper must be exactly 0 or 1")
        if not isinstance(self.pose_source, str) or not self.pose_source.strip():
            raise MotionIntentError("motion intent requires a recorded pose source")

    def _validate_grasp(self) -> None:
        if self.grip_index != 0 or isinstance(self.grip_index, bool):
            raise MotionIntentError("grip_index must select the single reviewed T29 grip 0")
        if self.grasp_source_sha256 != TRUSTED_GRASP_SOURCE.sha256:
            raise MotionIntentError("motion intent T29 grasp source does not match the reviewed SHA-256")
        try:
            reviewed = _reviewed_grasps()[self.object_id]
        except KeyError as error:
            raise MotionIntentError(f"{self.object_id!r} has no reviewed T29 grasp") from error
        yaw = math.radians(self.object_pose[2])
        local_x, local_y, local_z = reviewed.tcp_local_mm
        expected_xyz = (
            self.object_pose[0] + math.cos(yaw) * local_x - math.sin(yaw) * local_y,
            self.object_pose[1] + math.sin(yaw) * local_x + math.cos(yaw) * local_y,
            self.object_origin_z_mm + local_z,
        )
        if max(abs(actual - expected) for actual, expected in zip(self.grasp_tcp[:3], expected_xyz)) > 1e-6:
            raise MotionIntentError("grasp_tcp does not match the reviewed T29 fingertip offset")
        expected_turn = self.object_pose[2] + reviewed.turn_deg
        if any(
            (
                _angle_error(self.grasp_tcp[3], -180.0) > 1e-6,
                _angle_error(self.grasp_tcp[4], 0.0) > 1e-6,
                _angle_error(self.grasp_tcp[5], expected_turn) > 1e-6,
                _angle_error(self.wrist_roll_deg, expected_turn) > 1e-6,
            )
        ):
            raise MotionIntentError("wrist roll does not match the reviewed T29 object turn")

    def _validate_release(self) -> None:
        if type(self.release_authorized) is not bool:
            raise MotionIntentError("release_authorized must be a boolean")
        release_skills = {
            SemanticSkill.RELEASE_RETREAT,
            SemanticSkill.PULL_OUT,
            SemanticSkill.PLACE_FREE,
            SemanticSkill.DEFECT_DROP,
        }
        if self.skill in release_skills:
            if self.release_tcp is None or not self.release_authorized or self.final_gripper != 0.0:
                raise MotionIntentError(f"{self.skill.value} requires an authorized release TCP and open endpoint")
        elif self.release_tcp is not None or self.release_authorized:
            raise MotionIntentError(f"{self.skill.value} cannot authorize release")

    def _validate_pick_insert(self) -> None:
        if self.skill is SemanticSkill.PICK_INSERT_HOLD:
            if self.final_gripper != 1.0 or self.base_instance_id is None or self.base_object_id is None:
                raise MotionIntentError("pick_insert_hold requires a base and closed endpoint")
            if self.contact is None:
                raise MotionIntentError("pick_insert_hold requires a sourced contact envelope")
            xy_error = math.hypot(self.endpoint_tcp[0] - self.contact.x_mm, self.endpoint_tcp[1] - self.contact.y_mm)
            if xy_error > self.contact.radius_mm or abs(self.endpoint_tcp[2] - self.contact.z_contact_mm) > 1e-6:
                raise MotionIntentError("pick_insert_hold endpoint must agree with its sourced contact envelope")

    @property
    def plan_target(self) -> tuple[float, float, float, float, float, float, float]:
        """The exact T12 language target: raw object pose, roll, then raw goal pose."""
        return (*self.object_pose, float(self.wrist_roll_deg), *self.goal_pose)

    @property
    def endpoint(self) -> tuple[float, float, float, float, float, float, float]:
        """Measured completion condition for this semantic operation."""
        return (*self.endpoint_tcp, self.final_gripper)
