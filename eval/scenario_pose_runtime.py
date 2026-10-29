"""Mock-only calibrated pose and recovery-target providers for T17."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from assembly.catalog import ProductRecipe, RecipeStep
from assembly.controller_ports import ControllerPortError
from assembly.inventory import InventoryItem, InventoryZone
from assembly.motion import AbsolutePoseUF0, OrientedPartPose, RestingState
from eval.scenario_data import Scenario

Pose4 = tuple[float, float, float, float]


@dataclass(slots=True)
class ScenarioPoseState:
    pose_by_instance: dict[str, Pose4]
    observed_pose_by_instance: dict[str, Pose4]
    target_by_object: dict[str, Pose4]
    transfer_by_key: dict[tuple[str, InventoryZone], Pose4]
    transfer_result_by_key: dict[tuple[str, InventoryZone], Pose4]


def observed_pose(state: ScenarioPoseState, instance_id: str) -> Pose4:
    try:
        return state.observed_pose_by_instance[instance_id]
    except KeyError as error:
        raise ControllerPortError(f"instance {instance_id!r} was not observed by perception") from error


class ScenarioPoseProvider:
    def __init__(self, state: ScenarioPoseState, tcp_local_mm: tuple[float, float, float]) -> None:
        self._state = state
        self._tcp_local_mm = tcp_local_mm

    def grasp_for(self, part: InventoryItem, *, observed_at_s: float) -> OrientedPartPose:
        del part, observed_at_s
        raise ControllerPortError("FoundationPose wrist evidence required for every part grasp")

    def grasp_from_foundationpose(
        self,
        part: InventoryItem,
        detection: dict[str, object],
        *,
        observed_at_s: float,
    ) -> OrientedPartPose:
        del observed_at_s
        if detection.get("instance_id") != part.instance_id or detection.get("object_id") != part.object_id:
            raise ControllerPortError("FoundationPose detection identity does not match reserved part")
        resting_side = detection.get("resting_side")
        if resting_side == "flipped":
            raise ControllerPortError(f"no approved grasp for resting side of {part.instance_id!r}")
        if resting_side not in {"upright", "side"}:
            raise ControllerPortError("FoundationPose detection resting side is unsupported")
        state = RestingState.FALLEN if resting_side == "side" else RestingState.INSTALLED
        tcp = _pose_with_local_offset(_detection_pose(detection), self._tcp_local_mm)
        return _oriented(tcp, part.object_id, state)

    def base_for(self, base: InventoryItem, *, observed_at_s: float) -> OrientedPartPose:
        del observed_at_s
        return _oriented(observed_pose(self._state, base.instance_id), base.object_id)

    def target_for(
        self,
        product: ProductRecipe,
        base: InventoryItem,
        step: RecipeStep,
        *,
        observed_at_s: float,
    ) -> OrientedPartPose:
        del product, observed_at_s
        canonical = self._install_target(base, step)
        return _oriented(_pose_with_local_offset(canonical, self._tcp_local_mm), step.object_id)

    def verification_target_for(
        self,
        product: ProductRecipe,
        base: InventoryItem,
        step: RecipeStep,
        *,
        observed_at_s: float,
    ) -> OrientedPartPose:
        del product, observed_at_s
        return _oriented(self._install_target(base, step), step.object_id)

    def _install_target(self, base: InventoryItem, step: RecipeStep) -> Pose4:
        canonical = _offset_pose(observed_pose(self._state, base.instance_id), step.insert_offset_mm)
        self._state.target_by_object[step.object_id] = canonical
        return canonical


class ScenarioRecoveryTargets:
    def __init__(self, scenario: Scenario, state: ScenarioPoseState) -> None:
        self._scenario = scenario
        self._state = state

    def staging_target_for(self, product: int, part: InventoryItem, slot: int) -> OrientedPartPose:
        del product
        pose = self._staging_pose(slot)
        self._state.transfer_by_key[(part.instance_id, InventoryZone.STAGED)] = pose
        tcp = _pose_with_local_offset(pose, self._scenario.bounds.tcp_local_mm)
        return _oriented(tcp, part.object_id)

    def staging_verification_target_for(
        self,
        product: int,
        part: InventoryItem,
        slot: int,
    ) -> OrientedPartPose:
        del product
        pose = self._staging_pose(slot)
        self._state.transfer_by_key[(part.instance_id, InventoryZone.STAGED)] = pose
        return _oriented(pose, part.object_id)

    def _staging_pose(self, slot: int) -> Pose4:
        low, high = self._scenario.bounds.staging_y_mm
        fraction = (slot % 10) / 10.0
        return (
            self._scenario.bounds.staging_x_mm,
            low + (high - low) * fraction,
            self._scenario.bounds.part_z_mm,
            0.0,
        )

    def reject_target_for(self, part: InventoryItem) -> OrientedPartPose:
        pose = self._scenario.bounds.reject_pose
        self._state.transfer_by_key[(part.instance_id, InventoryZone.REJECT)] = pose
        tcp = _pose_with_local_offset(pose, self._scenario.bounds.tcp_local_mm)
        return _oriented(tcp, part.object_id)


def _detection_pose(detection: Mapping[str, object]) -> Pose4:
    pose = detection.get("pose")
    if not isinstance(pose, Mapping):
        raise ControllerPortError("FoundationPose detection pose must be a mapping")
    values = tuple(pose.get(field) for field in ("x", "y", "z", "yaw"))
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ControllerPortError("FoundationPose detection pose must contain numeric x, y, z, and yaw")
    numeric = tuple(float(value) for value in values)  # type: ignore[arg-type]
    if not all(math.isfinite(value) for value in numeric):
        raise ControllerPortError("FoundationPose detection pose must be finite")
    return numeric  # type: ignore[return-value]


def _oriented(
    pose: Pose4,
    object_id: str,
    resting_state: RestingState = RestingState.INSTALLED,
) -> OrientedPartPose:
    x, y, z, yaw = pose
    tcp = AbsolutePoseUF0(x, y, z, -180.0, 0.0, yaw, "UF0", "calibrated")
    return OrientedPartPose(tcp, resting_state, f"installed:{object_id}")


def _offset_pose(base: Pose4, offset: tuple[float, float, float]) -> Pose4:
    x, y, z, yaw = base
    angle = math.radians(yaw)
    dx, dy, dz = offset
    return (
        x + dx * math.cos(angle) - dy * math.sin(angle),
        y + dx * math.sin(angle) + dy * math.cos(angle),
        z + dz,
        yaw,
    )


def _pose_with_local_offset(pose: Pose4, offset: tuple[float, float, float]) -> Pose4:
    return _offset_pose(pose, offset)
