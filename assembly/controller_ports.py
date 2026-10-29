"""Fresh, complete runtime inputs required by the assembly controller."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Number
from typing import Protocol, TypeVar

from assembly.inventory import Inventory, InventoryItem, InventoryZone, ObjectCondition, UnknownInventoryItemError
from assembly.motion import OrientedPartPose, Waypoint
from assembly.safety import AABB
from robot.adapter import Observation


class ControllerPortError(ValueError):
    """A runtime boundary returned stale, partial, or malformed data."""


PortResult = TypeVar("PortResult")


def call_port(label: str, callback: Callable[..., PortResult], *args: object) -> PortResult:
    """Normalize untrusted perception/runtime callback failures."""
    try:
        return callback(*args)
    except ControllerPortError:
        raise
    except Exception as error:
        raise ControllerPortError(f"{label} failed: {error}") from error


@dataclass(frozen=True, slots=True)
class ObservedInventoryItem:
    instance_id: str
    object_id: str
    zone: InventoryZone = InventoryZone.TABLE
    condition: ObjectCondition = ObjectCondition.VERIFIED
    coarse_xy_mm: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class InventorySnapshot:
    observed_at_s: float
    complete: bool
    items: tuple[ObservedInventoryItem, ...]


@dataclass(frozen=True, slots=True)
class EvidencePair:
    detection: Mapping[str, object]
    inspection: Mapping[str, object]
    observed_at_s: float


@dataclass(frozen=True, slots=True)
class PlacementEvidence(EvidencePair):
    """Independent seated and canonical object-pose checks after insertion."""

    seated: bool
    pose_agrees: bool


@dataclass(frozen=True, slots=True)
class RecoveryTransferEvidence(EvidencePair):
    """Transfer check where pose_agrees means calibrated zone/slot agreement, never TCP equality."""

    target_zone: InventoryZone
    pose_agrees: bool


@dataclass(frozen=True, slots=True)
class CollisionScene:
    observed_at_s: float
    complete: bool
    base_instance_id: str
    base_pose: tuple[float, float, float, float]
    moving_instance_id: str
    moving_attached: bool
    fixtures: tuple[AABB, ...] = ()
    parts: tuple[AABB, ...] = ()


@dataclass(frozen=True, slots=True)
class ControllerEvent:
    kind: str
    product: int | None
    object_id: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class ExecutionTolerances:
    """Measured endpoint limits; no unmeasured defaults are permitted."""

    max_position_error_mm: float
    max_orientation_error_deg: float

    def __post_init__(self) -> None:
        _nonnegative(self.max_position_error_mm, "max_position_error_mm")
        _nonnegative(self.max_orientation_error_deg, "max_orientation_error_deg")


@dataclass(frozen=True, slots=True)
class PerceptionTolerances:
    """Measured cross-sensor pose limits; callers must supply both values."""

    max_position_error_mm: float
    max_yaw_error_deg: float

    def __post_init__(self) -> None:
        _nonnegative(self.max_position_error_mm, "max_position_error_mm")
        _nonnegative(self.max_yaw_error_deg, "max_yaw_error_deg")


class PerceptionPort(Protocol):
    """T27-shaped inference boundary; tests may inject deterministic evidence."""

    def scan(self, observation: Observation) -> InventorySnapshot: ...

    def pre_pick(self, observation: Observation, part: InventoryItem) -> EvidencePair: ...

    def post_place(self, observation: Observation, part: InventoryItem) -> PlacementEvidence: ...

    def post_transfer(
        self,
        observation: Observation,
        part: InventoryItem,
        target_zone: InventoryZone,
    ) -> RecoveryTransferEvidence: ...

    def collision_scene(
        self,
        observation: Observation,
        moving_instance_id: str,
        base_instance_id: str,
    ) -> CollisionScene: ...


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Number):
        raise ControllerPortError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ControllerPortError(f"{field} must be a finite number")
    return number


def _nonnegative(value: object, field: str) -> float:
    number = _finite(value, field)
    if number < 0.0:
        raise ControllerPortError(f"{field} must be nonnegative")
    return number


def observation_time(observation: Observation) -> float:
    return _finite(observation.get("t"), "observation.t")


def reconcile_inventory(
    inventory: Inventory,
    snapshot: object,
    expected_time_s: float,
    *,
    held_instance_id: str | None = None,
) -> None:
    if not isinstance(snapshot, InventorySnapshot):
        raise ControllerPortError("perception scan must return InventorySnapshot")
    require_fresh(snapshot.observed_at_s, expected_time_s, "inventory snapshot")
    require_complete(snapshot.complete, "inventory snapshot")
    if not isinstance(snapshot.items, tuple):
        raise ControllerPortError("inventory snapshot items must be a tuple")
    seen: set[str] = set()
    for observed in snapshot.items:
        _reconcile_item(inventory, observed, seen)
    for item in inventory.items():
        if (
            item.zone
            in {
                InventoryZone.TABLE,
                InventoryZone.SPARE,
                InventoryZone.RESERVED,
                InventoryZone.STAGED,
            }
            and item.instance_id not in seen
            and item.instance_id != held_instance_id
        ):
            inventory.mark_unobserved(item.instance_id)
    _require_claimed_items_verified(inventory)


def _require_claimed_items_verified(inventory: Inventory) -> None:
    claimed = {InventoryZone.RESERVED, InventoryZone.INSTALLED, InventoryZone.STAGED}
    actionable = {ObjectCondition.FALLEN, ObjectCondition.DEFECTIVE}
    for item in inventory.items():
        if (
            item.reserved_for is not None
            and item.zone in claimed
            and item.condition is not ObjectCondition.VERIFIED
            and item.condition not in actionable
        ):
            raise ControllerPortError(f"claimed instance {item.instance_id!r} is not verified in the complete scan")


def _reconcile_item(inventory: Inventory, observed: object, seen: set[str]) -> None:
    if not isinstance(observed, ObservedInventoryItem):
        raise ControllerPortError("inventory snapshot contains an invalid item")
    if observed.instance_id in seen:
        raise ControllerPortError(f"duplicate observed instance {observed.instance_id!r}")
    if observed.coarse_xy_mm is not None:
        _numeric_values(observed.coarse_xy_mm, 2, "coarse_xy_mm must contain x and y", "coarse_xy_mm")
    seen.add(observed.instance_id)
    try:
        inventory.reconcile_observation(
            observed.instance_id,
            observed.object_id,
            zone=observed.zone,
            condition=observed.condition,
        )
    except UnknownInventoryItemError:
        inventory.add(
            observed.instance_id,
            observed.object_id,
            zone=observed.zone,
            condition=observed.condition,
        )


def require_fresh(observed_at_s: object, expected_s: float, label: str) -> None:
    if _finite(observed_at_s, f"{label}.observed_at_s") != expected_s:
        raise ControllerPortError(f"{label} does not belong to the current observation")


def require_complete(complete: object, label: str) -> None:
    if complete is not True:
        raise ControllerPortError(f"{label} must be explicitly complete")


def endpoint_reached(actual: Waypoint, expected: Waypoint, tolerances: ExecutionTolerances) -> bool:
    position_error = math.dist(actual.xyz, expected.xyz)
    angle_errors = tuple(_angle_error(left, right) for left, right in zip(_angles(actual), _angles(expected)))
    return (
        position_error <= tolerances.max_position_error_mm
        and max(angle_errors) <= tolerances.max_orientation_error_deg
        and actual.gripper == expected.gripper
    )


def pose_agrees(
    actual: object,
    expected: object,
    tolerances: PerceptionTolerances,
) -> bool:
    """Compare independently measured x/y/z/yaw poses with wrapped yaw."""
    left = _pose_values(actual, "actual pose")
    right = _pose_values(expected, "expected pose")
    return (
        math.dist(left[:3], right[:3]) <= tolerances.max_position_error_mm
        and _angle_error(left[3], right[3]) <= tolerances.max_yaw_error_deg
    )


def oriented_pose_values(pose: object) -> tuple[float, float, float, float]:
    if not isinstance(pose, OrientedPartPose):
        raise ControllerPortError("oriented pose must be an OrientedPartPose")
    return (pose.tcp.x, pose.tcp.y, pose.tcp.z, pose.tcp.r)


def _pose_values(value: object, field: str) -> tuple[float, float, float, float]:
    if isinstance(value, Mapping):
        if set(value) != {"x", "y", "z", "yaw"}:
            raise ControllerPortError(f"{field} must contain only x, y, z, and yaw")
        raw = tuple(value[name] for name in ("x", "y", "z", "yaw"))
    else:
        raw = _numeric_values(value, 4, f"{field} must contain x, y, z, and yaw", field)
    return tuple(_finite(item, field) for item in raw)  # type: ignore[return-value]


def _angles(waypoint: Waypoint) -> tuple[float, float, float]:
    return (waypoint.w, waypoint.p, waypoint.r)


def _angle_error(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def current_waypoint(observation: Observation) -> Waypoint:
    values = _numeric_values(
        observation.get("tcp_pose"),
        6,
        "observation tcp_pose must contain six absolute UF0 values",
        "observation.tcp_pose",
    )
    joints = _numeric_values(
        observation.get("joints"),
        7,
        "observation joints must contain six joints and gripper",
        "observation.joints",
    )
    gripper = _finite(joints[6], "observation.joints.gripper")
    return Waypoint(*values, gripper)


def _numeric_values(value: object, count: int, shape_error: str, field: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not hasattr(value, "__len__") or not hasattr(value, "__getitem__"):
        raise ControllerPortError(shape_error)
    try:
        if len(value) != count:  # type: ignore[arg-type]
            raise ControllerPortError(shape_error)
        raw = tuple(value[index] for index in range(count))  # type: ignore[index]
    except (IndexError, KeyError, TypeError) as error:
        raise ControllerPortError(shape_error) from error
    return tuple(_finite(item, field) for item in raw)
