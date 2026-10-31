"""Typed non-learned camera positioning for the CELL inspection gate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Number

from assembly.controller_ports import ExecutionTolerances


class FixedInspectionError(ValueError):
    """A fixed inspection request lacks reviewed finite pose evidence."""


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Number):
        raise FixedInspectionError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise FixedInspectionError(f"{field} must be finite")
    return numeric


@dataclass(frozen=True, slots=True)
class InspectionSlotPose:
    """One reviewed selection region and its exact camera pose in UF0."""

    slot_id: str
    selection_bounds_xy_mm: tuple[float, float, float, float]
    endpoint_tcp: tuple[float, float, float, float, float, float]
    safe_z_mm: float
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.slot_id, str) or not self.slot_id.strip():
            raise FixedInspectionError("inspection slot requires an identifier")
        if not isinstance(self.source, str) or not self.source.strip():
            raise FixedInspectionError("inspection slot requires recorded pose provenance")
        if not isinstance(self.selection_bounds_xy_mm, tuple) or len(self.selection_bounds_xy_mm) != 4:
            raise FixedInspectionError("inspection slot selection bounds must contain xmin, xmax, ymin, ymax")
        xmin, xmax, ymin, ymax = (
            _finite(value, "inspection slot selection bound")
            for value in self.selection_bounds_xy_mm
        )
        if xmin >= xmax or ymin >= ymax:
            raise FixedInspectionError("inspection slot selection bounds must increase")
        FixedInspectionMove(
            "validation", "validation", self.slot_id, self.endpoint_tcp,
            self.safe_z_mm, self.source,
        )

    def contains(self, coarse_xy_mm: tuple[float, float]) -> bool:
        x, y = coarse_xy_mm
        xmin, xmax, ymin, ymax = self.selection_bounds_xy_mm
        return xmin <= x <= xmax and ymin <= y <= ymax


@dataclass(frozen=True, slots=True)
class InspectionPoseRegistry:
    """Resolve a coarse detection to exactly one reviewed fixed slot pose."""

    slots: tuple[InspectionSlotPose, ...]

    def __post_init__(self) -> None:
        if not self.slots or any(not isinstance(slot, InspectionSlotPose) for slot in self.slots):
            raise FixedInspectionError("inspection registry requires reviewed slot poses")
        identifiers = [slot.slot_id for slot in self.slots]
        if len(identifiers) != len(set(identifiers)):
            raise FixedInspectionError("inspection slot identifiers must be unique")
        for index, left in enumerate(self.slots):
            for right in self.slots[index + 1 :]:
                if _regions_overlap(left, right):
                    raise FixedInspectionError(
                        f"inspection slot selection regions overlap: {left.slot_id!r}, {right.slot_id!r}"
                    )

    def resolve(self, coarse_xy_mm: tuple[float, float]) -> InspectionSlotPose:
        if not isinstance(coarse_xy_mm, tuple) or len(coarse_xy_mm) != 2:
            raise FixedInspectionError("coarse inspection center must contain x and y")
        center = tuple(_finite(value, "coarse inspection center") for value in coarse_xy_mm)
        matches = tuple(slot for slot in self.slots if slot.contains(center))
        if len(matches) != 1:
            raise FixedInspectionError(
                f"no reviewed inspection slot contains coarse center {center!r}"
            )
        return matches[0]


def _regions_overlap(left: InspectionSlotPose, right: InspectionSlotPose) -> bool:
    lx0, lx1, ly0, ly1 = left.selection_bounds_xy_mm
    rx0, rx1, ry0, ry1 = right.selection_bounds_xy_mm
    return lx0 <= rx1 and rx0 <= lx1 and ly0 <= ry1 and ry0 <= ly1


@dataclass(frozen=True, slots=True)
class FixedInspectionMove:
    """One reviewed camera-only pose; execution requires an open gripper."""

    instance_id: str
    base_instance_id: str
    slot_id: str
    endpoint_tcp: tuple[float, float, float, float, float, float]
    safe_z_mm: float
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise FixedInspectionError("fixed inspection requires a part instance")
        if not isinstance(self.base_instance_id, str) or not self.base_instance_id.strip():
            raise FixedInspectionError("fixed inspection requires a base instance")
        if not isinstance(self.slot_id, str) or not self.slot_id.strip():
            raise FixedInspectionError("fixed inspection requires a reviewed slot identifier")
        if not isinstance(self.source, str) or not self.source.strip():
            raise FixedInspectionError("fixed inspection requires recorded pose provenance")
        if not isinstance(self.endpoint_tcp, tuple) or len(self.endpoint_tcp) != 6:
            raise FixedInspectionError("fixed inspection endpoint must contain six values")
        if any(isinstance(value, bool) or not isinstance(value, Number) for value in self.endpoint_tcp):
            raise FixedInspectionError("fixed inspection endpoint must be numeric")
        if not all(math.isfinite(float(value)) for value in self.endpoint_tcp):
            raise FixedInspectionError("fixed inspection endpoint must be finite")
        if isinstance(self.safe_z_mm, bool) or not isinstance(self.safe_z_mm, Number):
            raise FixedInspectionError("fixed inspection safe Z must be numeric")
        if not math.isfinite(float(self.safe_z_mm)):
            raise FixedInspectionError("fixed inspection safe Z must be finite")
        if abs(abs(self.endpoint_tcp[3]) - 180.0) > 1e-9 or abs(self.endpoint_tcp[4]) > 1e-9:
            raise FixedInspectionError("fixed inspection must keep UTOOL1 straight down")

    @property
    def label(self) -> str:
        """Operator-visible authority label required for non-learned motion."""
        return "non_learned_inspection"


def next_inspection_leg(
    current_tcp: tuple[float, float, float, float, float, float],
    move: FixedInspectionMove,
    tolerances: ExecutionTolerances,
) -> tuple[float, float, float, float, float, float]:
    """Choose rise, safe-plane transit, or final camera approach."""
    endpoint = move.endpoint_tcp
    if current_tcp[2] < move.safe_z_mm - tolerances.max_position_error_mm:
        return (*current_tcp[:2], move.safe_z_mm, *current_tcp[3:])
    xy_error = math.hypot(current_tcp[0] - endpoint[0], current_tcp[1] - endpoint[1])
    angle_error = max(
        abs((current_tcp[index] - endpoint[index] + 180.0) % 360.0 - 180.0)
        for index in range(3, 6)
    )
    if (
        xy_error > tolerances.max_position_error_mm
        or angle_error > tolerances.max_orientation_error_deg
    ):
        transit_z = max(current_tcp[2], move.safe_z_mm)
        return (*endpoint[:2], transit_z, *endpoint[3:])
    return endpoint
