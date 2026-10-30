"""Typed non-learned camera positioning for the CELL inspection gate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Number


class FixedInspectionError(ValueError):
    """A fixed inspection request lacks reviewed finite pose evidence."""


@dataclass(frozen=True, slots=True)
class FixedInspectionMove:
    """One reviewed camera-only pose; it carries no gripper command."""

    instance_id: str
    base_instance_id: str
    endpoint_tcp: tuple[float, float, float, float, float, float]
    safe_z_mm: float
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise FixedInspectionError("fixed inspection requires a part instance")
        if not isinstance(self.base_instance_id, str) or not self.base_instance_id.strip():
            raise FixedInspectionError("fixed inspection requires a base instance")
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
) -> tuple[float, float, float, float, float, float]:
    """Choose rise, safe-plane transit, or final camera approach."""
    endpoint = move.endpoint_tcp
    if current_tcp[2] < move.safe_z_mm - 1e-9:
        return (*current_tcp[:2], move.safe_z_mm, *current_tcp[3:])
    xy_error = math.hypot(current_tcp[0] - endpoint[0], current_tcp[1] - endpoint[1])
    angle_error = max(
        abs((current_tcp[index] - endpoint[index] + 180.0) % 360.0 - 180.0)
        for index in range(3, 6)
    )
    if xy_error > 1e-9 or angle_error > 1e-9:
        transit_z = max(current_tcp[2], move.safe_z_mm)
        return (*endpoint[:2], transit_z, *endpoint[3:])
    return endpoint
