"""Sourced fixed camera views required before production may complete a product."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Number


class FinalViewError(ValueError):
    """A production final-view definition is absent, ambiguous, or unsourced."""


_MIN_INDEPENDENT_POSITION_MM = 0.3
_MIN_INDEPENDENT_ANGLE_DEG = 1.0


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Number):
        raise FinalViewError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise FinalViewError(f"{field} must be a finite number")
    return number


@dataclass(frozen=True, slots=True)
class FinalViewPose:
    """One measured camera pose and the identities it must independently see."""

    product: int
    base_object_id: str
    view_id: str
    endpoint_tcp_uf0: tuple[float, float, float, float, float, float]
    safe_z_mm: float
    expected_instance_roles: tuple[str, str, str, str]
    source: str
    base_center_xy_mm: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.product, bool) or not isinstance(self.product, int) or self.product <= 0:
            raise FinalViewError("final view product must be a positive integer")
        if not isinstance(self.base_object_id, str) or not self.base_object_id.startswith("F"):
            raise FinalViewError("final view requires a fixture object ID")
        if not isinstance(self.view_id, str) or not self.view_id.strip():
            raise FinalViewError("final view requires a nonempty view ID")
        if not isinstance(self.endpoint_tcp_uf0, tuple) or len(self.endpoint_tcp_uf0) != 6:
            raise FinalViewError("final view endpoint must contain six UF0 values")
        pose = tuple(_finite(value, "final view endpoint") for value in self.endpoint_tcp_uf0)
        _finite(self.safe_z_mm, "final view safe Z")
        if abs(abs(pose[3]) - 180.0) > 1e-9 or abs(pose[4]) > 1e-9:
            raise FinalViewError("final view must keep UTOOL1 straight down")
        roles = self.expected_instance_roles
        if type(roles) is not tuple or len(roles) != 4:
            raise FinalViewError("final view requires exactly four expected instance roles")
        if any(not isinstance(role, str) or not role.strip() for role in roles):
            raise FinalViewError("final view instance roles must be nonempty strings")
        if len(set(roles)) != len(roles):
            raise FinalViewError("final view instance roles must be unique")
        if not isinstance(self.source, str) or not self.source.strip():
            raise FinalViewError("final view requires recorded measurement source")
        center = self.base_center_xy_mm
        if center is not None:
            if type(center) is not tuple or len(center) != 2:
                raise FinalViewError("final view base center must contain x and y")
            tuple(_finite(value, "final view base center") for value in center)


@dataclass(frozen=True, slots=True)
class FinalViewRegistry:
    """Resolve exactly two independently measured views for a product/base pair."""

    views: tuple[FinalViewPose, ...]

    def __post_init__(self) -> None:
        if type(self.views) is not tuple or any(type(view) is not FinalViewPose for view in self.views):
            raise FinalViewError("final-view registry requires typed views")
        grouped: dict[tuple[int, str, tuple[float, float] | None], list[FinalViewPose]] = {}
        for view in self.views:
            grouped.setdefault(
                (view.product, view.base_object_id, view.base_center_xy_mm),
                [],
            ).append(view)
        if not grouped:
            raise FinalViewError("each product/base pair requires exactly two final views")
        for key, pair in grouped.items():
            self._validate_pair(key, pair)

    @staticmethod
    def _validate_pair(
        key: tuple[int, str, tuple[float, float] | None],
        pair: list[FinalViewPose],
    ) -> None:
        if len(pair) != 2:
            raise FinalViewError(f"{key!r} requires exactly two final views")
        if pair[0].view_id == pair[1].view_id:
            raise FinalViewError(f"{key!r} requires distinct final-view IDs")
        if pair[0].expected_instance_roles != pair[1].expected_instance_roles:
            raise FinalViewError(f"{key!r} final views must bind the same identity roles")
        position_delta = math.dist(
            pair[0].endpoint_tcp_uf0[:3],
            pair[1].endpoint_tcp_uf0[:3],
        )
        angle_delta = max(
            _wrapped_angle_delta(first, second)
            for first, second in zip(
                pair[0].endpoint_tcp_uf0[3:],
                pair[1].endpoint_tcp_uf0[3:],
                strict=True,
            )
        )
        if (
            position_delta <= _MIN_INDEPENDENT_POSITION_MM
            and angle_delta <= _MIN_INDEPENDENT_ANGLE_DEG
        ):
            raise FinalViewError(f"{key!r} requires physically distinct final-view poses")

    def views_for(
        self,
        product: int,
        base_object_id: str,
        *,
        base_xy_mm: tuple[float, float] | None = None,
        max_distance_mm: float | None = None,
    ) -> tuple[FinalViewPose, FinalViewPose]:
        groups = self._groups_for(product, base_object_id)
        if base_xy_mm is not None:
            groups = self._nearby_groups(groups, base_xy_mm, max_distance_mm)
        if len(groups) != 1:
            raise FinalViewError(
                f"no reviewed final views for product {product} on {base_object_id}"
            )
        return groups[0]

    def _groups_for(
        self,
        product: int,
        base_object_id: str,
    ) -> tuple[tuple[FinalViewPose, FinalViewPose], ...]:
        centers = {
            view.base_center_xy_mm
            for view in self.views
            if view.product == product and view.base_object_id == base_object_id
        }
        return tuple(
            tuple(
                view
                for view in self.views
                if view.product == product
                and view.base_object_id == base_object_id
                and view.base_center_xy_mm == center
            )
            for center in centers
        )  # type: ignore[return-value]

    @staticmethod
    def _nearby_groups(
        groups: tuple[tuple[FinalViewPose, FinalViewPose], ...],
        base_xy_mm: tuple[float, float],
        max_distance_mm: float | None,
    ) -> tuple[tuple[FinalViewPose, FinalViewPose], ...]:
        if type(base_xy_mm) is not tuple or len(base_xy_mm) != 2:
            raise FinalViewError("measured base center must contain x and y")
        center = tuple(_finite(value, "measured base center") for value in base_xy_mm)
        limit = _finite(max_distance_mm, "final-view selection tolerance")
        if limit < 0.0:
            raise FinalViewError("final-view selection tolerance must be nonnegative")
        nearby = []
        for pair in groups:
            reviewed = pair[0].base_center_xy_mm
            if reviewed is None:
                continue
            if math.hypot(center[0] - reviewed[0], center[1] - reviewed[1]) <= limit:
                nearby.append(pair)
        if not nearby:
            raise FinalViewError("measured base center is outside the reviewed final-view spots")
        return tuple(nearby)


def _wrapped_angle_delta(first_deg: float, second_deg: float) -> float:
    return abs((first_deg - second_deg + 180.0) % 360.0 - 180.0)
