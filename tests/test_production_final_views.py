"""Production completion views require sourced CSI poses; there are no defaults."""

from __future__ import annotations

from dataclasses import replace

import pytest

from assembly.final_views import (
    FinalViewError,
    FinalViewPose,
    FinalViewRegistry,
)
from robot.cell.real_layout import reviewed_final_view_registry


def _view(view_id: str, x_mm: float) -> FinalViewPose:
    return FinalViewPose(
        product=1,
        base_object_id="F01",
        view_id=view_id,
        endpoint_tcp_uf0=(x_mm, 0.0, -168.0, -180.0, 0.0, 180.0),
        safe_z_mm=-180.0,
        expected_instance_roles=("step:1", "step:2", "step:3", "step:4"),
        source="measured CSI final-view fixture",
    )


def test_registry_returns_exactly_two_distinct_sourced_views() -> None:
    left = _view("left", 300.0)
    right = _view("right", 340.0)

    registry = FinalViewRegistry((left, right))

    assert registry.views_for(1, "F01") == (left, right)


@pytest.mark.parametrize(
    "views, message",
    [
        ((), "exactly two"),
        ((_view("left", 300.0),), "exactly two"),
        ((_view("same", 300.0), _view("same", 340.0)), "distinct"),
    ],
)
def test_registry_refuses_missing_or_self_attested_view_pairs(views, message: str) -> None:
    with pytest.raises(FinalViewError, match=message):
        FinalViewRegistry(views)


def test_registry_refuses_unmeasured_or_non_camera_safe_pose() -> None:
    with pytest.raises(FinalViewError, match="source"):
        FinalViewPose(
            product=1,
            base_object_id="F01",
            view_id="left",
            endpoint_tcp_uf0=(300.0, 0.0, -168.0, -180.0, 0.0, 180.0),
            safe_z_mm=-180.0,
            expected_instance_roles=("step:1", "step:2", "step:3", "step:4"),
            source="",
        )
    with pytest.raises(FinalViewError, match="straight down"):
        FinalViewPose(
            product=1,
            base_object_id="F01",
            view_id="left",
            endpoint_tcp_uf0=(300.0, 0.0, -168.0, -160.0, 0.0, 180.0),
            safe_z_mm=-180.0,
            expected_instance_roles=("step:1", "step:2", "step:3", "step:4"),
            source="measured CSI final-view fixture",
        )


def test_registry_refuses_unknown_product_base_pair() -> None:
    registry = FinalViewRegistry((_view("left", 300.0), _view("right", 340.0)))

    with pytest.raises(FinalViewError, match="no reviewed final views"):
        registry.views_for(2, "F02")


def test_registry_selects_the_reviewed_base_spot_when_an_object_moves_between_rollouts() -> None:
    left = replace(_view("left-b1", 300.0), base_center_xy_mm=(320.0, 0.0))
    right = replace(_view("right-b1", 340.0), base_center_xy_mm=(320.0, 0.0))
    upper_left = replace(
        left,
        view_id="left-b3",
        endpoint_tcp_uf0=(300.0, 165.0, -168.0, -180.0, 0.0, 180.0),
        base_center_xy_mm=(320.0, 165.0),
    )
    upper_right = replace(
        right,
        view_id="right-b3",
        endpoint_tcp_uf0=(340.0, 165.0, -168.0, -180.0, 0.0, 180.0),
        base_center_xy_mm=(320.0, 165.0),
    )
    registry = FinalViewRegistry((left, right, upper_left, upper_right))

    assert registry.views_for(1, "F01", base_xy_mm=(320.0, 164.0), max_distance_mm=10.0) == (
        upper_left,
        upper_right,
    )
    with pytest.raises(FinalViewError, match="outside the reviewed"):
        registry.views_for(1, "F01", base_xy_mm=(320.0, 80.0), max_distance_mm=10.0)


def test_reviewed_registry_has_product_five_b2_views_and_catalog_roles() -> None:
    views = reviewed_final_view_registry().views_for(
        5,
        "F05",
        base_xy_mm=(420.0, 0.0),
        max_distance_mm=10.0,
    )

    assert tuple(view.endpoint_tcp_uf0[:2] for view in views) == ((420.0, 0.0),) * 2
    assert {view.endpoint_tcp_uf0[5] for view in views} == {45.0, 225.0}
    assert all(view.expected_instance_roles == ("M5_01", "M5_02", "M5_03", "M5_04") for view in views)
    assert all("PRELIMINARY" in view.source and "1f97ff5" in view.source for view in views)
