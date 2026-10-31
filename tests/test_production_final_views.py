"""Production completion views require sourced CSI poses; there are no defaults."""

from __future__ import annotations

import pytest

from assembly.final_views import (
    FinalViewError,
    FinalViewPose,
    FinalViewRegistry,
)


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
