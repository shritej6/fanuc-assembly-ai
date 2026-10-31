"""Production completion views require sourced CSI poses; there are no defaults."""

from __future__ import annotations

from dataclasses import replace

import pytest

from assembly.final_views import (
    FinalViewError,
    FinalViewPose,
    FinalViewRegistry,
)
from perception.cell.targets import MAX_HELD_BELOW_TCP_MM, TABLE_Z_UF0_MM
from perception.port.cad import load_cad
from perception.port.placement import precision_gate_tolerances
from robot.cell.real_layout import INSPECTION_SAFE_Z_MM, reviewed_final_view_registry


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


def test_registry_refuses_equivalent_views_or_mismatched_identity_roles() -> None:
    left = _view("left", 300.0)
    equivalent = replace(
        left,
        view_id="equivalent",
        endpoint_tcp_uf0=(*left.endpoint_tcp_uf0[:5], left.endpoint_tcp_uf0[5] + 360.0),
    )
    with pytest.raises(FinalViewError, match="physically distinct"):
        FinalViewRegistry((left, equivalent))

    wrong_roles = replace(
        _view("right", 340.0),
        expected_instance_roles=("step:1", "step:2", "step:3", "wrong"),
    )
    with pytest.raises(FinalViewError, match="identity roles"):
        FinalViewRegistry((left, wrong_roles))
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


def test_reviewed_registry_has_product_five_views_at_the_new_base_pose() -> None:
    views = reviewed_final_view_registry().views_for(
        5,
        "F05",
        base_xy_mm=(275.0, -123.0),
        max_distance_mm=10.0,
    )

    assert tuple(view.endpoint_tcp_uf0[:2] for view in views) == ((275.0, -148.0), (275.0, -98.0))
    assert {view.endpoint_tcp_uf0[5] for view in views} == {326.0}
    assert all(view.expected_instance_roles == ("M5_01", "M5_02", "M5_03", "M5_04") for view in views)
    assert all("PRELIMINARY" in view.source and "434f488" in view.source for view in views)


def test_every_reviewed_final_view_clears_the_assembled_cad_stack() -> None:
    cad = load_cad()
    execution, _perception = precision_gate_tolerances()
    required_mm = MAX_HELD_BELOW_TCP_MM + execution.max_position_error_mm

    for view in reviewed_final_view_registry().views:
        object_ids = (view.base_object_id, *view.expected_instance_roles)
        stack_top_z = TABLE_Z_UF0_MM + max(cad.get(object_id).seated_top_z_mm for object_id in object_ids)
        lowest_tcp_z = min(view.safe_z_mm, view.endpoint_tcp_uf0[2])

        assert view.safe_z_mm == INSPECTION_SAFE_Z_MM
        assert lowest_tcp_z - stack_top_z >= required_mm
