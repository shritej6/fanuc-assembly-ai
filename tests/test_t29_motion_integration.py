"""End-to-end T29 fingertip-frame regressions for every catalog part."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from assembly.calibrated_poses import (
    ApprovedInstalledOrientation,
    FoundationPoseFrameBoundary,
    MeasuredPoseRegistry,
    MeasuredRecipePoses,
)
from assembly.catalog import CAD_TRANSLATION_TOLERANCE_MM, RecipeStep, load_catalog
from assembly.inventory import InventoryItem, InventoryZone, ObjectCondition
from assembly.motion import (
    AbsolutePoseUF0,
    MotionConfig,
    MotionPhase,
    OrientedPartPose,
    RestingState,
    Waypoint,
    compile_pick_to_insert,
    compile_release_retreat,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog()
T29 = json.loads((ROOT / CATALOG.grasp_source.snapshot).read_text())["objects"]
CASES = [
    pytest.param(product.product, step.step, yaw, id=f"{step.object_id}-yaw-{yaw}")
    for product in CATALOG.products
    for step in product.steps
    for yaw in (0.0, 90.0, -135.0)
]


def _pose(x: float, y: float, z: float, yaw: float) -> OrientedPartPose:
    tcp = AbsolutePoseUF0(x, y, z, -180.0, 0.0, yaw, "UF0", "calibrated")
    return OrientedPartPose(tcp, RestingState.INSTALLED, "installed:+z")


def _item(instance_id: str, object_id: str, product: int) -> InventoryItem:
    return InventoryItem(
        instance_id,
        object_id,
        InventoryZone.RESERVED,
        ObjectCondition.VERIFIED,
        reserved_for=product,
        source_zone=InventoryZone.TABLE,
    )


def _provider(product: int, base_yaw: float) -> MeasuredRecipePoses:
    orientations = {
        step.object_id: ApprovedInstalledOrientation(-180.0, 0.0, 0.0, "installed:+z")
        for recipe in CATALOG.products
        for step in recipe.steps
    }
    base = _pose(400.0, 100.0, -320.0, base_yaw)
    measurements = MeasuredPoseRegistry({"base": base}, observed_at_s=10.0, complete=True)
    return MeasuredRecipePoses(
        catalog=CATALOG,
        measurements=measurements,
        installed_orientations=orientations,
        orientation_tolerance_deg=0.5,
        foundationpose_boundary=FoundationPoseFrameBoundary("installed_local", "T30 adapter contract"),
    )


def _world_point(origin: tuple[float, float, float], local: list[float], yaw: float) -> tuple[float, float, float]:
    angle = math.radians(yaw)
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        origin[0] + cosine * local[0] - sine * local[1],
        origin[1] + sine * local[0] + cosine * local[1],
        origin[2] + local[2],
    )


def _config() -> MotionConfig:
    return MotionConfig(30.0, 50.0, 20.0, 40.0, 15.0, 10.0, 3, -180.0, "test fixture")


def _endpoint(chunks, phase: MotionPhase) -> Waypoint:
    return next(chunk for chunk in reversed(chunks) if chunk.phase is phase).waypoints[-1]


def _yaw_error(actual: float, expected: float) -> float:
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


@pytest.mark.parametrize(
    ("product_id", "step_number", "expected_tcp", "superseded_pad_point"),
    [
        (1, 1, (0.0, -19.26, 8.4), (0.0, -21.06, 8.4)),
        (1, 4, (0.0, 1.8, 27.9), (0.0, 0.0, 27.9)),
    ],
)
def test_t29_preferred_turn_uses_flange_axis_tcp_not_pad_point(
    product_id: int,
    step_number: int,
    expected_tcp: tuple[float, float, float],
    superseded_pad_point: tuple[float, float, float],
) -> None:
    step = CATALOG.product(product_id).steps[step_number - 1]

    assert step.preferred_grasp.turn_deg == 90.0
    assert step.preferred_grasp.tcp_local_mm == expected_tcp
    assert step.preferred_grasp.tcp_local_mm != superseded_pad_point


def _assert_t29_target(
    product_id: int,
    step_number: int,
    base_yaw: float,
) -> tuple[RecipeStep, dict[str, Any], MeasuredRecipePoses, OrientedPartPose]:
    product = CATALOG.product(product_id)
    step = product.steps[step_number - 1]
    entry, provider = T29[step.object_id], _provider(product_id, base_yaw)
    base = _item("base", f"F0{product_id}", product_id)
    target = provider.target_for(product, base, step, observed_at_s=10.0)
    grip = entry["grips"][0]
    turn_key = f"{step.preferred_grasp.turn_deg:g}"
    assert tuple(grip["tcp_local_mm"][turn_key]) == step.preferred_grasp.tcp_local_mm
    installed_tcp = [
        left + right for left, right in zip(entry["installed_origin_mm"], step.preferred_grasp.tcp_local_mm)
    ]
    expected = _world_point((400.0, 100.0, -320.0), installed_tcp, base_yaw)
    assert target.tcp.x == pytest.approx(expected[0], abs=CAD_TRANSLATION_TOLERANCE_MM)
    assert target.tcp.y == pytest.approx(expected[1], abs=CAD_TRANSLATION_TOLERANCE_MM)
    assert target.tcp.z == pytest.approx(expected[2], abs=CAD_TRANSLATION_TOLERANCE_MM)
    return step, grip, provider, target


def _compile_t29_motion(
    product_id: int,
    base_yaw: float,
    step: RecipeStep,
    grip: dict[str, Any],
    provider: MeasuredRecipePoses,
    target: OrientedPartPose,
) -> tuple[Waypoint, Waypoint]:
    pick_yaw, pick_origin = base_yaw + 17.0, (250.0, -50.0, -300.0)
    detection = {
        "instance_id": "part",
        "object_id": step.object_id,
        "pose_source": "foundationpose",
        "resting_side": "upright",
        "pose": dict(zip(("x", "y", "z", "yaw"), (*pick_origin, pick_yaw))),
    }
    part = _item("part", step.object_id, product_id)
    grasp = provider.grasp_from_foundationpose(part, detection, observed_at_s=10.0)
    chunks = compile_pick_to_insert(
        Waypoint(200.0, 0.0, -180.0, -180.0, 0.0, 0.0, 0.0),
        grasp,
        target,
        gripper_roll_deg=step.preferred_grasp.turn_deg,
        config=_config(),
    )
    grip_endpoint = _endpoint(chunks, MotionPhase.GRIP)
    insert_endpoint = _endpoint(chunks, MotionPhase.VERTICAL_INSERT)
    assert grip_endpoint.xyz == pytest.approx(
        _world_point(pick_origin, list(step.preferred_grasp.tcp_local_mm), pick_yaw)
    )
    assert _yaw_error(grip_endpoint.r, pick_yaw + step.preferred_grasp.turn_deg) < 1e-9
    return grip_endpoint, insert_endpoint


@pytest.mark.parametrize(("product_id", "step_number", "base_yaw"), CASES)
def test_all_t29_fingertip_targets_and_rolls_reach_emitted_waypoints(
    product_id: int,
    step_number: int,
    base_yaw: float,
) -> None:
    step, grip, provider, target = _assert_t29_target(product_id, step_number, base_yaw)
    _, insert_endpoint = _compile_t29_motion(product_id, base_yaw, step, grip, provider, target)
    assert insert_endpoint.xyz == pytest.approx((target.tcp.x, target.tcp.y, target.tcp.z))
    assert _yaw_error(insert_endpoint.r, base_yaw + step.preferred_grasp.turn_deg) < 1e-9

    release = compile_release_retreat(
        insert_endpoint,
        target,
        gripper_roll_deg=step.preferred_grasp.turn_deg,
        config=_config(),
        max_position_error_mm=0.0,
        max_orientation_error_deg=0.0,
    )
    assert _endpoint(release, MotionPhase.RELEASE).xyz == pytest.approx(insert_endpoint.xyz)
