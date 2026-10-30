"""Measured-pose projection into absolute UF0 assembly targets."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from assembly.calibrated_poses import (
    ApprovedInstalledOrientation,
    CalibratedPoseError,
    FoundationPoseFrameBoundary,
    MeasuredPoseRegistry,
    MeasuredRecipePoses,
)
from assembly.catalog import load_catalog
from assembly.inventory import InventoryItem, InventoryZone, ObjectCondition
from assembly.motion import AbsolutePoseUF0, OrientedPartPose, RestingState


def _pose(
    x: float,
    y: float,
    z: float,
    yaw: float,
    *,
    up_face: str = "installed:+z",
) -> OrientedPartPose:
    return OrientedPartPose(
        AbsolutePoseUF0(x, y, z, -180.0, 0.0, yaw, "UF0", "calibrated"),
        RestingState.INSTALLED,
        up_face,
    )


def _item(instance_id: str, object_id: str) -> InventoryItem:
    return InventoryItem(
        instance_id,
        object_id,
        InventoryZone.RESERVED,
        ObjectCondition.VERIFIED,
        reserved_for=4,
        source_zone=InventoryZone.TABLE,
    )


def _provider(
    *,
    product_id: int = 4,
    base: OrientedPartPose | None = None,
    part: OrientedPartPose | None = None,
    observed_at_s: float = 10.0,
    complete: bool = True,
):
    catalog = load_catalog()
    measurements = {}
    if base is not None:
        measurements["base-4"] = base
    if part is not None:
        measurements["part-4"] = part
    orientations = {
        step.object_id: ApprovedInstalledOrientation(-180.0, 0.0, 0.0, "installed:+z")
        for step in catalog.product(product_id).steps
    }
    return MeasuredRecipePoses(
        catalog=catalog,
        measurements=MeasuredPoseRegistry(measurements, observed_at_s=observed_at_s, complete=complete),
        installed_orientations=orientations,
        orientation_tolerance_deg=0.5,
        foundationpose_boundary=FoundationPoseFrameBoundary(
            "installed_local",
            "test canonical-pose adapter",
        ),
    )


def test_raw_stl_foundationpose_frame_cannot_be_silently_configured() -> None:
    with pytest.raises(CalibratedPoseError, match="installed_local"):
        FoundationPoseFrameBoundary("stl_mesh", "raw FoundationPose output")


def test_target_transforms_cad_local_offset_through_measured_base_yaw() -> None:
    catalog = load_catalog()
    product = catalog.product(4)
    step = product.steps[3]
    provider = _provider(base=_pose(400.0, 100.0, -320.0, 90.0), part=_pose(250.0, -50.0, -300.0, 90.0))

    target = provider.target_for(product, _item("base-4", "F04"), step, observed_at_s=10.0)

    assert target.tcp.x == pytest.approx(411.5)
    assert target.tcp.y == pytest.approx(100.0)
    assert target.tcp.z == pytest.approx(-290.6)
    assert target.tcp.r == pytest.approx(90.0)
    assert target.tcp.source == "calibrated"


def test_verification_target_is_canonical_object_pose_not_fingertip_tcp() -> None:
    catalog = load_catalog()
    product = catalog.product(4)
    step = product.steps[0]
    provider = _provider(base=_pose(400.0, 100.0, -320.0, 90.0))

    motion_target = provider.target_for(product, _item("base-4", "F04"), step, observed_at_s=10.0)
    object_target = provider.verification_target_for(
        product,
        _item("base-4", "F04"),
        step,
        observed_at_s=10.0,
    )

    assert object_target.tcp.x == pytest.approx(400.0)
    assert object_target.tcp.y == pytest.approx(100.0)
    assert object_target.tcp.z == pytest.approx(-320.0)
    assert object_target.tcp.r == pytest.approx(90.0)
    assert motion_target.tcp.y == pytest.approx(124.06)
    assert motion_target.tcp.z == pytest.approx(-309.6)


def test_rear_guard_verification_target_preserves_nonidentity_mesh_frame() -> None:
    catalog = load_catalog()
    product = catalog.product(1)
    step = product.steps[3]
    provider = _provider(product_id=1, base=_pose(400.0, 100.0, -320.0, 90.0))

    object_target = provider.verification_target_for(
        product,
        _item("base-4", "F01"),
        step,
        observed_at_s=10.0,
    )

    assert object_target.tcp.x == pytest.approx(382.5)
    assert object_target.tcp.y == pytest.approx(100.0)
    assert object_target.tcp.z == pytest.approx(-316.0)
    assert object_target.tcp.r == pytest.approx(90.0)


def test_rear_guard_target_composes_installed_and_mesh_frame_transforms() -> None:
    catalog = load_catalog()
    product = catalog.product(1)
    step = product.steps[3]
    provider = _provider(product_id=1, base=_pose(400.0, 100.0, -320.0, 90.0))

    target = provider.target_for(product, _item("base-4", "F01"), step, observed_at_s=10.0)

    assert target.tcp.x == pytest.approx(382.5)
    assert target.tcp.y == pytest.approx(100.0)
    assert target.tcp.z == pytest.approx(-288.1)
    assert target.tcp.r == pytest.approx(90.0)


def test_foundationpose_canonical_origin_is_shifted_to_t29_fingertip_tcp() -> None:
    provider = _provider(product_id=4)
    part = _item("part-4", "M4_01")
    detection = {
        "instance_id": "part-4",
        "object_id": "M4_01",
        "pose_source": "foundationpose",
        "resting_side": "upright",
        "pose": {"x": 250.0, "y": -50.0, "z": -300.0, "yaw": 90.0},
    }

    grasp = provider.grasp_from_foundationpose(part, detection, observed_at_s=10.0)

    assert grasp.tcp.x == pytest.approx(250.0)
    assert grasp.tcp.y == pytest.approx(-25.94)
    assert grasp.tcp.z == pytest.approx(-289.6)
    assert grasp.tcp.r == pytest.approx(90.0)


def test_foundationpose_fallen_pose_without_full_6d_grasp_fails_closed() -> None:
    provider = _provider(product_id=4)
    part = _item("part-4", "M4_01")
    detection = {
        "instance_id": "part-4",
        "object_id": "M4_01",
        "pose_source": "foundationpose",
        "resting_side": "side",
        "pose": {"x": 250.0, "y": -50.0, "z": -300.0, "yaw": 90.0},
    }

    with pytest.raises(CalibratedPoseError, match="full 6D grasp"):
        provider.grasp_from_foundationpose(part, detection, observed_at_s=10.0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"instance_id": "another-part"}, "identity"),
        ({"object_id": "M4_02"}, "identity"),
        ({"pose_source": "detector"}, "FoundationPose"),
        ({"pose": {"x": math.nan, "y": 0.0, "z": 0.0, "yaw": 0.0}}, "finite"),
    ],
)
def test_foundationpose_grasp_rejects_untrusted_identity_or_pose(
    change: dict[str, object],
    message: str,
) -> None:
    provider = _provider(product_id=4)
    part = _item("part-4", "M4_01")
    detection = {
        "instance_id": "part-4",
        "object_id": "M4_01",
        "pose_source": "foundationpose",
        "resting_side": "upright",
        "pose": {"x": 250.0, "y": -50.0, "z": -300.0, "yaw": 90.0},
        **change,
    }

    with pytest.raises(CalibratedPoseError, match=message):
        provider.grasp_from_foundationpose(part, detection, observed_at_s=10.0)


def test_base_pose_exposes_the_same_current_measurement_used_by_target() -> None:
    measured = _pose(400.0, 100.0, -320.0, 90.0)
    provider = _provider(base=measured)

    assert provider.base_for(_item("base-4", "F04"), observed_at_s=10.0) == measured


def test_grasp_returns_the_current_measured_pose_without_mutating_it() -> None:
    measured = _pose(250.0, -50.0, -300.0, -35.0)
    provider = _provider(base=_pose(400.0, 100.0, -320.0, 0.0), part=measured)

    assert provider.grasp_for(_item("part-4", "M4_04"), observed_at_s=10.0) == measured


def test_registry_snapshot_replaces_stale_measurements() -> None:
    registry = MeasuredPoseRegistry(
        {"part-4": _pose(250.0, -50.0, -300.0, 0.0)},
        observed_at_s=10.0,
        complete=True,
    )
    registry.replace_snapshot(
        {"part-5": _pose(260.0, -40.0, -300.0, 0.0)},
        observed_at_s=11.0,
        complete=True,
    )

    with pytest.raises(CalibratedPoseError, match="part-4.*missing"):
        registry.pose_for("part-4", expected_at_s=11.0)
    assert registry.pose_for("part-5", expected_at_s=11.0).tcp.x == 260.0


def test_registry_rejects_stale_snapshot_for_current_observation() -> None:
    registry = MeasuredPoseRegistry(
        {"part-4": _pose(250.0, -50.0, -300.0, 0.0)},
        observed_at_s=10.0,
        complete=True,
    )

    with pytest.raises(CalibratedPoseError, match="current observation"):
        registry.pose_for("part-4", expected_at_s=11.0)


def test_registry_rejects_incomplete_snapshot_even_when_timestamp_matches() -> None:
    provider = _provider(part=_pose(250.0, -50.0, -300.0, 0.0), complete=False)

    with pytest.raises(CalibratedPoseError, match="explicitly complete"):
        provider.grasp_for(_item("part-4", "M4_04"), observed_at_s=10.0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"source": "simulator_ground_truth"}, "calibrated"),
        ({"frame": "base"}, "UF0"),
        ({"x": math.nan}, "finite"),
    ],
)
def test_provider_rejects_untrusted_or_nonfinite_measurements(change: dict[str, object], message: str) -> None:
    part = _pose(250.0, -50.0, -300.0, 0.0)
    part = replace(part, tcp=replace(part.tcp, **change))
    provider = _provider(base=_pose(400.0, 100.0, -320.0, 0.0), part=part)

    with pytest.raises(CalibratedPoseError, match=message):
        provider.grasp_for(_item("part-4", "M4_04"), observed_at_s=10.0)


@pytest.mark.parametrize(
    ("part", "message"),
    [
        (replace(_pose(250.0, -50.0, -300.0, 0.0), resting_state=RestingState.FALLEN), "installed"),
        (_pose(250.0, -50.0, -300.0, 0.0, up_face="installed:-z"), "up-face"),
    ],
)
def test_provider_rejects_fallen_or_wrong_up_face(part: OrientedPartPose, message: str) -> None:
    provider = _provider(base=_pose(400.0, 100.0, -320.0, 0.0), part=part)

    with pytest.raises(CalibratedPoseError, match=message):
        provider.grasp_for(_item("part-4", "M4_04"), observed_at_s=10.0)


def test_provider_rejects_missing_base_or_part_measurement() -> None:
    provider = _provider()

    with pytest.raises(CalibratedPoseError, match="part-4.*missing"):
        provider.grasp_for(_item("part-4", "M4_04"), observed_at_s=10.0)
    with pytest.raises(CalibratedPoseError, match="base-4.*missing"):
        provider.target_for(
            load_catalog().product(4),
            _item("base-4", "F04"),
            load_catalog().product(4).steps[0],
            observed_at_s=10.0,
        )


def test_provider_rejects_wrong_base_family_and_step_ownership() -> None:
    catalog = load_catalog()
    provider = _provider(base=_pose(400.0, 100.0, -320.0, 0.0), part=_pose(250.0, -50.0, -300.0, 0.0))

    with pytest.raises(CalibratedPoseError, match="base type"):
        provider.target_for(
            catalog.product(4),
            _item("base-4", "F01"),
            catalog.product(4).steps[0],
            observed_at_s=10.0,
        )
    with pytest.raises(CalibratedPoseError, match="recipe step"):
        provider.target_for(
            catalog.product(4),
            _item("base-4", "F04"),
            catalog.product(1).steps[0],
            observed_at_s=10.0,
        )
