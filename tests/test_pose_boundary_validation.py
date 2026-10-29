"""Nested pose values must fail as typed boundary errors."""

from __future__ import annotations

import pytest

from assembly.calibrated_poses import (
    ApprovedInstalledOrientation,
    CalibratedPoseError,
    FoundationPoseFrameBoundary,
    MeasuredPoseRegistry,
    MeasuredRecipePoses,
)
from assembly.catalog import load_catalog
from assembly.inventory import Inventory
from assembly.motion import (
    AbsolutePoseUF0,
    MotionConfig,
    MotionInputError,
    OrientedPartPose,
    RestingState,
    Waypoint,
    compile_pick_insert,
)
from assembly.planner import PlannerPoseError, RecipePlanner


def _valid_pose() -> OrientedPartPose:
    tcp = AbsolutePoseUF0(200.0, 0.0, -300.0, -180.0, 0.0, 0.0, "UF0", "calibrated")
    return OrientedPartPose(tcp, RestingState.INSTALLED, "installed:+z")


def _broken_pose(kind: str) -> OrientedPartPose:
    if kind == "tcp":
        return OrientedPartPose(None, RestingState.INSTALLED, "installed:+z")  # type: ignore[arg-type]
    return OrientedPartPose(_valid_pose().tcp, RestingState.INSTALLED, 7)  # type: ignore[arg-type]


def _inventory() -> Inventory:
    inventory = Inventory(load_catalog())
    inventory.add("base-1", "F01")
    for step in range(1, 5):
        inventory.add(f"p1-{step}", f"M1_{step:02d}")
    inventory.reserve_product(1)
    return inventory


class BrokenProvider:
    def __init__(self, broken: OrientedPartPose) -> None:
        self.broken = broken

    def grasp_for(self, part, *, observed_at_s: float) -> OrientedPartPose:
        return self.broken

    def base_for(self, base, *, observed_at_s: float) -> OrientedPartPose:
        return _valid_pose()

    def target_for(self, product, base, step, *, observed_at_s: float) -> OrientedPartPose:
        return _valid_pose()

    def verification_target_for(self, product, base, step, *, observed_at_s: float) -> OrientedPartPose:
        return _valid_pose()


@pytest.mark.parametrize("kind", ["tcp", "up_face"])
def test_planner_rejects_malformed_nested_pose(kind: str) -> None:
    planner = RecipePlanner(load_catalog(), BrokenProvider(_broken_pose(kind)))

    with pytest.raises(PlannerPoseError):
        planner.next_step(_inventory(), observed_at_s=1.0)


@pytest.mark.parametrize("kind", ["tcp", "up_face"])
def test_motion_rejects_malformed_nested_pose(kind: str) -> None:
    current = Waypoint(200.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0)
    config = MotionConfig(10.0, 10.0, 10.0, 10.0, 20.0, 10.0, 10, -180.0, "test fixture")

    with pytest.raises(MotionInputError):
        compile_pick_insert(current, _broken_pose(kind), _valid_pose(), gripper_roll_deg=0.0, config=config)


@pytest.mark.parametrize("kind", ["tcp", "up_face"])
def test_calibrated_provider_rejects_malformed_nested_measurement(kind: str) -> None:
    inventory = _inventory()
    registry = MeasuredPoseRegistry({"p1-1": _broken_pose(kind)}, observed_at_s=1.0, complete=True)
    provider = MeasuredRecipePoses(
        catalog=load_catalog(),
        measurements=registry,
        installed_orientations={"M1_01": ApprovedInstalledOrientation(-180.0, 0.0, 0.0, "installed:+z")},
        orientation_tolerance_deg=0.5,
        foundationpose_boundary=FoundationPoseFrameBoundary("installed_local", "test fixture"),
    )

    with pytest.raises(CalibratedPoseError):
        provider.grasp_for(inventory.item("p1-1"), observed_at_s=1.0)
