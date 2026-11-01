"""Checks-only scene-safety regression tests.
Authored by Shritej Gunda.
"""

import pytest

from assembly.motion import MotionChunk, MotionPhase, Waypoint
from assembly.safety import (
    AABB,
    CartesianSpeedViolation,
    CollisionViolation,
    EmptyPlanViolation,
    MotionOwner,
    OfficialActionGuardViolation,
    SafetyConfigurationViolation,
    SafetySupervisor,
)
from robot.cell.saturday_scene_safety import (
    LayoutWorkspaceViolation,
    SaturdaySceneSafetySupervisor,
)


def _point(
    x: float = 300.0,
    y: float = 0.0,
    z: float = -200.0,
    w: float = -180.0,
    p: float = 0.0,
    r: float = 0.0,
) -> Waypoint:
    return Waypoint(x, y, z, w, p, r, 0.0)


def _plan(target: Waypoint) -> tuple[MotionChunk, ...]:
    return (MotionChunk(MotionPhase.CARRY_YAW, (target,)),)


OWNER = MotionOwner("part-1", "M1_01", False)


def test_supervisor_is_explicitly_preliminary_and_never_claims_full_geometry() -> None:
    supervisor = SaturdaySceneSafetySupervisor()

    assert isinstance(supervisor, SafetySupervisor)
    assert supervisor.status == "PRELIMINARY"
    assert supervisor.checks_only is True
    assert supervisor.reachability_capabilities.full_robot_kinematics is False
    assert supervisor.collision_capabilities.robot_links is False
    assert supervisor.collision_capabilities.held_part_geometry is False


def test_one_safe_row_passes_all_preliminary_checks() -> None:
    supervisor = SaturdaySceneSafetySupervisor()
    plan = _plan(_point(x=310.0))

    assert supervisor.validate(
        plan,
        start=_point(),
        fixtures=(),
        parts=(),
        owner=OWNER,
    ) == plan


def test_plan_requires_nonempty_exactly_one_row_chunks() -> None:
    supervisor = SaturdaySceneSafetySupervisor()

    with pytest.raises(EmptyPlanViolation, match="non-empty"):
        supervisor.validate((), start=_point(), fixtures=(), parts=(), owner=OWNER)
    with pytest.raises(SafetyConfigurationViolation, match="exactly one waypoint"):
        supervisor.validate(
            (MotionChunk(MotionPhase.CARRY_YAW, (_point(x=301.0), _point(x=302.0))),),
            start=_point(),
            fixtures=(),
            parts=(),
            owner=OWNER,
        )


@pytest.mark.parametrize(
    "target",
    [
        _point(x=float("nan")),
        _point(w=-173.9),
    ],
)
def test_official_guard_rejects_nonfinite_rows_and_tool_tilt_over_six_degrees(
    target: Waypoint,
) -> None:
    supervisor = SaturdaySceneSafetySupervisor()

    with pytest.raises(OfficialActionGuardViolation):
        supervisor.validate(
            _plan(target),
            start=_point(),
            fixtures=(),
            parts=(),
            owner=OWNER,
        )


def test_layout_workspace_includes_reject_zone_and_refuses_points_outside_its_envelope() -> None:
    supervisor = SaturdaySceneSafetySupervisor()
    reject_start = _point(x=200.0, y=340.0)
    reject_plan = _plan(_point(x=210.0, y=340.0))

    assert supervisor.validate(
        reject_plan,
        start=reject_start,
        fixtures=(),
        parts=(),
        owner=OWNER,
    ) == reject_plan
    assert supervisor.table_z_mm == -335.0
    assert supervisor.workspace_xy_mm == ((190.0, 470.0), (-380.0, 390.0))
    assert supervisor.layout_source.endswith("sdg/configs/cell_layout.yaml")

    with pytest.raises(LayoutWorkspaceViolation, match="outside layout workspace"):
        supervisor.validate(
            _plan(_point(x=189.0, y=340.0)),
            start=reject_start,
            fixtures=(),
            parts=(),
            owner=OWNER,
        )


@pytest.mark.parametrize(
    ("start", "target"),
    [
        (_point(), _point(x=330.1)),
        (_point(), _point(r=12.1)),
        (_point(z=-330.0), _point(x=302.1, z=-330.0)),
    ],
)
def test_each_row_obeys_preliminary_free_and_slow_zone_step_limits(
    start: Waypoint,
    target: Waypoint,
) -> None:
    supervisor = SaturdaySceneSafetySupervisor()

    with pytest.raises(CartesianSpeedViolation, match="PRELIMINARY"):
        supervisor.validate(
            _plan(target),
            start=start,
            fixtures=(),
            parts=(),
            owner=OWNER,
        )


@pytest.mark.parametrize("collection", ["fixtures", "parts"])
def test_tcp_point_collision_refuses_fixture_and_part_aabbs(collection: str) -> None:
    supervisor = SaturdaySceneSafetySupervisor()
    obstacle = AABB("occupied", (309.0, -1.0, -201.0), (311.0, 1.0, -199.0))
    arguments = {"fixtures": (), "parts": ()}
    arguments[collection] = (obstacle,)

    with pytest.raises(CollisionViolation, match="TCP point.*occupied"):
        supervisor.validate(
            _plan(_point(x=310.0)),
            start=_point(),
            owner=OWNER,
            **arguments,
        )
