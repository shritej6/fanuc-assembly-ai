"""Fail-closed safety supervision for compiled absolute TCP motion."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace

import pytest

from assembly.motion import MotionChunk, MotionPhase, Waypoint
from assembly.safety import (
    AABB,
    AABBCollisionChecker,
    CartesianSpeedViolation,
    CollisionCapabilities,
    CollisionChecker,
    MotionOwner,
    OfficialActionGuardViolation,
    ReachabilityCapabilities,
    ReachabilityChecker,
    SafetyConfig,
    SafetyConfigurationViolation,
    SafetySupervisor,
    SafetyViolation,
)
from robot.adapter import ActionBoundsError, ActionValueError


class AlwaysReachable:
    """Deterministic in-process reachability checker for supervisor tests."""

    capabilities = ReachabilityCapabilities(full_robot_kinematics=True)

    def is_reachable(self, pose: tuple[float, float, float, float, float, float]) -> bool:
        return True


class NeverReachable:
    """A deterministic negative kinematic result."""

    capabilities = ReachabilityCapabilities(full_robot_kinematics=False)

    def is_reachable(self, pose: tuple[float, float, float, float, float, float]) -> bool:
        return False


class BrokenReachability:
    """A checker whose backing service is unavailable."""

    capabilities = ReachabilityCapabilities(full_robot_kinematics=False)

    def is_reachable(self, pose: tuple[float, float, float, float, float, float]) -> bool:
        raise RuntimeError("IK service offline")


class RejectingCollisionChecker:
    """Injected deterministic collision result independent of AABB geometry."""

    capabilities = CollisionCapabilities(robot_links=True, held_part_geometry=True)

    def first_collision(
        self,
        start: Waypoint,
        end: Waypoint,
        obstacles: Sequence[AABB],
        sweep_radius_mm: float,
        owner: MotionOwner,
    ) -> str | None:
        del start, end, obstacles, sweep_radius_mm, owner
        return "injected-obstacle"


class BrokenCollisionChecker:
    """A collision checker whose backing service is unavailable."""

    capabilities = CollisionCapabilities(robot_links=True, held_part_geometry=True)

    def first_collision(
        self,
        start: Waypoint,
        end: Waypoint,
        obstacles: Sequence[AABB],
        sweep_radius_mm: float,
        owner: MotionOwner,
    ) -> str | None:
        del start, end, obstacles, sweep_radius_mm, owner
        raise RuntimeError("collision service offline")


class WrappedAABBCollisionChecker:
    """A wrapper cannot turn a TCP-only sweep into production geometry."""

    capabilities = CollisionCapabilities(robot_links=False, held_part_geometry=False)

    def first_collision(self, start, end, obstacles, sweep_radius_mm, owner):
        return AABBCollisionChecker().first_collision(start, end, obstacles, sweep_radius_mm, owner)


class RecordingCollisionChecker:
    """Production-capable test double that records the geometry owner."""

    capabilities = CollisionCapabilities(robot_links=True, held_part_geometry=True)

    def __init__(self) -> None:
        self.owner: MotionOwner | None = None

    def first_collision(self, start, end, obstacles, sweep_radius_mm, owner):
        del start, end, obstacles, sweep_radius_mm
        self.owner = owner


class UndeclaredCollisionChecker:
    def first_collision(self, start, end, obstacles, sweep_radius_mm, owner):
        del start, end, obstacles, sweep_radius_mm, owner


_DEFAULT_REACHABILITY = object()


def _waypoint(x: float, y: float, z: float, *, yaw: float = 0.0) -> Waypoint:
    return Waypoint(x, y, z, -180.0, 0.0, yaw, 0.0)


def _owner(*, attached: bool = False) -> MotionOwner:
    return MotionOwner("part-1", "M1_01", attached)


def _config() -> SafetyConfig:
    return SafetyConfig(
        dt_s=0.1,
        table_z_mm=-335.0,
        min_table_clearance_mm=1.0,
        expected_w_deg=-180.0,
        expected_p_deg=0.0,
        orientation_tolerance_deg=0.5,
        sweep_radius_mm=2.0,
        max_cartesian_speed_mm_s=10000.0,
        max_cartesian_speed_source="test fixture",
    )


def _supervisor(*, reachability=_DEFAULT_REACHABILITY) -> SafetySupervisor:
    return SafetySupervisor(
        config=_config(),
        reachability=AlwaysReachable() if reachability is _DEFAULT_REACHABILITY else reachability,
        collision=AABBCollisionChecker(),
        allow_tcp_sweep_only_for_mock=True,
    )


def test_supervisor_accepts_guarded_reachable_clear_plan() -> None:
    """A valid plan should survive every gate without mutation."""
    plan = (
        MotionChunk(MotionPhase.APPROACH, (_waypoint(220.0, 0.0, -250.0),)),
        MotionChunk(MotionPhase.DESCEND, (_waypoint(220.0, 0.0, -280.0),)),
    )
    start = _waypoint(200.0, 0.0, -250.0)
    supervisor = _supervisor()

    assert supervisor.validate(plan, start=start, fixtures=(), parts=(), owner=_owner()) == plan


def test_supervisor_rejects_segment_above_configured_cartesian_speed() -> None:
    plan = (MotionChunk(MotionPhase.APPROACH, (_waypoint(220.0, 0.0, -250.0),)),)
    supervisor = SafetySupervisor(
        config=replace(_config(), max_cartesian_speed_mm_s=100.0),
        reachability=AlwaysReachable(),
        collision=AABBCollisionChecker(),
        allow_tcp_sweep_only_for_mock=True,
    )

    with pytest.raises(CartesianSpeedViolation, match="speed"):
        supervisor.validate(
            plan,
            start=_waypoint(200.0, 0.0, -250.0),
            fixtures=(),
            parts=(),
            owner=_owner(),
        )


def test_supervisor_rejects_empty_plan() -> None:
    """An empty plan is not an executable guarded action and must fail closed."""
    supervisor = _supervisor()

    with pytest.raises(SafetyViolation, match="empty"):
        supervisor.validate((), start=_waypoint(200.0, 0.0, -250.0), fixtures=(), parts=(), owner=_owner())


@pytest.mark.parametrize(
    ("bad", "error"),
    [
        (_waypoint(149.9, 0.0, -250.0), ActionBoundsError),
        (_waypoint(math.nan, 0.0, -250.0), ActionValueError),
    ],
)
def test_supervisor_delegates_every_tcp_row_to_t05_guard(bad: Waypoint, error: type[ValueError]) -> None:
    """The supervisor must not clamp or weaken T05's finite/bounds checks."""
    plan = (
        MotionChunk(MotionPhase.APPROACH, (_waypoint(220.0, 0.0, -250.0),)),
        MotionChunk(MotionPhase.DESCEND, (bad,)),
    )
    supervisor = _supervisor()

    with pytest.raises(OfficialActionGuardViolation) as raised:
        supervisor.validate(
            plan, start=_waypoint(200.0, 0.0, -250.0), fixtures=(), parts=(), owner=_owner()
        )
    assert isinstance(raised.value.__cause__, error)


@pytest.mark.parametrize(
    ("waypoint", "message"),
    [
        (Waypoint(220.0, 0.0, -250.0, -170.0, 0.0, 0.0, 0.0), "orientation"),
        (Waypoint(220.0, 0.0, -250.0, -180.0, 2.0, 0.0, 0.0), "orientation"),
        (_waypoint(220.0, 0.0, -331.0), "table clearance"),
    ],
)
def test_supervisor_rejects_wrong_tool_orientation_and_table_clearance(waypoint: Waypoint, message: str) -> None:
    """Downward UTOOL1 and explicit clearance are independent fail-closed gates."""
    plan = (MotionChunk(MotionPhase.APPROACH, (waypoint,)),)
    config = replace(_config(), min_table_clearance_mm=5.0)
    supervisor = SafetySupervisor(
        config=config,
        reachability=AlwaysReachable(),
        collision=AABBCollisionChecker(),
        allow_tcp_sweep_only_for_mock=True,
    )

    with pytest.raises(SafetyViolation, match=message):
        supervisor.validate(
            plan, start=_waypoint(200.0, 0.0, -250.0), fixtures=(), parts=(), owner=_owner()
        )


@pytest.mark.parametrize(
    ("config", "waypoint"),
    [
        (replace(_config(), expected_w_deg=0.0), Waypoint(220.0, 0.0, -250.0, 0.0, 0.0, 0.0, 0.0)),
        (replace(_config(), expected_p_deg=10.0), Waypoint(220.0, 0.0, -250.0, -180.0, 10.0, 0.0, 0.0)),
        (replace(_config(), orientation_tolerance_deg=180.0), Waypoint(220.0, 0.0, -250.0, 0.0, 0.0, 0.0, 0.0)),
    ],
)
def test_supervisor_config_cannot_redefine_or_disable_tool_down(config: SafetyConfig, waypoint: Waypoint) -> None:
    """Configuration must not turn an upright or sideways tool into an accepted pose."""
    plan = (MotionChunk(MotionPhase.APPROACH, (waypoint,)),)
    supervisor = SafetySupervisor(
        config=config,
        reachability=AlwaysReachable(),
        collision=AABBCollisionChecker(),
        allow_tcp_sweep_only_for_mock=True,
    )

    with pytest.raises(SafetyConfigurationViolation, match="orientation"):
        supervisor.validate(plan, start=waypoint, fixtures=(), parts=(), owner=_owner())


@pytest.mark.parametrize("reachability", [None, NeverReachable(), BrokenReachability()])
def test_supervisor_fails_closed_without_positive_reachability(
    reachability: ReachabilityChecker | None,
) -> None:
    """Missing, negative, and failed IK results must all reject motion."""
    plan = (MotionChunk(MotionPhase.APPROACH, (_waypoint(220.0, 0.0, -250.0),)),)
    supervisor = _supervisor(reachability=reachability)

    with pytest.raises(SafetyViolation, match="reachab"):
        supervisor.validate(
            plan, start=_waypoint(200.0, 0.0, -250.0), fixtures=(), parts=(), owner=_owner()
        )


def _box(name: str, minimum: tuple[float, float, float], maximum: tuple[float, float, float]) -> AABB:
    return AABB(name=name, minimum=minimum, maximum=maximum)


def test_supervisor_checks_current_tcp_to_first_approach_segment() -> None:
    """An obstacle between observed TCP and approach is caught even when endpoints are clear."""
    plan = (MotionChunk(MotionPhase.APPROACH, (_waypoint(230.0, 0.0, -250.0),)),)
    fixture = _box("fixture-start", (210.0, -3.0, -253.0), (215.0, 3.0, -247.0))
    config = replace(_config(), sweep_radius_mm=0.0)
    supervisor = SafetySupervisor(
        config=config,
        reachability=AlwaysReachable(),
        collision=AABBCollisionChecker(),
        allow_tcp_sweep_only_for_mock=True,
    )

    with pytest.raises(SafetyViolation, match="fixture-start"):
        supervisor.validate(
            plan,
            start=_waypoint(200.0, 0.0, -250.0),
            fixtures=(fixture,),
            parts=(),
            owner=_owner(),
        )


def test_supervisor_checks_swept_collision_across_chunk_boundary() -> None:
    """Flattening each chunk separately must not omit the segment between chunks."""
    plan = (
        MotionChunk(MotionPhase.LIFT, (_waypoint(240.0, 0.0, -250.0),)),
        MotionChunk(MotionPhase.CARRY_YAW, (_waypoint(300.0, 0.0, -250.0),)),
    )
    part = _box("part-between-chunks", (265.0, -3.0, -253.0), (275.0, 3.0, -247.0))
    config = replace(_config(), sweep_radius_mm=0.0)
    supervisor = SafetySupervisor(
        config=config,
        reachability=AlwaysReachable(),
        collision=AABBCollisionChecker(),
        allow_tcp_sweep_only_for_mock=True,
    )

    with pytest.raises(SafetyViolation, match="part-between-chunks"):
        supervisor.validate(
            plan,
            start=_waypoint(220.0, 0.0, -250.0),
            fixtures=(),
            parts=(part,),
            owner=_owner(attached=True),
        )


def test_supervisor_checks_swept_radius_not_only_centerline() -> None:
    """A TCP sweep that grazes an AABB is a collision even if its centerline misses."""
    plan = (MotionChunk(MotionPhase.CARRY_YAW, (_waypoint(260.0, 0.0, -250.0),)),)
    fixture = _box("grazed-fixture", (230.0, 1.5, -252.0), (240.0, 3.0, -248.0))
    supervisor = _supervisor()

    with pytest.raises(SafetyViolation, match="grazed-fixture"):
        supervisor.validate(
            plan,
            start=_waypoint(220.0, 0.0, -250.0),
            fixtures=(fixture,),
            parts=(),
            owner=_owner(attached=True),
        )


@pytest.mark.parametrize("checker", [RejectingCollisionChecker(), BrokenCollisionChecker()])
def test_supervisor_fails_closed_on_injected_collision_result(checker: CollisionChecker) -> None:
    """Injected collision decisions and checker outages both reject the plan."""
    plan = (MotionChunk(MotionPhase.APPROACH, (_waypoint(220.0, 0.0, -250.0),)),)
    supervisor = SafetySupervisor(
        config=_config(),
        reachability=AlwaysReachable(),
        collision=checker,
    )

    with pytest.raises(SafetyViolation, match="collision"):
        supervisor.validate(
            plan, start=_waypoint(200.0, 0.0, -250.0), fixtures=(), parts=(), owner=_owner()
        )


@pytest.mark.parametrize(
    "obstacle",
    [
        lambda: _box("nonfinite", (math.nan, 0.0, 0.0), (1.0, 1.0, 1.0)),
        lambda: _box("inverted", (2.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    ],
)
def test_supervisor_rejects_invalid_obstacle_definition(obstacle: Callable[[], AABB]) -> None:
    """Malformed scene geometry cannot silently disable collision checking."""
    plan = (MotionChunk(MotionPhase.APPROACH, (_waypoint(220.0, 0.0, -250.0),)),)
    supervisor = _supervisor()

    with pytest.raises(SafetyViolation, match="obstacle"):
        supervisor.validate(
            plan,
            start=_waypoint(200.0, 0.0, -250.0),
            fixtures=(obstacle(),),
            parts=(),
            owner=_owner(),
        )


def test_supervisor_requires_explicit_collision_checker() -> None:
    with pytest.raises(SafetyConfigurationViolation, match="explicit collision"):
        SafetySupervisor(config=_config(), reachability=AlwaysReachable(), collision=None)


@pytest.mark.parametrize("checker", [AABBCollisionChecker(), WrappedAABBCollisionChecker()])
def test_tcp_only_collision_checker_requires_explicit_mock_opt_in(checker: CollisionChecker) -> None:
    with pytest.raises(SafetyConfigurationViolation, match="held parts"):
        SafetySupervisor(
            config=_config(),
            reachability=AlwaysReachable(),
            collision=checker,
        )


def test_checker_must_declare_capabilities_even_in_mock_mode() -> None:
    with pytest.raises(SafetyConfigurationViolation, match="declare capabilities"):
        SafetySupervisor(
            config=_config(),
            reachability=AlwaysReachable(),
            collision=UndeclaredCollisionChecker(),
            allow_tcp_sweep_only_for_mock=True,
        )


def test_collision_checker_receives_exact_motion_owner_context() -> None:
    checker = RecordingCollisionChecker()
    supervisor = SafetySupervisor(config=_config(), reachability=AlwaysReachable(), collision=checker)
    plan = (MotionChunk(MotionPhase.CARRY_YAW, (_waypoint(260.0, 0.0, -250.0),)),)
    owner = _owner(attached=True)

    supervisor.validate(
        plan,
        start=_waypoint(220.0, 0.0, -250.0),
        fixtures=(),
        parts=(),
        owner=owner,
    )

    assert checker.owner == owner
