"""Production safety providers fail closed until every sourced model is present."""

from __future__ import annotations

import importlib.util
import random
from dataclasses import dataclass

import pytest

from assembly.motion import Waypoint
from assembly.safety import AABB, MotionOwner


def test_cell_runtime_safety_module_is_available() -> None:
    assert importlib.util.find_spec("assembly.cell_runtime_safety") is not None


def test_sdg_reachability_uses_sourced_full_ik_result() -> None:
    from assembly.cell_runtime_safety import SDGReachabilityChecker

    calls: list[list[float]] = []

    def solve(pose: list[float]) -> bool:
        calls.append(pose)
        return pose[0] == 410.0

    checker = SDGReachabilityChecker(solve, source="sdg/wrist_ik.py:reachable")

    assert checker.capabilities.full_robot_kinematics is True
    assert checker.is_reachable((410.0, 20.0, -200.0, -180.0, 0.0, 45.0)) is True
    assert calls == [[410.0, 20.0, -200.0, -180.0, 0.0, 45.0]]


@pytest.mark.parametrize("source", ["", "   "])
def test_production_safety_rejects_unrecorded_sources(source: str) -> None:
    from assembly.cell_runtime_safety import ProductionSafetyConfigurationError, SDGReachabilityChecker

    with pytest.raises(ProductionSafetyConfigurationError, match="source"):
        SDGReachabilityChecker(lambda pose: True, source=source)


def test_reachability_rejects_non_boolean_solver_result() -> None:
    from assembly.cell_runtime_safety import ProductionSafetyUnavailable, SDGReachabilityChecker

    checker = SDGReachabilityChecker(lambda pose: 1, source="measured solver")

    with pytest.raises(ProductionSafetyUnavailable, match="boolean"):
        checker.is_reachable((400.0, 0.0, -200.0, -180.0, 0.0, 0.0))


@dataclass
class _Geometry:
    source: str = "CAD robot links + part STL + T29 grasp frames"
    models_robot_links: bool = True
    models_held_parts: bool = True
    links: tuple[AABB, ...] = (AABB("J1_link", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),)
    held: AABB | None = None

    def swept_robot_links(self, start: Waypoint, end: Waypoint) -> tuple[AABB, ...]:
        del start, end
        return self.links

    def swept_held_part(self, start: Waypoint, end: Waypoint, owner: MotionOwner) -> AABB:
        del start, end, owner
        if self.held is None:
            raise RuntimeError("held geometry unavailable")
        return self.held


@dataclass
class _RobotLinks:
    source: str = "ER-4iA CAD link meshes + full IK/FK"

    def swept_bounds(self, start: Waypoint, end: Waypoint) -> tuple[AABB, ...]:
        del start, end
        return (AABB("J2_link", (200.0, -10.0, -200.0), (250.0, 10.0, -150.0)),)


@dataclass
class _HeldParts:
    source: str = "cad/stl + T29 grasp frames"

    def swept_bounds(self, start: Waypoint, end: Waypoint, owner: MotionOwner) -> AABB:
        del start, end
        return AABB(f"held:{owner.object_id}", (300.0, -5.0, -250.0), (310.0, 5.0, -230.0))


def _point(x: float, y: float, z: float) -> Waypoint:
    return Waypoint(x, y, z, -180.0, 0.0, 0.0, 1.0)


def test_cad_collision_checker_covers_robot_links_and_held_part() -> None:
    from assembly.cell_runtime_safety import CADCollisionChecker

    obstacle = AABB("base", (300.0, -20.0, -334.0), (360.0, 20.0, -250.0))
    owner = MotionOwner("part-1", "M1_01", True)
    link_hit = _Geometry(links=(AABB("J4_link", (320.0, 0.0, -300.0), (330.0, 10.0, -280.0)),))
    held_hit = _Geometry(held=AABB("held:M1_01", (330.0, 0.0, -300.0), (340.0, 10.0, -280.0)))

    assert CADCollisionChecker(link_hit).first_collision(
        _point(200.0, 0.0, -200.0), _point(210.0, 0.0, -200.0), (obstacle,), 0.0, owner
    ) == "robot-link:J4_link with base"
    assert CADCollisionChecker(held_hit).first_collision(
        _point(200.0, 0.0, -200.0), _point(210.0, 0.0, -200.0), (obstacle,), 0.0, owner
    ) == "held:M1_01 with base"


def test_collision_checker_ignores_the_moving_instances_old_box() -> None:
    from assembly.cell_runtime_safety import CADCollisionChecker

    owner = MotionOwner("part-1", "M1_01", True)
    geometry = _Geometry(held=AABB("held:M1_01", (330.0, 0.0, -300.0), (340.0, 10.0, -280.0)))
    old_box = AABB("part-1", (330.0, 0.0, -300.0), (340.0, 10.0, -280.0))

    assert CADCollisionChecker(geometry).first_collision(
        _point(200.0, 0.0, -200.0), _point(210.0, 0.0, -200.0), (old_box,), 0.0, owner
    ) is None


def test_cad_collision_checker_matches_seeded_random_layouts() -> None:
    from assembly.cell_runtime_safety import CADCollisionChecker

    rng = random.Random(20261101)
    owner = MotionOwner("moving", "M4_04", True)
    held = AABB("held:M4_04", (300.0, -5.0, -250.0), (310.0, 5.0, -230.0))
    checker = CADCollisionChecker(_Geometry(held=held))
    for index in range(40):
        x = rng.uniform(150.0, 500.0)
        obstacle = AABB(f"part-{index}", (x, 40.0, -250.0), (x + 8.0, 48.0, -230.0))
        assert checker.first_collision(
            _point(300.0, 0.0, -220.0), _point(310.0, 0.0, -220.0), (obstacle,), 0.0, owner
        ) is None


def test_collision_checker_refuses_partial_geometry_capabilities() -> None:
    from assembly.cell_runtime_safety import CADCollisionChecker, ProductionSafetyConfigurationError

    with pytest.raises(ProductionSafetyConfigurationError, match="robot links and held parts"):
        CADCollisionChecker(_Geometry(models_held_parts=False))


def test_collision_checker_fails_closed_when_robot_link_sweep_is_empty() -> None:
    from assembly.cell_runtime_safety import CADCollisionChecker, ProductionSafetyUnavailable

    checker = CADCollisionChecker(_Geometry(links=(), held=AABB("held", (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))))
    owner = MotionOwner("moving", "M1_01", True)

    with pytest.raises(ProductionSafetyUnavailable, match="robot-link"):
        checker.first_collision(_point(300.0, 0.0, -220.0), _point(310.0, 0.0, -220.0), (), 0.0, owner)


def test_sourced_cad_geometry_combines_robot_and_t29_sources() -> None:
    from assembly.cell_runtime_safety import SourcedCADGeometry

    geometry = SourcedCADGeometry(_RobotLinks(), _HeldParts())
    owner = MotionOwner("moving", "M4_04", True)

    assert geometry.models_robot_links is True
    assert geometry.models_held_parts is True
    assert "ER-4iA CAD" in geometry.source
    assert "T29 grasp frames" in geometry.source
    assert geometry.swept_robot_links(_point(300.0, 0.0, -220.0), _point(310.0, 0.0, -220.0))[0].name == "J2_link"
    assert geometry.swept_held_part(
        _point(300.0, 0.0, -220.0), _point(310.0, 0.0, -220.0), owner
    ).name == "held:M4_04"
