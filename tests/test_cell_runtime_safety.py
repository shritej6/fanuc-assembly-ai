"""Production safety providers must prove both capability and reviewed provenance."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

import pytest

from assembly.cell_runtime_safety import (
    CADCollisionChecker,
    ProductionSafetyConfigurationError,
    ProductionSafetyUnavailable,
    ReviewedSafetySource,
    SafetySourceKind,
    SDGReachabilityChecker,
    SourcedCADGeometry,
)
from assembly.motion import Waypoint
from assembly.safety import AABB, MotionOwner, SafetyConfig, SafetyConfigurationViolation, SafetySupervisor

SHA = "a" * 64


def _source(kind: SafetySourceKind, *, sha256: str = SHA, status: str = "REVIEWED") -> ReviewedSafetySource:
    return ReviewedSafetySource(kind, f"review/{kind.value}.json", sha256, status)


def _point(x: float, y: float, z: float, *, closed: bool = True) -> Waypoint:
    return Waypoint(x, y, z, -180.0, 0.0, 0.0, float(closed))


@dataclass
class _RobotLinks:
    provenance: ReviewedSafetySource = field(default_factory=lambda: _source(SafetySourceKind.ROBOT_LINK_SWEEP))
    links: tuple[AABB, ...] = field(
        default_factory=lambda: (AABB("J2", (200.0, -10.0, -200.0), (250.0, 10.0, -150.0)),)
    )

    def swept_bounds(self, start: Waypoint, end: Waypoint) -> tuple[AABB, ...]:
        del start, end
        return self.links


@dataclass
class _HeldPart:
    provenance: ReviewedSafetySource = field(default_factory=lambda: _source(SafetySourceKind.HELD_PART_SWEEP))
    held: AABB = field(default_factory=lambda: AABB("held", (300.0, -5.0, -250.0), (310.0, 5.0, -230.0)))

    def swept_bounds(self, start: Waypoint, end: Waypoint, owner: MotionOwner) -> AABB:
        del start, end, owner
        return self.held


def test_reviewed_source_rejects_unreviewed_or_malformed_evidence() -> None:
    with pytest.raises(ProductionSafetyConfigurationError, match="REVIEWED"):
        _source(SafetySourceKind.FULL_IK_JOINT_LIMITS, status="DRAFT")
    with pytest.raises(ProductionSafetyConfigurationError, match="SHA-256"):
        _source(SafetySourceKind.FULL_IK_JOINT_LIMITS, sha256="not-a-digest")


def test_sdg_reachability_never_claims_physical_authority() -> None:
    calls: list[list[float]] = []

    def solve(pose: list[float]) -> bool:
        calls.append(pose)
        return pose[0] == 410.0

    checker = SDGReachabilityChecker(solve, _source(SafetySourceKind.FULL_IK_JOINT_LIMITS))
    assert checker.capabilities.full_robot_kinematics is False
    assert checker.is_reachable((410.0, 20.0, -200.0, -180.0, 0.0, 45.0)) is True
    assert calls == [[410.0, 20.0, -200.0, -180.0, 0.0, 45.0]]


def test_reachability_rejects_wrong_provenance_kind_and_non_boolean_result() -> None:
    with pytest.raises(ProductionSafetyConfigurationError, match="full_ik_joint_limits"):
        SDGReachabilityChecker(lambda pose: True, _source(SafetySourceKind.ROBOT_LINK_SWEEP))
    checker = SDGReachabilityChecker(
        lambda pose: 1,
        _source(SafetySourceKind.FULL_IK_JOINT_LIMITS),
    )
    with pytest.raises(ProductionSafetyUnavailable, match="boolean"):
        checker.is_reachable((400.0, 0.0, -200.0, -180.0, 0.0, 0.0))


def test_from_sdg_requires_exact_reviewed_file_hash(monkeypatch, tmp_path) -> None:
    source = tmp_path / "wrist_ik.py"
    source.write_text("reviewed implementation\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    class _Module:
        __file__ = str(source)

        @staticmethod
        def reachable(pose):
            return bool(pose)

    monkeypatch.setattr("assembly.cell_runtime_safety.importlib.import_module", lambda name: _Module)
    provenance = ReviewedSafetySource(
        SafetySourceKind.FULL_IK_JOINT_LIMITS,
        "sdg/wrist_ik.py + reviewed physical joint limits",
        digest,
    )
    assert SDGReachabilityChecker.from_sdg(provenance).is_reachable((1.0, 2.0, 3.0, -180.0, 0.0, 0.0))

    wrong = ReviewedSafetySource(
        SafetySourceKind.FULL_IK_JOINT_LIMITS,
        provenance.source,
        "0" * 64,
    )
    with pytest.raises(ProductionSafetyConfigurationError, match="reviewed SHA-256"):
        SDGReachabilityChecker.from_sdg(wrong)


def test_sourced_geometry_requires_independent_reviewed_kinds() -> None:
    with pytest.raises(ProductionSafetyConfigurationError, match="held_part_sweep"):
        SourcedCADGeometry(_RobotLinks(), _HeldPart(provenance=_source(SafetySourceKind.ROBOT_LINK_SWEEP)))

    geometry = SourcedCADGeometry(_RobotLinks(), _HeldPart())
    assert geometry.models_robot_links is False
    assert geometry.models_held_parts is False
    assert CADCollisionChecker(geometry).capabilities.robot_links is False
    assert CADCollisionChecker(geometry).capabilities.held_part_geometry is False


def test_preflight_providers_cannot_arm_the_production_supervisor() -> None:
    reachability = SDGReachabilityChecker(
        lambda pose: True,
        _source(SafetySourceKind.FULL_IK_JOINT_LIMITS),
    )
    geometry = SourcedCADGeometry(_RobotLinks(), _HeldPart())
    collision = CADCollisionChecker(geometry)
    config = SafetyConfig(0.2, -335.0, 1.0, -180.0, 0.0, 1.0, 1.0, 10.0, "test only")
    with pytest.raises(SafetyConfigurationViolation, match="full robot kinematics"):
        SafetySupervisor(config=config, reachability=reachability, collision=collision)
    assert "@" in geometry.source


def test_collision_checker_covers_robot_links_and_held_part() -> None:
    obstacle = AABB("base", (300.0, -20.0, -334.0), (360.0, 20.0, -250.0))
    owner = MotionOwner("part-1", "M1_01", True)
    link_hit = _RobotLinks(links=(AABB("J4", (320.0, 0.0, -300.0), (330.0, 10.0, -280.0)),))
    held_hit = _HeldPart(held=AABB("held:M1_01", (330.0, 0.0, -300.0), (340.0, 10.0, -280.0)))

    assert (
        CADCollisionChecker(SourcedCADGeometry(link_hit, _HeldPart())).first_collision(
            _point(200.0, 0.0, -200.0), _point(210.0, 0.0, -200.0), (obstacle,), 0.0, owner
        )
        == "robot-link:J4 with base"
    )
    assert (
        CADCollisionChecker(SourcedCADGeometry(_RobotLinks(), held_hit)).first_collision(
            _point(200.0, 0.0, -200.0), _point(210.0, 0.0, -200.0), (obstacle,), 0.0, owner
        )
        == "held:M1_01 with base"
    )


def test_collision_checker_ignores_old_box_for_moving_instance() -> None:
    owner = MotionOwner("part-1", "M4_04", True)
    held = _HeldPart(held=AABB("held:M4_04", (330.0, 0.0, -300.0), (340.0, 10.0, -280.0)))
    old_box = AABB("part-1", (330.0, 0.0, -300.0), (340.0, 10.0, -280.0))
    checker = CADCollisionChecker(SourcedCADGeometry(_RobotLinks(), held))
    assert (
        checker.first_collision(_point(200.0, 0.0, -200.0), _point(210.0, 0.0, -200.0), (old_box,), 0.0, owner) is None
    )


def test_collision_checker_matches_seeded_clear_layouts() -> None:
    rng = random.Random(20261101)
    owner = MotionOwner("moving", "M4_04", True)
    checker = CADCollisionChecker(SourcedCADGeometry(_RobotLinks(), _HeldPart()))
    for index in range(40):
        x = rng.uniform(150.0, 500.0)
        obstacle = AABB(f"part-{index}", (x, 40.0, -250.0), (x + 8.0, 48.0, -230.0))
        assert (
            checker.first_collision(_point(300.0, 0.0, -220.0), _point(310.0, 0.0, -220.0), (obstacle,), 0.0, owner)
            is None
        )


@pytest.mark.parametrize(
    "links",
    [(), (AABB("bad", (20.0, 0.0, 0.0), (10.0, 1.0, 1.0)),)],
)
def test_collision_checker_fails_closed_on_missing_or_invalid_geometry(links) -> None:
    checker = CADCollisionChecker(SourcedCADGeometry(_RobotLinks(links=links), _HeldPart()))
    owner = MotionOwner("moving", "M1_01", True)
    with pytest.raises(ProductionSafetyUnavailable, match="geometry|inverted"):
        checker.first_collision(_point(300.0, 0.0, -220.0), _point(310.0, 0.0, -220.0), (), 0.0, owner)
