"""Production controller composition requires every sourced cell dependency."""

from __future__ import annotations

from dataclasses import dataclass

from assembly.motion import Waypoint
from assembly.safety import AABB, MotionOwner


@dataclass
class _RobotLinks:
    source: str = "reviewed ER-4iA CAD link sweep"

    def swept_bounds(self, start: Waypoint, end: Waypoint) -> tuple[AABB, ...]:
        del start, end
        return (AABB("robot-link", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),)


@dataclass
class _HeldParts:
    source: str = "T29 grasp frames + printed-part STL"

    def swept_bounds(self, start: Waypoint, end: Waypoint, owner: MotionOwner) -> AABB:
        del start, end
        return AABB(f"held:{owner.object_id}", (2.0, 2.0, 2.0), (3.0, 3.0, 3.0))


def test_factory_builds_controller_without_a_fixed_product_scenario(monkeypatch) -> None:
    import assembly.cell_runtime as runtime
    from assembly.cell_runtime_safety import SDGReachabilityChecker
    from assembly.safety import SafetyConfig, SafetySupervisor

    sentinel = object()
    captured: dict[str, object] = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(runtime, "AssemblyController", capture)
    monkeypatch.setattr(
        runtime.SDGReachabilityChecker,
        "from_sdg",
        lambda: SDGReachabilityChecker(lambda pose: True, source="sdg/wrist_ik.py:reachable"),
    )
    inputs = runtime.CellRuntimeInputs(
        catalog=object(),
        perception=object(),
        poses=object(),
        motion_config=object(),
        inspection_stop=object(),
        startup_hold=object(),
        safety_config=SafetyConfig(
            0.1,
            -335.0,
            1.0,
            -180.0,
            0.0,
            3.0,
            1.0,
            1000.0,
            "test fixture",
        ),
        verification_thresholds=object(),
        execution_tolerances=object(),
        perception_tolerances=object(),
        recovery_targets=object(),
        robot_links=_RobotLinks(),
        held_parts=_HeldParts(),
    )

    assert runtime.build_cell_controller(inputs) is sentinel
    assert "products" not in captured
    assert isinstance(captured["safety"], SafetySupervisor)
    assert captured["startup_hold"] is inputs.startup_hold


def test_factory_rejects_unsourced_collision_geometry(monkeypatch) -> None:
    import pytest

    import assembly.cell_runtime as runtime
    from assembly.cell_runtime_safety import ProductionSafetyConfigurationError, SDGReachabilityChecker
    from assembly.safety import SafetyConfig

    monkeypatch.setattr(
        runtime.SDGReachabilityChecker,
        "from_sdg",
        lambda: SDGReachabilityChecker(lambda pose: True, source="sdg/wrist_ik.py:reachable"),
    )
    inputs = runtime.CellRuntimeInputs(
        catalog=object(),
        perception=object(),
        poses=object(),
        motion_config=object(),
        inspection_stop=object(),
        startup_hold=object(),
        safety_config=SafetyConfig(
            0.1,
            -335.0,
            1.0,
            -180.0,
            0.0,
            3.0,
            1.0,
            1000.0,
            "test fixture",
        ),
        verification_thresholds=object(),
        execution_tolerances=object(),
        perception_tolerances=object(),
        recovery_targets=object(),
        robot_links=_RobotLinks(source=""),
        held_parts=_HeldParts(),
    )

    with pytest.raises(ProductionSafetyConfigurationError, match="source"):
        runtime.build_cell_controller(inputs)
