"""Seeded scenario-fleet regression coverage for the T09 controller."""

from __future__ import annotations

from collections import Counter

import pytest

from assembly.catalog import load_catalog
from assembly.controller import AssemblyController
from assembly.controller_ports import ControllerPortError, ExecutionTolerances, PerceptionTolerances
from assembly.inventory import InventoryItem, InventoryZone, ObjectCondition
from assembly.motion import MotionConfig
from assembly.safety import AABBCollisionChecker, SafetyConfig, SafetySupervisor
from assembly.verification import VerificationThresholds
from eval.scenario_fleet import (
    FaultKind,
    FleetBounds,
    FleetReport,
    FleetRunner,
    ScenarioGenerator,
    ScenarioRunStatus,
    ScenarioRuntime,
    parse_instruction_products,
)
from robot.cell.t09_mock_runtime import MockBoxReachability

BOUNDS = FleetBounds(
    x_mm=(220.0, 520.0),
    y_mm=(-260.0, 260.0),
    part_z_mm=-300.0,
    base_z_mm=-320.0,
    reject_pose=(560.0, -360.0, -300.0, 0.0),
    staging_x_mm=520.0,
    staging_y_mm=(-260.0, -120.0),
    home_pose=(200.0, 0.0, -250.0, -180.0, 0.0, 0.0),
)


def _generator(seed: int) -> ScenarioGenerator:
    return ScenarioGenerator(load_catalog(), bounds=BOUNDS, seed=seed)


def _controller(runtime: ScenarioRuntime, products: tuple[int, ...]) -> AssemblyController:
    safety = SafetySupervisor(
        config=SafetyConfig(0.1, -335.0, 1.0, -180.0, 0.0, 0.5, 1.0),
        reachability=MockBoxReachability(),
        collision=AABBCollisionChecker(),
        allow_tcp_sweep_only_for_mock=True,
    )
    return AssemblyController(
        catalog=load_catalog(),
        products=products,
        perception=runtime.perception,
        poses=runtime.poses,
        motion_config=MotionConfig(20.0, 30.0, 15.0, 20.0, 1000.0, 360.0, 100),
        safety=safety,
        verification_thresholds=VerificationThresholds(0.9, 0.8, 0.9, 0.1, 1.0),
        execution_tolerances=ExecutionTolerances(0.0, 0.0),
        perception_tolerances=PerceptionTolerances(0.0, 0.0),
        recovery_targets=runtime.recovery_targets,
    )


def test_seeded_generator_uses_prompt_and_random_perception_layouts() -> None:
    first = _generator(1701).generate(60)
    repeated = _generator(1701).generate(60)
    different = _generator(1702).generate(60)

    assert first == repeated
    assert first != different
    assert {scenario.fault for scenario in first} == set(FaultKind)
    assert len({scenario.products for scenario in first}) > 10
    assert len({scenario.layout_signature for scenario in first}) == len(first)
    assert all(parse_instruction_products(s.instruction, load_catalog()) == s.products for s in first)
    assert all(len(s.products) == len(set(s.products)) == 3 for s in first)


def test_same_shape_bases_are_interchangeable_and_never_fixed_by_product() -> None:
    catalog = load_catalog()
    scenarios = _generator(17).generate(80)
    bases = [item for scenario in scenarios for item in scenario.objects if item.object_id.startswith("F")]

    assert all(catalog.product(item.owner_product).base_type == item.base_type for item in bases)
    assert any(item.object_id != f"F0{item.owner_product}" for item in bases)


def test_faults_are_injected_through_the_perception_boundary() -> None:
    scenarios = _generator(91).generate(30)
    by_fault = {scenario.fault: scenario for scenario in scenarios}

    fallen = ScenarioRuntime(by_fault[FaultKind.FALLEN])
    fallen_item = fallen.scenario.hazardous_objects[0]
    observed = fallen.perception.scan(fallen.observation(0.0))
    assert next(item for item in observed.items if item.instance_id == fallen_item.instance_id).condition.value == "fallen"

    defect = ScenarioRuntime(by_fault[FaultKind.DEFECT])
    defect_item = defect.scenario.hazardous_objects[0]
    observed = defect.perception.scan(defect.observation(0.0))
    assert next(item for item in observed.items if item.instance_id == defect_item.instance_id).condition.value == "defective"

    shortage = by_fault[FaultKind.SPARE_SHORTAGE]
    missing = shortage.required_missing_object_id
    assert missing is not None
    assert all(item.object_id != missing for item in shortage.objects)


def test_grasp_pose_is_unavailable_until_the_perception_scan_observes_it() -> None:
    scenario = next(item for item in _generator(71).generate(6) if item.fault is FaultKind.CLEAN)
    runtime = ScenarioRuntime(scenario)
    scripted = next(item for item in scenario.objects if item.object_id.startswith("M"))
    inventory_item = InventoryItem(
        scripted.instance_id,
        scripted.object_id,
        InventoryZone.TABLE,
        ObjectCondition.VERIFIED,
    )

    with pytest.raises(ControllerPortError, match="not observed by perception"):
        runtime.poses.grasp_for(inventory_item, observed_at_s=0.0)

    runtime.perception.scan(runtime.observation(0.0))
    assert runtime.poses.grasp_for(inventory_item, observed_at_s=0.0).tcp.x == scripted.pose[0]


def test_ungraspable_fallen_parts_are_represented_fail_closed() -> None:
    scenarios = _generator(301).generate(120)
    fallen = [scenario.hazardous_objects[0] for scenario in scenarios if scenario.fault is FaultKind.FALLEN]

    assert {item.grasp_approved for item in fallen} == {True, False}


def test_fleet_runs_real_t09_controller_and_reports_only_measured_counts() -> None:
    scenarios = _generator(90210).generate(12)
    report = FleetRunner(_controller, max_callbacks=1500, dt_s=0.1).run(scenarios)

    assert isinstance(report, FleetReport)
    assert report.scenario_count == len(scenarios)
    assert report.requested_products == 3 * len(scenarios)
    assert report.completed_products == sum(len(result.completed_products) for result in report.results)
    assert report.action_rows == sum(result.action_rows for result in report.results)
    assert report.assists == sum(result.assists for result in report.results)
    assert report.retries == sum(result.retries for result in report.results)
    assert report.escapes == sum(result.escapes for result in report.results)
    assert Counter(result.fault for result in report.results) == Counter(s.fault for s in scenarios)
    assert all(result.products == parse_instruction_products(result.instruction, load_catalog()) for result in report.results)
    assert all(result.status is not ScenarioRunStatus.ERROR for result in report.results)
    assert any(result.status is ScenarioRunStatus.COMPLETE for result in report.results)
    summary = report.as_dict()
    assert summary["scenario_count"] == len(scenarios)
    assert summary["status_counts"] == dict(Counter(result.status.value for result in report.results))
    assert summary["fault_counts"] == dict(Counter(result.fault.value for result in report.results))

