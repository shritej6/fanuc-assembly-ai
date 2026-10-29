"""Seeded scenario-fleet regression coverage for the T09 controller."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import pytest

from assembly.catalog import load_catalog
from assembly.controller import AssemblyController
from assembly.controller_ports import ControllerPortError, ExecutionTolerances, PerceptionTolerances
from assembly.controller_startup import StartupHoldConfig
from assembly.inventory import InventoryItem, InventoryZone, ObjectCondition
from assembly.motion import InspectionStopConfig, MotionConfig, Waypoint
from assembly.safety import AABBCollisionChecker, SafetyConfig, SafetySupervisor
from assembly.state_machine import PackMLState
from assembly.verification import VerificationThresholds
from eval.scenario_fleet import (
    FaultKind,
    FleetBounds,
    FleetInputError,
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


def _mock_safety() -> SafetySupervisor:
    return SafetySupervisor(
        config=SafetyConfig(
            0.1,
            -335.0,
            1.0,
            -180.0,
            0.0,
            0.5,
            1.0,
            10000.0,
            "T17 mock-only speed fixture",
        ),
        reachability=MockBoxReachability(),
        collision=AABBCollisionChecker(),
        allow_tcp_sweep_only_for_mock=True,
    )


def _motion_config() -> MotionConfig:
    return MotionConfig(
        20.0,
        30.0,
        15.0,
        20.0,
        1000.0,
        360.0,
        100,
        -180.0,
        "T17 mock-only safe transit fixture",
    )


def _controller(runtime: ScenarioRuntime) -> AssemblyController:
    return AssemblyController(
        catalog=load_catalog(),
        perception=runtime.perception,
        poses=runtime.poses,
        motion_config=_motion_config(),
        inspection_stop=InspectionStopConfig(
            -168.0,
            -180.0,
            0.0,
            "T17 mock-only wrist inspection fixture",
        ),
        startup_hold=StartupHoldConfig(
            Waypoint(*BOUNDS.home_pose, 0.0),
            "T17 caller-supplied mock home pose",
        ),
        safety=_mock_safety(),
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


def test_generator_balances_every_unordered_three_product_combination() -> None:
    expected = {tuple(combo) for combo in combinations(range(1, 7), 3)}

    three_times = Counter(tuple(sorted(scenario.products)) for scenario in _generator(1704).generate(60))

    for seed in (11, 1703):
        once = Counter(tuple(sorted(scenario.products)) for scenario in _generator(seed).generate(20))
        assert once == Counter({combo: 1 for combo in expected})
    assert three_times == Counter({combo: 3 for combo in expected})


def test_random_layout_xy_values_are_diverse_and_inside_configured_bounds() -> None:
    scenarios = _generator(1705).generate(20)
    xy_values = [(item.pose[0], item.pose[1]) for scenario in scenarios for item in scenario.objects]
    x_values = {x for x, _ in xy_values}
    y_values = {y for _, y in xy_values}

    assert all(BOUNDS.x_mm[0] <= x <= BOUNDS.x_mm[1] for x in x_values)
    assert all(BOUNDS.y_mm[0] <= y <= BOUNDS.y_mm[1] for y in y_values)
    assert len(x_values) == len(xy_values)
    assert len(y_values) == len(xy_values)
    assert max(x_values) - min(x_values) > (BOUNDS.x_mm[1] - BOUNDS.x_mm[0]) / 2
    assert max(y_values) - min(y_values) > (BOUNDS.y_mm[1] - BOUNDS.y_mm[0]) / 2


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


def test_coarse_xy_centers_enter_only_through_perception_scan() -> None:
    scenario = _generator(92).generate(1)[0]
    runtime = ScenarioRuntime(scenario)

    observed = runtime.perception.scan(runtime.observation(0.0))

    expected = {item.instance_id: item.pose[:2] for item in scenario.objects}
    assert {item.instance_id: item.coarse_xy_mm for item in observed.items} == expected


def test_legacy_grasp_path_stays_disabled_after_perception_scan() -> None:
    scenario = next(item for item in _generator(71).generate(6) if item.fault is FaultKind.CLEAN)
    runtime = ScenarioRuntime(scenario)
    scripted = next(item for item in scenario.objects if item.object_id.startswith("M"))
    inventory_item = InventoryItem(
        scripted.instance_id,
        scripted.object_id,
        InventoryZone.TABLE,
        ObjectCondition.VERIFIED,
    )

    with pytest.raises(ControllerPortError, match="FoundationPose wrist evidence required"):
        runtime.poses.grasp_for(inventory_item, observed_at_s=0.0)

    runtime.perception.scan(runtime.observation(0.0))
    with pytest.raises(ControllerPortError, match="FoundationPose wrist evidence required"):
        runtime.poses.grasp_for(inventory_item, observed_at_s=0.0)


def test_foundationpose_grasp_uses_detection_pose_not_hidden_scenario_pose() -> None:
    scenario = next(item for item in _generator(72).generate(6) if item.fault is FaultKind.CLEAN)
    runtime = ScenarioRuntime(scenario)
    scripted = next(item for item in scenario.objects if item.object_id.startswith("M"))
    part = InventoryItem(scripted.instance_id, scripted.object_id, InventoryZone.TABLE, ObjectCondition.VERIFIED)
    runtime.perception.scan(runtime.observation(0.0))
    evidence = runtime.perception.pre_pick(runtime.observation(0.0), part)
    detection = dict(evidence.detection)
    detection["pose"] = {"x": 444.0, "y": -111.0, "z": -299.0, "yaw": 37.0}

    grasp = runtime.poses.grasp_from_foundationpose(part, detection, observed_at_s=0.0)

    assert (grasp.tcp.x, grasp.tcp.y, grasp.tcp.z, grasp.tcp.r) == (444.0, -111.0, -299.0, 37.0)

    detection["resting_side"] = "side"
    fallen_grasp = runtime.poses.grasp_from_foundationpose(part, detection, observed_at_s=0.0)
    assert fallen_grasp.resting_state.value == "fallen"

    detection["resting_side"] = "flipped"
    with pytest.raises(ControllerPortError, match="no approved grasp for resting side"):
        runtime.poses.grasp_from_foundationpose(part, detection, observed_at_s=0.0)


def test_ungraspable_fallen_parts_are_represented_fail_closed() -> None:
    scenarios = _generator(301).generate(120)
    fallen = [scenario.hazardous_objects[0] for scenario in scenarios if scenario.fault is FaultKind.FALLEN]

    assert {item.grasp_approved for item in fallen} == {True, False}


def test_graspable_fallen_part_is_physically_rejected_before_spare_selection() -> None:
    scenario = next(
        item
        for item in _generator(301).generate(120)
        if item.fault is FaultKind.FALLEN and item.hazardous_objects[0].grasp_approved
        and item.hazardous_objects[0].owner_product == item.products[0]
    )
    fallen = scenario.hazardous_objects[0]
    spare = next(
        item
        for item in scenario.objects
        if item.object_id == fallen.object_id and item.zone is InventoryZone.SPARE
    )

    result = FleetRunner(_controller, max_callbacks=1500, dt_s=0.1).run((scenario,)).results[0]

    physical_reject_at = next(
        index
        for index, event in enumerate(result.physical_events)
        if event.kind == "reject_transfer" and event.instance_id == fallen.instance_id
    )
    spare_pick_at = next(
        index
        for index, event in enumerate(result.physical_events)
        if event.kind == "pre_pick" and event.instance_id == spare.instance_id
    )
    rejected_at = next(
        index
        for index, event in enumerate(result.events)
        if event.kind == "rejected" and event.detail == fallen.instance_id
    )
    spare_at = next(
        index
        for index, event in enumerate(result.events)
        if event.kind in {"spare", "replacement"} and event.detail == spare.instance_id
    )
    assert physical_reject_at < spare_pick_at
    assert rejected_at < spare_at
    assert result.status is ScenarioRunStatus.COMPLETE
    assert result.escapes == 0


def test_fallen_part_without_approved_grasp_holds_before_any_reject_motion() -> None:
    scenario = next(
        item
        for item in _generator(301).generate(120)
        if item.fault is FaultKind.FALLEN and not item.hazardous_objects[0].grasp_approved
        and item.hazardous_objects[0].owner_product == item.products[0]
    )
    fallen = scenario.hazardous_objects[0]

    result = FleetRunner(_controller, max_callbacks=1500, dt_s=0.1).run((scenario,)).results[0]

    assert result.status is ScenarioRunStatus.HELD
    assert result.escapes is None
    assert result.unresolved_hazards == 1
    assert any(event.kind == "pre_pick" and event.instance_id == fallen.instance_id for event in result.physical_events)
    assert any(
        event.kind == "andon" and "no approved grasp for resting side" in event.detail for event in result.events
    )
    assert all(event.instance_id != fallen.instance_id for event in result.physical_events if event.kind == "reject_transfer")
    assert all(event.detail != fallen.instance_id for event in result.events if event.kind == "rejected")


def test_incomplete_hazard_run_is_indeterminate_and_blocks_acceptance() -> None:
    scenario = next(item for item in _generator(402).generate(6) if item.fault is FaultKind.FALLEN)

    report = FleetRunner(_controller, max_callbacks=1, dt_s=0.1).run((scenario,))

    result = report.results[0]
    assert result.status is ScenarioRunStatus.CALLBACK_LIMIT
    assert result.escapes is None
    assert result.unresolved_hazards == 1
    assert report.escapes is None
    assert report.unresolved_hazards == 1
    assert report.acceptance_ready is False


def test_error_after_finished_flag_still_has_indeterminate_escape_count() -> None:
    class FinishedWithMalformedAction:
        finished = True
        completed_products: tuple[int, ...] = ()
        events: tuple[object, ...] = ()

        @staticmethod
        def act(observation: object) -> list[object]:
            del observation
            return []

    def factory(runtime: ScenarioRuntime) -> FinishedWithMalformedAction:
        del runtime
        return FinishedWithMalformedAction()

    scenario = next(item for item in _generator(404).generate(6) if item.fault is FaultKind.FALLEN)

    result = FleetRunner(factory, max_callbacks=1, dt_s=0.1).run((scenario,)).results[0]  # type: ignore[arg-type]

    assert result.status is ScenarioRunStatus.ERROR
    assert result.escapes is None
    assert result.unresolved_hazards == 1


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("held", ScenarioRunStatus.HELD),
        ("suspended", ScenarioRunStatus.SUSPENDED),
        ("aborted", ScenarioRunStatus.ABORTED),
        ("callback_limit", ScenarioRunStatus.CALLBACK_LIMIT),
        ("error", ScenarioRunStatus.ERROR),
    ],
)
def test_every_incomplete_status_is_indeterminate_and_blocks_acceptance(
    mode: str,
    expected: ScenarioRunStatus,
) -> None:
    state_by_mode = {
        "held": PackMLState.HELD,
        "suspended": PackMLState.SUSPENDED,
        "aborted": PackMLState.ABORTED,
    }

    class IncompleteController:
        finished = False
        completed_products: tuple[int, ...] = ()
        events: tuple[object, ...] = ()
        state_machine = SimpleNamespace(state=state_by_mode.get(mode, PackMLState.EXECUTE))

        @staticmethod
        def act(observation: object) -> list[float]:
            del observation
            return [] if mode == "error" else [*BOUNDS.home_pose, 0.0]

    scenario = next(item for item in _generator(405).generate(6) if item.fault is FaultKind.FALLEN)
    report = FleetRunner(lambda runtime: IncompleteController(), max_callbacks=1, dt_s=0.1).run((scenario,))  # type: ignore[arg-type]

    assert report.results[0].status is expected
    assert report.results[0].escapes is None
    assert report.acceptance_ready is False


def test_runner_rejects_products_that_disagree_with_trial_instruction() -> None:
    scenario = _generator(403).generate(1)[0]
    inconsistent = replace(scenario, products=tuple(reversed(scenario.products)))

    with pytest.raises(FleetInputError, match="products disagree with trial instruction"):
        FleetRunner(_controller, max_callbacks=1, dt_s=0.1).run((inconsistent,))


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
    assert report.known_escapes == sum(result.escapes or 0 for result in report.results)
    assert report.unresolved_hazards == sum(result.unresolved_hazards for result in report.results)
    if any(result.escapes is None for result in report.results):
        assert report.escapes is None
        assert report.acceptance_ready is False
    else:
        assert report.escapes == report.known_escapes
    assert Counter(result.fault for result in report.results) == Counter(s.fault for s in scenarios)
    assert all(result.products == parse_instruction_products(result.instruction, load_catalog()) for result in report.results)
    assert all(result.status is not ScenarioRunStatus.ERROR for result in report.results)
    assert any(result.status is ScenarioRunStatus.COMPLETE for result in report.results)
    summary = report.as_dict()
    assert summary["scenario_count"] == len(scenarios)
    assert summary["status_counts"] == dict(Counter(result.status.value for result in report.results))
    assert summary["fault_counts"] == dict(Counter(result.fault.value for result in report.results))


def test_unseen_seed_runs_end_to_end_without_fixed_prompt_or_layout() -> None:
    scenarios = _generator(24_681_357).generate(20)

    report = FleetRunner(_controller, max_callbacks=1500, dt_s=0.1).run(scenarios)

    assert report.scenario_count == 20
    assert len({tuple(sorted(result.products)) for result in report.results}) == 20
    assert {result.fault for result in report.results} == set(FaultKind)
    assert {result.status for result in report.results} <= {
        ScenarioRunStatus.COMPLETE,
        ScenarioRunStatus.HELD,
        ScenarioRunStatus.SUSPENDED,
    }
    assert report.known_escapes == 0
    assert all(
        result.products == parse_instruction_products(result.instruction, load_catalog()) for result in report.results
    )
