"""The real-cell controller factory exists but remains fail-closed until reviewed."""

from __future__ import annotations

from dataclasses import fields, replace
from importlib.util import find_spec
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _dependency_bundle(runtime, perception: object):
    assert hasattr(runtime, "RealCellDependencies")
    values = {
        field.name: perception if field.name == "perception" else object()
        for field in fields(runtime.RealCellDependencies)
    }
    return runtime.RealCellDependencies(**values)


def test_real_cell_runtime_factory_module_exists() -> None:
    assert find_spec("robot.cell.t09_real_runtime") is not None


def test_real_placement_artifact_has_three_reviewed_rollouts() -> None:
    placement = yaml.safe_load(
        (ROOT / "robot/cell/submission/placement.yaml").read_text(encoding="utf-8")
    )

    assert len(placement["rollouts"]) == 3
    assert all(len(rollout["objects"]) == 15 for rollout in placement["rollouts"])
    assert all("Assemble products" in rollout["prompt"] for rollout in placement["rollouts"])


def test_real_cell_requirements_name_every_calibration_and_runtime_dependency() -> None:
    import robot.cell.t09_real_runtime as runtime

    assert hasattr(runtime, "RealCellRequirement")
    assert hasattr(runtime, "RequirementKind")
    assert hasattr(runtime, "required_real_cell_requirements")
    RealCellRequirement = runtime.RealCellRequirement
    RequirementKind = runtime.RequirementKind

    requirements = runtime.required_real_cell_requirements()

    assert type(requirements) is tuple
    assert all(type(item) is RealCellRequirement for item in requirements)
    assert {item.code for item in requirements} == {
        "inspection_slots",
        "reject_spots",
        "staging_slots",
        "pull_out_targets",
        "final_verification_views",
        "motion_clearances",
        "cell_frames",
        "camera_extrinsics",
        "base_regions",
        "cell_perception_port",
        "cell_startup_hold_contract",
        "measured_pose_bridge",
        "perception_reset",
        "production_final_verifier",
        "production_safety",
        "model_adapters",
        "watchdog_limits",
        "expected_layout_verifier",
    }
    assert len({item.code for item in requirements}) == len(requirements)
    assert {item.kind for item in requirements} == {
        RequirementKind.CSI_CALIBRATION,
        RequirementKind.RUNTIME_DEPENDENCY,
    }
    assert all(item.fields and item.detail for item in requirements)
    for item in requirements:
        if item.kind is RequirementKind.CSI_CALIBRATION:
            assert "source" in item.fields


def test_csi_pose_requirements_expose_exact_fields_without_invented_coordinates() -> None:
    import robot.cell.t09_real_runtime as runtime

    assert hasattr(runtime, "RequirementKind")
    assert hasattr(runtime, "required_real_cell_requirements")
    RequirementKind = runtime.RequirementKind

    by_code = {
        item.code: item
        for item in runtime.required_real_cell_requirements()
        if item.kind is RequirementKind.CSI_CALIBRATION
    }

    assert by_code["inspection_slots"].fields == (
        "slot_id",
        "selection_bounds_xy_mm",
        "endpoint_tcp_uf0",
        "safe_z_mm",
        "source",
    )
    assert by_code["reject_spots"].fields == (
        "spot_id",
        "release_tcp_uf0",
        "verification_tcp_uf0",
        "initially_occupied",
        "source",
    )
    assert by_code["staging_slots"].fields == (
        "slot_id",
        "release_tcp_uf0",
        "verification_tcp_uf0",
        "initially_occupied",
        "source",
    )
    assert by_code["pull_out_targets"].fields == (
        "product",
        "object_id",
        "set_down_tcp_uf0",
        "clear_z_mm",
        "source",
    )
    assert by_code["final_verification_views"].fields == (
        "product",
        "base_object_id",
        "view_id",
        "endpoint_tcp_uf0",
        "safe_z_mm",
        "expected_instance_roles",
        "source",
    )


def test_public_real_factory_rejects_every_non_executor_object() -> None:
    import robot.cell.t09_real_runtime as runtime

    assert hasattr(runtime, "build_groot_real_controller")
    with pytest.raises(runtime.RealCellFactoryConfigurationError, match="exact T09GrootExecutor"):
        runtime.build_groot_real_controller(object())


def test_public_real_factory_lists_all_missing_requirements_and_stays_unpinned() -> None:
    import robot.cell.t09_real_runtime as runtime
    from assembly import cell_runtime
    from policy_learn.deploy.cell_executor import T09GrootExecutor

    executor = object.__new__(T09GrootExecutor)

    assert cell_runtime.REVIEWED_REAL_CELL_FACTORY is None
    assert hasattr(runtime, "REVIEWED_REAL_CELL_DEPENDENCY_LOADER")
    assert runtime.REVIEWED_REAL_CELL_DEPENDENCY_LOADER is None
    with pytest.raises(runtime.RealCellFactoryBlockedError) as captured:
        runtime.build_groot_real_controller(executor)

    message = str(captured.value)
    assert "no reviewed real-cell dependency loader" in message
    assert all(item.code in message for item in runtime.required_real_cell_requirements())


def test_real_dependency_bundle_names_every_controller_boundary_explicitly() -> None:
    import robot.cell.t09_real_runtime as runtime

    assert hasattr(runtime, "RealCellDependencies")
    assert {field.name for field in fields(runtime.RealCellDependencies)} == {
        "catalog",
        "perception",
        "poses",
        "motion_config",
        "inspection_stop",
        "startup_hold",
        "safety",
        "verification_thresholds",
        "execution_tolerances",
        "perception_tolerances",
        "recovery_targets",
        "final_verifier",
        "inspection_poses",
        "endpoint_settle_limits",
        "watchdog_limits",
        "expected_layout_verifier",
    }


def test_internal_real_factory_rejects_non_cell_perception_before_construction() -> None:
    import robot.cell.t09_real_runtime as runtime
    from policy_learn.deploy.cell_executor import T09GrootExecutor

    executor = object.__new__(T09GrootExecutor)
    dependencies = _dependency_bundle(runtime, object())

    with pytest.raises(runtime.RealCellFactoryConfigurationError, match="exact CellPerceptionPort"):
        runtime._build_groot_real_controller(executor, dependencies)


def test_internal_real_factory_rechecks_exact_executor_identity() -> None:
    import robot.cell.t09_real_runtime as runtime
    from perception.port.port import CellPerceptionPort

    dependencies = _dependency_bundle(runtime, object.__new__(CellPerceptionPort))

    with pytest.raises(runtime.RealCellFactoryConfigurationError, match="exact T09GrootExecutor"):
        runtime._build_groot_real_controller(object(), dependencies)


def test_internal_real_factory_requires_the_exact_reviewed_layout_verifier() -> None:
    import robot.cell.t09_real_runtime as runtime
    from perception.port.port import CellPerceptionPort
    from policy_learn.deploy.cell_executor import T09GrootExecutor

    perception = object.__new__(CellPerceptionPort)
    dependencies = _dependency_bundle(runtime, perception)
    executor = object.__new__(T09GrootExecutor)

    with pytest.raises(runtime.RealCellFactoryConfigurationError, match="ExpectedLayoutVerifier"):
        runtime._build_groot_real_controller(executor, dependencies)


def test_internal_real_factory_requires_the_exact_t38_inspection_registry() -> None:
    import robot.cell.t09_real_runtime as runtime
    from assembly.catalog import load_catalog
    from perception.port.port import CellPerceptionPort
    from policy_learn.deploy.cell_executor import T09GrootExecutor
    from robot.cell.real_layout import ExpectedLayoutVerifier, load_real_cell_layout

    catalog = load_catalog()
    verifier = ExpectedLayoutVerifier(load_real_cell_layout(), catalog)
    perception = object.__new__(CellPerceptionPort)
    perception._expected_layout_verifier = verifier
    dependencies = replace(
        _dependency_bundle(runtime, perception),
        expected_layout_verifier=verifier,
    )

    with pytest.raises(runtime.RealCellFactoryConfigurationError, match="T38 fixed inspection"):
        runtime._build_groot_real_controller(object.__new__(T09GrootExecutor), dependencies)
