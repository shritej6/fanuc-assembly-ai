"""The real-cell controller factory exists but remains fail-closed until reviewed.
Authored by Shritej Gunda.
"""

from __future__ import annotations

from dataclasses import fields, replace
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _dependency_bundle(runtime, perception: object):
    from assembly.safety import SafetySupervisor

    assert hasattr(runtime, "RealCellDependencies")
    values = {
        field.name: (
            perception
            if field.name == "perception"
            else object.__new__(SafetySupervisor)
            if field.name == "safety"
            else object()
        )
        for field in fields(runtime.RealCellDependencies)
    }
    return runtime.RealCellDependencies(**values)


def _executor_for(dependencies):
    from policy_learn.deploy.cell_executor import T09GrootExecutor

    executor = object.__new__(T09GrootExecutor)
    executor._require_scene_safety = True
    executor._safety = dependencies.safety
    return executor


def test_real_cell_runtime_factory_module_exists() -> None:
    assert find_spec("robot.cell.t09_real_runtime") is not None


def test_real_placement_artifact_has_three_reviewed_rollouts() -> None:
    placement = yaml.safe_load(
        (ROOT / "robot/cell/submission/placement.yaml").read_text(encoding="utf-8")
    )

    assert len(placement["rollouts"]) == 3
    assert all(len(rollout["objects"]) == 5 for rollout in placement["rollouts"])
    assert all("Assemble product 5" in rollout["prompt"] for rollout in placement["rollouts"])


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
        "final_view_registry",
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


def test_public_real_factory_refuses_unverified_presence_by_default(monkeypatch) -> None:
    import robot.cell.t09_real_runtime as runtime
    from assembly import cell_runtime
    from assembly.safety import SafetySupervisor
    from policy_learn.deploy.cell_executor import T09GrootExecutor
    from robot.cell.saturday_runtime_adapters import UNVERIFIED_PRESENCE_ENV

    executor = object.__new__(T09GrootExecutor)
    executor._require_scene_safety = True
    executor._safety = object.__new__(SafetySupervisor)
    monkeypatch.delenv(UNVERIFIED_PRESENCE_ENV, raising=False)

    assert cell_runtime.REVIEWED_REAL_CELL_FACTORY is None
    assert hasattr(runtime, "REVIEWED_REAL_CELL_DEPENDENCY_LOADER")
    assert runtime.REVIEWED_REAL_CELL_DEPENDENCY_LOADER is runtime.load_saturday_real_cell_dependencies
    with pytest.raises(
        runtime.RealCellFactoryBlockedError,
        match="detector_presence_unverified",
    ) as captured:
        runtime.build_groot_real_controller(executor)

    message = str(captured.value)
    assert "model.safetensors" in message
    assert UNVERIFIED_PRESENCE_ENV in message


def test_saturday_adapter_boundary_imports_the_pinned_module(monkeypatch) -> None:
    import robot.cell.t09_real_runtime as runtime

    expected = object()
    calls: list[tuple[object, ...]] = []

    def load_adapters(*args):
        calls.append(args)
        return expected

    module = SimpleNamespace(
        SaturdayAdapterBlockedError=type("SaturdayAdapterBlockedError", (RuntimeError,), {}),
        load_saturday_runtime_adapters=load_adapters,
    )
    monkeypatch.setattr(runtime, "import_module", lambda name: module)
    inputs = (object(), object(), object(), object())

    assert runtime._load_saturday_runtime_adapters(*inputs) is expected
    assert calls == [inputs]


def test_saturday_adapter_boundary_preserves_typed_fail_closed_reason(monkeypatch) -> None:
    import robot.cell.t09_real_runtime as runtime

    class SaturdayAdapterBlockedError(RuntimeError):
        pass

    def load_adapters(*args):
        del args
        raise SaturdayAdapterBlockedError("detector weights are absent; explicit arming required")

    module = SimpleNamespace(
        SaturdayAdapterBlockedError=SaturdayAdapterBlockedError,
        load_saturday_runtime_adapters=load_adapters,
    )
    monkeypatch.setattr(runtime, "import_module", lambda name: module)

    with pytest.raises(
        runtime.RealCellFactoryBlockedError,
        match="detector weights are absent; explicit arming required",
    ):
        runtime._load_saturday_runtime_adapters(object(), object(), object(), object())


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
        "final_view_registry",
    }


def test_internal_real_factory_rejects_non_cell_perception_before_construction() -> None:
    import robot.cell.t09_real_runtime as runtime

    dependencies = _dependency_bundle(runtime, object())
    executor = _executor_for(dependencies)

    with pytest.raises(runtime.RealCellFactoryConfigurationError, match="exact CellPerceptionPort"):
        runtime._build_groot_real_controller(executor, dependencies)


def test_internal_real_factory_rechecks_exact_executor_identity() -> None:
    import robot.cell.t09_real_runtime as runtime
    from perception.port.port import CellPerceptionPort

    dependencies = _dependency_bundle(runtime, object.__new__(CellPerceptionPort))

    with pytest.raises(runtime.RealCellFactoryConfigurationError, match="exact T09GrootExecutor"):
        runtime._build_groot_real_controller(object(), dependencies)


def test_internal_real_factory_rejects_disabled_scene_safety(monkeypatch, tmp_path) -> None:
    import robot.cell.t09_real_runtime as runtime
    from assembly.controller_ports import ExecutionTolerances
    from policy_learn.deploy.cell_executor import T09GrootExecutor
    from robot.cell.saturday_runtime_adapters import (
        RUN_DIR_ENV,
        UNVERIFIED_PRESENCE_ENV,
    )

    class ResetOnlyPolicy:
        last_guard_result = None

        def reset(self):
            return None

        def close(self):
            return None

    monkeypatch.setenv(UNVERIFIED_PRESENCE_ENV, "1")
    monkeypatch.setenv(RUN_DIR_ENV, str(tmp_path))
    dependencies = runtime.load_saturday_real_cell_dependencies()
    executor = T09GrootExecutor(
        policy=ResetOnlyPolicy(),
        safety=None,
        tolerances=ExecutionTolerances(0.3, 1.0),
        require_scene_safety=False,
    )

    with pytest.raises(
        runtime.RealCellFactoryConfigurationError,
        match="scene safety must be enabled",
    ):
        runtime._build_groot_real_controller(executor, dependencies)


def test_internal_real_factory_binds_executor_to_dependency_safety(monkeypatch, tmp_path) -> None:
    import robot.cell.t09_real_runtime as runtime
    from assembly.controller_ports import ExecutionTolerances
    from assembly.safety import SafetySupervisor
    from policy_learn.deploy.cell_executor import T09GrootExecutor
    from robot.cell.saturday_runtime_adapters import (
        RUN_DIR_ENV,
        UNVERIFIED_PRESENCE_ENV,
    )

    class ResetOnlyPolicy:
        last_guard_result = None

        def reset(self):
            return None

        def close(self):
            return None

    monkeypatch.setenv(UNVERIFIED_PRESENCE_ENV, "1")
    monkeypatch.setenv(RUN_DIR_ENV, str(tmp_path))
    dependencies = runtime.load_saturday_real_cell_dependencies()
    executor = T09GrootExecutor(
        policy=ResetOnlyPolicy(),
        safety=object.__new__(SafetySupervisor),
        tolerances=ExecutionTolerances(0.3, 1.0),
        require_scene_safety=True,
    )

    with pytest.raises(
        runtime.RealCellFactoryConfigurationError,
        match="same SafetySupervisor",
    ):
        runtime._build_groot_real_controller(executor, dependencies)


def test_public_real_factory_binds_scene_safety_before_controller_exists(
    monkeypatch,
    tmp_path,
) -> None:
    import robot.cell.t09_real_runtime as runtime
    from assembly.controller_ports import ExecutionTolerances
    from policy_learn.deploy.cell_executor import T09GrootExecutor
    from robot.cell.saturday_runtime_adapters import (
        RUN_DIR_ENV,
        UNVERIFIED_PRESENCE_ENV,
    )

    class ResetOnlyPolicy:
        last_guard_result = None

        def reset(self):
            return None

        def close(self):
            return None

    monkeypatch.setenv(UNVERIFIED_PRESENCE_ENV, "1")
    monkeypatch.setenv(RUN_DIR_ENV, str(tmp_path))
    dependencies = runtime.load_saturday_real_cell_dependencies()
    monkeypatch.setattr(
        runtime,
        "REVIEWED_REAL_CELL_DEPENDENCY_LOADER",
        lambda: dependencies,
    )
    executor = T09GrootExecutor(
        policy=ResetOnlyPolicy(),
        safety=None,
        tolerances=ExecutionTolerances(0.3, 1.0),
        require_scene_safety=False,
    )

    controller = runtime.build_groot_real_controller(executor)

    assert controller.groot_executor is executor
    assert executor._require_scene_safety is True
    assert executor._safety is dependencies.safety


def test_internal_real_factory_requires_the_exact_reviewed_layout_verifier() -> None:
    import robot.cell.t09_real_runtime as runtime
    from perception.port.port import CellPerceptionPort

    perception = object.__new__(CellPerceptionPort)
    dependencies = _dependency_bundle(runtime, perception)
    executor = _executor_for(dependencies)

    with pytest.raises(runtime.RealCellFactoryConfigurationError, match="ExpectedLayoutVerifier"):
        runtime._build_groot_real_controller(executor, dependencies)


def test_internal_real_factory_requires_the_exact_t38_inspection_registry() -> None:
    import robot.cell.t09_real_runtime as runtime
    from assembly.catalog import load_catalog
    from perception.port.port import CellPerceptionPort
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
        runtime._build_groot_real_controller(_executor_for(dependencies), dependencies)


def test_internal_real_factory_requires_cell_layout_recovery_targets() -> None:
    import robot.cell.t09_real_runtime as runtime
    from assembly.catalog import load_catalog
    from perception.port.port import CellPerceptionPort
    from robot.cell.real_layout import (
        ExpectedLayoutVerifier,
        load_real_cell_layout,
        reviewed_inspection_registry,
    )

    catalog = load_catalog()
    verifier = ExpectedLayoutVerifier(load_real_cell_layout(), catalog)
    perception = object.__new__(CellPerceptionPort)
    perception._expected_layout_verifier = verifier
    dependencies = replace(
        _dependency_bundle(runtime, perception),
        expected_layout_verifier=verifier,
        inspection_poses=reviewed_inspection_registry(),
    )

    with pytest.raises(runtime.RealCellFactoryConfigurationError, match="RealCellRecoveryTargets"):
        runtime._build_groot_real_controller(_executor_for(dependencies), dependencies)


def test_internal_real_factory_requires_sourced_final_view_registry() -> None:
    import robot.cell.t09_real_runtime as runtime
    from assembly.catalog import load_catalog
    from perception.port.port import CellPerceptionPort
    from robot.cell.real_layout import (
        ExpectedLayoutVerifier,
        load_real_cell_layout,
        reviewed_inspection_registry,
    )
    from robot.cell.real_recovery import RealCellRecoveryTargets

    catalog = load_catalog()
    verifier = ExpectedLayoutVerifier(load_real_cell_layout(), catalog)
    perception = object.__new__(CellPerceptionPort)
    perception._expected_layout_verifier = verifier
    dependencies = replace(
        _dependency_bundle(runtime, perception),
        expected_layout_verifier=verifier,
        inspection_poses=reviewed_inspection_registry(),
        recovery_targets=object.__new__(RealCellRecoveryTargets),
    )

    with pytest.raises(runtime.RealCellFactoryConfigurationError, match="FinalViewRegistry"):
        runtime._build_groot_real_controller(_executor_for(dependencies), dependencies)


def test_pinned_loader_builds_real_authority_with_typed_test_adapters(monkeypatch) -> None:
    import robot.cell.t09_real_runtime as runtime
    from assembly.controller_ports import ExecutionTolerances
    from assembly.groot_controller import GrootControllerAuthority
    from perception.port.port import CellPerceptionPort
    from policy_learn.deploy.cell_executor import T09GrootExecutor
    from robot.cell.t09_mock_runtime import (
        MockCalibratedPoses,
        MockPerception,
        _controller_kwargs,
    )

    class ResetOnlyPolicy:
        last_guard_result = None

        def reset(self):
            return None

        def close(self):
            return None

    class FinalIdentityBackend:
        def capture(self, observation, view, base_instance_id, installed_instance_ids):
            raise AssertionError("construction must not capture final evidence")

    def load_test_adapters(catalog, cad, expected, final_views):
        del catalog, cad, final_views
        perception = object.__new__(CellPerceptionPort)
        perception._expected_layout_verifier = expected
        perception.reset = expected.reset
        mock = _controller_kwargs(MockPerception())
        return runtime.SaturdayRuntimeAdapters(
            perception,
            MockCalibratedPoses(),
            mock["motion_config"],
            mock["safety"],
            mock["verification_thresholds"],
            FinalIdentityBackend(),
        )

    monkeypatch.setattr(runtime, "_load_saturday_runtime_adapters", load_test_adapters)
    dependencies = runtime.load_saturday_real_cell_dependencies()
    executor = T09GrootExecutor(
        policy=ResetOnlyPolicy(),
        safety=dependencies.safety,
        tolerances=ExecutionTolerances(0.3, 1.0),
        require_scene_safety=True,
    )
    controller = runtime._build_groot_real_controller(executor, dependencies)

    assert controller.runtime_authority is GrootControllerAuthority.REAL_CELL
    assert controller.groot_executor is executor
    assert controller._execution_tolerances == ExecutionTolerances(0.3, 1.0)
    assert dependencies.startup_hold is None
    assert dependencies.perception.expected_layout_verifier is dependencies.expected_layout_verifier
    views = dependencies.final_view_registry.views_for(
        5,
        "F05",
        base_xy_mm=(275.0, -123.0),
        max_distance_mm=10.0,
    )
    assert tuple(view.view_id for view in views) == (
        "XY-275--123-view-1",
        "XY-275--123-view-2",
    )


def test_pinned_loader_without_injection_builds_product_five_real_controller(
    monkeypatch,
    tmp_path,
) -> None:
    import robot.cell.t09_real_runtime as runtime
    from assembly.controller_ports import ExecutionTolerances
    from assembly.groot_controller import GrootControllerAuthority
    from assembly.instruction import parse_requested_products
    from policy_learn.deploy.cell_executor import T09GrootExecutor
    from robot.cell.real_layout import load_real_cell_layout
    from robot.cell.saturday_runtime_adapters import (
        RUN_DIR_ENV,
        UNVERIFIED_PRESENCE_ENV,
    )

    class ResetOnlyPolicy:
        last_guard_result = None

        def reset(self):
            return None

        def close(self):
            return None

    monkeypatch.setenv(UNVERIFIED_PRESENCE_ENV, "1")
    monkeypatch.setenv(RUN_DIR_ENV, str(tmp_path))
    dependencies = runtime.load_saturday_real_cell_dependencies()
    executor = T09GrootExecutor(
        policy=ResetOnlyPolicy(),
        safety=dependencies.safety,
        tolerances=ExecutionTolerances(0.3, 1.0),
        require_scene_safety=True,
    )
    controller = runtime._build_groot_real_controller(executor, dependencies)

    assert controller.runtime_authority is GrootControllerAuthority.REAL_CELL
    assert controller.groot_executor is executor
    assert controller._execution_tolerances == ExecutionTolerances(0.3, 1.0)
    assert dependencies.startup_hold is None
    assert dependencies.perception.expected_layout_verifier is dependencies.expected_layout_verifier
    rollout = load_real_cell_layout().rollouts[0]
    assert parse_requested_products(rollout.prompt, dependencies.catalog) == (5,)
    assert {item.object_id for item in rollout.objects} == {
        "F05",
        "M5_01",
        "M5_02",
        "M5_03",
        "M5_04",
    }
    views = dependencies.final_view_registry.views_for(
        5,
        "F05",
        base_xy_mm=(275.0, -123.0),
        max_distance_mm=10.0,
    )
    assert tuple(view.view_id for view in views) == (
        "XY-275--123-view-1",
        "XY-275--123-view-2",
    )


def test_rollout_two_pins_product_five_parts_and_b2_pose() -> None:
    from robot.cell.real_layout import load_real_cell_layout

    layout = load_real_cell_layout()
    rollout = layout.rollout_for_observation(
        {"trial": {"index": 1, "instruction": layout.rollouts[1].prompt}}
    )
    by_id = {item.object_id: item for item in rollout.objects}

    assert (by_id["F05"].center_x_mm, by_id["F05"].center_y_mm, by_id["F05"].yaw_deg) == (
        275.0,
        -123.0,
        -34.0,
    )
    assert set(by_id) == {"F05", "M5_01", "M5_02", "M5_03", "M5_04"}
