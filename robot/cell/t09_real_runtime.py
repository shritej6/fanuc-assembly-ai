"""Fail-closed construction boundary for the physical T09/GR00T cell."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from assembly.catalog import Catalog
from assembly.controller_ports import (
    ExecutionTolerances,
    FinalProductVerifier,
    PerceptionTolerances,
)
from assembly.controller_startup import StartupHoldConfig
from assembly.execution import EndpointSettleLimits, GrootWatchdogLimits
from assembly.final_verification import ControllerOwnedFinalVerifier
from assembly.final_views import FinalViewRegistry
from assembly.fixed_inspection import InspectionPoseRegistry
from assembly.groot_controller import GrootAssemblyController, GrootControllerAuthority
from assembly.motion import InspectionStopConfig, MotionConfig
from assembly.planner import CalibratedPoseProvider
from assembly.recovery import RecoveryTargetProvider
from assembly.safety import SafetySupervisor
from assembly.verification import VerificationThresholds
from perception.port.port import CellPerceptionPort
from policy_learn.deploy.cell_executor import T09GrootExecutor
from robot.cell.real_layout import ExpectedLayoutVerifier, reviewed_inspection_registry
from robot.cell.real_recovery import RealCellRecoveryTargets


class RequirementKind(StrEnum):
    """Separate operator measurements from software/runtime dependencies."""

    CSI_CALIBRATION = "csi_calibration"
    RUNTIME_DEPENDENCY = "runtime_dependency"


@dataclass(frozen=True, slots=True)
class RealCellRequirement:
    """One named input that must exist before the real factory is reviewed."""

    code: str
    kind: RequirementKind
    fields: tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        if not self.code or not isinstance(self.kind, RequirementKind):
            raise ValueError("real-cell requirement needs a code and typed kind")
        if not self.fields or len(self.fields) != len(set(self.fields)):
            raise ValueError("real-cell requirement fields must be nonempty and unique")
        if any(not field for field in self.fields) or not self.detail:
            raise ValueError("real-cell requirement fields and detail must be nonempty")


_CSI_REQUIREMENTS = (
    RealCellRequirement(
        "inspection_slots",
        RequirementKind.CSI_CALIBRATION,
        ("slot_id", "selection_bounds_xy_mm", "endpoint_tcp_uf0", "safe_z_mm", "source"),
        "Measure each table/spare slot camera pose with the gripper open.",
    ),
    RealCellRequirement(
        "reject_spots",
        RequirementKind.CSI_CALIBRATION,
        ("spot_id", "release_tcp_uf0", "verification_tcp_uf0", "initially_occupied", "source"),
        "Measure each reject release and verification pose and record occupancy.",
    ),
    RealCellRequirement(
        "staging_slots",
        RequirementKind.CSI_CALIBRATION,
        ("slot_id", "release_tcp_uf0", "verification_tcp_uf0", "initially_occupied", "source"),
        "Measure each dismantle/rebuild staging release and verification pose.",
    ),
    RealCellRequirement(
        "pull_out_targets",
        RequirementKind.CSI_CALIBRATION,
        ("product", "object_id", "set_down_tcp_uf0", "clear_z_mm", "source"),
        "Measure an empty-table set-down pose and absolute clear-Z for each pull-out case.",
    ),
    RealCellRequirement(
        "final_verification_views",
        RequirementKind.CSI_CALIBRATION,
        (
            "product",
            "base_object_id",
            "view_id",
            "endpoint_tcp_uf0",
            "safe_z_mm",
            "expected_instance_roles",
            "source",
        ),
        "Measure two distinct camera views and expected visible identities per product.",
    ),
    RealCellRequirement(
        "motion_clearances",
        RequirementKind.CSI_CALIBRATION,
        (
            "safe_z_mm",
            "approach_mm",
            "lift_mm",
            "hover_mm",
            "retreat_mm",
            "max_linear_step_mm",
            "max_yaw_step_deg",
            "source",
        ),
        "Measure cell clearances and preserve the preliminary Saturday motion limits until replaced.",
    ),
    RealCellRequirement(
        "cell_frames",
        RequirementKind.CSI_CALIBRATION,
        ("uf0_transform", "utool1_transform", "table_surface_z_mm", "source"),
        "Verify UF0, UTOOL1, and the table surface after the physical rebuild.",
    ),
    RealCellRequirement(
        "camera_extrinsics",
        RequirementKind.CSI_CALIBRATION,
        ("table_camera_to_uf0", "wrist_camera_to_utool1", "source"),
        "Verify both camera extrinsics against the rebuilt cell.",
    ),
    RealCellRequirement(
        "base_regions",
        RequirementKind.CSI_CALIBRATION,
        ("region_id", "selection_bounds_xy_mm", "allowed_base_object_ids", "source"),
        "Measure the physical regions in which each interchangeable base may be placed.",
    ),
)


_RUNTIME_REQUIREMENTS = (
    RealCellRequirement(
        "cell_perception_port",
        RequirementKind.RUNTIME_DEPENDENCY,
        ("exact_type", "detector", "classifier", "pose_estimator", "geometry", "image_sink"),
        "Bind the exact CellPerceptionPort with reviewed production backends.",
    ),
    RealCellRequirement(
        "cell_startup_hold_contract",
        RequirementKind.RUNTIME_DEPENDENCY,
        ("constructor_contract", "measured_tcp_only"),
        "Remove the inherited static startup-hold dependency; missing TCP must hold the trial without motion.",
    ),
    RealCellRequirement(
        "measured_pose_bridge",
        RequirementKind.RUNTIME_DEPENDENCY,
        ("fresh_pose_snapshot", "installed_orientations", "foundationpose_frame_boundary"),
        "Publish each fresh CellPerceptionPort pose snapshot to MeasuredRecipePoses.",
    ),
    RealCellRequirement(
        "perception_reset",
        RequirementKind.RUNTIME_DEPENDENCY,
        ("trial_reset",),
        "Reset scanner, tracker, ledger, fitted poses, and evidence caches between trials.",
    ),
    RealCellRequirement(
        "production_final_verifier",
        RequirementKind.RUNTIME_DEPENDENCY,
        ("two_view_motion", "identity_checks", "reviewed_motion_source"),
        "Own guarded motion to two reviewed views and return identity-bound evidence.",
    ),
    RealCellRequirement(
        "final_view_registry",
        RequirementKind.RUNTIME_DEPENDENCY,
        ("exact_type", "two_views_per_product_base", "identity_roles", "source"),
        "Bind the exact sourced final-view registry used by controller-owned verification.",
    ),
    RealCellRequirement(
        "production_safety",
        RequirementKind.RUNTIME_DEPENDENCY,
        ("full_ik", "joint_limits", "robot_link_sweep", "held_part_sweep", "source_hashes"),
        "Provide reviewed production-capable reachability and collision checks.",
    ),
    RealCellRequirement(
        "model_adapters",
        RequirementKind.RUNTIME_DEPENDENCY,
        ("detector_artifact", "classifier_artifact", "foundationpose_service", "thresholds"),
        "Bind reviewed artifacts, service adapters, and measured thresholds.",
    ),
    RealCellRequirement(
        "watchdog_limits",
        RequirementKind.RUNTIME_DEPENDENCY,
        ("per_intent_budget_s", "scene_interval_rows", "base_tilt_limit_deg", "source"),
        "Supply sourced per-intent and periodic perception watchdog limits.",
    ),
    RealCellRequirement(
        "expected_layout_verifier",
        RequirementKind.RUNTIME_DEPENDENCY,
        ("exact_type", "placement_source", "trial_index", "prompt", "start_tolerance"),
        "Verify the reviewed placement once per trial without using it as grasp authority.",
    ),
)


def required_real_cell_requirements() -> tuple[RealCellRequirement, ...]:
    """Return the complete, immutable real-cell readiness manifest."""
    return (*_CSI_REQUIREMENTS, *_RUNTIME_REQUIREMENTS)


class RealCellFactoryConfigurationError(TypeError):
    """A caller attempted to bypass the exact reviewed factory types."""


class RealCellFactoryBlockedError(RuntimeError):
    """The physical controller cannot be built from reviewed dependencies."""


@dataclass(frozen=True, slots=True)
class RealCellDependencies:
    """Explicit reviewed inputs needed by the physical controller constructor."""

    catalog: Catalog
    perception: CellPerceptionPort
    poses: CalibratedPoseProvider
    motion_config: MotionConfig
    inspection_stop: InspectionStopConfig
    startup_hold: StartupHoldConfig
    safety: SafetySupervisor
    verification_thresholds: VerificationThresholds
    execution_tolerances: ExecutionTolerances
    perception_tolerances: PerceptionTolerances
    recovery_targets: RecoveryTargetProvider
    final_verifier: FinalProductVerifier
    inspection_poses: InspectionPoseRegistry
    endpoint_settle_limits: EndpointSettleLimits
    watchdog_limits: GrootWatchdogLimits
    expected_layout_verifier: ExpectedLayoutVerifier
    final_view_registry: FinalViewRegistry


# Source review must replace this with a committed loader. Environment variables
# cannot provide or self-authorize physical dependencies.
REVIEWED_REAL_CELL_DEPENDENCY_LOADER: Callable[[], RealCellDependencies] | None = None


def _missing_requirements_message() -> str:
    fields = "; ".join(
        f"{item.code}=[{', '.join(item.fields)}]"
        for item in required_real_cell_requirements()
    )
    return f"no reviewed real-cell dependency loader is pinned; required inputs: {fields}"


def _require_executor(executor: object) -> T09GrootExecutor:
    if type(executor) is not T09GrootExecutor:
        raise RealCellFactoryConfigurationError(
            "real CELL factory requires the exact T09GrootExecutor type"
        )
    return executor


def build_groot_real_controller(executor: object) -> GrootAssemblyController:
    """Build the exact physical controller, or fail before any motion authority exists."""
    safe_executor = _require_executor(executor)
    loader = REVIEWED_REAL_CELL_DEPENDENCY_LOADER
    if loader is None:
        raise RealCellFactoryBlockedError(_missing_requirements_message())
    return _build_groot_real_controller(safe_executor, loader())


def _build_groot_real_controller(
    executor: object,
    dependencies: object,
) -> GrootAssemblyController:
    safe_executor = _require_executor(executor)
    if type(dependencies) is not RealCellDependencies:
        raise RealCellFactoryConfigurationError(
            "real CELL factory requires the exact RealCellDependencies type"
        )
    if type(dependencies.perception) is not CellPerceptionPort:
        raise RealCellFactoryConfigurationError(
            "real CELL factory requires the exact CellPerceptionPort type"
        )
    verifier = dependencies.expected_layout_verifier
    if type(verifier) is not ExpectedLayoutVerifier:
        raise RealCellFactoryConfigurationError(
            "real CELL factory requires the exact ExpectedLayoutVerifier type"
        )
    if dependencies.perception.expected_layout_verifier is not verifier:
        raise RealCellFactoryConfigurationError(
            "real CELL perception must retain the reviewed expected-layout verifier"
        )
    if dependencies.inspection_poses != reviewed_inspection_registry():
        raise RealCellFactoryConfigurationError(
            "real CELL factory requires the reviewed T38 fixed inspection registry"
        )
    if type(dependencies.recovery_targets) is not RealCellRecoveryTargets:
        raise RealCellFactoryConfigurationError(
            "real CELL factory requires the exact RealCellRecoveryTargets type"
        )
    if type(dependencies.final_view_registry) is not FinalViewRegistry:
        raise RealCellFactoryConfigurationError(
            "real CELL factory requires the exact FinalViewRegistry type"
        )
    if type(dependencies.final_verifier) is not ControllerOwnedFinalVerifier:
        raise RealCellFactoryConfigurationError(
            "real CELL factory requires the controller-owned final verifier"
        )
    if dependencies.final_verifier.registry is not dependencies.final_view_registry:
        raise RealCellFactoryConfigurationError(
            "real CELL final verifier must retain the reviewed FinalViewRegistry"
        )
    return GrootAssemblyController(
        groot_executor=safe_executor,
        inspection_poses=dependencies.inspection_poses,
        endpoint_settle_limits=dependencies.endpoint_settle_limits,
        watchdog_limits=dependencies.watchdog_limits,
        runtime_authority=GrootControllerAuthority.REAL_CELL,
        final_view_registry=dependencies.final_view_registry,
        catalog=dependencies.catalog,
        perception=dependencies.perception,
        poses=dependencies.poses,
        motion_config=dependencies.motion_config,
        inspection_stop=dependencies.inspection_stop,
        startup_hold=dependencies.startup_hold,
        safety=dependencies.safety,
        verification_thresholds=dependencies.verification_thresholds,
        execution_tolerances=dependencies.execution_tolerances,
        perception_tolerances=dependencies.perception_tolerances,
        recovery_targets=dependencies.recovery_targets,
        final_verifier=dependencies.final_verifier,
        allow_mock_final_verification=False,
    )
