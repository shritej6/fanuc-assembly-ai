"""Compose the opt-in real-cell controller from explicitly sourced ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from assembly.cell_runtime_safety import (
    CADCollisionChecker,
    HeldPartSweepSource,
    RobotLinkSweepSource,
    SDGReachabilityChecker,
    SourcedCADGeometry,
)
from assembly.controller import AssemblyController
from assembly.controller_ports import (
    ExecutionTolerances,
    PerceptionPort,
    PerceptionTolerances,
)
from assembly.controller_startup import StartupHoldConfig
from assembly.motion import InspectionStopConfig, MotionConfig
from assembly.planner import CalibratedPoseProvider
from assembly.recovery import RecoveryTargetProvider
from assembly.safety import SafetyConfig, SafetySupervisor
from assembly.verification import VerificationThresholds


@dataclass(frozen=True, slots=True)
class CellRuntimeInputs:
    """All reviewed dependencies required before cell motion can be enabled."""

    catalog: Any
    perception: PerceptionPort
    poses: CalibratedPoseProvider
    motion_config: MotionConfig
    inspection_stop: InspectionStopConfig
    startup_hold: StartupHoldConfig
    safety_config: SafetyConfig
    verification_thresholds: VerificationThresholds
    execution_tolerances: ExecutionTolerances
    perception_tolerances: PerceptionTolerances
    recovery_targets: RecoveryTargetProvider
    robot_links: RobotLinkSweepSource
    held_parts: HeldPartSweepSource


def build_cell_controller(inputs: CellRuntimeInputs) -> AssemblyController:
    """Build a prompt-driven controller with production-only safety providers."""
    geometry = SourcedCADGeometry(inputs.robot_links, inputs.held_parts)
    safety = SafetySupervisor(
        config=inputs.safety_config,
        reachability=SDGReachabilityChecker.from_sdg(),
        collision=CADCollisionChecker(geometry),
    )
    return AssemblyController(
        catalog=inputs.catalog,
        perception=inputs.perception,
        poses=inputs.poses,
        motion_config=inputs.motion_config,
        inspection_stop=inputs.inspection_stop,
        startup_hold=inputs.startup_hold,
        safety=safety,
        verification_thresholds=inputs.verification_thresholds,
        execution_tolerances=inputs.execution_tolerances,
        perception_tolerances=inputs.perception_tolerances,
        recovery_targets=inputs.recovery_targets,
    )
