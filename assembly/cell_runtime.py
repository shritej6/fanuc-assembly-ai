"""CELL entry point that pairs controller decisions with guarded GR00T motion.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from importlib import import_module

from assembly.controller_ports import ExecutionTolerances
from assembly.execution import EndpointSettleLimits, GrootWatchdogLimits
from assembly.groot_controller import GrootAssemblyController, GrootControllerAuthority
from assembly.safety import SafetySupervisor
from policy_learn.deploy.cell_executor import T09GrootExecutor
from policy_learn.deploy.cell_groot_policy import GrootPolicy
from policy_learn.deploy.cell_guards import GuardLimits
from robot import Controller

CELL_FACTORY_ENV = "FANUC_T09_CELL_FACTORY"
SOFTWARE_PROOF_ENV = "FANUC_T09_SOFTWARE_PROOF"
MOCK_PROOF_FACTORY = "robot.cell.t09_mock_runtime:build_groot_mock_controller"
# No reviewed production factory exists yet. Adding one requires a reviewed source
# change; an environment variable cannot self-authorize.
REVIEWED_REAL_CELL_FACTORY: str | None = None
SATURDAY_EXECUTION_TOLERANCES = ExecutionTolerances(
    0.3,
    1.0,
)  # Precision gates from sim/assembly_lab/INTERFACES.md.
SATURDAY_ENDPOINT_SETTLE_LIMITS = EndpointSettleLimits(
    3.0,
    3.0,
    "PRELIMINARY",
    "T38 realistic-arm rehearsal: 2.5 s failed and 3.0 s cleared TCP settlement at the unchanged "
    "0.3 mm / 1 deg gate; pending CSI TCP/gripper measurement",
)
SATURDAY_GROOT_WATCHDOG_LIMITS = GrootWatchdogLimits(
    pick_insert_s=90.0,
    release_retreat_s=30.0,
    pull_out_s=60.0,
    defect_drop_s=60.0,
    inspection_s=30.0,
    scene_interval_rows=5,
    base_tilt_limit_deg=3.0,
    status="PRELIMINARY",
    source=(
        "Engineering estimate: 60 s pre-grasp, 90 s post-grasp insertion, "
        "and collision-scene evidence at least every 1 s; pending CSI measurement"
    ),
    pick_s=60.0,
    insert_s=90.0,
    scene_interval_s=1.0,
)


class CellRuntimeBlockedError(RuntimeError):
    """A required CELL dependency is absent or unsafe."""


class CellTransport(str, Enum):
    """Identity of the official client transport hosting the CELL policy."""

    REAL = "real"
    OFFICIAL_MOCK = "official_mock"


@dataclass(frozen=True, slots=True)
class RuntimeBlocker:
    """Known deferred risk intentionally excluded from motion authority."""

    code: str
    owner: str
    detail: str


_KNOWN_RISKS = (
    RuntimeBlocker(
        "checkpoint_attestation",
        "T12 Sunday hardening",
        "Immutable checkpoint/process attestation is deferred.",
    ),
    RuntimeBlocker(
        "live_collision_backends",
        "T28/T29 Sunday hardening",
        "Live collision geometry is not Saturday motion authority.",
    ),
    RuntimeBlocker(
        "full_robot_sweeps",
        "T28/T29 Sunday hardening",
        "Full IK, joint, robot-link, and held-part sweeps are deferred.",
    ),
    RuntimeBlocker(
        "measured_motion_limits",
        "CSI calibration",
        "Step and speed limits remain preliminary.",
    ),
    RuntimeBlocker(
        "identity_bound_release_tokens",
        "T09/T12 Sunday hardening",
        "Saturday keeps the one-step release fence; identity-bound tokens are deferred.",
    ),
    RuntimeBlocker(
        "live_force_torque",
        "user-accepted Saturday risk, 2026-10-31",
        "The official observation has no force/torque channel; the supervised Saturday run relies on existing "
        "workspace/tilt/step/speed/contact-floor guards, slow insertion, FANUC's external collision stop, and the "
        "physical E-stop. This does not waive any other CELL gate.",
    ),
    RuntimeBlocker(
        "cosmos_pin_disabled",
        "T18 Sunday hardening",
        "Cosmos is advisory and disabled in T37.",
    ),
)


def runtime_blockers() -> tuple[RuntimeBlocker, ...]:
    """Return known risks accepted for the current CELL scope."""
    return _KNOWN_RISKS


def _saturday_limits() -> GuardLimits:
    return GuardLimits(
        max_tilt_deg=6.0,
        status="PRELIMINARY",
        source="2026-10-31 supervised Saturday 6-degree fingers-down cone, mirrored by CellAdapter",
    )


def _resolve_factory(transport: CellTransport) -> Callable[[T09GrootExecutor], Controller]:
    if transport is CellTransport.REAL and SOFTWARE_PROOF_ENV in os.environ:
        raise CellRuntimeBlockedError(
            f"real CELL transport refuses the software-proof flag {SOFTWARE_PROOF_ENV}"
        )
    spec = os.environ.get(CELL_FACTORY_ENV, "").strip()
    if not spec:
        raise CellRuntimeBlockedError(
            f"CELL runtime needs {CELL_FACTORY_ENV}=<module>:<factory>; no mock fallback is allowed"
        )
    if transport is CellTransport.REAL and spec != REVIEWED_REAL_CELL_FACTORY:
        raise CellRuntimeBlockedError(
            "real CELL transport has no reviewed real-cell factory pin in this source revision"
        )
    if spec == MOCK_PROOF_FACTORY:
        if transport is not CellTransport.OFFICIAL_MOCK:
            raise CellRuntimeBlockedError(
                f"{MOCK_PROOF_FACTORY} is forbidden on the real CELL transport"
            )
        if os.environ.get(SOFTWARE_PROOF_ENV) != "1":
            raise CellRuntimeBlockedError(
                f"{MOCK_PROOF_FACTORY} is software-proof-only; set {SOFTWARE_PROOF_ENV}=1 only with mock_cell"
            )
    elif transport is CellTransport.OFFICIAL_MOCK:
        raise CellRuntimeBlockedError(
            f"official-mock transport requires the reviewed proof factory {MOCK_PROOF_FACTORY}"
        )
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise CellRuntimeBlockedError(f"{CELL_FACTORY_ENV} must use <module>:<factory> syntax")
    try:
        factory = getattr(import_module(module_name), attribute)
    except (AttributeError, ImportError) as error:
        raise CellRuntimeBlockedError(f"CELL controller factory {spec!r} is unavailable") from error
    if not callable(factory):
        raise CellRuntimeBlockedError(f"CELL controller factory {spec!r} is not callable")
    return factory


def create_cell_policy(
    *,
    checkpoint: str | None,
    transport: CellTransport = CellTransport.REAL,
) -> Controller:
    """Build the explicit controller to semantic executor to guarded GR00T CELL path."""
    if not isinstance(transport, CellTransport):
        raise CellRuntimeBlockedError("CELL transport must be an explicit CellTransport value")
    if not isinstance(checkpoint, str) or not checkpoint.startswith("groot://"):
        raise CellRuntimeBlockedError("CELL runtime requires an explicit groot://host:port server URL")
    factory = _resolve_factory(transport)
    policy = GrootPolicy(checkpoint, limits=_saturday_limits(), saturday_cell=True)
    executor = T09GrootExecutor(
        policy=policy,
        safety=None,
        tolerances=SATURDAY_EXECUTION_TOLERANCES,
        require_scene_safety=False,
    )
    try:
        controller = factory(executor)
        _validate_controller(controller, executor, transport)
    except CellRuntimeBlockedError:
        _close_failed_executor(executor)
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        _close_failed_executor(executor)
        raise CellRuntimeBlockedError(f"CELL controller factory failed: {error}") from error
    return controller


def _validate_controller(
    controller: Controller,
    executor: T09GrootExecutor,
    transport: CellTransport,
) -> None:
    if type(controller) is not GrootAssemblyController:
        raise CellRuntimeBlockedError(
            "CELL controller factory must return the exact GrootAssemblyController type"
        )
    if controller.groot_executor is not executor:
        raise CellRuntimeBlockedError(
            "CELL controller factory did not retain the exact guarded executor"
        )
    expected_authority = (
        GrootControllerAuthority.REAL_CELL
        if transport is CellTransport.REAL
        else GrootControllerAuthority.OFFICIAL_MOCK_PROOF
    )
    if controller.runtime_authority is not expected_authority:
        raise CellRuntimeBlockedError(
            "CELL controller runtime authority does not match the selected transport"
        )
    if transport is CellTransport.REAL and (
        getattr(executor, "_require_scene_safety", None) is not True
        or not isinstance(getattr(executor, "_safety", None), SafetySupervisor)
    ):
        raise CellRuntimeBlockedError(
            "real CELL controller factory returned without reviewed scene safety"
        )


def _close_failed_executor(executor: T09GrootExecutor) -> None:
    try:
        executor.close()
    except (OSError, RuntimeError, ValueError) as error:
        raise CellRuntimeBlockedError(
            f"CELL factory rejection could not close the GR00T transport: {error}"
        ) from error
