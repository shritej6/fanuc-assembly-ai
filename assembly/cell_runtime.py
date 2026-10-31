"""Saturday CELL entrypoint for T09 decisions and guarded GR00T motion.

The controller factory is supplied explicitly because the repository has more
than one perception/calibration deployment. CELL never substitutes the local
mock or scripted motion when that production dependency is absent.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from importlib import import_module

from assembly.controller_ports import ExecutionTolerances
from assembly.groot_controller import GrootAssemblyController
from policy_learn.deploy.cell_executor import T09GrootExecutor
from policy_learn.deploy.groot_policy import GrootPolicy
from policy_learn.deploy.guards import GuardLimits
from robot import Controller

CELL_FACTORY_ENV = "FANUC_T09_CELL_FACTORY"
SOFTWARE_PROOF_ENV = "FANUC_T09_SOFTWARE_PROOF"
MOCK_PROOF_FACTORY = "robot.cell.t09_mock_runtime:build_groot_mock_controller"
SATURDAY_EXECUTION_TOLERANCES = ExecutionTolerances(
    0.3,
    1.0,
)  # sim/assembly_lab/INTERFACES.md precision gates, user-approved for Saturday.


class CellRuntimeBlockedError(RuntimeError):
    """A required Saturday CELL dependency is absent or unsafe."""


class CellTransport(str, Enum):
    """Identity of the official client transport hosting the CELL policy."""

    REAL = "real"
    OFFICIAL_MOCK = "official_mock"


@dataclass(frozen=True, slots=True)
class RuntimeBlocker:
    """Known post-Saturday risk intentionally excluded from motion authority."""

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
        "cell hardware/API",
        "The official observation has no force/torque channel.",
    ),
    RuntimeBlocker(
        "cosmos_pin_disabled",
        "T18 Sunday hardening",
        "Cosmos is advisory and disabled in T37.",
    ),
)


def runtime_blockers() -> tuple[RuntimeBlocker, ...]:
    """Return known risks accepted by the user's Saturday scope decision."""
    return _KNOWN_RISKS


def _saturday_limits() -> GuardLimits:
    return GuardLimits(
        max_tilt_deg=3.0,
        status="PRELIMINARY",
        source="T05 final CellAdapter W/P envelope, mirrored before CELL dispatch",
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
    """Build the explicit T09 -> semantic executor -> guarded GR00T CELL path."""
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
    except CellRuntimeBlockedError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise CellRuntimeBlockedError(f"CELL controller factory failed: {error}") from error
    if not isinstance(controller, GrootAssemblyController):
        raise CellRuntimeBlockedError(
            "CELL controller factory must return a GrootAssemblyController"
        )
    if controller.groot_executor is not executor:
        raise CellRuntimeBlockedError(
            "CELL controller factory did not retain the exact guarded executor"
        )
    return controller
