"""Strict production entrypoint for the T09 + GR00T physical-cell runtime.

The repo contains the independently tested controller, learned-policy guard, and
semantic executor boundary.  It does not yet contain reviewed physical providers or a
T09 semantic queue that can preserve the hold-before-release protocol.  This module is
therefore the single explicit CELL gate; it never substitutes mock or scripted motion.
"""

from __future__ import annotations

from dataclasses import dataclass

from robot import Controller


class CellRuntimeBlockedError(RuntimeError):
    """The physical-cell dependency graph is incomplete or unreviewed."""


@dataclass(frozen=True, slots=True)
class RuntimeBlocker:
    code: str
    owner: str
    detail: str


_BLOCKERS = (
    RuntimeBlocker(
        "semantic_t09_executor",
        "T09/T12 integration",
        "T09 still queues scripted MotionChunk values; every CELL phase must instead emit a MotionIntent and "
        "consume T09GrootExecutor results, including a controller-owned, identity/evidence-bound release token; "
        "the current release_authorized Boolean is not physical authority.",
    ),
    RuntimeBlocker(
        "served_checkpoint_attestation",
        "T12",
        "The served checkpoint must attest all required semantic skills, exact hashes, one-row horizon, "
        "evaluation approval, and the T09 release protocol. The launcher must also bind every action to the exact "
        "supervised child process and an immutable content-addressed checkpoint; a static proxy envelope is not "
        "live process identity.",
    ),
    RuntimeBlocker(
        "measured_motion_limits",
        "CSI calibration",
        "GuardLimits must be MEASURED with a recorded source; preliminary speed, force, and contact values "
        "cannot authorize physical motion.",
    ),
    RuntimeBlocker(
        "live_perception_backends",
        "T28/T30",
        "The complete live detector, classifier, FoundationPose/depth, final-view verifier, and collision scene "
        "must be configured with their reviewed artifacts.",
    ),
    RuntimeBlocker(
        "reviewed_physical_safety",
        "T28/T29/CSI",
        "Full IK and joint limits plus robot-link and held-part swept geometry must match reviewed immutable "
        "physical-cell sources.",
    ),
)


def runtime_blockers() -> tuple[RuntimeBlocker, ...]:
    """Return immutable, actionable reasons CELL authorization is withheld."""
    return _BLOCKERS


def create_cell_policy(*, checkpoint: str | None) -> Controller:
    """Create the physical controller only after every fail-closed gate is implemented.

    ``checkpoint`` is accepted so ``robot/cell/policy.py`` has the final stable API.
    It is never interpreted as permission to bypass missing physical dependencies.
    """
    del checkpoint
    details = "; ".join(f"{item.code}: {item.detail}" for item in _BLOCKERS)
    raise CellRuntimeBlockedError(f"CELL runtime is not armed: {details}")
