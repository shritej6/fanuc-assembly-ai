"""Passive identity evidence for controller-owned physical final views."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from assembly.controller_ports import (
    ControllerPortError,
    FinalProductEvidence,
    call_port,
    observation_time,
    require_complete,
    require_fresh,
)
from assembly.final_views import FinalViewPose, FinalViewRegistry
from robot.adapter import Observation


@dataclass(frozen=True, slots=True)
class FinalViewIdentityEvidence:
    """Identity result captured after one guarded view endpoint settles."""

    observed_at_s: float
    complete: bool
    view_id: str
    instance_ids: tuple[str, str, str, str]
    object_ids: tuple[str, str, str, str]
    passed: bool
    failed_instance_id: str | None = None


@dataclass(frozen=True, slots=True)
class PendingFinalVerification:
    """Controller state spanning two guarded view motions and captures."""

    product: int
    base_instance_id: str
    installed_instance_ids: tuple[str, str, str, str]
    views: tuple[FinalViewPose, FinalViewPose]
    captures: tuple[FinalViewIdentityEvidence, ...] = ()
    awaiting_capture_index: int | None = None


class FinalViewIdentityBackend(Protocol):
    """Perception-only boundary; implementations have no motion authority."""

    def capture(
        self,
        observation: Observation,
        view: FinalViewPose,
        base_instance_id: str,
        installed_instance_ids: tuple[str, str, str, str],
    ) -> FinalViewIdentityEvidence: ...


class ControllerOwnedFinalVerifier:
    """Validate two passive captures made by the guarded controller state machine."""

    mock_only = False

    def __init__(
        self,
        registry: FinalViewRegistry,
        backend: FinalViewIdentityBackend,
        *,
        selection_tolerance_mm: float,
    ) -> None:
        if type(registry) is not FinalViewRegistry:
            raise ControllerPortError("controller-owned final verifier requires FinalViewRegistry")
        if not callable(getattr(backend, "capture", None)):
            raise ControllerPortError("controller-owned final verifier requires an identity capture backend")
        if (
            isinstance(selection_tolerance_mm, bool)
            or not isinstance(selection_tolerance_mm, (int, float))
            or not math.isfinite(float(selection_tolerance_mm))
            or selection_tolerance_mm < 0.0
        ):
            raise ControllerPortError("final-view selection tolerance must be finite and nonnegative")
        sources = {view.source for view in registry.views}
        if len(sources) != 1:
            raise ControllerPortError("controller-owned final views must share one reviewed motion source")
        self.registry = registry
        self._backend = backend
        self.selection_tolerance_mm = float(selection_tolerance_mm)
        self.reviewed_motion_source = next(iter(sources))

    def capture_view(
        self,
        observation: Observation,
        view: FinalViewPose,
        base_instance_id: str,
        installed_instance_ids: tuple[str, str, str, str],
    ) -> FinalViewIdentityEvidence:
        evidence = call_port(
            "final-view identity capture",
            self._backend.capture,
            observation,
            view,
            base_instance_id,
            installed_instance_ids,
        )
        return self._validate_capture(
            evidence,
            observation_time(observation),
            view,
            installed_instance_ids,
        )

    @staticmethod
    def _validate_capture(
        evidence: object,
        observed_at_s: float,
        view: FinalViewPose,
        installed_instance_ids: tuple[str, str, str, str],
    ) -> FinalViewIdentityEvidence:
        if type(evidence) is not FinalViewIdentityEvidence:
            raise ControllerPortError("final-view backend must return FinalViewIdentityEvidence")
        require_fresh(evidence.observed_at_s, observed_at_s, "final-view identity evidence")
        require_complete(evidence.complete, "final-view identity evidence")
        if evidence.view_id != view.view_id:
            raise ControllerPortError("final-view identity evidence is bound to the wrong view")
        if evidence.instance_ids != installed_instance_ids:
            raise ControllerPortError("final-view identities do not match installed instances")
        if evidence.object_ids != view.expected_instance_roles:
            raise ControllerPortError("final-view object roles do not match the catalog recipe")
        if type(evidence.passed) is not bool:
            raise ControllerPortError("final-view pass must be a boolean")
        if evidence.failed_instance_id not in {None, *installed_instance_ids}:
            raise ControllerPortError("final-view failed instance is not installed")
        if evidence.passed and evidence.failed_instance_id is not None:
            raise ControllerPortError("passing final-view evidence cannot name a failed instance")
        return evidence

    def combine(
        self,
        captures: tuple[FinalViewIdentityEvidence, FinalViewIdentityEvidence],
        *,
        observed_at_s: float,
        product: int,
        base_instance_id: str,
        installed_instance_ids: tuple[str, str, str, str],
    ) -> FinalProductEvidence:
        if type(captures) is not tuple or len(captures) != 2:
            raise ControllerPortError("controller-owned final verification requires two captures")
        if any(type(capture) is not FinalViewIdentityEvidence for capture in captures):
            raise ControllerPortError("controller-owned final captures must be typed evidence")
        failed = next((capture.failed_instance_id for capture in captures if not capture.passed), None)
        identities = tuple(capture.instance_ids for capture in captures)
        return FinalProductEvidence(
            observed_at_s,
            True,
            product,
            base_instance_id,
            installed_instance_ids,
            tuple(capture.view_id for capture in captures),
            identities,
            tuple(capture.passed for capture in captures),
            identities[0] == identities[1] == installed_instance_ids,
            self.reviewed_motion_source,
            failed,
        )
