"""Typed final-product verification and PackML completion."""

from __future__ import annotations

from assembly.controller_ports import (
    ControllerEvent,
    ControllerPortError,
    FinalProductEvidence,
    call_port,
    require_complete,
    require_fresh,
)
from assembly.final_verification import ControllerOwnedFinalVerifier
from assembly.state_machine import PackMLState
from robot.adapter import Observation


def validate_final_product_verifier(
    verifier: object,
    *,
    allow_mock: bool,
    controller_owned: bool = False,
) -> str:
    source = getattr(verifier, "reviewed_motion_source", None)
    mock_only = getattr(verifier, "mock_only", None)
    if type(source) is not str or not source.strip():
        raise ControllerPortError("final-product verifier requires reviewed motion-source provenance")
    if type(mock_only) is not bool:
        raise ControllerPortError("final-product verifier mock_only must be a boolean")
    if type(verifier) is ControllerOwnedFinalVerifier:
        if not controller_owned:
            raise ControllerPortError(
                "production final-product verification requires controller-owned physical view motion"
            )
        return source
    if not callable(getattr(verifier, "verify", None)):
        raise ControllerPortError("final-product verifier requires a callable verify method")
    if mock_only and not allow_mock:
        raise ControllerPortError("mock-only final-product verification is forbidden in production")
    if not mock_only:
        raise ControllerPortError(
            "production final-product verification requires controller-owned physical view motion"
        )
    return source


def _validate_views(evidence: FinalProductEvidence) -> None:
    view_ids = evidence.view_ids
    if type(view_ids) is not tuple or len(view_ids) != 2:
        raise ControllerPortError("final-product evidence requires exactly two view IDs")
    if any(type(view_id) is not str or not view_id.strip() for view_id in view_ids):
        raise ControllerPortError("final-product view IDs must be nonempty strings")
    if view_ids[0] == view_ids[1]:
        raise ControllerPortError("final-product view IDs must be distinct")
    passes = evidence.view_passes
    if type(passes) is not tuple or len(passes) != 2 or any(type(value) is not bool for value in passes):
        raise ControllerPortError("final-product evidence requires two boolean view results")


def _require_instance_ids(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) != 4:
        raise ControllerPortError(f"{label} must contain exactly four instance IDs")
    if any(type(instance_id) is not str or not instance_id.strip() for instance_id in value):
        raise ControllerPortError(f"{label} must contain nonempty strings")
    if len(set(value)) != len(value):
        raise ControllerPortError(f"{label} must contain unique instance IDs")
    return value


def _validate_view_identities(
    evidence: FinalProductEvidence,
    installed_instance_ids: tuple[str, ...],
) -> None:
    views = evidence.view_instance_ids
    if type(views) is not tuple or len(views) != 2:
        raise ControllerPortError("final-product evidence requires identities for exactly two views")
    for view in views:
        identities = _require_instance_ids(view, "final-product view identities")
        if identities != installed_instance_ids:
            raise ControllerPortError("final-product view identities do not match installed instances")


def validate_final_product_evidence(
    evidence: object,
    *,
    observed_at: float,
    product: int,
    base_instance_id: str,
    installed_instance_ids: tuple[str, ...],
    motion_source: str,
) -> FinalProductEvidence:
    if type(evidence) is not FinalProductEvidence:
        raise ControllerPortError("final-product verifier must return FinalProductEvidence")
    require_fresh(evidence.observed_at_s, observed_at, "final-product evidence")
    require_complete(evidence.complete, "final-product evidence")
    if type(evidence.product) is not int:
        raise ControllerPortError("final-product evidence product must be an integer")
    if type(evidence.base_instance_id) is not str or not evidence.base_instance_id.strip():
        raise ControllerPortError("final-product evidence base instance ID must be a nonempty string")
    expected_ids = _require_instance_ids(installed_instance_ids, "active installed instances")
    evidence_ids = _require_instance_ids(evidence.installed_instance_ids, "final-product installed instances")
    if evidence.product != product or evidence.base_instance_id != base_instance_id:
        raise ControllerPortError("final-product evidence is bound to the wrong product or base")
    if evidence_ids != expected_ids:
        raise ControllerPortError("final-product evidence is bound to the wrong installed instances")
    _validate_views(evidence)
    _validate_view_identities(evidence, expected_ids)
    if evidence.checks_agree is not True:
        raise ControllerPortError("final-product two-view identity checks disagree")
    if type(evidence.motion_source) is not str or not evidence.motion_source.strip():
        raise ControllerPortError("final-product evidence motion source must be a nonempty string")
    if evidence.motion_source != motion_source:
        raise ControllerPortError("final-product evidence motion source is not the reviewed source")
    failed = evidence.failed_instance_id
    if failed is not None and (type(failed) is not str or not failed.strip()):
        raise ControllerPortError("final-product failed instance ID must be a nonempty string")
    if failed is not None and failed not in expected_ids:
        raise ControllerPortError("final-product failed instance is not installed in this product")
    if all(evidence.view_passes) and failed is not None:
        raise ControllerPortError("passing final-product views cannot identify a failed instance")
    return evidence


class CompletionControllerMixin:
    """Complete a product only after a trusted two-view gate passes."""

    def _verify_finished_product(self, observation: Observation, observed_at: float) -> bool:
        product = self.state_machine.current_product
        if product is None:
            raise RuntimeError("final-product verification requires an active product")
        reservation = self.inventory.active_reservation
        installed_ids = tuple(part.instance_id for part in reservation.parts)
        evidence = call_port(
            "final-product verification",
            self._final_verifier.verify,
            observation,
            product,
            reservation.base.instance_id,
            installed_ids,
        )
        verified = validate_final_product_evidence(
            evidence,
            observed_at=observed_at,
            product=product,
            base_instance_id=reservation.base.instance_id,
            installed_instance_ids=installed_ids,
            motion_source=self._final_motion_source,
        )
        if not all(verified.view_passes):
            self._hold(self._finished_failure_reason(product, verified.failed_instance_id))
            return False
        self._finish_product(verified)
        return False

    @staticmethod
    def _finished_failure_reason(product: int, failed_instance_id: str | None) -> str:
        failed = failed_instance_id or "unidentified installed instance"
        return (
            f"finished-product verification failed for product {product}: {failed}; "
            "reverse repair requires a reviewed installed-part grasp and staging sequence"
        )

    def _finish_product(self, evidence: FinalProductEvidence) -> None:
        product = self.state_machine.current_product
        if product is None:
            raise RuntimeError("completed PackML cycle has no product owner")
        self.inventory.complete_product(product)
        self.state_machine.complete()
        self.state_machine.transition(PackMLState.COMPLETE)
        self._completed.append(product)
        detail = f"two verified views {evidence.view_ids[0]}, {evidence.view_ids[1]}"
        self._events.append(ControllerEvent("complete", product, None, detail))
