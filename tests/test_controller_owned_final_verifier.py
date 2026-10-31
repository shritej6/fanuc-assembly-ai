"""Controller-owned final verification keeps motion out of perception callbacks."""

from __future__ import annotations

from dataclasses import replace

import pytest

from assembly.controller_ports import ControllerPortError
from assembly.final_verification import (
    ControllerOwnedFinalVerifier,
    FinalViewIdentityEvidence,
)
from assembly.final_views import FinalViewPose, FinalViewRegistry


def _view(view_id: str, wrist_yaw: float) -> FinalViewPose:
    return FinalViewPose(
        5,
        "F05",
        view_id,
        (420.0, 0.0, -168.0, -180.0, 0.0, wrist_yaw),
        -215.0,
        ("M5_01", "M5_02", "M5_03", "M5_04"),
        "PRELIMINARY reviewed B2 views",
        (420.0, 0.0),
    )


class IdentityBackend:
    def __init__(self) -> None:
        self.result: FinalViewIdentityEvidence | None = None

    def capture(self, observation, view, base_instance_id, installed_instance_ids):
        del base_instance_id
        if self.result is not None:
            return self.result
        return FinalViewIdentityEvidence(
            float(observation["t"]),
            True,
            view.view_id,
            installed_instance_ids,
            view.expected_instance_roles,
            True,
        )


def _verifier() -> tuple[ControllerOwnedFinalVerifier, IdentityBackend, tuple[FinalViewPose, ...]]:
    views = (_view("B2-left", 45.0), _view("B2-right", 225.0))
    backend = IdentityBackend()
    return (
        ControllerOwnedFinalVerifier(
            FinalViewRegistry(views),
            backend,
            selection_tolerance_mm=10.0,
        ),
        backend,
        views,
    )


def test_two_passive_captures_combine_into_identity_bound_product_evidence() -> None:
    verifier, _backend, views = _verifier()
    instances = ("p5-1", "p5-2", "p5-3", "p5-4")
    captures = tuple(
        verifier.capture_view({"t": float(index)}, view, "base-5", instances)
        for index, view in enumerate(views, start=1)
    )

    evidence = verifier.combine(
        captures,
        observed_at_s=3.0,
        product=5,
        base_instance_id="base-5",
        installed_instance_ids=instances,
    )

    assert evidence.view_ids == ("B2-left", "B2-right")
    assert evidence.view_passes == (True, True)
    assert evidence.checks_agree is True
    assert evidence.motion_source == "PRELIMINARY reviewed B2 views"


def test_capture_refuses_backend_roles_that_do_not_match_the_catalog() -> None:
    verifier, backend, views = _verifier()
    instances = ("p5-1", "p5-2", "p5-3", "p5-4")
    backend.result = FinalViewIdentityEvidence(
        1.0,
        True,
        views[0].view_id,
        instances,
        ("M5_01", "M5_02", "M5_03", "M4_04"),
        True,
    )

    with pytest.raises(ControllerPortError, match="catalog recipe"):
        verifier.capture_view({"t": 1.0}, views[0], "base-5", instances)


def test_capture_refuses_a_passing_result_that_names_a_failed_instance() -> None:
    verifier, backend, views = _verifier()
    instances = ("p5-1", "p5-2", "p5-3", "p5-4")
    backend.result = replace(
        IdentityBackend().capture({"t": 1.0}, views[0], "base-5", instances),
        failed_instance_id="p5-2",
    )

    with pytest.raises(ControllerPortError, match="cannot name a failed"):
        verifier.capture_view({"t": 1.0}, views[0], "base-5", instances)
