"""Behavioral regressions for the trusted final-product verification gate."""

import inspect
from collections import Counter
from dataclasses import replace

import pytest

from assembly.controller import AssemblyController
from assembly.controller_ports import ControllerPortError
from assembly.state_machine import PackMLState
from tests.test_controller import (
    MockFinalProductVerifier,
    ScriptedPerception,
    _controller,
    _drive,
    _observed_products,
)


class MutatingFinalVerifier(MockFinalProductVerifier):
    def __init__(self, **changes) -> None:
        self._changes = changes

    def verify(self, observation, product, base_instance_id, installed_instance_ids):
        evidence = super().verify(observation, product, base_instance_id, installed_instance_ids)
        return replace(evidence, **self._changes)


class TupleSubclass(tuple):
    """Proves boundary validation does not accept tuple lookalikes."""


class MissingVerifyMethod:
    reviewed_motion_source = "reviewed but unavailable workflow"
    mock_only = False


class SelfAttestedProductionVerifier(MockFinalProductVerifier):
    """A callback must not authorize its own physical-view provenance."""

    reviewed_motion_source = "self-attested production views"
    mock_only = False

    def __init__(self) -> None:
        self.called = False

    def verify(self, observation, product, base_instance_id, installed_instance_ids):
        self.called = True
        return super().verify(observation, product, base_instance_id, installed_instance_ids)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"observed_at_s": -1.0}, "current observation"),
        ({"complete": False}, "explicitly complete"),
        ({"product": True}, "product must be an integer"),
        ({"product": 2}, "wrong product or base"),
        ({"base_instance_id": True}, "base instance ID"),
        ({"base_instance_id": "wrong-base"}, "wrong product or base"),
        ({"installed_instance_ids": ("p1-1", "p1-2", "p1-3")}, "exactly four"),
        ({"view_ids": TupleSubclass(("left", "right"))}, "exactly two view IDs"),
        ({"view_ids": ("", "right")}, "nonempty strings"),
        ({"view_ids": ("same", "same")}, "view IDs must be distinct"),
        ({"view_passes": (True, 1)}, "two boolean view results"),
        ({"checks_agree": False}, "checks disagree"),
        ({"motion_source": ""}, "nonempty string"),
        ({"motion_source": "unreviewed"}, "not the reviewed source"),
        ({"failed_instance_id": True}, "failed instance ID"),
        ({"failed_instance_id": "other"}, "not installed"),
    ],
)
def test_final_gate_holds_on_untrusted_evidence(changes, reason) -> None:
    perception = ScriptedPerception(_observed_products((1,)), set(), Counter(), set())
    controller = _controller(perception, final_verifier=MutatingFinalVerifier(**changes))

    _drive(controller, products=(1,))

    assert controller.state_machine.state is PackMLState.HELD
    assert controller.completed_products == ()
    assert reason in controller.events[-1].detail


def test_final_gate_derives_identity_agreement_from_both_views() -> None:
    installed = ("p1-1", "p1-2", "p1-3", "p1-4")
    perception = ScriptedPerception(_observed_products((1,)), set(), Counter(), set())
    verifier = MutatingFinalVerifier(
        view_instance_ids=(installed, ("p1-1", "p1-2", "p1-3", "other")),
    )
    controller = _controller(perception, final_verifier=verifier)

    _drive(controller, products=(1,))

    assert controller.state_machine.state is PackMLState.HELD
    assert controller.completed_products == ()
    assert "view identities do not match" in controller.events[-1].detail


def test_final_gate_completes_only_after_two_bound_views_pass() -> None:
    perception = ScriptedPerception(_observed_products((1,)), set(), Counter(), set())
    controller = _controller(perception)

    _drive(controller, products=(1,))

    assert controller.completed_products == (1,)
    assert controller.finished is True
    assert controller.events[-1].kind == "complete"


def test_mock_final_gate_requires_explicit_mock_authorization() -> None:
    perception = ScriptedPerception(_observed_products((1,)), set(), Counter(), set())

    with pytest.raises(ControllerPortError, match="mock-only final-product verification is forbidden"):
        _controller(perception, allow_mock_final_verification=False)


def test_production_final_gate_cannot_self_attest_physical_view_motion() -> None:
    perception = ScriptedPerception(_observed_products((1,)), set(), Counter(), set())
    verifier = SelfAttestedProductionVerifier()

    with pytest.raises(ControllerPortError, match="controller-owned physical view motion"):
        _controller(
            perception,
            final_verifier=verifier,
            allow_mock_final_verification=False,
        )
    assert verifier.called is False


def test_generic_controller_cannot_claim_controller_owned_view_motion() -> None:
    from tests.test_controller_owned_final_verifier import _verifier

    perception = ScriptedPerception(_observed_products((1,)), set(), Counter(), set())
    verifier, _backend, _views = _verifier()

    with pytest.raises(ControllerPortError, match="controller-owned physical view motion"):
        _controller(
            perception,
            final_verifier=verifier,
            allow_mock_final_verification=False,
        )


def test_generic_controller_has_no_public_controller_owned_authority_flag() -> None:
    assert "controller_owned_final_verification" not in inspect.signature(
        AssemblyController,
    ).parameters


def test_final_gate_holds_when_verifier_has_no_callable_boundary() -> None:
    perception = ScriptedPerception(_observed_products((1,)), set(), Counter(), set())

    with pytest.raises(ControllerPortError, match="callable verify method"):
        _controller(perception, final_verifier=MissingVerifyMethod())


def test_verified_failure_names_reverse_repair_dependency() -> None:
    perception = ScriptedPerception(_observed_products((1,)), set(), Counter(), set())
    verifier = MutatingFinalVerifier(view_passes=(True, False), failed_instance_id="p1-2")
    controller = _controller(perception, final_verifier=verifier)

    _drive(controller, products=(1,))

    assert controller.state_machine.state is PackMLState.HELD
    assert "p1-2" in controller.events[-1].detail
    assert "reviewed installed-part grasp and staging sequence" in controller.events[-1].detail
