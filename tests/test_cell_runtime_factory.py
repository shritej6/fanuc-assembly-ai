"""Saturday CELL factory is explicit, GR00T-only, and fail closed."""

from __future__ import annotations

import pytest

import assembly.cell_runtime as runtime
from assembly.groot_controller import GrootAssemblyController


class _FakeGrootPolicy:
    def __init__(self, checkpoint, limits, *, saturday_cell):
        self.checkpoint = checkpoint
        self.limits = limits
        self.saturday_cell = saturday_cell
        self.last_guard_result = None
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


def test_runtime_risks_name_every_explicit_sunday_deferral() -> None:
    codes = {risk.code for risk in runtime.runtime_blockers()}

    assert codes == {
        "checkpoint_attestation",
        "live_collision_backends",
        "full_robot_sweeps",
        "measured_motion_limits",
        "identity_bound_release_tokens",
        "live_force_torque",
        "cosmos_pin_disabled",
    }
    assert all(risk.owner and risk.detail for risk in runtime.runtime_blockers())


@pytest.mark.parametrize("checkpoint", [None, "stub", "checkpoint.pt", "http://127.0.0.1:5555"])
def test_cell_factory_requires_explicit_groot_server_url(checkpoint: str | None) -> None:
    with pytest.raises(runtime.CellRuntimeBlockedError, match="explicit groot://"):
        runtime.create_cell_policy(checkpoint=checkpoint)


def test_cell_factory_never_uses_an_implicit_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime.CELL_FACTORY_ENV, raising=False)

    with pytest.raises(runtime.CellRuntimeBlockedError, match=runtime.CELL_FACTORY_ENV):
        runtime.create_cell_policy(checkpoint="groot://127.0.0.1:5555")


def test_mock_factory_is_proof_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runtime.CELL_FACTORY_ENV, runtime.MOCK_PROOF_FACTORY)
    monkeypatch.delenv(runtime.SOFTWARE_PROOF_ENV, raising=False)

    with pytest.raises(runtime.CellRuntimeBlockedError, match="software-proof-only"):
        runtime.create_cell_policy(checkpoint="groot://127.0.0.1:5555")


def test_groot_url_arms_t09_semantic_controller_for_explicit_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(runtime.CELL_FACTORY_ENV, runtime.MOCK_PROOF_FACTORY)
    monkeypatch.setenv(runtime.SOFTWARE_PROOF_ENV, "1")
    monkeypatch.setattr(runtime, "GrootPolicy", _FakeGrootPolicy)

    controller = runtime.create_cell_policy(checkpoint="groot://127.0.0.1:5555")

    assert isinstance(controller, GrootAssemblyController)
    policy = controller._groot_executor._policy
    assert policy.checkpoint == "groot://127.0.0.1:5555"
    assert policy.saturday_cell is True
    assert policy.limits.max_tilt_deg == 3.0
    assert policy.resets == 1
