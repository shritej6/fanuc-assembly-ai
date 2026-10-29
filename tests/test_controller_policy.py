"""Policy-selection behavior independent of the assembly scenarios."""

from pathlib import Path

import pytest


class _TcpHold:
    def reset(self) -> None:
        pass

    def act(self, observation):
        del observation
        return [400.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0]


def _policy_type(monkeypatch):
    cell_dir = Path(__file__).resolve().parents[1] / "robot" / "cell"
    monkeypatch.syspath_prepend(str(cell_dir))
    from policy import Policy

    return Policy


def test_policy_can_opt_into_tcp_controller_without_changing_safe_default(monkeypatch) -> None:
    policy_type = _policy_type(monkeypatch)

    observation = {"joints": [0.0] * 7, "tcp_pose": None}
    policy = policy_type(controller=_TcpHold(), action_mode="tcp")

    assert policy_type.ACTION == "joints"
    assert policy.ACTION == "tcp"
    assert policy.act(observation) == [400.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0]


def test_t09_cell_checkpoint_requires_cell_runtime(monkeypatch) -> None:
    policy_type = _policy_type(monkeypatch)

    with pytest.raises(ValueError, match="forbidden outside the real cell runtime"):
        policy_type("t09-cell", runtime="mock", cell_runtime_factory=_TcpHold)


def test_t09_cell_checkpoint_selects_injected_factory(monkeypatch) -> None:
    policy_type = _policy_type(monkeypatch)
    built: list[_TcpHold] = []

    def factory() -> _TcpHold:
        controller = _TcpHold()
        built.append(controller)
        return controller

    policy = policy_type("t09-cell", runtime="cell", cell_runtime_factory=factory)

    assert policy.ACTION == "tcp"
    assert policy.controller is built[0]


def test_t09_cell_checkpoint_fails_closed_without_production_factory(monkeypatch) -> None:
    policy_type = _policy_type(monkeypatch)

    with pytest.raises(RuntimeError, match="production cell runtime factory"):
        policy_type("t09-cell", runtime="cell")


def test_default_policy_does_not_invoke_cell_factory(monkeypatch) -> None:
    policy_type = _policy_type(monkeypatch)

    def forbidden() -> _TcpHold:
        raise AssertionError("default policy must not construct the cell controller")

    policy = policy_type(runtime="cell", cell_runtime_factory=forbidden)

    assert policy.ACTION == "joints"
