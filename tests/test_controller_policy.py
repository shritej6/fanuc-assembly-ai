"""Policy-selection behavior independent of the assembly scenarios."""

from pathlib import Path


def test_policy_can_opt_into_tcp_controller_without_changing_safe_default(monkeypatch) -> None:
    cell_dir = Path(__file__).resolve().parents[1] / "robot" / "cell"
    monkeypatch.syspath_prepend(str(cell_dir))
    from policy import Policy

    class TcpHold:
        def reset(self) -> None:
            pass

        def act(self, observation):
            del observation
            return [400.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0]

    observation = {"joints": [0.0] * 7, "tcp_pose": None}
    policy = Policy(controller=TcpHold(), action_mode="tcp")

    assert Policy.ACTION == "joints"
    assert policy.ACTION == "tcp"
    assert policy.act(observation) == [400.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0]
