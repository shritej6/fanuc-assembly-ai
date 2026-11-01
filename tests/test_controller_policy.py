"""Policy-selection behavior independent of the assembly scenarios.
Authored by Shritej Gunda.
"""

from pathlib import Path


def test_policy_can_inject_mock_tcp_controller_without_changing_cell_protocol(monkeypatch) -> None:
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
    policy = Policy(controller=TcpHold(), action_mode="tcp", runtime="mock")

    assert Policy.ACTION == "tcp"
    assert policy.ACTION == "tcp"
    assert policy.act(observation) == [400.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0]


def test_policy_close_releases_an_owned_controller(monkeypatch) -> None:
    cell_dir = Path(__file__).resolve().parents[1] / "robot" / "cell"
    monkeypatch.syspath_prepend(str(cell_dir))
    from policy import Policy

    class ClosableController:
        def __init__(self) -> None:
            self.closed = False

        def reset(self) -> None:
            pass

        def act(self, observation):
            del observation
            return [400.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0]

        def close(self) -> None:
            self.closed = True

    controller = ClosableController()
    policy = Policy(controller=controller, action_mode="tcp", runtime="mock")

    policy.close()

    assert controller.closed is True
