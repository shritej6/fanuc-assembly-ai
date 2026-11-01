"""GrootPolicy: official policy interface, model output always guarded, instruction routing, GR00T I/O mapping.
Authored by Shritej Gunda.
"""
from __future__ import annotations

import numpy as np
import pytest

from policy_learn.deploy.groot_policy import (
    GrootPolicy,
    GrootServerModel,
    StubModel,
    load_model,
    tcp_from_joints,
)
from policy_learn.deploy.guards import ContactTarget

MOCK_HOME_JOINTS = [-0.0065, 3.2909, -11.8967, -0.0007, -74.8218, -0.0286, 0.0]  # robot/cell/mock_cell.py HOME
TCP = [300.0, 0.0, -90.0, -180.0, 0.0, 0.0]


def obs(tcp=TCP, grip=0.0, instruction="assemble everything"):
    return {"images": {c: np.zeros((360, 640, 3), np.uint8) for c in ("table", "side", "wrist")},
            "joints": np.array(MOCK_HOME_JOINTS[:6] + [grip], np.float32), "tcp_pose": tcp,
            "trial": {"instruction": instruction}}


class Fixed:
    def __init__(self, chunk):
        self.chunk, self.seen = np.asarray(chunk, float), []

    def predict(self, model_obs):
        assert "sim" not in model_obs
        self.seen.append(model_obs["instruction"])
        return self.chunk.copy()


def test_official_interface_attributes():
    assert GrootPolicy.ACTION == "tcp" and set(GrootPolicy.CAMERAS) <= {"table", "side", "wrist"}
    assert 0.1 <= GrootPolicy.DT <= 1.0 and GrootPolicy.PREFETCH is False


def test_fk_estimate_at_mock_home_is_fingers_down():
    pose = tcp_from_joints(MOCK_HOME_JOINTS)
    np.testing.assert_allclose(pose[:3], [300.0, 0.0, -90.1], atol=0.5)
    assert abs(abs(pose[3]) - 180.0) < 0.5 and abs(pose[4]) < 0.5 and abs(pose[5]) < 0.5


def test_missing_tcp_pose_holds_instead_of_moving():
    pol = GrootPolicy(model=Fixed([[400, 0, -100, -180, 0, 0, 0]]))
    pol.reset()
    rows = pol.act(obs(tcp=None))
    assert rows.shape == (1, 7) and pol.guard_log[-1]["held"]
    assert pol.model.seen == []  # the model is not even asked without a known pose


def test_bad_model_output_becomes_hold_with_reason():
    pol = GrootPolicy(model=Fixed([[700, 0, -100, -180, 0, 0, 0]]))
    pol.reset()
    rows = pol.act(obs(grip=1.0))
    np.testing.assert_allclose(rows, [TCP + [1.0]])
    assert "outside workspace" in pol.guard_log[-1]["reasons"][0]


def test_contact_target_from_planner_is_enforced():
    pol = GrootPolicy(model=Fixed([[300, 0, -95.0, -180, 0, 0, 1]]))  # 3.5 mm below the floor
    pol.reset()
    pol.set_task("insert M4_02 on F04", ContactTarget(300.0, 0.0, -90.5))
    pol.act(obs(grip=1.0))
    assert pol.guard_log[-1]["held"] and "contact floor" in pol.guard_log[-1]["reasons"][0]
    assert pol.model.seen == ["insert M4_02 on F04"]


def test_instruction_falls_back_to_the_trial_prompt():
    pol = GrootPolicy(model=Fixed([TCP + [0.0]]))
    pol.reset()
    pol.act(obs(instruction="build the rotary switch"))
    assert pol.model.seen == ["build the rotary switch"]


def test_stub_chunks_pass_the_guard():
    pol = GrootPolicy(model=StubModel())
    pol.reset()
    rows = pol.act(obs())
    assert rows.shape == (16, 7) and pol.guard_log == []


def test_load_model_never_falls_back_silently(monkeypatch):
    monkeypatch.delenv("GROOT_POLICY", raising=False)
    with pytest.raises(ValueError):
        load_model(None)
    with pytest.raises(ValueError):
        load_model("checkpoints/model.pt")
    assert isinstance(load_model("stub"), StubModel)


class FakeClient:
    def __init__(self):
        self.resets, self.obs = 0, None

    def reset(self):
        self.resets += 1

    def act(self, o):
        self.obs = o
        return [TCP + [0.0]] * 4


def test_server_model_gets_planner_instruction_and_no_sim_extras():
    fake = FakeClient()
    pol = GrootPolicy(model=GrootServerModel("x", 0, client=fake))
    pol.reset()
    pol.set_task("pick M4_02 at (400, 0, 30) gripping X")
    o = obs() | {"sim": {"object_poses": {}}}
    rows = pol.act(o)
    assert rows.shape == (4, 7) and fake.resets == 1
    assert fake.obs["instruction"] == "pick M4_02 at (400, 0, 30) gripping X" and "sim" not in fake.obs


def test_set_plan_uses_the_training_instruction_format():
    pol = GrootPolicy(model=Fixed([TCP + [0.0]]))
    pol.reset()
    text = pol.set_plan("pick_insert", "M4_02", "F04", 0, "nominal", [300, -50, 37, 127, 400, 50, 90],
                        ContactTarget(400.0, 50.0, -301.2))
    assert text == "pick M4_02 at (300, -50, 37) gripping X at roll 127, insert on F04 at (400, 50, 90)"
    pol.act(obs())
    assert pol.model.seen == [text]


def test_model_rows_across_r_180_are_one_small_turn_not_a_hold():
    """A 6D-rotation decode flips r between +179.8 and -179.9."""
    tcp = [300.0, 0.0, -90.0, -180.0, 0.0, -179.9]
    pol = GrootPolicy(model=Fixed([[300, 1, -90, 180.0, 0, 179.8, 0], [300, 2, -90, -180.0, 0, -179.7, 0]]))
    pol.reset()
    rows = pol.act(obs(tcp=tcp))
    assert pol.guard_log == []
    np.testing.assert_allclose(rows[:, [3, 5]], [[-180.0, -180.2], [-180.0, -179.7]], atol=1e-9)


def test_policy_sends_the_valid_prefix_and_logs_the_truncation():
    tcp = [380.0, 74.0, -300.0, -180.0, 0.0, 180.0]
    chunk = [[380.0, 74.0, -300.0 - 2.8 * (k + 1), -180.0, 0.0, 180.0, 0] for k in range(10)]
    chunk += [[380.0, 74.0, -334.5, -180.0, 0.0, 180.0, 0]] * 6
    pol = GrootPolicy(model=Fixed(chunk))
    pol.reset()
    rows = pol.act(obs(tcp=tcp))
    assert rows.shape == (10, 7)
    entry = pol.guard_log[-1]
    assert not entry["held"] and entry["truncated_at"] == 10 and "outside workspace" in entry["reasons"][0]
    pol = GrootPolicy(model=Fixed(chunk[10:]))
    pol.reset()
    rows = pol.act(obs(tcp=tcp))
    assert rows.shape == (1, 7) and pol.guard_log[-1]["held"] and pol.guard_log[-1]["truncated_at"] is None


def test_exec_horizon_sends_only_the_first_rows_and_guards_only_those(monkeypatch):
    tcp = [380.0, 74.0, -300.0, -180.0, 0.0, 180.0]
    chunk = [[380.0, 74.0, -300.0 - 2.8 * (k + 1), -180.0, 0.0, 180.0, 0] for k in range(10)]
    chunk += [[380.0, 74.0, -334.5, -180.0, 0.0, 180.0, 0]] * 6
    pol = GrootPolicy(model=Fixed(chunk), exec_horizon=8)
    pol.reset()
    rows = pol.act(obs(tcp=tcp))
    np.testing.assert_allclose(rows, np.array(chunk[:8], float))
    assert pol.guard_log == []                         # row 10 under the floor is never sent, so never refused
    monkeypatch.setenv("GROOT_EXEC_HORIZON", "4")
    pol = GrootPolicy(model=Fixed(chunk))
    pol.reset()
    assert pol.exec_horizon == 4 and len(pol.act(obs(tcp=tcp))) == 4
    monkeypatch.delenv("GROOT_EXEC_HORIZON")
    assert GrootPolicy(model=Fixed(chunk)).exec_horizon == 16
    with pytest.raises(ValueError):
        GrootPolicy(model=Fixed(chunk), exec_horizon=101)


def test_a_real_large_wrist_turn_is_still_refused():
    tcp = [300.0, 0.0, -90.0, -180.0, 0.0, 0.0]
    pol = GrootPolicy(model=Fixed([[300, 0, -90, -180.0, 0, 90.0, 0]]))
    pol.reset()
    pol.act(obs(tcp=tcp))
    assert pol.guard_log[-1]["held"]
