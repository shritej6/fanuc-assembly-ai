"""GrootPolicy: official policy interface, model output always guarded, instruction routing, GR00T I/O mapping."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace

import numpy as np
import pytest

from assembly.catalog import TRUSTED_GRASP_SOURCE
from assembly.controller_ports import ExecutionTolerances
from policy_learn.deploy import attestation as attestation_module
from policy_learn.deploy.attestation import (
    REQUIRED_CELL_SKILLS,
    DeploymentAttestation,
    DeploymentPin,
    load_reviewed_deployment,
)
from policy_learn.deploy.groot_policy import (
    CellPolicySafetyError,
    GrootPolicy,
    GrootServerModel,
    StubModel,
    load_model,
    tcp_from_joints,
)
from policy_learn.deploy.guards import ContactTarget, GuardLimits

MOCK_HOME_JOINTS = [-0.0065, 3.2909, -11.8967, -0.0007, -74.8218, -0.0286, 0.0]  # robot/cell/mock_cell.py HOME
TCP = [300.0, 0.0, -90.0, -180.0, 0.0, 0.0]


def obs(tcp=TCP, grip=0.0, instruction="assemble everything"):
    return {
        "images": {c: np.zeros((360, 640, 3), np.uint8) for c in ("table", "side", "wrist")},
        "joints": np.array(MOCK_HOME_JOINTS[:6] + [grip], np.float32),
        "tcp_pose": tcp,
        "trial": {"instruction": instruction},
    }


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
    pol = GrootPolicy(model=Fixed([[300, 0, -95.0, -180, 0, 0, 1]]))
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
    def __init__(self, rows=None, attestation=None):
        self.resets, self.obs = 0, None
        self.closed = False
        self.rows = rows or [TCP + [0.0]] * 4
        self.attestation = attestation or {"checkpoint_sha256": "a" * 64}

    def reset(self):
        self.resets += 1

    def act(self, o):
        self.obs = o
        return self.rows

    def describe(self):
        return self.attestation

    def close(self):
        self.closed = True


def _approved() -> DeploymentAttestation:
    return DeploymentAttestation(
        checkpoint_sha256="a" * 64,
        model_id="GR00T-N1.7",
        model_revision="NVIDIA/GR00T-N1.7@reviewed",
        dataset_manifest_sha256="b" * 64,
        catalog_sha256="c" * 64,
        grasp_source_sha256=TRUSTED_GRASP_SOURCE.sha256,
        guard_calibration_sha256="e" * 64,
        skills=REQUIRED_CELL_SKILLS,
        cameras=("table", "side", "wrist"),
        depth=(),
        image_size=(640, 360),
        dt_s=0.2,
        execution_horizon=1,
        action="tcp",
        embodiment="fanuc_er4ia_team_v_fingers",
        release_protocol="t09_verified_release_v1",
        eval_approved=True,
        server_verified=True,
    )


def _attestation_mapping(value: DeploymentAttestation) -> dict:
    mapping = {field.name: getattr(value, field.name) for field in fields(value)}
    mapping["skills"] = sorted(value.skills)
    mapping["cameras"] = list(value.cameras)
    mapping["depth"] = list(value.depth)
    mapping["image_size"] = list(value.image_size)
    return mapping


def _measured_limits(tmp_path) -> GuardLimits:
    defaults = GuardLimits()
    values = {
        field.name: getattr(defaults, field.name)
        for field in fields(defaults)
        if field.init and field.name not in {"status", "source"}
    }
    report = tmp_path / "guard-calibration.json"
    report.write_text(json.dumps({"status": "MEASURED", "source": "CSI test report", "limits": values}))
    return GuardLimits.from_calibration(report, hashlib.sha256(report.read_bytes()).hexdigest())


def _reviewed_deployment(tmp_path, approved: DeploymentAttestation):
    payload = _attestation_mapping(approved)
    payload.pop("server_verified")
    path = tmp_path / "reviewed-deployment.json"
    path.write_text(json.dumps(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    previous = attestation_module.APPROVED_DEPLOYMENT_PINS
    attestation_module.APPROVED_DEPLOYMENT_PINS = {"test": DeploymentPin(str(path), digest)}
    try:
        return load_reviewed_deployment("test")
    finally:
        attestation_module.APPROVED_DEPLOYMENT_PINS = previous


def _production_policy(rows, tmp_path, *, limits=None, attestation=None) -> GrootPolicy:
    active_limits = limits or _measured_limits(tmp_path)
    approved = replace(_approved(), guard_calibration_sha256=active_limits.calibration_sha256)
    reviewed = _reviewed_deployment(tmp_path, approved)
    client = FakeClient(rows=rows, attestation=_attestation_mapping(attestation or approved))
    model = GrootServerModel.__new__(GrootServerModel)
    model.client = client
    model.injected_client = False
    return GrootPolicy(
        model=model,
        production=True,
        limits=active_limits,
        approved_deployment=reviewed,
    )


def _saturday_policy(rows) -> GrootPolicy:
    model = GrootServerModel.__new__(GrootServerModel)
    model.client = FakeClient(rows=rows)
    model.injected_client = False
    return GrootPolicy(
        model=model,
        saturday_cell=True,
        limits=GuardLimits(max_tilt_deg=6.0, source="Saturday fingers-down cone"),
    )


def test_server_model_gets_planner_instruction_and_no_sim_extras():
    fake = FakeClient()
    pol = GrootPolicy(model=GrootServerModel("x", 0, client=fake))
    pol.reset()
    pol.set_task("pick M4_02 at (400, 0, 30) gripping X")
    o = obs() | {"sim": {"object_poses": {}}}
    rows = pol.act(o)
    assert rows.shape == (4, 7) and fake.resets == 1
    assert fake.obs["instruction"] == "pick M4_02 at (400, 0, 30) gripping X" and "sim" not in fake.obs
    assert pol.model.describe() == {"checkpoint_sha256": "a" * 64}


def test_policy_close_releases_the_server_transport() -> None:
    client = FakeClient()
    policy = GrootPolicy(model=GrootServerModel("x", 0, client=client))

    policy.close()

    assert client.closed is True


def test_set_plan_uses_the_training_instruction_format():
    pol = GrootPolicy(model=Fixed([TCP + [0.0]]))
    pol.reset()
    text = pol.set_plan(
        "pick_insert",
        "M4_02",
        "F04",
        0,
        "nominal",
        [300, -50, 37, 127, 400, 50, 90],
        ContactTarget(400.0, 50.0, -301.2),
    )
    assert text == "pick M4_02 at (300, -50, 37) gripping X at roll 127, insert on F04 at (400, 50, 90)"
    pol.act(obs())
    assert pol.model.seen == [text]


def test_model_rows_across_r_180_are_one_small_turn_not_a_hold():
    """A 6D-rotation decode flips r between +179.8 and -179.9; the adapter sends the same orientation next to the
    reported r, so the arm makes a 0.3 deg turn instead of freezing on a guard hold (or spinning J6)."""
    tcp = [300.0, 0.0, -90.0, -180.0, 0.0, -179.9]
    pol = GrootPolicy(model=Fixed([[300, 1, -90, 180.0, 0, 179.8, 0], [300, 2, -90, -180.0, 0, -179.7, 0]]))
    pol.reset()
    rows = pol.act(obs(tcp=tcp))
    assert pol.guard_log == []
    np.testing.assert_allclose(rows[:, [3, 5]], [[-180.0, -180.2], [-180.0, -179.7]], atol=1e-9)


def test_a_real_large_wrist_turn_is_still_refused():
    tcp = [300.0, 0.0, -90.0, -180.0, 0.0, 0.0]
    pol = GrootPolicy(model=Fixed([[300, 0, -90, -180.0, 0, 90.0, 0]]))
    pol.reset()
    pol.act(obs(tcp=tcp))
    assert pol.guard_log[-1]["held"]


def test_production_policy_rejects_stub_and_preliminary_limits(tmp_path):
    with pytest.raises(CellPolicySafetyError, match="stub"):
        GrootPolicy(
            model=StubModel(),
            production=True,
            limits=_measured_limits(tmp_path),
        )
    with pytest.raises(CellPolicySafetyError, match="attested GrootServerModel"):
        GrootPolicy(
            model=Fixed([TCP + [0.0]]),
            production=True,
            limits=_measured_limits(tmp_path),
        )
    with pytest.raises(CellPolicySafetyError, match="injected"):
        GrootPolicy(
            model=GrootServerModel("x", 0, client=FakeClient()),
            production=True,
            limits=_measured_limits(tmp_path),
        )
    with pytest.raises(CellPolicySafetyError, match="MEASURED"):
        _production_policy([TCP + [0.0]], tmp_path, limits=GuardLimits())


def test_saturday_cell_requires_a_noninjected_live_server() -> None:
    with pytest.raises(CellPolicySafetyError, match="stub"):
        GrootPolicy(model=StubModel(), saturday_cell=True)
    with pytest.raises(CellPolicySafetyError, match="live GrootServerModel"):
        GrootPolicy(model=Fixed([TCP + [0.0]]), saturday_cell=True)
    with pytest.raises(CellPolicySafetyError, match="injected"):
        GrootPolicy(model=GrootServerModel("x", 0, client=FakeClient()), saturday_cell=True)


def test_saturday_policy_defers_moving_close_before_the_gripper_guard() -> None:
    grasp = (299.1, -48.441154, -270.0, -180.0, 0.0, 120.0)
    current = [grasp[0], grasp[1], grasp[2] + 2.7, *grasp[3:]]
    policy = _saturday_policy([[*grasp, 1.0]])
    policy.reset()
    policy.set_grasp_fence(grasp, ExecutionTolerances(0.3, 1.0))

    rows = policy.act(obs(tcp=current, grip=0.0))

    assert rows[0, 6] == 0.0
    assert policy.last_guard_result is not None
    assert policy.last_guard_result.held is False
    assert "deferred" in policy.last_guard_result.reasons[0]


def test_saturday_policy_allows_stationary_close_at_the_grasp_fence() -> None:
    grasp = (299.1, -48.441154, -270.0, -180.0, 0.0, 120.0)
    policy = _saturday_policy([[*grasp, 1.0]])
    policy.reset()
    policy.set_grasp_fence(grasp, ExecutionTolerances(0.3, 1.0))

    rows = policy.act(obs(tcp=list(grasp), grip=0.0))

    assert rows[0, 6] == 1.0
    assert policy.last_guard_result is not None
    assert policy.last_guard_result.held is False


def test_saturday_policy_does_not_defer_close_at_the_wrong_pose() -> None:
    grasp = (299.1, -48.441154, -270.0, -180.0, 0.0, 120.0)
    wrong = (309.1, -48.441154, -270.0, -180.0, 0.0, 120.0)
    current = [wrong[0], wrong[1], wrong[2] + 2.7, *wrong[3:]]
    policy = _saturday_policy([[*wrong, 1.0]])
    policy.reset()
    policy.set_grasp_fence(grasp, ExecutionTolerances(0.3, 1.0))

    policy.act(obs(tcp=current, grip=0.0))

    assert policy.last_guard_result is not None
    assert policy.last_guard_result.held is True
    assert "gripper changes while moving" in policy.last_guard_result.reasons[0]


def test_saturday_cell_keeps_one_row_release_and_final_adapter_tilt_fences() -> None:
    many = _saturday_policy([TCP + [0.0], TCP + [0.0]])
    many.act(obs())
    assert many.last_guard_result is not None and many.last_guard_result.held
    assert "exactly one row" in many.last_guard_result.reasons[0]

    release = _saturday_policy([TCP + [0.0]])
    release.act(obs(grip=1.0))
    assert release.last_guard_result is not None and release.last_guard_result.held
    assert "release is not authorized" in release.last_guard_result.reasons[0]

    tilted = _saturday_policy([[300.0, 0.0, -90.0, -173.0, 0.0, 0.0, 0.0]])
    tilted.act(obs())
    assert tilted.last_guard_result is not None and tilted.last_guard_result.held
    assert "tilted" in tilted.last_guard_result.reasons[0]


def test_production_policy_rejects_a_server_not_matching_the_approved_record(tmp_path):
    mismatch = replace(_approved(), checkpoint_sha256="d" * 64)
    with pytest.raises(CellPolicySafetyError, match="approved deployment"):
        _production_policy([TCP + [0.0]], tmp_path, attestation=mismatch)


def test_production_policy_requires_tcp_and_one_row_execution(tmp_path):
    missing = _production_policy([TCP + [0.0]], tmp_path)
    with pytest.raises(CellPolicySafetyError, match="tcp_pose"):
        missing.act(obs(tcp=None))

    many = _production_policy([TCP + [0.0], TCP + [0.0]], tmp_path)
    rows = many.act(obs())
    assert rows.shape == (1, 7)
    assert many.last_guard_result is not None and many.last_guard_result.held
    assert "exactly one row" in many.last_guard_result.reasons[0]

    for malformed in ([1.0] * 5, [1.0] * 7, [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0]):
        with pytest.raises(CellPolicySafetyError, match="tcp_pose"):
            _production_policy([TCP + [0.0]], tmp_path).act(obs(tcp=malformed))


def test_release_is_fenced_until_controller_authorizes_it(tmp_path):
    pol = _production_policy([TCP + [0.0]], tmp_path)
    pol.reset()
    held = pol.act(obs(grip=1.0))
    assert held[0, 6] == 1.0
    assert pol.last_guard_result is not None and pol.last_guard_result.held
    assert "release is not authorized" in pol.last_guard_result.reasons[0]

    pol.authorize_release()
    released = pol.act(obs(grip=1.0))
    assert released[0, 6] == 0.0
    assert pol.release_authorized is False


@pytest.mark.parametrize("grip", [float("nan"), float("inf"), 0.05, 0.5, 0.95])
def test_production_policy_rejects_ambiguous_observed_gripper(grip, tmp_path):
    pol = _production_policy([TCP + [0.0]], tmp_path)
    with pytest.raises(CellPolicySafetyError, match="gripper|finite"):
        pol.act(obs(grip=grip))
