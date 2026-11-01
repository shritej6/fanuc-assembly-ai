"""Host checks of the Mimic plan (skills.yaml), plan-id catalog, instructions, pose conversions and force guard.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from sim.assembly_lab.mimic import catalog, frames, instructions
from sim.assembly_lab.mimic.force_guard import ForceLimitError, ForceLimits, check_trace, load_limits
from sim.assembly_lab.mimic.spec import SPEC_PATH, SpecError, load_spec


def test_spec_loads_with_object_centric_frames():
    spec = load_spec()
    assert set(spec.skills) == set(catalog.SKILLS)
    pi = spec.skills["pick_insert"]
    assert [s.object_ref for s in pi.subtasks] == ["grasp_frame", "seat_frame", "seat_frame"]
    assert pi.signals == ["grasped", "seated"]
    contact = next(s for s in pi.subtasks if s.name == "carry_insert")
    assert contact.noise_pos_m < 0.25e-3  # well under the 0.5 mm insert clearance


@pytest.mark.parametrize("mutate, msg", [
    (lambda r: r["noise_profiles"]["contact"].update(pos_mm=0.4), "insert clearance"),
    (lambda r: r["skills"]["pick_insert"]["subtasks"][-1].update(term_signal="done"), "last subtask"),
    (lambda r: r["skills"]["pick_insert"]["subtasks"][0].update(object_ref="M4_02"), "object_ref"),
    (lambda r: r["skills"]["pick_insert"]["subtasks"][1].update(term_signal="grasped"), "unique"),
])
def test_spec_rejects_bad_plans(tmp_path, mutate, msg):
    raw = copy.deepcopy(yaml.safe_load(SPEC_PATH.read_text()))
    mutate(raw)
    path = tmp_path / "skills.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(SpecError, match=msg):
        load_spec(path)


def test_plan_ids_roundtrip_and_bounds():
    ids = catalog.encode_plan_ids("defect_drop", "M1_03", None, 0, "defect")
    d = catalog.decode_plan_ids(ids)
    assert (d["object_id"], d["base_id"], d["product"], d["case"]) == ("M1_03", None, 1, "defect")
    assert len(catalog.object_ids()) == 31
    with pytest.raises(IndexError):
        catalog.decode_plan_ids([0, 99, -1, 0, 0])
    with pytest.raises(IndexError):
        catalog.decode_plan_ids(catalog.encode_plan_ids("pick_insert", "M4_02", "F04", 5, "nominal"))


def test_instruction_carries_plan_target():
    ids = catalog.encode_plan_ids("pull_out", "M4_02", "F04", 0, "fix_above")
    text = instructions.instruction(ids, [310, 20, 45, 135, 250, -100, 0])
    assert text == "pull out M4_02 at (310, 20, 45) gripping X at roll 135, set it down at (250, -100, 0) [fix_above]"
    with pytest.raises(ValueError):
        instructions.instruction(ids, [1, 2, 3])


def test_instruction_names_trained_base_for_an_interchangeable_physical_base():
    target = [410, 162, -1, 179, 375, -132, -105]
    expected = (
        "pick M5_01 at (410, 162, -1) gripping Y at roll 179, "
        "insert on F05 at (375, -132, -105)"
    )

    for physical in ("F04", "F06"):
        ids = catalog.encode_plan_ids("pick_insert", "M5_01", physical, 0, "nominal")
        assert instructions.instruction(ids, target) == expected

    ids = catalog.encode_plan_ids("pick_insert", "M2_01", "F01", 0, "nominal")
    assert "insert on F02 at" in instructions.instruction(ids, target)


def test_instruction_with_the_trained_base_is_unchanged():
    ids = catalog.encode_plan_ids("pick_insert", "M5_01", "F05", 0, "nominal")
    text = instructions.instruction(ids, [332, -47, 30, 210, 420, 0, 45])
    assert text == (
        "pick M5_01 at (332, -47, 30) gripping Y at roll 210, "
        "insert on F05 at (420, 0, 45)"
    )


def test_wpr_roundtrip_and_w_wrap():
    rng = np.random.default_rng(1)
    wpr = np.stack([rng.uniform(-269, 89, 200), rng.uniform(-80, 80, 200), rng.uniform(-179, 179, 200)], 1)
    back = frames.rot_to_wpr(frames.wpr_to_rot(wpr))
    assert np.allclose(back, wpr, atol=1e-6)
    down = frames.rot_to_wpr(frames.wpr_to_rot(np.array([[179.9, 0, 10], [-179.9, 0, 10]])))
    assert np.allclose(down[:, 0], [-180.1, -179.9], atol=1e-6)


def test_matches_skills_geometry_convention():
    geometry = pytest.importorskip("sim.assembly_lab.skills.geometry")
    for w, p, r in [(-180, 0, 30), (-177, 2, -150), (10, 20, 30)]:
        assert np.allclose(frames.wpr_to_rot(np.array([w, p, r])), geometry.wpr_to_rot(w, p, r), atol=1e-9)


def test_force_guard(tmp_path):
    lim = ForceLimits(15.0, 5.0, 0.5, "PRELIMINARY", "test")
    ok = check_trace(np.array([[1, 1, 0.1], [14.9, 4.9, 0.4]]), lim)
    assert ok.ok and ok.first_violation_step is None
    bad = check_trace(np.array([[1, 1, 0.1], [1, 5.2, 0.1], [20, 1, 0.1]]), lim)
    assert not bad.ok and bad.first_violation_step == 1
    assert not check_trace(np.array([[1, np.nan, 0.1]]), lim).ok
    (tmp_path / "l.yaml").write_text("axial_N: 15\nlateral_N: 5\ntorque_Nm: 0.5\nstatus: GUESS\nsource: x\n")
    with pytest.raises(ForceLimitError):
        load_limits(tmp_path / "l.yaml")
    with pytest.raises(ForceLimitError):
        load_limits(tmp_path / "none.yaml")
