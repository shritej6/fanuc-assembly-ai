"""Correction demos (skills/correct.py): the staged wrong pose is masked and comes first, the fix is learned.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.assembly_lab.mimic.instructions import instruction
from sim.assembly_lab.skills import correct, dry_run, episodes, record
from sim.assembly_lab.skills.force_guard import ForceGuard, load_limits
from sim.assembly_lab.skills.geometry import tilt_deg
from sim.assembly_lab.skills.runner import EpisodeRecorder

LIMITS = "axial_N: 15.0\nlateral_N: 4.5\ntorque_Nm: 0.1\nstatus: PRELIMINARY\nsource: test\n"


@pytest.fixture
def limits(tmp_path):
    p = tmp_path / "limits.yaml"
    p.write_text(LIMITS)
    return load_limits(p)


def _record(limits, kind: str, seed: int) -> EpisodeRecorder:
    objs, seats = dry_run.demo_world()
    env = dry_run.KinematicEnv(objs, seats)
    pick, ins, *_ = dry_run.demo_plans(objs, seats)
    rec = EpisodeRecorder(env, ForceGuard(limits), env.reset())
    getattr(episodes, kind)(rec, pick, ins, np.random.default_rng(seed))
    return rec, pick, ins


def _mask(rec) -> np.ndarray:
    return np.array([f.seg.learn for f in rec.frames])


@pytest.mark.parametrize("kind", ["align_insert", "align_grasp"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_staging_is_masked_and_comes_first(limits, kind, seed):
    rec, *_ = _record(limits, kind, seed)
    mask = _mask(rec)
    first = int(np.argmax(mask))
    assert first > 0 and not mask[:first].any(), "the staged rows come first and are masked"
    assert mask[first:].all(), "no masked gap after the fix starts (a gap teaches a jump)"
    assert rec.marks["grasped"] < rec.marks["seated"] and not rec.gate_log.failed


@pytest.mark.parametrize("seed", range(5))
def test_align_insert_learned_part_starts_off_target(limits, seed):
    rec, _, ins = _record(limits, "align_insert", seed)
    mask = _mask(rec)
    start = rec.frames[int(np.argmax(mask))].obs["tcp_pose"]  # where the first learned frame sees the arm
    xy = float(np.hypot(*(np.asarray(start[:2]) - ins.seated_tcp[:2])))
    assert correct.INSERT_XY_MM[0] - 1e-6 <= xy <= correct.INSERT_XY_MM[1] + 1e-6  # the lean pivots about the TCP
    assert start[2] > ins.seated_tcp[2] + correct.INSERT_DZ_MM[0]
    turned = abs((start[5] - ins.seated_tcp[5] + 180.0) % 360.0 - 180.0)
    assert correct.INSERT_YAW_DEG[0] - 1e-6 <= turned <= correct.INSERT_YAW_DEG[1] + 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_align_grasp_learned_fingers_start_above_and_off_the_pad(limits, seed):
    rec, pick, _ = _record(limits, "align_grasp", seed)
    f = rec.frames[int(np.argmax(_mask(rec)))]
    start = [*f.obs["tcp_pose"][:6], f.action[6]]
    xy = float(np.hypot(*(np.asarray(start[:2]) - pick.grasp_tcp[:2])))
    assert correct.GRASP_XY_MM[0] - 1e-6 <= xy <= correct.GRASP_XY_MM[1] + 1e-6
    assert start[2] >= pick.grasp_tcp[2] + correct.GRASP_DZ_MM[0] - 1e-6 and start[6] == 0.0  # fingers open


@pytest.mark.parametrize("kind", ["align_insert", "align_grasp"])
def test_instruction_reads_like_the_planner_with_no_case_tag(limits, kind):
    rec, *_ = _record(limits, kind, 0)
    text = instruction(rec.meta["plan_ids"], rec.meta["plan_target"])
    assert text.startswith("pick ") and ", insert on " in text and "[" not in text


def test_staged_insert_pose_stays_inside_the_tool_tilt_limit():
    objs, seats = dry_run.demo_world()
    _, ins, *_ = dry_run.demo_plans(objs, seats)
    rng = np.random.default_rng(7)
    for _ in range(200):
        assert tilt_deg(correct.staged_insert_pose(ins, (0.0, 0.0, 0.0), rng)) <= correct.INSERT_TILT_DEG + 1e-6


@pytest.mark.parametrize("scenario", ["align_insert", "align_grasp"])
def test_record_keeps_correction_demos(tmp_path, scenario):
    lim = tmp_path / "limits.yaml"
    lim.write_text(LIMITS)
    summary = record.record(record.KinematicBackend(), scenario, [1], 2, tmp_path, 0, 5, lim)
    assert summary["1:M1_02"]["kept_or_would_keep"] == 2
    assert (tmp_path / "pick_insert" / "product_1.hdf5").exists()


def test_give_up_after_stops_a_part_that_keeps_failing(tmp_path, monkeypatch):
    """A part failing its gates every time must not burn every attempt of a job."""
    lim = tmp_path / "limits.yaml"
    lim.write_text(LIMITS)

    def always_fails(*_a, **_k):
        raise record.EpisodeFailed("force guard: M1_02 normal_N 3.0 > 2.5 at step 80")

    monkeypatch.setattr(record, "_episode", always_fails)
    stats = record.record(record.KinematicBackend(), "align_insert", [1], 6, tmp_path, 0, 10, lim,
                          give_up_after=3)["1:M1_02"]
    assert stats["attempts"] == 3 and stats["kept_or_would_keep"] == 0
    assert "3 robot failures in a row" in stats["aborted"]
