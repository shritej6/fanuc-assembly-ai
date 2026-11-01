"""look-alike size gate and the close-up judgement rules.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.port.backends import WristReading
from perception.port.cad import load_cad
from perception.port.judge import judge
from perception.port.size_gate import (
    TABLE_DEPTH_PRELIMINARY,
    SizeOutcome,
    SizeUncertainty,
    SizeVerdict,
    Status,
    gate,
    measure,
    measure_unposed,
)
from tests.port_fakes import TABLE_Z, THRESHOLDS, WRIST_SIZE, loose_pose, top_points

CAD = load_cad()


def _measured(object_id: str, as_id: str | None = None, top_offset: float = 0.0, region_z: tuple | None = None):
    """Measure the top face of `object_id` posed as `as_id` would be (same frame), optionally with a raised region."""
    pose = loose_pose(object_id, 400.0, 0.0, 20.0)
    points = top_points(object_id, pose, top_offset)
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    if region_z is not None:
        (x0, y0, x1, y1), z = region_z
        inside = (local[:, 0] >= x0) & (local[:, 0] <= x1) & (local[:, 1] >= y0) & (local[:, 1] <= y1)
        local[inside, 2] = z
    target = as_id or object_id
    cues = tuple(c for other in CAD.candidates_for(target) for c in CAD.get(target).cues.get(other, ()))
    return measure(local, points, TABLE_Z, cues)


def test_height_separates_the_m1_m2_carriers_at_the_wrist():
    verdict = gate("M1_02", _measured("M1_02"), CAD, WRIST_SIZE)
    assert verdict.outcome is SizeOutcome.CONFIRMS and verdict.residual_mm == pytest.approx(0.0, abs=0.05)
    assert gate("M1_02", _measured("M2_02", "M1_02"), CAD, WRIST_SIZE).outcome is SizeOutcome.CONTRADICTS


def test_a_coarse_bound_cannot_separate_the_carriers():
    coarse = SizeUncertainty(0.5, 1.5, Status.PRELIMINARY, "test")
    verdict = gate("M1_02", _measured("M1_02"), CAD, coarse)
    assert verdict.outcome is SizeOutcome.CANNOT_SEPARATE and "M2_02" in verdict.matching


def test_same_envelope_twins_are_told_apart_by_the_cad_top_cue():
    m1_pocket = ((-7.0, -10.5, 7.0, -6.0), 7.0)
    m3_solid = ((-7.0, -10.5, 7.0, -6.0), 16.0)
    assert gate("M1_01", _measured("M1_01", region_z=m1_pocket), CAD, WRIST_SIZE).outcome is SizeOutcome.CONFIRMS
    contradicted = gate("M1_01", _measured("M1_01", region_z=m3_solid), CAD, WRIST_SIZE)
    assert contradicted.outcome is SizeOutcome.CONTRADICTS and "M3_01" in contradicted.matching


def test_a_piece_that_fits_none_of_the_30_and_a_depth_hole():
    cube = np.array([(x, y, TABLE_Z + 60.0) for x in np.arange(0, 70, 0.5) for y in np.arange(0, 70, 0.5)])
    assert gate("M1_02", measure_unposed(cube, TABLE_Z), CAD, WRIST_SIZE).outcome is SizeOutcome.MATCHES_NOTHING
    assert gate("M1_02", measure(cube[:5], cube[:5], TABLE_Z), CAD, WRIST_SIZE).outcome is SizeOutcome.UNMEASURED


def test_unposed_measurement_ignores_yaw():
    for yaw in (0.0, 33.0, 90.0):
        points = top_points("M4_01", loose_pose("M4_01", 400.0, 0.0, yaw))
        size = measure_unposed(points, TABLE_Z)
        assert size.footprint_mm == pytest.approx(sorted(CAD.get("M4_01").size_mm[:2]), abs=0.05)


def test_table_depth_bound_is_labelled_preliminary():
    assert TABLE_DEPTH_PRELIMINARY.status is Status.PRELIMINARY and "rms_plane_mm 2.31" in TABLE_DEPTH_PRELIMINARY.source


def _size(outcome: SizeOutcome, matching=("M1_02",)) -> SizeVerdict:
    return SizeVerdict(outcome, "M1_02", tuple(matching), 0.1, WRIST_SIZE)


def _judge(reading=("M1_02", 0.97, 0.01), fit=0.9, side="upright", size=None):
    return judge("M1_02", WristReading(*reading), fit, side, size or _size(SizeOutcome.CONFIRMS), THRESHOLDS)


@pytest.mark.parametrize(("kwargs", "verdict", "action", "unsure", "agree"), [
    ({}, "ok", "use", False, True),
    ({"fit": 0.3}, "ok", "use", False, False),
    ({"side": "side"}, "ok", "use", False, False),
    ({"reading": ("M1_02", 0.4, 0.01), "size": _size(SizeOutcome.MATCHES_NOTHING, ())}, "foreign", "reject", False,
     True),
    ({"reading": ("M1_02", 0.97, 0.995)}, "print_defect", "reject", False, True),
    ({"reading": ("M1_02", 0.97, 0.7)}, "uncertain", "ask_operator", True, False),
    ({"reading": ("M1_02", 0.5, 0.01)}, "uncertain", "ask_operator", True, False),
    ({"reading": ("M2_02", 0.97, 0.01), "size": _size(SizeOutcome.CONTRADICTS, ("M2_02",))}, "wrong_part", "reject",
     False, True),
    ({"reading": ("M2_02", 0.97, 0.01)}, "uncertain", "ask_operator", True, False),
    ({"size": _size(SizeOutcome.CONTRADICTS, ("M2_02",))}, "uncertain", "ask_operator", True, False),
    ({"size": _size(SizeOutcome.UNMEASURED, ())}, "uncertain", "ask_operator", True, False),
    ({"size": _size(SizeOutcome.CANNOT_SEPARATE, ("M1_02", "M2_02"))}, "ok", "use", False, True),
])
def test_closeup_rules(kwargs, verdict, action, unsure, agree):
    result = _judge(**kwargs)
    assert (result.verdict, result.action, result.unsure) == (verdict, action, unsure)
    assert result.checks_agree is agree


def test_explicit_defect_still_rejects_with_foundationpose_warnings():
    result = _judge(reading=("M1_02", 0.97, 0.995), fit=0.3, side="side")

    assert (result.verdict, result.action, result.unsure, result.checks_agree) == (
        "print_defect", "reject", False, False)
    assert "FoundationPose advisory" in result.reason


def test_a_base_of_the_same_mesh_is_the_expected_base():
    result = judge("F01", WristReading("F02", 0.95, 0.0), 0.9, "upright",
                   SizeVerdict(SizeOutcome.CONFIRMS, "F01", ("F01", "F02"), 0.0, WRIST_SIZE), THRESHOLDS)
    assert (result.verdict, result.object_id) == ("ok", "F01")


def test_with_the_measured_wrist_noise_height_alone_cannot_separate_the_carriers():
    from perception.port.size_gate import WRIST_150MM_PRELIMINARY

    verdict = gate("M1_02", _measured("M1_02"), CAD, WRIST_150MM_PRELIMINARY)
    assert verdict.outcome is SizeOutcome.CANNOT_SEPARATE and verdict.uncertainty.status is Status.PRELIMINARY


def test_height_is_not_biased_upward_by_depth_noise():
    pose = loose_pose("M4_01", 400.0, 0.0)
    points = top_points("M4_01", pose)
    points[:, 2] += np.random.default_rng(0).normal(0.0, 1.3, len(points))
    size = measure_unposed(points, TABLE_Z)
    assert abs(size.height_mm - CAD.get("M4_01").size_mm[2]) < 1.3 / 2   # well inside the wrist z bound
