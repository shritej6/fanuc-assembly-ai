"""Insertion tolerance table (sdg/data/insertion_tolerance.json) and the helpers that build it.
Authored by Shritej Gunda.
"""

import json
from pathlib import Path

import pytest

from sdg.tools.insertion_tolerance import YAW_GRID_DEG, largest_ok, symmetry

TABLE = json.loads((Path(__file__).resolve().parents[1] / "sdg/data/insertion_tolerance.json").read_text())["parts"]
CAD_CLEARANCE_MM = 0.5   # docs/PLAN.md: designed insertion clearance


@pytest.mark.parametrize("fit, expected", [
    ([0], "every 360 deg"), ([0, 180], "every 180 deg"), ([0, 90, 180, 270], "every 90 deg"),
    ([0, 5, 175, 180, 185, 355], "every 180 deg"),   # a tolerant part also fits at the neighbouring grid yaws
    ([355, 0, 5], "every 360 deg"), (list(range(0, 360, YAW_GRID_DEG)), "any"), ([0, 30, 180], "irregular"),
])
def test_symmetry_groups_neighbouring_yaws_into_one_fit_window(fit, expected):
    assert symmetry(fit) == expected


def test_largest_ok_finds_the_boundary_within_its_resolution():
    assert largest_ok(lambda x: x <= 1.234, 4.0, 0.01) == pytest.approx(1.234, abs=0.01)
    assert largest_ok(lambda x: True, 4.0, 0.01) == 4.0


def test_every_part_has_a_row_that_fits_at_its_cad_pose():
    assert sorted(TABLE) == sorted(f"M{p}_0{s}" for p in range(1, 7) for s in range(1, 5))
    for part, row in TABLE.items():
        assert 0 in row["fit_yaws_deg"], part
        assert row["symmetry"] == symmetry(row["fit_yaws_deg"]), part


def test_tolerances_stay_within_the_designed_clearance():
    for part, row in TABLE.items():
        assert 0 < row["shift_tol_mm"] <= CAD_CLEARANCE_MM + 0.02, part
        assert row["yaw_tol_deg"] > 0, part
        if row["symmetry"] != "any":
            assert row["yaw_tol_deg"] < YAW_GRID_DEG * 2, part   # a keyed part cannot be turned far
