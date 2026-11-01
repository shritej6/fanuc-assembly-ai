"""Tests for objects.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

from sim.assembly_lab.objects import lookalikes as L
from sim.assembly_lab.objects import planner

DOC = yaml.safe_load(planner.OBJECTS_YAML.read_text())
OBJ = DOC["objects"]
REQUIRED = ("id", "product", "role", "sheet_order", "bbox_mm", "distinguishing_dims_mm", "mass_g", "com_mm", "grips",
            "allowed_rolls_deg", "pegs", "holes", "mates_with", "seated_height_mm", "insert_axis", "pull_out_axis",
            "max_force", "insertion", "installed_origin_mm")


def test_all_30_objects_with_every_interface_field():
    assert len(OBJ) == 30
    assert sorted(OBJ) == sorted([f"F0{n}" for n in range(1, 7)] + [f"M{n}_0{k}" for n in range(1, 7)
                                                                     for k in range(1, 5)])
    for oid, e in OBJ.items():
        missing = [f for f in REQUIRED if f not in e]
        assert not missing, (oid, missing)


@pytest.fixture(scope="module")
def fresh():
    pytest.importorskip("cadquery")
    from sim.assembly_lab.objects import build_objects
    doc, _ = build_objects.build()
    return yaml.safe_load(yaml.safe_dump(build_objects._clean(doc), sort_keys=False))


def test_committed_yaml_is_what_the_cad_produces(fresh):
    assert fresh == DOC


def test_mass_is_pla_density_times_cad_volume():
    for e in OBJ.values():
        assert e["mass_g"] == pytest.approx(e["volume_mm3"] * 1.24e-3, abs=0.01)
        assert 0 < e["mass_g"] < 60.0          # parameters.py MAX_PART_MASS_G


def test_sheet_order_and_base_types():
    for n, base_type in zip(range(1, 7), "AABCCC"):
        assert OBJ[f"F0{n}"]["sheet_order"] == 0 and OBJ[f"F0{n}"]["base_type"] == base_type
        assert [OBJ[f"M{n}_0{k}"]["sheet_order"] for k in range(1, 5)] == [1, 2, 3, 4]


def test_seated_heights_from_the_cad_stack():
    # base 8 mm thick (top at z 0); guard origin z 4 + 34 tall; lock origin z 18 + 17 tall; housings 16 / 20 tall
    assert OBJ["M1_04"]["seated_height_mm"] == pytest.approx(46.0)
    assert OBJ["M4_04"]["seated_height_mm"] == pytest.approx(43.0)
    assert OBJ["M1_01"]["seated_height_mm"] == pytest.approx(24.0)
    assert OBJ["M4_01"]["seated_height_mm"] == pytest.approx(28.0)


def test_every_fit_is_half_a_millimetre_and_chamfers_only_widen_capture():
    for oid, e in OBJ.items():
        ins = e["insertion"]
        if e["sheet_order"] == 0:
            assert ins is None
            continue
        assert ins["xy_clearance_mm"] == pytest.approx(0.5), oid
        assert ins["xy_capture_mm"] >= ins["xy_clearance_mm"], oid
        assert ins["axis"] == [0.0, 0.0, -1.0] and e["pull_out_axis"] == [0.0, 0.0, 1.0]


def test_yaw_tolerance_matches_feature_radius_and_beats_the_1_deg_gate():
    for oid, e in OBJ.items():
        ins = e["insertion"]
        if not ins or ins["r_eff_mm"] is None:
            continue
        assert ins["yaw_tol_cad_deg"] == pytest.approx(math.degrees(0.5 / ins["r_eff_mm"]), abs=1e-3)
        assert ins["yaw_tol_deg"] <= ins["yaw_tol_cad_deg"] + 1e-9
        assert ins["yaw_tol_deg"] > 1.0, oid     # INTERFACES.md precision gate: yaw within 1 deg
    # housings: footprint half-length in the stops sets the CAD bound (0.5 / 17 rad for the 34 mm control housing)
    assert OBJ["M1_01"]["insertion"]["r_eff_mm"] == pytest.approx(17.0)


def test_round_carriers_are_yaw_free():
    for oid in ("M1_02", "M2_02"):
        assert OBJ[oid]["insertion"]["r_eff_mm"] is None


def test_grips_are_the_cad_pads_with_both_rolls_allowed():
    for e in OBJ.values():
        if e["sheet_order"] == 0:
            assert e["grips"] == [] and e["allowed_rolls_deg"] == []
            continue
        g = e["grips"][0]
        assert 4.5 < g["pad_width_mm"] < 16.5
        assert len(e["allowed_rolls_deg"]) == 2
        assert abs(abs(e["allowed_rolls_deg"][0] - e["allowed_rolls_deg"][1]) - 180.0) < 1e-6


def test_identical_meshes_have_zero_difference_and_others_a_positive_one():
    pairs = {k: v for g in DOC["lookalike_groups"] for k, v in g["pairs"].items()}
    for k in ("F01~F02", "F04~F05", "F04~F06", "F05~F06"):
        assert pairs[k]["identical"] and pairs[k]["best_cue_mm"] == 0.0
    for k, v in pairs.items():
        if k not in ("F01~F02", "F04~F05", "F04~F06", "F05~F06"):
            assert v["best_cue_mm"] >= 0.8, k


def test_known_lookalike_differences():
    vs = OBJ["M1_02"]["distinguishing_dims_mm"]["vs"]["M2_02"]
    assert vs["bbox_size_diff_mm"] == [0.0, 0.0, 1.0]           # post 6 vs 5 mm
    guards = OBJ["M1_04"]["distinguishing_dims_mm"]["vs"]
    assert guards["M2_04"]["bbox_size_diff_mm"][0] == -2.0       # bridge 25 vs 27 mm
    assert guards["M3_04"]["bbox_size_diff_mm"][0] == -4.0       # bridge 25 vs 29 mm


def test_compare_finds_a_step():
    flat = {"origin": (0.0, 0.0), "res": 0.5, "top": np.full((20, 20), 5.0), "bottom": np.zeros((20, 20))}
    bump = {**flat, "top": flat["top"].copy()}
    bump["top"][4:10, 4:8] = 6.5
    cues = L.compare(bump, flat, 0.0, 0.0)
    assert len(cues) == 1 and cues[0]["size_mm"] == pytest.approx(1.5) and cues[0]["area_mm2"] == pytest.approx(6.0)


def test_feature_check_rejects_a_moved_peg():
    pytest.importorskip("cadquery")
    from sim.assembly_lab.objects import features as F
    from sim.assembly_lab.objects import meshes as M
    frame = M.grasp_frames()["M4_04"]
    tri = M.local_triangles(frame["name"], frame["T_local_from_stl"])
    feats = F.object_features(4, 4, {0: "F04", 1: "M4_01", 2: "M4_02", 3: "M4_03", 4: "M4_04"})
    F.check_features("M4_04", tri, feats)
    feats["pegs"][0]["xy_mm"] = [0.0, 0.0]        # the leg is not at the lock centre
    with pytest.raises(ValueError):
        F.check_features("M4_04", tri, feats)


def _run(pair, cond, cap, seated, lat=0.1, tq=0.001):
    return {"pair": pair, "condition": cond, "cap_N": cap, "seated": seated, "final_dz_mm": 0.0 if seated else 3.0,
            "final_xy_err_mm": [0.0, 0.0], "peak_axial_N": cap, "peak_lateral_N": lat, "peak_torque_Nm": tq,
            "pre_seat_peak_axial_N": 0.5, "pre_seat_peak_lateral_N": lat, "pre_seat_peak_torque_Nm": tq}


def test_seating_summary_brackets_the_force_and_flags_limit_breaks():
    from sim.assembly_lab.physics import summarize_seating as S
    rows = [_run("A", "aligned", 15, True), _run("A", "aligned", 5, True), _run("A", "aligned", 2, False),
            _run("B", "offset_x_0.5", 15, True, lat=6.0), _run("B", "offset_x_0.5", 5, False, lat=6.0),
            _run("C", "aligned", 15, False),
            _run("D_onto_X", "aligned", 15, False), _run("D_onto_X", "aligned", 5, True), _run("D_onto_X", "aligned", 2, True)]
    table = S.summarize(rows)
    assert table["A"]["aligned"]["min_cap_seated_N"] == 5
    assert table["C"]["aligned"]["min_cap_seated_N"] is None
    assert table["D_onto_X"]["aligned"]["press_sink_cap_N"] == 15
    assert table["D_onto_X"]["aligned"]["max_cap_seated_N"] == 5
    base = {"axial_N": 15.0, "lateral_N": 4.5, "torque_Nm": 0.1, "method": {"safety_factor_min": 2.0}}
    res = S.check(table, base)
    assert res["max_min_cap_seated_N_allowed_conditions"] == 15 and res["axial_limit_covers_seating"]
    assert any("B offset_x_0.5: lateral" in v for v in res["violations"])
    assert any("C aligned: does not seat" in v for v in res["violations"])
    assert any("D_onto_X aligned: sinks" in v for v in res["violations"])
    capped = S.check(table, {**base, "per_part_axial_N": {"D": {"axial_N": 7.5}}})
    assert not any(v.startswith("D_onto_X") for v in capped["violations"])
    too_low = S.check(table, {**base, "per_part_axial_N": {"D": {"axial_N": 1.0}}})
    assert any("D_onto_X aligned: needs 2 N > axial limit 1.0" in v for v in too_low["violations"])
