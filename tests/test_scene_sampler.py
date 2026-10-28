"""Behaviour of the synthetic scene sampler: every guarantee is re-checked from the spec, not from sampler internals."""

import collections
import json
import math

import numpy as np
import pytest

from sdg.scene_sampler import FOREIGN_RADIUS_MM, TCP_BOX, Inputs, chunk_range, n_chunks, sample_scene
from sdg.transforms import translation

N = 120


@pytest.fixture(scope="module")
def inp() -> Inputs:
    return Inputs.load()


@pytest.fixture(scope="module")
def scenes(inp) -> dict[str, list[dict]]:
    return {s: [sample_scene(inp, s, i) for i in range(N)] for s in "ABCD"}


def _placement(inp, obj) -> tuple[np.ndarray, dict]:
    """Undo the object's resting pose: returns the table placement (yaw + xy at table height) and its footprint."""
    tf = np.array(obj["transform_uf0"])
    if obj["kind"] == "foreign":
        return tf, {"footprint_xy": [0.0, 0.0], "footprint_r": FOREIGN_RADIUS_MM[obj["foreign_type"]]}
    geo = inp.geometry[obj["object_id"]]
    if obj["kind"] == "base":
        return tf @ np.linalg.inv(translation(0, 0, -geo["bounds_mm"][0][2])), geo["upright"]
    pose = geo["stable_poses"][obj["stable_pose"]]
    return tf @ np.linalg.inv(np.array(pose["transform"])), pose


def _circle(inp, obj) -> tuple[np.ndarray, float]:
    at, fp = _placement(inp, obj)
    return (at @ np.array([*fp["footprint_xy"], 0.0, 1.0]))[:2], fp["footprint_r"]


def test_same_index_gives_identical_spec_and_different_index_differs(inp):
    assert json.dumps(sample_scene(inp, "A", 7)) == json.dumps(sample_scene(inp, "A", 7))
    assert json.dumps(sample_scene(inp, "A", 7)) != json.dumps(sample_scene(inp, "A", 8))


def test_three_products_each_with_its_base_and_every_part_exactly_once(scenes, inp):
    for spec in (s for v in scenes.values() for s in v):
        assert len(set(spec["products"])) == 3
        for p in spec["products"]:
            mine = [o for o in spec["objects"] if o["product"] == p and o["kind"] in ("base", "part")]
            assert [o["object_id"] for o in mine if o["kind"] == "base"] == [f"F0{p}"]
            parts = [o for o in mine if o["kind"] == "part"]
            assert len(parts) == 4
            assert sum(o["state"] == "installed" for o in parts) == spec["assembly_step"][str(p)]


def test_installed_parts_sit_exactly_where_cad_puts_them(scenes, inp):
    for spec in scenes["A"]:
        bases = {o["uid"]: np.array(o["transform_uf0"]) for o in spec["objects"] if o["kind"] == "base"}
        for o in (o for o in spec["objects"] if o["state"] == "installed"):
            expected = bases[o["on_base"]] @ np.array(inp.geometry[o["object_id"]]["installed_in_base_frame"])
            np.testing.assert_allclose(o["transform_uf0"], expected, atol=1e-3)


def test_loose_parts_and_bases_rest_on_the_table(scenes, inp):
    """Removing the CAD resting pose must leave only a yaw about z at table height (no hidden tilt or float)."""
    table_z = inp.layout["table_z"]
    for spec in scenes["A"]:
        for o in (o for o in spec["objects"] if o["state"] != "installed"):
            at, _ = _placement(inp, o)
            assert at[2, 3] == pytest.approx(table_z, abs=1e-2)
            np.testing.assert_allclose(at[2, :3], [0, 0, 1], atol=1e-3)


def test_table_objects_do_not_overlap_and_stay_in_reach(scenes, inp):
    lay = inp.layout
    for spec in scenes["A"] + scenes["D"]:
        flat = [o for o in spec["objects"] if o["state"] != "installed"]
        circles = [_circle(inp, o) for o in flat]
        for i, (ci, ri) in enumerate(circles):
            if flat[i]["state"] != "in_store":
                assert lay["reach_radius_min"] <= math.hypot(*ci) <= lay["reach_radius_max"]
            for cj, rj in circles[i + 1:]:
                assert np.linalg.norm(ci - cj) >= ri + rj - 1e-6


def test_bases_never_touch_each_other(scenes, inp):
    gap = inp.layout["clearance"]["base_gap_mm"]
    for spec in scenes["A"]:
        circles = [_circle(inp, o) for o in spec["objects"] if o["kind"] == "base"]
        for i, (ci, ri) in enumerate(circles):
            for cj, rj in circles[i + 1:]:
                assert np.linalg.norm(ci - cj) >= ri + rj + gap - 1e-6


def test_set_b_shows_every_object_equally_often_and_target_is_loose(inp):
    counts = collections.Counter()
    for i in range(60):
        spec = sample_scene(inp, "B", i)
        target = next(o for o in spec["objects"] if o["uid"] == spec["wrist_target_uid"])
        counts[target["object_id"]] += 1
        assert target["state"] == "on_table"
    assert len(counts) == 30 and set(counts.values()) == {2}


def test_wrist_tcp_is_inside_the_cell_box_and_points_down(scenes):
    for spec in scenes["B"] + scenes["C"] + scenes["D"]:
        x, y, z, w, p, _ = spec["arm"]["tcp_uf0"]
        for v, axis in ((x, "x"), (y, "y"), (z, "z")):
            assert TCP_BOX[axis][0] <= v <= TCP_BOX[axis][1]
        assert abs(abs(w) - 180.0) <= 20.0 and abs(p) <= 20.0


def test_build_check_target_is_the_top_installed_part(scenes, inp):
    states = collections.Counter()
    for spec in scenes["D"]:
        target = next(o for o in spec["objects"] if o["uid"] == spec["wrist_target_uid"])
        states[target["build_state"]] += 1
        same_base = [o for o in spec["objects"] if o.get("on_base") == target["on_base"]]
        assert target is same_base[-1]
        if target["build_state"] == "wrong_part":
            assert inp.taxonomy[target["object_id"]]["product"] != target["product"]
    assert set(states) == {"seated", "high", "tilted", "wrong_part"}


def test_wrong_part_trap_comes_from_a_product_not_on_the_table(scenes, inp):
    traps = [(s, o) for s in scenes["A"] for o in s["objects"] if o["kind"] == "trap"]
    assert traps
    for spec, o in traps:
        assert inp.taxonomy[o["object_id"]]["product"] not in spec["products"]


def test_patchcore_set_keeps_most_scenes_defect_free(scenes):
    defective = sum(any(o["variant"] != "ok" for o in s["objects"]) for s in scenes["C"])
    assert 0 < defective < N * 0.5


def test_cad_geometry_is_registered_and_pose_probabilities_are_normalised(inp):
    for object_id, geo in inp.geometry.items():
        assert geo["registration_residual_mm"] < 0.01, object_id
        assert sum(p["probability"] for p in geo["stable_poses"]) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("set_name", ["A", "B", "C", "D"])
def test_chunks_cover_every_scene_once_and_match_the_image_budget(inp, set_name):
    cfg = inp.sets["sets"][set_name]
    indices = [i for c in range(n_chunks(inp, set_name)) for i in chunk_range(inp, set_name, c)]
    assert indices == list(range(len(indices)))
    assert len(indices) * len(cfg["cameras"]) == cfg["images"]
    with pytest.raises(ValueError):
        chunk_range(inp, set_name, n_chunks(inp, set_name))


def test_rest_shift_lands_the_lowest_vertex_on_the_table():
    from sdg.transforms import rest_on_table_dz, rot_axis

    verts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 3.0], [0.0, 10.0, -1.5]])
    tf = translation(5, 5, -330.0) @ rot_axis((1, 0, 0), 20)
    dz = rest_on_table_dz(verts, tf, -335.0)
    moved = (translation(0, 0, dz) @ tf @ np.c_[verts, np.ones(3)].T)[2]
    assert moved.min() == pytest.approx(-335.0)


def test_coverage_gate_reports_every_missing_case(inp, scenes):
    from sdg.coverage import count, missing_cases

    c = count(scenes["A"][:5], inp)
    missing = missing_cases(c, inp)
    assert any(m.startswith("defect_part ('M6_04', 'warp')") for m in missing)
    assert len([m for m in missing if m.startswith("product_combo")]) == 20


def test_every_arm_pose_has_an_ik_solution_on_the_render_host(scenes):
    from sdg.wrist_ik import reachable

    poses = [s["arm"]["tcp_uf0"] for set_scenes in scenes.values() for s in set_scenes[:30] if s["arm"]["mode"] != "parked"]
    assert poses and all(reachable(tcp) for tcp in poses)


def test_edge_set_shows_rare_resting_poses_more_often_than_the_nominal_set(inp):
    def rare_share(set_name):
        loose = [o for i in range(60) for o in sample_scene(inp, set_name, i)["objects"] if "stable_pose" in o
                 and o["kind"] == "part"]
        return sum(o["stable_pose"] > 0 for o in loose) / len(loose)

    assert rare_share("H") > rare_share("B") + 0.1


def test_edge_set_widens_the_look_ranges_beyond_the_nominal_ones(inp):
    nominal = inp.sets["randomization"]["camera_jitter"]["translate_mm"]
    shifts = [abs(v) for i in range(60) for v in sample_scene(inp, "F", i)["camera_jitter"]["table"]["translate_mm"]]
    assert max(shifts) > nominal


def test_borderline_build_states_carry_their_offset_inside_the_configured_range(inp):
    cfg = inp.sets["sets"]["J"]
    seen = collections.Counter()
    for i in range(80):
        spec = sample_scene(inp, "J", i)
        target = next(o for o in spec["objects"] if o["uid"] == spec["wrist_target_uid"])
        state, off = target["build_state"], target["build_offset"]
        seen[state] += 1
        if state == "seated":
            assert 0.0 <= off["high_mm"] <= cfg["seated_noise_mm"] and off["tilt_deg"] == 0.0
        elif state == "high":
            assert cfg["high_mm"][0] <= off["high_mm"] <= cfg["high_mm"][1]
        elif state == "tilted":
            assert cfg["tilt_deg"][0] <= off["tilt_deg"] <= cfg["tilt_deg"][1]
    assert set(seen) == {"seated", "high", "tilted", "wrong_part"}
