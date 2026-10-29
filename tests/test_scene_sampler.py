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
    pose = geo["installed_rest"] if obj.get("rest") == "installed" else geo["stable_poses"][obj["stable_pose"]]
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


@pytest.mark.parametrize("set_name", ["A", "B", "C", "D", "N", "R", "O", "P", "Q"])
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


def test_wrist_camera_keeps_the_nominal_jitter_even_in_hard_look_sets(inp):
    nominal = inp.sets["randomization"]["camera_jitter"]
    for set_name in ("H", "I", "K"):
        for i in range(40):
            j = sample_scene(inp, set_name, i)["camera_jitter"]["wrist"]
            assert max(abs(v) for v in j["translate_mm"]) <= nominal["translate_mm"]
            assert max(abs(v) for v in j["rotate_deg"]) <= nominal["rotate_deg"]


def _build_targets(inp, set_name, n=80):
    for i in range(n):
        spec = sample_scene(inp, set_name, i)
        yield next(o for o in spec["objects"] if o["uid"] == spec["wrist_target_uid"])


def test_tilted_and_swapped_parts_rest_on_the_stack_instead_of_sinking_into_it(inp):
    """A rigid tilt about the bottom centre pushes half the part into the fixture: the clearance lifts it out."""
    for t in _build_targets(inp, "D"):
        if t["build_state"] == "tilted":
            assert t["build_offset"]["high_mm"] > 0.0
        elif t["build_state"] == "wrong_part":
            original = next(k for k, v in inp.clearance["parts"].items() if t["object_id"] in v["wrong_mm"]
                            and k[3:] == t["object_id"][3:] and k[1] == str(t["product"]))
            expected = inp.clearance["parts"][original]["wrong_mm"][t["object_id"]]
            assert t["build_offset"]["high_mm"] == pytest.approx(expected + 0.05 * (expected > 0), abs=1e-3)


def test_misassembly_set_labels_every_shift_and_turn_with_its_measured_rest_height(inp):
    cfg, grid = inp.sets["sets"]["M"], inp.clearance["shift_mm"]
    seen = collections.Counter()
    for t in _build_targets(inp, "M", 120):
        state, off = t["build_state"], t["build_offset"]
        seen[state] += 1
        assert set(off) == {"high_mm", "tilt_deg", "shift_mm", "yaw_deg"}
        if state == "shifted":
            assert off["shift_mm"] in grid and cfg["shift_mm"][0] <= off["shift_mm"] <= cfg["shift_mm"][1]
        elif state == "rotated":
            yaw = inp.clearance["parts"][t["object_id"]]["yaw"][str(int(off["yaw_deg"]))]
            assert yaw["deviation_mm"] >= inp.clearance["yaw_visible_mm"]   # a turn you can actually see
            assert off["high_mm"] == pytest.approx(yaw["lift_mm"] + 0.05 * (yaw["lift_mm"] > 0), abs=1e-3)
    assert set(seen) == {"seated", "shifted", "rotated", "tilted", "high", "wrong_part"}


# ---------------------------------------------------------------- test-bed sets (T25)
TEST_BED = "NROPQ"


@pytest.fixture(scope="module")
def bed(inp) -> dict[str, list[dict]]:
    return {s: [sample_scene(inp, s, i) for i in range(80)] for s in TEST_BED}


def _work_area_loose(spec):
    return [o for o in spec["objects"] if o["state"] == "on_table" and o["kind"] in ("part", "trap")]


def test_every_loose_test_bed_part_is_labelled_installed_or_fallen(bed):
    for spec in (s for v in bed.values() for s in v):
        assert all(o["rest"] in ("installed", "fallen") for o in _work_area_loose(spec))


def test_installed_parts_stand_exactly_as_in_their_product_at_any_yaw_on_the_table(bed, inp):
    """The arm cannot turn a part over: up in the part frame must be the installed up; only yaw and xy are free."""
    table_z, yaws = inp.layout["table_z"], []
    for spec in (s for v in bed.values() for s in v):
        for o in (o for o in _work_area_loose(spec) + [o for o in spec["objects"] if o["state"] == "in_store"]
                  if o.get("rest") == "installed"):
            geo = inp.geometry[o["object_id"]]
            tf = np.array(o["transform_uf0"])
            np.testing.assert_allclose(tf[2, :3], np.array(geo["installed_rest"]["transform"])[2, :3], atol=1e-3)
            at, _ = _placement(inp, o)
            assert at[2, 3] == pytest.approx(table_z, abs=1e-2)
            yaws.append(math.degrees(math.atan2(at[1, 0], at[0, 0])))
    sectors = collections.Counter(int((y + 180) // 30) % 12 for y in yaws)
    assert len(sectors) == 12 and min(sectors.values()) > len(yaws) / 24   # every 30 degree sector, none starved


def test_installed_rest_has_the_same_up_face_as_the_part_inside_its_product(inp):
    """The bed pose must be the product pose: a guide that lies flat in the product lies flat on the bed."""
    parts = [p for p, g in inp.geometry.items() if "installed_rest" in g and not p.startswith("F")]
    assert len(parts) == 24
    for p in parts:
        product_up = np.array(inp.geometry[p]["installed_in_base_frame"])[2, :3]
        bed_up = np.array(inp.geometry[p]["installed_rest"]["transform"])[2, :3]
        np.testing.assert_allclose(bed_up, product_up, atol=1e-3, err_msg=p)


def test_fallen_parts_lie_in_a_different_resting_pose_at_the_configured_share(bed, inp):
    for set_name in ("N", "Q"):
        loose = [o for s in bed[set_name] for o in _work_area_loose(s)]
        fallen = [o for o in loose if o["rest"] == "fallen"]
        expected = inp.sets["sets"][set_name]["scene"]["fallen_prob"]
        assert abs(len(fallen) / len(loose) - expected) < 0.06, set_name
        for o in fallen:
            installed_up = np.array(inp.geometry[o["object_id"]]["installed_rest"]["transform"])[2, :3]
            assert not np.allclose(np.array(o["transform_uf0"])[2, :3], installed_up, atol=1e-2)


def test_edge_standing_plates_appear_standing_although_no_drop_pose_matches(bed):
    plates = [o for s in bed["N"] for o in _work_area_loose(s)
              if o["object_id"] in ("M1_04", "M2_04", "M3_04") and o["rest"] == "installed"]
    assert plates and all(o["stable_pose"] == -1 for o in plates)


def test_test_bed_objects_do_not_overlap_and_spares_stand_ready_to_pick(bed, inp):
    for spec in bed["N"] + bed["Q"]:
        flat = [o for o in spec["objects"] if o["state"] != "installed"]
        circles = [_circle(inp, o) for o in flat]
        for i, (ci, ri) in enumerate(circles):
            for cj, rj in circles[i + 1:]:
                assert np.linalg.norm(ci - cj) >= ri + rj - 1e-6
        assert all(o["rest"] == "installed" for o in spec["objects"] if o["state"] == "in_store")


def test_identity_close_ups_keep_one_wrist_height_so_pixel_size_stays_fixed(bed, inp):
    from sdg.transforms import world_aabb

    lo, hi = inp.sets["sets"]["O"]["wrist"]["standoff_mm"]
    assert hi - lo <= 6.0
    targets = set()
    for spec in bed["O"]:
        target = next(o for o in spec["objects"] if o["uid"] == spec["wrist_target_uid"])
        top = world_aabb(np.array(target["transform_uf0"]), inp.geometry[target["object_id"]]["bounds_mm"])[1, 2]
        assert lo - 1e-3 <= spec["arm"]["tcp_uf0"][2] - top <= hi + 1e-3   # camera height above the part's top face
        targets.add(target["object_id"])
    assert len(targets) == 30


def test_old_sets_never_carry_the_test_bed_label(scenes):
    assert not any("rest" in o for v in scenes.values() for s in v for o in s["objects"])


def test_test_bed_spares_can_be_any_part_but_never_a_base(bed, scenes, inp):
    spares = {o["object_id"] for s in bed["N"] + bed["Q"] for o in s["objects"] if o["kind"] == "spare"}
    assert spares and not any(p.startswith("F") for p in spares)
    assert {p[:2] for p in spares} == {f"M{i}" for i in range(1, 7)}       # every product, not a fixed short list
    old = {o["object_id"] for s in scenes["A"] for o in s["objects"] if o["kind"] == "spare"}
    assert old <= set(inp.sets["defaults"]["spares"])                       # older sets reproduce unchanged
