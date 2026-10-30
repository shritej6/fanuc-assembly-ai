"""MimicGen over agent 3's Isaac Lab recorder demos: reading kept episodes, object-centric transforms, the row
surgery generate.py does before replay, and the export's acceptance rule. Host-only (no Isaac Sim)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from sim.assembly_lab.mimic import generate as GEN
from sim.assembly_lab.mimic import mimicgen as M

Z_OFF_M = 1.23  # world z of UF0 (table top + 0.33 m), as measured in run10 files


def _quat_xyzw(rot: np.ndarray) -> np.ndarray:
    from sim.assembly_lab.env.layout import quat_xyzw

    return np.asarray(quat_xyzw(rot))


def _pose7_world(t_mm: np.ndarray) -> np.ndarray:
    return np.r_[t_mm[:3, 3] / 1000.0 + (0, 0, Z_OFF_M), _quat_xyzw(t_mm[:3, :3])]


def _write_episode(g, part_path: list[np.ndarray], tcp_rows: np.ndarray, frames: dict, marks: dict | None,
                   meta_attrs: bool, kept: bool) -> dict:
    """One recorder episode: states/ post-step (row t = after action t), obs/ pre-step, quats x, y, z, w."""
    n = len(tcp_rows)
    g["actions"] = np.c_[tcp_rows, np.r_[np.zeros(n // 2), np.ones(n - n // 2)]].astype(np.float32)
    g["obs/tcp_pose"] = tcp_rows.astype(np.float32)
    for name, t in frames.items():
        g[f"obs/{name}_pos"] = np.tile(t[:3, 3] / 1000.0, (n, 1)).astype(np.float32)
        g[f"obs/{name}_quat"] = np.tile(_quat_xyzw(t[:3, :3]), (n, 1)).astype(np.float32)
        g[f"states/rigid_object/{name}/root_pose"] = np.tile(_pose7_world(t), (n, 1)).astype(np.float32)
    poses = np.stack([_pose7_world(t) for t in part_path])
    g["initial_state/rigid_object/M4_01/root_pose"] = poses[:1].astype(np.float32)
    g["states/rigid_object/M4_01/root_pose"] = poses[1:].astype(np.float32)
    g.attrs["success"] = False  # what the recorder writes after its reset-time overwrite (run10)
    meta = {"steps": n, "kept": kept, "case": "nominal", "object_id": "M4_01", "product": 4,
            "marks": marks or {}, "seat_tries": [{"ok": True}]}
    if meta_attrs:
        for k, v in meta.items():
            g.attrs[k] = v if isinstance(v, (int, str, bool)) else json.dumps(v)
    return meta


@pytest.fixture
def recorded(tmp_path):
    """A pick-and-carry episode where the part sits 12 mm below the TCP, rolled 10 deg in the fingers."""
    n = 40
    hand = M.pose_to_matrix([0.0, 0.0, -12.0, 0.0, 0.0, 10.0])
    tcp = np.array([[300.0 + 2 * k, -40.0, -280.0 + (5.0 if k > 20 else 0.0), -180.0, 0.0, 30.0] for k in range(n)])
    seen = np.r_[tcp, tcp[-1:]]  # part[t] = what observation t saw; states row t = after action t = part[t + 1]
    part = [M.pose_to_matrix(r) @ hand for r in seen]
    grasp = M.pose_to_matrix(tcp[0])
    seat = M.pose_to_matrix([420.0, 60.0, -300.0, -180.0, 0.0, 75.0])
    frames = {"grasp_frame": grasp, "seat_frame": seat, "spot_frame": seat}
    path = tmp_path / "product_4_build_x.hdf5"
    with h5py.File(path, "w") as f:
        _write_episode(f.create_group("data/demo_0"), part, tcp, frames, {"grasped": 10, "seated": 30}, True, True)
        _write_episode(f.create_group("data/demo_1"), part, tcp, frames, {"grasped": 10, "seated": 30}, True, False)
        m2 = _write_episode(f.create_group("data/demo_2"), part, tcp, frames, None, False, True)
    m2["marks"] = {"grasped": 11, "seated": 31}
    metas = [{"steps": n, "kept": True}, {"steps": n, "kept": False}, m2]
    path.with_name(path.name + ".meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metas))
    return path, hand, part, frames


def _identity_local(uid, body):
    return body


def test_read_recorded_uses_kept_and_meta_fallback(recorded):
    path, hand, part, _ = recorded
    demos = M.read_recorded(str(path), to_local=_identity_local)
    assert [d.key for d in demos] == ["demo_0", "demo_2"]          # demo_1 not kept; success attr ignored
    assert demos[1].marks == {"grasped": 11, "seated": 31}         # from meta.jsonl (file not annotated yet)
    d = demos[0]
    assert np.allclose(d.part[0], part[0], atol=1e-3)               # world m -> UF0 mm, z shift recovered
    assert np.allclose(M.in_hand_at(d, 25), hand, atol=1e-3)        # obs[t] pairs with states[t - 1]


def test_held_segment_frame_is_the_planned_seated_part_pose(recorded):
    path, hand, part, frames = recorded
    d = M.read_recorded(str(path), to_local=_identity_local)[0]
    segs = M.demo_segments(d, ["grasped", "seated"], ["grasp_frame", "seat_frame", "seat_frame"], [False, True, True])
    want = frames["seat_frame"] @ np.linalg.inv(frames["grasp_frame"]) @ part[0]
    assert np.allclose(segs[1].frame_pose, want, atol=1e-3) and np.allclose(segs[1].in_hand, hand, atol=1e-3)
    assert segs[0].in_hand is None and len(segs[0].rows) == 10


def test_transform_held_moves_the_part_not_just_the_tcp():
    rng = np.random.default_rng(1)
    src_frame = M.pose_to_matrix([400.0, 0.0, -320.0, 0.0, 0.0, 20.0])
    new_frame = M.pose_to_matrix([300.0, 150.0, -320.0, 0.0, 0.0, -100.0])
    h_src = M.pose_to_matrix([0.4, -0.3, -12.0, 1.0, -0.5, 3.0])
    h_new = M.pose_to_matrix([-0.6, 0.2, -11.5, -0.8, 0.4, 183.0])  # other pad, part pivoted the other way
    rows = np.c_[np.tile([400.0, 0.0, -300.0], (5, 1)) + rng.normal(0, 3, (5, 3)), np.tile([-180.0, 0.0, 20.0], (5, 1)),
                 np.ones(5)]
    out = M.transform_held(rows, src_frame, new_frame, h_src, h_new)
    for a, b in zip(rows, out):
        part_src = np.linalg.inv(src_frame) @ M.pose_to_matrix(a[:6]) @ h_src
        part_new = np.linalg.inv(new_frame) @ M.pose_to_matrix(b[:6]) @ h_new
        assert np.allclose(part_src, part_new, atol=1e-9)


def _rows(xyz, grip):
    xyz = np.asarray(xyz, float)
    return np.c_[xyz, np.tile([-180.0, 0.0, 0.0], (len(xyz), 1)), grip]


def test_trim_to_corridor_drops_the_source_travel():
    travel = [[87.0 + 40 * k, 0.0, 100.0] for k in range(8)]            # from home across the table
    corridor = [[400.0, 5.0, 100.0 - 20 * k] for k in range(6)]          # down onto the pad
    rows = _rows(travel + corridor + [[400.0, 5.0, 0.0]] * 3, np.r_[np.zeros(14), np.ones(3)])
    out = GEN.trim_to_corridor(rows)
    assert np.allclose(out[0, :3], [400.0, 5.0, 100.0]) and len(out) == 9


def test_split_after_close_and_retreat_trim():
    rows = _rows([[0, 0, 10], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 5]], [0, 0, 1, 1, 1])
    head, tail = GEN.split_after_close(rows)
    assert len(head) == 4 and len(tail) == 1
    rel = _rows([[0, 0, 0], [0, 0, 0], [0, 0, 3], [0, 0, 6], [50, 0, 6]], [1, 0, 0, 0, 0])
    assert len(GEN.trim_after_retreat(rel)) == 4                      # the move home is re-planned live


def test_align_r_lands_on_the_planned_roll():
    rows = _rows([[0, 0, 0]] * 3, [0, 0, 0])
    rows[:, 5] = [170.0, 175.0, 178.0]
    out = GEN.align_r(rows, -182.0, at=2)
    assert np.allclose(out[:, 5], [-190.0, -185.0, -182.0])


def test_reason_buckets_drop_numbers():
    a = GEN.reason_key("EpisodeFailed: force guard: M4_02 normal_N 25.44 > 15.00 at step 270")
    b = GEN.reason_key("EpisodeFailed: force guard: M4_02 normal_N 19.10 > 15.00 at step 12")
    assert a == b and "25" not in a
    assert GEN.reason_key("TcpError: no IK solution for TCP [1.0, 2.0]") == "TcpError: no IK solution for TCP"


def test_export_accepts_on_agent3_kept():
    pytest.importorskip("pandas")
    from sim.assembly_lab.mimic.export_lerobot import accepted_attr

    assert accepted_attr({"kept": "True", "success": False})
    assert not accepted_attr({"kept": False, "success": True})
    assert accepted_attr({"success": True}) and not accepted_attr({})


def test_sources_skip_drop_tags_and_carry_drift(recorded, tmp_path):
    path, *_ = recorded
    sig, refs, held = ["grasped", "seated"], ["grasp_frame", "seat_frame", "seat_frame"], [False, True, True]
    segs, counts = M.load_sources([path], sig, refs, held, to_local=_identity_local)
    assert counts["used"] == 2 and len(segs[1]) == 2
    path.with_name(path.name + ".drop.json").write_text("[2]")              # agent 3 tagged meta line 2
    segs, counts = M.load_sources([path], sig, refs, held, to_local=_identity_local)
    assert counts["used"] == 1 and counts["skipped"] == {"drop tag": 1}
    with h5py.File(path, "a") as f:
        f["data/demo_0"].attrs["in_hand_drift_mm"] = json.dumps({"carry": 0.6, "contact": 0.1})
    _, counts = M.load_sources([path], sig, refs, held, to_local=_identity_local)
    assert counts["used"] == 0 and counts["skipped"]["carry drift"] == 1


def test_stage_seeds_copies_only_closed_files_with_drop_lists(tmp_path):
    from sim.assembly_lab.mimic.stage_seeds import stage

    demos = tmp_path / "demos"
    (demos / "pick_insert").mkdir(parents=True)
    for name in ("product_4_build_a_M4_01.hdf5", "product_4_build_b_M4_02.hdf5"):
        (demos / "pick_insert" / name).write_bytes(b"x")
        (demos / "pick_insert" / f"{name}.meta.jsonl").write_text(
            json.dumps({"kept": True, "product": 4, "case": "nominal"}) + "\n"
            + json.dumps({"kept": True, "product": 4, "case": "nominal"}) + "\n")
    (demos / "run10_closed.txt").write_text("demos/pick_insert/product_4_build_a_M4_01.hdf5\n")
    (demos / "run10_drop_tags.jsonl").write_text(json.dumps(
        {"file": "demos/pick_insert/product_4_build_a_M4_01.hdf5", "meta_line": 1}) + "\n")
    counts = stage(demos, tmp_path / "out", "run10_closed.txt", "run10_drop_tags.jsonl")
    staged = sorted(p.name for p in (tmp_path / "out/pick_insert").iterdir())
    assert staged == ["product_4_build_a_M4_01.hdf5", "product_4_build_a_M4_01.hdf5.drop.json",
                      "product_4_build_a_M4_01.hdf5.meta.jsonl"]
    assert json.loads((tmp_path / "out/pick_insert/product_4_build_a_M4_01.hdf5.drop.json").read_text()) == [1]
    assert counts == {"pick_insert:4:nominal": 1}


def test_chunk_paths_keep_worker_stats_at_the_base_name(tmp_path):
    out = tmp_path / "pick_insert/p4/host_g3_w0.hdf5"
    assert GEN.chunk_path(out, 0).name == "host_g3_w0_c000.hdf5"
    assert GEN.chunk_path(out, 12).name == "host_g3_w0_c012.hdf5"
    assert GEN.chunk_path(out, 3).stem.rsplit("_c", 1)[1] == "003"


def test_pull_out_trims_only_free_space_subtasks():
    from sim.assembly_lab.mimic.generate import SCENARIO, free_space_starts, plan_roll
    from sim.assembly_lab.mimic.spec import load_spec

    spec = load_spec()
    assert free_space_starts(spec.skills["pull_out"].subtasks) == [True, False, True, False]
    assert free_space_starts(spec.skills["pick_insert"].subtasks) == [True, True, False]
    assert SCENARIO["pull_out"] == "fix_above"

    plans = (SimpleNamespace(grasp_tcp=[0, 0, 0, 180, 0, 33.0]), SimpleNamespace(spot_tcp=[0, 0, 0, 180, 0, -90.0]))
    assert plan_roll(plans, "grasp_frame") == 33.0 and plan_roll(plans, "spot_frame") == -90.0
