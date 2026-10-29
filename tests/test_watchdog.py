"""Watchdog behaviour on fake chunk folders: sampled QA, incomplete chunks, bounded retries, render plan."""

import json

import numpy as np
import pytest
from PIL import Image

from sdg import watchdog


def _scene(chunk, idx: int, box=(1, 1, 3, 3)) -> None:
    name = f"{idx:06d}"
    seg = np.zeros((6, 8), dtype=np.uint16)
    seg[1:3, 1:3] = 1                                   # uid o00 -> 1
    for sub, img in (("seg", Image.fromarray(seg)), ("rgb", Image.new("RGB", (8, 6)))):
        (chunk / sub / "wrist").mkdir(parents=True, exist_ok=True)
        img.save(chunk / sub / "wrist" / f"{name}.{'png' if sub == 'seg' else 'jpg'}")
    (chunk / "labels").mkdir(exist_ok=True)
    label = {"spec": {"outputs": ["rgb"]}, "cameras": {"wrist": [
        {"uid": "o00", "object_id": "M1_01", "kind": "part", "variant": "ok", "bbox_xyxy": list(box)}]}}
    (chunk / "labels" / f"{name}.json").write_text(json.dumps(label))


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(watchdog, "DATA_ROOT", tmp_path)
    return tmp_path


def _chunk(root, scenes: int, expected: int, bad_box=False):
    chunk = root / "synthetic" / "B" / "chunk_0000"
    chunk.mkdir(parents=True)
    for i in range(scenes):
        _scene(chunk, i, box=(0, 0, 3, 3) if bad_box and i == 0 else (1, 1, 3, 3))
    (root / "specs" / "B").mkdir(parents=True)
    (root / "specs" / "B" / "chunk_0000.jsonl").write_text("{}\n" * expected)
    (chunk / "_DONE.json").write_text("{}")
    return chunk


def test_finished_chunk_is_checked_once_and_passes(root):
    chunk = _chunk(root, scenes=3, expected=3)
    assert watchdog.qa_finished(chunk)["ok"]
    assert (root / "watchdog" / "sheets" / "B_chunk_0000.png").exists()
    assert watchdog.qa_finished(chunk) is None          # already checked


def test_missing_scenes_fail_the_chunk(root):
    assert not watchdog.qa_finished(_chunk(root, scenes=2, expected=3))["ok"]


def test_wrong_box_fails_the_chunk(root):
    result = watchdog.qa_finished(_chunk(root, scenes=2, expected=2, bad_box=True))
    assert not result["ok"] and result["errors"] == 1


def test_a_failed_chunk_is_retried_at_most_three_times(tmp_path):
    for attempt in range(1, 5):
        (tmp_path / "_FAILED.txt").write_text("boom")
        assert watchdog.take_failure(tmp_path) is (attempt <= watchdog.MAX_RETRIES)
    assert watchdog.retries(tmp_path) == watchdog.MAX_RETRIES


def test_plan_lists_chunks_in_render_order():
    assert watchdog.parse_plan(["C:14-15", "D:0-1"]) == [("C", 14), ("C", 15), ("D", 0), ("D", 1)]


def test_render_noise_is_flagged_only_on_quiet_sensor_scenes(tmp_path):
    from sdg.qa import _check_render_noise

    rng = np.random.default_rng(0)
    seg = np.zeros((40, 60), dtype=np.uint16)
    clean, noisy = tmp_path / "clean.png", tmp_path / "noisy.png"
    Image.fromarray(np.full((40, 60, 3), 200, dtype=np.uint8)).save(clean)
    Image.fromarray(rng.integers(120, 255, (40, 60, 3), dtype=np.uint8)).save(noisy)
    quiet = {"post": {"motion_blur": False, "gaussian_noise_sigma": 0.001}}
    loud = {"post": {"motion_blur": False, "gaussian_noise_sigma": 0.03}}
    assert _check_render_noise("0", "wrist", clean, seg, quiet) == []
    assert _check_render_noise("0", "wrist", noisy, seg, quiet)
    assert _check_render_noise("0", "wrist", noisy, seg, loud) == []


def test_one_stalled_render_is_killed_while_the_parallel_ones_keep_going(root, monkeypatch):
    import os
    import time
    stalled, busy = (root / "synthetic" / "D" / "chunk_0003", root / "synthetic" / "A" / "chunk_0001")
    for chunk in (stalled, busy):
        _scene(chunk, 0)
    old = time.time() - (watchdog.HANG_MIN + 5) * 60
    for p in [stalled, *stalled.rglob("*")]:
        os.utime(p, (old, old))
    monkeypatch.setattr(watchdog, "render_containers", lambda: {"D/chunk_0003.jsonl": "c1", "A/chunk_0001.jsonl": "c2"})
    killed = []
    monkeypatch.setattr(watchdog, "run", lambda cmd, **kw: killed.append(cmd[-1]))
    events = []
    watchdog.spark_hang_check(events)
    assert killed == ["c1"]
    assert (stalled / "_FAILED.txt").exists() and not (busy / "_FAILED.txt").exists()
    assert events == ["HANG D/chunk_0003: killed its render, will retry"]
