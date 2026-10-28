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
