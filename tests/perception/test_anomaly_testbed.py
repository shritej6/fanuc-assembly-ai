"""Tests for anomaly testbed.
Authored by Shritej Gunda.
"""

import json
from pathlib import Path

from PIL import Image

from perception.anomaly.testbed import bucket, export, square_crop
from perception.manifest_types import split_for_scene


def _obj(object_id="M1_01", variant="ok", kind="part"):
    return {"uid": "o01", "object_id": object_id, "variant": variant, "kind": kind}


def _o_index(split: str) -> int:
    return next(i for i in range(1000) if split_for_scene("O", i, 1337) == split)


def test_o_good_targets_train_or_measure_false_alarms_by_their_scene_split():
    assert bucket("O", _o_index("train"), _obj(), None) == "train_good"
    assert bucket("O", _o_index("test"), _obj(), None) == "o_test_good"
    assert bucket("O", _o_index("validation"), _obj(), None) is None
    assert bucket("O", _o_index("train"), _obj(variant="warp"), None) is None      # never train on a defect


def test_old_sets_only_test_and_only_parts_lying_as_on_the_real_bed():
    installed, fallen = {"o01": "installed"}, {"o01": "fallen"}
    assert bucket("C", 3, _obj(), installed) == "test_good"
    assert bucket("I", 3, _obj(variant="snapped_tab"), installed) == "test_bad"
    assert bucket("C", 3, _obj(variant="snapped_tab"), fallen) is None
    assert bucket("C", 3, _obj(), None) is None
    assert bucket("C", 3, _obj(object_id="F01", kind="base"), installed) is None


def test_square_crop_keeps_context_and_stays_inside_the_image():
    crop = square_crop(Image.new("RGB", (100, 80)), [30, 10, 50, 50])
    assert crop.size == (52, 52)                       # 1.3 x the 40 px long side, square
    edge = square_crop(Image.new("RGB", (100, 80)), [0, 0, 20, 20])
    assert edge.size == (23, 23)


def _chunk(root: Path, set_name: str, variant: str, rest: str | None) -> None:
    chunk = root / set_name / "chunk_0000"
    (chunk / "labels").mkdir(parents=True)
    (chunk / "rgb" / "wrist").mkdir(parents=True)
    index = _o_index("train") if set_name == "O" else 0
    stem = f"{index:06d}"
    Image.new("RGB", (64, 48)).save(chunk / "rgb" / "wrist" / f"{stem}.jpg")
    spec = {"wrist_target_uid": "o01", "objects": [_obj(variant=variant)]}
    box = {"uid": "o01", "bbox_xyxy": [10, 10, 30, 30], "visible_px": 300}
    (chunk / "labels" / f"{stem}.json").write_text(json.dumps({"spec": spec, "cameras": {"wrist": [box]}}))
    if rest:
        (chunk / "_REST.json").write_text(json.dumps({"images": {stem: {"o01": rest}}}))


def test_export_writes_one_folder_per_part_and_counts_what_it_wrote(tmp_path):
    _chunk(tmp_path / "new", "O", "ok", None)
    _chunk(tmp_path / "old", "C", "warp", "installed")
    _chunk(tmp_path / "old", "I", "ok", "fallen")
    counts = export(tmp_path / "new", tmp_path / "old", tmp_path / "out")
    assert dict(counts) == {("M1_01", "train_good"): 1, ("M1_01", "test_bad"): 1}
    assert len(list((tmp_path / "out" / "M1_01" / "test_bad").glob("C_chunk_0000_*.png"))) == 1
