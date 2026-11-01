import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from perception.anomaly.coverage import bucket, eval_split, export
from perception.anomaly.coverage_eval import auroc, summarise, threshold_for_fpr
from perception.manifest_types import split_for_scene


def _obj(object_id="M1_04", variant="ok", rest="installed", kind="part"):
    return {"uid": "o01", "object_id": object_id, "variant": variant, "kind": kind, "rest": rest}


def _o_index(split: str) -> int:
    return next(i for i in range(1000) if split_for_scene("O", i, 1337) == split)


def test_only_good_installed_o_targets_train_and_held_out_o_goods_are_measured():
    assert bucket("O", _o_index("train"), _obj()) == ("train", "train_good")
    assert bucket("O", _o_index("validation"), _obj()) == ("val", "good")
    assert bucket("O", _o_index("test"), _obj()) == ("test", "good")
    assert bucket("O", _o_index("train"), _obj(rest="fallen")) is None          # fallen goes to the detector
    assert bucket("O", _o_index("train"), _obj(variant="warp")) is None


def test_u_and_t_are_evaluation_only_and_never_train():
    for index in range(50):
        for set_name in ("U", "T"):
            split, folder = bucket(set_name, index, _obj(variant="cracked"))
            assert split in ("val", "test") and folder == "bad"
            assert bucket(set_name, index, _obj())[1] == "good"
    assert bucket("U", 3, _obj(rest="fallen")) is None
    assert bucket("U", 3, _obj(object_id="F01", kind="base")) is None
    assert bucket("N", 3, _obj()) is None


def test_eval_split_is_deterministic_and_uses_both_halves():
    splits = [eval_split("U", i) for i in range(200)]
    assert splits == [eval_split("U", i) for i in range(200)]
    assert 60 < splits.count("val") < 140


def _scene(chunk: Path, stem: str, index: int, obj: dict) -> None:
    label = {"spec": {"index": index, "wrist_target_uid": "o01", "objects": [obj]},
             "cameras": {"wrist": [{"uid": "o01", "bbox_xyxy": [10, 10, 30, 30], "visible_px": 400}]}}
    (chunk / "labels").mkdir(parents=True, exist_ok=True)
    (chunk / "rgb" / "wrist").mkdir(parents=True, exist_ok=True)
    (chunk / "labels" / f"{stem}.json").write_text(json.dumps(label))
    Image.new("RGB", (64, 64)).save(chunk / "rgb" / "wrist" / f"{stem}.jpg")


def test_export_writes_layout_and_index(tmp_path):
    o_chunk, u_chunk = tmp_path / "O" / "chunk_0000", tmp_path / "U" / "chunk_0000"
    _scene(o_chunk, "000000", _o_index("train"), _obj())
    _scene(u_chunk, "000000", 0, _obj(variant="snapped_tab"))
    for chunk in (o_chunk, u_chunk):
        (chunk / "_DONE.json").write_text("{}")
    counts = export(tmp_path, tmp_path / "out", ("O", "U"))
    split = eval_split("U", 0)
    assert counts == {("M1_04", "train_good"): 1, ("M1_04", f"{split}_bad"): 1}
    row = json.loads((tmp_path / "out" / "M1_04" / "index.jsonl").read_text())
    assert row == {"file": f"{split}/bad/U_chunk_0000_000000.png", "split": split, "set": "U",
                   "variant": "snapped_tab"}
    assert (tmp_path / "out" / "M1_04" / row["file"]).is_file()


def test_export_refuses_an_incomplete_chunk(tmp_path):
    _scene(tmp_path / "U" / "chunk_0000", "000000", 0, _obj())
    with pytest.raises(FileNotFoundError, match="_DONE"):
        export(tmp_path, tmp_path / "out", ("U",))


def test_auroc_counts_ties_half_and_orders_correctly():
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 0.0
    assert auroc(np.array([1.0]), np.array([1.0])) == 0.5


def test_threshold_allows_at_most_the_false_alarm_budget():
    good = np.arange(100, dtype=float)                  # 0..99
    t = threshold_for_fpr(good, 0.02)
    assert (good > t).mean() <= 0.02 and t == 97.0
    assert threshold_for_fpr(good, 0.0) == 99.0


def _rows(split, good_scores, bad_scores, set_name="U"):
    return ([{"split": split, "variant": "ok", "set": set_name, "score": s} for s in good_scores]
            + [{"split": split, "variant": "warp", "set": set_name, "score": s} for s in bad_scores])


def test_summarise_sets_threshold_on_val_and_reports_test_once():
    rows = _rows("val", range(50), range(40, 90)) + _rows("test", range(50), range(45, 95))
    result = summarise(rows, 0.02)
    assert result["status"] == "measured"
    assert result["threshold"] == 48.0
    assert result["test"]["fpr"] == pytest.approx(1 / 50)
    assert result["test_per_defect"]["warp"]["bad"] == 50
    assert result["test_fpr_per_set"]["U"]["good"] == 50


def test_summarise_refuses_too_little_support():
    result = summarise(_rows("val", range(50), range(5)) + _rows("test", range(50), range(50)), 0.02)
    assert result["status"] == "insufficient_evaluation_support"


def test_mask_greys_everything_but_the_target_instance(tmp_path):
    from perception.anomaly.coverage import MASK_FILL, masked

    seg = np.zeros((4, 4), dtype=np.uint16)
    seg[1:3, 1:3] = 2                                 # uid o01 -> value 2
    Image.fromarray(seg).save(tmp_path / "seg.png")
    out = np.asarray(masked(Image.new("RGB", (4, 4), (10, 20, 30)), tmp_path / "seg.png", "o01"))
    assert (out[1, 1] == [10, 20, 30]).all() and (out[0, 0] == MASK_FILL).all()
    with pytest.raises(ValueError, match="no pixels"):
        masked(Image.new("RGB", (4, 4)), tmp_path / "seg.png", "o05")
