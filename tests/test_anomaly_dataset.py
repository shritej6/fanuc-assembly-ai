import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from perception.anomaly.dataset import (
    DatasetBoundaryError,
    discover_patchcore_samples,
    prepare_patchcore_folders,
)


def _label_payload(set_name: str, index: str, variant: str, object_id: str, rest: str | None) -> dict:
    target = {
        "uid": "o03",
        "object_id": object_id,
        "kind": "part",
        "variant": variant,
        "stable_pose": 2,
    }
    if rest is not None:
        target["rest"] = rest
    return {
        "spec": {
            "set": set_name,
            "index": int(index),
            "wrist_target_uid": "o03",
            "objects": [target],
        },
        "cameras": {
            "wrist": [
                {
                    "uid": "o03",
                    "object_id": object_id,
                    "variant": variant,
                    "bbox_xyxy": [10, 20, 50, 80],
                    "visible_px": 2400,
                }
            ]
        },
    }


def _write_scene(
    root: Path,
    set_name: str,
    index: str,
    variant: str,
    *,
    object_id: str = "M1_01",
    rest: str | None = None,
) -> Path:
    chunk = root / set_name / "chunk_0000"
    for relative in ("labels", "rgb/wrist", "depth/wrist", "seg/wrist"):
        (chunk / relative).mkdir(parents=True, exist_ok=True)
    rgb = chunk / "rgb/wrist" / f"{index}.jpg"
    Image.new("RGB", (100, 100), (20, 30, 40)).save(rgb)
    (chunk / "depth/wrist" / f"{index}.png").write_bytes(b"depth")
    (chunk / "seg/wrist" / f"{index}.png").write_bytes(b"seg")
    label_path = chunk / "labels" / f"{index}.json"
    label_path.write_text(json.dumps(_label_payload(set_name, index, variant, object_id, rest)))
    return rgb


def test_patchcore_discovery_keeps_defects_out_of_training(tmp_path: Path) -> None:
    good_rgb = _write_scene(tmp_path, "C", "000000", "ok")
    _write_scene(tmp_path, "I", "000001", "missing_feature")

    samples = discover_patchcore_samples(tmp_path)

    assert [(item.set_name, item.variant, item.eligibility) for item in samples] == [
        ("C", "ok", "train_eligible"),
        ("I", "missing_feature", "test_only"),
    ]
    assert samples[0].object_id == "M1_01"
    assert samples[0].stable_pose == 2
    assert samples[0].bbox_xyxy == (10, 20, 50, 80)
    assert samples[0].segmentation_value == 4
    assert samples[0].rgb_sha256 == hashlib.sha256(good_rgb.read_bytes()).hexdigest()


def test_t_discovery_accepts_only_installed_rear_guards_as_evaluation_only(tmp_path: Path) -> None:
    _write_scene(tmp_path, "T", "000000", "ok", object_id="M1_04", rest="installed")
    _write_scene(tmp_path, "T", "000001", "warp", object_id="M2_04", rest="installed")
    _write_scene(tmp_path, "T", "000002", "ok", object_id="M3_04", rest="installed")
    _write_scene(tmp_path, "T", "000003", "ok", object_id="M3_04", rest="fallen")
    _write_scene(tmp_path, "T", "000004", "ok", object_id="M4_04", rest="installed")

    samples = discover_patchcore_samples(tmp_path)

    assert [
        (sample.sample_id, sample.object_id, sample.variant, sample.eligibility, sample.rest) for sample in samples
    ] == [
        ("T/chunk_0000/000000", "M1_04", "ok", "evaluation_only", "installed"),
        ("T/chunk_0000/000001", "M2_04", "warp", "evaluation_only", "installed"),
        ("T/chunk_0000/000002", "M3_04", "ok", "evaluation_only", "installed"),
    ]


def test_t_discovery_supports_a_separate_evaluation_root(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    evaluation_root = tmp_path / "evaluation"
    _write_scene(training_root, "C", "000000", "ok", object_id="M1_04")
    _write_scene(evaluation_root, "T", "000100", "warp", object_id="M1_04", rest="installed")

    samples = discover_patchcore_samples(training_root, evaluation_root=evaluation_root)

    assert [(sample.sample_id, sample.eligibility) for sample in samples] == [
        ("C/chunk_0000/000000", "train_eligible"),
        ("T/chunk_0000/000100", "evaluation_only"),
    ]


def test_t_evaluation_samples_never_enter_patchcore_training(tmp_path: Path) -> None:
    for index in range(5):
        _write_scene(tmp_path, "C", f"{index:06d}", "ok", object_id="M1_04")
    _write_scene(tmp_path, "C", "000005", "missing_feature", object_id="M1_04")
    _write_scene(tmp_path, "T", "000100", "ok", object_id="M1_04", rest="installed")
    _write_scene(tmp_path, "T", "000101", "warp", object_id="M1_04", rest="installed")

    samples = discover_patchcore_samples(tmp_path)
    output = tmp_path / "prepared"
    summary = prepare_patchcore_folders(
        samples,
        output,
        expected_object_ids=("M1_04",),
        normal_test_modulus=5,
    )

    assert (summary.train_good, summary.test_good, summary.test_bad) == (4, 2, 2)
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    t_rows = [row for row in rows if row["sample_id"].startswith("T/")]
    assert [(row["variant"], row["rest"], row["split"]) for row in t_rows] == [
        ("ok", "installed", "test/good"),
        ("warp", "installed", "test/bad"),
    ]
    assert not list((output / "M1_04/train/good").glob("T__*.png"))


@pytest.mark.parametrize("rest", ["installed", "fallen"])
def test_t_discovery_validates_artifacts_before_pose_filter(tmp_path: Path, rest: str) -> None:
    _write_scene(tmp_path, "T", "000000", "ok", object_id="M1_04", rest=rest)
    (tmp_path / "T/chunk_0000/depth/wrist/000000.png").unlink()

    with pytest.raises(DatasetBoundaryError, match="required anomaly artifact is missing"):
        discover_patchcore_samples(tmp_path)


@pytest.mark.parametrize("rest", [None, "sideways"])
def test_t_discovery_rejects_missing_or_unknown_rest_provenance(tmp_path: Path, rest: str | None) -> None:
    _write_scene(tmp_path, "T", "000000", "ok", object_id="M1_04", rest=rest)

    with pytest.raises(DatasetBoundaryError, match="rest provenance"):
        discover_patchcore_samples(tmp_path)


def test_t_discovery_rejects_label_path_provenance_mismatch(tmp_path: Path) -> None:
    _write_scene(tmp_path, "T", "000000", "ok", object_id="M1_04", rest="installed")
    label_path = tmp_path / "T/chunk_0000/labels/000000.json"
    label = json.loads(label_path.read_text())
    label["spec"]["set"] = "S"
    label_path.write_text(json.dumps(label))

    with pytest.raises(DatasetBoundaryError, match="provenance"):
        discover_patchcore_samples(tmp_path)


def test_patchcore_discovery_requires_visible_wrist_target(tmp_path: Path) -> None:
    _write_scene(tmp_path, "C", "000000", "ok")
    label_path = tmp_path / "C/chunk_0000/labels/000000.json"
    label = json.loads(label_path.read_text())
    label["cameras"]["wrist"] = []
    label_path.write_text(json.dumps(label))

    with pytest.raises(DatasetBoundaryError, match="visible wrist target"):
        discover_patchcore_samples(tmp_path)


@pytest.mark.parametrize(("field", "value"), [("set", "REAL"), ("index", 99)])
def test_patchcore_discovery_rejects_label_provenance_mismatch(tmp_path: Path, field: str, value: str | int) -> None:
    _write_scene(tmp_path, "C", "000000", "ok")
    label_path = tmp_path / "C/chunk_0000/labels/000000.json"
    label = json.loads(label_path.read_text())
    label["spec"][field] = value
    label_path.write_text(json.dumps(label))

    with pytest.raises(DatasetBoundaryError, match="provenance"):
        discover_patchcore_samples(tmp_path)


def test_patchcore_discovery_rejects_noncanonical_chunk(tmp_path: Path) -> None:
    _write_scene(tmp_path, "C", "000000", "ok")
    (tmp_path / "C/chunk_0000").rename(tmp_path / "C/chunk_rejected")

    with pytest.raises(DatasetBoundaryError, match="chunk name"):
        discover_patchcore_samples(tmp_path)


def test_patchcore_discovery_rejects_invalid_bbox(tmp_path: Path) -> None:
    _write_scene(tmp_path, "C", "000000", "ok")
    label_path = tmp_path / "C/chunk_0000/labels/000000.json"
    label = json.loads(label_path.read_text())
    label["cameras"]["wrist"][0]["bbox_xyxy"] = [5, 5, 1, 1]
    label_path.write_text(json.dumps(label))

    with pytest.raises(DatasetBoundaryError, match="metadata"):
        discover_patchcore_samples(tmp_path)


def test_patchcore_discovery_wraps_malformed_target(tmp_path: Path) -> None:
    _write_scene(tmp_path, "C", "000000", "ok")
    label_path = tmp_path / "C/chunk_0000/labels/000000.json"
    label = json.loads(label_path.read_text())
    label["spec"]["objects"] = [None]
    label_path.write_text(json.dumps(label))

    with pytest.raises(DatasetBoundaryError, match="invalid wrist target"):
        discover_patchcore_samples(tmp_path)


def test_patchcore_discovery_wraps_invalid_visible_pixel_count(tmp_path: Path) -> None:
    _write_scene(tmp_path, "C", "000000", "ok")
    label_path = tmp_path / "C/chunk_0000/labels/000000.json"
    label = json.loads(label_path.read_text())
    label["cameras"]["wrist"][0]["visible_px"] = "bad"
    label_path.write_text(json.dumps(label))

    with pytest.raises(DatasetBoundaryError, match="visible wrist target"):
        discover_patchcore_samples(tmp_path)


def test_patchcore_discovery_rejects_symlinked_set(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _write_scene(outside, "C", "000000", "ok")
    synthetic = tmp_path / "synthetic"
    synthetic.mkdir()
    (synthetic / "C").symlink_to(outside / "C", target_is_directory=True)

    with pytest.raises(DatasetBoundaryError, match="symlink"):
        discover_patchcore_samples(synthetic)


def test_patchcore_discovery_rejects_symlinked_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _write_scene(outside, "C", "000000", "ok")
    synthetic = tmp_path / "synthetic"
    synthetic.symlink_to(outside, target_is_directory=True)

    with pytest.raises(DatasetBoundaryError, match="root.*symlink"):
        discover_patchcore_samples(synthetic)
