import hashlib
import json
from pathlib import Path

import pytest

from perception.anomaly.dataset import DatasetBoundaryError, discover_patchcore_samples


def _write_scene(root: Path, set_name: str, index: str, variant: str) -> Path:
    chunk = root / set_name / "chunk_0000"
    for relative in ("labels", "rgb/wrist", "depth/wrist", "seg/wrist"):
        (chunk / relative).mkdir(parents=True, exist_ok=True)
    rgb = chunk / "rgb/wrist" / f"{index}.jpg"
    rgb.write_bytes(f"rgb-{set_name}-{index}".encode())
    (chunk / "depth/wrist" / f"{index}.png").write_bytes(b"depth")
    (chunk / "seg/wrist" / f"{index}.png").write_bytes(b"seg")
    label = {
        "spec": {
            "set": set_name,
            "index": int(index),
            "wrist_target_uid": "o03",
            "objects": [
                {
                    "uid": "o03",
                    "object_id": "M1_01",
                    "kind": "part",
                    "variant": variant,
                    "stable_pose": 2,
                }
            ],
        },
        "cameras": {
            "wrist": [
                {
                    "uid": "o03",
                    "object_id": "M1_01",
                    "variant": variant,
                    "bbox_xyxy": [10, 20, 50, 80],
                    "visible_px": 2400,
                }
            ]
        },
    }
    label_path = chunk / "labels" / f"{index}.json"
    label_path.write_text(json.dumps(label))
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


def test_patchcore_discovery_requires_visible_wrist_target(tmp_path: Path) -> None:
    _write_scene(tmp_path, "C", "000000", "ok")
    label_path = tmp_path / "C/chunk_0000/labels/000000.json"
    label = json.loads(label_path.read_text())
    label["cameras"]["wrist"] = []
    label_path.write_text(json.dumps(label))

    with pytest.raises(DatasetBoundaryError, match="visible wrist target"):
        discover_patchcore_samples(tmp_path)


@pytest.mark.parametrize(("field", "value"), [("set", "REAL"), ("index", 99)])
def test_patchcore_discovery_rejects_label_provenance_mismatch(
    tmp_path: Path, field: str, value: str | int
) -> None:
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
