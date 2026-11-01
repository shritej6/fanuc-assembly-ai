"""Tests for anomaly prepare.
Authored by Shritej Gunda.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from perception.anomaly.dataset import (
    DatasetPreparationError,
    PatchCoreSample,
    prepare_patchcore_folders,
)


def _o_index(split: str) -> int:
    for index in range(10_000):
        token = f"1337:O:{index}".encode()
        bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % 10_000
        assigned = "train" if bucket < 8_000 else "validation" if bucket < 9_000 else "test"
        if assigned == split:
            return index
    raise AssertionError(f"no O index for split {split}")


def _rest_file(root: Path, set_name: str, index: int, rest: str | None) -> Path | None:
    if rest is None:
        return None
    path = root / f"rest-{set_name}-{index}.json"
    path.write_text(rest)
    return path


def _sample(
    root: Path,
    set_name: str,
    index: int,
    variant: str,
    eligibility: str,
    assigned_split: str,
    rest: str | None,
) -> PatchCoreSample:
    rgb = root / f"source-{set_name}-{index}.jpg"
    depth = root / f"depth-{set_name}-{index}.png"
    segmentation = root / f"seg-{set_name}-{index}.png"
    label = root / f"label-{set_name}-{index}.json"
    Image.new("RGB", (100, 80), (index % 10 * 20, 30, 40)).save(rgb)
    depth.write_bytes(b"depth")
    segmentation.write_bytes(b"segmentation")
    label.write_text("{}")
    rest_path = _rest_file(root, set_name, index, rest)
    return PatchCoreSample(
        set_name=set_name,
        sample_id=f"{set_name}/chunk_0000/{index:06d}",
        object_id="M1_01",
        variant=variant,
        stable_pose=index % 6,
        eligibility=eligibility,
        bbox_xyxy=(10, 20, 50, 60),
        segmentation_value=4,
        rgb_path=rgb,
        depth_path=depth,
        segmentation_path=segmentation,
        label_path=label,
        rgb_sha256=hashlib.sha256(rgb.read_bytes()).hexdigest(),
        depth_sha256=hashlib.sha256(depth.read_bytes()).hexdigest(),
        segmentation_sha256=hashlib.sha256(segmentation.read_bytes()).hexdigest(),
        label_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
        rest_sidecar_path=rest_path,
        rest_sidecar_sha256=hashlib.sha256(rest_path.read_bytes()).hexdigest() if rest_path else None,
        rest=rest,
        assigned_split=assigned_split,
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _complete_samples(root: Path) -> tuple[PatchCoreSample, ...]:
    return (
        _sample(root, "O", _o_index("train"), "ok", "train_eligible", "train/good", None),
        _sample(root, "O", _o_index("test"), "ok", "false_alarm_only", "o_test_good", None),
        _sample(root, "C", 100, "ok", "evaluation_only", "test/good", "installed"),
        _sample(root, "I", 101, "missing_feature", "evaluation_only", "test/bad", "installed"),
    )


def test_patchcore_preparation_is_deterministic_and_keeps_defects_out_of_train(
    tmp_path: Path,
) -> None:
    samples = _complete_samples(tmp_path)
    first, second = tmp_path / "prepared-a", tmp_path / "prepared-b"

    summary = prepare_patchcore_folders(samples, first, expected_object_ids=("M1_01",))
    prepare_patchcore_folders(samples, second, expected_object_ids=("M1_01",))

    assert (summary.train_good, summary.test_good, summary.test_bad, summary.o_test_good) == (1, 1, 1, 1)
    assert len(list((first / "M1_01/train/good").glob("*.png"))) == 1
    assert len(list((first / "M1_01/test/good").glob("*.png"))) == 1
    assert len(list((first / "M1_01/test/bad").glob("*.png"))) == 1
    assert Image.open(next((first / "M1_01/test/bad").glob("*.png"))).size == (40, 40)
    assert _tree(first) == _tree(second)
    rows = [json.loads(line) for line in (first / "manifest.jsonl").read_text().splitlines()]
    assert all(row["variant"] == "ok" for row in rows if row["split"] != "test/bad")
    assert all(row["variant"] != "ok" for row in rows if row["split"] == "test/bad")
    assert (first / "_READY.json").is_file()
    assert not (first / "_INCOMPLETE").exists()
    assert summary.per_object[0].object_id == "M1_01"
    assert (summary.per_object[0].train_good, summary.per_object[0].test_good) == (1, 1)
    source_fields = {
        "source_rgb_sha256",
        "source_depth_sha256",
        "source_segmentation_sha256",
        "source_label_sha256",
        "source_rest_sha256",
    }
    assert all(source_fields <= row.keys() for row in rows)


def test_patchcore_preparation_rejects_source_hash_drift(tmp_path: Path) -> None:
    samples = _complete_samples(tmp_path)
    samples[0].rgb_path.write_bytes(b"changed")

    with pytest.raises(DatasetPreparationError, match="hash changed"):
        prepare_patchcore_folders(
            samples,
            tmp_path / "prepared",
            expected_object_ids=("M1_01",),
        )


@pytest.mark.parametrize(
    "field",
    ["label_path", "depth_path", "segmentation_path", "rest_sidecar_path"],
)
def test_patchcore_preparation_rejects_semantic_source_drift(tmp_path: Path, field: str) -> None:
    samples = _complete_samples(tmp_path)
    sample = samples[2] if field == "rest_sidecar_path" else samples[0]
    path = getattr(sample, field)
    assert path is not None
    path.write_bytes(b"changed after discovery")

    with pytest.raises(DatasetPreparationError, match="source hash changed"):
        prepare_patchcore_folders(
            samples,
            tmp_path / "prepared",
            expected_object_ids=("M1_01",),
        )


def test_patchcore_preparation_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    output.mkdir()

    with pytest.raises(DatasetPreparationError, match="must not already exist"):
        prepare_patchcore_folders(
            _complete_samples(tmp_path),
            output,
            expected_object_ids=("M1_01",),
        )


def test_patchcore_preparation_requires_complete_splits_per_object(tmp_path: Path) -> None:
    samples = _complete_samples(tmp_path)[:-1]

    with pytest.raises(DatasetPreparationError, match="complete splits"):
        prepare_patchcore_folders(
            samples,
            tmp_path / "prepared",
            expected_object_ids=("M1_01",),
        )


def test_patchcore_preparation_rejects_unexpected_object_id(tmp_path: Path) -> None:
    samples = list(_complete_samples(tmp_path))
    samples[0] = replace(samples[0], object_id="M/../../escaped")

    with pytest.raises(DatasetPreparationError, match="expected object IDs"):
        prepare_patchcore_folders(
            samples,
            tmp_path / "prepared",
            expected_object_ids=("M1_01",),
        )


def test_patchcore_preparation_rejects_invalid_crop_coordinates(tmp_path: Path) -> None:
    samples = list(_complete_samples(tmp_path))
    samples[0] = replace(samples[0], bbox_xyxy=(-1, 20, 50, 60))

    with pytest.raises(DatasetPreparationError, match="invalid crop"):
        prepare_patchcore_folders(
            samples,
            tmp_path / "prepared",
            expected_object_ids=("M1_01",),
        )


def test_patchcore_preparation_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    samples = list(_complete_samples(tmp_path))
    samples.append(samples[0])

    with pytest.raises(DatasetPreparationError, match="duplicate sample ID"):
        prepare_patchcore_folders(
            samples,
            tmp_path / "prepared",
            expected_object_ids=("M1_01",),
        )
