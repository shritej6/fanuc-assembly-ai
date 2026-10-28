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


def _sample(root: Path, index: int, variant: str) -> PatchCoreSample:
    rgb = root / f"source-{index}.jpg"
    Image.new("RGB", (100, 80), (index * 20, 30, 40)).save(rgb)
    digest = hashlib.sha256(rgb.read_bytes()).hexdigest()
    return PatchCoreSample(
        set_name="C",
        sample_id=f"C/chunk_0000/{index:06d}",
        object_id="M1_01",
        variant=variant,
        stable_pose=index % 6,
        eligibility="train_eligible" if variant == "ok" else "test_only",
        bbox_xyxy=(10, 20, 50, 60),
        segmentation_value=4,
        rgb_path=rgb,
        depth_path=root / f"depth-{index}.png",
        segmentation_path=root / f"seg-{index}.png",
        rgb_sha256=digest,
    )


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _complete_samples(root: Path) -> tuple[PatchCoreSample, ...]:
    return tuple(_sample(root, index, "ok") for index in range(5)) + (
        _sample(root, 5, "missing_feature"),
    )


def test_patchcore_preparation_is_deterministic_and_keeps_defects_out_of_train(
    tmp_path: Path,
) -> None:
    samples = _complete_samples(tmp_path)
    first, second = tmp_path / "prepared-a", tmp_path / "prepared-b"

    summary = prepare_patchcore_folders(
        samples, first, expected_object_ids=("M1_01",), normal_test_modulus=5
    )
    prepare_patchcore_folders(
        samples, second, expected_object_ids=("M1_01",), normal_test_modulus=5
    )

    assert (summary.train_good, summary.test_good, summary.test_bad) == (4, 1, 1)
    assert len(list((first / "M1_01/train/good").glob("*.png"))) == 4
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
    assert (summary.per_object[0].train_good, summary.per_object[0].test_good) == (4, 1)


def test_patchcore_preparation_rejects_source_hash_drift(tmp_path: Path) -> None:
    samples = _complete_samples(tmp_path)
    samples[0].rgb_path.write_bytes(b"changed")

    with pytest.raises(DatasetPreparationError, match="hash changed"):
        prepare_patchcore_folders(
            samples,
            tmp_path / "prepared",
            expected_object_ids=("M1_01",),
            normal_test_modulus=5,
        )


def test_patchcore_preparation_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    output.mkdir()

    with pytest.raises(DatasetPreparationError, match="must not already exist"):
        prepare_patchcore_folders(
            _complete_samples(tmp_path),
            output,
            expected_object_ids=("M1_01",),
            normal_test_modulus=5,
        )


def test_patchcore_preparation_requires_complete_splits_per_object(tmp_path: Path) -> None:
    samples = tuple(_sample(tmp_path, index, "ok") for index in range(5))

    with pytest.raises(DatasetPreparationError, match="complete splits"):
        prepare_patchcore_folders(
            samples,
            tmp_path / "prepared",
            expected_object_ids=("M1_01",),
            normal_test_modulus=5,
        )


def test_patchcore_preparation_rejects_unexpected_object_id(tmp_path: Path) -> None:
    samples = list(_complete_samples(tmp_path))
    samples[0] = replace(samples[0], object_id="M/../../escaped")

    with pytest.raises(DatasetPreparationError, match="expected object IDs"):
        prepare_patchcore_folders(
            samples,
            tmp_path / "prepared",
            expected_object_ids=("M1_01",),
            normal_test_modulus=5,
        )


def test_patchcore_preparation_rejects_invalid_crop_coordinates(tmp_path: Path) -> None:
    samples = list(_complete_samples(tmp_path))
    samples[0] = replace(samples[0], bbox_xyxy=(-1, 20, 50, 60))

    with pytest.raises(DatasetPreparationError, match="invalid crop"):
        prepare_patchcore_folders(
            samples,
            tmp_path / "prepared",
            expected_object_ids=("M1_01",),
            normal_test_modulus=5,
        )


def test_patchcore_preparation_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    samples = list(_complete_samples(tmp_path))
    samples.append(samples[0])

    with pytest.raises(DatasetPreparationError, match="duplicate sample ID"):
        prepare_patchcore_folders(
            samples,
            tmp_path / "prepared",
            expected_object_ids=("M1_01",),
            normal_test_modulus=5,
        )
