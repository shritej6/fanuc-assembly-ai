"""Tests for training data.
Authored by Shritej Gunda.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from perception.classifier.dataset import CropDataset
from perception.detector.dataset import CocoDataset
from perception.train_common import configure_mixed_precision, seed_training, trainer_arguments


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_training_seed_is_applied_through_transformers(monkeypatch) -> None:
    calls = []
    fake_transformers = type("FakeTransformers", (), {"set_seed": staticmethod(calls.append)})
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    seed_training(1337)

    assert calls == [1337]


def test_mixed_precision_uses_a_bounded_scaler_only_for_fp16(monkeypatch) -> None:
    calls = []

    class FakeGradScaler:
        def __init__(self, device: str, init_scale: int) -> None:
            calls.append((device, init_scale))

    fake_amp = type("Amp", (), {"GradScaler": FakeGradScaler})
    monkeypatch.setitem(sys.modules, "torch", type("Torch", (), {"amp": fake_amp}))
    accelerator = type("Accelerator", (), {"scaler": object()})()
    fp16_trainer = type("Trainer", (), {"accelerator": accelerator})()
    bf16_accelerator = type("Accelerator", (), {"scaler": None})()
    bf16_trainer = type("Trainer", (), {"accelerator": bf16_accelerator})()

    configure_mixed_precision(
        fp16_trainer,
        {"fp16": True, "bf16": False, "fp16_initial_scale": 128},
        RuntimeError,
    )
    configure_mixed_precision(
        bf16_trainer,
        {"fp16": False, "bf16": True},
        RuntimeError,
    )

    assert calls == [("cuda", 128)]
    assert isinstance(fp16_trainer.accelerator.scaler, FakeGradScaler)


def test_mixed_precision_rejects_missing_or_unexpected_scalers(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", type("Torch", (), {}))
    missing_accelerator = type("Accelerator", (), {"scaler": None})()
    missing = type("Trainer", (), {"accelerator": missing_accelerator})()
    unexpected_accelerator = type("Accelerator", (), {"scaler": object()})()
    unexpected = type("Trainer", (), {"accelerator": unexpected_accelerator})()

    with pytest.raises(RuntimeError, match="FP16.*gradient scaler"):
        configure_mixed_precision(
            missing,
            {"fp16": True, "bf16": False, "fp16_initial_scale": 128},
            RuntimeError,
        )
    with pytest.raises(RuntimeError, match="BF16.*gradient scaler"):
        configure_mixed_precision(
            unexpected,
            {"fp16": False, "bf16": True},
            RuntimeError,
        )


class FakeTensor:
    def __init__(self, value: str) -> None:
        self.value = value

    def squeeze(self, dimension: int) -> str:
        assert dimension == 0
        return self.value


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "pixel_values": FakeTensor("processed-pixels"),
            "labels": [{"class_labels": [0]}],
        }


def test_coco_dataset_passes_real_image_and_annotations_to_processor(tmp_path: Path) -> None:
    image_path = tmp_path / "source.jpg"
    Image.new("RGB", (32, 24), "white").save(image_path)
    coco_path = tmp_path / "train.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 7,
                        "file_name": str(image_path),
                        "width": 32,
                        "height": 24,
                        "image_sha256": _sha256(image_path),
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 7,
                        "category_id": 0,
                        "bbox": [2, 3, 10, 11],
                        "area": 110,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 0, "name": "F01"}],
            }
        )
    )
    processor = FakeProcessor()

    dataset = CocoDataset(coco_path, processor)

    assert len(dataset) == 1
    assert dataset[0] == {
        "pixel_values": "processed-pixels",
        "labels": {"class_labels": [0]},
    }
    call = processor.calls[0]
    assert call["images"].size == (32, 24)
    assert call["annotations"] == {
        "image_id": 7,
        "annotations": [
            {
                "id": 1,
                "image_id": 7,
                "category_id": 0,
                "bbox": [2, 3, 10, 11],
                "area": 110,
                "iscrowd": 0,
            }
        ],
    }
    assert call["return_tensors"] == "pt"


def test_crop_dataset_filters_split_and_preserves_scenario_rows(tmp_path: Path) -> None:
    train_image = tmp_path / "train.jpg"
    test_image = tmp_path / "test.jpg"
    Image.new("RGB", (8, 6), "white").save(train_image)
    Image.new("RGB", (8, 6), "black").save(test_image)
    crops_path = tmp_path / "crops.jsonl"
    rows = [
        {
            "split": "train",
            "crop_path": str(train_image),
            "class_id": 2,
            "object_id": "F02",
            "set": "B",
            "variant": "native",
            "crop_sha256": _sha256(train_image),
        },
        {
            "split": "test",
            "crop_path": str(test_image),
            "class_id": 3,
            "object_id": "F03",
            "set": "H",
            "variant": "cosmos",
            "crop_sha256": _sha256(test_image),
        },
    ]
    crops_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    dataset = CropDataset(
        crops_path,
        "test",
        transform=lambda image: ("pixels", image.size),
        labels=("F00", "F01", "F02", "F03"),
    )

    assert len(dataset) == 1
    assert dataset[0] == {"pixel_values": ("pixels", (8, 6)), "labels": 3}
    assert dataset.rows == [rows[1]]


def test_crop_dataset_rejects_class_id_and_object_id_mismatch(tmp_path: Path) -> None:
    image = tmp_path / "crop.jpg"
    Image.new("RGB", (8, 6), "white").save(image)
    crops_path = tmp_path / "crops.jsonl"
    crops_path.write_text(
        json.dumps(
            {
                "split": "train",
                "crop_path": str(image),
                "class_id": 1,
                "object_id": "F00",
                "crop_sha256": _sha256(image),
            }
        )
        + "\n"
    )

    dataset = CropDataset(crops_path, "train", transform=lambda value: value, labels=("F00", "F01"))

    with pytest.raises(ValueError, match="class mapping mismatch"):
        dataset[0]


def test_detector_dataset_rejects_same_dimension_image_byte_replacement(tmp_path: Path) -> None:
    image = tmp_path / "source.jpg"
    Image.new("RGB", (32, 24), "white").save(image)
    coco_path = tmp_path / "train.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": str(image),
                        "width": 32,
                        "height": 24,
                        "image_sha256": _sha256(image),
                    }
                ],
                "annotations": [],
                "categories": [{"id": 0, "name": "F01"}],
            }
        )
    )
    processor = FakeProcessor()
    dataset = CocoDataset(coco_path, processor)
    Image.new("RGB", (32, 24), "black").save(image)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        dataset[0]
    assert processor.calls == []


def test_crop_dataset_rejects_same_dimension_crop_byte_replacement(tmp_path: Path) -> None:
    image = tmp_path / "crop.jpg"
    Image.new("RGB", (8, 6), "white").save(image)
    called = []
    crops_path = tmp_path / "crops.jsonl"
    crops_path.write_text(
        json.dumps(
            {
                "split": "train",
                "crop_path": str(image),
                "class_id": 0,
                "object_id": "F00",
                "crop_sha256": _sha256(image),
            }
        )
        + "\n"
    )
    dataset = CropDataset(crops_path, "train", transform=called.append, labels=("F00",))
    Image.new("RGB", (8, 6), "black").save(image)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        dataset[0]
    assert called == []


@pytest.mark.parametrize("digest", [None, "ABC", "A" * 64])
def test_detector_dataset_rejects_missing_or_malformed_digest(tmp_path: Path, digest: str | None) -> None:
    record = {"id": 1, "file_name": "source.jpg", "width": 1, "height": 1}
    if digest is not None:
        record["image_sha256"] = digest
    coco_path = tmp_path / "train.json"
    coco_path.write_text(json.dumps({"images": [record], "annotations": [], "categories": []}))

    with pytest.raises(ValueError, match="image_sha256"):
        CocoDataset(coco_path, FakeProcessor())


@pytest.mark.parametrize("digest", [None, "ABC", "A" * 64])
def test_crop_dataset_rejects_missing_or_malformed_digest(tmp_path: Path, digest: str | None) -> None:
    row = {"split": "train", "crop_path": "crop.jpg", "class_id": 0, "object_id": "F00"}
    if digest is not None:
        row["crop_sha256"] = digest
    crops_path = tmp_path / "crops.jsonl"
    crops_path.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="crop_sha256"):
        CropDataset(crops_path, "train", transform=lambda value: value, labels=("F00",))


def test_trainer_arguments_keep_precision_and_detection_batch_semantics(tmp_path: Path) -> None:
    training = {
        "seed": 17,
        "num_train_epochs": 2,
        "max_steps": 3,
        "max_train_samples": 16,
        "max_eval_samples": 8,
        "warmup_ratio": 0.05,
        "fp16": True,
        "fp16_initial_scale": 128,
        "bf16": False,
        "tf32": False,
        "remove_unused_columns": False,
        "eval_do_concat_batches": False,
        "per_device_train_batch_size": 2,
    }

    arguments = trainer_arguments(training, tmp_path / "run")

    assert arguments["output_dir"] == str((tmp_path / "run").resolve())
    assert arguments["fp16"] is True
    assert "fp16_initial_scale" not in arguments
    assert arguments["bf16"] is False
    assert arguments["tf32"] is False
    assert arguments["remove_unused_columns"] is False
    assert arguments["eval_do_concat_batches"] is False
    assert "max_train_samples" not in arguments
    assert "max_eval_samples" not in arguments
    assert "warmup_ratio" not in arguments
    assert arguments["warmup_steps"] == 0.05
