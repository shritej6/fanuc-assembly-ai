import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from perception.classifier.train import (
    ClassifierTrainingError,
    _check_class_support,
    _check_finite_metrics,
    _configure_fp16_scaler,
    _parser,
    _run_provenance,
    _validate_manifest_mode,
    classifier_report_from_logits,
    resolve_resume,
    validate_classifier_identity,
)


def _metadata() -> dict:
    labels = [f"class_{index:02d}" for index in range(30)]
    return {
        "labels": {
            "classifier": labels,
            "lookalike_groups": [labels[:2]],
        },
        "classifier": {"sha256": "manifest-abc"},
        "taxonomy_sha256": "taxonomy-abc",
        "split_seed": 1337,
        "git_commit": "abc1234",
    }


def _config() -> dict:
    return {
        "task": "classifier",
        "model": {
            "architecture": "resnet18",
            "weights": "DEFAULT",
            "num_labels": 30,
            "image_size": 224,
        },
        "training": {},
    }


def _provenance() -> dict:
    metadata = _metadata()
    return {
        "manifest_sha256": metadata["classifier"]["sha256"],
        "taxonomy_sha256": metadata["taxonomy_sha256"],
        "split_seed": metadata["split_seed"],
        "labels": metadata["labels"]["classifier"],
        "pretrained_model": "torchvision/resnet18",
        "pretrained_revision": "ResNet18_Weights.IMAGENET1K_V1",
        "manifest_code_commit": metadata["git_commit"],
        "derived_data_sha256": "derived-abc",
        "trainer_code_commit": "code-abc",
        "packages": {"torch": "2.5.1+cu118"},
        "training_config": {"max_steps": 2, "fp16": True},
    }


def test_classifier_report_reduces_logits_and_keeps_scenario_slices() -> None:
    report = classifier_report_from_logits(
        logits=[
            [0.1, 0.9, 0.0],
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
        ],
        label_ids=[0, 1, 2],
        rows=[
            {"set": "B", "variant": "native"},
            {"set": "B", "variant": "native"},
            {"set": "H", "variant": "cosmos"},
        ],
        labels=("A", "B", "C"),
        lookalike_groups=(("A", "B"),),
    )

    assert report["accuracy"] == pytest.approx(1 / 3)
    assert report["confusion_matrix"] == [[0, 1, 0], [0, 1, 0], [1, 0, 0]]
    assert report["lookalike_groups"]["A|B"]["confused_within_group"] == 1
    assert report["scenarios"]["set"] == {
        "B": {"samples": 2, "correct": 1, "accuracy": 0.5},
        "H": {"samples": 1, "correct": 0, "accuracy": 0.0},
    }


def test_classifier_identity_requires_exact_30_label_manifest() -> None:
    labels, groups = validate_classifier_identity(_config(), _metadata())

    assert len(labels) == 30
    assert groups == ((labels[0], labels[1]),)

    invalid = _metadata()
    invalid["labels"]["classifier"] = invalid["labels"]["classifier"][:-1]
    with pytest.raises(ClassifierTrainingError, match="30 classifier labels"):
        validate_classifier_identity(_config(), invalid)


def test_resume_requires_and_checks_run_provenance(tmp_path: Path) -> None:
    current = _provenance()

    with pytest.raises(ClassifierTrainingError, match="provenance"):
        resolve_resume("latest", tmp_path, current)

    (tmp_path / "run_provenance.json").write_text(json.dumps(current))
    (tmp_path / "checkpoint-1").mkdir()
    assert resolve_resume("latest", tmp_path, current) is True

    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    assert resolve_resume(str(checkpoint), tmp_path, current) == str(checkpoint.resolve())

    changed = dict(current)
    changed["manifest_sha256"] = "different"
    with pytest.raises(ClassifierTrainingError, match="manifest_sha256"):
        resolve_resume("latest", tmp_path, changed)


def test_new_run_writes_provenance_and_existing_checkpoint_requires_resume(tmp_path: Path) -> None:
    current = _provenance()

    assert resolve_resume(None, tmp_path, current) is None
    assert json.loads((tmp_path / "run_provenance.json").read_text()) == current
    (tmp_path / "checkpoint-2").mkdir()

    with pytest.raises(ClassifierTrainingError, match="resume"):
        resolve_resume(None, tmp_path, current)


def test_classifier_resume_rejects_changed_crop_bytes_identity(tmp_path: Path) -> None:
    crops = tmp_path / "crops.jsonl"
    crops.write_text(json.dumps({"crop_sha256": "a" * 64}) + "\n")
    first = _run_provenance(_metadata(), _config(), crops)
    output = tmp_path / "run"
    resolve_resume(None, output, first)
    (output / "checkpoint-1").mkdir()

    crops.write_text(json.dumps({"crop_sha256": "b" * 64}) + "\n")
    changed = _run_provenance(_metadata(), _config(), crops)

    assert first["derived_data_sha256"] != changed["derived_data_sha256"]
    with pytest.raises(ClassifierTrainingError, match="derived_data_sha256"):
        resolve_resume("latest", output, changed)


def test_classifier_parser_supports_bounded_resume_probe(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--crops",
            str(tmp_path / "crops.jsonl"),
            "--manifest-metadata",
            str(tmp_path / "metadata.json"),
            "--output-dir",
            str(tmp_path / "run"),
            "--smoke",
            "--max-steps",
            "3",
            "--resume-from-checkpoint",
        ]
    )

    assert args.smoke is True
    assert args.max_steps == 3
    assert args.resume_from_checkpoint == "latest"


def test_classifier_rejects_non_finite_measured_metrics() -> None:
    _check_finite_metrics({"loss": 0.4, "samples": 8}, "validation")

    with pytest.raises(ClassifierTrainingError, match="non-finite"):
        _check_finite_metrics({"loss": float("nan")}, "training")


def test_classifier_replaces_default_fp16_scaler_with_bounded_scale(monkeypatch) -> None:
    calls = []

    class FakeGradScaler:
        def __init__(self, device: str, init_scale: int) -> None:
            calls.append((device, init_scale))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(amp=SimpleNamespace(GradScaler=FakeGradScaler)))
    trainer = SimpleNamespace(accelerator=SimpleNamespace(scaler=object()))

    _configure_fp16_scaler(trainer, {"fp16_initial_scale": 128})

    assert calls == [("cuda", 128)]
    assert isinstance(trainer.accelerator.scaler, FakeGradScaler)


def test_classifier_rejects_missing_fp16_scaler(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    trainer = SimpleNamespace(accelerator=SimpleNamespace(scaler=None))

    with pytest.raises(ClassifierTrainingError, match="gradient scaler"):
        _configure_fp16_scaler(trainer, {"fp16_initial_scale": 128})


def test_classifier_production_requires_cosmos_manifest() -> None:
    _validate_manifest_mode({"require_cosmos": False, "include_cosmos": False}, smoke=True)

    with pytest.raises(ClassifierTrainingError, match="Cosmos"):
        _validate_manifest_mode({"require_cosmos": False, "include_cosmos": False}, smoke=False)
    with pytest.raises(ClassifierTrainingError, match="Cosmos"):
        _validate_manifest_mode({"require_cosmos": True, "include_cosmos": False}, smoke=False)


def test_classifier_requires_every_label_in_every_production_split() -> None:
    labels = ("F00", "F01", "F02")
    complete = SimpleNamespace(rows=[{"object_id": label} for label in labels])

    _check_class_support((complete, complete, complete), labels)

    incomplete = SimpleNamespace(rows=[{"object_id": "F00"}, {"object_id": "F01"}])
    with pytest.raises(ClassifierTrainingError, match=r"validation.*F02"):
        _check_class_support((complete, incomplete, complete), labels)
