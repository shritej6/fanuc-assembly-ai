import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from perception.detector.train import (
    DetectorTrainError,
    _check_class_support,
    _check_finite_history,
    _detection_metric_options,
    _evaluate_scenarios,
    _flatten_detection_metrics,
    _gather_for_metrics,
    _size_pair,
    _validate_manifest_mode,
    build_parser,
    build_run_provenance,
    prepare_resume,
)

CONFIG = {
    "task": "detector",
    "model": {
        "name": "PekingU/rtdetr_r50vd",
        "revision": "df939e661d8c52e80608d1ec566561aabd25a4e7",
        "image_size": 640,
    },
    "training": {},
}
METADATA = {
    "git_commit": "abc1234",
    "split_seed": 17,
    "taxonomy_sha256": "taxonomy-abc",
    "labels": {"detector": ["F01", "foreign_screw_m3"]},
    "detector": {"sha256": "detector-abc"},
}


def _coco_dir(tmp_path: Path) -> Path:
    path = tmp_path / "coco"
    path.mkdir()
    for split in ("train", "validation", "test"):
        (path / f"{split}.json").write_text(json.dumps({"split": split}))
    return path


def test_parser_supports_bounded_smoke_and_latest_resume(tmp_path: Path) -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--coco-dir",
            str(tmp_path / "coco"),
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
    assert args.coco_dir == tmp_path / "coco"


def test_run_provenance_keeps_frozen_data_and_model_identity(tmp_path: Path) -> None:
    provenance = build_run_provenance(METADATA, CONFIG, _coco_dir(tmp_path))

    assert provenance["manifest_sha256"] == "detector-abc"
    assert provenance["taxonomy_sha256"] == "taxonomy-abc"
    assert provenance["labels"] == ["F01", "foreign_screw_m3"]
    assert provenance["pretrained_revision"] == "df939e661d8c52e80608d1ec566561aabd25a4e7"
    assert set(provenance["packages"]) == {
        "accelerate",
        "pycocotools",
        "scipy",
        "torch",
        "torchmetrics",
        "torchvision",
        "transformers",
    }


def test_new_run_writes_provenance_and_latest_resume_checks_it(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    identity = build_run_provenance(METADATA, CONFIG, _coco_dir(tmp_path))

    assert prepare_resume(output_dir, None, identity) is None
    assert json.loads((output_dir / "run_provenance.json").read_text()) == identity
    (output_dir / "checkpoint-2").mkdir()
    assert prepare_resume(output_dir, "latest", identity) is True


def test_existing_checkpoint_requires_explicit_resume(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    identity = build_run_provenance(METADATA, CONFIG, _coco_dir(tmp_path))
    prepare_resume(output_dir, None, identity)
    (output_dir / "checkpoint-2").mkdir()

    with pytest.raises(DetectorTrainError, match="resume"):
        prepare_resume(output_dir, None, identity)


def test_detector_resume_rejects_changed_source_bytes_identity(tmp_path: Path) -> None:
    coco_dir = _coco_dir(tmp_path)
    train_path = coco_dir / "train.json"
    train_path.write_text(json.dumps({"images": [{"image_sha256": "a" * 64}]}))
    first = build_run_provenance(METADATA, CONFIG, coco_dir)
    output = tmp_path / "run"
    prepare_resume(output, None, first)
    (output / "checkpoint-1").mkdir()

    train_path.write_text(json.dumps({"images": [{"image_sha256": "b" * 64}]}))
    changed = build_run_provenance(METADATA, CONFIG, coco_dir)

    assert first["derived_data_sha256"] != changed["derived_data_sha256"]
    with pytest.raises(RuntimeError, match="derived_data_sha256"):
        prepare_resume(output, "latest", changed)


def test_explicit_resume_path_must_exist_and_match_provenance(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    identity = build_run_provenance(METADATA, CONFIG, _coco_dir(tmp_path))
    prepare_resume(output_dir, None, identity)
    checkpoint = output_dir / "checkpoint-2"
    checkpoint.mkdir()

    assert prepare_resume(output_dir, str(checkpoint), identity) == str(checkpoint.resolve())

    changed = {**identity, "manifest_sha256": "other"}
    with pytest.raises(RuntimeError, match="manifest_sha256"):
        prepare_resume(output_dir, str(checkpoint), changed)

    with pytest.raises(DetectorTrainError, match="does not exist"):
        prepare_resume(output_dir, str(output_dir / "checkpoint-9"), identity)

    outside = tmp_path / "other-run" / "checkpoint-2"
    outside.mkdir(parents=True)
    with pytest.raises(DetectorTrainError, match="inside"):
        prepare_resume(output_dir, str(outside), identity)


class _MetricValue:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


def test_undefined_torchmetrics_sentinels_are_not_reported_as_measurements() -> None:
    metrics = {
        "classes": [_MetricValue(0), _MetricValue(1)],
        "map_per_class": [_MetricValue(0.4), _MetricValue(-1.0)],
        "mar_100_per_class": [_MetricValue(0.5), _MetricValue(-1.0)],
        "map": _MetricValue(0.3),
        "map_small": _MetricValue(-1.0),
    }

    flattened = _flatten_detection_metrics(metrics, {0: "F01", 1: "F02"})

    assert flattened == {
        "map": 0.3,
        "map_F01": 0.4,
        "mar_100_F01": 0.5,
    }

    invalid = {**metrics, "map": _MetricValue(float("nan"))}
    with pytest.raises(DetectorTrainError, match="non-finite"):
        _flatten_detection_metrics(invalid, {0: "F01", 1: "F02"})


def test_detection_metric_does_not_resynchronize_trainer_gathered_predictions() -> None:
    assert _detection_metric_options() == {
        "box_format": "xyxy",
        "class_metrics": True,
        "sync_on_compute": False,
    }


def test_singleton_target_size_is_normalized_for_scenario_evaluation() -> None:
    assert _size_pair([[480, 640]]) == (480, 640)

    with pytest.raises(DetectorTrainError, match="two values"):
        _size_pair([480])


class _GatherRecorder:
    def __init__(self) -> None:
        self.object_flags = []

    def gather_for_metrics(self, value, use_gather_object: bool):
        self.object_flags.append(use_gather_object)
        return value


def test_ragged_detection_targets_use_object_gather_without_changing_logits() -> None:
    accelerator = _GatherRecorder()
    targets = [{"boxes": [[0.1, 0.2, 0.3, 0.4]], "class_labels": [1], "orig_size": [1080, 1920]}]
    logits = ([0.1], [[0.2, 0.3, 0.4, 0.5]])

    assert _gather_for_metrics(accelerator, targets) is targets
    assert _gather_for_metrics(accelerator, logits) is logits
    assert accelerator.object_flags == [True, False]


def test_detector_rejects_non_finite_gradient_history() -> None:
    _check_finite_history([{"loss": 2.0, "grad_norm": 4.0}])

    with pytest.raises(DetectorTrainError, match="grad_norm"):
        _check_finite_history([{"loss": 2.0, "grad_norm": float("nan")}])


def test_detector_production_requires_cosmos_manifest() -> None:
    _validate_manifest_mode({"require_cosmos": False, "include_cosmos": False}, smoke=True)

    with pytest.raises(DetectorTrainError, match="Cosmos"):
        _validate_manifest_mode({"require_cosmos": False, "include_cosmos": False}, smoke=False)
    with pytest.raises(DetectorTrainError, match="Cosmos"):
        _validate_manifest_mode({"require_cosmos": True, "include_cosmos": False}, smoke=False)


def test_detector_requires_every_category_in_every_production_split() -> None:
    labels = ["F01", "F02", "foreign_screw_m3"]
    complete = SimpleNamespace(
        annotations={1: [{"category_id": index} for index in range(len(labels))]}
    )

    _check_class_support((complete, complete, complete), labels)

    incomplete = SimpleNamespace(annotations={1: [{"category_id": 0}, {"category_id": 1}]})
    with pytest.raises(DetectorTrainError, match=r"test.*foreign_screw_m3"):
        _check_class_support((complete, complete, incomplete), labels)


class _ScenarioDataset:
    def __init__(self, images: list[dict]) -> None:
        self.images = images
        self.categories = ()
        self.annotations = {}
        self.processor = object()

    def __len__(self) -> int:
        return len(self.images)


class _ScenarioTrainer:
    def evaluate(self, eval_dataset, metric_key_prefix: str) -> dict[str, float]:
        return {f"{metric_key_prefix}_map": len(eval_dataset) / 10}


def test_detector_scenario_reports_include_measured_sample_counts() -> None:
    dataset = _ScenarioDataset(
        [
            {"set": "A", "camera": "table", "variant": "native", "has_trap": False},
            {"set": "G", "camera": "side", "variant": "native", "has_trap": True},
        ]
    )

    report = _evaluate_scenarios(_ScenarioTrainer(), dataset)

    assert report["set=A"]["samples"] == 1
    assert report["set=A"]["metrics"]["scenario_000_map"] == 0.1
    assert report["trap=present"]["samples"] == 1
