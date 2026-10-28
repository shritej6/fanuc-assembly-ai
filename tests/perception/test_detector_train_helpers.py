import json
from pathlib import Path

import pytest

from perception.detector.train import (
    DetectorTrainError,
    _evaluate_scenarios,
    _flatten_detection_metrics,
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


def test_detector_production_requires_cosmos_manifest() -> None:
    _validate_manifest_mode({"require_cosmos": False, "include_cosmos": False}, smoke=True)

    with pytest.raises(DetectorTrainError, match="Cosmos"):
        _validate_manifest_mode({"require_cosmos": False, "include_cosmos": False}, smoke=False)
    with pytest.raises(DetectorTrainError, match="Cosmos"):
        _validate_manifest_mode({"require_cosmos": True, "include_cosmos": False}, smoke=False)


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
