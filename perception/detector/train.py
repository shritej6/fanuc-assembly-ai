"""Fine-tune RT-DETR on a frozen T13 COCO manifest.

Heavy ML dependencies are imported only after argument, configuration, and
checkpoint-provenance validation so ordinary repository tests stay lightweight.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from perception.detector.dataset import CocoDataset
from perception.detector.scenarios import scenario_datasets
from perception.provenance import (
    assert_resume_compatible,
    current_git_commit,
    derived_data_identity,
    package_versions,
    provenance_from_manifest,
    read_provenance,
    write_provenance,
)
from perception.train_common import load_manifest_metadata, trainer_arguments
from perception.training_config import apply_smoke_step_override, load_training_config

PACKAGE_NAMES = (
    "accelerate",
    "pycocotools",
    "scipy",
    "torch",
    "torchmetrics",
    "torchvision",
    "transformers",
)


class DetectorTrainError(RuntimeError):
    """Raised when a detector run cannot start or finish safely."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--coco-dir", type=Path, required=True)
    parser.add_argument("--manifest-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="latest",
        help="resume the newest checkpoint, or the explicit checkpoint path",
    )
    return parser


def build_run_provenance(
    metadata: dict[str, Any],
    config: dict[str, Any],
    coco_dir: Path,
) -> dict[str, Any]:
    model = config["model"]
    provenance = provenance_from_manifest(
        metadata,
        task="detector",
        pretrained_model=model["name"],
        pretrained_revision=model["revision"],
    )
    provenance["packages"] = package_versions(PACKAGE_NAMES)
    provenance["trainer_code_commit"] = current_git_commit()
    provenance["training_config"] = dict(config["training"])
    provenance.update(
        derived_data_identity(
            {split: coco_dir / f"{split}.json" for split in ("train", "validation", "test")}
        )
    )
    return provenance


def _validate_manifest_mode(metadata: dict[str, Any], smoke: bool) -> None:
    require_cosmos = metadata.get("require_cosmos")
    if not isinstance(require_cosmos, bool):
        raise DetectorTrainError("manifest metadata lacks a boolean require_cosmos gate")
    includes_cosmos = metadata.get("include_cosmos")
    if not smoke and (not require_cosmos or includes_cosmos is not True):
        raise DetectorTrainError("production detector training requires a Cosmos-complete manifest")


def _checkpoint_dirs(output_dir: Path) -> list[Path]:
    return sorted(path for path in output_dir.glob("checkpoint-*") if path.is_dir())


def prepare_resume(
    output_dir: Path,
    requested: str | None,
    current: dict[str, Any],
) -> bool | str | None:
    """Persist identity for a new run or validate it before checkpoint resume."""
    output_dir = output_dir.resolve()
    provenance_path = output_dir / "run_provenance.json"
    if not provenance_path.exists():
        if requested:
            raise DetectorTrainError(f"cannot resume without {provenance_path}")
        write_provenance(provenance_path, current)
        return None
    assert_resume_compatible(current, read_provenance(provenance_path))
    if requested is None:
        if _checkpoint_dirs(output_dir):
            raise DetectorTrainError("output contains checkpoints; pass --resume-from-checkpoint")
        return None
    if requested == "latest":
        if not _checkpoint_dirs(output_dir):
            raise DetectorTrainError(f"no checkpoint exists in {output_dir}")
        return True
    checkpoint = Path(requested).resolve()
    if not checkpoint.is_dir():
        raise DetectorTrainError(f"checkpoint does not exist: {checkpoint}")
    if checkpoint.parent != output_dir:
        raise DetectorTrainError(f"checkpoint must be inside {output_dir}: {checkpoint}")
    return str(checkpoint)


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack fixed-size RT-DETR images while retaining variable-length labels."""
    import torch

    collated = {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": [item["labels"] for item in batch],
    }
    if "pixel_mask" in batch[0]:
        collated["pixel_mask"] = torch.stack([item["pixel_mask"] for item in batch])
    return collated


def _targets_for_metrics(target_batches: Any) -> tuple[list[Any], list[dict[str, Any]]]:
    import torch
    from transformers.image_transforms import center_to_corners_format

    image_sizes = []
    targets = []
    for batch in target_batches:
        image_sizes.append(torch.tensor([target["orig_size"] for target in batch]))
        for target in batch:
            size = target["orig_size"]
            boxes = center_to_corners_format(torch.tensor(target["boxes"]))
            height, width = size
            boxes = boxes * torch.tensor([[width, height, width, height]])
            targets.append({"boxes": boxes, "labels": torch.tensor(target["class_labels"])})
    return image_sizes, targets


def _predictions_for_metrics(prediction_batches: Any, image_sizes: list[Any], processor: Any) -> list[Any]:
    import torch

    predictions = []
    for batch, target_sizes in zip(prediction_batches, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        outputs = SimpleNamespace(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        predictions.extend(
            processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)
        )
    return predictions


def _flatten_detection_metrics(metrics: dict[str, Any], id2label: dict[int, str]) -> dict[str, float]:
    values = dict(metrics)
    classes = values.pop("classes")
    maps = values.pop("map_per_class")
    recalls = values.pop("mar_100_per_class")
    measured = {}
    for class_id, class_map, class_recall in zip(classes, maps, recalls):
        label = id2label[int(class_id.item())]
        class_map_value = float(class_map.item())
        class_recall_value = float(class_recall.item())
        if not math.isfinite(class_map_value) or not math.isfinite(class_recall_value):
            raise DetectorTrainError(f"detection metrics contain non-finite values for {label}")
        if class_map_value >= 0:
            measured[f"map_{label}"] = class_map_value
        if class_recall_value >= 0:
            measured[f"mar_100_{label}"] = class_recall_value
    aggregate = {key: float(value.item()) for key, value in values.items()}
    if any(not math.isfinite(value) for value in aggregate.values()):
        raise DetectorTrainError("detection metrics contain non-finite aggregate values")
    measured.update(aggregate)
    measured = {key: value for key, value in measured.items() if value >= 0}
    return {key: round(value, 6) for key, value in measured.items()}


def _detection_metric_options() -> dict[str, Any]:
    return {"box_format": "xyxy", "class_metrics": True, "sync_on_compute": False}


def metric_function(processor: Any, id2label: dict[int, str]):
    """Create the official threshold-zero TorchMetrics COCO evaluator."""
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    def compute(evaluation_results):
        prediction_batches, target_batches = evaluation_results.predictions, evaluation_results.label_ids
        image_sizes, targets = _targets_for_metrics(target_batches)
        predictions = _predictions_for_metrics(prediction_batches, image_sizes, processor)
        metric = MeanAveragePrecision(**_detection_metric_options())
        metric.update(predictions, targets)
        return _flatten_detection_metrics(metric.compute(), id2label)

    return compute


def _load_model(config: dict[str, Any], labels: list[str]):
    from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrImageProcessor

    model_config = config["model"]
    common = {"revision": model_config["revision"]}
    id2label = dict(enumerate(labels))
    label2id = {label: index for index, label in id2label.items()}
    processor = RTDetrImageProcessor.from_pretrained(
        model_config["name"],
        size={"height": model_config["image_size"], "width": model_config["image_size"]},
        do_pad=True,
        pad_size={"height": model_config["image_size"], "width": model_config["image_size"]},
        **common,
    )
    architecture = RTDetrConfig.from_pretrained(
        model_config["name"],
        id2label=id2label,
        label2id=label2id,
        **common,
    )
    model = RTDetrForObjectDetection.from_pretrained(
        model_config["name"],
        config=architecture,
        ignore_mismatched_sizes=True,
        **common,
    )
    return processor, model, id2label


def _load_datasets(args: argparse.Namespace, config: dict[str, Any], processor: Any):
    training = config["training"]
    train = CocoDataset(args.coco_dir / "train.json", processor, training.get("max_train_samples"))
    validation = CocoDataset(
        args.coco_dir / "validation.json",
        processor,
        training.get("max_eval_samples"),
    )
    test = CocoDataset(args.coco_dir / "test.json", processor, training.get("max_eval_samples"))
    if not train or not validation or not test:
        raise DetectorTrainError("train, validation, and test COCO splits must all be non-empty")
    return train, validation, test


def _check_labels(datasets: tuple[CocoDataset, ...], labels: list[str]) -> None:
    expected = tuple(labels)
    for dataset in datasets:
        if dataset.categories != expected:
            raise DetectorTrainError(
                f"COCO categories differ from frozen labels: {dataset.categories!r} != {expected!r}"
            )


def _check_finite(metrics: dict[str, Any], phase: str) -> None:
    numeric = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
    invalid = {key: value for key, value in numeric.items() if not math.isfinite(float(value))}
    if invalid:
        raise DetectorTrainError(f"{phase} produced non-finite metrics: {invalid}")


def _evaluate_scenarios(trainer: Any, dataset: CocoDataset) -> dict[str, Any]:
    report = {}
    for index, (name, selected) in enumerate(scenario_datasets(dataset).items()):
        metrics = trainer.evaluate(eval_dataset=selected, metric_key_prefix=f"scenario_{index:03d}")
        _check_finite(metrics, f"detector scenario {name}")
        report[name] = {"samples": len(selected), "metrics": metrics}
    return report


def _run_ml(
    args: argparse.Namespace,
    config: dict[str, Any],
    labels: list[str],
    resume: bool | str | None,
) -> None:
    from transformers import Trainer, TrainingArguments

    processor, model, id2label = _load_model(config, labels)
    datasets = _load_datasets(args, config, processor)
    _check_labels(datasets, labels)
    training_args = TrainingArguments(**trainer_arguments(config["training"], args.output_dir))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets[0],
        eval_dataset=datasets[1],
        processing_class=processor,
        data_collator=collate_fn,
        compute_metrics=metric_function(processor, id2label),
    )
    train_result = trainer.train(resume_from_checkpoint=resume)
    _check_finite(train_result.metrics, "training")
    trainer.save_model()
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    test_metrics = trainer.evaluate(eval_dataset=datasets[2], metric_key_prefix="test")
    _check_finite(test_metrics, "test evaluation")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)
    scenario_report = _evaluate_scenarios(trainer, datasets[2])
    if trainer.is_world_process_zero():
        write_provenance(args.output_dir / "test_scenario_metrics.json", scenario_report)


def main() -> None:
    args = build_parser().parse_args()
    config = load_training_config(args.config, smoke=args.smoke)
    config = apply_smoke_step_override(config, smoke=args.smoke, max_steps=args.max_steps)
    if config["task"] != "detector":
        raise DetectorTrainError(f"detector trainer received {config['task']!r} config")
    metadata = load_manifest_metadata(args.manifest_metadata)
    _validate_manifest_mode(metadata, args.smoke)
    args.output_dir = args.output_dir.resolve()
    identity = build_run_provenance(metadata, config, args.coco_dir)
    resume = prepare_resume(args.output_dir, args.resume_from_checkpoint, identity)
    labels = metadata["labels"]["detector"]
    if not isinstance(labels, list) or not labels:
        raise DetectorTrainError("frozen detector label map is empty")
    _run_ml(args, config, labels, resume)


if __name__ == "__main__":
    main()
