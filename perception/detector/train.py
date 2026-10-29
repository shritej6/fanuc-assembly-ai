"""Fine-tune RT-DETR on a frozen T13 COCO manifest.

Heavy ML dependencies are imported only after argument, configuration, and
checkpoint-provenance validation so ordinary repository tests stay lightweight.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

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
from perception.train_common import (
    configure_mixed_precision,
    load_manifest_metadata,
    seed_training,
    trainer_arguments,
)
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
    provenance["hardware_profile"] = config["hardware_profile"]
    provenance["model_config"] = dict(model)
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


def _load_converted_checkpoint(model: Any, checkpoint: str | Path) -> None:
    """Restore an RT-DETR checkpoint through its public key-conversion path."""
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_dir():
        raise DetectorTrainError(f"checkpoint does not exist: {checkpoint_path}")
    loader = getattr(type(model), "from_pretrained", None)
    if not callable(loader):
        raise DetectorTrainError(f"model cannot load converted checkpoints: {type(model).__name__}")
    try:
        restored = loader(
            str(checkpoint_path),
            local_files_only=True,
            use_safetensors=True,
        )
    except Exception as exc:
        raise DetectorTrainError(f"could not load detector checkpoint: {checkpoint_path}") from exc
    restored_state = restored.state_dict()
    expected_keys = set(model.state_dict())
    restored_keys = set(restored_state)
    if restored_keys != expected_keys:
        missing = sorted(expected_keys - restored_keys)
        unexpected = sorted(restored_keys - expected_keys)
        raise DetectorTrainError(
            f"converted checkpoint keys differ: missing={missing}, unexpected={unexpected}"
        )
    model.load_state_dict(restored_state, strict=True)


class _RTDetrCheckpointMixin:
    """Use RT-DETR's conversion-aware loader for resume and best-model restore."""

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Any = None) -> None:
        target = model if model is not None else self.model
        _load_converted_checkpoint(target, resume_from_checkpoint)

    def _load_best_model(self) -> None:
        checkpoint = self.state.best_model_checkpoint
        if not checkpoint:
            raise DetectorTrainError("best-model restore requested without a checkpoint")
        _load_converted_checkpoint(self.model, checkpoint)


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
        sizes = [_size_pair(target["orig_size"]) for target in batch]
        image_sizes.append(torch.tensor(sizes))
        for target, size in zip(batch, sizes):
            boxes = center_to_corners_format(torch.as_tensor(target["boxes"]).reshape(-1, 4))
            height, width = size
            boxes = boxes * torch.tensor([[width, height, width, height]])
            labels = torch.as_tensor(target["class_labels"]).reshape(-1)
            targets.append({"boxes": boxes, "labels": labels})
    return image_sizes, targets


def _size_pair(raw: Any) -> tuple[int, int]:
    values = np.asarray(raw).reshape(-1)
    if values.size != 2:
        raise DetectorTrainError(f"target orig_size must contain two values: {values.tolist()}")
    height, width = (int(value) for value in values)
    if height < 1 or width < 1:
        raise DetectorTrainError(f"target orig_size must be positive: {(height, width)}")
    return height, width


def _is_detection_targets(value: Any) -> bool:
    required = {"boxes", "class_labels", "orig_size"}
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(item, Mapping) and required <= item.keys() for item in value)
    )


def _gather_for_metrics(accelerator: Any, value: Any) -> Any:
    return accelerator.gather_for_metrics(
        value,
        use_gather_object=_is_detection_targets(value),
    )


def _keep_metric_outputs(outputs: tuple[Any, ...], labels: Any) -> tuple[Any, ...]:
    """Drop RT-DETR's encoder and intermediate-layer outputs before Trainer accumulates them on the GPU.

    The metric reads only indices 1 and 2 (logits, pred_boxes); the rest (per-anchor encoder logits, six decoder
    layers) grows with the eval set and ran 8xH100 out of memory at 1280 px on 3540 validation images.
    """
    return outputs[:3]


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


def _check_class_support(datasets: tuple[CocoDataset, ...], labels: list[str]) -> None:
    """Require every frozen detector category in every production split."""
    expected_ids = set(range(len(labels)))
    for split, dataset in zip(("train", "validation", "test"), datasets):
        observed = {
            int(annotation["category_id"])
            for annotations in dataset.annotations.values()
            for annotation in annotations
        }
        unknown = sorted(observed - expected_ids)
        if unknown:
            raise DetectorTrainError(f"{split} detector split has unknown category IDs: {unknown}")
        missing = [labels[index] for index in sorted(expected_ids - observed)]
        if missing:
            raise DetectorTrainError(f"{split} detector split lacks labels: {missing}")


def _check_finite(metrics: dict[str, Any], phase: str) -> None:
    numeric = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
    invalid = {key: value for key, value in numeric.items() if not math.isfinite(float(value))}
    if invalid:
        raise DetectorTrainError(f"{phase} produced non-finite metrics: {invalid}")


def _check_finite_history(history: list[dict[str, Any]]) -> None:
    for index, metrics in enumerate(history):
        _check_finite(metrics, f"training history entry {index}")


def _evaluate_scenarios(trainer: Any, dataset: CocoDataset) -> dict[str, Any]:
    report = {}
    for index, (name, selected) in enumerate(scenario_datasets(dataset).items()):
        metrics = trainer.evaluate(eval_dataset=selected, metric_key_prefix=f"scenario_{index:03d}")
        _check_finite(metrics, f"detector scenario {name}")
        report[name] = {"samples": len(selected), "metrics": metrics}
    return report


def _build_trainer(
    args: argparse.Namespace,
    config: dict[str, Any],
    processor: Any,
    model: Any,
    id2label: dict[int, str],
    datasets: tuple[CocoDataset, CocoDataset, CocoDataset],
) -> Any:
    from transformers import Trainer, TrainingArguments

    class DetectionTrainer(_RTDetrCheckpointMixin, Trainer):
        def _gather_detection_metrics(self, value: Any) -> Any:
            return _gather_for_metrics(self.accelerator, value)

        def evaluation_loop(self, *loop_args: Any, **loop_kwargs: Any):
            self.gather_function = self._gather_detection_metrics
            try:
                return super().evaluation_loop(*loop_args, **loop_kwargs)
            finally:
                self.gather_function = self._gather_detection_metrics

    training_args = TrainingArguments(**trainer_arguments(config["training"], args.output_dir))
    return DetectionTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets[0],
        eval_dataset=datasets[1],
        processing_class=processor,
        data_collator=collate_fn,
        compute_metrics=metric_function(processor, id2label),
        preprocess_logits_for_metrics=_keep_metric_outputs,
    )


def _run_ml(
    args: argparse.Namespace,
    config: dict[str, Any],
    labels: list[str],
    resume: bool | str | None,
) -> None:
    seed_training(config["training"]["seed"])
    processor, model, id2label = _load_model(config, labels)
    datasets = _load_datasets(args, config, processor)
    _check_labels(datasets, labels)
    if not args.smoke:
        _check_class_support(datasets, labels)
    trainer = _build_trainer(args, config, processor, model, id2label, datasets)
    configure_mixed_precision(trainer, config["training"], DetectorTrainError)
    train_result = trainer.train(resume_from_checkpoint=resume)
    _check_finite(train_result.metrics, "training")
    _check_finite_history(trainer.state.log_history)
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
