"""Train and evaluate the 30-class wrist-camera ResNet-18 classifier."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from perception.classifier.dataset import CropDataset
from perception.metrics import classification_report
from perception.provenance import (
    ProvenanceMismatchError,
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

PRETRAINED_MODEL = "torchvision/resnet18"
PRETRAINED_REVISION = "ResNet18_Weights.IMAGENET1K_V1"
PACKAGE_NAMES = ("torch", "torchvision", "transformers", "accelerate")


class ClassifierTrainingError(RuntimeError):
    """Raised when classifier training would violate run identity or data gates."""


def classifier_report_from_logits(
    logits: Any,
    label_ids: Any,
    rows: Sequence[dict[str, Any]] | None,
    labels: tuple[str, ...],
    lookalike_groups: tuple[tuple[str, ...], ...],
) -> dict[str, Any]:
    """Convert logits to measured exact-ID, look-alike, and scenario metrics."""
    values = np.asarray(logits)
    truth = np.asarray(label_ids).reshape(-1)
    if values.ndim != 2 or values.shape[0] != len(truth):
        raise ClassifierTrainingError(f"logit/label shape mismatch: {values.shape}/{truth.shape}")
    if rows is not None and len(rows) != len(truth):
        raise ClassifierTrainingError("scenario rows and labels must have the same length")
    predictions = values.argmax(axis=1)
    return classification_report(
        y_true=[int(value) for value in truth],
        y_pred=[int(value) for value in predictions],
        labels=labels,
        lookalike_groups=lookalike_groups,
        scenarios=rows,
        scenario_keys=("set", "variant") if rows is not None else (),
    )


def validate_classifier_identity(
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Require the user-approved 30-label order and valid look-alike groups."""
    try:
        labels = tuple(metadata["labels"]["classifier"])
        groups = tuple(tuple(group) for group in metadata["labels"]["lookalike_groups"])
    except (KeyError, TypeError) as exc:
        raise ClassifierTrainingError("manifest metadata lacks classifier labels") from exc
    if config.get("task") != "classifier" or config.get("model", {}).get("num_labels") != 30:
        raise ClassifierTrainingError("configuration must declare 30 classifier labels")
    if len(labels) != 30 or len(set(labels)) != 30:
        raise ClassifierTrainingError("manifest must contain exactly 30 classifier labels")
    unknown = {label for group in groups for label in group} - set(labels)
    if unknown:
        raise ClassifierTrainingError(f"look-alike groups contain unknown labels: {sorted(unknown)}")
    return labels, groups


def resolve_resume(
    requested: str | None,
    output_dir: Path,
    current_provenance: dict[str, Any],
) -> bool | str | None:
    """Validate run identity before asking Trainer to restore a checkpoint."""
    output_dir = output_dir.resolve()
    provenance_path = output_dir / "run_provenance.json"
    if not provenance_path.is_file():
        if requested:
            raise ClassifierTrainingError(f"cannot resume without provenance: {provenance_path}")
        write_provenance(provenance_path, current_provenance)
        return None
    try:
        assert_resume_compatible(current_provenance, read_provenance(provenance_path))
    except (OSError, ValueError, ProvenanceMismatchError) as exc:
        raise ClassifierTrainingError(str(exc)) from exc
    checkpoints = [path for path in output_dir.glob("checkpoint-*") if path.is_dir()]
    if requested is None:
        if checkpoints:
            raise ClassifierTrainingError("output contains checkpoints; pass --resume-from-checkpoint")
        return None
    if requested == "latest":
        if not checkpoints:
            raise ClassifierTrainingError(f"no checkpoint exists in {output_dir}")
        return True
    checkpoint = Path(requested).resolve()
    if not checkpoint.is_dir():
        raise ClassifierTrainingError(f"checkpoint does not exist: {checkpoint}")
    if checkpoint.parent != output_dir:
        raise ClassifierTrainingError(f"checkpoint must be inside {output_dir}: {checkpoint}")
    return str(checkpoint)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--manifest-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint", nargs="?", const="latest")
    return parser


def _validate_manifest_mode(metadata: dict[str, Any], smoke: bool) -> None:
    require_cosmos = metadata.get("require_cosmos")
    if not isinstance(require_cosmos, bool):
        raise ClassifierTrainingError("manifest metadata lacks a boolean require_cosmos gate")
    includes_cosmos = metadata.get("include_cosmos")
    if not smoke and (not require_cosmos or includes_cosmos is not True):
        raise ClassifierTrainingError("production classifier training requires a Cosmos-complete manifest")


def _check_finite_metrics(metrics: dict[str, Any], phase: str) -> None:
    stack: list[tuple[str, Any]] = [(phase, metrics)]
    invalid: dict[str, Any] = {}
    while stack:
        path, value = stack.pop()
        if isinstance(value, dict):
            stack.extend((f"{path}.{key}", item) for key, item in value.items())
        elif isinstance(value, (list, tuple)):
            stack.extend((f"{path}[{index}]", item) for index, item in enumerate(value))
        elif isinstance(value, (int, float, np.number)) and not math.isfinite(float(value)):
            invalid[path] = value
    if invalid:
        raise ClassifierTrainingError(f"{phase} produced non-finite metrics: {invalid}")


def _build_transforms(image_size: int) -> tuple[Any, Any]:
    from torchvision import transforms

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluate = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train, evaluate


def _build_model(num_labels: int) -> Any:
    from torch import nn
    from torch.nn import functional
    from torchvision.models import ResNet18_Weights, resnet18
    from transformers.modeling_outputs import ImageClassifierOutput

    class ResNetClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_labels)

        def forward(self, pixel_values=None, labels=None):
            logits = self.backbone(pixel_values)
            loss = functional.cross_entropy(logits, labels) if labels is not None else None
            return ImageClassifierOutput(loss=loss, logits=logits)

    return ResNetClassifier()


def _compute_metrics(
    labels: tuple[str, ...],
    groups: tuple[tuple[str, ...], ...],
):
    def calculate(eval_prediction):
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        report = classifier_report_from_logits(logits, eval_prediction.label_ids, None, labels, groups)
        return {"accuracy": report["accuracy"], "macro_f1": report["macro_f1"]}

    return calculate


def _datasets(
    crops_path: Path,
    config: dict[str, Any],
    labels: tuple[str, ...],
) -> tuple[CropDataset, CropDataset, CropDataset]:
    model, training = config["model"], config["training"]
    train_transform, eval_transform = _build_transforms(model["image_size"])
    train = CropDataset(crops_path, "train", train_transform, labels, training.get("max_train_samples"))
    validation = CropDataset(crops_path, "validation", eval_transform, labels, training.get("max_eval_samples"))
    test = CropDataset(crops_path, "test", eval_transform, labels, training.get("max_eval_samples"))
    counts = {"train": len(train), "validation": len(validation), "test": len(test)}
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise ClassifierTrainingError(f"empty classifier splits: {empty}")
    return train, validation, test


def _run_provenance(
    metadata: dict[str, Any],
    config: dict[str, Any],
    crops_path: Path,
) -> dict[str, Any]:
    provenance = provenance_from_manifest(
        metadata,
        task="classifier",
        pretrained_model=PRETRAINED_MODEL,
        pretrained_revision=PRETRAINED_REVISION,
    )
    provenance["packages"] = package_versions(PACKAGE_NAMES)
    provenance["trainer_code_commit"] = current_git_commit()
    provenance["training_config"] = dict(config["training"])
    provenance.update(derived_data_identity({"crops": crops_path}))
    return provenance


def _detailed_report(
    trainer: Any,
    dataset: CropDataset,
    prefix: str,
    labels: tuple[str, ...],
    groups: tuple[tuple[str, ...], ...],
) -> dict[str, Any]:
    output = trainer.predict(dataset, metric_key_prefix=prefix)
    logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    report = classifier_report_from_logits(logits, output.label_ids, dataset.rows, labels, groups)
    report["trainer_metrics"] = {
        key: float(value) if isinstance(value, (int, float, np.number)) else value
        for key, value in output.metrics.items()
    }
    return report


def _train(
    args: argparse.Namespace,
    config: dict[str, Any],
    labels: tuple[str, ...],
    groups: tuple[tuple[str, ...], ...],
    resume: bool | str | None,
) -> None:
    from transformers import Trainer, TrainingArguments

    train, validation, test = _datasets(args.crops, config, labels)
    trainer = Trainer(
        model=_build_model(len(labels)),
        args=TrainingArguments(**trainer_arguments(config["training"], args.output_dir)),
        train_dataset=train,
        eval_dataset=validation,
        compute_metrics=_compute_metrics(labels, groups),
    )
    result = trainer.train(resume_from_checkpoint=resume)
    _check_finite_metrics(result.metrics, "training")
    _check_finite_metrics({"log_history": trainer.state.log_history}, "training history")
    trainer.save_model()
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    validation_report = _detailed_report(trainer, validation, "validation_details", labels, groups)
    test_report = _detailed_report(trainer, test, "test", labels, groups)
    _check_finite_metrics(validation_report, "validation")
    _check_finite_metrics(test_report, "test")
    if trainer.is_world_process_zero():
        write_provenance(args.output_dir / "validation_classification_report.json", validation_report)
        write_provenance(args.output_dir / "test_classification_report.json", test_report)


def main(argv: Sequence[str] | None = None) -> None:
    """Validate identity, train with HF Trainer, and write final measured reports."""
    args = _parser().parse_args(argv)
    config = load_training_config(args.config, smoke=args.smoke)
    config = apply_smoke_step_override(config, smoke=args.smoke, max_steps=args.max_steps)
    metadata = load_manifest_metadata(args.manifest_metadata)
    _validate_manifest_mode(metadata, args.smoke)
    labels, groups = validate_classifier_identity(config, metadata)
    provenance = _run_provenance(metadata, config, args.crops)
    args.output_dir = args.output_dir.resolve()
    resume = resolve_resume(args.resume_from_checkpoint, args.output_dir, provenance)
    _train(args, config, labels, groups, resume)


if __name__ == "__main__":
    main()
