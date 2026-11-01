"""Benchmark RT-DETR capacity and throughput on real frozen render data.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any

from perception.benchmark import (
    SynchronizedStepTimer,
    benchmark_training_config,
    cpu_affinity_count,
    cuda_timing_callback,
    ensure_empty_output,
    projection_report,
    validate_benchmark_shape,
)
from perception.detector.train import (
    DetectorTrainError,
    _build_trainer,
    _check_finite,
    _check_finite_history,
    _check_labels,
    _load_datasets,
    _load_model,
    _validate_manifest_mode,
    build_run_provenance,
)
from perception.provenance import write_provenance
from perception.train_common import configure_mixed_precision, load_manifest_metadata, seed_training
from perception.training_config import load_training_config


def benchmark_config(
    config: dict[str, Any],
    *,
    batch_size: int,
    workers: int,
    steps: int,
) -> dict[str, Any]:
    """Create a bounded, no-eval overlay without changing production settings."""
    return benchmark_training_config(
        config,
        batch_size=batch_size,
        workers=workers,
        steps=steps,
        max_batch_size=256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--coco-dir", type=Path, required=True)
    parser.add_argument("--manifest-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--project-train-samples", type=int)
    parser.add_argument("--worst-case-annotations", action="store_true")
    return parser


def _repeat_worst_annotation_batch(dataset: Any, batch_size: int, steps: int) -> None:
    ranked = sorted(
        dataset.images,
        key=lambda image: len(dataset.annotations.get(int(image["id"]), ())),
        reverse=True,
    )
    if len(ranked) < batch_size:
        raise DetectorTrainError("worst-case benchmark has fewer images than the requested batch")
    dataset.images = ranked[:batch_size] * steps


def _prepare(args: argparse.Namespace) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    validate_benchmark_shape(args.steps, args.warmup_steps)
    base_config = load_training_config(args.config)
    if base_config["task"] != "detector":
        raise DetectorTrainError("detector benchmark requires a detector config")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise DetectorTrainError("detector benchmark requires exactly one GPU process")
    if args.project_train_samples is not None and args.project_train_samples < 1:
        raise DetectorTrainError("projected train samples must be positive")
    config = benchmark_config(
        base_config,
        batch_size=args.batch_size,
        workers=args.workers,
        steps=args.steps,
    )
    metadata = load_manifest_metadata(args.manifest_metadata)
    _validate_manifest_mode(metadata, smoke=True)
    labels = metadata.get("labels", {}).get("detector")
    if not isinstance(labels, list) or not labels:
        raise DetectorTrainError("frozen detector label map is empty")
    args.output_dir = ensure_empty_output(args.output_dir, DetectorTrainError)
    identity = build_run_provenance(metadata, config, args.coco_dir)
    write_provenance(args.output_dir / "run_provenance.json", identity)
    return config, labels, identity


def _train(
    args: argparse.Namespace,
    config: dict[str, Any],
    labels: list[str],
) -> tuple[Any, SynchronizedStepTimer, dict[str, Any], dict[str, int]]:
    import torch

    seed_training(config["training"]["seed"])
    processor, model, id2label = _load_model(config, labels)
    datasets = _load_datasets(args, config, processor)
    _check_labels(datasets, labels)
    if args.worst_case_annotations:
        _repeat_worst_annotation_batch(datasets[0], args.batch_size, args.steps)
    trainer = _build_trainer(args, config, processor, model, id2label, datasets)
    configure_mixed_precision(trainer, config["training"], DetectorTrainError)
    timer = SynchronizedStepTimer(args.warmup_steps, torch.cuda.synchronize, time.perf_counter)
    callback, memory = cuda_timing_callback(timer)
    trainer.add_callback(callback)
    result = trainer.train()
    _check_finite(result.metrics, "benchmark training")
    _check_finite_history(trainer.state.log_history)
    sample_counts = dict(zip(("train", "validation", "test"), map(len, datasets)))
    return result, timer, memory, sample_counts


def _projection(
    args: argparse.Namespace,
    config: dict[str, Any],
    step_seconds: float,
) -> dict[str, Any] | None:
    if args.project_train_samples is None:
        return None
    return projection_report(args.project_train_samples, config, step_seconds)


def _report(
    args: argparse.Namespace,
    config: dict[str, Any],
    identity: dict[str, Any],
    result: Any,
    timer: SynchronizedStepTimer,
    memory: dict[str, Any],
    sample_counts: dict[str, int],
) -> dict[str, Any]:
    timing = timer.summary()
    batch_size = config["training"]["per_device_train_batch_size"]
    accumulation = config["training"]["gradient_accumulation_steps"]
    timing["images_per_second"] = (
        batch_size * accumulation / timing["median_seconds_per_optimizer_step"]
    )
    report = {
        "schema_version": 1,
        "hardware_profile": config["hardware_profile"],
        "image_size": config["model"]["image_size"],
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "dataloader_workers": config["training"]["dataloader_num_workers"],
        "cpu_affinity_count": cpu_affinity_count(),
        "warmup_steps": args.warmup_steps,
        "worst_case_annotations": args.worst_case_annotations,
        "benchmark_dataset_samples": sample_counts,
        "timing": timing,
        "memory": memory,
        "finite_train_loss": math.isfinite(float(result.metrics["train_loss"])),
        "provenance": identity,
    }
    projection = _projection(args, config, timing["median_seconds_per_optimizer_step"])
    if projection is not None:
        report["projection"] = projection
    return report


def _run(args: argparse.Namespace) -> dict[str, Any]:
    config, labels, identity = _prepare(args)
    result, timer, memory, sample_counts = _train(args, config, labels)
    return _report(args, config, identity, result, timer, memory, sample_counts)


def main() -> None:
    """Run the benchmark and print its report."""
    args = _parser().parse_args()
    report = _run(args)
    write_provenance(args.output_dir / "benchmark.json", report)
    print((args.output_dir / "benchmark.json").read_text(), end="")


if __name__ == "__main__":
    main()
