"""Timing and projection helpers for guarded training benchmarks.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import copy
import math
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class BenchmarkError(ValueError):
    """Raised when a benchmark cannot produce a defensible measurement."""


def benchmark_training_config(
    config: dict[str, Any],
    *,
    batch_size: int,
    workers: int,
    steps: int,
    max_batch_size: int,
) -> dict[str, Any]:
    """Create a bounded no-eval overlay without mutating production settings."""
    if type(batch_size) is not int or not 1 <= batch_size <= max_batch_size:
        raise BenchmarkError(f"benchmark batch size must be in [1, {max_batch_size}]")
    if type(workers) is not int or not 0 <= workers <= 208:
        raise BenchmarkError("benchmark workers must be in [0, 208]")
    updated = copy.deepcopy(config)
    updated["training"].update(
        {
            "max_steps": steps,
            "per_device_train_batch_size": batch_size,
            "dataloader_num_workers": workers,
            "eval_strategy": "no",
            "save_strategy": "no",
            "logging_strategy": "no",
            "load_best_model_at_end": False,
        }
    )
    return updated


def validate_benchmark_shape(steps: int, warmup_steps: int) -> None:
    """Bound a capacity/throughput probe and require enough timed samples."""
    if type(steps) is not int or not 3 <= steps <= 200:
        raise BenchmarkError("benchmark steps must be in [3, 200]")
    if type(warmup_steps) is not int or warmup_steps < 2:
        raise BenchmarkError("benchmark warmup steps must be at least 2")
    if steps - warmup_steps < 5:
        raise BenchmarkError("benchmark must retain at least 5 measured steps")


def optimizer_steps(
    train_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    world_size: int = 1,
) -> int:
    """Match Transformers 5.21's epoch and accumulation rounding."""
    values = (train_samples, batch_size, gradient_accumulation_steps, epochs, world_size)
    if any(type(value) is not int or value < 1 for value in values):
        raise BenchmarkError("optimizer-step inputs must be positive integers")
    dataloader_batches = math.ceil(train_samples / (batch_size * world_size))
    updates_per_epoch = math.ceil(dataloader_batches / gradient_accumulation_steps)
    return epochs * updates_per_epoch


def projected_train_loop_hours(step_seconds: float, steps: int) -> float:
    """Project only the optimizer loop; evaluation and checkpoint time are separate."""
    if not math.isfinite(step_seconds) or step_seconds <= 0:
        raise BenchmarkError("step seconds must be finite and positive")
    if type(steps) is not int or steps < 1:
        raise BenchmarkError("projected optimizer steps must be a positive integer")
    return step_seconds * steps / 3600


def projection_report(
    train_samples: int,
    config: dict[str, Any],
    step_seconds: float,
) -> dict[str, Any]:
    """Bind an exact train count and production schedule to a timing result."""
    training = config["training"]
    steps = optimizer_steps(
        train_samples,
        training["per_device_train_batch_size"],
        training["gradient_accumulation_steps"],
        training["num_train_epochs"],
    )
    return {
        "train_samples": train_samples,
        "epochs": training["num_train_epochs"],
        "optimizer_steps": steps,
        "train_loop_hours": projected_train_loop_hours(step_seconds, steps),
        "excludes": ["evaluation", "checkpoint_writes", "final_test", "scenario_reports"],
    }


def cpu_affinity_count() -> int:
    """Return schedulable logical CPUs rather than the host-wide CPU count."""
    affinity = getattr(os, "sched_getaffinity", None)
    return len(affinity(0)) if affinity is not None else int(os.cpu_count() or 1)


def ensure_empty_output(path: Path, error_type: type[Exception]) -> Path:
    """Refuse to merge measurements from different benchmark processes."""
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise error_type(f"benchmark output is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


@dataclass
class SynchronizedStepTimer:
    """Record compute and end-to-end optimizer-step cadence after warm-up."""

    warmup_steps: int
    synchronize: Callable[[], None]
    clock: Callable[[], float]
    compute_seconds: list[float] = field(default_factory=list)
    end_to_end_seconds: list[float] = field(default_factory=list)
    _step_started: float | None = None
    _previous_step_ended: float | None = None

    def step_begin(self) -> None:
        """Mark the start of one optimizer step."""
        self.synchronize()
        self._step_started = self.clock()

    def step_end(self, completed_step: int) -> None:
        """Record the timings of one finished optimizer step."""
        if self._step_started is None:
            raise BenchmarkError("step_end called without step_begin")
        self.synchronize()
        ended = self.clock()
        if completed_step > self.warmup_steps:
            self.compute_seconds.append(ended - self._step_started)
            if self._previous_step_ended is not None:
                self.end_to_end_seconds.append(ended - self._previous_step_ended)
        self._previous_step_ended = ended
        self._step_started = None

    def summary(self) -> dict[str, float | int]:
        """Return the median step timings, or raise when too few were measured."""
        values = self.end_to_end_seconds
        if len(values) < 5 or any(not math.isfinite(value) or value <= 0 for value in values):
            raise BenchmarkError("benchmark requires at least 5 finite positive step timings")
        return {
            "measured_steps": len(values),
            "median_seconds_per_optimizer_step": statistics.median(values),
            "min_seconds_per_optimizer_step": min(values),
            "max_seconds_per_optimizer_step": max(values),
            "median_compute_seconds": statistics.median(self.compute_seconds),
        }


def cuda_timing_callback(timer: SynchronizedStepTimer) -> tuple[Any, dict[str, Any]]:
    """Build a Trainer callback that captures synchronized timing and CUDA peaks."""
    import torch
    from transformers import TrainerCallback
    memory: dict[str, Any] = {
        "gpu_name": torch.cuda.get_device_name(),
        "total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "minimum_free_at_step_end_bytes": None,
    }

    class TimingCallback(TrainerCallback):
        """Trainer callback that feeds the step timer and records CUDA memory peaks."""
        def on_train_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            return control

        def on_step_begin(self, args, state, control, **kwargs):
            timer.step_begin()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            completed_step = int(state.global_step)
            timer.step_end(completed_step)
            if completed_step == timer.warmup_steps:
                torch.cuda.reset_peak_memory_stats()
            elif completed_step > timer.warmup_steps:
                free_bytes, _ = torch.cuda.mem_get_info()
                current = memory["minimum_free_at_step_end_bytes"]
                memory["minimum_free_at_step_end_bytes"] = (
                    free_bytes if current is None else min(current, free_bytes)
                )
            return control

        def on_train_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            memory["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            memory["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            return control

    return TimingCallback(), memory
