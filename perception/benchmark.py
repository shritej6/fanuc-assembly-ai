"""Dependency-light timing and projection helpers for guarded ML benchmarks."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field


class BenchmarkError(ValueError):
    """Raised when a benchmark cannot produce a defensible measurement."""


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
        self.synchronize()
        self._step_started = self.clock()

    def step_end(self, completed_step: int) -> None:
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
