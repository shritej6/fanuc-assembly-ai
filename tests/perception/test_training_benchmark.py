import sys
from types import SimpleNamespace

import pytest

from perception.benchmark import (
    BenchmarkError,
    SynchronizedStepTimer,
    cuda_timing_callback,
    optimizer_steps,
    projected_train_loop_hours,
    validate_benchmark_shape,
)
from perception.classifier.benchmark import _repeat_training_rows
from perception.classifier.benchmark import benchmark_config as classifier_benchmark_config
from perception.detector.benchmark import (
    _repeat_worst_annotation_batch,
    benchmark_config,
)


def test_optimizer_step_projection_matches_transformers_rounding() -> None:
    assert optimizer_steps(65, batch_size=8, gradient_accumulation_steps=2, epochs=20) == 100
    assert projected_train_loop_hours(0.5, 100) == pytest.approx(50 / 3600)


def test_synchronized_timer_discards_warmup_and_reports_step_cadence() -> None:
    moments = iter(float(value) for value in range(16))
    sync_calls = []
    timer = SynchronizedStepTimer(
        warmup_steps=2,
        synchronize=lambda: sync_calls.append(True),
        clock=lambda: next(moments),
    )

    for step in range(1, 9):
        timer.step_begin()
        timer.step_end(step)

    summary = timer.summary()

    assert len(sync_calls) == 16
    assert summary["measured_steps"] == 6
    assert summary["median_seconds_per_optimizer_step"] == 2.0
    assert summary["median_compute_seconds"] == 1.0


@pytest.mark.parametrize(("steps", "warmup"), [(6, 2), (20, 1), (201, 2)])
def test_benchmark_shape_rejects_short_or_unbounded_runs(steps: int, warmup: int) -> None:
    with pytest.raises(BenchmarkError):
        validate_benchmark_shape(steps, warmup)


def test_benchmark_overlay_is_bounded_and_does_not_mutate_production() -> None:
    production = {
        "task": "detector",
        "hardware_profile": "h100",
        "model": {"image_size": 1280},
        "training": {
            "num_train_epochs": 20,
            "max_steps": -1,
            "per_device_train_batch_size": 2,
            "dataloader_num_workers": 32,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_strategy": "steps",
            "load_best_model_at_end": True,
        },
    }

    benchmark = benchmark_config(production, batch_size=16, workers=52, steps=8)

    assert production["training"]["max_steps"] == -1
    assert benchmark["training"]["num_train_epochs"] == 20
    assert benchmark["training"]["max_steps"] == 8
    assert benchmark["training"]["per_device_train_batch_size"] == 16
    assert benchmark["training"]["dataloader_num_workers"] == 52
    assert benchmark["training"]["eval_strategy"] == "no"
    assert benchmark["training"]["save_strategy"] == "no"
    assert benchmark["training"]["load_best_model_at_end"] is False


def test_capacity_probe_repeats_the_highest_annotation_batch() -> None:
    dataset = SimpleNamespace(
        images=[{"id": 1}, {"id": 2}, {"id": 3}],
        annotations={1: [{}], 2: [{}, {}, {}], 3: [{}, {}]},
    )

    _repeat_worst_annotation_batch(dataset, batch_size=2, steps=3)

    assert [image["id"] for image in dataset.images] == [2, 3, 2, 3, 2, 3]


def test_cuda_callback_resets_warmup_peaks_and_serializes_memory(monkeypatch) -> None:
    calls = []

    class FakeCuda:
        @staticmethod
        def get_device_name() -> str:
            return "H100-test"

        @staticmethod
        def get_device_properties(index: int):
            return SimpleNamespace(total_memory=80)

        @staticmethod
        def synchronize() -> None:
            calls.append("sync")

        @staticmethod
        def reset_peak_memory_stats() -> None:
            calls.append("reset")

        @staticmethod
        def mem_get_info() -> tuple[int, int]:
            return 20, 80

        @staticmethod
        def max_memory_allocated() -> int:
            return 50

        @staticmethod
        def max_memory_reserved() -> int:
            return 60

    class FakeTimer:
        warmup_steps = 2

        def step_begin(self) -> None:
            calls.append("begin")

        def step_end(self, step: int) -> None:
            calls.append(("end", step))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda()))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(TrainerCallback=object))
    callback, memory = cuda_timing_callback(FakeTimer())
    control = object()

    callback.on_train_begin(None, None, control)
    callback.on_step_begin(None, None, control)
    callback.on_step_end(None, SimpleNamespace(global_step=2), control)
    callback.on_step_end(None, SimpleNamespace(global_step=3), control)
    callback.on_train_end(None, None, control)

    assert calls.count("reset") == 2
    assert memory == {
        "gpu_name": "H100-test",
        "total_bytes": 80,
        "minimum_free_at_step_end_bytes": 20,
        "peak_allocated_bytes": 50,
        "peak_reserved_bytes": 60,
    }


def test_classifier_overlay_preserves_epochs_and_repeats_frozen_rows() -> None:
    production = {
        "task": "classifier",
        "hardware_profile": "h100",
        "model": {"image_size": 224},
        "training": {
            "num_train_epochs": 15,
            "max_steps": -1,
            "per_device_train_batch_size": 64,
            "dataloader_num_workers": 16,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_strategy": "steps",
            "load_best_model_at_end": True,
        },
    }
    dataset = SimpleNamespace(rows=[{"id": 1}, {"id": 2}, {"id": 3}])

    benchmark = classifier_benchmark_config(production, batch_size=4, workers=8, steps=3)
    _repeat_training_rows(dataset, batch_size=4, steps=3)

    assert production["training"]["max_steps"] == -1
    assert benchmark["training"]["num_train_epochs"] == 15
    assert benchmark["training"]["per_device_train_batch_size"] == 4
    assert [row["id"] for row in dataset.rows] == [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
