from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from perception.brev.t14_benchmark_gate import create_gate
from perception.rosie.t14_finalize_patchcore import FinalizationError, finalize

MODEL_CONFIG = {
    "backbone": "wide_resnet50_2",
    "layers": ["layer2", "layer3"],
    "coreset_sampling_ratio": 0.1,
    "num_neighbors": 9,
}
MODEL_BYTES = b"deployable-model"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parts() -> list[str]:
    return [f"M{product}_{position:02d}" for product in range(1, 7) for position in range(1, 5)]


def _model_evidence(root: Path, part: str) -> tuple[Path, Path]:
    metrics = root / "metrics" / part / "metrics.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    model = metrics.parent / "model.ckpt"
    model.write_bytes(MODEL_BYTES)
    payload = {
        "part": part,
        "status": "measured",
        "test": {"image_AUROC": 0.75, "image_F1Score": 0.5},
        "o_test_false_alarm_share": 0.0,
        "model": {
            "path": "model.ckpt",
            "format": "anomalib-lightning-ckpt",
            "size_bytes": len(MODEL_BYTES),
            "config": MODEL_CONFIG,
        },
    }
    metrics.write_text(json.dumps(payload))
    return metrics, model


def _benchmark_inputs(root: Path) -> tuple[Path, Path]:
    dataset = root / "dataset-artifact.json"
    dataset.write_text("{}")
    platform = root / "platform.json"
    platform.write_text(json.dumps({"brev_instance_id": "test-id", "brev_machine_type": "test-machine"}))
    environment = root / "env-artifact.json"
    environment.write_text(json.dumps({"platform_identity": str(platform), "platform_identity_sha256": _sha(platform)}))
    return dataset, environment


def _create_gate(root: Path, dataset: Path, environment: Path) -> Path:
    metrics, model = root / "benchmark-metrics.json", root / "benchmark-model.ckpt"
    metrics.write_text("{}")
    model.write_bytes(b"benchmark-model")
    benchmark = root / "benchmark-artifact.json"
    benchmark.write_text(
        json.dumps(
            {
                "part_index": 0,
                "source_commit": "a" * 40,
                "runtime_sha256": "8" * 64,
                "dataset_artifact_sha256": _sha(dataset),
                "env_artifact_sha256": _sha(environment),
                "elapsed_seconds": 10.0,
                "metrics_path": str(metrics),
                "metrics_sha256": _sha(metrics),
                "model_path": str(model),
                "model_sha256": _sha(model),
            }
        )
    )
    rate = root / "rate.json"
    rate.write_text(
        json.dumps(
            {
                "source_command": "brev search",
                "captured_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "hourly_rate_usd": 1.0,
                "brev_instance_id": "test-id",
                "brev_machine_type": "test-machine",
            }
        )
    )
    gate = root / "benchmark-gate.json"
    if not gate.exists():
        create_gate(benchmark, dataset, environment, "a" * 40, "8" * 64, 1.0, rate, gate)
    return gate


def _generation(root: Path) -> dict[str, object]:
    dataset, environment = (
        _benchmark_inputs(root)
        if not (root / "dataset-artifact.json").exists()
        else (
            root / "dataset-artifact.json",
            root / "env-artifact.json",
        )
    )
    gate = (
        _create_gate(root, dataset, environment)
        if not (root / "benchmark-gate.json").exists()
        else (root / "benchmark-gate.json")
    )
    return {
        "source_commit": "a" * 40,
        "dataset_ready_sha256": "1" * 64,
        "dataset_manifest_sha256": "2" * 64,
        "dataset_artifact_path": str(dataset),
        "dataset_artifact_sha256": _sha(dataset),
        "env_complete_sha256": "4" * 64,
        "env_freeze_sha256": "5" * 64,
        "env_audit_sha256": "6" * 64,
        "env_artifact_path": str(environment),
        "env_artifact_sha256": _sha(environment),
        "runtime_sha256": "8" * 64,
        "dataset_preparation_runtime_sha256": "9" * 64,
        "benchmark_gate_sha256": _sha(gate),
        "benchmark_gate_path": str(gate),
    }


def _write_artifact(root: Path, part: str) -> Path:
    metrics, model = _model_evidence(root, part)
    artifact = root / "artifacts" / f"patchcore_run-{part}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **_generation(root),
        "part": part,
        "part_index": _parts().index(part),
        "metrics_path": str(metrics),
        "metrics_sha256": _sha(metrics),
        "model_path": str(model),
        "model_sha256": _sha(model),
        "model_format": "anomalib-lightning-ckpt",
        "model_size_bytes": len(MODEL_BYTES),
        "model_config": MODEL_CONFIG,
        "elapsed_seconds": 1.25,
        "runner_id": "123",
        "hostname": "node",
        "run_kind": "production",
    }
    artifact.write_text(json.dumps(payload))
    pointer = root / "approved" / f"patchcore-{part}.txt"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(f"{artifact}\n")
    return pointer


def _evidence(root: Path) -> list[Path]:
    return [_write_artifact(root, part) for part in _parts()]


def _last_payload(pointers: list[Path]) -> tuple[Path, dict]:
    artifact = Path(pointers[-1].read_text().strip())
    return artifact, json.loads(artifact.read_text())


def test_finalize_requires_exactly_24_matching_measured_parts(tmp_path: Path) -> None:
    summary = finalize(_evidence(tmp_path), tmp_path / "summary.json")

    assert summary["status"] == "approved"
    assert summary["parts"] == _parts()
    assert summary["total_elapsed_seconds"] == 30.0


def test_finalize_rejects_stale_source(tmp_path: Path) -> None:
    pointers = _evidence(tmp_path)
    artifact, payload = _last_payload(pointers)
    payload["source_commit"] = "b" * 40
    artifact.write_text(json.dumps(payload))

    with pytest.raises(FinalizationError, match="benchmark|generation"):
        finalize(pointers, tmp_path / "summary.json")


def test_finalize_rejects_mixed_environment(tmp_path: Path) -> None:
    pointers = _evidence(tmp_path)
    artifact, payload = _last_payload(pointers)
    payload["env_artifact_sha256"] = "b" * 64
    artifact.write_text(json.dumps(payload))

    with pytest.raises(FinalizationError, match="generation"):
        finalize(pointers, tmp_path / "summary.json")


def test_finalize_rejects_mixed_benchmark_authorization(tmp_path: Path) -> None:
    pointers = _evidence(tmp_path)
    artifact, payload = _last_payload(pointers)
    alternate = tmp_path / "alternate-benchmark-gate.json"
    alternate.write_text('{"status":"different-benchmark"}\n')
    payload.update(benchmark_gate_path=str(alternate), benchmark_gate_sha256=_sha(alternate))
    artifact.write_text(json.dumps(payload))

    with pytest.raises(FinalizationError, match="benchmark"):
        finalize(pointers, tmp_path / "summary.json")


def test_finalize_rejects_changed_benchmark_gate(tmp_path: Path) -> None:
    pointers = _evidence(tmp_path)
    _, payload = _last_payload(pointers)
    Path(payload["benchmark_gate_path"]).write_text("changed\n")

    with pytest.raises(FinalizationError, match="benchmark gate hash"):
        finalize(pointers, tmp_path / "summary.json")


def test_finalize_rejects_incomplete_identity(tmp_path: Path) -> None:
    pointers = _evidence(tmp_path)
    artifact, payload = _last_payload(pointers)
    del payload["runner_id"]
    artifact.write_text(json.dumps(payload))

    with pytest.raises(FinalizationError, match="run identity"):
        finalize(pointers, tmp_path / "summary.json")


def test_finalize_rejects_changed_model(tmp_path: Path) -> None:
    pointers = _evidence(tmp_path)
    _, payload = _last_payload(pointers)
    Path(payload["model_path"]).write_bytes(b"changed")

    with pytest.raises(FinalizationError, match="model"):
        finalize(pointers, tmp_path / "summary.json")
