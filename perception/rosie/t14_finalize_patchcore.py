"""Approve a PatchCore generation only when all 24 part runs are measured.
Authored by Shritej Gunda."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from perception.anomaly.patchcore_contract import (
    MODEL_CONFIG,
    MODEL_FORMAT,
    REQUIRED_IMAGE_METRICS,
)
from perception.brev.t14_benchmark_gate import BenchmarkGateError, verify_gate

PARTS = tuple(f"M{product}_{position:02d}" for product in range(1, 7) for position in range(1, 5))
GENERATION_FIELDS = (
    "source_commit",
    "runtime_sha256",
    "dataset_preparation_runtime_sha256",
    "dataset_ready_sha256",
    "dataset_manifest_sha256",
    "dataset_artifact_sha256",
    "env_complete_sha256",
    "env_freeze_sha256",
    "env_audit_sha256",
    "env_artifact_sha256",
    "benchmark_gate_sha256",
)


class FinalizationError(ValueError):
    """Raised when per-part evidence cannot prove one complete generation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FinalizationError(f"evidence must be a real file: {path}")
    try:
        payload = json.loads(path.read_text(), parse_constant=_reject_constant)
    except (OSError, ValueError) as exc:
        raise FinalizationError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise FinalizationError(f"JSON evidence must be an object: {path}")
    return payload


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _metrics_are_measured(metrics: dict[str, Any], part: str) -> bool:
    test = metrics.get("test")
    required = REQUIRED_IMAGE_METRICS
    values = tuple(test.get(key) for key in required) if isinstance(test, dict) else ()
    finite_test = len(values) == 2 and all(
        type(value) in (int, float) and math.isfinite(value) and 0.0 <= value <= 1.0 for value in values
    )
    false_alarm = metrics.get("o_test_false_alarm_share")
    finite_false_alarm = type(false_alarm) in (int, float) and math.isfinite(false_alarm) and 0.0 <= false_alarm <= 1.0
    return metrics.get("part") == part and metrics.get("status") == "measured" and finite_test and finite_false_alarm


def _model(artifact: dict[str, Any], metrics: dict[str, Any], metrics_path: Path) -> Path:
    model_path = Path(str(artifact.get("model_path", "")))
    model_info = metrics.get("model")
    relative = model_info.get("path") if isinstance(model_info, dict) else None
    if model_path.is_symlink() or not model_path.is_file() or not isinstance(relative, str):
        raise FinalizationError("part run deployable model is missing")
    if relative != "model.ckpt":
        raise FinalizationError("part run deployable model path is invalid")
    if model_path.resolve() != (metrics_path.parent / relative).resolve():
        raise FinalizationError("part run model path disagrees with its metrics")
    exact_contract = (
        model_info.get("format") == MODEL_FORMAT
        and model_info.get("config") == MODEL_CONFIG
        and model_info.get("size_bytes") == model_path.stat().st_size
        and artifact.get("model_format") == MODEL_FORMAT
        and artifact.get("model_config") == MODEL_CONFIG
        and artifact.get("model_size_bytes") == model_path.stat().st_size
    )
    if not exact_contract:
        raise FinalizationError("part run model contract is invalid")
    if _sha256(model_path) != artifact.get("model_sha256"):
        raise FinalizationError("part run model hash is invalid")
    return model_path.resolve()


def _artifact(pointer: Path) -> tuple[Path, dict[str, Any]]:
    if pointer.is_symlink() or not pointer.is_file():
        raise FinalizationError(f"approval pointer must be a real file: {pointer}")
    lines = pointer.read_text().splitlines()
    if len(lines) != 1 or not lines[0]:
        raise FinalizationError(f"approval pointer must contain one artifact path: {pointer}")
    path = Path(lines[0])
    return path, _json(path)


def _authorization(artifact: dict[str, Any]) -> None:
    gate = Path(str(artifact.get("benchmark_gate_path", "")))
    if gate.is_symlink() or not gate.is_file():
        raise FinalizationError("part run benchmark gate is missing")
    if _sha256(gate) != artifact.get("benchmark_gate_sha256"):
        raise FinalizationError("part run benchmark gate hash is invalid")
    dataset = Path(str(artifact.get("dataset_artifact_path", "")))
    environment = Path(str(artifact.get("env_artifact_path", "")))
    try:
        verified = verify_gate(
            gate,
            dataset,
            environment,
            str(artifact.get("source_commit", "")),
            str(artifact.get("runtime_sha256", "")),
        )
    except BenchmarkGateError as exc:
        raise FinalizationError("part run benchmark gate is not authorized") from exc
    expected = (artifact.get("dataset_artifact_sha256"), artifact.get("env_artifact_sha256"))
    actual = (verified.get("dataset_artifact_sha256"), verified.get("env_artifact_sha256"))
    if actual != expected:
        raise FinalizationError("part run benchmark gate belongs to another generation")


def _record(pointer: Path) -> dict[str, Any]:
    artifact_path, artifact = _artifact(pointer)
    part = artifact.get("part")
    metrics_path = Path(str(artifact.get("metrics_path", "")))
    metrics = _json(metrics_path)
    elapsed = artifact.get("elapsed_seconds")
    valid_time = type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0
    if part not in PARTS or not _metrics_are_measured(metrics, part):
        raise FinalizationError(f"part run is not measured: {artifact_path}")
    if _sha256(metrics_path) != artifact.get("metrics_sha256") or not valid_time:
        raise FinalizationError(f"part run hash or elapsed time is invalid: {artifact_path}")
    _model(artifact, metrics, metrics_path)
    _authorization(artifact)
    valid_identity = (
        artifact.get("part_index") == PARTS.index(part)
        and artifact.get("run_kind") == "production"
        and isinstance(artifact.get("runner_id"), str)
        and bool(artifact["runner_id"])
        and isinstance(artifact.get("hostname"), str)
        and bool(artifact["hostname"])
    )
    if not valid_identity:
        raise FinalizationError(f"part run identity is incomplete: {artifact_path}")
    source_ok = re.fullmatch(r"[0-9a-f]{40}", str(artifact.get("source_commit", ""))) is not None
    hashes_ok = all(
        re.fullmatch(r"[0-9a-f]{64}", str(artifact.get(field, ""))) is not None
        for field in GENERATION_FIELDS
        if field != "source_commit"
    )
    if not source_ok or not hashes_ok:
        raise FinalizationError(f"part run lacks generation provenance: {artifact_path}")
    return {**artifact, "artifact_path": str(artifact_path), "artifact_sha256": _sha256(artifact_path)}


def finalize(pointers: list[Path], output: Path) -> dict[str, Any]:
    """Write an atomic summary after proving exact coverage and one immutable generation."""
    if output.exists():
        raise FinalizationError(f"summary output must not already exist: {output}")
    records = [_record(Path(pointer)) for pointer in pointers]
    parts = [record["part"] for record in records]
    if len(parts) != len(set(parts)) or set(parts) != set(PARTS):
        raise FinalizationError("approved runs must cover each of the 24 parts exactly once")
    generations = {tuple(record[field] for field in GENERATION_FIELDS) for record in records}
    if len(generations) != 1:
        raise FinalizationError("approved runs do not belong to one generation")
    ordered = sorted(records, key=lambda record: PARTS.index(record["part"]))
    summary = {
        **dict(zip(GENERATION_FIELDS, next(iter(generations)), strict=True)),
        "status": "approved",
        "parts": list(PARTS),
        "total_elapsed_seconds": sum(record["elapsed_seconds"] for record in ordered),
        "runs": ordered,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return summary


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pointers = sorted(args.approved_dir.glob("patchcore-*.txt"))
    print(json.dumps(finalize(pointers, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
