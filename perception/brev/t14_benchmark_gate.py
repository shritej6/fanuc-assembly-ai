"""Create and verify the measured cost gate before Brev T14 production runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

COST_CAP_USD = 120.0
RATE_EVIDENCE_MAX_AGE = timedelta(minutes=15)
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class BenchmarkGateError(ValueError):
    """Raised when a benchmark cannot authorize the matching production generation."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BenchmarkGateError(f"evidence must be a real file: {path}")
    try:
        payload = json.loads(path.read_text(), parse_constant=_reject_constant)
    except (OSError, ValueError) as exc:
        raise BenchmarkGateError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkGateError(f"JSON evidence must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value) is None:
        raise BenchmarkGateError(f"{label} must be an exact UTC timestamp")
    try:
        return datetime.strptime(value, UTC_FORMAT).replace(tzinfo=UTC)
    except ValueError as exc:
        raise BenchmarkGateError(f"{label} must be an exact UTC timestamp") from exc


def _rate_timestamp(captured_at: Any, reference_time: datetime) -> None:
    captured = _utc_timestamp(captured_at, "hourly-rate captured_at_utc")
    age = reference_time - captured
    if age < timedelta(0) or age > RATE_EVIDENCE_MAX_AGE:
        raise BenchmarkGateError("hourly-rate evidence is stale or future-dated")


def _identifiers(source_commit: str, runtime_sha256: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise BenchmarkGateError("source commit must be a full lowercase SHA-1")
    if re.fullmatch(r"[0-9a-f]{64}", runtime_sha256) is None:
        raise BenchmarkGateError("runtime identity must be a lowercase SHA-256")


def _identity(
    artifact: dict[str, Any], source_commit: str, runtime_sha256: str, dataset: Path, environment: Path
) -> None:
    valid = (
        artifact.get("part_index") == 0
        and artifact.get("source_commit") == source_commit
        and artifact.get("runtime_sha256") == runtime_sha256
        and artifact.get("dataset_artifact_sha256") == _sha256(dataset)
        and artifact.get("env_artifact_sha256") == _sha256(environment)
        and _finite_positive(artifact.get("elapsed_seconds"))
    )
    if not valid:
        raise BenchmarkGateError("benchmark does not match the requested immutable generation")


def _artifact_files(artifact: dict[str, Any]) -> None:
    for path_key, hash_key in (("metrics_path", "metrics_sha256"), ("model_path", "model_sha256")):
        path = Path(str(artifact.get(path_key, "")))
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise BenchmarkGateError(f"benchmark {path_key} is missing or changed")
        if _sha256(path) != artifact.get(hash_key):
            raise BenchmarkGateError(f"benchmark {path_key} is missing or changed")


def _projection(elapsed_seconds: float, hourly_rate_usd: float) -> tuple[float, float, float]:
    projected_hours = elapsed_seconds * 24 / 3600
    return elapsed_seconds, projected_hours, projected_hours * hourly_rate_usd


def _destination(output: Path, hourly_rate_usd: float) -> Path:
    if output.exists() or output.is_symlink() or not _finite_positive(hourly_rate_usd):
        raise BenchmarkGateError("gate output must be fresh and hourly rate must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise BenchmarkGateError(f"temporary gate output must be fresh: {temporary}")
    return temporary


def _rate_evidence(
    path: Path,
    environment: Path,
    hourly_rate_usd: float,
    reference_time: datetime,
) -> dict[str, Any]:
    evidence = _json(path)
    env = _json(environment)
    platform_path = Path(str(env.get("platform_identity", "")))
    platform = _json(platform_path)
    valid = (
        evidence.get("source_command") == "brev search"
        and evidence.get("hourly_rate_usd") == hourly_rate_usd
        and evidence.get("brev_instance_id") == platform.get("brev_instance_id")
        and evidence.get("brev_machine_type") == platform.get("brev_machine_type")
        and _sha256(platform_path) == env.get("platform_identity_sha256")
    )
    if not valid:
        raise BenchmarkGateError("hourly-rate evidence does not match this Brev runtime")
    _rate_timestamp(evidence.get("captured_at_utc"), reference_time)
    return evidence


def _budget_deadline(created_at: datetime, benchmark_seconds: float, hourly_rate_usd: float) -> tuple[float, datetime]:
    budget_seconds = COST_CAP_USD * 3600 / hourly_rate_usd - benchmark_seconds
    if not _finite_positive(budget_seconds):
        raise BenchmarkGateError(f"benchmark already exhausts the USD {COST_CAP_USD:.2f} cap")
    return budget_seconds, created_at + timedelta(seconds=budget_seconds)


def _gate_payload(
    benchmark: Path,
    dataset: Path,
    environment: Path,
    source_commit: str,
    runtime_sha256: str,
    hourly_rate_usd: float,
    hourly_rate_evidence: Path,
    rate_evidence: dict[str, Any],
    created_at: datetime,
    projection: tuple[float, float, float],
) -> dict[str, Any]:
    seconds_per_part, projected_hours, projected_cost = projection
    budget_seconds, deadline = _budget_deadline(created_at, seconds_per_part, hourly_rate_usd)
    return {
        "status": "benchmark_measured",
        "source_commit": source_commit,
        "runtime_sha256": runtime_sha256,
        "dataset_artifact_sha256": _sha256(dataset),
        "env_artifact_sha256": _sha256(environment),
        "benchmark_artifact": str(benchmark.resolve()),
        "benchmark_artifact_sha256": _sha256(benchmark),
        "seconds_per_part": seconds_per_part,
        "projected_parts": 24,
        "projected_hours": projected_hours,
        "cost_cap_usd": COST_CAP_USD,
        "hourly_rate_usd": hourly_rate_usd,
        "hourly_rate_source": rate_evidence["source_command"],
        "hourly_rate_evidence": str(hourly_rate_evidence.resolve()),
        "hourly_rate_evidence_sha256": _sha256(hourly_rate_evidence),
        "projected_cost_usd": projected_cost,
        "gate_created_at_utc": created_at.strftime(UTC_FORMAT),
        "budget_seconds_remaining_at_gate": budget_seconds,
        "budget_deadline_utc": deadline.strftime(UTC_FORMAT),
    }


def create_gate(
    benchmark: Path,
    dataset: Path,
    environment: Path,
    source_commit: str,
    runtime_sha256: str,
    hourly_rate_usd: float,
    hourly_rate_evidence: Path,
    output: Path,
) -> dict[str, Any]:
    _identifiers(source_commit, runtime_sha256)
    temporary = _destination(output, hourly_rate_usd)
    created_at = _utc_now()
    rate_evidence = _rate_evidence(hourly_rate_evidence, environment, hourly_rate_usd, created_at)
    artifact = _json(benchmark)
    _identity(artifact, source_commit, runtime_sha256, dataset, environment)
    _artifact_files(artifact)
    projection = _projection(artifact["elapsed_seconds"], hourly_rate_usd)
    if projection[2] > COST_CAP_USD:
        raise BenchmarkGateError(f"projected cost exceeds the USD {COST_CAP_USD:.2f} cap")
    payload = _gate_payload(
        benchmark,
        dataset,
        environment,
        source_commit,
        runtime_sha256,
        hourly_rate_usd,
        hourly_rate_evidence,
        rate_evidence,
        created_at,
        projection,
    )
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)
    return payload


def _verify_gate_identity(
    gate: dict[str, Any],
    benchmark_path: Path,
    dataset: Path,
    environment: Path,
    source_commit: str,
    runtime_sha256: str,
) -> None:
    expected = {
        "status": "benchmark_measured",
        "source_commit": source_commit,
        "runtime_sha256": runtime_sha256,
        "dataset_artifact_sha256": _sha256(dataset),
        "env_artifact_sha256": _sha256(environment),
        "benchmark_artifact_sha256": _sha256(benchmark_path),
        "projected_parts": 24,
        "cost_cap_usd": COST_CAP_USD,
    }
    if any(gate.get(key) != value for key, value in expected.items()):
        raise BenchmarkGateError("benchmark gate identity changed or belongs to another generation")


def _verify_rate(gate: dict[str, Any], environment: Path, created_at: datetime) -> float:
    rate = gate.get("hourly_rate_usd")
    if not _finite_positive(rate):
        raise BenchmarkGateError("benchmark gate lacks a finite positive hourly_rate_usd")
    evidence_path = Path(str(gate.get("hourly_rate_evidence", "")))
    _rate_evidence(evidence_path, environment, rate, created_at)
    if gate.get("hourly_rate_evidence_sha256") != _sha256(evidence_path):
        raise BenchmarkGateError("benchmark gate hourly-rate evidence changed")
    if gate.get("hourly_rate_source") != "brev search":
        raise BenchmarkGateError("benchmark gate hourly-rate source changed")
    return rate


def _verify_budget(gate: dict[str, Any], benchmark: dict[str, Any], rate: float, created_at: datetime) -> None:
    measured = tuple(gate.get(key) for key in ("seconds_per_part", "projected_hours", "projected_cost_usd"))
    if measured != _projection(benchmark["elapsed_seconds"], rate):
        raise BenchmarkGateError("benchmark gate projection no longer matches measured evidence")
    if gate["projected_cost_usd"] > COST_CAP_USD:
        raise BenchmarkGateError(f"projected cost exceeds the USD {COST_CAP_USD:.2f} cap")
    budget_seconds, deadline = _budget_deadline(created_at, benchmark["elapsed_seconds"], rate)
    budget_fields = (gate.get("budget_seconds_remaining_at_gate"), gate.get("budget_deadline_utc"))
    expected_budget = (budget_seconds, deadline.strftime(UTC_FORMAT))
    if budget_fields != expected_budget or _utc_now() >= deadline:
        raise BenchmarkGateError("benchmark gate runtime budget is changed or expired")


def verify_gate(
    gate_path: Path,
    dataset: Path,
    environment: Path,
    source_commit: str,
    runtime_sha256: str,
) -> dict[str, Any]:
    _identifiers(source_commit, runtime_sha256)
    gate = _json(gate_path)
    benchmark_path = Path(str(gate.get("benchmark_artifact", "")))
    benchmark = _json(benchmark_path)
    _identity(benchmark, source_commit, runtime_sha256, dataset, environment)
    _artifact_files(benchmark)
    _verify_gate_identity(gate, benchmark_path, dataset, environment, source_commit, runtime_sha256)
    created_at = _utc_timestamp(gate.get("gate_created_at_utc"), "benchmark gate creation time")
    if created_at > _utc_now():
        raise BenchmarkGateError("benchmark gate creation time is future-dated")
    rate = _verify_rate(gate, environment, created_at)
    _verify_budget(gate, benchmark, rate, created_at)
    return gate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "verify"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--dataset-artifact", type=Path, required=True)
        subparser.add_argument("--env-artifact", type=Path, required=True)
        subparser.add_argument("--source-commit", required=True)
        subparser.add_argument("--runtime-sha256", required=True)
    create = subparsers.choices["create"]
    create.add_argument("--benchmark-artifact", type=Path, required=True)
    create.add_argument("--hourly-rate-usd", type=float, required=True)
    create.add_argument("--hourly-rate-evidence", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.choices["verify"]
    verify.add_argument("--gate", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "create":
        payload = create_gate(
            args.benchmark_artifact,
            args.dataset_artifact,
            args.env_artifact,
            args.source_commit,
            args.runtime_sha256,
            args.hourly_rate_usd,
            args.hourly_rate_evidence,
            args.output,
        )
    else:
        payload = verify_gate(
            args.gate,
            args.dataset_artifact,
            args.env_artifact,
            args.source_commit,
            args.runtime_sha256,
        )
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
