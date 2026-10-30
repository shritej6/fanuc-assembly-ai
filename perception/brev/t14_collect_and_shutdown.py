"""Copy, independently verify, and retire the dedicated Brev T14 instance."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

PARTS = tuple(f"M{product}_{position:02d}" for product in range(1, 7) for position in range(1, 5))
INSTANCE_NAME = "fa-t14"


class CollectionError(ValueError):
    """Raised when off-host T14 evidence cannot be proven complete."""


@dataclass(frozen=True)
class CopySpec:
    remote: str
    relative: str
    expected_sha256: str | None = None


CopyFile = Callable[[str, str, Path], None]
Lifecycle = Callable[[str, str], None]
ResolveInstance = Callable[[str], str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CollectionError(f"evidence must be a real file: {path}")
    try:
        payload = json.loads(path.read_text(), parse_constant=_reject_constant)
    except (OSError, ValueError) as exc:
        raise CollectionError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise CollectionError(f"JSON evidence must be an object: {path}")
    return payload


def _remote(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CollectionError(f"{label} remote path is missing")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or "\x00" in value:
        raise CollectionError(f"{label} remote path is unsafe: {value}")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CollectionError(f"{label} SHA-256 is invalid")
    return value


def _destination(path: Path) -> tuple[Path, Path]:
    destination = path.expanduser()
    if not destination.is_absolute() or destination == Path("/"):
        raise CollectionError("destination must be a safe absolute path")
    partial = destination.with_name(f"{destination.name}.partial")
    if destination.exists() or destination.is_symlink() or partial.exists() or partial.is_symlink():
        raise CollectionError("destination and partial directory must be fresh")
    if not destination.parent.is_dir():
        raise CollectionError("destination parent must already exist")
    partial.mkdir()
    return destination, partial


def _brev_copy(instance: str, remote: str, local: Path) -> None:
    subprocess.run(["brev", "copy", f"{instance}:{remote}", str(local)], check=True)


def _brev_lifecycle(action: str, instance: str) -> None:
    if action not in {"delete", "stop"}:
        raise CollectionError(f"unsupported Brev lifecycle action: {action}")
    subprocess.run(["brev", action, instance], check=True)


def _brev_resolve_instance_id(instance_name: str) -> str:
    result = subprocess.run(
        ["brev", "ls", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(result.stdout, parse_constant=_reject_constant)
    except ValueError as exc:
        raise CollectionError("brev ls returned invalid JSON") from exc
    records = payload.get("workspaces") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise CollectionError("brev ls JSON does not contain a workspaces list")
    matches = []
    for record in records:
        if not isinstance(record, dict):
            raise CollectionError("brev ls JSON contains an invalid instance record")
        if record.get("name") == instance_name:
            matches.append(record.get("id"))
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise CollectionError(f"brev ls did not resolve exactly one {instance_name} instance")
    if re.fullmatch(r"[a-z0-9]+", matches[0]) is None:
        raise CollectionError("brev ls returned an invalid instance ID")
    return matches[0]


def _copy(spec: CopySpec, root: Path, instance: str, copy_file: CopyFile) -> None:
    target = root / spec.relative
    if target.exists() or target.is_symlink():
        raise CollectionError(f"duplicate collection target: {spec.relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    copy_file(instance, spec.remote, target)
    if target.is_symlink() or not target.is_file():
        raise CollectionError(f"Brev copy did not produce a real file: {spec.relative}")
    if spec.expected_sha256 is not None and _sha256(target) != spec.expected_sha256:
        raise CollectionError(f"copied evidence hash mismatch: {spec.relative}")


def _copy_many(specs: Iterable[CopySpec], root: Path, instance: str, copy_file: CopyFile) -> list[CopySpec]:
    copied = []
    for spec in specs:
        _copy(spec, root, instance, copy_file)
        copied.append(spec)
    return copied


def _seed_specs(args: argparse.Namespace) -> tuple[CopySpec, ...]:
    return (
        CopySpec(_remote(args.summary_remote, "summary"), "summary.json"),
        CopySpec(_remote(args.dataset_artifact_remote, "dataset artifact"), "dataset/artifact.json"),
        CopySpec(_remote(args.env_artifact_remote, "environment artifact"), "environment/artifact.json"),
        CopySpec(_remote(args.env_audit_remote, "environment audit"), "environment/pip_audit.json"),
        CopySpec(_remote(args.benchmark_gate_remote, "benchmark gate"), "benchmark/gate.json"),
    )


def _runs(summary: dict[str, Any]) -> list[dict[str, Any]]:
    runs = summary.get("runs")
    if summary.get("status") != "approved" or not isinstance(runs, list) or len(runs) != 24:
        raise CollectionError("final summary is not an approved 24-part generation")
    parts = [run.get("part") for run in runs if isinstance(run, dict)]
    invalid_parts = len(parts) != 24 or any(not isinstance(part, str) for part in parts)
    if invalid_parts or set(parts) != set(PARTS) or len(set(parts)) != 24:
        raise CollectionError("final summary does not cover the exact 24 parts")
    return sorted(runs, key=lambda run: PARTS.index(run["part"]))


def _one_value(records: Iterable[dict[str, Any]], key: str) -> str:
    values = [_remote(record.get(key), key) for record in records]
    if not values or any(value != values[0] for value in values[1:]):
        raise CollectionError(f"run records disagree on {key}")
    return values[0]


def _run_specs(runs: list[dict[str, Any]]) -> list[CopySpec]:
    specs = []
    for run in runs:
        part = run["part"]
        specs.extend(
            (
                CopySpec(
                    _remote(run.get("artifact_path"), "run artifact"),
                    f"runs/{part}/artifact.json",
                    _digest(run.get("artifact_sha256"), "run artifact"),
                ),
                CopySpec(
                    _remote(run.get("metrics_path"), "run metrics"),
                    f"runs/{part}/metrics.json",
                    _digest(run.get("metrics_sha256"), "run metrics"),
                ),
                CopySpec(
                    _remote(run.get("model_path"), "run model"),
                    f"runs/{part}/model.ckpt",
                    _digest(run.get("model_sha256"), "run model"),
                ),
            )
        )
    return specs


def _dataset_specs(dataset_dir: str, dataset: dict[str, Any]) -> list[CopySpec]:
    return [
        CopySpec(
            f"{dataset_dir}/_READY.json",
            "dataset/_READY.json",
            _digest(dataset.get("ready_sha256"), "dataset readiness"),
        ),
        CopySpec(
            f"{dataset_dir}/manifest.jsonl",
            "dataset/manifest.jsonl",
            _digest(dataset.get("manifest_sha256"), "dataset manifest"),
        ),
    ]


def _environment_specs(
    env_dir: str,
    environment: dict[str, Any],
    summary: dict[str, Any],
) -> list[CopySpec]:
    specs = [
        CopySpec(
            f"{env_dir}/.complete",
            "environment/.complete",
            _digest(summary.get("env_complete_sha256"), "environment completion"),
        ),
        CopySpec(
            f"{env_dir}/requirements.freeze.txt",
            "environment/requirements.freeze.txt",
            _digest(environment.get("freeze_sha256"), "environment freeze"),
        ),
        CopySpec(
            f"{env_dir}/requirements.audit.txt",
            "environment/requirements.audit.txt",
            _digest(environment.get("audit_requirements_sha256"), "environment audit requirements"),
        ),
        CopySpec(
            _remote(environment.get("platform_identity"), "platform identity"),
            "environment/platform.json",
            _digest(environment.get("platform_identity_sha256"), "platform identity"),
        ),
    ]
    return specs + _bootstrap_specs(environment)


def _bootstrap_specs(environment: dict[str, Any]) -> list[CopySpec]:
    fields = (
        ("uv_executable", "uv_executable_sha256", "environment/bootstrap/uv"),
        ("bootstrap_freeze", "bootstrap_freeze_sha256", "environment/bootstrap/requirements.freeze.txt"),
        (
            "bootstrap_audit_requirements",
            "bootstrap_audit_requirements_sha256",
            "environment/bootstrap/requirements.audit.txt",
        ),
        ("bootstrap_pip_audit", "bootstrap_pip_audit_sha256", "environment/bootstrap/pip_audit.json"),
        ("python_executable", "python_executable_sha256", "environment/python/python3.11"),
    )
    return [
        CopySpec(_remote(environment.get(path), path), relative, _digest(environment.get(digest), digest))
        for path, digest, relative in fields
    ]


def _gate_specs(gate: dict[str, Any]) -> list[CopySpec]:
    return [
        CopySpec(
            _remote(gate.get("hourly_rate_evidence"), "rate evidence"),
            "benchmark/rate.json",
            _digest(gate.get("hourly_rate_evidence_sha256"), "rate evidence"),
        ),
        CopySpec(
            _remote(gate.get("benchmark_artifact"), "benchmark artifact"),
            "benchmark/artifact.json",
            _digest(gate.get("benchmark_artifact_sha256"), "benchmark artifact"),
        ),
    ]


def _support_specs(root: Path, runs: list[dict[str, Any]]) -> list[CopySpec]:
    summary = _json(root / "summary.json")
    dataset = _json(root / "dataset/artifact.json")
    environment = _json(root / "environment/artifact.json")
    gate = _json(root / "benchmark/gate.json")
    dataset_dir = _one_value(runs, "dataset_dir")
    env_dir = _one_value(runs, "env_dir")
    if _remote(environment.get("env_dir"), "environment directory") != env_dir:
        raise CollectionError("environment artifact and run records disagree on env_dir")
    specs = _dataset_specs(dataset_dir, dataset)
    specs += _environment_specs(env_dir, environment, summary)
    specs += _gate_specs(gate)
    return specs + _run_specs(runs)


def _benchmark_specs(root: Path) -> tuple[CopySpec, CopySpec]:
    artifact = _json(root / "benchmark/artifact.json")
    metrics = CopySpec(
        _remote(artifact.get("metrics_path"), "benchmark metrics"),
        "benchmark/metrics.json",
        _digest(artifact.get("metrics_sha256"), "benchmark metrics"),
    )
    model = CopySpec(
        _remote(artifact.get("model_path"), "benchmark model"),
        "benchmark/model.ckpt",
        _digest(artifact.get("model_sha256"), "benchmark model"),
    )
    return metrics, model


def _verify_seed_hashes(root: Path, summary: dict[str, Any]) -> None:
    checks = (
        ("dataset/artifact.json", "dataset_artifact_sha256"),
        ("environment/artifact.json", "env_artifact_sha256"),
        ("environment/pip_audit.json", "env_audit_sha256"),
        ("benchmark/gate.json", "benchmark_gate_sha256"),
    )
    for relative, field in checks:
        if _sha256(root / relative) != _digest(summary.get(field), field):
            raise CollectionError(f"off-host {field} mismatch")
    complete = (root / "environment/.complete").read_text().strip()
    if complete != _sha256(root / "environment/artifact.json"):
        raise CollectionError("environment completion marker disagrees with its artifact")


def _verify_run_records(root: Path, runs: list[dict[str, Any]]) -> None:
    for run in runs:
        part = run["part"]
        artifact = _json(root / f"runs/{part}/artifact.json")
        if artifact.get("part") != part:
            raise CollectionError(f"run artifact part mismatch: {part}")
        for name, field in (("metrics.json", "metrics_sha256"), ("model.ckpt", "model_sha256")):
            if _sha256(root / f"runs/{part}/{name}") != _digest(artifact.get(field), field):
                raise CollectionError(f"run {part} {field} mismatch")


def _manifest(root: Path, specs: list[CopySpec], instance_id: str) -> None:
    seen = set()
    files = []
    for spec in specs:
        if spec.relative in seen:
            raise CollectionError(f"duplicate manifest path: {spec.relative}")
        seen.add(spec.relative)
        path = root / spec.relative
        files.append(
            {
                "local_path": spec.relative,
                "remote_path": spec.remote,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload = {"status": "off_host_verified", "brev_instance_id": instance_id, "files": files}
    manifest = root / "OFFHOST_MANIFEST.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    verified = {"status": "verified", "manifest_sha256": _sha256(manifest), "file_count": len(files)}
    (root / "OFFHOST_VERIFIED.json").write_text(json.dumps(verified, indent=2, sort_keys=True) + "\n")


def _verify_instance(root: Path, instance_id: str) -> None:
    platform = _json(root / "environment/platform.json")
    if platform.get("brev_instance_id") != instance_id:
        raise CollectionError("requested Brev instance ID disagrees with platform evidence")


def _failure_marker(root: Path, error: Exception) -> None:
    payload = {"status": "collection_failed", "error_type": type(error).__name__, "message": str(error)}
    (root / "COLLECTION_FAILED.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _preserve_and_stop(
    partial: Path,
    destination: Path,
    instance_id: str,
    error: Exception,
    lifecycle: Lifecycle,
) -> None:
    root = partial if partial.is_dir() else destination
    marker_error = None
    try:
        _failure_marker(root, error)
    except OSError as exc:
        marker_error = exc
    try:
        lifecycle("stop", instance_id)
    except (CollectionError, OSError, subprocess.CalledProcessError) as stop_error:
        raise CollectionError(f"collection failed ({error}); stopping fa-t14 also failed ({stop_error})") from error
    if marker_error is not None:
        raise CollectionError(
            f"collection failed ({error}); failure evidence could not be written ({marker_error})"
        ) from error


def collect_and_shutdown(
    args: argparse.Namespace,
    *,
    copy_file: CopyFile = _brev_copy,
    lifecycle: Lifecycle = _brev_lifecycle,
    resolve_instance: ResolveInstance = _brev_resolve_instance_id,
) -> Path:
    """Collect one approved generation and retire only the dedicated T14 instance."""
    if args.instance_name != INSTANCE_NAME or not re.fullmatch(r"[a-z0-9]+", args.instance_id):
        raise CollectionError("collector is restricted to the exact fa-t14 instance identity")
    if args.confirm_delete != INSTANCE_NAME:
        raise CollectionError("explicit --confirm-delete fa-t14 is required")
    live_instance_id = resolve_instance(args.instance_name)
    if live_instance_id != args.instance_id:
        raise CollectionError("requested Brev instance ID disagrees with the live fa-t14 instance")
    destination, partial = _destination(Path(args.destination))
    specs: list[CopySpec] = []
    try:
        specs += _copy_many(_seed_specs(args), partial, INSTANCE_NAME, copy_file)
        summary = _json(partial / "summary.json")
        runs = _runs(summary)
        support = _support_specs(partial, runs)
        specs += _copy_many(support, partial, INSTANCE_NAME, copy_file)
        specs += _copy_many(_benchmark_specs(partial), partial, INSTANCE_NAME, copy_file)
        _verify_seed_hashes(partial, summary)
        _verify_run_records(partial, runs)
        _verify_instance(partial, args.instance_id)
        _manifest(partial, specs, args.instance_id)
        partial.replace(destination)
        lifecycle("delete", args.instance_id)
        return destination
    except (CollectionError, OSError, subprocess.CalledProcessError) as exc:
        _preserve_and_stop(partial, destination, args.instance_id, exc, lifecycle)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-name", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--confirm-delete", required=True)
    parser.add_argument("--summary-remote", required=True)
    parser.add_argument("--dataset-artifact-remote", required=True)
    parser.add_argument("--env-artifact-remote", required=True)
    parser.add_argument("--env-audit-remote", required=True)
    parser.add_argument("--benchmark-gate-remote", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main() -> None:
    result = collect_and_shutdown(_parser().parse_args())
    print(json.dumps({"status": "collected_and_deleted", "destination": str(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
