"""Execute one immutable T14 PatchCore part run and emit hash-linked evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from perception.anomaly.patchcore_contract import (
    MODEL_CONFIG,
    MODEL_FORMAT,
    REQUIRED_IMAGE_METRICS,
)
from perception.brev.t14_benchmark_gate import BenchmarkGateError, verify_gate

ENV_PACKAGES = {
    "anomalib": "anomalib",
    "torch": "torch",
    "torchvision": "torchvision",
    "pillow": "Pillow",
    "filelock": "filelock",
    "pip": "pip",
    "setuptools": "setuptools",
    "pip_audit": "pip-audit",
}
ENV_VERSIONS = {
    "anomalib": "2.7.0",
    "torch": "2.14.1+cu126",
    "torchvision": "0.29.1+cu126",
    "pillow": "13.0.0",
    "filelock": "4.2.4",
    "pip": "26.2.2",
    "setuptools": "84.2.1",
    "pip_audit": "2.10.1",
}
ENV_PYTHON_VERSION = "3.11.18"
ENV_UV_VERSION = "0.13.6"
ROSIE_SINGULARITY_IMAGE = "/data/containers/msoe-pytorch-23.05-py3.sif"
BREV_PROVENANCE_PREFIXES = ("python", "uv", "bootstrap_", "platform_identity")


class RunBoundaryError(ValueError):
    """Raised when a T14 run input or result cannot be proven safe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RunBoundaryError(f"required provenance file is missing or a symlink: {path}")
    try:
        payload = json.loads(path.read_text(), parse_constant=_reject_constant)
    except (OSError, ValueError) as exc:
        raise RunBoundaryError(f"invalid JSON provenance: {path}") from exc
    if not isinstance(payload, dict):
        raise RunBoundaryError(f"JSON provenance must be an object: {path}")
    return payload


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _real_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RunBoundaryError(f"{label} must be a real directory: {path}")
    return path.resolve()


def _real_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RunBoundaryError(f"{label} must be a real file: {path}")
    return path.resolve()


def _source_snapshot(code: Path, source_commit: str) -> None:
    try:
        head = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(
            ["git", "-C", str(code), "status", "--porcelain", "--untracked-files=all"], text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunBoundaryError("code root must be an exact Git snapshot") from exc
    if head != source_commit:
        raise RunBoundaryError("code root HEAD disagrees with source commit")
    if status:
        raise RunBoundaryError("code root must be a clean exact source snapshot")


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line in path.read_text().splitlines():
            row = json.loads(line, parse_constant=_reject_constant)
            if not isinstance(row, dict):
                raise TypeError("manifest row is not an object")
            rows.append(row)
    except (OSError, TypeError, ValueError) as exc:
        raise RunBoundaryError(f"invalid dataset manifest rows: {path}") from exc
    if not rows:
        raise RunBoundaryError("dataset manifest must contain crop rows")
    return rows


def _crop_file(dataset: Path, row: dict[str, Any]) -> Path:
    output, digest = row.get("output"), row.get("crop_sha256")
    if not isinstance(output, str) or not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
        raise RunBoundaryError("dataset manifest crop identity is invalid")
    relative = Path(output)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != output:
        raise RunBoundaryError(f"dataset manifest crop path is unsafe: {output}")
    cursor = dataset
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise RunBoundaryError(f"dataset crop path contains a symlink: {output}")
    crop = _real_file(dataset / relative, "dataset crop")
    if not crop.is_relative_to(dataset) or _sha256(crop) != digest:
        raise RunBoundaryError(f"dataset crop hash disagrees with manifest: {output}")
    return crop


def _verify_crops(dataset: Path, manifest: Path) -> None:
    crops = [_crop_file(dataset, row) for row in _manifest_rows(manifest)]
    expected = set(crops)
    if len(expected) != len(crops):
        raise RunBoundaryError("dataset manifest contains duplicate crop outputs")
    entries = tuple(dataset.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise RunBoundaryError("dataset contains a symlink")
    metadata = {dataset / "_READY.json", manifest}
    actual = {path.resolve() for path in entries if path.is_file() and path not in metadata}
    if actual != expected:
        raise RunBoundaryError("dataset contains missing or unlisted crop files")


def _dataset(dataset: Path, artifact_path: Path, source_commit: str) -> dict[str, str]:
    artifact = _json(artifact_path)
    ready = _real_file(dataset / "_READY.json", "dataset readiness marker")
    manifest = _real_file(dataset / "manifest.jsonl", "dataset manifest")
    if artifact.get("source_commit") != source_commit:
        raise RunBoundaryError("dataset source commit disagrees with this run")
    origin = artifact.get("output_root")
    prep_runtime = artifact.get("preparation_runtime_sha256")
    if not isinstance(origin, str) or not Path(origin).is_absolute():
        raise RunBoundaryError("dataset artifact lacks an absolute preparation origin")
    if not isinstance(prep_runtime, str) or re.fullmatch(r"[0-9a-f]{64}", prep_runtime) is None:
        raise RunBoundaryError("dataset artifact lacks its preparation runtime identity")
    hashes = {"dataset_ready_sha256": _sha256(ready), "dataset_manifest_sha256": _sha256(manifest)}
    if hashes["dataset_ready_sha256"] != artifact.get("ready_sha256"):
        raise RunBoundaryError("dataset readiness hash disagrees with its artifact")
    if hashes["dataset_manifest_sha256"] != artifact.get("manifest_sha256"):
        raise RunBoundaryError("dataset manifest hash disagrees with its artifact")
    _verify_crops(dataset, manifest)
    return {
        **hashes,
        "dataset_artifact_path": str(artifact_path.resolve()),
        "dataset_artifact_sha256": _sha256(artifact_path),
        "dataset_origin": origin,
        "dataset_preparation_runtime_sha256": prep_runtime,
    }


def _runtime_versions(env_dir: Path) -> dict[str, str]:
    probe = (
        "import json,sys; from importlib.metadata import version; "
        "p=json.loads(sys.argv[1]); print(json.dumps({k:version(v) for k,v in p.items()},sort_keys=True))"
    )
    try:
        subprocess.run([str(env_dir / "bin/python"), "-m", "pip", "check"], check=True, capture_output=True, text=True)
        output = subprocess.check_output(
            [str(env_dir / "bin/python"), "-c", probe, json.dumps(ENV_PACKAGES)], text=True
        )
        versions = json.loads(output, parse_constant=_reject_constant)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise RunBoundaryError("environment runtime validation failed") from exc
    if not isinstance(versions, dict) or not all(isinstance(value, str) for value in versions.values()):
        raise RunBoundaryError("environment runtime versions are invalid")
    return versions


def _python_identity(env_dir: Path) -> dict[str, str]:
    executable = env_dir / "bin/python"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RunBoundaryError("environment Python is missing or not executable")
    try:
        reported = subprocess.check_output([str(executable), "--version"], stderr=subprocess.STDOUT, text=True).strip()
        resolved = executable.resolve(strict=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunBoundaryError("environment Python identity probe failed") from exc
    if reported != f"Python {ENV_PYTHON_VERSION}":
        raise RunBoundaryError(f"environment requires exact Python {ENV_PYTHON_VERSION}")
    return {
        "python": ENV_PYTHON_VERSION,
        "python_executable": str(resolved),
        "python_executable_sha256": _sha256(resolved),
    }


def _bootstrap_provenance(artifact: dict[str, Any]) -> None:
    if artifact.get("uv") != ENV_UV_VERSION:
        raise RunBoundaryError(f"environment requires exact uv {ENV_UV_VERSION}")
    uv = _real_file(Path(str(artifact.get("uv_executable", ""))), "bootstrap uv executable")
    if _sha256(uv) != artifact.get("uv_executable_sha256"):
        raise RunBoundaryError("bootstrap uv executable hash disagrees with its artifact")
    try:
        reported = subprocess.check_output([str(uv), "--version"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunBoundaryError("bootstrap uv identity probe failed") from exc
    if re.fullmatch(rf"uv {re.escape(ENV_UV_VERSION)}(?:\s.*)?", reported) is None:
        raise RunBoundaryError("bootstrap uv runtime version disagrees with its artifact")
    evidence = (
        ("bootstrap_freeze", "bootstrap_freeze_sha256"),
        ("bootstrap_audit_requirements", "bootstrap_audit_requirements_sha256"),
        ("bootstrap_pip_audit", "bootstrap_pip_audit_sha256"),
    )
    for path_field, hash_field in evidence:
        path = _real_file(Path(str(artifact.get(path_field, ""))), path_field)
        if _sha256(path) != artifact.get(hash_field):
            raise RunBoundaryError(f"{path_field} hash disagrees with its artifact")


def _brev_runtime(env_dir: Path, artifact: dict[str, Any]) -> None:
    identity = _python_identity(env_dir)
    if any(artifact.get(field) != value for field, value in identity.items()):
        raise RunBoundaryError("environment Python identity disagrees with its artifact")
    install_dir = Path(str(artifact.get("python_install_dir", ""))).resolve()
    if install_dir not in Path(identity["python_executable"]).parents:
        raise RunBoundaryError("environment Python escaped its managed install directory")
    _bootstrap_provenance(artifact)


def _runtime_contract(env_dir: Path, artifact: dict[str, Any]) -> None:
    runtime_kind = artifact.get("runtime_kind")
    if runtime_kind == "brev-vm":
        _brev_runtime(env_dir, artifact)
    elif "runtime_kind" in artifact:
        raise RunBoundaryError("environment runtime kind is unsupported")
    elif artifact.get("singularity_image") != ROSIE_SINGULARITY_IMAGE:
        raise RunBoundaryError("legacy Rosie environment has an invalid Singularity identity")
    elif any(field.startswith(BREV_PROVENANCE_PREFIXES) for field in artifact):
        raise RunBoundaryError("Brev environment cannot be downgraded to legacy Rosie")


def _environment(
    env_dir: Path, artifact_path: Path, audit_path: Path, source_commit: str, runtime_hash: str
) -> dict[str, str]:
    artifact = _json(artifact_path)
    complete = _real_file(env_dir / ".complete", "environment completion marker")
    freeze = _real_file(env_dir / "requirements.freeze.txt", "environment freeze")
    audit_path = _real_file(audit_path, "environment audit")
    if artifact.get("source_commit") != source_commit:
        raise RunBoundaryError("environment source commit disagrees with this run")
    if artifact.get("runtime_sha256") != runtime_hash:
        raise RunBoundaryError("environment runtime identity disagrees with this run")
    if Path(str(artifact.get("env_dir", ""))).resolve() != env_dir:
        raise RunBoundaryError("environment artifact points at a different environment")
    if complete.read_text().strip() != _sha256(artifact_path):
        raise RunBoundaryError("environment completion marker disagrees with its artifact")
    hashes = {
        "env_complete_sha256": _sha256(complete),
        "env_freeze_sha256": _sha256(freeze),
        "env_audit_sha256": _sha256(audit_path),
    }
    if hashes["env_freeze_sha256"] != artifact.get("freeze_sha256"):
        raise RunBoundaryError("environment freeze hash disagrees with its artifact")
    if hashes["env_audit_sha256"] != artifact.get("pip_audit_sha256"):
        raise RunBoundaryError("environment audit hash disagrees with its artifact")
    expected_versions = {field: artifact.get(field) for field in ENV_PACKAGES}
    if expected_versions != ENV_VERSIONS or _runtime_versions(env_dir) != ENV_VERSIONS:
        raise RunBoundaryError("environment runtime versions disagree with its artifact")
    _runtime_contract(env_dir, artifact)
    return {
        **hashes,
        "env_artifact_path": str(artifact_path.resolve()),
        "env_artifact_sha256": _sha256(artifact_path),
    }


def _part(code_root: Path, index: int) -> str:
    taxonomy = _json(code_root / "contracts/taxonomy.json")
    objects = taxonomy.get("objects")
    if not isinstance(objects, list):
        raise RunBoundaryError("taxonomy objects must be a list")
    parts = [
        item["object_id"]
        for item in objects
        if isinstance(item, dict) and isinstance(item.get("object_id"), str) and item["object_id"].startswith("M")
    ]
    if (
        len(parts) != 24
        or len(set(parts)) != 24
        or any(re.fullmatch(r"M[1-6]_0[1-4]", part or "") is None for part in parts)
    ):
        raise RunBoundaryError("taxonomy must contain the exact 24 part IDs")
    if not 0 <= index < len(parts):
        raise RunBoundaryError("part index must be in [0, 23]")
    return parts[index]


def _measured(metrics_path: Path, part: str, part_root: Path) -> tuple[Path, dict[str, Any]]:
    metrics = _json(metrics_path)
    if metrics.get("part") != part or metrics.get("status") != "measured":
        raise RunBoundaryError(f"part metrics are not completely measured: {metrics_path}")
    test = metrics.get("test")
    required = REQUIRED_IMAGE_METRICS
    values = tuple(test.get(key) for key in required) if isinstance(test, dict) else ()
    finite_test = len(values) == 2 and all(
        type(value) in (int, float) and math.isfinite(value) and 0.0 <= value <= 1.0 for value in values
    )
    false_alarm = metrics.get("o_test_false_alarm_share")
    finite_false_alarm = type(false_alarm) in (int, float) and math.isfinite(false_alarm) and 0.0 <= false_alarm <= 1.0
    if not finite_test or not finite_false_alarm:
        raise RunBoundaryError(f"part metrics must be non-empty and finite: {metrics_path}")
    model_info = metrics.get("model")
    relative = model_info.get("path") if isinstance(model_info, dict) else None
    if relative != "model.ckpt":
        raise RunBoundaryError("part metrics model path is invalid")
    exact_model = (
        model_info.get("format") == MODEL_FORMAT
        and model_info.get("config") == MODEL_CONFIG
        and type(model_info.get("size_bytes")) is int
        and model_info["size_bytes"] > 0
    )
    if not exact_model:
        raise RunBoundaryError("part metrics model contract is invalid")
    model = _real_file(part_root / relative, "deployable PatchCore model")
    if not model.is_relative_to(part_root.resolve()) or model.stat().st_size != model_info["size_bytes"]:
        raise RunBoundaryError("deployable PatchCore model escapes its part output")
    return model, model_info


def _train(env_dir: Path, code: Path, dataset: Path, cache: Path, args: argparse.Namespace, part: str) -> float:
    command = [
        str(env_dir / "bin/python"),
        "-m",
        "perception.anomaly.train_patchcore",
        "--crops",
        str(dataset),
        "--part",
        part,
        "--out",
        str(args.output_root),
        "--workers",
        str(args.workers),
    ]
    environment = {
        **os.environ,
        "PYTHONPATH": str(code),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TORCH_HOME": str(cache),
    }
    started = time.perf_counter()
    subprocess.run(command, cwd=code, env=environment, check=True)
    return time.perf_counter() - started


def _payload(
    args: argparse.Namespace,
    evidence: dict[str, str],
    code: Path,
    dataset: Path,
    env_dir: Path,
    part: str,
    metrics: Path,
    model: Path,
    model_info: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    payload = {
        **evidence,
        "part": part,
        "part_index": args.part_index,
        "source_commit": args.source_commit,
        "code_root": str(code),
        "dataset_dir": str(dataset),
        "env_dir": str(env_dir),
        "metrics_path": str(metrics),
        "metrics_sha256": _sha256(metrics),
        "model_path": str(model),
        "model_sha256": _sha256(model),
        "model_format": model_info["format"],
        "model_size_bytes": model_info["size_bytes"],
        "model_config": model_info["config"],
        "elapsed_seconds": elapsed,
        "runner_id": args.runner_id,
        "hostname": args.hostname,
        "runtime_sha256": args.runtime_sha256,
        "run_kind": args.run_kind,
    }
    if args.run_kind == "production":
        payload["benchmark_gate_sha256"] = args.benchmark_gate_sha256
        payload["benchmark_gate_path"] = str(args.benchmark_gate.resolve())
    return payload


def _authorization(args: argparse.Namespace) -> None:
    gate_hash = getattr(args, "benchmark_gate_sha256", None)
    gate_path = getattr(args, "benchmark_gate", None)
    if args.run_kind == "benchmark":
        if gate_hash is not None or gate_path is not None:
            raise RunBoundaryError("benchmark runs must not claim a production gate")
        return
    if re.fullmatch(r"[0-9a-f]{64}", gate_hash or "") is None:
        raise RunBoundaryError("production requires a benchmark gate SHA-256")
    gate = _real_file(gate_path, "benchmark gate") if isinstance(gate_path, Path) else None
    if gate is None or _sha256(gate) != gate_hash:
        raise RunBoundaryError("production benchmark gate hash disagrees with its file")
    try:
        verify_gate(
            gate,
            args.dataset_artifact,
            args.env_artifact,
            args.source_commit,
            args.runtime_sha256,
        )
    except BenchmarkGateError as exc:
        raise RunBoundaryError("production benchmark gate is not authorized") from exc


def _write_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RunBoundaryError(f"temporary artifact path must be fresh: {temporary}")
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    """Validate immutable inputs, execute one part, and atomically write its evidence."""
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        raise RunBoundaryError("source commit must be a full lowercase SHA-1")
    if re.fullmatch(r"[0-9a-f]{64}", args.runtime_sha256) is None:
        raise RunBoundaryError("runtime identity must be a lowercase SHA-256")
    _authorization(args)
    code = _real_directory(args.code_root, "code root")
    dataset = _real_directory(args.dataset_dir, "dataset directory")
    env_dir = _real_directory(args.env_dir, "environment directory")
    cache = _real_directory(args.model_cache, "model cache")
    if args.output_root.exists() or args.artifact.exists():
        raise RunBoundaryError("run output and artifact paths must be fresh")
    _source_snapshot(code, args.source_commit)
    part = _part(code, args.part_index)
    evidence = _dataset(dataset, args.dataset_artifact, args.source_commit)
    evidence.update(_environment(env_dir, args.env_artifact, args.env_audit, args.source_commit, args.runtime_sha256))
    args.output_root.mkdir(parents=True)
    elapsed = _train(env_dir, code, dataset, cache, args, part)
    part_root = (args.output_root / part).resolve()
    metrics = part_root / "metrics.json"
    model, model_info = _measured(metrics, part, part_root)
    payload = _payload(args, evidence, code, dataset, env_dir, part, metrics, model, model_info, elapsed)
    _write_artifact(args.artifact, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-artifact", type=Path, required=True)
    parser.add_argument("--env-dir", type=Path, required=True)
    parser.add_argument("--env-artifact", type=Path, required=True)
    parser.add_argument("--env-audit", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--part-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--runner-id", required=True)
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--run-kind", choices=("benchmark", "production"), required=True)
    parser.add_argument("--benchmark-gate", type=Path)
    parser.add_argument("--benchmark-gate-sha256")
    return parser


def main() -> None:
    print(json.dumps(execute(_parser().parse_args()), allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
