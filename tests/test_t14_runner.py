"""Tests for runner.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from perception.brev.t14_benchmark_gate import create_gate
from perception.rosie.t14_execute_patchcore import RunBoundaryError, _part, execute

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "perception/rosie/t14_run_patchcore.sbatch"
ENV_VERSIONS = {
    "anomalib": "2.7.0",
    "filelock": "4.2.4",
    "pillow": "13.0.0",
    "pip": "26.2.2",
    "pip_audit": "2.10.1",
    "setuptools": "84.2.1",
    "torch": "2.14.1+cu126",
    "torchvision": "0.29.1+cu126",
}
FAKE_PYTHON = f"""#!/bin/sh
if [ "$1" = "--version" ]; then printf 'Python %s\n' "${{FAKE_T14_PYTHON_VERSION:-3.11.18}}"; exit 0; fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "check" ]; then exit 0; fi
if [ "$1" = "-c" ]; then
  if [ "${{FAKE_T14_BAD_VERSION:-0}}" = "1" ]; then echo '{{"anomalib":"wrong"}}';
  else echo '{json.dumps(ENV_VERSIONS, sort_keys=True)}'; fi
  exit 0
fi
while [ $# -gt 0 ]; do
  case "$1" in --part) part=$2; shift 2;; --out) out=$2; shift 2;; *) shift;; esac
done
rel=model.ckpt
mkdir -p "$out/$part"
if [ "${{FAKE_T14_CASE:-ok}}" != "no_model" ]; then printf model > "$out/$part/$rel"; fi
if [ "${{FAKE_T14_CASE:-ok}}" = "nan_metrics" ]; then
  printf '{{"part":"%s","status":"measured","test":{{"image_AUROC":NaN,"image_F1Score":0.5}},"o_test_false_alarm_share":0.0,"model":{{"path":"%s","format":"anomalib-lightning-ckpt","size_bytes":5,"config":{{"backbone":"wide_resnet50_2","layers":["layer2","layer3"],"coreset_sampling_ratio":0.1,"num_neighbors":9}}}}}}\n' "$part" "$rel" > "$out/$part/metrics.json"
elif [ "${{FAKE_T14_CASE:-ok}}" = "bad_model_contract" ]; then
  printf '{{"part":"%s","status":"measured","test":{{"image_AUROC":0.75,"image_F1Score":0.5}},"o_test_false_alarm_share":0.0,"model":{{"path":"%s","format":"anomalib-lightning-ckpt","size_bytes":5,"config":{{"backbone":"wrong","layers":["layer2","layer3"],"coreset_sampling_ratio":0.1,"num_neighbors":9}}}}}}\n' "$part" "$rel" > "$out/$part/metrics.json"
else
  printf '{{"part":"%s","status":"measured","test":{{"image_AUROC":0.75,"image_F1Score":0.5}},"o_test_false_alarm_share":0.0,"model":{{"path":"%s","format":"anomalib-lightning-ckpt","size_bytes":5,"config":{{"backbone":"wide_resnet50_2","layers":["layer2","layer3"],"coreset_sampling_ratio":0.1,"num_neighbors":9}}}}}}\n' "$part" "$rel" > "$out/$part/metrics.json"
fi
"""

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parts() -> list[str]:
    return [f"M{product}_{position:02d}" for product in range(1, 7) for position in range(1, 5)]


def test_production_taxonomy_resolves_exact_24_parts() -> None:
    assert [_part(ROOT, index) for index in range(24)] == [
        "M1_01",
        "M1_02",
        "M1_03",
        "M1_04",
        "M2_01",
        "M2_02",
        "M2_03",
        "M2_04",
        "M3_01",
        "M3_02",
        "M3_03",
        "M3_04",
        "M4_01",
        "M4_02",
        "M4_03",
        "M4_04",
        "M5_01",
        "M5_02",
        "M5_03",
        "M5_04",
        "M6_01",
        "M6_02",
        "M6_03",
        "M6_04",
    ]


def _code_snapshot(root: Path) -> tuple[Path, str]:
    code = root / "code"
    (code / "contracts").mkdir(parents=True)
    objects = [{"object_id": part} for part in _parts()]
    (code / "contracts/taxonomy.json").write_text(json.dumps({"objects": objects}))
    subprocess.run(["git", "init", "-q", str(code)], check=True)
    subprocess.run(["git", "-C", str(code), "add", "contracts/taxonomy.json"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(code),
            "-c",
            "user.name=T14",
            "-c",
            "user.email=t14@example.invalid",
            "commit",
            "-q",
            "-m",
            "snapshot",
        ],
        check=True,
    )
    source = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
    return code, source


def _dataset_inputs(root: Path, source: str, runtime_hash: str) -> tuple[Path, Path]:
    dataset = root / "dataset"
    crop = dataset / "M1_01/train/good/crop.png"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"verified-crop")
    ready, manifest = dataset / "_READY.json", dataset / "manifest.jsonl"
    ready.write_text("{}")
    manifest.write_text(json.dumps({"output": "M1_01/train/good/crop.png", "crop_sha256": _sha(crop)}) + "\n")
    artifact = root / "dataset-artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "source_commit": source,
                "preparation_runtime_sha256": runtime_hash,
                "output_root": str(dataset),
                "ready_sha256": _sha(ready),
                "manifest_sha256": _sha(manifest),
            }
        )
    )
    return dataset, artifact


def _runtime_provenance(root: Path, env_dir: Path) -> dict[str, str]:
    python_install = root / "managed-python/cpython-3.11.18/bin"
    python_install.mkdir(parents=True)
    fake_python = python_install / "python3.11"
    fake_python.write_text(FAKE_PYTHON)
    fake_python.chmod(0o755)
    (env_dir / "bin/python").symlink_to(fake_python)
    bootstrap = root / "bootstrap"
    bootstrap.mkdir()
    uv = bootstrap / "uv"
    uv.write_text("#!/bin/sh\necho 'uv 0.13.6'\n")
    uv.chmod(0o755)
    bootstrap_freeze = bootstrap / "requirements.freeze.txt"
    bootstrap_requirements = bootstrap / "requirements.audit.txt"
    bootstrap_audit = bootstrap / "pip-audit.json"
    bootstrap_freeze.write_text("pip==26.2.2\nsetuptools==84.2.1\nuv==0.13.6\n")
    bootstrap_requirements.write_text("pip==26.2.2\nsetuptools==84.2.1\nuv==0.13.6\n")
    bootstrap_audit.write_text("{}\n")
    return {
        "python": "3.11.18",
        "python_install_dir": str(python_install.parent.resolve()),
        "python_executable": str(fake_python.resolve()),
        "python_executable_sha256": _sha(fake_python),
        "uv": "0.13.6",
        "uv_executable": str(uv.resolve()),
        "uv_executable_sha256": _sha(uv),
        "bootstrap_freeze": str(bootstrap_freeze.resolve()),
        "bootstrap_freeze_sha256": _sha(bootstrap_freeze),
        "bootstrap_audit_requirements": str(bootstrap_requirements.resolve()),
        "bootstrap_audit_requirements_sha256": _sha(bootstrap_requirements),
        "bootstrap_pip_audit": str(bootstrap_audit.resolve()),
        "bootstrap_pip_audit_sha256": _sha(bootstrap_audit),
    }


def _environment_inputs(root: Path, source: str, runtime_hash: str) -> tuple[Path, Path, Path]:
    env_dir = root / "env"
    (env_dir / "bin").mkdir(parents=True)
    (env_dir / ".complete").write_text("")
    freeze = env_dir / "requirements.freeze.txt"
    freeze.write_text("anomalib==2.7.0\n")
    runtime = _runtime_provenance(root, env_dir)
    audit = root / "audit.json"
    audit.write_text("{}")
    platform = root / "platform.json"
    platform.write_text(json.dumps({"brev_instance_id": "test-id", "brev_machine_type": "test-machine"}))
    artifact = root / "env-artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "source_commit": source,
                "runtime_sha256": runtime_hash,
                "runtime_kind": "brev-vm",
                "env_dir": str(env_dir.resolve()),
                "freeze_sha256": _sha(freeze),
                "pip_audit_sha256": _sha(audit),
                "platform_identity": str(platform.resolve()),
                "platform_identity_sha256": _sha(platform),
                **runtime,
                **ENV_VERSIONS,
            }
        )
    )
    (env_dir / ".complete").write_text(f"{_sha(artifact)}\n")
    return env_dir, artifact, audit


def _benchmark_gate(root: Path, source: str, runtime_hash: str, dataset: Path, environment: Path) -> Path:
    evidence = root / "benchmark-evidence"
    evidence.mkdir()
    metrics, model = evidence / "metrics.json", evidence / "model.ckpt"
    metrics.write_text("{}")
    model.write_bytes(b"benchmark-model")
    benchmark = root / "benchmark-artifact.json"
    benchmark.write_text(
        json.dumps(
            {
                "part_index": 0,
                "source_commit": source,
                "runtime_sha256": runtime_hash,
                "dataset_artifact_sha256": _sha(dataset),
                "env_artifact_sha256": _sha(environment),
                "elapsed_seconds": 10.0,
                "metrics_path": str(metrics.resolve()),
                "metrics_sha256": _sha(metrics),
                "model_path": str(model.resolve()),
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
    create_gate(benchmark, dataset, environment, source, runtime_hash, 1.0, rate, gate)
    return gate


def _execution_inputs(root: Path) -> Namespace:
    runtime_hash = "f" * 64
    code, source = _code_snapshot(root)
    dataset, dataset_artifact = _dataset_inputs(root, source, runtime_hash)
    env_dir, env_artifact, audit = _environment_inputs(root, source, runtime_hash)
    cache = root / "cache"
    cache.mkdir()
    gate = _benchmark_gate(root, source, runtime_hash, dataset_artifact, env_artifact)
    return Namespace(
        code_root=code,
        source_commit=source,
        dataset_dir=dataset,
        dataset_artifact=dataset_artifact,
        env_dir=env_dir,
        env_artifact=env_artifact,
        env_audit=audit,
        model_cache=cache,
        part_index=0,
        output_root=root / "run",
        artifact=root / "run-artifact.json",
        workers=1,
        runner_id="123",
        hostname="node",
        runtime_sha256=runtime_hash,
        run_kind="production",
        benchmark_gate=gate,
        benchmark_gate_sha256=_sha(gate),
    )


def test_runner_is_one_v100_part_and_fails_closed_on_immutable_inputs() -> None:
    script = RUNNER.read_text()

    assert "#SBATCH --partition=dgx" in script
    assert "#SBATCH --gres=gpu:v100:1" in script
    assert "PART_INDEX" in script and "SLURM_ARRAY_TASK_ID" in script
    for variable in (
        "RUN_ROOT",
        "CODE_ROOT",
        "SOURCE_COMMIT",
        "DATASET_DIR",
        "DATASET_ARTIFACT",
        "ENV_DIR",
        "ENV_ARTIFACT",
        "ENV_AUDIT",
        "BENCHMARK_GATE",
        "BENCHMARK_GATE_SHA256",
    ):
        assert f"${{{variable}:?" in script
    assert '-B "$CODE_ROOT:$CODE_ROOT:ro"' in script
    assert '-B "$DATASET_DIR:$DATASET_DIR:ro"' in script
    assert '-B "$DATASET_ARTIFACT:$DATASET_ARTIFACT:ro"' in script
    assert '-B "$ENV_DIR:$ENV_DIR:ro"' in script
    assert '-B "$ENV_ARTIFACT:$ENV_ARTIFACT:ro"' in script
    assert '-B "$ENV_AUDIT:$ENV_AUDIT:ro"' in script
    assert 'git -C "$CODE_ROOT" rev-parse HEAD' in script
    assert "t14_execute_patchcore.py" in script
    assert 'test ! -e "$OUTPUT_ROOT"' in script
    assert 'test ! -e "$ARTIFACT"' in script
    assert 'test -s "$ARTIFACT"' in script
    assert 'mv "$POINTER_TMP" "$APPROVED"' in script
    assert '--run-kind production --benchmark-gate "$BENCHMARK_GATE"' in script
    assert '--benchmark-gate-sha256 "$BENCHMARK_GATE_SHA256"' in script


def test_execute_writes_evidence_only_after_measured_metrics(tmp_path: Path) -> None:
    args = _execution_inputs(tmp_path)

    artifact = execute(args)

    assert artifact["part"] == "M1_01"
    assert artifact["elapsed_seconds"] > 0
    assert artifact["metrics_sha256"] == _sha(args.output_root / "M1_01/metrics.json")
    assert artifact["model_sha256"] == _sha(Path(artifact["model_path"]))
    assert artifact["benchmark_gate_sha256"] == _sha(args.benchmark_gate)
    assert json.loads(args.artifact.read_text()) == artifact


def test_execute_benchmark_is_distinct_from_production_authorization(tmp_path: Path) -> None:
    args = _execution_inputs(tmp_path)
    args.run_kind = "benchmark"
    args.benchmark_gate = None
    args.benchmark_gate_sha256 = None

    artifact = execute(args)

    assert artifact["run_kind"] == "benchmark"
    assert "benchmark_gate_sha256" not in artifact


def test_execute_production_requires_benchmark_authorization(tmp_path: Path) -> None:
    args = _execution_inputs(tmp_path)
    args.benchmark_gate_sha256 = None

    with pytest.raises(RunBoundaryError, match="benchmark gate"):
        execute(args)

    assert not args.output_root.exists()


def test_execute_rejects_benchmark_gate_hash_drift(tmp_path: Path) -> None:
    args = _execution_inputs(tmp_path)
    args.benchmark_gate.write_text("changed\n")

    with pytest.raises(RunBoundaryError, match="benchmark gate hash"):
        execute(args)

    assert not args.output_root.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_commit", "0" * 40),
        ("runtime_sha256", "0" * 64),
        ("dataset_artifact_sha256", "0" * 64),
        ("env_artifact_sha256", "0" * 64),
        ("projected_cost_usd", 121.0),
    ),
)
def test_execute_rejects_hash_matching_semantically_invalid_gate(tmp_path: Path, field: str, value: object) -> None:
    args = _execution_inputs(tmp_path)
    gate = json.loads(args.benchmark_gate.read_text())
    gate[field] = value
    args.benchmark_gate.write_text(json.dumps(gate))
    args.benchmark_gate_sha256 = _sha(args.benchmark_gate)

    with pytest.raises(RunBoundaryError, match="not authorized"):
        execute(args)

    assert not args.output_root.exists()


def test_execute_rejects_a_changed_dataset_before_running(tmp_path: Path) -> None:
    args = _execution_inputs(tmp_path)
    (args.dataset_dir / "manifest.jsonl").write_text("changed\n")

    with pytest.raises(RunBoundaryError, match="manifest hash"):
        execute(args)

    assert not args.output_root.exists()


def test_execute_accepts_a_fully_verified_relocated_dataset(tmp_path: Path) -> None:
    args = _execution_inputs(tmp_path)
    relocated = tmp_path / "relocated-dataset"
    args.dataset_dir.rename(relocated)
    args.dataset_dir = relocated

    artifact = execute(args)

    assert artifact["dataset_dir"] == str(relocated.resolve())


def test_execute_rejects_changed_or_extra_crop_bytes(tmp_path: Path) -> None:
    args = _execution_inputs(tmp_path)
    crop = args.dataset_dir / "M1_01/train/good/crop.png"
    crop.write_bytes(b"changed-after-preparation")

    with pytest.raises(RunBoundaryError, match="crop hash"):
        execute(args)

    assert not args.output_root.exists()


def test_execute_rejects_an_unlisted_dataset_file(tmp_path: Path) -> None:
    args = _execution_inputs(tmp_path)
    (args.dataset_dir / "extra.png").write_bytes(b"unlisted")

    with pytest.raises(RunBoundaryError, match="unlisted"):
        execute(args)


def test_execute_rejects_a_symlinked_crop(tmp_path: Path) -> None:
    args = _execution_inputs(tmp_path)
    crop = args.dataset_dir / "M1_01/train/good/crop.png"
    outside = tmp_path / "outside.png"
    outside.write_bytes(crop.read_bytes())
    crop.unlink()
    crop.symlink_to(outside)

    with pytest.raises(RunBoundaryError, match="symlink"):
        execute(args)


def test_execute_rejects_dirty_source_snapshot(tmp_path: Path) -> None:
    args = _execution_inputs(tmp_path)
    (args.code_root / "untracked.py").write_text("raise RuntimeError\n")

    with pytest.raises(RunBoundaryError, match="clean"):
        execute(args)

    assert not args.output_root.exists()


def test_execute_rejects_nonfinite_metrics_or_missing_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _execution_inputs(tmp_path)
    monkeypatch.setenv("FAKE_T14_CASE", "nan_metrics")

    with pytest.raises(RunBoundaryError, match="JSON|finite"):
        execute(args)

    assert not args.artifact.exists()


def test_execute_requires_deployable_model_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _execution_inputs(tmp_path)
    monkeypatch.setenv("FAKE_T14_CASE", "no_model")

    with pytest.raises(RunBoundaryError, match="model"):
        execute(args)

    assert not args.artifact.exists()


def test_execute_requires_exact_deployable_model_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _execution_inputs(tmp_path)
    monkeypatch.setenv("FAKE_T14_CASE", "bad_model_contract")

    with pytest.raises(RunBoundaryError, match="model contract"):
        execute(args)

    assert not args.artifact.exists()


def test_execute_binds_and_revalidates_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _execution_inputs(tmp_path)
    monkeypatch.setenv("FAKE_T14_BAD_VERSION", "1")

    with pytest.raises(RunBoundaryError, match="versions"):
        execute(args)

    assert not args.output_root.exists()


def test_execute_rejects_a_different_environment_directory(tmp_path: Path) -> None:
    args = _execution_inputs(tmp_path)
    copied = tmp_path / "copied-env"
    shutil.copytree(args.env_dir, copied)
    args.env_dir = copied

    with pytest.raises(RunBoundaryError, match="different environment"):
        execute(args)

    assert not args.output_root.exists()


def test_execute_rejects_source_or_runtime_provenance_mismatch(tmp_path: Path) -> None:
    args = _execution_inputs(tmp_path)
    env_artifact = json.loads(args.env_artifact.read_text())
    env_artifact["runtime_sha256"] = "0" * 64
    args.env_artifact.write_text(json.dumps(env_artifact))

    with pytest.raises(RunBoundaryError, match="benchmark gate|runtime identity"):
        execute(args)

    assert not args.output_root.exists()


def test_execute_rejects_code_root_at_a_different_commit(tmp_path: Path) -> None:
    args = _execution_inputs(tmp_path)
    (args.code_root / "changed.txt").write_text("new commit\n")
    subprocess.run(["git", "-C", str(args.code_root), "add", "changed.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(args.code_root),
            "-c",
            "user.name=T14",
            "-c",
            "user.email=t14@example.invalid",
            "commit",
            "-q",
            "-m",
            "changed",
        ],
        check=True,
    )

    with pytest.raises(RunBoundaryError, match="code root HEAD"):
        execute(args)

    assert not args.output_root.exists()
