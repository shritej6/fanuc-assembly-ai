from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from perception.brev.t14_benchmark_gate import BenchmarkGateError, create_gate, verify_gate
from perception.brev.t14_guarded_production import RuntimeGuardError, _run_remote, run_guarded
from perception.rosie.t14_execute_patchcore import (
    ENV_VERSIONS,
    RunBoundaryError,
    _bootstrap_provenance,
    _environment,
    _python_identity,
)

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "perception/brev/t14_setup_patchcore.sh"
RUN = ROOT / "perception/brev/t14_run_patchcore.sh"
RUN_ALL = ROOT / "perception/brev/t14_run_all_patchcore.sh"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_artifact(root: Path) -> dict[str, str]:
    uv = root / "uv"
    uv.write_text("#!/bin/sh\necho 'uv 0.13.6'\n")
    uv.chmod(0o755)
    artifact = {"uv": "0.13.6", "uv_executable": str(uv), "uv_executable_sha256": _sha(uv)}
    bootstrap_packages = "pip==26.2.2\nsetuptools==84.2.1\nuv==0.13.6\n"
    for field, content in (
        ("bootstrap_freeze", bootstrap_packages),
        ("bootstrap_audit_requirements", bootstrap_packages),
        ("bootstrap_pip_audit", "{}\n"),
    ):
        path = root / field
        path.write_text(content)
        artifact[field] = str(path)
        artifact[f"{field}_sha256"] = _sha(path)
    return artifact


def _evidence(root: Path) -> tuple[Path, Path, Path, Path]:
    dataset, environment = root / "dataset.json", root / "environment.json"
    dataset.write_text("{}")
    platform = root / "platform.json"
    platform.write_text(json.dumps({"brev_instance_id": "instance-1", "brev_machine_type": "l4-1"}))
    environment.write_text(json.dumps({"platform_identity": str(platform), "platform_identity_sha256": _sha(platform)}))
    rate = root / "rate.json"
    rate.write_text(
        json.dumps(
            {
                "source_command": "brev search",
                "captured_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "brev_instance_id": "instance-1",
                "brev_machine_type": "l4-1",
                "hourly_rate_usd": 0.85,
            }
        )
    )
    metrics, model = root / "metrics.json", root / "model.ckpt"
    metrics.write_text("{}")
    model.write_bytes(b"model")
    benchmark = root / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "part_index": 0,
                "source_commit": "a" * 40,
                "runtime_sha256": "b" * 64,
                "dataset_artifact_sha256": _sha(dataset),
                "env_artifact_sha256": _sha(environment),
                "metrics_path": str(metrics),
                "metrics_sha256": _sha(metrics),
                "model_path": str(model),
                "model_sha256": _sha(model),
                "elapsed_seconds": 30.0,
            }
        )
    )
    return dataset, environment, benchmark, rate


def test_brev_environment_is_secure_pinned_and_l4_gated() -> None:
    script = SETUP.read_text()
    for requirement in (
        "torch==2.14.1",
        "torchvision==0.29.1",
        "anomalib==2.7.0",
        "pillow==13.0.0",
        "filelock==4.2.4",
        "pip==26.2.2",
        "setuptools==84.2.1",
        "pip-audit==2.10.1",
        "uv==0.13.6",
    ):
        assert requirement in script
    assert "UV_VERSION=0.13.6" in script and "PYTHON_VERSION=3.11.18" in script
    assert 'UV_PYTHON_INSTALL_DIR="$PYTHON_INSTALL_DIR"' in script
    assert "python install --managed-python --no-bin --no-config" in script
    assert "venv --managed-python --no-python-downloads --no-config --seed" in script
    assert "PIP_*|UV_*) unset" in script
    assert script.count("-m pip check") == 2 and script.count("pip freeze --all") == 2
    assert "get_device_capability() != (8, 9)" in script
    assert "torch.cuda.device_count() != 1" in script
    assert 'torch.cuda.get_device_name().strip() != "NVIDIA L4"' in script
    assert "torch.cuda.synchronize()" in script
    assert "expected_checksum = float(1024**3)" in script
    assert '"sm_89" not in torch.cuda.get_arch_list()' not in script
    assert '"runtime_kind": "brev-vm"' in script
    assert '"runtime_sha256": sys.argv[7]' in script
    assert '"torch": version("torch")' in script


def test_brev_bootstrap_is_minimal_and_audited_by_the_final_environment() -> None:
    script = SETUP.read_text()
    bootstrap_install = script[
        script.index('"$BOOTSTRAP_DIR/bin/python" -m pip install') : script.index(
            '"$BOOTSTRAP_DIR/bin/python" -m pip check'
        )
    ]
    assert "pip==26.2.2 setuptools==84.2.1 uv==0.13.6" in bootstrap_install
    assert "pip-audit" not in bootstrap_install and "filelock" not in bootstrap_install
    assert script.count("-m pip check") == 2 and script.count("pip freeze --all") == 2
    assert script.count("--strict --no-deps --disable-pip") == 2
    assert script.count("filelock==4.2.4") == 1 and script.count("pip-audit==2.10.1") == 1
    assert '"$BOOTSTRAP_DIR/bin/pip-audit"' not in script
    assert script.count('"$ENV_DIR/bin/pip-audit"') == 2
    bootstrap_audit = '--requirement "$BOOTSTRAP_DIR/requirements.audit.txt" --format=json --output "$BOOTSTRAP_AUDIT"'
    final_audit = '--requirement "$ENV_DIR/requirements.audit.txt" --format=json --output "$AUDIT"'
    assert script.index(bootstrap_audit) < script.index(final_audit) < script.index('"$ARTIFACT" "$AUDIT"')
    for field in (
        "python_executable_sha256",
        "uv_executable_sha256",
        "bootstrap_freeze_sha256",
        "bootstrap_audit_requirements_sha256",
        "bootstrap_pip_audit_sha256",
    ):
        assert f'"{field}"' in script
    assert "unset PYTHONPATH PYTHONHOME VIRTUAL_ENV" in script
    assert 'git -C "$CODE_ROOT" clean -ndx' in script
    assert "printf '%s\\n' \"$ARTIFACT\"" in script
    assert "singularity" not in script.lower() and "#SBATCH" not in script


def test_brev_runner_is_sequential_benchmark_first_and_runtime_neutral() -> None:
    script, all_script = RUN.read_text(), RUN_ALL.read_text()

    assert "python3" not in script
    assert "python3" not in all_script
    assert script.count('"$ENV_DIR/bin/python"') == 4
    assert all_script.count('"$ENV_DIR/bin/python"') == 2
    assert ': "${ENV_DIR:?set ENV_DIR to the approved Brev environment}"' in all_script
    assert "flock -n 9" in script
    assert 'test "$PART_INDEX" = 0' in script
    assert 't14_benchmark_gate.py" verify' in script
    assert script.index('t14_benchmark_gate.py" verify') < script.index("t14_execute_patchcore.py")
    assert '--runtime-sha256 "$RUNTIME_SHA256"' in script
    assert '--runner-id "brev-${BREV_INSTANCE_ID}-${ATTEMPT_ID}-part-${PART_INDEX}"' in script
    assert 'sha256sum "$PLATFORM_IDENTITY"' in script
    assert "--query-gpu=name,driver_version,compute_cap" in script
    assert "platform.get(key) != value" in script
    assert '"brev_instance_id": sys.argv[2]' in script
    assert '"hostname": sys.argv[3]' in script
    assert '"gpu_query": sys.argv[4]' in script
    assert "HOURLY_RATE_EVIDENCE" in script
    assert "BENCHMARK_GATE_SHA256" in script
    assert '--benchmark-gate "$BENCHMARK_GATE" --benchmark-gate-sha256 "$BENCHMARK_GATE_SHA256"' in script
    assert 'EXECUTION_KIND=(--run-kind "$RUN_MODE")' in script
    assert 'PYTHONPATH="$CODE_ROOT"' in script and 'git -C "$CODE_ROOT" clean -ndx' in script
    assert "PYTHONDONTWRITEBYTECODE=1" in script
    assert "--singularity-sha256" not in script
    assert "for PART_INDEX in $(seq 0 23)" in all_script
    assert "export PART_INDEX RUN_MODE=production" in all_script
    assert "t14_finalize_patchcore.py" in all_script
    assert "T14_GUARDED_ORCHESTRATION" in all_script
    assert "timeout --foreground --kill-after=30s" in all_script
    assert "budget_deadline_utc" in all_script


def test_runtime_rejects_changed_bootstrap_evidence(tmp_path: Path) -> None:
    artifact = _bootstrap_artifact(tmp_path)
    _bootstrap_provenance(artifact)
    Path(artifact["bootstrap_freeze"]).write_text("tampered\n")

    with pytest.raises(RunBoundaryError, match="bootstrap_freeze hash"):
        _bootstrap_provenance(artifact)


def test_runtime_requires_exact_managed_python(tmp_path: Path) -> None:
    executable = tmp_path / "bin/python"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\necho 'Python 3.10.12'\n")
    executable.chmod(0o755)

    with pytest.raises(RunBoundaryError, match="exact Python 3.11.18"):
        _python_identity(tmp_path)


def _rosie_environment(root: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    env = root / "env"
    (env / "bin").mkdir(parents=True)
    python = env / "bin/python"
    python.write_text(
        f'#!/bin/sh\nif [ "$1" = "-m" ]; then exit 0; fi\necho \'{json.dumps(ENV_VERSIONS, sort_keys=True)}\'\n'
    )
    python.chmod(0o755)
    freeze, audit = env / "requirements.freeze.txt", root / "audit.json"
    freeze.write_text("anomalib==2.7.0\n")
    audit.write_text("{}\n")
    artifact = root / "environment.json"
    payload: dict[str, object] = {
        "source_commit": "a" * 40,
        "runtime_sha256": "b" * 64,
        "env_dir": str(env.resolve()),
        "freeze_sha256": _sha(freeze),
        "pip_audit_sha256": _sha(audit),
        "singularity_image": "/data/containers/msoe-pytorch-23.05-py3.sif",
        **ENV_VERSIONS,
    }
    artifact.write_text(json.dumps(payload))
    (env / ".complete").write_text(f"{_sha(artifact)}\n")
    return env, artifact, audit, payload


def _rewrite_environment(env: Path, artifact: Path, payload: dict[str, object]) -> None:
    artifact.write_text(json.dumps(payload))
    (env / ".complete").write_text(f"{_sha(artifact)}\n")


def test_runner_preserves_existing_rosie_environment_support(tmp_path: Path) -> None:
    env, artifact, audit, _ = _rosie_environment(tmp_path)

    assert _environment(env, artifact, audit, "a" * 40, "b" * 64)["env_artifact_sha256"] == _sha(artifact)


@pytest.mark.parametrize("runtime_kind", ("brev_vm", "rosie", "", None))
def test_runner_rejects_runtime_kind_typo_bypass(tmp_path: Path, runtime_kind: object) -> None:
    env, artifact, audit, payload = _rosie_environment(tmp_path)
    payload["runtime_kind"] = runtime_kind
    _rewrite_environment(env, artifact, payload)

    with pytest.raises(RunBoundaryError, match="runtime kind is unsupported"):
        _environment(env, artifact, audit, "a" * 40, "b" * 64)


@pytest.mark.parametrize("singularity_image", ("", "/tmp/not-the-approved-image.sif"))
def test_runner_rejects_unapproved_legacy_rosie_image(tmp_path: Path, singularity_image: str) -> None:
    env, artifact, audit, payload = _rosie_environment(tmp_path)
    payload["singularity_image"] = singularity_image
    _rewrite_environment(env, artifact, payload)

    with pytest.raises(RunBoundaryError, match="invalid Singularity identity"):
        _environment(env, artifact, audit, "a" * 40, "b" * 64)


def test_runner_rejects_brev_runtime_kind_downgrade(tmp_path: Path) -> None:
    env, artifact, audit, payload = _rosie_environment(tmp_path)
    payload["uv"] = "0.13.6"
    _rewrite_environment(env, artifact, payload)

    with pytest.raises(RunBoundaryError, match="cannot be downgraded"):
        _environment(env, artifact, audit, "a" * 40, "b" * 64)


def test_benchmark_gate_measures_projection_and_revalidates_artifacts(tmp_path: Path) -> None:
    dataset, environment, benchmark, rate = _evidence(tmp_path)
    output = tmp_path / "gate.json"

    gate = create_gate(
        dataset=dataset,
        environment=environment,
        benchmark=benchmark,
        source_commit="a" * 40,
        runtime_sha256="b" * 64,
        hourly_rate_usd=0.85,
        hourly_rate_evidence=rate,
        output=output,
    )
    verified = verify_gate(output, dataset, environment, "a" * 40, "b" * 64)

    assert gate == verified
    assert gate["seconds_per_part"] == 30.0
    assert gate["projected_hours"] == 0.2
    assert gate["projected_cost_usd"] == pytest.approx(0.17)
    assert gate["hourly_rate_source"] == "brev search"
    assert gate["hourly_rate_evidence_sha256"] == _sha(rate)
    assert gate["cost_cap_usd"] == 120.0
    assert gate["budget_seconds_remaining_at_gate"] > 0
    assert gate["budget_deadline_utc"] > gate["gate_created_at_utc"]


def test_benchmark_gate_rejects_model_drift_and_nonfinite_json(tmp_path: Path) -> None:
    dataset, environment, benchmark, rate = _evidence(tmp_path)
    output = tmp_path / "gate.json"
    create_gate(benchmark, dataset, environment, "a" * 40, "b" * 64, 0.85, rate, output)
    payload = json.loads(benchmark.read_text())
    Path(payload["model_path"]).write_bytes(b"changed")

    with pytest.raises(BenchmarkGateError, match="missing or changed"):
        verify_gate(output, dataset, environment, "a" * 40, "b" * 64)

    output.write_text('{"projected_hours": NaN}')
    with pytest.raises(BenchmarkGateError, match="invalid JSON"):
        verify_gate(output, dataset, environment, "a" * 40, "b" * 64)


def test_benchmark_gate_rejects_edited_projection(tmp_path: Path) -> None:
    dataset, environment, benchmark, rate = _evidence(tmp_path)
    output = tmp_path / "gate.json"
    create_gate(benchmark, dataset, environment, "a" * 40, "b" * 64, 0.85, rate, output)
    payload = json.loads(output.read_text())
    payload["projected_cost_usd"] = 0.01
    output.write_text(json.dumps(payload))

    with pytest.raises(BenchmarkGateError, match="projection no longer matches"):
        verify_gate(output, dataset, environment, "a" * 40, "b" * 64)


def test_benchmark_gate_rejects_rate_drift_and_cost_over_cap(tmp_path: Path) -> None:
    dataset, environment, benchmark, rate = _evidence(tmp_path)
    output = tmp_path / "gate.json"
    rate_payload = json.loads(rate.read_text())
    rate_payload["hourly_rate_usd"] = 1_000.0
    rate.write_text(json.dumps(rate_payload))

    with pytest.raises(BenchmarkGateError, match="USD 120.00 cap"):
        create_gate(benchmark, dataset, environment, "a" * 40, "b" * 64, 1_000.0, rate, output)

    rate_payload["hourly_rate_usd"] = 0.84
    rate.write_text(json.dumps(rate_payload))
    with pytest.raises(BenchmarkGateError, match="hourly-rate"):
        create_gate(benchmark, dataset, environment, "a" * 40, "b" * 64, 0.85, rate, output)


@pytest.mark.parametrize(
    ("field", "value"),
    [("source_command", "brev ls"), ("captured_at_utc", "not-a-utc-timestamp")],
)
def test_benchmark_gate_requires_real_search_rate_provenance(tmp_path: Path, field: str, value: str) -> None:
    dataset, environment, benchmark, rate = _evidence(tmp_path)
    payload = json.loads(rate.read_text())
    payload[field] = value
    rate.write_text(json.dumps(payload))

    with pytest.raises(BenchmarkGateError, match="hourly-rate"):
        create_gate(
            benchmark,
            dataset,
            environment,
            "a" * 40,
            "b" * 64,
            0.85,
            rate,
            tmp_path / "gate.json",
        )


@pytest.mark.parametrize("offset", [timedelta(minutes=-16), timedelta(minutes=1)])
def test_benchmark_gate_rejects_stale_or_future_rate(tmp_path: Path, offset: timedelta) -> None:
    dataset, environment, benchmark, rate = _evidence(tmp_path)
    payload = json.loads(rate.read_text())
    payload["captured_at_utc"] = (datetime.now(UTC) + offset).strftime("%Y-%m-%dT%H:%M:%SZ")
    rate.write_text(json.dumps(payload))

    with pytest.raises(BenchmarkGateError, match="stale or future-dated"):
        create_gate(benchmark, dataset, environment, "a" * 40, "b" * 64, 0.85, rate, tmp_path / "gate.json")


def _runtime_gate(path: Path, deadline: datetime) -> Path:
    path.write_text(json.dumps({"budget_deadline_utc": deadline.strftime("%Y-%m-%dT%H:%M:%SZ")}))
    return path


def test_guarded_production_stops_exact_id_on_hung_run(tmp_path: Path) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    gate = _runtime_gate(tmp_path / "gate.json", now + timedelta(seconds=30))
    stopped: list[str] = []

    def hung(instance_id: str, remote_env: str, remote_code: str, timeout: float) -> None:
        assert (instance_id, remote_env, remote_code, timeout) == ("fisjik4qw", "/run.env", "/code", 30.0)
        raise subprocess.TimeoutExpired("brev exec", timeout)

    with pytest.raises(RuntimeGuardError, match="exact Brev instance was stopped"):
        run_guarded(
            instance_name="fa-t14",
            instance_id="fisjik4qw",
            gate_path=gate,
            remote_env="/run.env",
            remote_code_root="/code",
            list_instances=lambda: [{"name": "fa-t14", "id": "fisjik4qw"}],
            run_remote=hung,
            stop_instance=stopped.append,
            now=now,
        )

    assert stopped == ["fisjik4qw"]


def test_guarded_production_never_stops_unverified_id(tmp_path: Path) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    gate = _runtime_gate(tmp_path / "gate.json", now + timedelta(seconds=30))
    stopped: list[str] = []

    with pytest.raises(RuntimeGuardError, match="name-to-ID mapping"):
        run_guarded(
            instance_name="fa-t14",
            instance_id="wrongid",
            gate_path=gate,
            remote_env="/run.env",
            remote_code_root="/code",
            list_instances=lambda: [{"name": "fa-t14", "id": "fisjik4qw"}],
            run_remote=lambda *_: None,
            stop_instance=stopped.append,
            now=now,
        )

    assert stopped == []


def test_guarded_production_stops_verified_instance_when_budget_expired(tmp_path: Path) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    gate = _runtime_gate(tmp_path / "gate.json", now - timedelta(seconds=1))
    stopped: list[str] = []

    with pytest.raises(RuntimeGuardError, match="exact Brev instance was stopped"):
        run_guarded(
            instance_name="fa-t14",
            instance_id="fisjik4qw",
            gate_path=gate,
            remote_env="/run.env",
            remote_code_root="/code",
            list_instances=lambda: [{"name": "fa-t14", "id": "fisjik4qw"}],
            run_remote=lambda *_: pytest.fail("expired budget must not run production"),
            stop_instance=stopped.append,
            now=now,
        )

    assert stopped == ["fisjik4qw"]


def test_guarded_production_uses_brev_exec_command_string(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv: list[str], **kwargs: object) -> None:
        calls.append((argv, kwargs))

    monkeypatch.setattr(subprocess, "run", fake_run)
    _run_remote("fisjik4qw", "/remote/run env", "/remote/code root", 42.0)

    expected = (
        'bash -c \'set -a; source "$1"; set +a; export T14_GUARDED_ORCHESTRATION=1; '
        "exec \"$2/perception/brev/t14_run_all_patchcore.sh\"' t14 '/remote/run env' '/remote/code root'"
    )
    assert calls == [(["brev", "exec", "fisjik4qw", expected], {"check": True, "timeout": 42.0})]
