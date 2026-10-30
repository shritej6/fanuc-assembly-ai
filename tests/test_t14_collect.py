from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from perception.brev.t14_collect_and_shutdown import (
    PARTS,
    CollectionError,
    _brev_resolve_instance_id,
    collect_and_shutdown,
)

INSTANCE_ID = "fisjik4qw"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _environment_evidence(remote: dict[str, bytes]) -> tuple[bytes, bytes]:
    env_dir = "/remote/env"
    platform = json.dumps({"brev_instance_id": INSTANCE_ID, "brev_machine_type": "l4"}).encode()
    remote["/remote/platform.json"] = platform
    rate = json.dumps({"source_command": "brev search", "hourly_rate_usd": 0.85}).encode()
    remote["/remote/rate.json"] = rate
    env = {
        "env_dir": env_dir,
        "platform_identity": "/remote/platform.json",
        "platform_identity_sha256": _sha(platform),
    }
    freeze, audit_requirements, audit = b"frozen\n", b"audit requirements\n", b"{}"
    env.update(
        freeze_sha256=_sha(freeze),
        audit_requirements_sha256=_sha(audit_requirements),
        pip_audit_sha256=_sha(audit),
    )
    env_bytes = json.dumps(env).encode()
    remote["/remote/env.json"] = env_bytes
    remote["/remote/audit.json"] = audit
    remote[f"{env_dir}/requirements.freeze.txt"] = freeze
    remote[f"{env_dir}/requirements.audit.txt"] = audit_requirements
    remote[f"{env_dir}/.complete"] = f"{_sha(env_bytes)}\n".encode()
    return env_bytes, audit


def _dataset_evidence(remote: dict[str, bytes]) -> bytes:
    dataset_dir = "/remote/dataset"
    ready, manifest = b"{}", b'{"row":1}\n'
    dataset = {"ready_sha256": _sha(ready), "manifest_sha256": _sha(manifest)}
    dataset_bytes = json.dumps(dataset).encode()
    remote["/remote/dataset.json"] = dataset_bytes
    remote[f"{dataset_dir}/_READY.json"] = ready
    remote[f"{dataset_dir}/manifest.jsonl"] = manifest
    return dataset_bytes


def _benchmark_evidence(remote: dict[str, bytes]) -> bytes:
    benchmark_metrics, benchmark_model = b"{}", b"benchmark-model"
    remote["/remote/benchmark_metrics.json"] = benchmark_metrics
    remote["/remote/benchmark_model.ckpt"] = benchmark_model
    benchmark = {
        "metrics_path": "/remote/benchmark_metrics.json",
        "metrics_sha256": _sha(benchmark_metrics),
        "model_path": "/remote/benchmark_model.ckpt",
        "model_sha256": _sha(benchmark_model),
    }
    benchmark_bytes = json.dumps(benchmark).encode()
    remote["/remote/benchmark.json"] = benchmark_bytes
    gate = {
        "benchmark_artifact": "/remote/benchmark.json",
        "benchmark_artifact_sha256": _sha(benchmark_bytes),
        "hourly_rate_evidence": "/remote/rate.json",
        "hourly_rate_evidence_sha256": _sha(remote["/remote/rate.json"]),
    }
    gate_bytes = json.dumps(gate).encode()
    remote["/remote/gate.json"] = gate_bytes
    return gate_bytes


def _run_evidence(remote: dict[str, bytes]) -> list[dict[str, str]]:
    env_dir, dataset_dir = "/remote/env", "/remote/dataset"
    runs = []
    for part in PARTS:
        metrics, model = json.dumps({"part": part}).encode(), f"model-{part}".encode()
        artifact = {
            "part": part,
            "dataset_dir": dataset_dir,
            "env_dir": env_dir,
            "metrics_path": f"/remote/{part}/metrics.json",
            "metrics_sha256": _sha(metrics),
            "model_path": f"/remote/{part}/model.ckpt",
            "model_sha256": _sha(model),
        }
        artifact_bytes = json.dumps(artifact).encode()
        remote[f"/remote/{part}/artifact.json"] = artifact_bytes
        remote[f"/remote/{part}/metrics.json"] = metrics
        remote[f"/remote/{part}/model.ckpt"] = model
        runs.append(
            {
                **artifact,
                "artifact_path": f"/remote/{part}/artifact.json",
                "artifact_sha256": _sha(artifact_bytes),
            }
        )
    return runs


def _arguments(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        instance_name="fa-t14",
        instance_id=INSTANCE_ID,
        confirm_delete="fa-t14",
        summary_remote="/remote/summary.json",
        dataset_artifact_remote="/remote/dataset.json",
        env_artifact_remote="/remote/env.json",
        env_audit_remote="/remote/audit.json",
        benchmark_gate_remote="/remote/gate.json",
        destination=tmp_path / "offhost",
    )


def _remote_fixture(tmp_path: Path) -> tuple[dict[str, bytes], argparse.Namespace]:
    remote: dict[str, bytes] = {}
    env_bytes, audit = _environment_evidence(remote)
    dataset_bytes = _dataset_evidence(remote)
    gate_bytes = _benchmark_evidence(remote)
    runs = _run_evidence(remote)
    summary = {
        "status": "approved",
        "runs": runs,
        "dataset_artifact_sha256": _sha(dataset_bytes),
        "env_artifact_sha256": _sha(env_bytes),
        "env_audit_sha256": _sha(audit),
        "env_complete_sha256": _sha(remote["/remote/env/.complete"]),
        "benchmark_gate_sha256": _sha(gate_bytes),
    }
    remote["/remote/summary.json"] = json.dumps(summary).encode()
    return remote, _arguments(tmp_path)


def _copy_from(remote: dict[str, bytes]):
    def copy(instance: str, source: str, destination: Path) -> None:
        assert instance == "fa-t14"
        destination.write_bytes(remote[source])

    return copy


def _resolve_fa_t14(instance_name: str) -> str:
    assert instance_name == "fa-t14"
    return INSTANCE_ID


def test_collection_verifies_all_evidence_before_deleting(tmp_path: Path) -> None:
    remote, args = _remote_fixture(tmp_path)
    actions = []

    result = collect_and_shutdown(
        args,
        copy_file=_copy_from(remote),
        lifecycle=lambda action, instance: actions.append((action, instance)),
        resolve_instance=_resolve_fa_t14,
    )

    assert result == args.destination
    assert actions == [("delete", INSTANCE_ID)]
    verified = json.loads((result / "OFFHOST_VERIFIED.json").read_text())
    manifest = json.loads((result / "OFFHOST_MANIFEST.json").read_text())
    assert verified["status"] == "verified"
    assert verified["file_count"] == 87
    assert len(manifest["files"]) == 87
    assert len(list(result.glob("runs/*/model.ckpt"))) == 24


def test_hash_failure_preserves_partial_evidence_and_stops_instance(tmp_path: Path) -> None:
    remote, args = _remote_fixture(tmp_path)
    remote["/remote/M1_01/model.ckpt"] = b"tampered"
    actions = []

    with pytest.raises(CollectionError, match="hash mismatch"):
        collect_and_shutdown(
            args,
            copy_file=_copy_from(remote),
            lifecycle=lambda action, instance: actions.append((action, instance)),
            resolve_instance=_resolve_fa_t14,
        )

    assert actions == [("stop", INSTANCE_ID)]
    partial = args.destination.with_name("offhost.partial")
    assert (partial / "COLLECTION_FAILED.json").is_file()
    assert not args.destination.exists()


def test_collector_refuses_any_instance_except_exact_fa_t14(tmp_path: Path) -> None:
    remote, args = _remote_fixture(tmp_path)
    args.instance_name = "fa-sim"
    actions = []

    with pytest.raises(CollectionError, match="exact fa-t14"):
        collect_and_shutdown(
            args,
            copy_file=_copy_from(remote),
            lifecycle=lambda action, instance: actions.append((action, instance)),
            resolve_instance=_resolve_fa_t14,
        )

    assert actions == []
    assert not args.destination.exists()


def test_delete_failure_preserves_verified_copy_and_stops_instance(tmp_path: Path) -> None:
    remote, args = _remote_fixture(tmp_path)
    actions = []

    def lifecycle(action: str, instance: str) -> None:
        actions.append((action, instance))
        if action == "delete":
            raise CollectionError("delete failed")

    with pytest.raises(CollectionError, match="delete failed"):
        collect_and_shutdown(
            args,
            copy_file=_copy_from(remote),
            lifecycle=lifecycle,
            resolve_instance=_resolve_fa_t14,
        )

    assert actions == [("delete", INSTANCE_ID), ("stop", INSTANCE_ID)]
    assert (args.destination / "OFFHOST_VERIFIED.json").is_file()
    assert (args.destination / "COLLECTION_FAILED.json").is_file()


def test_live_instance_id_mismatch_cannot_copy_or_stop(tmp_path: Path) -> None:
    _remote, args = _remote_fixture(tmp_path)
    copies = []
    actions = []

    def copy(instance: str, source: str, destination: Path) -> None:
        copies.append((instance, source, destination))
        raise OSError("copy must not be attempted")

    with pytest.raises(CollectionError, match="live fa-t14"):
        collect_and_shutdown(
            args,
            copy_file=copy,
            lifecycle=lambda action, instance: actions.append((action, instance)),
            resolve_instance=lambda _name: "wronginstance",
        )

    assert copies == []
    assert actions == []
    assert not args.destination.exists()


def test_brev_resolver_accepts_measured_workspaces_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    command = ["brev", "ls", "--json"]
    payload = {"workspaces": [{"name": "fa-t14", "id": INSTANCE_ID, "status": "RUNNING"}]}
    result = subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(
        "perception.brev.t14_collect_and_shutdown.subprocess.run",
        lambda *args, **kwargs: result,
    )

    assert _brev_resolve_instance_id("fa-t14") == INSTANCE_ID


@pytest.mark.parametrize("payload", [[], {"instances": []}, {"workspaces": "fa-t14"}])
def test_brev_resolver_rejects_unsupported_json_shapes(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
) -> None:
    result = subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")
    monkeypatch.setattr(
        "perception.brev.t14_collect_and_shutdown.subprocess.run",
        lambda *args, **kwargs: result,
    )

    with pytest.raises(CollectionError, match="workspaces list"):
        _brev_resolve_instance_id("fa-t14")
