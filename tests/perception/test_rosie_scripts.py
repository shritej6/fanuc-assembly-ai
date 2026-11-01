"""Behavior tests for the guarded Rosie launch scripts.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from perception.rosie.gates import RosieGateError, verify_production_manifest
from perception.rosie.prepare import preparation_plan

REPO = Path(__file__).resolve().parents[2]
ROSIE = REPO / "perception" / "rosie"


def _run(script: str, tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "REPO": str(REPO),
        "DATA_ROOT": str(tmp_path / "fanuc-assembly-data"),
        "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "RUN_ROOT": str(tmp_path / "runs"),
        "PYTHON": sys.executable,
        "DRY_RUN": "1",
        **overrides,
    }
    return subprocess.run(
        ["bash", str(ROSIE / script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
    )


def _metadata(path: Path, *, require_cosmos: bool) -> None:
    sets = {
        name: {"scenes": 50, "native_records": 100, "cosmos_accepted": 10, "cosmos_used": 8}
        for name in ("A", "F", "G", "B", "H", "K")
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "content_hash_algorithm": "sha256",
                "require_cosmos": require_cosmos,
                "include_cosmos": require_cosmos,
                "sets": sets,
            }
        )
    )


def test_setup_replays_hash_locked_cuda_126_runtime_without_pip_cache(tmp_path: Path) -> None:
    result = _run("setup_ml.sh", tmp_path, VENV_PATH=str(tmp_path / "venv"))
    lock = (ROSIE / "requirements-ml-cu126.txt").read_text()

    assert result.returncode == 0, result.stderr
    assert "-m venv --clear" in result.stdout
    assert "--require-hashes" in result.stdout
    assert "--only-binary=:all:" in result.stdout
    assert "--no-cache-dir" in result.stdout
    assert "--no-index" in result.stdout
    assert "--index-url" not in result.stdout
    assert "--extra-index-url" not in result.stdout
    assert "torch-2.14.1%2Bcu126" in lock
    assert "torchvision-0.29.1%2Bcu126" in lock
    assert "transformers-5.21.0" in lock
    assert "--hash=sha256:" in lock
    assert not (tmp_path / "venv").exists()


def test_rosie_runtime_lock_has_no_unhashed_requirement_lines() -> None:
    lock = (ROSIE / "requirements-ml-cu126.txt").read_text()
    logical_lines = lock.replace("\\\n", " ").splitlines()
    requirements = [
        line
        for line in logical_lines
        if line.strip() and not line.lstrip().startswith(("#", "--"))
    ]

    assert requirements
    artifact = re.compile(
        r"^[a-z0-9][a-z0-9._-]* @ https://[^ ]+ --hash=sha256:[0-9a-f]{64}$"
    )
    assert all(artifact.fullmatch(line) for line in requirements)


def test_runtime_scripts_load_the_archived_commit_identity() -> None:
    for name in ("prepare.sbatch", "detector.sbatch", "classifier.sbatch"):
        content = (ROSIE / name).read_text()
        assert ".t13-code-commit" in content
        assert "FANUC_CODE_COMMIT" in content


@pytest.mark.parametrize("script", ["detector.sbatch", "classifier.sbatch"])
def test_slurm_launch_resolves_repo_from_submit_directory(
    script: str,
    tmp_path: Path,
) -> None:
    archived_repo = tmp_path / "fanuc-t13-code"
    archived_repo.mkdir()

    result = _run(
        script,
        tmp_path,
        REPO="",
        SLURM_SUBMIT_DIR=str(archived_repo),
        SMOKE="1",
    )

    assert result.returncode == 0, result.stderr
    assert str(archived_repo / "perception" / "configs") in result.stdout


@pytest.mark.parametrize(
    ("script", "config"),
    [("detector.sbatch", "rtdetr_h100.yaml"), ("classifier.sbatch", "classifier_h100.yaml")],
)
def test_no_slurm_launch_accepts_an_explicit_h100_config(
    script: str,
    config: str,
    tmp_path: Path,
) -> None:
    path = REPO / "perception" / "configs" / config

    result = _run(script, tmp_path, SMOKE="1", CONFIG=str(path))

    assert result.returncode == 0, result.stderr
    assert f"config={path}" in result.stdout
    assert f"--config {path}" in result.stdout
    assert "--nproc_per_node 1" in result.stdout


def test_preparation_plan_allows_native_only_only_for_smoke(tmp_path: Path) -> None:
    smoke = preparation_plan(tmp_path / "data", tmp_path / "smoke", REPO / "contracts/taxonomy.json", True)
    full = preparation_plan(tmp_path / "data", tmp_path / "full", REPO / "contracts/taxonomy.json", False)

    assert smoke["require_cosmos"] is False
    assert smoke["include_cosmos"] is False
    assert smoke["max_scenes_per_set"] == 128
    assert full["require_cosmos"] is True
    assert full["include_cosmos"] is True
    assert full["max_scenes_per_set"] is None
    assert smoke["detector_sets"] == ["A"]
    assert smoke["classifier_sets"] == ["B"]


@pytest.mark.parametrize(
    ("script", "data_flag"),
    [("detector.sbatch", "--coco-dir"), ("classifier.sbatch", "--crops")],
)
def test_smoke_launch_is_one_gpu_two_steps_and_resumable(
    script: str,
    data_flag: str,
    tmp_path: Path,
) -> None:
    initial = _run(script, tmp_path, SMOKE="1", SLURM_GPUS_ON_NODE="1")
    resumed = _run(
        script,
        tmp_path,
        SMOKE="1",
        MAX_STEPS="3",
        RESUME_FROM_CHECKPOINT="latest",
        SLURM_GPUS_ON_NODE="1",
    )

    assert initial.returncode == 0, initial.stderr
    assert "--partition=teaching" in (ROSIE / script).read_text()
    assert "--gres=gpu:t4:1" in (ROSIE / script).read_text()
    assert "--nproc_per_node 1" in initial.stdout
    assert "--smoke" in initial.stdout
    assert "--max-steps 2" in initial.stdout
    assert data_flag in initial.stdout
    assert "--max-steps 3" in resumed.stdout
    assert "--resume-from-checkpoint latest" in resumed.stdout


def test_full_launch_rejects_native_only_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "artifacts" / "full" / "manifests" / "metadata.json"
    _metadata(metadata, require_cosmos=False)

    result = _run("detector.sbatch", tmp_path, SMOKE="0")

    assert result.returncode != 0
    assert "Cosmos" in result.stderr


def test_full_manifest_gate_rejects_out_of_range_set(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    _metadata(metadata, require_cosmos=True)
    payload = json.loads(metadata.read_text())
    payload["sets"]["G"]["cosmos_accepted"] = 9
    metadata.write_text(json.dumps(payload))

    with pytest.raises(RosieGateError, match="G"):
        verify_production_manifest(metadata)


def test_full_manifest_gate_uses_native_image_records_as_denominator(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.json"
    _metadata(metadata, require_cosmos=True)
    payload = json.loads(metadata.read_text())
    payload["sets"]["A"].update({"scenes": 10, "native_records": 20, "cosmos_accepted": 1})
    metadata.write_text(json.dumps(payload))

    with pytest.raises(RosieGateError, match="A"):
        verify_production_manifest(metadata)


@pytest.mark.parametrize(
    "invalid",
    [
        {"schema_version": 1, "content_hash_algorithm": "sha256"},
        {"schema_version": 2},
    ],
)
def test_full_manifest_gate_requires_pixel_bound_schema(
    tmp_path: Path,
    invalid: dict[str, object],
) -> None:
    metadata = tmp_path / "metadata.json"
    _metadata(metadata, require_cosmos=True)
    payload = json.loads(metadata.read_text())
    payload.pop("content_hash_algorithm")
    payload.update(invalid)
    metadata.write_text(json.dumps(payload))

    with pytest.raises(RosieGateError, match="schema"):
        verify_production_manifest(metadata)


def test_full_launch_uses_frozen_cosmos_artifacts(tmp_path: Path) -> None:
    metadata = tmp_path / "artifacts" / "full" / "manifests" / "metadata.json"
    _metadata(metadata, require_cosmos=True)

    result = _run("classifier.sbatch", tmp_path, SMOKE="0")

    assert result.returncode == 0, result.stderr
    assert "--smoke" not in result.stdout
    assert "--max-steps" not in result.stdout
    assert str(metadata) in result.stdout
