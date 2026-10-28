from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_patchcore_environment_smoke_is_pinned_and_v100_gated() -> None:
    script = (ROOT / "perception/rosie/t14_patchcore_env_smoke.sbatch").read_text()

    assert "#SBATCH --partition=dgx" in script
    assert "#SBATCH --gres=gpu:v100:1" in script
    assert "/data/containers/msoe-pytorch-23.05-py3.sif" in script
    assert "torch==2.10.0" in script
    assert "torchvision==0.25.0" in script
    assert "anomalib==2.7.0" in script
    assert "pip-audit==2.10.1" in script
    assert "export PIP_CONFIG_FILE=/dev/null" in script
    assert "unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_NO_INDEX" in script
    assert "unset PIP_REQUIREMENT PIP_CONSTRAINT PIP_TRUSTED_HOST" in script
    assert "https://download.pytorch.org/whl/cu126" in script
    assert "--index-url https://pypi.org/simple" in script
    assert "pip check" in script
    assert "pip freeze --all" in script
    assert "requirements.audit.txt" in script
    assert "Version(raw_version).public" in script
    assert "--no-deps --disable-pip" in script
    assert "audit_requirements_sha256" in script
    assert "--format=json" in script
    assert "--strict" in script
    assert "get_device_capability() != (7, 0)" in script
    assert '"sm_70" not in torch.cuda.get_arch_list()' in script
    assert "from anomalib.models import Patchcore" in script
    assert "SOURCE_COMMIT:?set SOURCE_COMMIT" in script
    assert 'sha256sum "$SIF"' in script
    assert "freeze_sha256" in script
    assert "attempts/patchcore-${SLURM_JOB_ID}" in script
    assert "patchcore_env_smoke-${SLURM_JOB_ID}.json" in script
    assert "pip_audit-${SLURM_JOB_ID}.json" in script
    assert script.index('test -s "$ARTIFACT"') < script.index('touch "$ENV_DIR/.complete"')


def test_patchcore_preparation_job_is_cpu_only_and_contract_complete() -> None:
    script = (ROOT / "perception/rosie/t14_prepare_patchcore.sbatch").read_text()

    assert "#SBATCH --partition=teaching" in script
    assert "#SBATCH --gres" not in script
    assert "RUN_ROOT:?set RUN_ROOT" in script
    assert "DATA_ROOT:?set DATA_ROOT" in script
    assert "CODE_ROOT:?set CODE_ROOT" in script
    assert "SOURCE_COMMIT:?set SOURCE_COMMIT" in script
    assert "#   export RUN_ROOT DATA_ROOT CODE_ROOT SOURCE_COMMIT" in script
    assert "discover_patchcore_samples" in script
    assert "prepare_patchcore_folders" in script
    assert 'item["object_id"] for item in taxonomy["objects"]' in script
    assert 'if item["object_id"].startswith("M")' in script
    assert "expected_object_ids=expected_object_ids" in script
    assert "normal_test_modulus=5" in script
    assert "attempts/patchcore-${SLURM_JOB_ID}" in script
    assert 'test ! -e "$OUTPUT_ROOT/_INCOMPLETE"' in script
    assert 'test -s "$OUTPUT_ROOT/_READY.json"' in script
    assert "patchcore_prepare-${SLURM_JOB_ID}.json" in script
