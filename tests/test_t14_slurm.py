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
    assert "https://download.pytorch.org/whl/cu126" in script
    assert "pip check" in script
    assert "--format=json" in script
    assert "--strict" in script
    assert 'get_device_capability() != (7, 0)' in script
    assert '"sm_70" not in torch.cuda.get_arch_list()' in script
    assert "from anomalib.models import Patchcore" in script
    assert 'SOURCE_COMMIT:?set SOURCE_COMMIT' in script
    assert 'sha256sum "$SIF"' in script
    assert "freeze_sha256" in script
    assert 'attempts/patchcore-${SLURM_JOB_ID}' in script
    assert 'patchcore_env_smoke-${SLURM_JOB_ID}.json' in script
    assert 'pip_audit-${SLURM_JOB_ID}.json' in script
    assert script.index('test -s "$ARTIFACT"') < script.index('touch "$ENV_DIR/.complete"')
