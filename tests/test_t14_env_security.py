from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "perception/rosie/t14_patchcore_env_smoke.sbatch"

SECURE_PINS = (
    "torch==2.14.1",
    "torchvision==0.29.1",
    "anomalib==2.7.0",
    "pillow==13.0.0",
    "filelock==4.2.4",
    "pip==26.2.2",
    "setuptools==84.2.1",
    "pip-audit==2.10.1",
)


def test_patchcore_environment_uses_the_measured_secure_dependency_set() -> None:
    script = SCRIPT.read_text()

    for requirement in SECURE_PINS:
        assert requirement in script
    assert "torch==2.10.0" not in script
    assert "torchvision==0.25.0" not in script


def test_patchcore_environment_approves_only_after_audit_and_v100_smoke() -> None:
    script = SCRIPT.read_text()

    assert script.index("pip check") < script.index("pip freeze --all")
    assert script.index('"$ENV_DIR/bin/pip-audit"') < script.index("torch.cuda.is_available()")
    assert script.index("torch.cuda.is_available()") < script.index('sha256sum "$ARTIFACT"')
    assert '"sm_70" not in torch.cuda.get_arch_list()' in script
    assert "freeze_sha256" in script
    assert "audit_requirements_sha256" in script
    assert "pip_audit_sha256" in script
    for package in ("pillow", "filelock", "pip", "setuptools", "pip-audit"):
        assert f'version("{package}")' in script


def test_patchcore_environment_binds_clean_source_and_exact_environment() -> None:
    script = SCRIPT.read_text()

    assert "CODE_ROOT:?set CODE_ROOT" in script
    assert 'git -C "$CODE_ROOT" rev-parse HEAD' in script
    assert "status --porcelain --untracked-files=all" in script
    assert '"env_dir": str(env_dir)' in script
    assert '"runner_id": sys.argv[6]' in script
    assert '"runtime_sha256": sys.argv[10]' in script
    assert 'sha256sum "$ARTIFACT"' in script
    assert script.index('sha256sum "$ARTIFACT"') < script.index("printf '%s\\n' \"$ENV_DIR\"")
