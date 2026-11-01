from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_patchcore_environment_smoke_is_pinned_and_v100_gated() -> None:
    script = (ROOT / "perception/rosie/t14_patchcore_env_smoke.sbatch").read_text()

    assert "#SBATCH --partition=dgx" in script
    assert "#SBATCH --gres=gpu:v100:1" in script
    assert "/data/containers/msoe-pytorch-23.05-py3.sif" in script
    assert "torch==2.14.1" in script
    assert "torchvision==0.29.1" in script
    assert "anomalib==2.7.0" in script
    assert "pillow==13.0.0" in script
    assert "filelock==4.2.4" in script
    assert "pip==26.2.2" in script
    assert "setuptools==84.2.1" in script
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
    assert script.index('test -s "$ARTIFACT"') < script.index('sha256sum "$ARTIFACT"')


def test_patchcore_preparation_job_gates_resources_and_inputs() -> None:
    script = (ROOT / "perception/rosie/t14_prepare_patchcore.sbatch").read_text()

    assert "#SBATCH --partition=teaching" in script
    assert "#SBATCH --gres" not in script
    assert "RUN_ROOT:?set RUN_ROOT" in script
    assert "NEW_DATA_ROOT:?set NEW_DATA_ROOT" in script
    assert "OLD_DATA_ROOT:?set OLD_DATA_ROOT" in script
    assert "REST_TOOL:?set REST_TOOL" in script
    assert "REST_GEOMETRY:?set REST_GEOMETRY" in script
    assert "CODE_ROOT:?set CODE_ROOT" in script
    assert "SOURCE_COMMIT:?set SOURCE_COMMIT" in script
    assert "RUN_ROOT,NEW_DATA_ROOT,OLD_DATA_ROOT,REST_TOOL,REST_GEOMETRY" in script
    assert "derive_rest_sidecars" in script
    assert "verify_rest_sidecars(rest_data_root, rest_summary)" in script
    assert "validate_patchcore_source_inventory" in script
    assert "source_inventory = validate_patchcore_source_inventory" in script
    assert "discover_patchcore_samples" in script
    assert 'defect_manifest=code_root / "cad/defects/manifest.json"' in script
    assert "prepare_patchcore_folders" in script
    assert 'item["object_id"] for item in taxonomy["objects"]' in script
    assert 'if item["object_id"].startswith("M")' in script
    assert "expected_object_ids=expected_object_ids" in script
    assert "normal_test_modulus" not in script
    assert 'test -d "$NEW_DATA_ROOT/O"' in script
    assert 'test -d "$NEW_DATA_ROOT/T"' in script
    assert 'test -d "$OLD_DATA_ROOT/C"' in script
    assert 'test -d "$OLD_DATA_ROOT/I"' in script


def test_patchcore_preparation_job_records_complete_provenance() -> None:
    script = (ROOT / "perception/rosie/t14_prepare_patchcore.sbatch").read_text()

    assert '"new_data_root": str(new_data_root)' in script
    assert '"old_data_root": str(old_data_root)' in script
    assert '"rest_data_root": str(rest_data_root)' in script
    assert '"rest_tool": str(rest_tool)' in script
    assert '"rest_geometry": str(rest_geometry)' in script
    assert '"rest_derivation": asdict(rest_summary)' in script
    assert '"source_inventory": source_inventory' in script
    assert '"defect_manifest_sha256": hashlib.sha256' in script
    assert '-B "$NEW_DATA_ROOT:$NEW_DATA_ROOT:ro"' in script
    assert '-B "$OLD_DATA_ROOT:$OLD_DATA_ROOT:ro"' in script
    assert '-B "$REST_TOOL:$REST_TOOL:ro"' in script
    assert '-B "$REST_GEOMETRY:$REST_GEOMETRY:ro"' in script
    argv = (
        '"$CODE_ROOT" "$NEW_DATA_ROOT" "$OLD_DATA_ROOT" "$REST_TOOL" \\\n'
        '    "$REST_GEOMETRY" "$REST_DATA_ROOT" "$OUTPUT_ROOT" "$ARTIFACT" \\\n'
        '    "$SLURM_JOB_ID" "$NODE_NAME" "$SOURCE_COMMIT" "$SIF_SHA256"'
    )
    assert argv in script
    assert "new_data_root = Path(sys.argv[2])" in script
    assert "old_data_root = Path(sys.argv[3])" in script
    assert "rest_data_root = Path(sys.argv[6])" in script
    assert '"source_commit": sys.argv[11]' in script
    assert '"runner_id": sys.argv[9]' in script
    assert '"preparation_runtime_sha256": sys.argv[12]' in script
    assert "b89e81c7bcc11f76ec59d88a8127d1997bf5871d849037346d3f0b7084e2d10f" in script
    assert "bf98f0473e9c3e5a5a9fbb1bae801434c5085b094b56d28eb03860386245d91c" in script
    assert "attempts/patchcore-${SLURM_JOB_ID}" in script
    assert "rest/attempts/rest-${SLURM_JOB_ID}" in script
    assert 'test ! -e "$OUTPUT_ROOT/_INCOMPLETE"' in script
    assert 'test -s "$OUTPUT_ROOT/_READY.json"' in script
    assert "patchcore_prepare-${SLURM_JOB_ID}.json" in script
