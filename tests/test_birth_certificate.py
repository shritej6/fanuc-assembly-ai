"""Behavioral coverage for immutable JSON/PDF product traceability."""

from __future__ import annotations

import hashlib
import json
import math
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from pypdf import PdfReader

import assembly.birth_certificate as birth_certificate_module
import assembly.birth_certificate_artifacts as artifact_module
from assembly.birth_certificate import (
    BirthCertificateInputError,
    BirthCertificateWriteError,
    CertificateArtifacts,
    ProductOutcome,
    ProductResult,
    StepInspectionEvidence,
    build_birth_certificate,
    validate_birth_certificate,
    write_birth_certificate,
)
from assembly.catalog import load_catalog
from assembly.controller_ports import ControllerEvent
from assembly.planner import ContractPoseUF0, PlanStepRecord


def _plan(product: int, step: int, instance_id: str, attempt: int = 1) -> PlanStepRecord:
    pose = ContractPoseUF0(300.0 + step, -40.0, -300.0, 90.0)
    return PlanStepRecord(
        step_id=f"product-{product}-step-{step:02d}-attempt-{attempt}",
        product=product,
        instance_id=instance_id,
        object_id=f"M{product}_{step:02d}",
        grasp=pose,
        target=pose,
        gripper_roll_deg=0.0,
        attempt=attempt,
    )


def _inspection(
    product: int,
    step: int,
    instance_id: str,
    *,
    passed: bool = True,
    image_ref: str | None = None,
) -> StepInspectionEvidence:
    object_id = f"M{product}_{step:02d}"
    inspection = {
        "instance_id": instance_id,
        "object_id": object_id,
        "verdict": "ok" if passed else "print_defect",
        "action": "use" if passed else "reject",
        "checks_agree": passed,
        "evidence": {
            "classifier_score": 0.97,
            "patchcore_score": 0.08,
            "depth_residual_mm": 0.21,
            "image_ref": image_ref or f"images/product-{product}/step-{step}.png",
        },
    }
    return StepInspectionEvidence(product, instance_id, object_id, inspection, passed, passed)


def _first_pass_inputs(product: int = 1):
    plans = tuple(_plan(product, step, f"part-{product}-{step}") for step in range(1, 5))
    inspections = tuple(_inspection(product, step, f"part-{product}-{step}") for step in range(1, 5))
    events = tuple(
        event
        for step in range(1, 5)
        for event in (
            ControllerEvent("motion", product, f"M{product}_{step:02d}", "guarded"),
            ControllerEvent("installed", product, f"M{product}_{step:02d}", "verified"),
        )
    ) + (ControllerEvent("complete", product, None, "four verified steps"),)
    return plans, inspections, events


def _build_good(product: int = 1) -> dict[str, object]:
    plans, inspections, events = _first_pass_inputs(product)
    return build_birth_certificate(
        catalog=load_catalog(),
        run_id="run-2026-10-29",
        rollout=3,
        product=product,
        base_instance=f"base-{product}",
        plan_records=plans,
        events=events,
        inspections=inspections,
        outcome=ProductOutcome(ProductResult.GOOD_FIRST_PASS, True, True, 100.0, 112.5),
    )


def test_builds_contract_record_from_t09_evidence() -> None:
    record = _build_good()

    assert record["result"] == "good_first_pass"
    assert record["cycle_time_s"] == 12.5
    assert record["rebuilds"] == 0
    assert record["assists"] == 0
    assert [step["object_id"] for step in record["steps"]] == [f"M1_{step:02d}" for step in range(1, 5)]
    assert all(step["attempts"] == 1 for step in record["steps"])
    assert all(step["spare_used"] is False for step in record["steps"])
    assert all(step["seated_check"] == "passed" for step in record["steps"])


def test_writes_immutable_json_and_single_page_pdf(tmp_path: Path) -> None:
    record = _build_good()

    artifacts = write_birth_certificate(record, output_root=tmp_path / "records")

    artifact_dir = tmp_path / "records" / "run-2026-10-29" / "rollout-000003" / "product-1"
    assert artifacts.json_path == artifact_dir / "certificate.json"
    assert artifacts.pdf_path == artifact_dir / "certificate.pdf"
    assert artifacts.manifest_path == artifact_dir / "manifest.json"
    assert json.loads(artifacts.json_path.read_text()) == record
    assert artifacts.pdf_path.read_bytes().startswith(b"%PDF-")
    canonical = json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    canonical_digest = hashlib.sha256(canonical).hexdigest()
    manifest = json.loads(artifacts.manifest_path.read_text())
    assert manifest["canonical_json_sha256"] == canonical_digest
    assert manifest["pdf_sha256"] == hashlib.sha256(artifacts.pdf_path.read_bytes()).hexdigest()
    assert birth_certificate_module.verify_birth_certificate_artifacts(artifacts) == record
    for path in (artifacts.json_path, artifacts.pdf_path, artifacts.manifest_path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert stat.S_IMODE(artifact_dir.stat().st_mode) == 0o555
    reader = PdfReader(artifacts.pdf_path)
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    assert "PRODUCT BIRTH CERTIFICATE" in text
    assert "Product 1" in text
    assert "M1_01" in text
    assert "M1_04" in text
    assert canonical_digest in text.replace(" ", "").replace("\n", "")
    assert "Page 1 of 1" in text
    with pytest.raises(BirthCertificateWriteError, match="immutable"):
        write_birth_certificate(record, output_root=tmp_path / "records")


def test_rollout_is_part_of_artifact_identity(tmp_path: Path) -> None:
    first = _build_good()
    second = dict(first, rollout=4)

    first_artifacts = write_birth_certificate(first, output_root=tmp_path)
    second_artifacts = write_birth_certificate(second, output_root=tmp_path)

    assert first_artifacts.json_path != second_artifacts.json_path
    assert "rollout-000003" in first_artifacts.json_path.parts
    assert "rollout-000004" in second_artifacts.json_path.parts


def test_verifier_detects_modified_json_or_pdf(tmp_path: Path) -> None:
    artifacts = write_birth_certificate(_build_good(), output_root=tmp_path)
    artifacts.json_path.chmod(0o644)
    artifacts.json_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(BirthCertificateWriteError, match="JSON digest"):
        birth_certificate_module.verify_birth_certificate_artifacts(artifacts)

    artifacts = write_birth_certificate(dict(_build_good(), rollout=4), output_root=tmp_path)
    artifacts.pdf_path.chmod(0o644)
    artifacts.pdf_path.write_bytes(artifacts.pdf_path.read_bytes() + b"tampered")

    with pytest.raises(BirthCertificateWriteError, match="PDF digest"):
        birth_certificate_module.verify_birth_certificate_artifacts(artifacts)


def test_bundle_publish_leaves_no_visible_partial_artifact_on_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_commit(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("injected directory commit failure")

    monkeypatch.setattr(artifact_module.os, "rename", fail_commit)

    with pytest.raises(BirthCertificateWriteError, match="publish certificate artifact bundle"):
        write_birth_certificate(_build_good(), output_root=tmp_path)

    assert not (tmp_path / "run-2026-10-29" / "rollout-000003" / "product-1").exists()


def test_controller_import_does_not_require_pdf_dependencies() -> None:
    code = """
import builtins
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.startswith((\"reportlab\", \"PIL\")):
        raise AssertionError(f\"unexpected PDF dependency import: {name}\")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import assembly.state_machine
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_validation_rejects_more_than_four_product_steps() -> None:
    record = _build_good()
    record["steps"] = [*record["steps"], record["steps"][0]]

    with pytest.raises(BirthCertificateInputError, match="at most four"):
        validate_birth_certificate(record)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rollout", -1, "rollout"),
        ("rollout", True, "rollout"),
        ("rollout", 1.5, "rollout"),
        ("product", 0, "product"),
        ("product", 7, "product"),
        ("product", True, "product"),
        ("product", 1.5, "product"),
        ("base_instance", "", "base_instance"),
        ("base_instance", 42, "base_instance"),
    ],
)
def test_public_validation_enforces_domain_identity(field: str, value: object, message: str) -> None:
    record = dict(_build_good(), **{field: value})

    with pytest.raises(BirthCertificateInputError, match=message):
        validate_birth_certificate(record)


def test_writer_enforces_domain_identity_before_creating_directories(tmp_path: Path) -> None:
    record = dict(_build_good(), rollout=-1)

    with pytest.raises(BirthCertificateInputError, match="rollout"):
        write_birth_certificate(record, output_root=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_verifier_revalidates_identity_after_digest_checks(tmp_path: Path) -> None:
    artifacts = write_birth_certificate(_build_good(), output_root=tmp_path)
    invalid = dict(_build_good(), product=7)
    canonical = json.dumps(invalid, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    artifacts.json_path.chmod(0o644)
    artifacts.manifest_path.chmod(0o644)
    artifacts.json_path.write_text(json.dumps(invalid, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads(artifacts.manifest_path.read_text())
    manifest["canonical_json_sha256"] = hashlib.sha256(canonical).hexdigest()
    artifacts.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(BirthCertificateInputError, match="product"):
        birth_certificate_module.verify_birth_certificate_artifacts(artifacts)


@pytest.mark.parametrize(
    "artifacts",
    [
        CertificateArtifacts("certificate.json", Path("certificate.pdf"), Path("manifest.json")),  # type: ignore[arg-type]
        CertificateArtifacts(Path("wrong.json"), Path("certificate.pdf"), Path("manifest.json")),
        CertificateArtifacts(Path("a/certificate.json"), Path("b/certificate.pdf"), Path("a/manifest.json")),
    ],
)
def test_verifier_rejects_malformed_artifact_paths(artifacts: CertificateArtifacts) -> None:
    with pytest.raises(BirthCertificateWriteError, match="artifact paths"):
        birth_certificate_module.verify_birth_certificate_artifacts(artifacts)


def test_rebuild_tracks_all_attempts_spare_and_assist_episodes() -> None:
    plans, inspections, events = _first_pass_inputs()
    replacement = _plan(1, 2, "spare-1-2", attempt=1)
    plans = (plans[0], plans[1], _plan(1, 2, "part-1-2", attempt=2), replacement, *plans[2:])
    inspections = (*inspections, _inspection(1, 2, "spare-1-2"))
    events = (
        ControllerEvent("retry", 1, "M1_02", "1"),
        ControllerEvent("andon", 1, "M1_02", "operator reconciliation"),
        ControllerEvent("andon", 1, "M1_02", "operator reconciliation"),
        ControllerEvent("resume", 1, None, "inventory restored"),
        ControllerEvent("rebuild", 1, "M1_02", "started"),
        ControllerEvent("reject_failed", 1, "M1_02", "spare-1-2"),
        ControllerEvent("rebuild", 1, None, "verified complete"),
        *events,
    )

    record = build_birth_certificate(
        catalog=load_catalog(),
        run_id="rebuild-001",
        rollout=0,
        product=1,
        base_instance="base-1",
        plan_records=plans,
        events=events,
        inspections=inspections,
        outcome=ProductOutcome(ProductResult.GOOD_AFTER_REBUILD, True, True),
    )

    step = record["steps"][1]
    assert step["instance_id"] == "spare-1-2"
    assert step["attempts"] == 3
    assert step["spare_used"] is True
    assert record["rebuilds"] == 1
    assert record["assists"] == 1
    assert record["cycle_time_s"] is None


def test_blocked_record_keeps_failed_partial_step_and_deduplicates_andon() -> None:
    plan = _plan(2, 1, "part-2-1")
    events = (
        ControllerEvent("andon", 2, "M2_01", "checks disagree"),
        ControllerEvent("andon", 2, "M2_01", "checks disagree"),
    )

    record = build_birth_certificate(
        catalog=load_catalog(),
        run_id="blocked-001",
        rollout=1,
        product=2,
        base_instance="base-2",
        plan_records=(plan,),
        events=events,
        inspections=(_inspection(2, 1, "part-2-1", passed=False),),
        outcome=ProductOutcome(ProductResult.BLOCKED, False, None),
    )

    assert record["result"] == "blocked"
    assert record["assists"] == 1
    assert record["steps"][0]["seated_check"] == "failed"
    assert record["final_checks"] == {"recipe": False, "cosmos_reason": None}


def test_blocked_outcome_requires_andon_to_remain_active_at_terminal_state() -> None:
    plan = _plan(2, 1, "part-2-1")
    events = (
        ControllerEvent("andon", 2, "M2_01", "checks disagree"),
        ControllerEvent("resume", 2, None, "inventory restored"),
    )

    with pytest.raises(BirthCertificateInputError, match="Andon evidence"):
        build_birth_certificate(
            catalog=load_catalog(),
            run_id="resumed-not-blocked",
            rollout=1,
            product=2,
            base_instance="base-2",
            plan_records=(plan,),
            events=events,
            inspections=(_inspection(2, 1, "part-2-1", passed=False),),
            outcome=ProductOutcome(ProductResult.BLOCKED, False, None),
        )


@pytest.mark.parametrize("run_id", ["../escape", "has/slash", "", "x" * 129])
def test_rejects_unsafe_run_id(run_id: str) -> None:
    plans, inspections, events = _first_pass_inputs()
    with pytest.raises(BirthCertificateInputError, match="run_id"):
        build_birth_certificate(
            catalog=load_catalog(),
            run_id=run_id,
            rollout=1,
            product=1,
            base_instance="base-1",
            plan_records=plans,
            events=events,
            inspections=inspections,
            outcome=ProductOutcome(ProductResult.GOOD_FIRST_PASS, True, True),
        )


def test_good_outcome_requires_two_final_checks() -> None:
    plans, inspections, events = _first_pass_inputs()
    with pytest.raises(BirthCertificateInputError, match="recipe and Cosmos"):
        build_birth_certificate(
            catalog=load_catalog(),
            run_id="missing-final-check",
            rollout=1,
            product=1,
            base_instance="base-1",
            plan_records=plans,
            events=events,
            inspections=inspections,
            outcome=ProductOutcome(ProductResult.GOOD_FIRST_PASS, True, None),
        )


def test_first_pass_rejects_retry_or_spare_recovery() -> None:
    plans, inspections, events = _first_pass_inputs()
    with pytest.raises(BirthCertificateInputError, match="good_first_pass"):
        build_birth_certificate(
            catalog=load_catalog(),
            run_id="hidden-retry",
            rollout=1,
            product=1,
            base_instance="base-1",
            plan_records=plans,
            events=(ControllerEvent("retry", 1, "M1_01", "1"), *events),
            inspections=inspections,
            outcome=ProductOutcome(ProductResult.GOOD_FIRST_PASS, True, True),
        )


def test_good_outcome_rejects_mismatched_or_failed_step_evidence() -> None:
    plans, inspections, events = _first_pass_inputs()
    bad = _inspection(1, 4, "wrong-instance")
    with pytest.raises(BirthCertificateInputError, match="missing archived inspection"):
        build_birth_certificate(
            catalog=load_catalog(),
            run_id="mismatch",
            rollout=1,
            product=1,
            base_instance="base-1",
            plan_records=plans,
            events=events,
            inspections=(*inspections[:3], bad),
            outcome=ProductOutcome(ProductResult.GOOD_FIRST_PASS, True, True),
        )


def test_filters_other_products_from_shared_controller_streams() -> None:
    plans, inspections, events = _first_pass_inputs()
    other_plans, other_inspections, other_events = _first_pass_inputs(2)
    record = build_birth_certificate(
        catalog=load_catalog(),
        run_id="multi-product",
        rollout=2,
        product=1,
        base_instance="base-1",
        plan_records=(*other_plans, *plans),
        events=(*other_events, *events),
        inspections=(*other_inspections, *inspections),
        outcome=ProductOutcome(ProductResult.GOOD_FIRST_PASS, True, True),
    )
    assert {step["object_id"] for step in record["steps"]} == {f"M1_{step:02d}" for step in range(1, 5)}


def test_rejects_unmeasured_or_nonfinite_values() -> None:
    record = _build_good()
    record["cycle_time_s"] = math.nan
    with pytest.raises(BirthCertificateInputError, match="finite JSON"):
        validate_birth_certificate(record)

    plans, inspections, events = _first_pass_inputs()
    with pytest.raises(BirthCertificateInputError, match="both start and finish"):
        build_birth_certificate(
            catalog=load_catalog(),
            run_id="partial-time",
            rollout=1,
            product=1,
            base_instance="base-1",
            plan_records=plans,
            events=events,
            inspections=inspections,
            outcome=ProductOutcome(ProductResult.GOOD_FIRST_PASS, True, True, started_at_s=1.0),
        )


def test_failed_outcome_requires_abort_and_rejects_completion() -> None:
    plan = _plan(3, 1, "part-3-1")
    inspection = _inspection(3, 1, "part-3-1", passed=False)
    with pytest.raises(BirthCertificateInputError, match="abort evidence"):
        build_birth_certificate(
            catalog=load_catalog(),
            run_id="failed-without-abort",
            rollout=1,
            product=3,
            base_instance="base-3",
            plan_records=(plan,),
            events=(ControllerEvent("complete", 3, None, "incorrect"),),
            inspections=(inspection,),
            outcome=ProductOutcome(ProductResult.FAILED, False, False),
        )
