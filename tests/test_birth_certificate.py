"""Behavioral coverage for immutable JSON/PDF product traceability."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pypdf import PdfReader

from assembly.birth_certificate import (
    BirthCertificateInputError,
    BirthCertificateWriteError,
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

    assert artifacts.json_path == tmp_path / "records" / "run-2026-10-29" / "1.json"
    assert json.loads(artifacts.json_path.read_text()) == record
    assert artifacts.pdf_path.read_bytes().startswith(b"%PDF-")
    reader = PdfReader(artifacts.pdf_path)
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    assert "PRODUCT BIRTH CERTIFICATE" in text
    assert "Product 1" in text
    assert "M1_01" in text
    assert "M1_04" in text
    assert "Page 1 of 1" in text
    with pytest.raises(BirthCertificateWriteError, match="immutable"):
        write_birth_certificate(record, output_root=tmp_path / "records")


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
