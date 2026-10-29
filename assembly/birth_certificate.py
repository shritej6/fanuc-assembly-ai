"""Validated product traceability records backed by T09 controller evidence."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from referencing import Registry, Resource
from referencing.exceptions import CannotDetermineSpecification, Unresolvable
from referencing.jsonschema import UnknownDialect

from assembly.birth_certificate_artifacts import (
    ArtifactBundleError,
    verify_artifact_bundle,
    write_artifact_bundle,
)
from assembly.catalog import Catalog
from assembly.controller_ports import ControllerEvent
from assembly.metrics import count_assist_episodes
from assembly.planner import PlanStepRecord

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_PATH = ROOT / "contracts"
SCHEMA_PATH = CONTRACTS_PATH / "birth_certificate.schema.json"
DEFAULT_OUTPUT_ROOT = ROOT / "eval" / "records"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class BirthCertificateError(Exception):
    """Base class for traceability failures."""


class BirthCertificateInputError(BirthCertificateError, ValueError):
    """Evidence is incomplete, inconsistent, or not JSON-safe."""


class BirthCertificateSchemaError(BirthCertificateError):
    """The assembled record violates the shared contract."""


class BirthCertificateResourceError(BirthCertificateError):
    """A shared schema resource cannot be loaded safely."""


class BirthCertificateWriteError(BirthCertificateError, OSError):
    """The immutable JSON/PDF artifact pair cannot be published."""


class ProductResult(StrEnum):
    GOOD_FIRST_PASS = "good_first_pass"
    GOOD_AFTER_REBUILD = "good_after_rebuild"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StepInspectionEvidence:
    """Archived post-placement evidence for one physical instance."""

    product: int
    instance_id: str
    object_id: str
    inspection: Mapping[str, object]
    seated: bool
    pose_agrees: bool


@dataclass(frozen=True, slots=True)
class ProductOutcome:
    """Measured terminal outcome; timing stays null when it was not measured."""

    result: ProductResult | str
    recipe_check: bool
    cosmos_reason: bool | None
    started_at_s: float | None = None
    finished_at_s: float | None = None


@dataclass(frozen=True, slots=True)
class CertificateArtifacts:
    json_path: Path
    pdf_path: Path
    manifest_path: Path


def build_birth_certificate(
    *,
    catalog: Catalog,
    run_id: str,
    rollout: int,
    product: int,
    base_instance: str | None,
    plan_records: Sequence[PlanStepRecord],
    events: Sequence[ControllerEvent],
    inspections: Sequence[StepInspectionEvidence],
    outcome: ProductOutcome,
) -> dict[str, Any]:
    """Assemble and validate one contract-shaped traceability record."""
    _validate_identity(run_id, rollout, product, base_instance)
    recipe = catalog.product(product)
    plans = _product_plans(plan_records, product)
    product_events = _product_events(events, product)
    evidence = _product_inspections(inspections, product)
    result = _normalize_outcome(outcome)
    steps = _build_steps(recipe.steps, plans, product_events, evidence)
    rebuilds = _rebuild_count(product_events)
    _validate_terminal(result.result, steps, rebuilds, product_events, base_instance)
    record = {
        "run_id": run_id,
        "rollout": rollout,
        "product": product,
        "base_instance": base_instance,
        "result": result.result.value,
        "steps": steps,
        "rebuilds": rebuilds,
        "assists": count_assist_episodes(product_events),
        "cycle_time_s": _cycle_time(result),
        "final_checks": {
            "recipe": result.recipe_check,
            "cosmos_reason": result.cosmos_reason,
        },
    }
    return validate_birth_certificate(record)


def validate_birth_certificate(record: Mapping[str, object]) -> dict[str, Any]:
    """Return a detached JSON-safe record after shared-contract validation."""
    detached = _json_copy(record, "birth certificate")
    schema, validator_type = _load_schema(SCHEMA_PATH)
    try:
        errors = sorted(validator_type(schema, registry=_schema_registry()).iter_errors(detached), key=_error_path)
    except Unresolvable as error:
        raise BirthCertificateResourceError(f"Cannot resolve birth certificate schema: {error}") from error
    if errors:
        detail = "; ".join(f"{error.json_path}: {error.message}" for error in errors)
        raise BirthCertificateSchemaError(f"Birth certificate schema validation failed: {detail}")
    if len(detached.get("steps", ())) > 4:
        raise BirthCertificateInputError("birth certificate records at most four product steps")
    return detached


def write_birth_certificate(
    record: Mapping[str, object],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> CertificateArtifacts:
    """Publish an immutable, verifiable bundle below ``eval/records/<run>``."""
    validated = validate_birth_certificate(record)
    run_id = str(validated["run_id"])
    rollout = int(validated["rollout"])
    product = int(validated["product"])
    _validate_run_id(run_id)
    try:
        paths = write_artifact_bundle(validated, output_root, run_id, rollout, product)
    except ArtifactBundleError as error:
        raise BirthCertificateWriteError(str(error)) from error
    return CertificateArtifacts(*paths)


def verify_birth_certificate_artifacts(artifacts: CertificateArtifacts) -> dict[str, Any]:
    """Verify stored JSON/PDF digests and revalidate the decoded record."""
    if not isinstance(artifacts, CertificateArtifacts):
        raise BirthCertificateWriteError("artifacts must be CertificateArtifacts")
    try:
        record = verify_artifact_bundle(
            artifacts.json_path,
            artifacts.pdf_path,
            artifacts.manifest_path,
        )
    except ArtifactBundleError as error:
        raise BirthCertificateWriteError(str(error)) from error
    return validate_birth_certificate(record)


def _validate_identity(run_id: object, rollout: object, product: object, base_instance: object) -> None:
    _validate_run_id(run_id)
    if isinstance(rollout, bool) or not isinstance(rollout, int) or rollout < 0:
        raise BirthCertificateInputError("rollout must be a nonnegative integer")
    if isinstance(product, bool) or not isinstance(product, int) or not 1 <= product <= 6:
        raise BirthCertificateInputError("product must be an integer from 1 through 6")
    if base_instance is not None and (not isinstance(base_instance, str) or not base_instance.strip()):
        raise BirthCertificateInputError("base_instance must be a non-empty string or null")


def _validate_run_id(run_id: object) -> None:
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise BirthCertificateInputError("run_id must be a safe 1-128 character identifier")


def _product_plans(records: Sequence[PlanStepRecord], product: int) -> tuple[PlanStepRecord, ...]:
    normalized = tuple(records)
    if not all(isinstance(record, PlanStepRecord) for record in normalized):
        raise BirthCertificateInputError("plan_records must contain PlanStepRecord values")
    return tuple(record for record in normalized if record.product == product)


def _product_events(events: Sequence[ControllerEvent], product: int) -> tuple[ControllerEvent, ...]:
    normalized = tuple(events)
    if not all(isinstance(event, ControllerEvent) for event in normalized):
        raise BirthCertificateInputError("events must contain ControllerEvent values")
    return tuple(event for event in normalized if event.product == product)


def _product_inspections(
    inspections: Sequence[StepInspectionEvidence],
    product: int,
) -> tuple[StepInspectionEvidence, ...]:
    normalized = tuple(inspections)
    if not all(isinstance(item, StepInspectionEvidence) for item in normalized):
        raise BirthCertificateInputError("inspections must contain StepInspectionEvidence values")
    return tuple(item for item in normalized if item.product == product)


def _normalize_outcome(outcome: object) -> ProductOutcome:
    if not isinstance(outcome, ProductOutcome):
        raise BirthCertificateInputError("outcome must be ProductOutcome")
    try:
        result = ProductResult(outcome.result)
    except (TypeError, ValueError) as error:
        raise BirthCertificateInputError(f"unknown product result {outcome.result!r}") from error
    if not isinstance(outcome.recipe_check, bool):
        raise BirthCertificateInputError("recipe_check must be boolean")
    if outcome.cosmos_reason is not None and not isinstance(outcome.cosmos_reason, bool):
        raise BirthCertificateInputError("cosmos_reason must be boolean or null")
    _validate_times(outcome.started_at_s, outcome.finished_at_s)
    good = result in {ProductResult.GOOD_FIRST_PASS, ProductResult.GOOD_AFTER_REBUILD}
    if good and (outcome.recipe_check is not True or outcome.cosmos_reason is not True):
        raise BirthCertificateInputError("good products require passing recipe and Cosmos Reason checks")
    return ProductOutcome(result, outcome.recipe_check, outcome.cosmos_reason, outcome.started_at_s, outcome.finished_at_s)


def _validate_times(started: object, finished: object) -> None:
    if (started is None) != (finished is None):
        raise BirthCertificateInputError("cycle timing requires both start and finish measurements")
    if started is None:
        return
    if not _finite_number(started) or not _finite_number(finished):
        raise BirthCertificateInputError("cycle timing measurements must be finite numbers")
    if float(finished) < float(started):
        raise BirthCertificateInputError("cycle finish cannot precede cycle start")


def _build_steps(recipe_steps, plans, events, evidence) -> list[dict[str, Any]]:
    expected = {step.object_id for step in recipe_steps}
    unexpected = sorted({record.object_id for record in plans} - expected)
    if unexpected:
        raise BirthCertificateInputError(f"plan records contain objects outside the product recipe: {unexpected}")
    built: list[dict[str, Any]] = []
    for recipe_step in recipe_steps:
        attempts = tuple(record for record in plans if record.object_id == recipe_step.object_id)
        if not attempts:
            continue
        final = attempts[-1]
        inspection = _final_inspection(evidence, final)
        passed = _seated_passed(inspection)
        built.append(
            {
                "step": recipe_step.step,
                "instance_id": final.instance_id,
                "object_id": final.object_id,
                "inspection": _validated_inspection(inspection),
                "attempts": len(attempts),
                "spare_used": _used_spare(events, final.instance_id),
                "seated_check": "passed" if passed else "failed",
            }
        )
    return built


def _final_inspection(evidence, record: PlanStepRecord) -> StepInspectionEvidence:
    matches = tuple(
        item for item in evidence if item.instance_id == record.instance_id and item.object_id == record.object_id
    )
    if not matches:
        raise BirthCertificateInputError(f"missing archived inspection for {record.instance_id!r}")
    return matches[-1]


def _validated_inspection(evidence: StepInspectionEvidence) -> dict[str, Any]:
    if not isinstance(evidence.seated, bool) or not isinstance(evidence.pose_agrees, bool):
        raise BirthCertificateInputError("seated and pose_agrees must be boolean measurements")
    inspection = _json_copy(evidence.inspection, "inspection")
    if inspection.get("instance_id") != evidence.instance_id or inspection.get("object_id") != evidence.object_id:
        raise BirthCertificateInputError("inspection identity does not match archived step evidence")
    schema, validator_type = _load_schema(CONTRACTS_PATH / "inspection.schema.json")
    try:
        errors = sorted(validator_type(schema, registry=_schema_registry()).iter_errors(inspection), key=_error_path)
    except Unresolvable as error:
        raise BirthCertificateResourceError(f"Cannot resolve inspection schema: {error}") from error
    if errors:
        detail = "; ".join(f"{error.json_path}: {error.message}" for error in errors)
        raise BirthCertificateInputError(f"inspection schema validation failed: {detail}")
    return inspection


def _seated_passed(evidence: StepInspectionEvidence) -> bool:
    inspection = evidence.inspection
    return (
        evidence.seated is True
        and evidence.pose_agrees is True
        and inspection.get("checks_agree") is True
        and inspection.get("verdict") == "ok"
        and inspection.get("action") == "use"
    )


def _used_spare(events: Sequence[ControllerEvent], instance_id: str) -> bool:
    return any(event.kind in {"spare", "reject_failed"} and event.detail == instance_id for event in events)


def _rebuild_count(events: Sequence[ControllerEvent]) -> int:
    return sum(event.kind == "rebuild" and event.detail == "started" for event in events)


def _validate_terminal(result, steps, rebuilds, events, base_instance) -> None:
    completed = any(event.kind == "complete" for event in events)
    aborted = any(event.kind == "abort" for event in events)
    held = _andon_active(events)
    good = result in {ProductResult.GOOD_FIRST_PASS, ProductResult.GOOD_AFTER_REBUILD}
    if good and (not completed or aborted or base_instance is None or len(steps) != 4):
        raise BirthCertificateInputError("good outcome requires a base, four recorded steps, and a clean completion event")
    if good and any(step["seated_check"] != "passed" for step in steps):
        raise BirthCertificateInputError("good outcome requires every final seated check to pass")
    recovered = (
        rebuilds > 0
        or any(step["attempts"] > 1 or step["spare_used"] for step in steps)
        or any(event.kind in {"retry", "spare", "reject_failed"} for event in events)
    )
    if result is ProductResult.GOOD_FIRST_PASS and recovered:
        raise BirthCertificateInputError("good_first_pass cannot contain retry, spare, or rebuild evidence")
    if result is ProductResult.GOOD_AFTER_REBUILD and not recovered:
        raise BirthCertificateInputError("good_after_rebuild requires recorded recovery evidence")
    if result is ProductResult.BLOCKED and (completed or not held):
        raise BirthCertificateInputError("blocked outcome requires Andon evidence and no completion")
    if result is ProductResult.FAILED and (completed or not aborted):
        raise BirthCertificateInputError("failed outcome requires abort evidence and no completion")


def _andon_active(events: Sequence[ControllerEvent]) -> bool:
    active = False
    for event in events:
        if event.kind == "andon":
            active = True
        elif event.kind in {"resume", "complete", "abort"}:
            active = False
    return active


def _cycle_time(outcome: ProductOutcome) -> float | None:
    if outcome.started_at_s is None or outcome.finished_at_s is None:
        return None
    return float(outcome.finished_at_s) - float(outcome.started_at_s)


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _json_copy(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BirthCertificateInputError(f"{label} must be a mapping")
    try:
        detached = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise BirthCertificateInputError(f"{label} must contain only finite JSON values") from error
    if not isinstance(detached, dict):
        raise BirthCertificateInputError(f"{label} must encode as an object")
    return detached


def _schema_registry() -> Registry:
    registry = Registry()
    for path in sorted(CONTRACTS_PATH.glob("*.schema.json")):
        schema, _ = _load_schema(path)
        registry = registry.with_resource(path.name, Resource.from_contents(schema))
    return registry


def _load_schema(path: Path):
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
        validator_type = validator_for(schema)
        validator_type.check_schema(schema)
        Resource.from_contents(schema)
    except (OSError, UnicodeError, json.JSONDecodeError, SchemaError) as error:
        raise BirthCertificateResourceError(f"Cannot load schema resource {path}: {error}") from error
    except (CannotDetermineSpecification, UnknownDialect, Unresolvable) as error:
        raise BirthCertificateResourceError(f"Cannot identify schema resource {path}: {error}") from error
    return schema, validator_type


def _error_path(error) -> str:
    return error.json_path
