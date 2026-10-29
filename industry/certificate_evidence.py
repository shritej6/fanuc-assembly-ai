"""Semantic checks for projecting T15 certificates into dashboard events."""

from __future__ import annotations

from collections.abc import Mapping

from assembly.birth_certificate import ProductResult as CertificateResult
from assembly.catalog import CatalogError, ProductRecipe, load_catalog


class CertificateEvidenceError(ValueError):
    """A schema-valid certificate cannot support its claimed terminal result."""


def validate_certificate_evidence(certificate: Mapping[str, object]) -> None:
    """Fail closed when certificate fields cannot prove the terminal result."""
    try:
        result = CertificateResult(certificate.get("result"))
    except (TypeError, ValueError) as error:
        raise CertificateEvidenceError("certificate result is invalid") from error
    if result not in {CertificateResult.GOOD_FIRST_PASS, CertificateResult.GOOD_AFTER_REBUILD}:
        return
    product = certificate.get("product")
    if isinstance(product, bool) or not isinstance(product, int):
        raise CertificateEvidenceError("good certificate product is invalid")
    recipe = _recipe(product)
    steps = _good_steps(certificate, recipe)
    _require_final_checks(certificate)
    recovered = _has_recovery(certificate, steps)
    if result is CertificateResult.GOOD_FIRST_PASS and recovered:
        raise CertificateEvidenceError("good_first_pass forbids rebuild, retry, and spare evidence")
    if result is CertificateResult.GOOD_AFTER_REBUILD and not recovered:
        raise CertificateEvidenceError("good_after_rebuild requires recorded recovery evidence")


def _recipe(product: int) -> ProductRecipe:
    try:
        return load_catalog().product(product)
    except CatalogError as error:
        raise CertificateEvidenceError(f"cannot validate product {product} recipe: {error}") from error


def _good_steps(certificate: Mapping[str, object], recipe: ProductRecipe) -> tuple[Mapping[str, object], ...]:
    if certificate.get("base_instance") is None:
        raise CertificateEvidenceError("good certificate requires a non-null base")
    raw_steps = certificate.get("steps")
    if not isinstance(raw_steps, list) or any(not isinstance(step, Mapping) for step in raw_steps):
        raise CertificateEvidenceError("good certificate requires four unique recipe steps")
    steps = tuple(raw_steps)
    actual = {(step.get("step"), step.get("object_id")) for step in steps}
    expected = {(step.step, step.object_id) for step in recipe.steps}
    if len(steps) != 4 or len(actual) != 4 or actual != expected:
        raise CertificateEvidenceError("good certificate requires four unique recipe steps")
    if any(not _passing_step(step) for step in steps):
        raise CertificateEvidenceError("good certificate requires passed seated checks and positive attempts")
    if any(not _passing_inspection(step) for step in steps):
        raise CertificateEvidenceError("good certificate requires matching passing inspection evidence")
    return steps


def _passing_step(step: Mapping[str, object]) -> bool:
    attempts = step.get("attempts")
    return (
        not isinstance(attempts, bool)
        and isinstance(attempts, int)
        and attempts >= 1
        and step.get("seated_check") == "passed"
        and isinstance(step.get("spare_used"), bool)
    )


def _passing_inspection(step: Mapping[str, object]) -> bool:
    inspection = step.get("inspection")
    return (
        isinstance(inspection, Mapping)
        and inspection.get("instance_id") == step.get("instance_id")
        and inspection.get("object_id") == step.get("object_id")
        and inspection.get("checks_agree") is True
        and inspection.get("verdict") == "ok"
        and inspection.get("action") == "use"
    )


def _require_final_checks(certificate: Mapping[str, object]) -> None:
    checks = certificate.get("final_checks")
    if not isinstance(checks, Mapping) or checks.get("recipe") is not True or checks.get("cosmos_reason") is not True:
        raise CertificateEvidenceError("good certificate requires both final checks to be true")


def _has_recovery(certificate: Mapping[str, object], steps: tuple[Mapping[str, object], ...]) -> bool:
    rebuilds = certificate.get("rebuilds")
    if isinstance(rebuilds, bool) or not isinstance(rebuilds, int) or rebuilds < 0:
        raise CertificateEvidenceError("good certificate requires a nonnegative rebuild count")
    if rebuilds > 0:
        return True
    return any(int(step["attempts"]) > 1 or step["spare_used"] is True for step in steps)
