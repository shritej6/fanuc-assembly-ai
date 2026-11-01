"""Trusted trial-instruction evidence for operator task authorization.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime

from assembly.catalog import Catalog
from assembly.instruction import InstructionError, parse_requested_products


class AuthorityEvidenceError(ValueError):
    """Trial authority evidence is malformed, stale, or inconsistent."""


@dataclass(frozen=True, slots=True)
class TrialInstructionEvidence:
    run_id: str
    rollout: int
    observation_id: str
    instruction: str
    captured_at: datetime
    expires_at: datetime
    source: str

    def __post_init__(self) -> None:
        _safe_id(self.run_id, "run_id")
        if isinstance(self.rollout, bool) or not isinstance(self.rollout, int) or self.rollout < 0:
            raise AuthorityEvidenceError("rollout must be a nonnegative integer")
        _safe_id(self.observation_id, "observation_id")
        _bounded_text(self.instruction, "trial instruction", 4096)
        captured = _aware(self.captured_at, "captured_at")
        expires = _aware(self.expires_at, "expires_at")
        if expires < captured:
            raise AuthorityEvidenceError("trial instruction evidence expires before capture")
        object.__setattr__(self, "captured_at", captured)
        object.__setattr__(self, "expires_at", expires)
        _safe_id(self.source, "trial instruction source")


def authorized_trial_products(
    evidence: object,
    catalog: Catalog,
    now: datetime,
) -> tuple[int, ...]:
    """Parse products only from fresh, typed trial evidence."""
    if type(evidence) is not TrialInstructionEvidence:
        raise AuthorityEvidenceError("trial instruction evidence must be exactly TrialInstructionEvidence")
    current = _aware(now, "authorization clock")
    if evidence.captured_at > current or current > evidence.expires_at:
        raise AuthorityEvidenceError("trial instruction evidence is missing, stale, or from the future")
    try:
        return parse_requested_products(evidence.instruction, catalog)
    except InstructionError as error:
        raise AuthorityEvidenceError(str(error)) from error


def _safe_id(value: object, field: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not isinstance(value, str) or not value or len(value) > 128:
        raise AuthorityEvidenceError(f"{field} must be a safe 1-128 character identifier")
    if any(character not in allowed for character in value):
        raise AuthorityEvidenceError(f"{field} contains unsafe characters")
    return value


def _aware(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise AuthorityEvidenceError(f"{field} must include a timezone")
    return value.astimezone(UTC)


def _bounded_text(value: object, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise AuthorityEvidenceError(f"{field} must be non-empty text no longer than {limit} characters")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise AuthorityEvidenceError(f"{field} must not contain control characters")
    return value
