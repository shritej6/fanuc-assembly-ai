"""Fail-closed identity and capability checks for a served GR00T checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from numbers import Number
from pathlib import Path
from types import MappingProxyType

from assembly.catalog import TRUSTED_GRASP_SOURCE

REQUIRED_CELL_SKILLS = frozenset(
    {
        "inspect_view",
        "pick_insert_hold",
        "release_retreat",
        "pull_out",
        "place_free",
        "defect_drop",
    }
)
_SHA256_LEN = 64
_REVIEWED_DEPLOYMENT_SEAL = object()
_ROOT = Path(__file__).resolve().parents[2]


class AttestationError(RuntimeError):
    """A served checkpoint cannot prove the exact Saturday deployment contract."""


@dataclass(frozen=True, slots=True)
class DeploymentAttestation:
    checkpoint_sha256: str
    model_id: str
    model_revision: str
    dataset_manifest_sha256: str
    catalog_sha256: str
    grasp_source_sha256: str
    guard_calibration_sha256: str
    skills: frozenset[str]
    cameras: tuple[str, ...]
    depth: tuple[str, ...]
    image_size: tuple[int, int]
    dt_s: float
    execution_horizon: int
    action: str
    embodiment: str
    release_protocol: str
    eval_approved: bool
    server_verified: bool


@dataclass(frozen=True, slots=True)
class DeploymentPin:
    """Repository-reviewed pointer to one immutable deployment manifest."""

    manifest: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ReviewedDeployment:
    """Attestation decoded only from a registry-pinned local record."""

    name: str
    attestation: DeploymentAttestation
    record_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _REVIEWED_DEPLOYMENT_SEAL:
            raise AttestationError("reviewed deployments can only be created by the pinned-record loader")


# Deliberately empty until the trained checkpoint, evaluation, and CSI calibration
# have a dual-reviewed immutable manifest. An empty registry makes production fail closed.
APPROVED_DEPLOYMENT_PINS: Mapping[str, DeploymentPin] = MappingProxyType({})


def attestation_from_mapping(value: object) -> DeploymentAttestation:
    """Decode the versioned server payload without accepting implicit defaults."""
    if not isinstance(value, Mapping):
        raise AttestationError("server attestation must be a mapping")
    expected = {field.name for field in fields(DeploymentAttestation)}
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise AttestationError(f"server attestation fields mismatch; unknown={unknown}, missing={missing}")
    converted = dict(value)
    converted["skills"] = _string_set(value["skills"], "skills")
    converted["cameras"] = _string_tuple(value["cameras"], "cameras")
    converted["depth"] = _string_tuple(value["depth"], "depth")
    converted["image_size"] = _integer_pair(value["image_size"], "image_size")
    try:
        return DeploymentAttestation(**converted)
    except TypeError as error:
        raise AttestationError(f"server attestation cannot be decoded: {error}") from error


def _read_record(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AttestationError("reviewed deployment record is unreadable or a symlink") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise AttestationError("reviewed deployment record must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            return source.read()
    except OSError as error:
        raise AttestationError("reviewed deployment record could not be read") from error
    finally:
        os.close(descriptor)


def load_reviewed_deployment(name: str) -> ReviewedDeployment:
    """Load one local approval only when its bytes match the repository trust registry."""
    if not isinstance(name, str) or not name.strip():
        raise AttestationError("deployment name must be a non-empty string")
    try:
        pin = APPROVED_DEPLOYMENT_PINS[name]
    except KeyError as error:
        raise AttestationError(f"deployment {name!r} is not in the reviewed registry") from error
    if not isinstance(pin, DeploymentPin):
        raise AttestationError("reviewed deployment registry entry has the wrong type")
    if not isinstance(pin.manifest, str) or not pin.manifest.strip():
        raise AttestationError("reviewed deployment manifest path must be a non-empty string")
    _require_sha256(pin.sha256, "deployment record")
    path = Path(pin.manifest)
    path = path if path.is_absolute() else _ROOT / path
    raw = _read_record(path)
    if hashlib.sha256(raw).hexdigest() != pin.sha256:
        raise AttestationError("reviewed deployment record differs from its registry SHA-256")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise AttestationError("reviewed deployment record is not valid JSON") from error
    if not isinstance(payload, dict) or "server_verified" in payload:
        raise AttestationError("reviewed deployment record cannot self-assert server verification")
    payload["server_verified"] = True
    attestation = attestation_from_mapping(payload)
    return ReviewedDeployment(name, attestation, pin.sha256, _REVIEWED_DEPLOYMENT_SEAL)


def _string_set(value: object, label: str) -> frozenset[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise AttestationError(f"{label} must be a sequence of strings")
    try:
        result = frozenset(value)
    except TypeError as error:
        raise AttestationError(f"{label} must contain hashable strings") from error
    if not result or any(not isinstance(item, str) or not item.strip() for item in result):
        raise AttestationError(f"{label} must contain non-empty strings")
    return result


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise AttestationError(f"{label} must be an ordered sequence of strings")
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise AttestationError(f"{label} must contain non-empty strings")
    return result


def _integer_pair(value: object, label: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise AttestationError(f"{label} must contain two integers")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise AttestationError(f"{label} must contain two positive integers")
    return (value[0], value[1])


def _require_sha256(value: object, label: str) -> None:
    valid = isinstance(value, str) and len(value) == _SHA256_LEN and all(c in "0123456789abcdef" for c in value)
    if not valid:
        raise AttestationError(f"{label} must be a lowercase SHA-256 digest")


def _validate_identity(value: DeploymentAttestation) -> None:
    for digest, label in (
        (value.checkpoint_sha256, "checkpoint"),
        (value.dataset_manifest_sha256, "dataset manifest"),
        (value.catalog_sha256, "catalog"),
        (value.grasp_source_sha256, "T29 grasp source"),
        (value.guard_calibration_sha256, "guard calibration"),
    ):
        _require_sha256(digest, label)
    if value.grasp_source_sha256 != TRUSTED_GRASP_SOURCE.sha256:
        raise AttestationError("T29 grasp source SHA-256 does not match the reviewed pin")


def _validate_interface(value: DeploymentAttestation) -> None:
    if value.model_id != "GR00T-N1.7" or not isinstance(value.model_revision, str) or not value.model_revision.strip():
        raise AttestationError("model identity or revision is not the reviewed GR00T N1.7 build")
    if not isinstance(value.skills, frozenset):
        raise AttestationError("checkpoint skills must be a frozenset")
    if not REQUIRED_CELL_SKILLS.issubset(value.skills):
        missing = sorted(REQUIRED_CELL_SKILLS - value.skills)
        raise AttestationError(f"checkpoint skills are incomplete: {missing}")
    expected = (("table", "side", "wrist"), (), (640, 360), "tcp")
    actual = (value.cameras, value.depth, value.image_size, value.action)
    if actual != expected:
        raise AttestationError("camera, depth, image, or action interface does not match CELL")
    valid_dt = isinstance(value.dt_s, Number) and not isinstance(value.dt_s, bool)
    valid_horizon = isinstance(value.execution_horizon, int) and not isinstance(value.execution_horizon, bool)
    if not valid_dt or not math.isclose(float(value.dt_s), 0.2) or not valid_horizon or value.execution_horizon != 1:
        raise AttestationError("CELL requires dt 0.2 s and execution horizon 1")
    if value.embodiment != "fanuc_er4ia_team_v_fingers":
        raise AttestationError("checkpoint embodiment does not match the physical cell")
    if value.release_protocol != "t09_verified_release_v1":
        raise AttestationError("checkpoint release protocol cannot enforce T09 verification")


def validate_attestation(
    value: DeploymentAttestation,
    *,
    expected: DeploymentAttestation,
) -> DeploymentAttestation:
    """Return the attestation only after identity, interface, evaluation, and server proof agree."""
    for candidate, label in ((expected, "approved"), (value, "server")):
        if not isinstance(candidate, DeploymentAttestation):
            raise AttestationError(f"{label} attestation has the wrong type")
        if candidate.server_verified is not True:
            raise AttestationError(f"{label} attestation lacks server verification")
        if candidate.eval_approved is not True:
            raise AttestationError(f"{label} attestation lacks reviewed evaluation approval")
        _validate_identity(candidate)
        _validate_interface(candidate)
    if value != expected:
        mismatched = [
            field.name for field in fields(value) if getattr(value, field.name) != getattr(expected, field.name)
        ]
        raise AttestationError(f"server attestation differs from the approved deployment: {mismatched}")
    return value
