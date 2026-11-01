"""CELL model identity is verified before any learned action can reach the robot.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace

import pytest

from assembly.catalog import TRUSTED_GRASP_SOURCE
from policy_learn.deploy import attestation as attestation_module
from policy_learn.deploy.attestation import (
    REQUIRED_CELL_SKILLS,
    AttestationError,
    DeploymentAttestation,
    DeploymentPin,
    attestation_from_mapping,
    load_reviewed_deployment,
    validate_attestation,
)


def _approved(**changes) -> DeploymentAttestation:
    value = DeploymentAttestation(
        checkpoint_sha256="a" * 64,
        model_id="GR00T-N1.7",
        model_revision="NVIDIA/GR00T-N1.7@reviewed",
        dataset_manifest_sha256="b" * 64,
        catalog_sha256="c" * 64,
        grasp_source_sha256=TRUSTED_GRASP_SOURCE.sha256,
        guard_calibration_sha256="e" * 64,
        skills=REQUIRED_CELL_SKILLS,
        cameras=("table", "side", "wrist"),
        depth=(),
        image_size=(640, 360),
        dt_s=0.2,
        execution_horizon=1,
        action="tcp",
        embodiment="fanuc_er4ia_team_v_fingers",
        release_protocol="t09_verified_release_v1",
        eval_approved=True,
        server_verified=True,
    )
    return replace(value, **changes)


def _mapping(value: DeploymentAttestation) -> dict:
    payload = {field.name: getattr(value, field.name) for field in fields(value)}
    payload["skills"] = sorted(payload["skills"])
    payload["cameras"] = list(payload["cameras"])
    payload["depth"] = list(payload["depth"])
    payload["image_size"] = list(payload["image_size"])
    return payload


def test_approved_attestation_covers_every_semantic_motion() -> None:
    approved = _approved()
    assert validate_attestation(approved, expected=approved) is approved


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"skills": frozenset({"pick_insert"})}, "skills"),
        ({"execution_horizon": 16}, "horizon"),
        ({"grasp_source_sha256": "d" * 64}, "T29"),
        ({"release_protocol": "model_decides"}, "release"),
        ({"eval_approved": False}, "evaluation"),
        ({"server_verified": False}, "server"),
        ({"depth": ("table",)}, "depth"),
    ],
)
def test_attestation_mismatch_blocks_cell(changes: dict, message: str) -> None:
    with pytest.raises(AttestationError, match=message):
        validate_attestation(_approved(**changes), expected=_approved())


def test_sidecar_only_claim_cannot_self_attest() -> None:
    with pytest.raises(AttestationError, match="server"):
        validate_attestation(_approved(server_verified=False), expected=_approved())


@pytest.mark.parametrize(
    "changes",
    [
        {"checkpoint_sha256": "d" * 64},
        {"dataset_manifest_sha256": "d" * 64},
        {"catalog_sha256": "d" * 64},
        {"model_revision": "NVIDIA/GR00T-N1.7@different"},
    ],
)
def test_server_identity_must_exactly_match_the_local_approved_record(changes: dict) -> None:
    with pytest.raises(AttestationError, match="approved deployment"):
        validate_attestation(_approved(**changes), expected=_approved())


def test_versioned_server_mapping_is_decoded_without_defaults() -> None:
    approved = _approved()
    payload = _mapping(approved)

    decoded = attestation_from_mapping(payload)

    assert decoded == _approved()


@pytest.mark.parametrize(
    "change",
    [
        {"unknown": "field"},
        {"skills": "pick_insert_hold"},
        {"skills": [["not-hashable"]]},
        {"image_size": [True, 360]},
        {"cameras": ["table", ""]},
    ],
)
def test_malformed_server_mapping_is_rejected(change: dict) -> None:
    approved = _approved()
    payload = _mapping(approved)
    payload.update(change)
    with pytest.raises(AttestationError):
        attestation_from_mapping(payload)


def test_local_approval_requires_a_registry_pinned_record(tmp_path, monkeypatch) -> None:
    with pytest.raises(AttestationError, match="not in the reviewed registry"):
        load_reviewed_deployment("missing")

    payload = _mapping(_approved())
    payload.pop("server_verified")
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        attestation_module,
        "APPROVED_DEPLOYMENT_PINS",
        {"saturday": DeploymentPin(str(path), digest)},
    )

    reviewed = load_reviewed_deployment("saturday")

    assert reviewed.attestation == _approved()
    assert reviewed.record_sha256 == digest


def test_local_approval_rejects_record_self_attestation(tmp_path, monkeypatch) -> None:
    payload = _mapping(_approved())
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        attestation_module,
        "APPROVED_DEPLOYMENT_PINS",
        {"saturday": DeploymentPin(str(path), digest)},
    )
    with pytest.raises(AttestationError, match="self-assert"):
        load_reviewed_deployment("saturday")


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"manifest": "deployment.json", "sha256": "a" * 64}, "wrong type"),
        (DeploymentPin("", "a" * 64), "manifest path"),
    ],
)
def test_local_approval_rejects_malformed_registry_entries(entry, message, monkeypatch) -> None:
    monkeypatch.setattr(attestation_module, "APPROVED_DEPLOYMENT_PINS", {"saturday": entry})

    with pytest.raises(AttestationError, match=message):
        load_reviewed_deployment("saturday")
