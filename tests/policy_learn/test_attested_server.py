"""The GR00T proxy only attests the exact immutable checkpoint it fronts.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json

import pytest

from policy_learn.groot.attested_server import (
    AttestationLoadError,
    checkpoint_tree_sha256,
    load_verified_envelope,
)


def _manifest(checkpoint_sha256: str) -> dict[str, object]:
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "model_id": "GR00T-N1.7",
        "model_revision": "reviewed",
        "dataset_manifest_sha256": "b" * 64,
        "catalog_sha256": "c" * 64,
        "grasp_source_sha256": "d" * 64,
        "guard_calibration_sha256": "e" * 64,
        "skills": ["pick_insert_hold"],
        "cameras": ["table", "side", "wrist"],
        "depth": [],
        "image_size": [640, 360],
        "dt_s": 0.2,
        "execution_horizon": 1,
        "action": "tcp",
        "embodiment": "fanuc_er4ia_team_v_fingers",
        "release_protocol": "t09_verified_release_v1",
        "eval_approved": True,
    }


def test_verified_envelope_is_derived_from_checkpoint_bytes(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "weights.bin").write_bytes(b"reviewed weights")
    manifest = tmp_path / "attestation.json"
    manifest.write_text(json.dumps(_manifest(checkpoint_tree_sha256(checkpoint))), encoding="utf-8")

    envelope = load_verified_envelope(checkpoint, manifest)

    assert envelope["protocol"] == "fanuc-groot-attestation-v1"
    assert envelope["attestation"]["server_verified"] is True
    assert envelope["attestation"]["checkpoint_sha256"] == checkpoint_tree_sha256(checkpoint)


def test_modified_checkpoint_or_self_asserted_verification_is_rejected(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "weights.bin"
    weights.write_bytes(b"original")
    manifest = tmp_path / "attestation.json"
    data = _manifest(checkpoint_tree_sha256(checkpoint))
    data["server_verified"] = True
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(AttestationLoadError, match="server_verified"):
        load_verified_envelope(checkpoint, manifest)

    data.pop("server_verified")
    manifest.write_text(json.dumps(data), encoding="utf-8")
    weights.write_bytes(b"changed")
    with pytest.raises(AttestationLoadError, match="checkpoint SHA-256"):
        load_verified_envelope(checkpoint, manifest)


def test_checkpoint_tree_rejects_symlinks(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"mutable")
    (checkpoint / "weights.bin").symlink_to(target)

    with pytest.raises(AttestationLoadError, match="symlink"):
        checkpoint_tree_sha256(checkpoint)
