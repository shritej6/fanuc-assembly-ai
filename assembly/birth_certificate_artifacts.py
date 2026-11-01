"""Atomic, hash-verifiable storage for product birth certificates."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

JSON_NAME = "certificate.json"
PDF_NAME = "certificate.pdf"
MANIFEST_NAME = "manifest.json"


class ArtifactBundleError(OSError):
    """A certificate bundle cannot be rendered, committed, or verified."""


def write_artifact_bundle(
    record: Mapping[str, object],
    output_root: Path,
    run_id: str,
    rollout: int,
    product: int,
) -> tuple[Path, Path, Path]:
    """Render and atomically publish one read-only artifact directory."""
    canonical_digest = hashlib.sha256(_canonical_json(record)).hexdigest()
    pretty_json = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        from assembly.birth_certificate_pdf import render_birth_certificate_pdf

        pdf_bytes = render_birth_certificate_pdf(dict(record), canonical_digest)
    except Exception as error:
        raise ArtifactBundleError(f"Cannot render birth certificate PDF: {error}") from error
    manifest = _manifest_bytes(canonical_digest, pdf_bytes)
    directory = _artifact_directory(output_root, run_id, rollout, product)
    _publish_directory(directory, {JSON_NAME: pretty_json, PDF_NAME: pdf_bytes, MANIFEST_NAME: manifest})
    return directory / JSON_NAME, directory / PDF_NAME, directory / MANIFEST_NAME


def verify_artifact_bundle(json_path: Path, pdf_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify both artifact digests and return the decoded JSON record."""
    manifest = _read_json(manifest_path, "manifest")
    _validate_manifest(manifest)
    record = _read_json(json_path, "certificate JSON")
    actual_json = hashlib.sha256(_canonical_json(record)).hexdigest()
    if not hmac.compare_digest(actual_json, manifest["canonical_json_sha256"]):
        raise ArtifactBundleError("Birth certificate JSON digest does not match its manifest")
    try:
        actual_pdf = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ArtifactBundleError(f"Cannot read certificate PDF: {error}") from error
    if not hmac.compare_digest(actual_pdf, manifest["pdf_sha256"]):
        raise ArtifactBundleError("Birth certificate PDF digest does not match its manifest")
    return record


def _canonical_json(record: Mapping[str, object]) -> bytes:
    return json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()


def _manifest_bytes(canonical_digest: str, pdf_bytes: bytes) -> bytes:
    manifest = {
        "algorithm": "sha256",
        "canonical_json_sha256": canonical_digest,
        "files": {"json": JSON_NAME, "pdf": PDF_NAME},
        "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
    }
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()


def _artifact_directory(output_root: Path, run_id: str, rollout: int, product: int) -> Path:
    if not isinstance(output_root, Path):
        raise ArtifactBundleError("output_root must be a pathlib.Path")
    parent = output_root / run_id / f"rollout-{rollout:06d}"
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ArtifactBundleError(f"Cannot create certificate directory {parent}: {error}") from error
    return parent / f"product-{product}"


def _publish_directory(directory: Path, files: Mapping[str, bytes]) -> None:
    if directory.exists():
        raise ArtifactBundleError("birth certificate artifacts are immutable and already exist")
    staged = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    committed = False
    try:
        for name, data in files.items():
            path = staged / name
            _write_file(path, data)
            path.chmod(0o444)
        staged.chmod(0o555)
        _fsync_directory(staged)
        os.rename(staged, directory)
        committed = True
        _fsync_directory(directory.parent)
    except OSError as error:
        raise ArtifactBundleError(f"Cannot publish certificate artifact bundle: {error}") from error
    finally:
        if not committed:
            _remove_staged(staged)


def _write_file(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_staged(directory: Path) -> None:
    if not directory.exists():
        return
    directory.chmod(0o755)
    shutil.rmtree(directory)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactBundleError(f"Cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactBundleError(f"{label} must encode a JSON object")
    return value


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    if manifest.get("algorithm") != "sha256":
        raise ArtifactBundleError("Certificate manifest must use sha256")
    files = manifest.get("files")
    if files != {"json": JSON_NAME, "pdf": PDF_NAME}:
        raise ArtifactBundleError("Certificate manifest contains unexpected artifact names")
    for field in ("canonical_json_sha256", "pdf_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ArtifactBundleError(f"Certificate manifest {field} is not a SHA-256 digest")
