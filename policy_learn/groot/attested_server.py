"""Versioned attestation endpoint in front of the pinned Isaac-GR00T server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

from policy_learn.deploy.attestation import DeploymentAttestation

ATTESTATION_PROTOCOL = "fanuc-groot-attestation-v1"
_HASH_FORMAT = b"fanuc-checkpoint-tree-v1\0"


class AttestationLoadError(RuntimeError):
    """The deployment manifest cannot prove the checkpoint served by this process."""


def _checkpoint_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise AttestationLoadError(f"checkpoint is not a directory: {root}")
    entries = sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix())
    if any(entry.is_symlink() for entry in entries):
        raise AttestationLoadError("checkpoint tree contains a symlink")
    files = [entry for entry in entries if entry.is_file()]
    if not files:
        raise AttestationLoadError("checkpoint directory contains no files")
    return files


def _hash_regular_file(path: Path, digest: Any) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AttestationLoadError(f"cannot open checkpoint file without following links: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AttestationLoadError(f"checkpoint entry is not a regular file: {path}")
        digest.update(before.st_size.to_bytes(8, "big"))
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        if identity(before) != identity(after):
            raise AttestationLoadError(f"checkpoint file changed while hashing: {path}")
    except OSError as error:
        raise AttestationLoadError(f"cannot hash checkpoint file: {path}") from error
    finally:
        os.close(descriptor)


def checkpoint_tree_sha256(checkpoint: str | Path) -> str:
    """Hash relative paths and bytes so a checkpoint tree has one reproducible identity."""
    supplied = Path(checkpoint)
    if supplied.is_symlink():
        raise AttestationLoadError("checkpoint root cannot be a symlink")
    root = supplied.resolve()
    digest = hashlib.sha256(_HASH_FORMAT)
    for path in _checkpoint_files(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        _hash_regular_file(path, digest)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AttestationLoadError(f"cannot read attestation manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise AttestationLoadError("attestation manifest must be a JSON object")
    expected = {field.name for field in fields(DeploymentAttestation)} - {"server_verified"}
    unknown, missing = sorted(set(value) - expected), sorted(expected - set(value))
    if unknown or missing:
        raise AttestationLoadError(f"attestation fields mismatch; unknown={unknown}, missing={missing}")
    return value


def load_verified_envelope(checkpoint: str | Path, manifest: str | Path) -> dict[str, Any]:
    """Bind manifest identity to checkpoint bytes and derive the server verification flag."""
    attestation = _load_manifest(Path(manifest))
    measured = checkpoint_tree_sha256(checkpoint)
    if attestation["checkpoint_sha256"] != measured:
        raise AttestationLoadError("attestation checkpoint SHA-256 does not match served checkpoint bytes")
    attestation["server_verified"] = True
    return {"protocol": ATTESTATION_PROTOCOL, "attestation": attestation}


class AttestedProxy:
    """Expose local attestation and forward every model operation byte-for-byte upstream."""

    def __init__(self, upstream_socket: Any, envelope: Mapping[str, Any]):
        self._upstream = upstream_socket
        self._envelope = dict(envelope)

    def handle(self, request_bytes: bytes) -> bytes:
        from policy_learn.groot.groot_client import pack, unpack

        request = unpack(request_bytes)
        if not isinstance(request, dict) or not isinstance(request.get("endpoint"), str):
            return pack({"error": "request must contain a string endpoint"})
        if request["endpoint"] == "describe":
            return pack(self._envelope)
        self._upstream.send(request_bytes)
        return self._upstream.recv()


def serve(bind: str, upstream: str, envelope: Mapping[str, Any], timeout_ms: int) -> None:
    """Serve until interrupted; an upstream timeout is returned as a typed server error."""
    import zmq

    from policy_learn.groot.groot_client import pack

    context = zmq.Context()
    frontend, backend = context.socket(zmq.REP), context.socket(zmq.REQ)
    backend.setsockopt(zmq.RCVTIMEO, timeout_ms)
    backend.setsockopt(zmq.SNDTIMEO, timeout_ms)
    frontend.bind(bind)
    backend.connect(upstream)
    proxy = AttestedProxy(backend, envelope)
    try:
        while True:
            request = frontend.recv()
            try:
                frontend.send(proxy.handle(request))
            except zmq.error.Again as exc:
                frontend.send(pack({"error": "upstream GR00T server timed out"}))
                raise RuntimeError("upstream GR00T server timed out; proxy stopped fail-closed") from exc
    finally:
        frontend.close(linger=0)
        backend.close(linger=0)
        context.term()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--upstream-port", type=int, default=5556)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    envelope = load_verified_envelope(args.checkpoint, args.attestation)
    if not args.verify_only:
        serve(
            f"tcp://{args.host}:{args.port}",
            f"tcp://127.0.0.1:{args.upstream_port}",
            envelope,
            args.timeout_ms,
        )


if __name__ == "__main__":
    main()
