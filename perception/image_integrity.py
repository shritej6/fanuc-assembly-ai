"""Exact-byte image identities and verified JPEG decoding."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ImageIntegrityError(ValueError):
    """Raised when an image does not match its frozen byte identity."""


def require_sha256(value: object, field: str) -> str:
    """Return a canonical lowercase SHA-256 value or reject it."""
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ImageIntegrityError(f"{field} must be a canonical lowercase SHA-256")
    return value


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ImageIntegrityError(f"cannot read image {path}: {exc}") from exc


def _decode_rgb(path: Path, payload: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(payload)) as opened:
            return opened.convert("RGB")
    except OSError as exc:
        raise ImageIntegrityError(f"cannot decode image {path}: {exc}") from exc


def image_identity(path: Path) -> tuple[int, int, str]:
    """Return dimensions and the SHA-256 of the exact source file bytes."""
    payload = _read_bytes(path)
    image = _decode_rgb(path, payload)
    return image.width, image.height, hashlib.sha256(payload).hexdigest()


def read_verified_rgb(path: Path, expected_sha256: object) -> Image.Image:
    """Verify exact file bytes, then decode those same bytes as RGB."""
    expected = require_sha256(expected_sha256, "expected image SHA-256")
    payload = _read_bytes(path)
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise ImageIntegrityError(
            f"image SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    return _decode_rgb(path, payload)


def encode_jpeg(image: Image.Image, quality: int = 95) -> tuple[bytes, str]:
    """Encode once and return the exact JPEG bytes plus their identity."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace a file atomically with already-hashed bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)
