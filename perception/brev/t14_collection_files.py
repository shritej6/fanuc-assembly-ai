"""Fail-closed filesystem operations for resumable T14 collection."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class CollectionError(ValueError):
    """Raised when off-host T14 evidence cannot be proven complete."""


@dataclass(frozen=True)
class CopySpec:
    remote: str
    relative: str
    expected_sha256: str | None = None


CopyFile = Callable[[str, str, Path], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _destination(path: Path, *, resume_partial: bool) -> tuple[Path, Path]:
    destination = path.expanduser()
    if not destination.is_absolute() or destination == Path("/"):
        raise CollectionError("destination must be a safe absolute path")
    partial = destination.with_name(f"{destination.name}.partial")
    if destination.exists() or destination.is_symlink():
        raise CollectionError("destination must be fresh")
    if partial.is_symlink():
        raise CollectionError("partial collection must not be a symlink")
    if partial.exists():
        if not resume_partial:
            raise CollectionError("existing partial collection requires explicit resume")
        if not partial.is_dir():
            raise CollectionError("partial collection must be a real directory")
        return destination, partial
    if not destination.parent.is_dir():
        raise CollectionError("destination parent must already exist")
    partial.mkdir()
    return destination, partial


def _assert_real_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CollectionError(f"evidence file is unavailable: {label}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise CollectionError(f"evidence must be one real, unlinked file: {label}")


def _atomic_copy(spec: CopySpec, target: Path, instance: str, copy_file: CopyFile) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.t14-copy")
    if temporary.exists() or temporary.is_symlink():
        _assert_real_file(temporary, spec.relative)
        temporary.unlink()
    try:
        copy_file(instance, spec.remote, temporary)
        _assert_real_file(temporary, spec.relative)
        if spec.expected_sha256 is not None and _sha256(temporary) != spec.expected_sha256:
            raise CollectionError(f"copied evidence hash mismatch: {spec.relative}")
        temporary.replace(target)
    finally:
        if temporary.exists() or temporary.is_symlink():
            _assert_real_file(temporary, spec.relative)
            temporary.unlink()


def _copy(
    spec: CopySpec,
    root: Path,
    instance: str,
    copy_file: CopyFile,
    *,
    resume_partial: bool,
) -> None:
    target = root / spec.relative
    if target.exists() or target.is_symlink():
        _assert_real_file(target, spec.relative)
        if not resume_partial:
            raise CollectionError(f"duplicate collection target: {spec.relative}")
        if spec.expected_sha256 is not None:
            if _sha256(target) != spec.expected_sha256:
                raise CollectionError(f"partial evidence hash mismatch: {spec.relative}")
            return
    _atomic_copy(spec, target, instance, copy_file)


def _copy_many(
    specs: Iterable[CopySpec],
    root: Path,
    instance: str,
    copy_file: CopyFile,
    *,
    resume_partial: bool,
) -> list[CopySpec]:
    copied = []
    for spec in specs:
        _copy(spec, root, instance, copy_file, resume_partial=resume_partial)
        copied.append(spec)
    return copied


def _expected_directories(relatives: set[str]) -> set[str]:
    directories = set()
    for relative in relatives:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _partial_entries(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise CollectionError(f"partial evidence must not contain symlinks: {relative}")
        metadata = path.lstat()
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise CollectionError(f"partial evidence must not contain hard links: {relative}")
            files.add(relative)
        elif stat.S_ISDIR(metadata.st_mode):
            directories.add(relative)
        else:
            raise CollectionError(f"partial evidence contains a special file: {relative}")
    return files, directories


def _verify_partial_tree(root: Path, relatives: set[str], *, complete: bool) -> None:
    files, directories = _partial_entries(root)
    allowed_files = relatives | {"COLLECTION_FAILED.json"}
    unexpected = (files - allowed_files) | (directories - _expected_directories(relatives))
    if unexpected:
        raise CollectionError(f"untracked partial evidence: {sorted(unexpected)}")
    if complete and files - {"COLLECTION_FAILED.json"} != relatives:
        missing = relatives - files
        raise CollectionError(f"partial collection is incomplete: {sorted(missing)}")


def _remove_stale_copy_temporaries(root: Path, relatives: set[str]) -> None:
    for relative in relatives:
        target = root / relative
        temporary = target.with_name(f".{target.name}.t14-copy")
        if temporary.exists() or temporary.is_symlink():
            _assert_real_file(temporary, temporary.relative_to(root).as_posix())
            temporary.unlink()


def _clear_failure_marker(root: Path) -> None:
    marker = root / "COLLECTION_FAILED.json"
    if marker.exists() or marker.is_symlink():
        _assert_real_file(marker, "COLLECTION_FAILED.json")
        marker.unlink()


def _verify_spec_hashes(root: Path, specs: Iterable[CopySpec]) -> None:
    for spec in specs:
        path = root / spec.relative
        _assert_real_file(path, spec.relative)
        if spec.expected_sha256 is not None and _sha256(path) != spec.expected_sha256:
            raise CollectionError(f"final evidence hash mismatch: {spec.relative}")


def _failure_marker(root: Path, error: Exception) -> None:
    payload = {"status": "collection_failed", "error_type": type(error).__name__, "message": str(error)}
    descriptor, name = tempfile.mkstemp(prefix=".COLLECTION_FAILED.", suffix=".tmp", dir=root)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _assert_real_file(temporary, temporary.name)
        temporary.replace(root / "COLLECTION_FAILED.json")
    finally:
        temporary.unlink(missing_ok=True)
