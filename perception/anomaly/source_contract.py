"""Immutable source-corpus contract for T14 PatchCore preparation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SOURCE_CHUNK_COUNTS = {"O": 48, "T": 3, "C": 16, "I": 12}
SOURCE_LABELS_PER_CHUNK = 500
SOURCE_O_TARGET_COUNTS = {
    **{f"F{product:02d}": 800 for product in range(1, 7)},
    **{f"M{product}_0{part}": 800 for product in range(1, 7) for part in range(1, 5)},
}
O_BASE_TARGETS = frozenset(f"F{product:02d}" for product in range(1, 7))
REVIEWED_DEFECT_MANIFEST_SHA256 = "4918fec9d73a9a0fdbeb64bb878fe60d21e476ca4e9ce81c1a45665ca2e36659"


class DatasetBoundaryError(ValueError):
    """Raised when synthetic anomaly data cannot be proven safe to consume."""


def _canonical_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4 or any(type(item) is not int for item in value):
        return False
    x0, y0, x1, y1 = value
    return 0 <= x0 < x1 and 0 <= y0 < y1


def exclude_o_base_target(
    set_name: str,
    target: Mapping[str, Any],
    box: Mapping[str, Any],
    label_path: Path,
) -> bool:
    """Exclude only the six measured Set-O fixture targets; reject every other non-part base."""
    if target.get("kind") != "base":
        return False
    uid, object_id = target.get("uid"), target.get("object_id")
    target_ok = (
        set_name == "O"
        and isinstance(uid, str)
        and uid.startswith("o")
        and uid[1:].isdigit()
        and object_id in O_BASE_TARGETS
        and target.get("variant") == "ok"
        and target.get("state") == "on_table"
        and target.get("build_state") is None
        and "stable_pose" not in target
        and "rest" not in target
    )
    fields = ("uid", "object_id", "kind", "state", "variant", "build_state")
    box_ok = all(box.get(field) == target.get(field) for field in fields)
    if not target_ok or not box_ok or not _canonical_bbox(box.get("bbox_xyxy")):
        raise DatasetBoundaryError(f"invalid excluded O base target: {label_path}")
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reviewed_variants(path: Path) -> frozenset[str]:
    """Load the exact reviewed CAD defect variant keys."""
    if path.is_symlink() or not path.is_file() or _sha256(path) != REVIEWED_DEFECT_MANIFEST_SHA256:
        raise DatasetBoundaryError(f"reviewed defect manifest is missing or changed: {path}")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetBoundaryError(f"reviewed defect manifest is invalid: {path}") from exc
    variants = payload.get("variants") if isinstance(payload, Mapping) else None
    if not isinstance(variants, Mapping) or not variants:
        raise DatasetBoundaryError(f"reviewed defect manifest is invalid: {path}")
    return frozenset(str(key) for key in variants)


def validate_variant(object_id: str, variant: str, reviewed: frozenset[str], label_path: Path) -> None:
    """Reject labels that do not name an exact reviewed part/defect mesh."""
    if variant != "ok" and f"{object_id}__{variant}" not in reviewed:
        raise DatasetBoundaryError(f"unreviewed defect variant in anomaly label: {label_path}")


def _expected_labels(chunk_index: int) -> set[str]:
    first = chunk_index * SOURCE_LABELS_PER_CHUNK
    return {f"{index:06d}.json" for index in range(first, first + SOURCE_LABELS_PER_CHUNK)}


def _o_target_id(path: Path) -> str:
    try:
        payload = json.loads(path.read_bytes())
        spec = payload["spec"]
        uid = spec["wrist_target_uid"]
        objects = spec["objects"]
        matches = [item for item in objects if isinstance(item, Mapping) and item.get("uid") == uid]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DatasetBoundaryError(f"invalid O target inventory label: {path}") from exc
    provenance_ok = (
        isinstance(spec, Mapping)
        and spec.get("set") == "O"
        and type(spec.get("index")) is int
        and path.stem.isdigit()
        and spec["index"] == int(path.stem)
    )
    object_id = matches[0].get("object_id") if len(matches) == 1 else None
    if not provenance_ok or object_id not in SOURCE_O_TARGET_COUNTS:
        raise DatasetBoundaryError(f"invalid O target inventory label: {path}")
    return object_id


def _inventory_set(set_name: str, set_path: Path, chunk_count: int) -> dict[str, Any]:
    if set_path.is_symlink() or not set_path.is_dir():
        raise DatasetBoundaryError(f"required synthetic set is missing or symlinked: {set_path}")
    expected_chunks = {f"chunk_{index:04d}" for index in range(chunk_count)}
    chunks = {path.name: path for path in set_path.glob("chunk_*")}
    if set(chunks) != expected_chunks:
        raise DatasetBoundaryError(f"{set_name} source chunk inventory is incomplete or has extras")
    target_counts: Counter[str] = Counter()
    for index, chunk_name in enumerate(sorted(expected_chunks)):
        chunk = chunks[chunk_name]
        labels = chunk / "labels"
        invalid = chunk.is_symlink() or not chunk.is_dir() or labels.is_symlink() or not labels.is_dir()
        if invalid:
            raise DatasetBoundaryError(f"{set_name} source chunk is not a real complete directory: {chunk}")
        actual_labels = {path.name for path in labels.iterdir() if path.is_file() and not path.is_symlink()}
        if actual_labels != _expected_labels(index):
            raise DatasetBoundaryError(f"{set_name}/{chunk_name} label inventory is incomplete or has extras")
        if set_name == "O":
            target_counts.update(_o_target_id(labels / name) for name in sorted(actual_labels))
    if set_name == "O" and dict(target_counts) != SOURCE_O_TARGET_COUNTS:
        raise DatasetBoundaryError("O source target inventory is incomplete, changed, or has extras")
    result: dict[str, Any] = {"chunks": chunk_count, "labels": chunk_count * SOURCE_LABELS_PER_CHUNK}
    if set_name == "O":
        result["target_counts"] = dict(sorted(target_counts.items()))
    return result


def validate_patchcore_source_inventory(new_root: Path, old_root: Path) -> dict[str, dict[str, Any]]:
    """Require the exact reviewed O/T/C/I chunk and label inventory before preparation."""
    new, old = Path(new_root), Path(old_root)
    if new.is_symlink() or old.is_symlink():
        raise DatasetBoundaryError("synthetic source roots must not be symlinks")
    roots = {"O": new, "T": new, "C": old, "I": old}
    return {
        set_name: _inventory_set(set_name, roots[set_name] / set_name, chunk_count)
        for set_name, chunk_count in SOURCE_CHUNK_COUNTS.items()
    }
