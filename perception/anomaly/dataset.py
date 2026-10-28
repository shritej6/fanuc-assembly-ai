"""Deterministic C/I target discovery for per-part PatchCore datasets."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


class DatasetBoundaryError(ValueError):
    """Raised when synthetic anomaly data cannot be proven safe to consume."""


@dataclass(frozen=True)
class PatchCoreSample:
    set_name: Literal["C", "I"]
    sample_id: str
    object_id: str
    variant: str
    stable_pose: int
    eligibility: Literal["train_eligible", "test_only"]
    bbox_xyxy: tuple[int, int, int, int]
    segmentation_value: int
    rgb_path: Path
    depth_path: Path
    segmentation_path: Path
    rgb_sha256: str


def _read_label(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetBoundaryError(f"cannot read label {path}: {exc}") from exc


def _target(
    payload: Any, label_path: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    try:
        spec = payload["spec"]
        uid = spec["wrist_target_uid"]
        raw_objects = spec["objects"]
        raw_boxes = payload["cameras"]["wrist"]
        if not isinstance(spec, Mapping) or not all(isinstance(item, Mapping) for item in raw_objects):
            raise TypeError
        if not all(isinstance(item, Mapping) for item in raw_boxes):
            raise TypeError
        objects = [item for item in raw_objects if item.get("uid") == uid]
        boxes = [item for item in raw_boxes if item.get("uid") == uid]
    except (KeyError, TypeError, AttributeError) as exc:
        raise DatasetBoundaryError(f"invalid wrist target label: {label_path}") from exc
    visible_pixels = boxes[0].get("visible_px") if len(boxes) == 1 else None
    if len(objects) != 1 or len(boxes) != 1 or type(visible_pixels) is not int or visible_pixels <= 0:
        raise DatasetBoundaryError(f"label lacks exactly one visible wrist target: {label_path}")
    return spec, objects[0], boxes[0]


def _bbox(value: Any, label_path: Path) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4 or any(type(item) is not int for item in value):
        raise DatasetBoundaryError(f"invalid C/I target metadata: {label_path}")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 and 0 <= y0 < y1):
        raise DatasetBoundaryError(f"invalid C/I target metadata: {label_path}")
    return x0, y0, x1, y1


def _safe_artifact(chunk: Path, relative: Path) -> Path:
    path = chunk / relative
    cursor = chunk
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise DatasetBoundaryError(f"artifact path contains a symlink: {path}")
    if not path.is_file():
        raise DatasetBoundaryError(f"required anomaly artifact is missing: {path}")
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample(label_path: Path, chunk: Path, set_name: str) -> PatchCoreSample:
    payload = _read_label(label_path)
    spec, target, box = _target(payload, label_path)
    if not label_path.stem.isdigit() or spec.get("set") != set_name:
        raise DatasetBoundaryError(f"label provenance disagrees with its path: {label_path}")
    if type(spec.get("index")) is not int or spec["index"] != int(label_path.stem):
        raise DatasetBoundaryError(f"label provenance disagrees with its path: {label_path}")
    uid, object_id = target.get("uid"), target.get("object_id")
    variant, stable_pose = target.get("variant"), target.get("stable_pose")
    bbox = _bbox(box.get("bbox_xyxy"), label_path)
    valid_target = isinstance(uid, str) and uid.startswith("o") and uid[1:].isdigit()
    if not valid_target or not isinstance(object_id, str) or not object_id.startswith("M"):
        raise DatasetBoundaryError(f"invalid C/I target identity: {label_path}")
    if not isinstance(variant, str) or not variant or type(stable_pose) is not int:
        raise DatasetBoundaryError(f"invalid C/I target metadata: {label_path}")
    if box.get("object_id") != object_id or box.get("variant") != variant:
        raise DatasetBoundaryError(f"wrist box disagrees with target metadata: {label_path}")
    index = label_path.stem
    rgb = _safe_artifact(chunk, Path("rgb/wrist") / f"{index}.jpg")
    depth = _safe_artifact(chunk, Path("depth/wrist") / f"{index}.png")
    segmentation = _safe_artifact(chunk, Path("seg/wrist") / f"{index}.png")
    return PatchCoreSample(
        set_name=set_name, sample_id=f"{set_name}/{chunk.name}/{index}", object_id=object_id,
        variant=variant, stable_pose=stable_pose,
        eligibility="train_eligible" if variant == "ok" else "test_only",
        bbox_xyxy=bbox, segmentation_value=int(uid[1:]) + 1,
        rgb_path=rgb, depth_path=depth, segmentation_path=segmentation, rgb_sha256=_sha256(rgb),
    )


def discover_patchcore_samples(synthetic_root: Path) -> tuple[PatchCoreSample, ...]:
    """Discover only approved C/I wrist targets without assigning an unmeasured split."""
    root = Path(synthetic_root)
    if root.is_symlink():
        raise DatasetBoundaryError(f"synthetic root must not be a symlink: {root}")
    samples: list[PatchCoreSample] = []
    for set_name in ("C", "I"):
        set_path = root / set_name
        if not set_path.exists():
            continue
        if set_path.is_symlink():
            raise DatasetBoundaryError(f"synthetic set path must not be a symlink: {set_path}")
        for chunk in sorted(set_path.glob("chunk_*")):
            if chunk.is_symlink() or not chunk.is_dir():
                raise DatasetBoundaryError(f"chunk path must be a real directory, not a symlink: {chunk}")
            if not re.fullmatch(r"chunk_\d{4}", chunk.name):
                raise DatasetBoundaryError(f"noncanonical chunk name is forbidden: {chunk}")
            labels = chunk / "labels"
            if labels.is_symlink() or not labels.is_dir():
                raise DatasetBoundaryError(f"labels path must be a real directory, not a symlink: {labels}")
            for label_path in sorted(labels.glob("*.json")):
                if label_path.is_symlink():
                    raise DatasetBoundaryError(f"label path must not be a symlink: {label_path}")
                samples.append(_sample(label_path, chunk, set_name))
    if not samples:
        raise DatasetBoundaryError(f"no C/I anomaly labels found under {root}")
    return tuple(samples)
