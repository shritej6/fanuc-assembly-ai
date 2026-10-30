"""Deterministic synthetic target discovery for per-part PatchCore datasets."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, UnidentifiedImageError

SetName = Literal["C", "I", "T"]
Eligibility = Literal["train_eligible", "test_only", "evaluation_only"]
RestState = Literal["installed", "fallen"]
Split = Literal["train/good", "test/good", "test/bad"]
T_EVALUATION_PARTS = frozenset({"M1_04", "M2_04", "M3_04"})


class DatasetBoundaryError(ValueError):
    """Raised when synthetic anomaly data cannot be proven safe to consume."""


class DatasetPreparationError(DatasetBoundaryError):
    """Raised when a discovered sample cannot be materialized reproducibly."""


@dataclass(frozen=True)
class PatchCoreSample:
    set_name: SetName
    sample_id: str
    object_id: str
    variant: str
    stable_pose: int
    eligibility: Eligibility
    bbox_xyxy: tuple[int, int, int, int]
    segmentation_value: int
    rgb_path: Path
    depth_path: Path
    segmentation_path: Path
    rgb_sha256: str
    rest: RestState | None = None


@dataclass(frozen=True)
class PatchCoreObjectCounts:
    object_id: str
    train_good: int
    test_good: int
    test_bad: int


@dataclass(frozen=True)
class PatchCorePreparationSummary:
    object_ids: tuple[str, ...]
    per_object: tuple[PatchCoreObjectCounts, ...]
    total: int
    train_good: int
    test_good: int
    test_bad: int
    normal_test_modulus: int


def _read_label(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetBoundaryError(f"cannot read label {path}: {exc}") from exc


def _target(payload: Any, label_path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
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
        raise DatasetBoundaryError(f"invalid anomaly target metadata: {label_path}")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 and 0 <= y0 < y1):
        raise DatasetBoundaryError(f"invalid anomaly target metadata: {label_path}")
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


def _rest_state(target: Mapping[str, Any], set_name: SetName, label_path: Path) -> RestState | None:
    if set_name != "T":
        return None
    rest = target.get("rest")
    if rest not in ("installed", "fallen"):
        raise DatasetBoundaryError(f"invalid T rest provenance: {label_path}")
    return rest


def _eligibility(set_name: SetName, variant: str) -> Eligibility:
    if set_name == "T":
        return "evaluation_only"
    return "train_eligible" if variant == "ok" else "test_only"


def _metadata(target: Mapping[str, Any], box: Mapping[str, Any], label_path: Path) -> tuple[str, str, str, int]:
    uid, object_id = target.get("uid"), target.get("object_id")
    variant, stable_pose = target.get("variant"), target.get("stable_pose")
    valid_target = isinstance(uid, str) and uid.startswith("o") and uid[1:].isdigit()
    if (
        not valid_target
        or not isinstance(object_id, str)
        or not object_id.startswith("M")
        or target.get("kind") != "part"
    ):
        raise DatasetBoundaryError(f"invalid anomaly target identity: {label_path}")
    if not isinstance(variant, str) or not variant or type(stable_pose) is not int:
        raise DatasetBoundaryError(f"invalid anomaly target metadata: {label_path}")
    if box.get("object_id") != object_id or box.get("variant") != variant:
        raise DatasetBoundaryError(f"wrist box disagrees with target metadata: {label_path}")
    return uid, object_id, variant, stable_pose


def _artifacts(chunk: Path, index: str) -> tuple[Path, Path, Path, str]:
    rgb = _safe_artifact(chunk, Path("rgb/wrist") / f"{index}.jpg")
    depth = _safe_artifact(chunk, Path("depth/wrist") / f"{index}.png")
    segmentation = _safe_artifact(chunk, Path("seg/wrist") / f"{index}.png")
    return rgb, depth, segmentation, _sha256(rgb)


def _sample(label_path: Path, chunk: Path, set_name: SetName) -> PatchCoreSample:
    payload = _read_label(label_path)
    spec, target, box = _target(payload, label_path)
    if not label_path.stem.isdigit() or spec.get("set") != set_name:
        raise DatasetBoundaryError(f"label provenance disagrees with its path: {label_path}")
    if type(spec.get("index")) is not int or spec["index"] != int(label_path.stem):
        raise DatasetBoundaryError(f"label provenance disagrees with its path: {label_path}")
    uid, object_id, variant, stable_pose = _metadata(target, box, label_path)
    bbox = _bbox(box.get("bbox_xyxy"), label_path)
    rest = _rest_state(target, set_name, label_path)
    index = label_path.stem
    rgb, depth, segmentation, rgb_sha256 = _artifacts(chunk, index)
    return PatchCoreSample(
        set_name=set_name,
        sample_id=f"{set_name}/{chunk.name}/{index}",
        object_id=object_id,
        variant=variant,
        stable_pose=stable_pose,
        eligibility=_eligibility(set_name, variant),
        bbox_xyxy=bbox,
        segmentation_value=int(uid[1:]) + 1,
        rgb_path=rgb,
        depth_path=depth,
        segmentation_path=segmentation,
        rgb_sha256=rgb_sha256,
        rest=rest,
    )


def _approved_sample(sample: PatchCoreSample) -> bool:
    if sample.set_name != "T":
        return True
    return sample.object_id in T_EVALUATION_PARTS and sample.rest == "installed"


def discover_patchcore_samples(synthetic_root: Path) -> tuple[PatchCoreSample, ...]:
    """Discover C/I training candidates and installed rear-guard T evaluation targets."""
    root = Path(synthetic_root)
    if root.is_symlink():
        raise DatasetBoundaryError(f"synthetic root must not be a symlink: {root}")
    samples: list[PatchCoreSample] = []
    for set_name in ("C", "I", "T"):
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
                sample = _sample(label_path, chunk, set_name)
                if _approved_sample(sample):
                    samples.append(sample)
    if not samples:
        raise DatasetBoundaryError(f"no C/I/T anomaly labels found under {root}")
    return tuple(samples)


def _assign_splits(samples: Iterable[PatchCoreSample], modulus: int) -> list[tuple[PatchCoreSample, Split]]:
    positions: defaultdict[str, int] = defaultdict(int)
    assigned = []
    ordered = sorted(samples, key=lambda item: (item.object_id, item.sample_id))
    sample_ids = [sample.sample_id for sample in ordered]
    if len(sample_ids) != len(set(sample_ids)):
        raise DatasetPreparationError("duplicate sample ID in PatchCore preparation")
    for sample in ordered:
        expected = _eligibility(sample.set_name, sample.variant)
        if sample.eligibility != expected:
            raise DatasetPreparationError(f"eligibility disagrees with variant: {sample.sample_id}")
        if sample.eligibility == "evaluation_only":
            split = "test/good" if sample.variant == "ok" else "test/bad"
        elif sample.variant != "ok":
            split: Split = "test/bad"
        else:
            split = "test/good" if positions[sample.object_id] % modulus == 0 else "train/good"
            positions[sample.object_id] += 1
        assigned.append((sample, split))
    return assigned


def _validated_crop(sample: PatchCoreSample) -> tuple[int, int, int, int]:
    value = sample.bbox_xyxy
    if len(value) != 4 or any(type(item) is not int for item in value):
        raise DatasetPreparationError(f"invalid crop coordinates: {sample.sample_id}")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 and 0 <= y0 < y1):
        raise DatasetPreparationError(f"invalid crop coordinates: {sample.sample_id}")
    return x0, y0, x1, y1


def _crop_sample(sample: PatchCoreSample, destination: Path) -> str:
    if _sha256(sample.rgb_path) != sample.rgb_sha256:
        raise DatasetPreparationError(f"source RGB hash changed: {sample.sample_id}")
    try:
        with Image.open(sample.rgb_path) as source:
            image = source.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise DatasetPreparationError(f"cannot decode source RGB: {sample.sample_id}") from exc
    x0, y0, x1, y1 = _validated_crop(sample)
    if x1 > image.width or y1 > image.height:
        raise DatasetPreparationError(f"crop lies outside source RGB: {sample.sample_id}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.crop((x0, y0, x1, y1)).save(destination, format="PNG")
    return _sha256(destination)


def _prepared_name(sample: PatchCoreSample) -> str:
    stable_id = sample.sample_id.replace("/", "__")
    return f"{stable_id}__{sample.rgb_sha256[:12]}.png"


def _validate_populations(
    assigned: list[tuple[PatchCoreSample, str]], expected_object_ids: Iterable[str]
) -> tuple[PatchCoreObjectCounts, ...]:
    expected = tuple(expected_object_ids)
    if not expected or len(expected) != len(set(expected)):
        raise DatasetPreparationError("expected object IDs must be non-empty and unique")
    if any(re.fullmatch(r"M[1-6]_0[1-4]", object_id) is None for object_id in expected):
        raise DatasetPreparationError("expected object IDs must be exact taxonomy part IDs")
    actual = {sample.object_id for sample, _ in assigned}
    if actual != set(expected):
        raise DatasetPreparationError("samples must match the expected object IDs exactly")
    counts = {
        object_id: Counter(split for sample, split in assigned if sample.object_id == object_id)
        for object_id in expected
    }
    if any(not all(counter[split] for split in ("train/good", "test/good", "test/bad")) for counter in counts.values()):
        raise DatasetPreparationError("each expected object must have complete splits")
    return tuple(
        PatchCoreObjectCounts(object_id, counter["train/good"], counter["test/good"], counter["test/bad"])
        for object_id, counter in counts.items()
    )


def _destination(root: Path, sample: PatchCoreSample, split: str) -> tuple[Path, Path]:
    relative = Path(sample.object_id) / split / _prepared_name(sample)
    destination = (root / relative).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise DatasetPreparationError(f"crop destination escapes output root: {sample.sample_id}")
    return relative, destination


def _summary(
    assigned: list[tuple[PatchCoreSample, str]],
    per_object: tuple[PatchCoreObjectCounts, ...],
    modulus: int,
) -> PatchCorePreparationSummary:
    counts = Counter(split for _, split in assigned)
    return PatchCorePreparationSummary(
        object_ids=tuple(item.object_id for item in per_object),
        per_object=per_object,
        total=len(assigned),
        train_good=counts["train/good"],
        test_good=counts["test/good"],
        test_bad=counts["test/bad"],
        normal_test_modulus=modulus,
    )


def _finalize_preparation(root: Path, rows: list[dict[str, Any]], summary: PatchCorePreparationSummary) -> None:
    (root / "manifest.jsonl").write_text("".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows))
    temporary = root / "_READY.json.tmp"
    temporary.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n")
    (root / "_INCOMPLETE").unlink()
    temporary.replace(root / "_READY.json")


def _materialize_rows(root: Path, assigned: Iterable[tuple[PatchCoreSample, str]]) -> list[dict[str, Any]]:
    rows = []
    output_paths: set[Path] = set()
    for sample, split in assigned:
        relative, destination = _destination(root, sample, split)
        if destination in output_paths:
            raise DatasetPreparationError(f"duplicate crop destination: {relative}")
        output_paths.add(destination)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "object_id": sample.object_id,
                "variant": sample.variant,
                "stable_pose": sample.stable_pose,
                "rest": sample.rest,
                "split": split,
                "source_rgb_sha256": sample.rgb_sha256,
                "crop_sha256": _crop_sample(sample, destination),
                "output": relative.as_posix(),
            }
        )
    return rows


def prepare_patchcore_folders(
    samples: Iterable[PatchCoreSample],
    output_root: Path,
    *,
    expected_object_ids: Iterable[str],
    normal_test_modulus: int,
) -> PatchCorePreparationSummary:
    """Create deterministic per-part anomalib folders from verified bbox crops."""
    root = Path(output_root)
    if root.exists():
        raise DatasetPreparationError(f"output root must not already exist: {root}")
    if normal_test_modulus < 2:
        raise DatasetPreparationError("normal_test_modulus must be at least 2")
    assigned = _assign_splits(samples, normal_test_modulus)
    if not assigned:
        raise DatasetPreparationError("at least one PatchCore sample is required")
    per_object = _validate_populations(assigned, expected_object_ids)
    root.mkdir(parents=True)
    incomplete = root / "_INCOMPLETE"
    incomplete.write_text("preparation in progress\n")
    rows = _materialize_rows(root, assigned)
    summary = _summary(assigned, per_object, normal_test_modulus)
    _finalize_preparation(root, rows, summary)
    return summary
