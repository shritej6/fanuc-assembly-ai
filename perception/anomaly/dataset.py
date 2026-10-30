"""Deterministic synthetic target discovery for per-part PatchCore datasets."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, UnidentifiedImageError

from perception.anomaly.rest_provenance import REVIEWED_MAX_TILT_DEG, REVIEWED_RULE
from perception.anomaly.source_contract import (
    DatasetBoundaryError,
    exclude_o_base_target,
    validate_variant,
)
from perception.anomaly.source_contract import reviewed_variants as load_reviewed_variants

SetName = Literal["C", "I", "O", "T"]
Eligibility = Literal["train_eligible", "false_alarm_only", "evaluation_only", "excluded"]
RestState = Literal["installed", "fallen"]
Split = Literal["train/good", "test/good", "test/bad", "o_test_good"]
T_EVALUATION_PARTS = frozenset({"M1_04", "M2_04", "M3_04"})
SPLIT_SEED = 1337


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
    label_path: Path
    rgb_sha256: str
    depth_sha256: str
    segmentation_sha256: str
    label_sha256: str
    rest_sidecar_path: Path | None
    rest_sidecar_sha256: str | None
    rest: RestState | None = None
    assigned_split: Split | None = None


@dataclass(frozen=True)
class PatchCoreObjectCounts:
    object_id: str
    train_good: int
    test_good: int
    test_bad: int
    o_test_good: int


@dataclass(frozen=True)
class PatchCorePreparationSummary:
    object_ids: tuple[str, ...]
    per_object: tuple[PatchCoreObjectCounts, ...]
    total: int
    train_good: int
    test_good: int
    test_bad: int
    o_test_good: int
    split_seed: int


def _read_hashed_json(path: Path) -> tuple[Any, str]:
    try:
        content = path.read_bytes()
        return json.loads(content), hashlib.sha256(content).hexdigest()
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


def _scene_split(set_name: str, index: int) -> str:
    token = f"{SPLIT_SEED}:{set_name}:{index}".encode()
    bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % 10_000
    if bucket < 8_000:
        return "train"
    return "validation" if bucket < 9_000 else "test"


def _rest_sidecar(chunk: Path, set_name: SetName, rest_root: Path) -> tuple[Mapping[str, Any], Path | None, str | None]:
    if set_name not in ("C", "I"):
        return {}, None, None
    path = rest_root / set_name / chunk.name / "_REST.json"
    if path.is_symlink() or not path.is_file():
        raise DatasetBoundaryError(f"required C/I _REST.json is missing or symlinked: {path}")
    payload, digest = _read_hashed_json(path)
    images = payload.get("images") if isinstance(payload, Mapping) else None
    provenance_matches = (
        isinstance(payload, Mapping)
        and payload.get("rule") == REVIEWED_RULE
        and payload.get("max_tilt_deg") == REVIEWED_MAX_TILT_DEG
    )
    if not provenance_matches or not isinstance(images, Mapping):
        raise DatasetBoundaryError(f"invalid C/I _REST.json provenance: {path}")
    return images, path.resolve(), digest


def _rest_state(
    target: Mapping[str, Any],
    set_name: SetName,
    label_path: Path,
    rests: Mapping[str, Any],
) -> RestState | None:
    if set_name == "O":
        return None
    if set_name == "T":
        rest = target.get("rest")
    else:
        image_rests = rests.get(label_path.stem)
        rest = image_rests.get(target.get("uid")) if isinstance(image_rests, Mapping) else None
    if rest not in ("installed", "fallen"):
        raise DatasetBoundaryError(f"invalid {set_name} rest provenance: {label_path}")
    return rest


def _assignment(
    set_name: SetName,
    index: int,
    object_id: str,
    variant: str,
    rest: RestState | None,
) -> tuple[Eligibility, Split | None]:
    if set_name == "O":
        if variant != "ok":
            return "excluded", None
        split = _scene_split("O", index)
        if split == "train":
            return "train_eligible", "train/good"
        if split == "test":
            return "false_alarm_only", "o_test_good"
        return "excluded", None
    if rest != "installed" or (set_name == "T" and object_id not in T_EVALUATION_PARTS):
        return "excluded", None
    return "evaluation_only", "test/good" if variant == "ok" else "test/bad"


def _metadata(target: Mapping[str, Any], box: Mapping[str, Any], label_path: Path) -> tuple[str, str, str, int]:
    uid, object_id = target.get("uid"), target.get("object_id")
    variant, stable_pose = target.get("variant"), target.get("stable_pose")
    valid_target = isinstance(uid, str) and uid.startswith("o") and uid[1:].isdigit()
    if (
        not valid_target
        or not isinstance(object_id, str)
        or re.fullmatch(r"M[1-6]_0[1-4]", object_id) is None
        or target.get("kind") != "part"
    ):
        raise DatasetBoundaryError(f"invalid anomaly target identity: {label_path}")
    if not isinstance(variant, str) or not variant or type(stable_pose) is not int:
        raise DatasetBoundaryError(f"invalid anomaly target metadata: {label_path}")
    if box.get("object_id") != object_id or box.get("variant") != variant:
        raise DatasetBoundaryError(f"wrist box disagrees with target metadata: {label_path}")
    return uid, object_id, variant, stable_pose


def _artifacts(chunk: Path, index: str) -> tuple[tuple[Path, Path, Path], tuple[str, str, str]]:
    rgb = _safe_artifact(chunk, Path("rgb/wrist") / f"{index}.jpg")
    depth = _safe_artifact(chunk, Path("depth/wrist") / f"{index}.png")
    segmentation = _safe_artifact(chunk, Path("seg/wrist") / f"{index}.png")
    paths = rgb, depth, segmentation
    return paths, tuple(_sha256(path) for path in paths)


def _label_index(spec: Mapping[str, Any], set_name: SetName, label_path: Path) -> str:
    index = label_path.stem
    valid = index.isdigit() and spec.get("set") == set_name
    valid = valid and type(spec.get("index")) is int and spec["index"] == int(index)
    if not valid:
        raise DatasetBoundaryError(f"label provenance disagrees with its path: {label_path}")
    return index


def _sample(
    label_path: Path,
    chunk: Path,
    set_name: SetName,
    rest_evidence: tuple[Mapping[str, Any], Path | None, str | None],
    reviewed_variants: frozenset[str],
) -> PatchCoreSample | None:
    payload, label_sha256 = _read_hashed_json(label_path)
    spec, target, box = _target(payload, label_path)
    index = _label_index(spec, set_name, label_path)
    if exclude_o_base_target(set_name, target, box, label_path):
        return None
    uid, object_id, variant, stable_pose = _metadata(target, box, label_path)
    validate_variant(object_id, variant, reviewed_variants, label_path)
    rests, rest_path, rest_sha256 = rest_evidence
    rest = _rest_state(target, set_name, label_path, rests)
    (rgb, depth, segmentation), hashes = _artifacts(chunk, index)
    eligibility, assigned_split = _assignment(set_name, int(index), object_id, variant, rest)
    return PatchCoreSample(
        set_name=set_name,
        sample_id=f"{set_name}/{chunk.name}/{index}",
        object_id=object_id,
        variant=variant,
        stable_pose=stable_pose,
        eligibility=eligibility,
        bbox_xyxy=_bbox(box.get("bbox_xyxy"), label_path),
        segmentation_value=int(uid[1:]) + 1,
        rgb_path=rgb,
        depth_path=depth,
        segmentation_path=segmentation,
        label_path=label_path.resolve(),
        rgb_sha256=hashes[0],
        depth_sha256=hashes[1],
        segmentation_sha256=hashes[2],
        label_sha256=label_sha256,
        rest_sidecar_path=rest_path,
        rest_sidecar_sha256=rest_sha256,
        rest=rest,
        assigned_split=assigned_split,
    )


def _existing_sources(new: Path, old: Path) -> tuple[tuple[SetName, Path], ...]:
    candidates: tuple[tuple[SetName, Path], ...] = (
        ("O", new / "O"),
        ("C", old / "C"),
        ("I", old / "I"),
        ("T", new / "T"),
    )
    return tuple(item for item in candidates if item[1].exists())


def discover_patchcore_samples(
    new_root: Path,
    *,
    old_root: Path,
    rest_root: Path,
    defect_manifest: Path | None = None,
) -> tuple[PatchCoreSample, ...]:
    """Discover O-only training and installed C/I/T evaluation targets."""
    new, old, rest = Path(new_root), Path(old_root), Path(rest_root)
    if new.is_symlink() or old.is_symlink() or rest.is_symlink():
        raise DatasetBoundaryError("synthetic source roots must not be symlinks")
    manifest = defect_manifest or Path(__file__).resolve().parents[2] / "cad/defects/manifest.json"
    reviewed_variants = load_reviewed_variants(Path(manifest))
    samples: list[PatchCoreSample] = []
    for set_name, set_path in _existing_sources(new, old):
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
            rest_evidence = _rest_sidecar(chunk, set_name, rest)
            for label_path in sorted(labels.glob("*.json")):
                if label_path.is_symlink():
                    raise DatasetBoundaryError(f"label path must not be a symlink: {label_path}")
                sample = _sample(label_path, chunk, set_name, rest_evidence, reviewed_variants)
                if sample is not None and sample.assigned_split is not None:
                    samples.append(sample)
    if not samples:
        raise DatasetBoundaryError("no approved O/C/I/T anomaly labels found")
    return tuple(samples)


def _assign_splits(samples: Iterable[PatchCoreSample]) -> list[tuple[PatchCoreSample, Split]]:
    assigned = []
    ordered = sorted(samples, key=lambda item: (item.object_id, item.sample_id))
    sample_ids = [sample.sample_id for sample in ordered]
    if len(sample_ids) != len(set(sample_ids)):
        raise DatasetPreparationError("duplicate sample ID in PatchCore preparation")
    for sample in ordered:
        index = int(sample.sample_id.rsplit("/", 1)[1])
        expected = _assignment(sample.set_name, index, sample.object_id, sample.variant, sample.rest)
        if (sample.eligibility, sample.assigned_split) != expected or sample.assigned_split is None:
            raise DatasetPreparationError(f"sample assignment is inconsistent: {sample.sample_id}")
        assigned.append((sample, sample.assigned_split))
    return assigned


def _validated_crop(sample: PatchCoreSample) -> tuple[int, int, int, int]:
    value = sample.bbox_xyxy
    if len(value) != 4 or any(type(item) is not int for item in value):
        raise DatasetPreparationError(f"invalid crop coordinates: {sample.sample_id}")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 and 0 <= y0 < y1):
        raise DatasetPreparationError(f"invalid crop coordinates: {sample.sample_id}")
    return x0, y0, x1, y1


def _source_hashes(sample: PatchCoreSample) -> dict[str, str | None]:
    sources = {
        "source_rgb_sha256": (sample.rgb_path, sample.rgb_sha256),
        "source_depth_sha256": (sample.depth_path, sample.depth_sha256),
        "source_segmentation_sha256": (sample.segmentation_path, sample.segmentation_sha256),
        "source_label_sha256": (sample.label_path, sample.label_sha256),
    }
    for label, (path, expected) in sources.items():
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise DatasetPreparationError(f"source hash changed ({label}): {sample.sample_id}")
    rest_path, rest_hash = sample.rest_sidecar_path, sample.rest_sidecar_sha256
    if (rest_path is None) != (rest_hash is None):
        raise DatasetPreparationError(f"rest sidecar identity is incomplete: {sample.sample_id}")
    if rest_path is not None and (rest_path.is_symlink() or not rest_path.is_file() or _sha256(rest_path) != rest_hash):
        raise DatasetPreparationError(f"source hash changed (rest sidecar): {sample.sample_id}")
    return {label: expected for label, (_, expected) in sources.items()} | {"source_rest_sha256": rest_hash}


def _crop_sample(sample: PatchCoreSample, destination: Path) -> str:
    _source_hashes(sample)
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
    required = ("train/good", "test/good", "test/bad", "o_test_good")
    if any(not all(counter[split] for split in required) for counter in counts.values()):
        raise DatasetPreparationError("each expected object must have complete splits")
    return tuple(
        PatchCoreObjectCounts(
            object_id,
            counter["train/good"],
            counter["test/good"],
            counter["test/bad"],
            counter["o_test_good"],
        )
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
) -> PatchCorePreparationSummary:
    counts = Counter(split for _, split in assigned)
    return PatchCorePreparationSummary(
        object_ids=tuple(item.object_id for item in per_object),
        per_object=per_object,
        total=len(assigned),
        train_good=counts["train/good"],
        test_good=counts["test/good"],
        test_bad=counts["test/bad"],
        o_test_good=counts["o_test_good"],
        split_seed=SPLIT_SEED,
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
                **_source_hashes(sample),
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
) -> PatchCorePreparationSummary:
    """Create deterministic per-part anomalib folders from verified bbox crops."""
    root = Path(output_root)
    if root.exists():
        raise DatasetPreparationError(f"output root must not already exist: {root}")
    assigned = _assign_splits(samples)
    if not assigned:
        raise DatasetPreparationError("at least one PatchCore sample is required")
    per_object = _validate_populations(assigned, expected_object_ids)
    root.mkdir(parents=True)
    incomplete = root / "_INCOMPLETE"
    incomplete.write_text("preparation in progress\n")
    rows = _materialize_rows(root, assigned)
    summary = _summary(assigned, per_object)
    _finalize_preparation(root, rows, summary)
    return summary
