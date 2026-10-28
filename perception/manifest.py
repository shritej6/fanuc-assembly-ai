"""Validate SDG chunks and freeze leak-free T13 training manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from perception.provenance import current_git_commit
from perception.records import Annotation, ImageRecord
from perception.taxonomy import Ontology, load_ontology

CHUNK_PATTERN = re.compile(r"chunk_[0-9]{4}")
FORBIDDEN_PARTS = {"rejected", "superseded", "real", "real-cell", "real_cell"}


class ManifestError(ValueError):
    """Raised when mutable source data cannot be frozen safely."""


@dataclass(frozen=True)
class ManifestConfig:
    data_root: Path
    output_dir: Path
    taxonomy_path: Path
    detector_sets: tuple[str, ...] = ("A", "F", "G")
    classifier_sets: tuple[str, ...] = ("B", "H", "K")
    split_seed: int = 1337
    require_cosmos: bool = True
    include_cosmos: bool = True
    max_scenes_per_set: int | None = None
    min_cosmos_fraction: float = 0.10
    max_cosmos_fraction: float = 0.20


@dataclass(frozen=True)
class ManifestSummary:
    detector_path: Path
    classifier_path: Path
    metadata_path: Path
    detector_sha256: str
    classifier_sha256: str
    detector_records: int
    classifier_records: int
    skipped_classifier_targets: dict[str, int]


def _validate_config(config: ManifestConfig) -> None:
    if config.require_cosmos and not config.include_cosmos:
        raise ManifestError("a Cosmos-required manifest must include Cosmos records")
    limit = config.max_scenes_per_set
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ManifestError("max_scenes_per_set must be a positive integer")


def split_for_scene(set_name: str, index: int, seed: int) -> str:
    """Assign an entire scene to an 80/10/10 split deterministically."""
    token = f"{seed}:{set_name}:{index}".encode()
    bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % 10_000
    if bucket < 8_000:
        return "train"
    if bucket < 9_000:
        return "validation"
    return "test"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"expected JSON object in {path}")
    return value


def _parse_label(path: Path, raw_bytes: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot parse label {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"cannot parse label {path}: expected JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return current_git_commit()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _jsonl(records: list[ImageRecord]) -> str:
    ordered = sorted(records, key=ImageRecord.sort_key)
    lines = (json.dumps(record.as_dict(), sort_keys=True, separators=(",", ":")) for record in ordered)
    return "".join(f"{line}\n" for line in lines)


def _chunk_dirs(set_dir: Path) -> list[Path]:
    if not set_dir.is_dir():
        raise ManifestError(f"set directory does not exist: {set_dir}")
    chunks = sorted(path for path in set_dir.iterdir() if path.is_dir() and CHUNK_PATTERN.fullmatch(path.name))
    if not chunks:
        raise ManifestError(f"no completed chunks found in {set_dir}")
    return chunks


def _validated_labels(chunk: Path) -> tuple[list[Path], str]:
    done_path = chunk / "_DONE.json"
    qa_path = chunk / "_QA.json"
    if not done_path.is_file():
        raise ManifestError(f"missing _DONE.json in {chunk}")
    if not qa_path.is_file():
        raise ManifestError(f"missing _QA.json in {chunk}")
    done, qa = _read_json(done_path), _read_json(qa_path)
    if qa.get("ok") is not True:
        raise ManifestError(f"QA failed for {chunk}")
    labels = sorted((chunk / "labels").glob("*.json"))
    counts = (done.get("scenes"), qa.get("labelled_scenes"), qa.get("expected_scenes"), len(labels))
    if not all(isinstance(value, int) for value in counts) or len(set(counts)) != 1:
        raise ManifestError(f"scene count mismatch in {chunk}: {counts}")
    return labels, str(done.get("code_commit", "unknown"))


def _cosmos_entries(chunk: Path) -> dict[str, dict[str, Any]]:
    manifest_path = chunk / "_COSMOS.json"
    entries = _read_json(manifest_path) if manifest_path.exists() else {}
    expected = {key for key, value in entries.items() if value.get("accepted") is True}
    cosmos_root = chunk / "rgb_cosmos"
    existing = {
        f"{path.parent.name}/{path.stem}"
        for path in cosmos_root.glob("*/*.jpg")
        if path.is_file()
    }
    missing = expected - existing
    if missing:
        raise ManifestError(f"accepted Cosmos file is missing in {chunk}: {sorted(missing)}")
    orphan = existing - expected
    if orphan:
        raise ManifestError(f"orphan Cosmos file without acceptance evidence in {chunk}: {sorted(orphan)}")
    return entries


def _image_size(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise ManifestError(f"missing image: {path}")
    try:
        with Image.open(path) as image:
            return image.size
    except OSError as exc:
        raise ManifestError(f"unreadable image {path}: {exc}") from exc


def _bbox(raw: Any, width: int, height: int, context: str) -> tuple[int, int, int, int]:
    if not isinstance(raw, list) or len(raw) != 4 or any(not isinstance(value, int) for value in raw):
        raise ManifestError(f"invalid bbox in {context}: {raw!r}")
    x0, y0, x1, y1 = raw
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ManifestError(f"out-of-bounds bbox in {context}: {raw!r} for {width}x{height}")
    return x0, y0, x1, y1


def _annotation(
    raw: dict[str, Any],
    label2id: dict[str, int],
    width: int,
    height: int,
    context: str,
) -> Annotation:
    object_id = raw.get("object_id")
    if object_id not in label2id:
        raise ManifestError(f"unknown object_id {object_id!r} in {context}")
    return Annotation(
        uid=str(raw["uid"]),
        object_id=object_id,
        class_id=label2id[object_id],
        bbox_xyxy=_bbox(raw.get("bbox_xyxy"), width, height, context),
        kind=str(raw.get("kind", "")),
        state=str(raw.get("state", "")),
        variant=str(raw.get("variant", "")),
        build_state=raw.get("build_state"),
    )


class _ManifestBuilder:
    def __init__(self, config: ManifestConfig, ontology: Ontology) -> None:
        self.config = config
        self.ontology = ontology
        self.detector: list[ImageRecord] = []
        self.classifier: list[ImageRecord] = []
        self.skipped: dict[str, int] = {}
        self.stats: dict[str, dict[str, Any]] = {}
        self.label_hashes = {"detector": hashlib.sha256(), "classifier": hashlib.sha256()}

    def build(self) -> ManifestSummary:
        self._validate_root()
        for set_name in self.config.detector_sets:
            self._process_set(set_name, "detector")
        for set_name in self.config.classifier_sets:
            self._process_set(set_name, "classifier")
        self._check_cosmos_coverage()
        return self._write()

    def _validate_root(self) -> None:
        root = self.config.data_root.resolve()
        if {part.casefold() for part in root.parts} & FORBIDDEN_PARTS:
            raise ManifestError(f"forbidden data root: {root}")
        if not (root / "synthetic").is_dir():
            raise ManifestError(f"synthetic data root does not exist: {root / 'synthetic'}")

    def _process_set(self, set_name: str, task: str) -> None:
        stats = {
            "scenes": 0,
            "native_records": 0,
            "cosmos_checked": 0,
            "cosmos_accepted": 0,
            "cosmos_used": 0,
            "code_commits": set(),
        }
        self.stats[set_name] = stats
        set_dir = self.config.data_root.resolve() / "synthetic" / set_name
        remaining = self.config.max_scenes_per_set
        for chunk in _chunk_dirs(set_dir):
            if remaining == 0:
                break
            labels, code_commit = _validated_labels(chunk)
            if remaining is not None:
                labels = labels[:remaining]
                remaining -= len(labels)
            stats["code_commits"].add(code_commit)
            stats["scenes"] += len(labels)
            entries = _cosmos_entries(chunk) if self.config.include_cosmos else {}
            stats["cosmos_checked"] += len(entries)
            stats["cosmos_accepted"] += sum(value.get("accepted") is True for value in entries.values())
            self._process_chunk(chunk, labels, entries, set_name, task)

    def _process_chunk(
        self,
        chunk: Path,
        labels: list[Path],
        cosmos: dict[str, dict[str, Any]],
        set_name: str,
        task: str,
    ) -> None:
        native: dict[str, ImageRecord] = {}
        labelled_keys: set[str] = set()
        for label_path in labels:
            raw_bytes = label_path.read_bytes()
            self.label_hashes[task].update(str(label_path).encode() + b"\0" + raw_bytes)
            label = _parse_label(label_path, raw_bytes)
            labelled_keys.update(f"{camera}/{label_path.stem}" for camera in label.get("cameras", {}))
            records = self._records_for_label(chunk, label_path, label, set_name, task)
            for record in records:
                native[f"{record.camera}/{record.index:06d}"] = record
                self._append(record)
        accepted_keys = {key for key, value in cosmos.items() if value.get("accepted") is True}
        unknown_keys = accepted_keys - labelled_keys
        if unknown_keys:
            raise ManifestError(f"accepted Cosmos entry has no matching native record: {sorted(unknown_keys)}")
        for key, value in sorted(cosmos.items()):
            if value.get("accepted") is True and key in native:
                self._append_cosmos(chunk, key, native[key])

    def _records_for_label(
        self,
        chunk: Path,
        label_path: Path,
        label: dict[str, Any],
        set_name: str,
        task: str,
    ) -> list[ImageRecord]:
        spec = label.get("spec", {})
        index = int(label_path.stem)
        if spec.get("set") != set_name or spec.get("index") != index:
            raise ManifestError(f"label identity mismatch in {label_path}")
        if task == "detector":
            return [
                self._detector_record(chunk, label_path, set_name, index, camera, raw)
                for camera, raw in sorted(label.get("cameras", {}).items())
            ]
        record = self._classifier_record(chunk, label_path, set_name, index, label)
        return [record] if record else []

    def _detector_record(
        self,
        chunk: Path,
        label_path: Path,
        set_name: str,
        index: int,
        camera: str,
        raw_objects: list[dict[str, Any]],
    ) -> ImageRecord:
        image_path = (chunk / "rgb" / camera / f"{index:06d}.jpg").resolve()
        width, height = _image_size(image_path)
        context = f"{set_name}:{index:06d}/{camera}"
        objects = tuple(
            _annotation(raw, self.ontology.detector_label2id, width, height, context) for raw in raw_objects
        )
        return self._record("detector", set_name, index, camera, image_path, label_path, width, height, objects)

    def _classifier_record(
        self,
        chunk: Path,
        label_path: Path,
        set_name: str,
        index: int,
        label: dict[str, Any],
    ) -> ImageRecord | None:
        camera = "wrist"
        image_path = (chunk / "rgb" / camera / f"{index:06d}.jpg").resolve()
        width, height = _image_size(image_path)
        target_uid = label.get("spec", {}).get("wrist_target_uid")
        target_raw = next((item for item in label.get("cameras", {}).get(camera, []) if item.get("uid") == target_uid), None)
        if target_raw is None:
            self.skipped[set_name] = self.skipped.get(set_name, 0) + 1
            return None
        target = _annotation(
            target_raw,
            self.ontology.classifier_label2id,
            width,
            height,
            f"{set_name}:{index:06d}/{camera}",
        )
        return self._record("classifier", set_name, index, camera, image_path, label_path, width, height, (), target)

    def _record(
        self,
        task: str,
        set_name: str,
        index: int,
        camera: str,
        image_path: Path,
        label_path: Path,
        width: int,
        height: int,
        objects: tuple[Annotation, ...] = (),
        target: Annotation | None = None,
    ) -> ImageRecord:
        return ImageRecord(
            task=task,
            scene_id=f"{set_name}:{index:06d}",
            set_name=set_name,
            index=index,
            camera=camera,
            split=split_for_scene(set_name, index, self.config.split_seed),
            variant="native",
            image_path=image_path,
            label_path=label_path.resolve(),
            width=width,
            height=height,
            objects=objects,
            target=target,
        )

    def _append(self, record: ImageRecord) -> None:
        records = self.detector if record.task == "detector" else self.classifier
        records.append(record)
        self.stats[record.set_name]["native_records"] += 1

    def _append_cosmos(self, chunk: Path, key: str, native: ImageRecord) -> None:
        if native.split != "train":
            return
        image_path = (chunk / "rgb_cosmos" / f"{key}.jpg").resolve()
        width, height = _image_size(image_path)
        if (width, height) != (native.width, native.height):
            raise ManifestError(f"Cosmos size mismatch for {image_path}")
        records = self.detector if native.task == "detector" else self.classifier
        records.append(
            ImageRecord(
                **{
                    **native.__dict__,
                    "variant": "cosmos",
                    "image_path": image_path,
                }
            )
        )
        self.stats[native.set_name]["cosmos_used"] += 1

    def _check_cosmos_coverage(self) -> None:
        if not self.config.require_cosmos:
            return
        for set_name, stats in self.stats.items():
            fraction = stats["cosmos_accepted"] / stats["scenes"]
            if not self.config.min_cosmos_fraction <= fraction <= self.config.max_cosmos_fraction:
                raise ManifestError(
                    f"Cosmos coverage for {set_name} is {fraction:.4f}; expected "
                    f"{self.config.min_cosmos_fraction:.2f}-{self.config.max_cosmos_fraction:.2f}"
                )

    def _write(self) -> ManifestSummary:
        output = self.config.output_dir.resolve()
        detector_path, classifier_path = output / "detector.jsonl", output / "classifier.jsonl"
        _atomic_write(detector_path, _jsonl(self.detector))
        _atomic_write(classifier_path, _jsonl(self.classifier))
        detector_hash, classifier_hash = _sha256(detector_path), _sha256(classifier_path)
        metadata_path = output / "metadata.json"
        metadata = self._metadata(detector_path, classifier_path, detector_hash, classifier_hash)
        _atomic_write(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return ManifestSummary(
            detector_path=detector_path,
            classifier_path=classifier_path,
            metadata_path=metadata_path,
            detector_sha256=detector_hash,
            classifier_sha256=classifier_hash,
            detector_records=len(self.detector),
            classifier_records=len(self.classifier),
            skipped_classifier_targets=dict(sorted(self.skipped.items())),
        )

    def _metadata(
        self,
        detector_path: Path,
        classifier_path: Path,
        detector_hash: str,
        classifier_hash: str,
    ) -> dict[str, Any]:
        stats = {
            name: {**values, "code_commits": sorted(values["code_commits"])}
            for name, values in sorted(self.stats.items())
        }
        return {
            "schema_version": 1,
            "git_commit": _git_commit(),
            "split_seed": self.config.split_seed,
            "taxonomy_path": str(self.config.taxonomy_path.resolve()),
            "taxonomy_sha256": _sha256(self.config.taxonomy_path),
            "require_cosmos": self.config.require_cosmos,
            "include_cosmos": self.config.include_cosmos,
            "max_scenes_per_set": self.config.max_scenes_per_set,
            "labels": {
                "classifier": list(self.ontology.classifier_labels),
                "detector": list(self.ontology.detector_labels),
                "lookalike_groups": [list(group) for group in self.ontology.lookalike_groups],
            },
            "sets": stats,
            "skipped_classifier_targets": dict(sorted(self.skipped.items())),
            "detector": {
                "path": str(detector_path),
                "records": len(self.detector),
                "sha256": detector_hash,
                "source_labels_sha256": self.label_hashes["detector"].hexdigest(),
            },
            "classifier": {
                "path": str(classifier_path),
                "records": len(self.classifier),
                "sha256": classifier_hash,
                "source_labels_sha256": self.label_hashes["classifier"].hexdigest(),
            },
        }


def build_manifests(config: ManifestConfig) -> ManifestSummary:
    """Validate approved synthetic sets and atomically freeze both T13 manifests."""
    _validate_config(config)
    ontology = load_ontology(config.taxonomy_path)
    return _ManifestBuilder(config, ontology).build()
