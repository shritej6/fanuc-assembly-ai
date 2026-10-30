"""Validate SDG chunks and freeze leak-free T13 training manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from perception.image_integrity import ImageIntegrityError, image_identity
from perception.manifest_annotations import annotation as _annotation
from perception.manifest_annotations import rest_by_uid as _rest_by_uid
from perception.manifest_types import ManifestConfig, ManifestError, ManifestSummary, split_for_scene
from perception.provenance import current_git_commit
from perception.records import Annotation, ImageRecord
from perception.taxonomy import Ontology, load_ontology

CHUNK_PATTERN = re.compile(r"chunk_[0-9]{4}")
FORBIDDEN_PARTS = {"rejected", "superseded", "real", "real-cell", "real_cell"}
ALLOWED_CAMERAS = {"table", "side", "wrist"}


def _validate_config(config: ManifestConfig) -> None:
    if config.require_cosmos and not config.include_cosmos:
        raise ManifestError("a Cosmos-required manifest must include Cosmos records")
    limit = config.max_scenes_per_set
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ManifestError("max_scenes_per_set must be a positive integer")


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


def _contained_path(path: Path, root: Path, description: str) -> Path:
    resolved_root = root.resolve()
    try:
        relative = path.absolute().relative_to(resolved_root)
    except ValueError as exc:
        raise ManifestError(f"{description} is outside approved synthetic data: {path}") from exc
    cursor = resolved_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ManifestError(f"{description} symlink is outside approved synthetic structure: {cursor}")
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ManifestError(f"{description} is outside approved synthetic data: {resolved}")
    return resolved


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
    chunks = []
    for path in set_dir.iterdir():
        if CHUNK_PATTERN.fullmatch(path.name):
            resolved = _contained_path(path, set_dir, "chunk")
            if resolved.is_dir():
                chunks.append(resolved)
    chunks.sort()
    if not chunks:
        raise ManifestError(f"no completed chunks found in {set_dir}")
    return chunks


def _validated_labels(chunk: Path) -> tuple[list[Path], str]:
    done_path = _contained_path(chunk / "_DONE.json", chunk, "completion metadata")
    qa_path = _contained_path(chunk / "_QA.json", chunk, "QA metadata")
    if not done_path.is_file():
        raise ManifestError(f"missing _DONE.json in {chunk}")
    if not qa_path.is_file():
        raise ManifestError(f"missing _QA.json in {chunk}")
    done, qa = _read_json(done_path), _read_json(qa_path)
    if qa.get("ok") is not True:
        raise ManifestError(f"QA failed for {chunk}")
    labels = sorted(_contained_path(path, chunk, "label") for path in (chunk / "labels").glob("*.json"))
    counts = (done.get("scenes"), qa.get("labelled_scenes"), qa.get("expected_scenes"), len(labels))
    if not all(isinstance(value, int) for value in counts) or len(set(counts)) != 1:
        raise ManifestError(f"scene count mismatch in {chunk}: {counts}")
    return labels, str(done.get("code_commit", "unknown"))


def _cosmos_entries(chunk: Path) -> dict[str, dict[str, Any]]:
    manifest_path = _contained_path(chunk / "_COSMOS.json", chunk, "Cosmos evidence")
    entries = _read_json(manifest_path) if manifest_path.exists() else {}
    expected = {key for key, value in entries.items() if value.get("accepted") is True}
    cosmos_root = chunk / "rgb_cosmos"
    existing = set()
    for path in cosmos_root.glob("*/*.jpg"):
        resolved = _contained_path(path, chunk, "Cosmos image")
        if resolved.is_file():
            existing.add(f"{path.parent.name}/{path.stem}")
    missing = expected - existing
    if missing:
        raise ManifestError(f"accepted Cosmos file is missing in {chunk}: {sorted(missing)}")
    orphan = existing - expected
    if orphan:
        raise ManifestError(f"orphan Cosmos file without acceptance evidence in {chunk}: {sorted(orphan)}")
    return entries


def _image_identity(path: Path) -> tuple[int, int, str]:
    try:
        return image_identity(path)
    except ImageIntegrityError as exc:
        raise ManifestError(str(exc)) from exc


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
        synthetic_root = _contained_path(root / "synthetic", root, "synthetic root")
        if not synthetic_root.is_dir():
            raise ManifestError(f"synthetic data root does not exist: {synthetic_root}")

    def _process_set(self, set_name: str, task: str) -> None:
        stats = {
            "scenes": 0,
            "native_records": 0,
            "cosmos_checked": 0,
            "cosmos_accepted": 0,
            "cosmos_used": 0,
            "code_commits": set(),
            "chunks": [],
        }
        self.stats[set_name] = stats
        root = self.config.data_root.resolve()
        synthetic_root = _contained_path(root / "synthetic", root, "synthetic root")
        set_dir = _contained_path(synthetic_root / set_name, synthetic_root, f"set {set_name}")
        remaining = self.config.max_scenes_per_set
        for chunk in _chunk_dirs(set_dir):
            if remaining == 0:
                break
            labels, code_commit = _validated_labels(chunk)
            if remaining is not None:
                labels = labels[:remaining]
                remaining -= len(labels)
            stats["code_commits"].add(code_commit)
            stats["chunks"].append(chunk.name)
            stats["scenes"] += len(labels)
            entries = _cosmos_entries(chunk) if self.config.include_cosmos else {}
            stats["cosmos_checked"] += len(entries)
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
        eligible_keys = accepted_keys & native.keys()
        self.stats[set_name]["cosmos_accepted"] += len(eligible_keys)
        for key, value in sorted(cosmos.items()):
            if value.get("accepted") is True and key in eligible_keys:
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
        cameras = label.get("cameras")
        if not isinstance(cameras, dict) or not set(cameras) <= ALLOWED_CAMERAS:
            raise ManifestError(f"label contains invalid camera names in {label_path}")
        if task == "detector":
            rest = _rest_by_uid(spec, f"{set_name}:{index:06d}")
            return [
                self._detector_record(chunk, label_path, set_name, index, camera, raw, rest)
                for camera, raw in sorted(cameras.items())
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
        rest: dict[str, str | None],
    ) -> ImageRecord:
        image_path = _contained_path(
            chunk / "rgb" / camera / f"{index:06d}.jpg",
            chunk,
            "native image",
        )
        width, height, image_sha256 = _image_identity(image_path)
        context = f"{set_name}:{index:06d}/{camera}"
        objects = tuple(
            _annotation(
                raw,
                self.ontology.detector_label2id,
                width,
                height,
                context,
                rest.get(str(raw.get("uid"))),
                self.config.fallen_classes,
            )
            for raw in raw_objects
        )
        return self._record(
            "detector",
            set_name,
            index,
            camera,
            image_path,
            label_path,
            width,
            height,
            image_sha256,
            objects,
        )

    def _classifier_record(
        self,
        chunk: Path,
        label_path: Path,
        set_name: str,
        index: int,
        label: dict[str, Any],
    ) -> ImageRecord | None:
        camera = "wrist"
        image_path = _contained_path(
            chunk / "rgb" / camera / f"{index:06d}.jpg",
            chunk,
            "native image",
        )
        width, height, image_sha256 = _image_identity(image_path)
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
        return self._record(
            "classifier",
            set_name,
            index,
            camera,
            image_path,
            label_path,
            width,
            height,
            image_sha256,
            (),
            target,
        )

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
        image_sha256: str,
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
            image_sha256=image_sha256,
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
        image_path = _contained_path(
            chunk / "rgb_cosmos" / f"{key}.jpg",
            chunk,
            "Cosmos image",
        )
        width, height, image_sha256 = _image_identity(image_path)
        if (width, height) != (native.width, native.height):
            raise ManifestError(f"Cosmos size mismatch for {image_path}")
        records = self.detector if native.task == "detector" else self.classifier
        records.append(
            ImageRecord(
                **{
                    **native.__dict__,
                    "variant": "cosmos",
                    "image_path": image_path,
                    "image_sha256": image_sha256,
                }
            )
        )
        self.stats[native.set_name]["cosmos_used"] += 1

    def _check_cosmos_coverage(self) -> None:
        if not self.config.require_cosmos:
            return
        for set_name, stats in self.stats.items():
            native_records = stats["native_records"]
            if native_records < 1:
                raise ManifestError(f"no eligible native records for Cosmos coverage in {set_name}")
            fraction = stats["cosmos_accepted"] / native_records
            low, high = self.config.cosmos_range(set_name)
            if not low <= fraction <= high:
                raise ManifestError(f"Cosmos coverage for {set_name} is {fraction:.4f}; expected {low:.2f}-{high:.2f}")

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
            "schema_version": 2, "content_hash_algorithm": "sha256",
            "git_commit": current_git_commit(),
            "split_seed": self.config.split_seed,
            "taxonomy_path": str(self.config.taxonomy_path.resolve()),
            "taxonomy_sha256": _sha256(self.config.taxonomy_path),
            "require_cosmos": self.config.require_cosmos,
            "include_cosmos": self.config.include_cosmos,
            "max_scenes_per_set": self.config.max_scenes_per_set,
            "profile": self.config.profile,
            "fallen_classes": self.config.fallen_classes,
            "cosmos_ranges": {name: [low, high] for name, low, high in self.config.cosmos_ranges},
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
    ontology = load_ontology(config.taxonomy_path, config.fallen_classes)
    return _ManifestBuilder(config, ontology).build()
