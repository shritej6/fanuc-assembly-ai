"""Convert frozen detector records to COCO detection JSON."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

SPLITS = ("train", "validation", "test")


class ConversionError(ValueError):
    """Raised when a frozen record cannot be represented safely."""


def _records(path: Path) -> list[dict[str, Any]]:
    output = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConversionError(f"invalid JSONL at {path}:{line_number}") from exc
        if record.get("task") != "detector":
            raise ConversionError(f"non-detector record at {path}:{line_number}")
        output.append(record)
    return output


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _xywh(raw: Any, context: str) -> tuple[int, int, int, int]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise ConversionError(f"invalid bbox in {context}: {raw!r}")
    x0, y0, x1, y1 = raw
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        raise ConversionError(f"bbox must have positive area in {context}: {raw!r}")
    return x0, y0, width, height


def _annotation(
    raw: dict[str, Any],
    labels: tuple[str, ...],
    image_id: int,
    annotation_id: int,
    context: str,
) -> dict[str, Any]:
    class_id = raw.get("class_id")
    object_id = raw.get("object_id")
    if not isinstance(class_id, int) or not 0 <= class_id < len(labels) or labels[class_id] != object_id:
        raise ConversionError(f"class mapping mismatch in {context}: {class_id!r}/{object_id!r}")
    x, y, width, height = _xywh(raw.get("bbox_xyxy"), context)
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": class_id,
        "bbox": [x, y, width, height],
        "area": width * height,
        "iscrowd": 0,
    }


def _empty_payload(labels: tuple[str, ...]) -> dict[str, Any]:
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": index, "name": label} for index, label in enumerate(labels)],
    }


def export_coco(
    manifest_path: Path,
    output_dir: Path,
    labels: tuple[str, ...],
) -> dict[str, Path]:
    """Write one COCO file per frozen split without copying source images."""
    payloads = {split: _empty_payload(labels) for split in SPLITS}
    image_ids = {split: 0 for split in SPLITS}
    annotation_ids = {split: 0 for split in SPLITS}
    for record in _records(manifest_path):
        split = record.get("split")
        if split not in payloads:
            raise ConversionError(f"unknown split {split!r} in {manifest_path}")
        image_ids[split] += 1
        image_id = image_ids[split]
        payloads[split]["images"].append(
            {
                "id": image_id,
                "file_name": record["image_path"],
                "width": record["width"],
                "height": record["height"],
                "scene_id": record["scene_id"],
                "set": record["set"],
                "camera": record["camera"],
                "variant": record["variant"],
            }
        )
        for raw in record.get("objects", []):
            annotation_ids[split] += 1
            context = f"{record['scene_id']}/{record['camera']}/{raw.get('uid')}"
            payloads[split]["annotations"].append(
                _annotation(raw, labels, image_id, annotation_ids[split], context)
            )
    output_dir = output_dir.resolve()
    outputs = {split: output_dir / f"{split}.json" for split in SPLITS}
    for split, path in outputs.items():
        _atomic_json(path, payloads[split])
    return outputs
