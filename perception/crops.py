"""Export deterministic wrist-target crops from frozen classifier records."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


class CropError(ValueError):
    """Raised when a classifier record cannot produce a safe crop."""


def _records(path: Path) -> list[dict[str, Any]]:
    output = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CropError(f"invalid JSONL at {path}:{line_number}") from exc
        if record.get("task") != "classifier" or not isinstance(record.get("target"), dict):
            raise CropError(f"invalid classifier record at {path}:{line_number}")
        output.append(record)
    return output


def _crop_box(
    raw: Any,
    image_size: tuple[int, int],
    context_scale: float,
) -> tuple[int, int, int, int]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise CropError(f"invalid target bbox: {raw!r}")
    x0, y0, x1, y1 = raw
    if x1 <= x0 or y1 <= y0:
        raise CropError(f"target bbox must have positive area: {raw!r}")
    pad_x = (x1 - x0) * (context_scale - 1.0) / 2.0
    pad_y = (y1 - y0) * (context_scale - 1.0) / 2.0
    width, height = image_size
    return (
        max(0, math.floor(x0 - pad_x)),
        max(0, math.floor(y0 - pad_y)),
        min(width, math.ceil(x1 + pad_x)),
        min(height, math.ceil(y1 + pad_y)),
    )


def _crop_name(record: dict[str, Any]) -> str:
    return (
        f"{record['set']}__{record['index']:06d}__"
        f"{record['camera']}__{record['variant']}.jpg"
    )


def _save_crop(image: Image.Image, box: tuple[int, int, int, int], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".jpg", dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    image.crop(box).save(temporary, format="JPEG", quality=95)
    os.replace(temporary, target)


def _metadata_row(
    record: dict[str, Any],
    crop_path: Path,
    box: tuple[int, int, int, int],
) -> dict[str, Any]:
    target = record["target"]
    return {
        "schema_version": 1,
        "scene_id": record["scene_id"],
        "set": record["set"],
        "index": record["index"],
        "camera": record["camera"],
        "split": record["split"],
        "variant": record["variant"],
        "source_image_path": record["image_path"],
        "label_path": record["label_path"],
        "crop_path": str(crop_path),
        "crop_bbox_xyxy": list(box),
        "object_id": target["object_id"],
        "class_id": target["class_id"],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def export_crops(
    manifest_path: Path,
    output_dir: Path,
    context_scale: float = 1.3,
) -> Path:
    """Crop every visible wrist target and write a deterministic crop manifest."""
    if context_scale < 1.0:
        raise CropError("context_scale must be at least 1.0")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for record in _records(manifest_path):
        source = Path(record["image_path"])
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        if image.size != (record["width"], record["height"]):
            raise CropError(f"source size mismatch for {source}")
        target = record["target"]
        box = _crop_box(target.get("bbox_xyxy"), image.size, context_scale)
        crop_path = (output_dir / record["split"] / target["object_id"] / _crop_name(record)).resolve()
        if crop_path in seen:
            raise CropError(f"duplicate crop path: {crop_path}")
        seen.add(crop_path)
        _save_crop(image, box, crop_path)
        rows.append(_metadata_row(record, crop_path, box))
    metadata_path = output_dir / "crops.jsonl"
    _write_jsonl(metadata_path, rows)
    return metadata_path
