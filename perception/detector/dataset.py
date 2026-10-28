"""COCO-backed dataset adapter for RT-DETR's image processor."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


class DatasetError(ValueError):
    """Raised when a COCO split cannot be consumed safely."""


class CocoDataset:
    """Minimal PyTorch-compatible dataset without importing torch at module load."""

    def __init__(self, coco_path: Path, processor: Any, max_samples: int | None = None) -> None:
        payload = json.loads(coco_path.read_text())
        if not isinstance(payload, dict):
            raise DatasetError(f"COCO payload must be an object: {coco_path}")
        self.processor = processor
        self.images = list(payload.get("images", ()))
        if max_samples is not None:
            if max_samples < 1:
                raise DatasetError("max_samples must be positive")
            self.images = self.images[:max_samples]
        annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload.get("annotations", ()):
            annotations[int(annotation["image_id"])].append(annotation)
        self.annotations = dict(annotations)
        self.categories = tuple(item["name"] for item in payload.get("categories", ()))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.images[index]
        path = Path(record["file_name"])
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        except OSError as exc:
            raise DatasetError(f"cannot read detector image {path}: {exc}") from exc
        expected_size = (record["width"], record["height"])
        if image.size != expected_size:
            raise DatasetError(f"detector image size mismatch for {path}: {image.size} != {expected_size}")
        target = {
            "image_id": int(record["id"]),
            "annotations": self.annotations.get(int(record["id"]), []),
        }
        encoded = self.processor(images=image, annotations=target, return_tensors="pt")
        item = {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"][0],
        }
        if "pixel_mask" in encoded:
            item["pixel_mask"] = encoded["pixel_mask"].squeeze(0)
        return item
