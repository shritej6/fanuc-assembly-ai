"""Dataset adapter for frozen wrist-target crop records."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from perception.image_integrity import ImageIntegrityError, read_verified_rgb, require_sha256


class DatasetError(ValueError):
    """Raised when frozen classifier crops are malformed."""


class CropDataset:
    """PyTorch-compatible crop dataset with retained scenario rows."""

    def __init__(
        self,
        crops_path: Path,
        split: str,
        transform: Callable[[Image.Image], Any],
        labels: tuple[str, ...],
        max_samples: int | None = None,
    ) -> None:
        rows = []
        for line_number, line in enumerate(crops_path.read_text().splitlines(), start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"invalid crop JSONL at {crops_path}:{line_number}") from exc
            if row.get("split") == split:
                try:
                    require_sha256(row.get("crop_sha256"), "crop_sha256")
                except ImageIntegrityError as exc:
                    raise DatasetError(f"invalid crop row at {crops_path}:{line_number}: {exc}") from exc
                rows.append(row)
        if max_samples is not None:
            if max_samples < 1:
                raise DatasetError("max_samples must be positive")
            rows = rows[:max_samples]
        self.rows = rows
        self.transform = transform
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = Path(row["crop_path"])
        try:
            image = read_verified_rgb(path, row["crop_sha256"])
        except ImageIntegrityError as exc:
            raise DatasetError(str(exc)) from exc
        class_id = row.get("class_id")
        object_id = row.get("object_id")
        valid_id = type(class_id) is int and 0 <= class_id < len(self.labels)
        if not valid_id or self.labels[class_id] != object_id:
            raise DatasetError(f"class mapping mismatch for {path}: {class_id!r}/{object_id!r}")
        return {"pixel_values": self.transform(image), "labels": class_id}
