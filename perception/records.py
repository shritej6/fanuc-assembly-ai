"""Typed records written to the frozen JSONL training manifests.
Authored by Shritej Gunda."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Annotation:
    """One labelled object in an image record."""
    uid: str
    object_id: str
    class_id: int
    bbox_xyxy: tuple[int, int, int, int]
    kind: str
    state: str
    variant: str
    build_state: str | None
    rest: str | None = None
    label: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return the annotation as a JSON-ready dict."""
        return {
            "uid": self.uid,
            "object_id": self.object_id,
            "class_id": self.class_id,
            "bbox_xyxy": list(self.bbox_xyxy),
            "kind": self.kind,
            "state": self.state,
            "variant": self.variant,
            "build_state": self.build_state,
            "rest": self.rest,
            "label": self.label or self.object_id,
        }


@dataclass(frozen=True)
class ImageRecord:
    """One image entry of a frozen manifest with its annotations."""
    task: str
    scene_id: str
    set_name: str
    index: int
    camera: str
    split: str
    variant: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    image_sha256: str
    objects: tuple[Annotation, ...] = ()
    target: Annotation | None = None

    def sort_key(self) -> tuple[str, int, str, int]:
        """Return the stable sort key for manifest ordering."""
        variant_order = 0 if self.variant == "native" else 1
        return self.set_name, self.index, self.camera, variant_order

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-ready dict."""
        record: dict[str, Any] = {
            "schema_version": 2,
            "task": self.task,
            "scene_id": self.scene_id,
            "set": self.set_name,
            "index": self.index,
            "camera": self.camera,
            "split": self.split,
            "variant": self.variant,
            "image_path": str(self.image_path),
            "label_path": str(self.label_path),
            "width": self.width,
            "height": self.height,
            "image_sha256": self.image_sha256,
        }
        if self.task == "detector":
            record["objects"] = [item.as_dict() for item in self.objects]
        else:
            record["target"] = self.target.as_dict() if self.target else None
        return record
