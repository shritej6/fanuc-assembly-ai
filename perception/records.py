"""Typed records written to frozen T13 JSONL manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Annotation:
    uid: str
    object_id: str
    class_id: int
    bbox_xyxy: tuple[int, int, int, int]
    kind: str
    state: str
    variant: str
    build_state: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "object_id": self.object_id,
            "class_id": self.class_id,
            "bbox_xyxy": list(self.bbox_xyxy),
            "kind": self.kind,
            "state": self.state,
            "variant": self.variant,
            "build_state": self.build_state,
        }


@dataclass(frozen=True)
class ImageRecord:
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
    objects: tuple[Annotation, ...] = ()
    target: Annotation | None = None

    def sort_key(self) -> tuple[str, int, str, int]:
        variant_order = 0 if self.variant == "native" else 1
        return self.set_name, self.index, self.camera, variant_order

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": 1,
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
        }
        if self.task == "detector":
            record["objects"] = [item.as_dict() for item in self.objects]
        else:
            record["target"] = self.target.as_dict() if self.target else None
        return record
