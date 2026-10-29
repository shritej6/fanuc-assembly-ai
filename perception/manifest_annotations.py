"""Box annotations for frozen manifests: bbox checks, detector label names and test-bed rest states."""

from __future__ import annotations

from typing import Any

from perception.manifest_types import ManifestError
from perception.records import Annotation
from perception.taxonomy import fallen_label


def _bbox(raw: Any, width: int, height: int, context: str) -> tuple[int, int, int, int]:
    if not isinstance(raw, list) or len(raw) != 4 or any(not isinstance(value, int) for value in raw):
        raise ManifestError(f"invalid bbox in {context}: {raw!r}")
    x0, y0, x1, y1 = raw
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ManifestError(f"out-of-bounds bbox in {context}: {raw!r} for {width}x{height}")
    return x0, y0, x1, y1


def annotation(
    raw: dict[str, Any],
    label2id: dict[str, int],
    width: int,
    height: int,
    context: str,
    rest: str | None = None,
    fallen_classes: bool = False,
) -> Annotation:
    object_id = raw.get("object_id")
    if object_id not in label2id:
        raise ManifestError(f"unknown object_id {object_id!r} in {context}")
    label = object_id
    if fallen_classes and rest == "fallen":
        try:
            label = fallen_label(object_id)
        except ValueError as exc:
            raise ManifestError(f"{exc} in {context}") from exc
    return Annotation(
        uid=str(raw["uid"]),
        object_id=object_id,
        class_id=label2id[label],
        bbox_xyxy=_bbox(raw.get("bbox_xyxy"), width, height, context),
        kind=str(raw.get("kind", "")),
        state=str(raw.get("state", "")),
        variant=str(raw.get("variant", "")),
        build_state=raw.get("build_state"),
        rest=rest,
        label=label,
    )


def rest_by_uid(spec: dict[str, Any], context: str) -> dict[str, str | None]:
    """Test-bed labels keep `rest` (installed | fallen) on the scene objects, not on the per-camera boxes."""
    rest = {}
    for item in spec.get("objects", []):
        value = item.get("rest")
        if value not in (None, "installed", "fallen"):
            raise ManifestError(f"unknown rest state {value!r} in {context}")
        rest[str(item.get("uid"))] = value
    return rest
