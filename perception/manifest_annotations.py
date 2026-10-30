"""Box annotations for frozen manifests: bbox checks, detector label names and test-bed rest states."""

from __future__ import annotations

from typing import Any

from perception.manifest_types import ManifestError
from perception.records import Annotation
from perception.taxonomy import BUILD_STATES, fallen_label

# Scene states of parts that lie on their own (on the table, in the spare store), so their pose is labelled.
LOOSE_STATES = ("on_table", "in_store")


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
    # A loose or spare part must say how it lies; without `rest` it would silently train as the installed identity.
    if fallen_classes and rest is None and not object_id.startswith("F") and raw.get("state") in LOOSE_STATES:
        raise ManifestError(f"part {object_id} {raw.get('state')} has no rest state in {context}")
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


def build_state_annotation(raw: dict[str, Any], width: int, height: int, context: str) -> Annotation:
    """Set P target: the class is how the part sits on its pegs, the object stays the exact part."""
    state = raw.get("build_state")
    if state not in BUILD_STATES:
        raise ManifestError(f"unknown build_state {state!r} in {context}")
    return Annotation(
        uid=str(raw["uid"]),
        object_id=str(raw.get("object_id")),
        class_id=BUILD_STATES.index(state),
        bbox_xyxy=_bbox(raw.get("bbox_xyxy"), width, height, context),
        kind=str(raw.get("kind", "")),
        state=str(raw.get("state", "")),
        variant=str(raw.get("variant", "")),
        build_state=state,
        label=state,
    )
