"""Class-agnostic duplicate suppression for RT-DETR boxes: one physical object, one box.
Authored by Shritej Gunda."""

from __future__ import annotations

from dataclasses import dataclass

from perception.port.errors import ModelOutputError

DEFAULT_IOU = 0.5
DEFAULT_CONTAINMENT = 0.85
# A runner-up within this fraction of the winner's score is a contested identity (wrist close-up decides).
CONTESTED_RATIO = 0.8


@dataclass(frozen=True, slots=True)
class RawBox:
    """One raw detector box with its label and score."""
    label: str
    score: float
    bbox_px: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ObjectBox:
    """One physical object: the top-1 box plus the labels suppressed onto it."""

    label: str
    score: float
    bbox_px: tuple[float, float, float, float]
    alternatives: tuple[tuple[str, float], ...]

    def contested_by(self, same_shape) -> tuple[str, ...]:
        """Runner-up labels of another shape scoring within CONTESTED_RATIO of the winner."""
        return tuple(label for label, score in self.alternatives
                     if score >= CONTESTED_RATIO * self.score and not same_shape(label, self.label))


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float]:
    """(IoU, intersection over the smaller area)."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, width) * max(0.0, height)
    area_a, area_b = _area(a), _area(b)
    union = area_a + area_b - inter
    return (inter / union if union > 0 else 0.0), (inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0)


def _validate(box: RawBox) -> None:
    x0, y0, x1, y1 = box.bbox_px
    if not (0.0 <= box.score <= 1.0) or not (x1 > x0 and y1 > y0):
        raise ModelOutputError(f"detector box {box} has a score outside [0, 1] or no area")


def suppress(boxes: list[RawBox], iou: float = DEFAULT_IOU, containment: float = DEFAULT_CONTAINMENT) -> list[ObjectBox]:
    """Group boxes by overlap regardless of class; return one ObjectBox per group, highest score first."""
    for box in boxes:
        _validate(box)
    kept: list[tuple[RawBox, list[tuple[str, float]]]] = []
    for box in sorted(boxes, key=lambda b: b.score, reverse=True):
        for winner, alternatives in kept:
            box_iou, box_ios = _overlap(winner.bbox_px, box.bbox_px)
            if box_iou >= iou or box_ios >= containment:
                alternatives.append((box.label, box.score))
                break
        else:
            kept.append((box, []))
    return [ObjectBox(w.label, w.score, w.bbox_px, tuple(alt)) for w, alt in kept]
