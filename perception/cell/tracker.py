"""Keep stable instance ids for parts seen by the table camera across frames.
Authored by Shritej Gunda.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

# Twice the largest coarse localisation error measured on ray-cast parts (2.92 mm, perception/cell/localize.py) plus
# a margin for real depth noise; parts are never placed closer than this (sdg/configs/cell_layout.yaml gaps).
GATE_MM = 12.0
VOTE_FRAMES = 9
MISS_FRAMES = 5


class TrackerError(ValueError):
    """A tracker call named a track that does not exist or passed malformed data."""


@dataclass(frozen=True, slots=True)
class Detection:
    """One located table-camera detection."""
    label: str
    score: float
    x_mm: float
    y_mm: float
    top_z_mm: float
    bbox_px: tuple[float, float, float, float]


@dataclass(slots=True)
class Track:
    """One tracked part with its recent labels."""
    instance_id: str
    x_mm: float
    y_mm: float
    top_z_mm: float
    bbox_px: tuple[float, float, float, float]
    labels: deque[str] = field(default_factory=lambda: deque(maxlen=VOTE_FRAMES))
    missed: int = 0

    @property
    def label(self) -> str:
        """Return the most common recent label, preferring the latest on a tie."""
        counts = Counter(self.labels)
        best = max(counts.values())
        return next(label for label in reversed(self.labels) if counts[label] == best)

    @property
    def votes(self) -> tuple[int, int]:
        """Return how many remembered detections agree with the label."""
        return Counter(self.labels)[self.label], len(self.labels)


class Tracker:
    """Associate detections with tracks frame after frame."""
    def __init__(self) -> None:
        self._tracks: dict[str, Track] = {}
        self._next = 1

    def tracks(self) -> tuple[Track, ...]:
        """Return all current tracks."""
        return tuple(self._tracks.values())

    def update(self, detections: list[Detection]) -> tuple[Track, ...]:
        """Associate one frame's detections and return the tracks seen in it."""
        pairs = self._pairs(detections)
        matched_tracks, matched_dets = {t for t, _ in pairs}, {d for _, d in pairs}
        for track_id in [t for t in self._tracks if t not in matched_tracks]:
            self._tracks[track_id].missed += 1
            if self._tracks[track_id].missed >= MISS_FRAMES:
                del self._tracks[track_id]
        seen = [self._assign(self._tracks[t], detections[d]) for t, d in pairs]
        seen += [self._create(det) for index, det in enumerate(detections) if index not in matched_dets]
        return tuple(seen)

    def moved(self, instance_id: str, x_mm: float, y_mm: float) -> None:
        """Move a track to where the robot put the part (UF0 mm)."""
        if instance_id not in self._tracks:
            raise TrackerError(f"no track {instance_id!r}")
        if not (np.isfinite(x_mm) and np.isfinite(y_mm)):
            raise TrackerError(f"moved {instance_id!r} to a non-finite position")
        track = self._tracks[instance_id]
        track.x_mm, track.y_mm, track.missed = float(x_mm), float(y_mm), 0

    def _pairs(self, detections: list[Detection]) -> list[tuple[str, int]]:
        candidates = sorted(
            (float(np.hypot(t.x_mm - d.x_mm, t.y_mm - d.y_mm)), t.instance_id, i)
            for t in self._tracks.values() for i, d in enumerate(detections))
        used_t: set[str] = set()
        used_d: set[int] = set()
        pairs = []
        for dist, track_id, index in candidates:
            if dist > GATE_MM:
                break
            if track_id not in used_t and index not in used_d:
                used_t.add(track_id)
                used_d.add(index)
                pairs.append((track_id, index))
        return pairs

    def _assign(self, track: Track, det: Detection) -> Track:
        track.x_mm, track.y_mm, track.top_z_mm, track.bbox_px = det.x_mm, det.y_mm, det.top_z_mm, det.bbox_px
        track.labels.append(det.label)
        track.missed = 0
        return track

    def _create(self, det: Detection) -> Track:
        track = Track(f"p{self._next}", det.x_mm, det.y_mm, det.top_z_mm, det.bbox_px)
        self._next += 1
        self._tracks[track.instance_id] = track
        return self._assign(track, det)
