"""Stable instance ids published to the controller across scans and robot moves.
Authored by Shritej Gunda."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.port.errors import UnknownInstanceError

ADOPT_GATE_MM = 12.0   # perception/cell/tracker.py GATE_MM: the same association radius the tracker uses


@dataclass(frozen=True, slots=True)
class Published:
    """One instance id published to the controller."""
    instance_id: str
    object_id: str


@dataclass(frozen=True, slots=True)
class _Expected:
    object_id: str
    x_mm: float
    y_mm: float


class Ledger:
    """Keeps published instance ids stable across scans."""
    def __init__(self) -> None:
        self._by_track: dict[str, Published] = {}
        self._revisions: dict[str, int] = {}
        self._expected: dict[str, _Expected] = {}

    def publish(self, track_id: str, object_id: str, x_mm: float, y_mm: float) -> Published:
        """Return the published entry for a track, keeping its id stable."""
        current = self._by_track.get(track_id)
        if current is not None and current.object_id == object_id:
            return current
        adopted = self._adopt(object_id, x_mm, y_mm) if current is None else None
        if adopted is not None:
            published = Published(adopted, object_id)
        elif current is None:
            published = Published(track_id, object_id)
        else:
            self._revisions[track_id] = self._revisions.get(track_id, 0) + 1
            published = Published(f"{track_id}r{self._revisions[track_id]}", object_id)
        self._by_track[track_id] = published
        return published

    def track_of(self, instance_id: str) -> str | None:
        """Return the track behind a published instance id, if any."""
        return next((track for track, pub in self._by_track.items() if pub.instance_id == instance_id), None)

    def object_of(self, instance_id: str) -> str:
        """Return the object id of a published instance."""
        for pub in self._by_track.values():
            if pub.instance_id == instance_id:
                return pub.object_id
        expected = self._expected.get(instance_id)
        if expected is None:
            raise UnknownInstanceError(f"instance {instance_id!r} was never published by a scan")
        return expected.object_id

    def expect(self, instance_id: str, object_id: str, x_mm: float, y_mm: float) -> None:
        """The robot put this instance at x, y; a new track there inherits the id. Its old track is released."""
        track = self.track_of(instance_id)
        if track is not None:
            del self._by_track[track]
        self._expected[instance_id] = _Expected(object_id, float(x_mm), float(y_mm))

    def _adopt(self, object_id: str, x_mm: float, y_mm: float) -> str | None:
        near = sorted((float(np.hypot(e.x_mm - x_mm, e.y_mm - y_mm)), iid) for iid, e in self._expected.items()
                      if e.object_id == object_id)
        if not near or near[0][0] > ADOPT_GATE_MM:
            return None
        instance_id = near[0][1]
        del self._expected[instance_id]
        return instance_id
