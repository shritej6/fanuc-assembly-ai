"""scan(): the table and side cameras -> a complete InventorySnapshot plus a typed ScanReport.

Per table frame: RT-DETR boxes -> class-agnostic suppression (dedup.py) -> UF0 centre from table depth (T28 locate)
-> cell zone -> T28 tracker (stable track ids, label vote) -> ledger (stable published ids) -> size check against
the CAD with the table depth (catches a piece that fits none of the 30 objects; it cannot separate 1-2 mm look-alikes
at about 3 mm depth pixels, so those are flagged for the wrist close-up).

What the controller's InventorySnapshot cannot carry is in ScanReport: foreign/unknown pieces (the inventory accepts
only the 30 catalog ids), objects outside the work area, boxes with no depth, side-camera foreign boxes, and the
instances whose identity still needs a wrist close-up. Foreign pieces become obstacles in collision_scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from assembly.controller_ports import InventorySnapshot, ObservedInventoryItem, observation_time
from assembly.inventory import InventoryZone, ObjectCondition
from perception.cell.inspect import same_shape
from perception.cell.localize import Zone
from perception.cell.tracker import Detection, Track, Tracker
from perception.port.backends import CellGeometry, Detector
from perception.port.cad import CadCatalog
from perception.port.dedup import ObjectBox, suppress
from perception.port.errors import PerceptionPortError
from perception.port.geometry import image
from perception.port.ledger import Ledger
from perception.port.size_gate import TABLE_DEPTH_PRELIMINARY, SizeOutcome, SizeVerdict, gate, measure_unposed
from perception.taxonomy import FALLEN_SUFFIX, FOREIGN_LABELS
from robot.adapter import Observation

# A track label needs a majority of its remembered votes (perception/cell/tracker.py VOTE_FRAMES) to be VERIFIED.
MIN_VOTE_SHARE = 0.5
_ZONES = {Zone.TABLE: InventoryZone.TABLE, Zone.SPARE: InventoryZone.SPARE}


@dataclass(frozen=True, slots=True)
class ScannedObject:
    instance_id: str
    object_id: str | None          # None for a foreign / unknown piece
    label: str
    x_mm: float
    y_mm: float
    top_z_mm: float
    bbox_px: tuple[float, float, float, float]
    footprint_mm: tuple[float, float] | None
    zone: Zone
    condition: ObjectCondition
    size: SizeVerdict | None
    contested_by: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScanReport:
    observed_at_s: float
    objects: tuple[ScannedObject, ...]
    foreign: tuple[ScannedObject, ...]
    outside: tuple[ScannedObject, ...]
    unlocated: tuple[ObjectBox, ...]
    side_foreign: tuple[ObjectBox, ...]
    needs_closeup: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Located:
    box: ObjectBox
    detection: Detection
    footprint: tuple[float, float] | None
    size_points: np.ndarray


class Scanner:
    def __init__(self, detector: Detector, geometry: CellGeometry, cad: CadCatalog) -> None:
        self._detector = detector
        self._geometry = geometry
        self._cad = cad
        self.tracker = Tracker()
        self.ledger = Ledger()
        self.assembly_bases: dict[str, tuple[float, float, float]] = {}   # base instance -> (x, y, footprint radius)
        self.last_report: ScanReport | None = None

    def scan(self, observation: Observation) -> tuple[InventorySnapshot, ScanReport]:
        observed_at = observation_time(observation)
        boxes = suppress(self._detector(image(observation, "table")))
        located, unlocated = self._locate_all(observation, boxes)
        free, assembly_hits = self._split_assemblies(located)
        seen = self.tracker.update([loc.detection for loc in free])
        objects = tuple(self._scanned(track, self._match(track, free)) for track in seen)
        report = self._report(observed_at, objects, unlocated, observation)
        items = tuple(self._item(o) for o in objects if o.object_id is not None and o.zone in _ZONES)
        items += self._assembly_items(assembly_hits)
        self.last_report = report
        return InventorySnapshot(observed_at, True, items), report

    def _locate_all(self, observation: Observation, boxes: list[ObjectBox]) -> tuple[list[_Located], list[ObjectBox]]:
        located, unlocated = [], []
        for box in boxes:
            try:
                where = self._geometry.table_locate(observation, box.bbox_px)
                points = self._geometry.table_points(observation, box.bbox_px)
            except PerceptionPortError:
                unlocated.append(box)
                continue
            size = measure_unposed(points, self._geometry.table_surface_z_mm)
            detection = Detection(box.label, box.score, where.x_mm, where.y_mm, where.top_z_mm, box.bbox_px)
            located.append(_Located(box, detection, size.footprint_mm if size else None, points))
        return located, unlocated

    def _split_assemblies(self, located: list[_Located]) -> tuple[list[_Located], set[str]]:
        """Detections on a base that already carries installed parts belong to that assembly, not to the table."""
        free, hits = [], set()
        for loc in located:
            base = next((iid for iid, (x, y, r) in self.assembly_bases.items()
                         if np.hypot(loc.detection.x_mm - x, loc.detection.y_mm - y) <= r), None)
            if base is None:
                free.append(loc)
            else:
                hits.add(base)
        return free, hits

    @staticmethod
    def _match(track: Track, located: list[_Located]) -> _Located:
        for loc in located:
            if loc.detection.bbox_px == track.bbox_px and loc.detection.x_mm == track.x_mm:
                return loc
        raise PerceptionPortError(f"track {track.instance_id} has no detection in this frame")

    def _scanned(self, track: Track, loc: _Located) -> ScannedObject:
        zone, _slot = self._geometry.zone(track.x_mm, track.y_mm)
        label = track.label
        if label in FOREIGN_LABELS:
            return ScannedObject(track.instance_id, None, label, track.x_mm, track.y_mm, track.top_z_mm,
                                 track.bbox_px, loc.footprint, zone, ObjectCondition.UNCERTAIN, None, ())
        object_id = label.removesuffix(FALLEN_SUFFIX)
        size = gate(object_id, measure_unposed(loc.size_points, self._geometry.table_surface_z_mm), self._cad,
                    TABLE_DEPTH_PRELIMINARY) if not label.endswith(FALLEN_SUFFIX) else None
        published = self.ledger.publish(track.instance_id, object_id, track.x_mm, track.y_mm)
        contested = loc.box.contested_by(same_shape)
        condition = self._condition(track, label, contested)
        return ScannedObject(published.instance_id, object_id, label, track.x_mm, track.y_mm, track.top_z_mm,
                             track.bbox_px, loc.footprint, zone, condition, size, contested)

    @staticmethod
    def _condition(track: Track, label: str, contested: tuple[str, ...]) -> ObjectCondition:
        """Table size is reported, not used to downgrade: its xy bound is assumed; the wrist close-up gates the pick."""
        if label.endswith(FALLEN_SUFFIX):
            return ObjectCondition.FALLEN
        agree, total = track.votes
        if agree / total < MIN_VOTE_SHARE or contested:
            return ObjectCondition.UNCERTAIN
        return ObjectCondition.VERIFIED

    @staticmethod
    def _item(scanned: ScannedObject) -> ObservedInventoryItem:
        return ObservedInventoryItem(scanned.instance_id, scanned.object_id, _ZONES[scanned.zone],  # type: ignore[arg-type]
                                     scanned.condition, (scanned.x_mm, scanned.y_mm))

    def _assembly_items(self, hits: set[str]) -> tuple[ObservedInventoryItem, ...]:
        """A base with installed parts is observed when something stands on its footprint this frame."""
        return tuple(ObservedInventoryItem(iid, self.ledger.object_of(iid), InventoryZone.TABLE,
                                           ObjectCondition.VERIFIED, self.assembly_bases[iid][:2]) for iid in sorted(hits))

    def _report(self, observed_at: float, objects: tuple[ScannedObject, ...], unlocated: list[ObjectBox],
                observation: Observation) -> ScanReport:
        side = suppress(self._detector(image(observation, "side"))) if "side" in observation.get("images", {}) else []
        closeup = tuple(o.instance_id for o in objects if o.object_id is not None and o.zone in _ZONES
                        and (o.contested_by or self._cad.get(o.object_id).lookalikes
                             or (o.size is not None and o.size.outcome is not SizeOutcome.CONFIRMS)))
        return ScanReport(observed_at, objects,
                          tuple(o for o in objects if o.object_id is None),
                          tuple(o for o in objects if o.object_id is not None and o.zone not in _ZONES),
                          tuple(unlocated), tuple(b for b in side if b.label in FOREIGN_LABELS), closeup)
