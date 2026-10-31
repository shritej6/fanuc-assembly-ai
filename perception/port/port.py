"""CellPerceptionPort: the T09 `PerceptionPort` (assembly/controller_ports.py) on the real cell's cameras and models.

Every call returns evidence stamped with the observation's own time, or raises a PerceptionPortError (a
ControllerPortError, so the controller holds). Typed side information the Protocol's return types cannot carry is
kept on the port: `last_scan` (ScanReport), `closeup(instance_id)` (CloseupReport: use / reject / second look /
Andon with both pictures) and `placement(instance_id)` (SeatReport with the measured errors and their bounds).
"""

from __future__ import annotations

import math

from assembly.controller_ports import (
    CollisionScene,
    EvidencePair,
    InventorySnapshot,
    PlacementEvidence,
    RecoveryTransferEvidence,
    current_waypoint,
    observation_time,
)
from assembly.inventory import InventoryItem, InventoryZone
from perception.cell.localize import Zone
from perception.port.backends import CellGeometry, Detector, PoseEstimator, WristClassifier
from perception.port.cad import CadCatalog, load_cad
from perception.port.errors import PerceptionPortError, UnknownInstanceError
from perception.port.evidence import CloseupReport, another_angle, detection, inspection, view_request
from perception.port.frames import InstalledPose, installed_pose, posed_bbox_corners
from perception.port.geometry import joints
from perception.port.judge import CloseupThresholds, Judgement, NextStep, judge
from perception.port.placement import PoseUncertainty, SeatReport, seat_report
from perception.port.postplace import post_place_judgement, transfer_judgement
from perception.port.scan import Scanner, ScanReport
from perception.port.scene import (
    ArmCorridor,
    box_from_footprint,
    box_from_pose,
    cad_footprint,
    held_part_visible,
    intrusion,
    tallest_stack_mm,
)
from perception.port.size_gate import SizeUncertainty
from perception.port.wrist import ImageSink, WristInspector, WristView
from robot.adapter import Observation

_TRANSFER_ZONES = {InventoryZone.REJECT: Zone.REJECT, InventoryZone.STAGED: Zone.TABLE, InventoryZone.SPARE: Zone.SPARE}


class CellPerceptionPort:
    def __init__(self, *, detector: Detector, classifier: WristClassifier, pose_estimator: PoseEstimator,
                 geometry: CellGeometry, thresholds: CloseupThresholds, wrist_size: SizeUncertainty,
                 wrist_pose: PoseUncertainty, table_pose: PoseUncertainty, arm_corridor: ArmCorridor,
                 image_sink: ImageSink, cad: CadCatalog | None = None) -> None:
        self._cad = cad or load_cad()
        self._geometry = geometry
        self._pose = pose_estimator
        self._thresholds = thresholds
        self._wrist_pose, self._table_pose = wrist_pose, table_pose
        self._corridor = arm_corridor
        self._scanner = Scanner(detector, geometry, self._cad)
        self._inspector = WristInspector(geometry, pose_estimator, classifier, self._cad, wrist_size, image_sink)
        self._xy: dict[str, tuple[float, float]] = {}
        self._base_poses: dict[str, InstalledPose] = {}
        self.base_fit_errors: dict[str, str] = {}
        self._pending: dict[str, WristView] = {}
        self._closeups: dict[str, CloseupReport] = {}
        self._placements: dict[str, SeatReport] = {}

    @property
    def last_scan(self) -> ScanReport | None:
        return self._scanner.last_report

    def closeup(self, instance_id: str) -> CloseupReport:
        return self._report(self._closeups, instance_id, "close-up")

    def placement(self, instance_id: str) -> SeatReport:
        return self._report(self._placements, instance_id, "placement")

    @staticmethod
    def _report(table: dict, instance_id: str, label: str):
        if instance_id not in table:
            raise UnknownInstanceError(f"no {label} report for {instance_id!r}")
        return table[instance_id]

    # -- scan ---------------------------------------------------------------------------------------------------
    def scan(self, observation: Observation) -> InventorySnapshot:
        snapshot, report = self._scanner.scan(observation)
        for item in snapshot.items:
            if item.coarse_xy_mm is not None:
                self._xy[item.instance_id] = item.coarse_xy_mm
        for scanned in report.objects:
            if scanned.object_id is not None and scanned.object_id.startswith("F"):
                self._fit_base(observation, scanned.instance_id, scanned.object_id)
        return snapshot

    def _fit_base(self, observation: Observation, instance_id: str, object_id: str) -> None:
        """FoundationPose on the table camera for a base; a failed fit is recorded and blocks that base's scene."""
        scanned = next(o for o in self._scanner.last_report.objects if o.instance_id == instance_id)
        try:
            mask = self._geometry.table_mask(observation, scanned.bbox_px)
            fit = self._pose(object_id, "table", observation, mask)
            self._base_poses[instance_id] = installed_pose(fit.world_from_stl, self._cad.get(object_id))
            self.base_fit_errors.pop(instance_id, None)
        except PerceptionPortError as error:
            self._base_poses.pop(instance_id, None)
            self.base_fit_errors[instance_id] = str(error)

    # -- pre_pick -----------------------------------------------------------------------------------------------
    def pre_pick(self, observation: Observation, part: InventoryItem) -> EvidencePair:
        center = self._center(part.instance_id)
        view = self._inspector.look(observation, part.object_id, center, self._geometry.table_surface_z_mm)
        verdict = judge(part.object_id, view.reading, view.fit_score, view.side, view.size, self._thresholds)
        self._closeups[part.instance_id] = self._next_step(part, view, verdict)
        return EvidencePair(detection(part.instance_id, view, verdict.object_id),
                            inspection(part.instance_id, view, verdict, view.size.residual_mm),
                            observation_time(observation))

    def _center(self, instance_id: str) -> tuple[float, float]:
        if instance_id not in self._xy:
            raise UnknownInstanceError(f"instance {instance_id!r} has no measured position from a scan")
        return self._xy[instance_id]

    def _next_step(self, part: InventoryItem, view: WristView, verdict: Judgement) -> CloseupReport:
        first = self._pending.get(part.instance_id)
        if not verdict.unsure:
            self._pending.pop(part.instance_id, None)
            step = NextStep.USE if verdict.action == "use" else NextStep.REJECT
            refs = (first.image_ref, view.image_ref) if first else (view.image_ref,)
            return CloseupReport(part.instance_id, part.object_id, step, verdict.reason, refs, None)
        if first is not None and another_angle(first, view):
            del self._pending[part.instance_id]
            return CloseupReport(part.instance_id, part.object_id, NextStep.ANDON, verdict.reason,
                                 (first.image_ref, view.image_ref), None)
        self._pending.setdefault(part.instance_id, view)
        top = view.pose.z_mm + self._cad.get(part.object_id).bbox_max_mm[2]
        return CloseupReport(part.instance_id, part.object_id, NextStep.SECOND_LOOK, verdict.reason,
                             (view.image_ref,), view_request(part.instance_id, part.object_id, view.pose, top))

    # -- post_place ---------------------------------------------------------------------------------------------
    def post_place(self, observation: Observation, part: InventoryItem) -> PlacementEvidence:
        tcp = current_waypoint(observation)
        base_id, base = self._base_under(tcp.x, tcp.y)
        cad = self._cad.get(part.object_id)
        floor = base.z_mm + cad.seated_origin_z_mm + cad.bbox_min_mm[2]
        view = self._inspector.look(observation, part.object_id, (tcp.x, tcp.y), floor)
        seat = seat_report(view.pose.z_mm, base.z_mm, cad.seated_origin_z_mm, view.pose.tilt_deg, self._wrist_pose,
                           self._table_pose)
        self._placements[part.instance_id] = seat
        verdict, seated, agrees = post_place_judgement(part.object_id, view, seat, self._thresholds)
        if seated and agrees and verdict.action == "use":
            self._record_installed(base_id, base)
        return PlacementEvidence(detection(part.instance_id, view, verdict.object_id),
                                 inspection(part.instance_id, view, verdict, abs(seat.seat_error_mm)),
                                 observation_time(observation), seated=seated, pose_agrees=agrees)

    def _base_under(self, x_mm: float, y_mm: float) -> tuple[str, InstalledPose]:
        for base_id, pose in self._base_poses.items():
            half = math.hypot(*self._cad.get(self._scanner.ledger.object_of(base_id)).size_mm[:2]) / 2.0
            if math.hypot(pose.x_mm - x_mm, pose.y_mm - y_mm) <= half:
                return base_id, pose
        raise PerceptionPortError(f"no base with a measured pose under the TCP at ({x_mm:.1f}, {y_mm:.1f})")

    def _record_installed(self, base_id: str, base: InstalledPose) -> None:
        half = math.hypot(*self._cad.get(self._scanner.ledger.object_of(base_id)).size_mm[:2]) / 2.0
        self._scanner.assembly_bases[base_id] = (base.x_mm, base.y_mm, half)

    # -- post_transfer ------------------------------------------------------------------------------------------
    def post_transfer(self, observation: Observation, part: InventoryItem,
                      target_zone: InventoryZone) -> RecoveryTransferEvidence:
        if target_zone not in _TRANSFER_ZONES:
            raise PerceptionPortError(f"no calibrated zone check for a transfer to {target_zone.value}")
        tcp = current_waypoint(observation)
        view = self._inspector.look(observation, part.object_id, (tcp.x, tcp.y), self._geometry.table_surface_z_mm)
        zone, _slot = self._geometry.zone(view.pose.x_mm, view.pose.y_mm)
        verdict = transfer_judgement(part.object_id, view, self._thresholds)
        agrees = verdict.action == "use" and zone is _TRANSFER_ZONES[target_zone]
        if agrees:
            self._moved(part, view.pose.x_mm, view.pose.y_mm)
        return RecoveryTransferEvidence(detection(part.instance_id, view, verdict.object_id),
                                        inspection(part.instance_id, view, verdict, view.size.residual_mm),
                                        observation_time(observation), target_zone, agrees)

    def _moved(self, part: InventoryItem, x_mm: float, y_mm: float) -> None:
        self._xy[part.instance_id] = (x_mm, y_mm)
        track = self._scanner.ledger.track_of(part.instance_id)
        if track is not None and any(t.instance_id == track for t in self._scanner.tracker.tracks()):
            self._scanner.tracker.moved(track, x_mm, y_mm)
        else:
            self._scanner.ledger.expect(part.instance_id, part.object_id, x_mm, y_mm)

    # -- collision_scene ----------------------------------------------------------------------------------------
    def collision_scene(self, observation: Observation, moving_instance_id: str,
                        base_instance_id: str) -> CollisionScene:
        self._require_resolved_obstacles()
        observed_at = observation_time(observation)
        base = self._refit_base(observation, base_instance_id)
        tcp = current_waypoint(observation)
        attached = tcp.gripper == 1.0 and held_part_visible(
            self._geometry.wrist_points(observation)[0], (tcp.x, tcp.y, tcp.z),
            self._cad.get(self._scanner.ledger.object_of(moving_instance_id)))
        intruded = intrusion(self._geometry.table_cloud(observation), joints(observation), *self._geometry.work_area,
                             tallest_stack_mm(self._cad) + self._table_pose.position_mm,
                             self._geometry.table_surface_z_mm, self._corridor)
        fixtures = tuple(box_from_pose(f"base:{iid}", pose, self._cad.get(self._scanner.ledger.object_of(iid)))
                         for iid, pose in sorted(self._base_poses.items()))
        return CollisionScene(observed_at, True, base_instance_id, base.contract_tuple(), base.tilt_deg,
                              moving_instance_id, attached, intruded, fixtures,
                              self._part_boxes(moving_instance_id))

    def _require_resolved_obstacles(self) -> None:
        report = self._scanner.last_report
        if report is None:
            raise PerceptionPortError("collision scene requested before the first scan")
        unresolved = len(report.unlocated) + len(report.side_unmatched)
        if unresolved:
            raise PerceptionPortError(
                f"collision scene has {unresolved} unresolved obstacle detection(s); motion must hold"
            )

    def _refit_base(self, observation: Observation, base_instance_id: str) -> InstalledPose:
        if base_instance_id not in self._base_poses:
            error = self.base_fit_errors.get(base_instance_id, "never fitted")
            raise PerceptionPortError(f"no measured pose for base {base_instance_id!r}: {error}")
        cad = self._cad.get(self._scanner.ledger.object_of(base_instance_id))
        box = self._geometry.table_box(observation, posed_bbox_corners(self._base_poses[base_instance_id], cad))
        fit = self._pose(cad.object_id, "table", observation, self._geometry.table_mask(observation, box))
        pose = installed_pose(fit.world_from_stl, cad)
        self._base_poses[base_instance_id] = pose
        return pose

    def _part_boxes(self, moving_instance_id: str) -> tuple:
        report = self._scanner.last_report
        if report is None:  # guarded by _require_resolved_obstacles; retained for direct internal misuse
            raise PerceptionPortError("collision scene requested before the first scan")
        boxes = []
        for obj in report.objects:
            if obj.instance_id == moving_instance_id or (obj.object_id or "").startswith("F"):
                continue
            footprint = cad_footprint(self._cad.get(obj.object_id)) if obj.object_id else obj.footprint_mm
            if footprint is None:
                raise PerceptionPortError(f"foreign piece {obj.instance_id} has no measured footprint")
            name = f"part:{obj.instance_id}" if obj.object_id else f"foreign:{obj.instance_id}"
            boxes.append(box_from_footprint(name, obj.x_mm, obj.y_mm, obj.top_z_mm, footprint,
                                            self._geometry.table_surface_z_mm))
        return tuple(boxes)
