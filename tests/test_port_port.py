"""T30 CellPerceptionPort end to end on fakes: the evidence it returns drives the real T09 verification gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from assembly.catalog import load_catalog
from assembly.controller_ports import (
    CollisionScene,
    ControllerPortError,
    EvidencePair,
    InventorySnapshot,
    PerceptionPort,
    PlacementEvidence,
    reconcile_inventory,
)
from assembly.inventory import Inventory, InventoryItem, InventoryZone, ObjectCondition
from assembly.verification import (
    VerificationOutcome,
    VerificationThresholds,
    verify_post_place,
    verify_pre_pick,
    verify_recovery_transfer,
)
from perception.cell.localize import Located
from perception.port.backends import WristReading
from perception.port.dedup import RawBox
from perception.port.errors import PerceptionPortError, UnknownInstanceError
from perception.port.judge import NextStep
from perception.port.port import CellPerceptionPort
from perception.port.size_gate import Status
from tests.port_fakes import (
    CORRIDOR,
    POSE_MEASURED,
    POSE_PRELIMINARY,
    TABLE_Z,
    THRESHOLDS,
    WRIST_SIZE,
    FakeClassifier,
    FakeDetector,
    FakeGeometry,
    FakePose,
    MemorySink,
    loose_pose,
    observation,
    top_points,
    world_from_local,
)

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
REGISTRY = Registry().with_resources(
    (path.name, Resource.from_contents(json.loads(path.read_text()))) for path in CONTRACTS.glob("*.schema.json"))
DETECTION = Draft202012Validator(json.loads((CONTRACTS / "detection.schema.json").read_text()), registry=REGISTRY)
INSPECTION = Draft202012Validator(json.loads((CONTRACTS / "inspection.schema.json").read_text()), registry=REGISTRY)
VERIFY = VerificationThresholds(0.8, 0.6, 0.8, 1.0, 5.0)

BASE_BOX, PART_BOX, TAPE_BOX = (10, 10, 30, 30), (40, 10, 50, 20), (5, 30, 15, 35)
FALLEN_BOX, HOLE_BOX, REJECT_BOX = (55, 10, 60, 15), (0, 0, 2, 2), (60, 20, 63, 25)
BASE_POSE = world_from_local(450.0, 100.0, TABLE_Z + 8.0, 0.0)
PART_POSE = loose_pose("M1_02", 400.0, 0.0, 10.0)


def _located(x, y, top) -> Located:
    return Located(x, y, top, 200)


def _world(**overrides) -> tuple[CellPerceptionPort, FakeGeometry, FakeClassifier, FakePose, MemorySink]:
    detector = FakeDetector(table=[
        RawBox("F01", 0.95, BASE_BOX), RawBox("M1_02", 0.93, PART_BOX), RawBox("M2_02", 0.5, PART_BOX),
        RawBox("foreign_tape_roll", 0.9, TAPE_BOX), RawBox("M1_03__fallen", 0.9, FALLEN_BOX),
        RawBox("M1_04", 0.9, HOLE_BOX), RawBox("M2_04", 0.9, REJECT_BOX)],
        side=[RawBox("foreign_screw_m3", 0.8, (1, 1, 3, 3))])
    geometry = FakeGeometry(
        located={BASE_BOX: _located(450.0, 100.0, TABLE_Z + 11.0), PART_BOX: _located(400.0, 0.0, TABLE_Z + 17.0),
                 TAPE_BOX: _located(300.0, -200.0, TABLE_Z + 40.0), FALLEN_BOX: _located(350.0, 200.0, TABLE_Z + 20.0),
                 REJECT_BOX: _located(620.0, -350.0, TABLE_Z + 34.0)},
        table_part_points={PART_BOX: top_points("M1_02", PART_POSE),
                           TAPE_BOX: top_points("F01", world_from_local(300.0, -200.0, TABLE_Z + 37.0, 0.0))},
        wrist_cloud=top_points("M1_02", PART_POSE))
    pose = FakePose({("F01", "table"): BASE_POSE, ("M1_02", "wrist"): PART_POSE})
    classifier, sink = FakeClassifier(), MemorySink()
    kwargs = {"detector": detector, "classifier": classifier, "pose_estimator": pose, "geometry": geometry,
                  "thresholds": THRESHOLDS, "wrist_size": WRIST_SIZE, "wrist_pose": POSE_MEASURED, "table_pose": POSE_MEASURED,
                  "arm_corridor": CORRIDOR, "image_sink": sink} | overrides
    return CellPerceptionPort(**kwargs), geometry, classifier, pose, sink


def _scanned(port: CellPerceptionPort, t: float = 10.0) -> InventorySnapshot:
    return port.scan(observation(t))


def _clean_detector(*, side: list[RawBox] | None = None) -> FakeDetector:
    return FakeDetector(
        table=[
            RawBox("F01", 0.95, BASE_BOX),
            RawBox("M1_02", 0.93, PART_BOX),
            RawBox("foreign_tape_roll", 0.9, TAPE_BOX),
        ],
        side=[] if side is None else side,
    )


def _item(snapshot: InventorySnapshot, object_id: str, zone=InventoryZone.TABLE) -> InventoryItem:
    observed = next(i for i in snapshot.items if i.object_id == object_id)
    return InventoryItem(observed.instance_id, object_id, zone, observed.condition)


def _valid(evidence: EvidencePair) -> EvidencePair:
    assert list(DETECTION.iter_errors(evidence.detection)) == []
    assert list(INSPECTION.iter_errors(evidence.inspection)) == []
    return evidence


def _pre(evidence: EvidencePair, item: InventoryItem):
    return verify_pre_pick(item.object_id, evidence.detection, evidence.inspection,
                           expected_instance_id=item.instance_id, thresholds=VERIFY)


def test_the_port_satisfies_the_controller_protocol():
    port, *_ = _world()
    assert isinstance(port, CellPerceptionPort)
    _: PerceptionPort = port


# -- scan -------------------------------------------------------------------------------------------------------
def test_scan_reports_one_item_per_physical_object_and_reconciles():
    port, *_ = _world()
    snapshot = _scanned(port)
    assert snapshot.observed_at_s == 10.0 and snapshot.complete is False
    assert sorted(i.object_id for i in snapshot.items) == ["F01", "M1_02", "M1_03"]
    assert _item(snapshot, "M1_03").condition is ObjectCondition.FALLEN
    assert _item(snapshot, "M1_02").condition is ObjectCondition.VERIFIED
    report = port.last_scan
    assert [o.label for o in report.foreign] == ["foreign_tape_roll"]
    assert [o.object_id for o in report.outside] == ["M2_04"]
    assert [b.bbox_px for b in report.unlocated] == [HOLE_BOX]
    assert [b.label for b in report.side_unmatched] == ["foreign_screw_m3"]
    assert _item(snapshot, "M1_02").instance_id in report.needs_closeup
    with pytest.raises(ControllerPortError, match="inventory snapshot must be explicitly complete"):
        reconcile_inventory(Inventory(load_catalog()), snapshot, 10.0)


def test_scan_is_complete_only_when_every_detection_is_depth_located_and_cross_view_resolved():
    port, *_ = _world(detector=_clean_detector())
    snapshot = _scanned(port)
    assert snapshot.complete is True
    assert port.last_scan.unlocated == ()
    assert port.last_scan.side_unmatched == ()
    reconcile_inventory(Inventory(load_catalog()), snapshot, 10.0)


def test_known_side_only_detection_is_an_unresolved_obstacle():
    port, *_ = _world(detector=_clean_detector(side=[RawBox("M1_04", 0.9, (1, 1, 3, 3))]))
    snapshot = _scanned(port)
    assert snapshot.complete is False
    assert [box.label for box in port.last_scan.side_unmatched] == ["M1_04"]


def test_instance_ids_are_stable_across_scans():
    port, *_ = _world()
    first = {i.object_id: i.instance_id for i in _scanned(port, 10.0).items}
    second = {i.object_id: i.instance_id for i in _scanned(port, 11.0).items}
    assert first == second


def test_a_contested_identity_is_uncertain():
    port, *_ = _world(detector=FakeDetector(table=[RawBox("M1_02", 0.9, PART_BOX), RawBox("M2_02", 0.85, PART_BOX)]))
    assert _item(_scanned(port), "M1_02").condition is ObjectCondition.UNCERTAIN


# -- pre_pick ---------------------------------------------------------------------------------------------------
def test_a_good_part_passes_the_controller_pre_pick_gate():
    port, *_ = _world()
    item = _item(_scanned(port), "M1_02")
    evidence = _valid(port.pre_pick(observation(12.0), item))
    assert evidence.observed_at_s == 12.0
    assert evidence.detection["pose"] == pytest.approx({"x": 400.0, "y": 0.0, "z": TABLE_Z, "yaw": 10.0})
    assert _pre(evidence, item).outcome is VerificationOutcome.ACCEPT
    assert port.closeup(item.instance_id).next_step is NextStep.USE


def test_low_foundationpose_fit_is_a_warning_not_a_pre_pick_gate():
    port, _, _, pose, _ = _world()
    pose.fit = 0.3
    item = _item(_scanned(port), "M1_02")

    evidence = _valid(port.pre_pick(observation(12.0), item))
    decision = _pre(evidence, item)

    assert decision.outcome is VerificationOutcome.ACCEPT
    assert decision.advisories == ("foundationpose_low_fit", "foundationpose_disagreement")
    assert evidence.inspection["checks_agree"] is False
    assert port.closeup(item.instance_id).next_step is NextStep.USE


def test_a_cracked_part_is_rejected_to_a_spare():
    port, _, classifier, *_ = _world()
    classifier.reading = WristReading("M1_02", 0.97, 0.999)
    item = _item(_scanned(port), "M1_02")
    evidence = _valid(port.pre_pick(observation(12.0), item))
    assert evidence.inspection["verdict"] == "print_defect"
    assert _pre(evidence, item).outcome is VerificationOutcome.REJECT_TO_SPARE
    assert port.closeup(item.instance_id).next_step is NextStep.REJECT


def test_unsure_asks_for_a_second_look_then_andon_with_both_pictures():
    port, _, classifier, _, sink = _world()
    classifier.reading = WristReading("M1_02", 0.97, 0.7)
    item = _item(_scanned(port), "M1_02")
    first = _valid(port.pre_pick(observation(12.0), item))
    assert _pre(first, item).outcome is VerificationOutcome.HOLD_ANDON
    report = port.closeup(item.instance_id)
    assert report.next_step is NextStep.SECOND_LOOK and report.second_look.tilt_from_vertical_deg == 30.0
    port.pre_pick(observation(13.0), item)
    assert port.closeup(item.instance_id).next_step is NextStep.SECOND_LOOK   # same angle is not a second look
    port.pre_pick(observation(14.0, tcp=(400.0, 0.0, -168.0, -150.0, 0.0, 0.0)), item)
    andon = port.closeup(item.instance_id)
    assert andon.next_step is NextStep.ANDON and andon.image_refs == (sink.saved[0], sink.saved[2])


def test_a_part_the_port_never_saw_is_refused():
    port, *_ = _world()
    _scanned(port)
    with pytest.raises(UnknownInstanceError):
        port.pre_pick(observation(12.0), InventoryItem("ghost", "M1_02", InventoryZone.TABLE, ObjectCondition.VERIFIED))


def test_missing_depth_on_the_part_is_a_typed_error():
    port, geometry, *_ = _world()
    item = _item(_scanned(port), "M1_02")
    geometry.wrist_cloud = np.zeros((0, 3))
    with pytest.raises(PerceptionPortError):
        port.pre_pick(observation(12.0), item)


# -- post_place -------------------------------------------------------------------------------------------------
def _placed_world(part_z_offset: float = 0.0, tilt: float = 0.0, wrist_pose=POSE_MEASURED, pose_fit: float = 0.9):
    port, geometry, _classifier, pose, _ = _world(wrist_pose=wrist_pose)
    pose.fit = pose_fit
    snapshot = _scanned(port)
    seated = world_from_local(450.0, 100.0, BASE_POSE[2, 3] + 7.0 + part_z_offset, 0.0, tilt)
    pose.poses[("M1_02", "wrist")] = seated
    geometry.wrist_cloud = top_points("M1_02", seated)
    item = _item(snapshot, "M1_02", InventoryZone.RESERVED)
    obs = observation(20.0, tcp=(450.0, 100.0, -300.0, -180.0, 0.0, 0.0), gripper=1.0)
    return port, item, port.post_place(obs, item)


def _post(evidence: PlacementEvidence, item: InventoryItem):
    return verify_post_place(item.object_id, evidence.detection, evidence.inspection,
                             expected_instance_id=item.instance_id, thresholds=VERIFY, seated=evidence.seated,
                             pose_agrees=evidence.pose_agrees)


def test_a_certified_seat_is_accepted():
    port, item, evidence = _placed_world()
    _valid(evidence)
    assert (evidence.seated, evidence.pose_agrees) == (True, True)
    assert _post(evidence, item).outcome is VerificationOutcome.ACCEPT
    assert port.placement(item.instance_id).seat_error_mm == pytest.approx(0.0, abs=1e-6)


def test_low_foundationpose_fit_requests_operator_without_independent_seat_evidence():
    _, item, evidence = _placed_world(pose_fit=0.3)

    decision = _post(evidence, item)

    assert evidence.inspection["action"] == "ask_operator"
    assert decision.outcome is VerificationOutcome.HOLD_ANDON
    assert decision.advisories == ("foundationpose_low_fit", "foundationpose_disagreement")


def test_low_foundationpose_fit_never_turns_pose_error_into_automatic_recovery():
    _, item, evidence = _placed_world(part_z_offset=1.0, pose_fit=0.3)

    decision = _post(evidence, item)

    assert evidence.inspection["action"] == "ask_operator"
    assert decision.outcome is VerificationOutcome.HOLD_ANDON


def test_a_part_riding_high_is_recovered():
    _, item, evidence = _placed_world(part_z_offset=1.0)
    assert (evidence.seated, evidence.pose_agrees) == (False, False)
    assert _post(evidence, item).outcome is VerificationOutcome.RECOVER


def test_a_preliminary_bound_cannot_certify_the_seat_so_the_operator_decides():
    port, item, evidence = _placed_world(wrist_pose=POSE_PRELIMINARY)
    assert evidence.inspection["verdict"] == "uncertain"
    assert _post(evidence, item).outcome is VerificationOutcome.HOLD_ANDON
    assert port.placement(item.instance_id).uncertainty.status is Status.PRELIMINARY


def test_a_crooked_part_is_recovered():
    _, item, evidence = _placed_world(tilt=3.0)
    assert _post(evidence, item).outcome is VerificationOutcome.RECOVER


def test_an_accepted_install_keeps_the_base_observed_under_its_parts():
    port, _, evidence = _placed_world()
    assert evidence.seated
    base = next(i for i in port.scan(observation(21.0)).items if i.object_id == "F01")
    assert base.condition is ObjectCondition.VERIFIED
    assert base.instance_id in port._scanner.assembly_bases


# -- post_transfer ----------------------------------------------------------------------------------------------
def test_a_reject_transfer_is_verified_in_the_reject_zone():
    port, geometry, _, pose, _ = _world()
    item = _item(_scanned(port), "M1_02")
    dropped = loose_pose("M1_02", 620.0, -350.0)
    pose.poses[("M1_02", "wrist")] = dropped
    geometry.wrist_cloud = top_points("M1_02", dropped)
    evidence = port.post_transfer(observation(30.0, tcp=(620.0, -350.0, -250.0, -180.0, 0.0, 0.0)), item,
                                  InventoryZone.REJECT)
    assert evidence.pose_agrees is True and evidence.target_zone is InventoryZone.REJECT
    decision = verify_recovery_transfer(item.object_id, evidence.detection, evidence.inspection,
                                        expected_instance_id=item.instance_id, thresholds=VERIFY,
                                        pose_agrees=evidence.pose_agrees, allow_rejected=True)
    assert decision.outcome is VerificationOutcome.ACCEPT
    staged = port.post_transfer(observation(31.0, tcp=(620.0, -350.0, -250.0, -180.0, 0.0, 0.0)), item,
                                InventoryZone.STAGED)
    assert staged.pose_agrees is False


def test_low_foundationpose_fit_cannot_verify_a_recovery_zone():
    port, geometry, _, pose, _ = _world()
    item = _item(_scanned(port), "M1_02")
    original_xy = port._xy[item.instance_id]
    dropped = loose_pose("M1_02", 620.0, -350.0)
    pose.fit = 0.3
    pose.poses[("M1_02", "wrist")] = dropped
    geometry.wrist_cloud = top_points("M1_02", dropped)

    evidence = port.post_transfer(
        observation(30.0, tcp=(620.0, -350.0, -250.0, -180.0, 0.0, 0.0)),
        item,
        InventoryZone.REJECT,
    )
    decision = verify_recovery_transfer(
        item.object_id,
        evidence.detection,
        evidence.inspection,
        expected_instance_id=item.instance_id,
        thresholds=VERIFY,
        pose_agrees=evidence.pose_agrees,
        allow_rejected=True,
    )

    assert evidence.inspection["action"] == "ask_operator"
    assert evidence.pose_agrees is False
    assert decision.outcome is VerificationOutcome.HOLD_ANDON
    assert port._xy[item.instance_id] == original_xy


# -- collision_scene --------------------------------------------------------------------------------------------
def test_collision_scene_has_measured_boxes_and_a_held_part():
    port, geometry, *_ = _world(detector=_clean_detector())
    snapshot = _scanned(port)
    part, base = _item(snapshot, "M1_02"), _item(snapshot, "F01")
    held_at = world_from_local(400.0, 0.0, -150.0 - 11.4, 0.0)
    geometry.wrist_cloud = top_points("M1_02", held_at, top_offset_mm=-12.0)   # the side below the fingertips
    scene = port.collision_scene(observation(40.0, tcp=(400.0, 0.0, -150.0, -180.0, 0.0, 0.0), gripper=1.0),
                                 part.instance_id, base.instance_id)
    assert isinstance(scene, CollisionScene) and scene.observed_at_s == 40.0 and scene.complete
    assert scene.moving_attached is True and scene.intrusion_detected is False
    assert scene.base_pose == pytest.approx((450.0, 100.0, TABLE_Z + 8.0, 0.0))
    assert [b.name for b in scene.fixtures] == [f"base:{base.instance_id}"]
    names = sorted(b.name for b in scene.parts)
    assert any(n.startswith("foreign:") for n in names) and not any(part.instance_id in n for n in names)


def test_a_dropped_part_and_an_intrusion_are_reported():
    port, geometry, *_ = _world(detector=_clean_detector())
    snapshot = _scanned(port)
    geometry.wrist_cloud = np.zeros((0, 3))
    geometry.cloud = np.array([(300.0 + i * 0.1, 300.0, TABLE_Z + 120.0) for i in range(40)])
    scene = port.collision_scene(observation(41.0, gripper=1.0), _item(snapshot, "M1_02").instance_id,
                                 _item(snapshot, "F01").instance_id)
    assert scene.moving_attached is False and scene.intrusion_detected is True


def test_a_base_without_a_measured_pose_blocks_the_scene():
    port, _, _, pose, _ = _world(detector=_clean_detector())
    del pose.poses[("F01", "table")]
    snapshot = _scanned(port)
    base = _item(snapshot, "F01")
    assert "no fit" in port.base_fit_errors[base.instance_id]
    with pytest.raises(PerceptionPortError, match="no measured pose"):
        port.collision_scene(observation(42.0), _item(snapshot, "M1_02").instance_id, base.instance_id)


@pytest.mark.parametrize(
    "detector",
    [
        _clean_detector(side=[RawBox("foreign_screw_m3", 0.8, (1, 1, 3, 3))]),
        FakeDetector(
            table=[
                RawBox("F01", 0.95, BASE_BOX),
                RawBox("M1_02", 0.93, PART_BOX),
                RawBox("M1_04", 0.9, HOLE_BOX),
            ],
            side=[],
        ),
    ],
)
def test_collision_scene_refuses_unresolved_obstacles(detector: FakeDetector):
    port, *_ = _world(detector=detector)
    snapshot = _scanned(port)
    with pytest.raises(PerceptionPortError, match="unresolved obstacle"):
        port.collision_scene(
            observation(42.0),
            _item(snapshot, "M1_02").instance_id,
            _item(snapshot, "F01").instance_id,
        )
