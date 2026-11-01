"""Port building blocks: duplicate suppression, pinned CAD, frames, defect band, ledger, placement checks.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import shutil
from hashlib import sha256

import numpy as np
import pytest
import yaml

from assembly.calibrated_poses import _matmul, _rigid_inverse
from assembly.catalog import load_catalog
from perception.cell.inspect import same_shape
from perception.port import defect
from perception.port.cad import OBJECTS_SNAPSHOT, load_cad
from perception.port.dedup import RawBox, suppress
from perception.port.errors import CadSourceError, ModelOutputError, UnknownInstanceError
from perception.port.frames import installed_pose, resting_side
from perception.port.ledger import Ledger
from perception.port.placement import (
    Check,
    PoseUncertainty,
    check,
    combined,
    precision_gate_tolerances,
    seat_report,
)
from perception.port.size_gate import Status
from tests.port_fakes import POSE_MEASURED, POSE_PRELIMINARY, world_from_local

CAD = load_cad()


# -- dedup ------------------------------------------------------------------------------------------------------
def test_two_classes_on_one_part_become_one_box_with_the_runner_up_kept():
    boxes = [RawBox("F04", 0.68, (847, 770, 957, 800)), RawBox("F03", 0.5, (850, 771, 955, 801))]
    (kept,) = suppress(boxes)
    assert kept.label == "F04" and kept.alternatives == (("F03", 0.5),)


def test_separate_parts_are_not_merged():
    boxes = [RawBox("M1_02", 0.9, (0, 0, 10, 10)), RawBox("M1_02", 0.8, (20, 0, 30, 10))]
    assert len(suppress(boxes)) == 2


def test_a_box_nested_in_a_larger_one_is_a_duplicate_only_when_almost_covered():
    outer = RawBox("M4_01", 0.9, (0, 0, 100, 100))
    assert len(suppress([outer, RawBox("M4_02", 0.8, (10, 10, 30, 30))])) == 1
    assert len(suppress([outer, RawBox("M4_02", 0.8, (90, 90, 130, 130))])) == 2


def test_a_close_runner_up_of_another_shape_is_contested_but_a_same_mesh_base_is_not():
    (part,) = suppress([RawBox("M1_02", 0.9, (0, 0, 10, 10)), RawBox("M2_02", 0.85, (0, 0, 10, 10))])
    assert part.contested_by(same_shape) == ("M2_02",)
    (base,) = suppress([RawBox("F01", 0.9, (0, 0, 10, 10)), RawBox("F02", 0.89, (0, 0, 10, 10))])
    assert base.contested_by(same_shape) == ()


@pytest.mark.parametrize("box", [RawBox("M1_02", 1.2, (0, 0, 1, 1)), RawBox("M1_02", 0.5, (5, 0, 1, 1))])
def test_malformed_detector_boxes_are_typed_errors(box):
    with pytest.raises(ModelOutputError):
        suppress([box])


# -- CAD ----------------------------------------------------------------------------------------------------------
def test_an_altered_cad_snapshot_is_refused(tmp_path):
    copy = tmp_path / OBJECTS_SNAPSHOT.name
    shutil.copy(OBJECTS_SNAPSHOT, copy)
    copy.write_bytes(copy.read_bytes() + b"\n# edited\n")
    with pytest.raises(CadSourceError, match="sha256"):
        load_cad.__wrapped__(copy)


def test_cad_facts_come_from_the_t12_snapshot():
    assert CAD.get("F01").size_mm == CAD.get("F02").size_mm
    assert CAD.candidates_for("M1_01") == ("M1_01", "M2_01", "M3_01")
    assert CAD.get("M1_01").cues["M3_01"][0].top_z_other_mm == 16.0
    assert CAD.get("M3_01").cues["M1_01"][0].top_z_here_mm == 16.0
    assert round(CAD.get("M3_02").fallen_tilt_deg, 1) == 34.7
    assert CAD.get("M1_02").tcp_height_mm == 11.4 and CAD.get("F01").tcp_height_mm is None
    with pytest.raises(CadSourceError):
        CAD.get("M9_09")


def test_tcp_height_is_cross_checked_against_the_authorized_t29_grasp(tmp_path):
    catalog = load_catalog()
    authorized = {
        step.object_id: step.preferred_grasp.tcp_local_mm[2]
        for product in catalog.products
        for step in product.steps
    }
    assert all(CAD.get(object_id).tcp_height_mm == height for object_id, height in authorized.items())

    altered = yaml.safe_load(OBJECTS_SNAPSHOT.read_text())
    altered["objects"]["M1_02"]["grips"][0]["tcp_local_mm"][2] = 999.0
    copy = tmp_path / OBJECTS_SNAPSHOT.name
    copy.write_text(yaml.safe_dump(altered, sort_keys=False))
    with pytest.raises(CadSourceError, match="authorized T29 grasp"):
        load_cad.__wrapped__(copy, sha256(copy.read_bytes()).hexdigest())


# -- frames -----------------------------------------------------------------------------------------------------
def test_foundationpose_stl_pose_is_normalised_like_the_controller_targets():
    cad = CAD.get("M1_04")
    world_from_stl = world_from_local(400.0, 10.0, -300.0, 35.0) @ np.array(cad.local_from_stl)
    pose = installed_pose(world_from_stl, cad)
    expected = _matmul(tuple(map(tuple, world_from_stl)), _rigid_inverse(cad.local_from_stl))
    assert np.allclose(pose.world_from_local, np.array(expected))
    assert pose.contract_pose() == pytest.approx({"x": 400.0, "y": 10.0, "z": -300.0, "yaw": 35.0})
    assert pose.tilt_deg == pytest.approx(0.0, abs=1e-6)


def test_non_rigid_poses_are_refused():
    bad = np.eye(4)
    bad[0, 0] = 2.0
    with pytest.raises(ModelOutputError):
        installed_pose(bad, CAD.get("M1_02"))


@pytest.mark.parametrize(("tilt", "side"), [(0.0, "upright"), (30.0, "upright"), (40.0, "side"), (170.0, "flipped")])
def test_resting_side_uses_half_the_smallest_cad_stable_pose_tilt(tilt, side):
    cad = CAD.get("M3_02")
    pose = installed_pose(world_from_local(0, 0, 0, 0, tilt) @ np.array(cad.local_from_stl), cad)
    assert resting_side(pose, cad) == side


# -- defect band ------------------------------------------------------------------------------------------------
def test_careful_band_matches_the_user_decision():
    assert (defect.SECOND_LOOK_FROM, defect.REJECT_FROM) == (0.5, 0.994)
    assert defect.defect_call(0.4999) is defect.DefectCall.USE
    assert defect.defect_call(0.5) is defect.DefectCall.SECOND_LOOK
    assert defect.defect_call(0.9939) is defect.DefectCall.SECOND_LOOK
    assert defect.defect_call(0.994) is defect.DefectCall.REJECT


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan"), True])
def test_defect_probability_must_be_a_probability(p):
    with pytest.raises(ModelOutputError):
        defect.defect_call(p)


# -- ledger -----------------------------------------------------------------------------------------------------
def test_ledger_keeps_ids_and_mints_a_new_one_when_identity_changes():
    ledger = Ledger()
    assert ledger.publish("p1", "M1_02", 0, 0).instance_id == "p1"
    assert ledger.publish("p1", "M1_02", 0, 0).instance_id == "p1"
    assert ledger.publish("p1", "M2_02", 0, 0).instance_id == "p1r1"
    assert ledger.object_of("p1r1") == "M2_02"
    with pytest.raises(UnknownInstanceError):
        ledger.object_of("p1")


def test_a_moved_part_keeps_its_id_on_a_new_track():
    ledger = Ledger()
    ledger.publish("p1", "M1_02", 0, 0)
    ledger.expect("p1", "M1_02", 500.0, -300.0)
    assert ledger.publish("p9", "M1_02", 505.0, -300.0).instance_id == "p1"
    assert ledger.publish("p10", "M1_02", 900.0, 0.0).instance_id == "p10"


# -- placement --------------------------------------------------------------------------------------------------
def test_preliminary_bounds_never_certify_but_can_refute():
    assert check(0.0, 0.05, 0.3, Status.PRELIMINARY) is Check.UNSURE
    assert check(0.0, 0.05, 0.3, Status.MEASURED) is Check.CERTIFIED
    assert check(0.2, 0.2, 0.3, Status.MEASURED) is Check.UNSURE
    assert check(2.0, 0.8, 0.3, Status.PRELIMINARY) is Check.REFUTED


def test_seat_error_is_relative_to_the_base_and_bounds_add_in_quadrature():
    report = seat_report(-293.0, -300.0, 7.0, 0.1, POSE_MEASURED, POSE_MEASURED)
    assert report.seat_error_mm == pytest.approx(0.0) and report.seated and report.level
    high = seat_report(-292.0, -300.0, 7.0, 0.1, POSE_PRELIMINARY, POSE_MEASURED)
    assert high.seat is Check.UNSURE and high.uncertainty.status is Status.PRELIMINARY
    assert combined(POSE_MEASURED, POSE_MEASURED).position_mm == pytest.approx(0.05 * 2 ** 0.5)


def test_gate_tolerances_are_the_cad_precision_gates():
    execution, perception = precision_gate_tolerances()
    assert (execution.max_position_error_mm, execution.max_orientation_error_deg) == (0.3, 1.0)
    assert (perception.max_position_error_mm, perception.max_yaw_error_deg) == (0.3, 1.0)


def test_pose_uncertainty_must_be_sourced():
    with pytest.raises(Exception, match="source"):
        PoseUncertainty(0.1, 0.1, Status.MEASURED, "")
