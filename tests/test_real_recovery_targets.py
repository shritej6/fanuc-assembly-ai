"""Real recovery poses use only reviewed cell-layout and pinned CAD/T29 facts."""

from __future__ import annotations

import pytest

from assembly.catalog import load_catalog
from assembly.inventory import InventoryItem, InventoryZone, ObjectCondition
from assembly.recovery import RecoveryError
from perception.cell.localize import CellZones, Zone
from perception.cell.targets import RELEASE_GAP_MM, TABLE_Z_UF0_MM, CellTargets
from perception.port.cad import load_cad
from robot.cell.real_recovery import RealCellRecoveryTargets


def _part(object_id: str = "M1_02") -> InventoryItem:
    return InventoryItem("part-1", object_id, InventoryZone.INSTALLED, ObjectCondition.VERIFIED, 1)


def _targets(*, pull_out_slots=None) -> RealCellRecoveryTargets:
    return RealCellRecoveryTargets(
        catalog=load_catalog(),
        cad=load_cad(),
        cell_targets=CellTargets(CellZones.load()),
        staging_slots=("S1", "S2", "S3"),
        pull_out_slots={} if pull_out_slots is None else pull_out_slots,
        source="CSI operator verified cell_layout.yaml slots on 2026-10-31",
    )


def test_staging_target_preserves_installed_face_and_clears_table() -> None:
    part = _part()
    targets = _targets()
    target = targets.staging_target_for(1, part, 1)
    cad = load_cad().get(part.object_id)
    held_below = cad.tcp_height_mm - cad.bbox_min_mm[2]

    assert (target.tcp.x, target.tcp.y) == (195.0, -330.0)
    assert target.tcp.z - held_below == pytest.approx(TABLE_Z_UF0_MM + RELEASE_GAP_MM)
    assert target.tcp.w == -180.0 and target.tcp.p == 0.0
    assert target.resting_state.value == "installed"
    assert target.up_face == "installed:+z"
    assert targets.staging_verification_target_for(1, part, 1) == target


def test_reject_targets_are_unique_and_inside_cell_layout_zone() -> None:
    targets = _targets()
    first = targets.reject_target_for(_part())
    second = targets.reject_target_for(_part())
    zones = CellZones.load()

    assert (first.tcp.x, first.tcp.y) != (second.tcp.x, second.tcp.y)
    assert zones.zone(first.tcp.x, first.tcp.y)[0] is Zone.REJECT
    assert zones.zone(second.tcp.x, second.tcp.y)[0] is Zone.REJECT


def test_unreviewed_staging_and_pull_out_slots_fail_closed() -> None:
    part = _part()
    targets = _targets()

    with pytest.raises(RecoveryError, match="reviewed empty staging slot"):
        targets.staging_target_for(1, part, 4)
    with pytest.raises(RecoveryError, match="reviewed pull-out slot"):
        targets.pull_out_target_for(1, part)


def test_pull_out_uses_explicit_empty_slot_and_cell_clearance() -> None:
    part = _part()
    targets = _targets(pull_out_slots={(1, "M1_02"): "S4"})

    target = targets.pull_out_target_for(1, part)

    assert (target.pose.tcp.x, target.pose.tcp.y) == (195.0, -380.0)
    assert target.clear_z_mm > target.pose.tcp.z
    assert "cell_layout.yaml" in target.source
    assert "T29" in target.source


def test_product_identity_and_calibration_source_are_required() -> None:
    with pytest.raises(RecoveryError, match="calibration source"):
        RealCellRecoveryTargets(
            catalog=load_catalog(),
            cad=load_cad(),
            cell_targets=CellTargets(CellZones.load()),
            staging_slots=("S1",),
            pull_out_slots={},
            source="",
        )

    with pytest.raises(RecoveryError, match="does not belong to product 2"):
        _targets().staging_target_for(2, _part(), 1)
