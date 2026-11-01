"""Reject spots and store slots are reachable, used one at a time, and release parts above the table.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import pytest

from perception.cell.localize import CellZones, Zone
from perception.cell.targets import RELEASE_GAP_MM, TABLE_Z_UF0_MM, CellTargets, TargetError

ZONES = CellZones.load()


@pytest.fixture(scope="module")
def targets() -> CellTargets:
    return CellTargets(ZONES)


def test_every_reject_spot_is_inside_the_reject_zone(targets):
    assert len(targets.reject_spots) >= 3
    assert all(ZONES.zone(x, y)[0] is Zone.REJECT for x, y in targets.reject_spots)


def test_reject_spots_are_used_once_then_the_zone_is_full():
    fresh = CellTargets(ZONES)
    spots = {(r.x_mm, r.y_mm) for r in (fresh.next_reject(10.0) for _ in fresh.reject_spots)}
    assert len(spots) == len(fresh.reject_spots)
    with pytest.raises(TargetError, match="reject zone full"):
        fresh.next_reject(10.0)


def test_release_height_keeps_the_part_above_the_table(targets):
    release = targets.slot("S1", 12.0)
    assert release.z_mm - 12.0 == pytest.approx(TABLE_Z_UF0_MM + RELEASE_GAP_MM)
    assert release.approach_z_mm > release.z_mm
    assert (release.x_mm, release.y_mm) == (195.0, -330.0)


@pytest.mark.parametrize("below", [-1.0, float("nan"), 500.0])
def test_impossible_held_part_is_refused(targets, below):
    with pytest.raises(TargetError, match="not possible"):
        targets.slot("S1", below)


def test_unknown_slot_is_refused(targets):
    with pytest.raises(TargetError, match="no store slot"):
        targets.slot("S9", 10.0)


def test_an_unreachable_store_slot_fails_loading():
    from dataclasses import replace

    far = replace(ZONES, slots=(("S1", 900.0, 0.0),))
    with pytest.raises(TargetError, match="not reachable"):
        CellTargets(far)
