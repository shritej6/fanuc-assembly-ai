"""Tests for fixed inspection registry.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import pytest

from assembly.fixed_inspection import (
    FixedInspectionError,
    InspectionPoseRegistry,
    InspectionSlotPose,
)


def _slot(
    slot_id: str,
    bounds: tuple[float, float, float, float],
) -> InspectionSlotPose:
    return InspectionSlotPose(
        slot_id=slot_id,
        selection_bounds_xy_mm=bounds,
        endpoint_tcp=(321.0, -45.0, -168.0, -180.0, 0.0, 77.0),
        safe_z_mm=-180.0,
        source="reviewed CSI slot-pose test fixture",
    )


def test_registry_returns_exact_reviewed_pose_not_observed_center() -> None:
    registry = InspectionPoseRegistry((_slot("slot-a", (240.0, 260.0, -110.0, -90.0)),))

    resolved = registry.resolve((250.5, -99.5))

    assert resolved.slot_id == "slot-a"
    assert resolved.endpoint_tcp == (321.0, -45.0, -168.0, -180.0, 0.0, 77.0)


def test_registry_rejects_missing_or_ambiguous_slot_authority() -> None:
    registry = InspectionPoseRegistry((_slot("slot-a", (240.0, 260.0, -110.0, -90.0)),))

    with pytest.raises(FixedInspectionError, match="no reviewed inspection slot"):
        registry.resolve((400.0, 300.0))

    with pytest.raises(FixedInspectionError, match="overlap"):
        InspectionPoseRegistry(
            (
                _slot("slot-a", (240.0, 260.0, -110.0, -90.0)),
                _slot("slot-b", (255.0, 275.0, -100.0, -80.0)),
            )
        )
