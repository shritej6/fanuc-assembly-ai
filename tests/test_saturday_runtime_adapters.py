"""Saturday production-adapter regressions.

These tests cover the source-pinned adapters without loading GPU model weights.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from PIL import Image

from assembly.catalog import load_catalog
from assembly.controller_ports import (
    ControllerPortError,
    InventorySnapshot,
    ObservedInventoryItem,
)
from assembly.inventory import InventoryItem, InventoryZone, ObjectCondition
from assembly.safety import SafetyConfigurationViolation
from perception.port.cad import load_cad
from perception.port.frames import installed_pose
from perception.port.placement import SEAT_GATE_MM, TILT_GATE_DEG
from perception.port.size_gate import Status
from robot.cell.real_layout import ExpectedLayoutVerifier, load_real_cell_layout, reviewed_final_view_registry
from robot.cell.saturday_runtime_adapters import (
    PLACEMENT_POSE_UNCERTAINTY,
    UNVERIFIED_PRESENCE_ENV,
    DetectorPresenceGate,
    PlacementMeasuredPoseBridge,
    PlacementPoseEstimator,
    PngJsonlImageSink,
    PresenceState,
    PresenceStatus,
    SaturdayAdapterBlockedError,
    _wire_snapshot_bridge,
    catalog_installed_orientations,
    conservative_closeup_thresholds,
    conservative_corridor,
    load_saturday_runtime_adapters,
)


def _observation(index: int = 0) -> dict[str, object]:
    rollout = load_real_cell_layout().rollouts[index]
    return {
        "t": 10.0,
        "trial": {"index": index, "instruction": rollout.prompt},
    }


def _item(instance_id: str, object_id: str) -> InventoryItem:
    return InventoryItem(
        instance_id,
        object_id,
        InventoryZone.RESERVED,
        ObjectCondition.VERIFIED,
        reserved_for=5,
        source_zone=InventoryZone.TABLE,
    )


def _reviewed_snapshot(*, skip: str | None = None, extra: str | None = None) -> InventorySnapshot:
    items = [
        ObservedInventoryItem(
            f"seen-{index}",
            placed.object_id,
            InventoryZone.TABLE,
            ObjectCondition.VERIFIED,
            (placed.center_x_mm, placed.center_y_mm),
        )
        for index, placed in enumerate(load_real_cell_layout().rollouts[0].objects)
        if placed.object_id != skip
    ]
    if extra is not None:
        items.append(
            ObservedInventoryItem(
                "seen-extra",
                extra,
                InventoryZone.TABLE,
                ObjectCondition.VERIFIED,
                (200.0, 300.0),
            )
        )
    return InventorySnapshot(10.0, True, tuple(items))


def test_placement_pose_estimator_uses_current_reviewed_rollout() -> None:
    layout, cad = load_real_cell_layout(), load_cad()
    estimator = PlacementPoseEstimator(layout, cad)
    expected = next(item for item in layout.rollouts[0].objects if item.object_id == "M5_01")

    fit = estimator("M5_01", "table", _observation(), np.ones((2, 2), dtype=bool))
    pose = installed_pose(fit.world_from_stl, cad.get("M5_01"))

    assert (pose.x_mm, pose.y_mm, pose.yaw_deg) == pytest.approx(
        (expected.center_x_mm, expected.center_y_mm, expected.yaw_deg)
    )
    assert pose.z_mm + cad.get("M5_01").bbox_min_mm[2] == pytest.approx(
        estimator.table_surface_z_mm
    )
    assert fit.fit_score == 0.0
    assert estimator.presence.status is PresenceStatus.UNVERIFIED


def test_placement_pose_estimator_rejects_wrong_trial_prompt() -> None:
    estimator = PlacementPoseEstimator(load_real_cell_layout(), load_cad())
    observation = _observation()
    observation["trial"] = {"index": 0, "instruction": "unreviewed prompt"}

    with pytest.raises(ControllerPortError, match="does not match reviewed placement"):
        estimator("M5_01", "table", observation, np.ones((1, 1), dtype=bool))


def test_placement_prior_is_preliminary_and_does_not_widen_gates() -> None:
    assert PLACEMENT_POSE_UNCERTAINTY.status is Status.PRELIMINARY
    assert PLACEMENT_POSE_UNCERTAINTY.position_mm == 2.0
    assert PLACEMENT_POSE_UNCERTAINTY.angle_deg == 2.0
    assert PLACEMENT_POSE_UNCERTAINTY.position_mm > SEAT_GATE_MM
    assert PLACEMENT_POSE_UNCERTAINTY.angle_deg > TILT_GATE_DEG


def test_png_jsonl_sink_writes_auditable_pair(tmp_path) -> None:
    sink = PngJsonlImageSink(tmp_path)
    image = np.full((3, 4, 3), 127, dtype=np.uint8)

    reference = sink("pre pick/M5_01", image)

    path = tmp_path / "images" / "000001-pre-pick-M5_01.png"
    assert reference == path.resolve().as_uri()
    assert np.asarray(Image.open(path)).tolist() == image.tolist()
    records = [json.loads(line) for line in (tmp_path / "images.jsonl").read_text().splitlines()]
    assert records == [
        {
            "dtype": "uint8",
            "name": "pre pick/M5_01",
            "path": "images/000001-pre-pick-M5_01.png",
            "sequence": 1,
            "shape": [3, 4, 3],
        }
    ]


def test_measured_pose_bridge_maps_dynamic_instance_ids_to_placement() -> None:
    catalog, cad = load_catalog(), load_cad()
    estimator = PlacementPoseEstimator(load_real_cell_layout(), cad)
    estimator("F05", "table", _observation(), np.ones((1, 1), dtype=bool))
    bridge = PlacementMeasuredPoseBridge(catalog, estimator)

    base = bridge.base_for(_item("track-base-7", "F05"), observed_at_s=10.0)
    grasp = bridge.grasp_for(_item("track-part-12", "M5_01"), observed_at_s=10.0)

    assert (base.tcp.x, base.tcp.y, base.tcp.r) == pytest.approx((420.0, 0.0, 45.0))
    assert base.up_face == "placement-base:+z"
    local = catalog.product(5).steps[0].preferred_grasp.tcp_local_mm
    angle = np.radians(30.0)
    expected_tcp = (
        335.0 + np.cos(angle) * local[0] - np.sin(angle) * local[1],
        -45.0 + np.sin(angle) * local[0] + np.cos(angle) * local[1],
        30.0,
    )
    assert (grasp.tcp.x, grasp.tcp.y, grasp.tcp.r) == pytest.approx(expected_tcp)
    assert base.tcp.source == grasp.tcp.source == "calibrated"


def test_installed_up_faces_derive_from_each_catalog_rotation() -> None:
    orientations = catalog_installed_orientations(load_catalog())

    assert orientations["M1_01"].up_face == "installed:+z"
    assert orientations["M1_04"].up_face == "installed:+y"
    assert orientations["M1_01"].up_face != orientations["M1_04"].up_face


def test_pose_bridge_preserves_catalog_derived_flat_part_up_face() -> None:
    catalog, cad = load_catalog(), load_cad()
    estimator = PlacementPoseEstimator(load_real_cell_layout(), cad)
    estimator("M1_04", "table", _observation(), np.ones((1, 1), dtype=bool))
    bridge = PlacementMeasuredPoseBridge(catalog, estimator)

    grasp = bridge.grasp_for(_item("rear-guard", "M1_04"), observed_at_s=10.0)

    assert grasp.up_face == "installed:+y"


def test_verified_detector_presence_gate_rejects_missing_and_extra() -> None:
    catalog, layout = load_catalog(), load_real_cell_layout()
    gate = DetectorPresenceGate(catalog, layout, tolerance_mm=25.0)

    gate.verify(_observation(), _reviewed_snapshot())
    with pytest.raises(ControllerPortError, match="missing M5_01"):
        gate.verify(_observation(), _reviewed_snapshot(skip="M5_01"))
    with pytest.raises(ControllerPortError, match="unexpected table detection M6_01"):
        gate.verify(_observation(), _reviewed_snapshot(extra="M6_01"))


def test_interchangeable_base_uses_the_reviewed_slot_pose() -> None:
    catalog, cad, layout = load_catalog(), load_cad(), load_real_cell_layout()
    expected = next(item for item in layout.rollouts[0].objects if item.object_id == "F05")
    items = tuple(
        ObservedInventoryItem(
            item.instance_id,
            "F04" if item.object_id == "F05" else item.object_id,
            item.zone,
            item.condition,
            item.coarse_xy_mm,
        )
        for item in _reviewed_snapshot().items
    )
    snapshot = InventorySnapshot(10.0, True, items)
    actual = next(
        item
        for item in snapshot.items
        if item.object_id == "F04" and item.coarse_xy_mm == (expected.center_x_mm, expected.center_y_mm)
    )
    estimator = PlacementPoseEstimator(
        layout,
        cad,
        presence=PresenceState(PresenceStatus.VERIFIED, "test detector artifact"),
    )
    estimator.activate(_observation())
    bridge = PlacementMeasuredPoseBridge(catalog, estimator)
    gate = DetectorPresenceGate(catalog, layout, tolerance_mm=25.0)

    gate.verify(_observation(), snapshot)
    bridge.publish(snapshot)
    base = bridge.base_for(_item(actual.instance_id, "F04"), observed_at_s=10.0)

    assert (base.tcp.x, base.tcp.y, base.tcp.r) == pytest.approx(
        (expected.center_x_mm, expected.center_y_mm, expected.yaw_deg)
    )
    assert base.up_face == "placement-base:+z"


def test_presence_gate_uses_fixed_25_mm_axis_box() -> None:
    layout = load_real_cell_layout()
    expected = layout.rollouts[0].objects[0]
    snapshot = _reviewed_snapshot()
    shifted = tuple(
        ObservedInventoryItem(
            item.instance_id,
            item.object_id,
            item.zone,
            item.condition,
            (
                expected.center_x_mm + 25.1,
                expected.center_y_mm,
            )
            if item.object_id == expected.object_id
            else item.coarse_xy_mm,
        )
        for item in snapshot.items
    )
    gate = DetectorPresenceGate(load_catalog(), layout, tolerance_mm=25.0)

    with pytest.raises(ControllerPortError, match="outside the reviewed 25.0 mm presence box"):
        gate.verify(_observation(), InventorySnapshot(10.0, True, shifted))


def test_verified_detector_scan_path_applies_extra_presence_gate(tmp_path) -> None:
    class FakePerception:
        def scan(self, observation):
            del observation
            return _reviewed_snapshot(extra="M6_01")

    catalog, cad, layout = load_catalog(), load_cad(), load_real_cell_layout()
    estimator = PlacementPoseEstimator(
        layout,
        cad,
        presence=PresenceState(PresenceStatus.VERIFIED, "test detector artifact"),
    )
    perception = FakePerception()
    _wire_snapshot_bridge(
        perception,  # type: ignore[arg-type]
        estimator,
        PlacementMeasuredPoseBridge(catalog, estimator),
        DetectorPresenceGate(catalog, layout, tolerance_mm=25.0),
        PngJsonlImageSink(tmp_path),
    )

    with pytest.raises(ControllerPortError, match="unexpected table detection M6_01"):
        perception.scan(_observation())


def test_conservative_preliminary_values_are_no_looser_than_current_gates() -> None:
    thresholds = conservative_closeup_thresholds()
    corridor = conservative_corridor()

    assert thresholds.classifier_min == 1.0
    assert thresholds.pose_fit_min == 1.0
    assert corridor.radius_mm <= 25.0
    assert "CONSTRUCTION-ONLY" in thresholds.source
    assert "CONSTRUCTION-ONLY" in corridor.source


def test_loader_refuses_missing_weights_without_explicit_arming(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(UNVERIFIED_PRESENCE_ENV, raising=False)
    monkeypatch.setenv("FANUC_CELL_RUN_DIR", str(tmp_path))
    catalog, cad, layout = load_catalog(), load_cad(), load_real_cell_layout()

    with pytest.raises(SaturdayAdapterBlockedError, match=UNVERIFIED_PRESENCE_ENV):
        load_saturday_runtime_adapters(
            catalog,
            cad,
            ExpectedLayoutVerifier(layout, catalog),
            reviewed_final_view_registry(),
        )


def test_loader_constructs_fail_closed_bundle_when_explicitly_armed(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(UNVERIFIED_PRESENCE_ENV, "1")
    monkeypatch.setenv("FANUC_CELL_RUN_DIR", str(tmp_path))
    catalog, cad, layout = load_catalog(), load_cad(), load_real_cell_layout()

    adapters = load_saturday_runtime_adapters(
        catalog,
        cad,
        ExpectedLayoutVerifier(layout, catalog),
        reviewed_final_view_registry(),
    )

    assert type(adapters.perception).__name__ == "CellPerceptionPort"
    assert isinstance(adapters.poses, PlacementMeasuredPoseBridge)
    assert adapters.perception.expected_layout_verifier is not None
    assert adapters.motion_config.safe_z_mm == -180.0
    assert math.isnan(adapters.motion_config.approach_mm)
    with pytest.raises(SafetyConfigurationViolation, match="motion is blocked"):
        adapters.safety.validate()
    view = reviewed_final_view_registry().views[0]
    with pytest.raises(ControllerPortError, match="completion is blocked"):
        adapters.final_identity_backend.capture({}, view, "base-1", ("p1", "p2", "p3", "p4"))
