"""Fail-closed Saturday adapters for supervised construction proof only.

The explicit unverified-presence flag permits dependency construction, not a
successful build. Missing measured motion, collision, arm-corridor, and final
identity authority all remain typed stops before physical motion or DONE.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from types import MethodType

import numpy as np

from assembly.catalog import Catalog
from assembly.controller_ports import ControllerPortError, InventorySnapshot, observation_time
from assembly.final_verification import FinalViewIdentityEvidence
from assembly.final_views import FinalViewPose, FinalViewRegistry
from assembly.motion import MotionConfig
from assembly.verification import VerificationThresholds
from perception.cell.models import ModelLoadError
from perception.port.backends import DefectHeadClassifier, RtDetrDetector, WristReading
from perception.port.cad import CadCatalog
from perception.port.geometry import CalibratedCellGeometry
from perception.port.judge import CloseupThresholds
from perception.port.port import CellPerceptionPort
from perception.port.scene import ArmCorridor
from perception.port.size_gate import WRIST_150MM_PRELIMINARY
from perception.taxonomy import load_ontology
from robot.cell._saturday_runtime_support import (
    DETECTOR_PRESENCE_TOLERANCE_MM,
    PLACEMENT_POSE_UNCERTAINTY,
    DetectorPresenceGate,
    PlacementMeasuredPoseBridge,
    PlacementPoseEstimator,
    PngJsonlImageSink,
    PresenceState,
    PresenceStatus,
    SaturdayAdapterBlockedError,
    catalog_installed_orientations,
)
from robot.cell.real_layout import (
    INSPECTION_SAFE_Z_MM,
    ExpectedLayoutVerifier,
    load_real_cell_layout,
)
from robot.cell.saturday_placement_evidence import PlacementInspectionEvidence
from robot.cell.saturday_scene_safety import SaturdaySceneSafetySupervisor

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "contracts" / "taxonomy.json"
UNVERIFIED_PRESENCE_ENV = "FANUC_T09_ALLOW_UNVERIFIED_PRESENCE"
DETECTOR_RUN_ENV = "FANUC_T09_DETECTOR_RUN_DIR"
CLASSIFIER_RUN_ENV = "FANUC_T09_CLASSIFIER_RUN_DIR"
RUN_DIR_ENV = "FANUC_CELL_RUN_DIR"
_CONSTRUCTION_ONLY_SOURCE = "CONSTRUCTION-ONLY: required CSI measurement/review is absent"
_PRELIMINARY_MOTION_SOURCE = (
    "PRELIMINARY discarded deterministic-plan parameters: approach=30, lift=10 and hover=5 mm from "
    "sim/assembly_lab/objects/planner.py; retreat=10 mm from sim/assembly_lab/eval/director.py; "
    "15 mm and 6 deg row bounds derived from GuardLimits at minimum dt=0.1 s; "
    "measured CSI safe transit Z=-180 from robot/cell/real_layout.py. "
    "CELL sends semantic GR00T intents, never these deterministic waypoints"
)
__all__ = (
    "DETECTOR_PRESENCE_TOLERANCE_MM",
    "PLACEMENT_POSE_UNCERTAINTY",
    "UNVERIFIED_PRESENCE_ENV",
    "DetectorPresenceGate",
    "PlacementMeasuredPoseBridge",
    "PlacementPoseEstimator",
    "PngJsonlImageSink",
    "PresenceState",
    "PresenceStatus",
    "SaturdayAdapterBlockedError",
    "catalog_installed_orientations",
    "conservative_closeup_thresholds",
    "conservative_corridor",
    "load_saturday_runtime_adapters",
)
_DETECTOR_CANDIDATES = (
    ROOT / "runs" / "t27" / "detector",
    Path.home() / "fanuc-t31-results" / "fanuc-t27-runs" / "main" / "full" / "detector",
    Path("/home/ubuntu/fanuc-t31-results/fanuc-t27-runs/main/full/detector"),
)
_CLASSIFIER_CANDIDATES = (
    ROOT / "runs" / "t27" / "classifier",
    Path.home() / "fanuc-t31-results" / "fanuc-t27-runs" / "main" / "full" / "classifier",
    Path("/home/ubuntu/fanuc-t31-results/fanuc-t27-runs/main/full/classifier"),
)


class _UnverifiedDetector:
    def __call__(self, image: np.ndarray) -> list:
        del image
        log.warning("UNVERIFIED detector presence used; returning no detections so CELL holds")
        return []


class _UnverifiedClassifier:
    def __call__(self, image: np.ndarray, bbox_px: tuple[float, float, float, float]) -> WristReading:
        del image, bbox_px
        log.warning("UNVERIFIED wrist classifier used; returning an unsure reading")
        return WristReading("unverified", 0.0, 0.5)


def _artifact_path(env_name: str, candidates: tuple[Path, ...]) -> Path | None:
    configured = os.environ.get(env_name, "").strip()
    paths = (Path(configured).expanduser(),) if configured else candidates
    return next((path for path in paths if (path / "model.safetensors").is_file()), None)


def _allow_unverified() -> bool:
    value = os.environ.get(UNVERIFIED_PRESENCE_ENV)
    if value not in (None, "", "1"):
        raise SaturdayAdapterBlockedError(
            "unverified_presence_flag",
            f"{UNVERIFIED_PRESENCE_ENV} must be exactly '1' when explicitly accepted",
        )
    return value == "1"


def _model_backends(allow_unverified: bool):
    detector_path = _artifact_path(DETECTOR_RUN_ENV, _DETECTOR_CANDIDATES)
    classifier_path = _artifact_path(CLASSIFIER_RUN_ENV, _CLASSIFIER_CANDIDATES)
    if detector_path is None:
        if not allow_unverified:
            raise SaturdayAdapterBlockedError(
                "detector_presence_unverified",
                f"no RT-DETR model.safetensors; set {UNVERIFIED_PRESENCE_ENV}=1 only with supervised approval",
            )
        state = PresenceState(PresenceStatus.UNVERIFIED, "explicit construction-only operator authority")
        return _UnverifiedDetector(), _UnverifiedClassifier(), state
    try:
        detector = RtDetrDetector(detector_path, min_score=1.0)
        labels = load_ontology(TAXONOMY_PATH).classifier_labels
        classifier = DefectHeadClassifier(classifier_path, labels, 224) if classifier_path else None
    except (ModelLoadError, OSError, ValueError, ImportError) as error:
        if not allow_unverified:
            raise SaturdayAdapterBlockedError("model_artifact", str(error)) from error
        log.warning("UNVERIFIED model artifact; construction remains fail-closed: %s", error)
        state = PresenceState(PresenceStatus.UNVERIFIED, str(error))
        return _UnverifiedDetector(), _UnverifiedClassifier(), state
    if classifier is None:
        if not allow_unverified:
            raise SaturdayAdapterBlockedError("classifier_unavailable", "no classifier model.safetensors")
        classifier = _UnverifiedClassifier()
    return detector, classifier, PresenceState(PresenceStatus.VERIFIED, str(detector_path))


def conservative_closeup_thresholds() -> CloseupThresholds:
    """Use the strictest valid probabilities; other construction-only gates still block."""
    return CloseupThresholds(1.0, 1.0, _CONSTRUCTION_ONLY_SOURCE)


def conservative_corridor() -> ArmCorridor:
    """Strictest representable sentinel; never presented as physical arm geometry."""
    return ArmCorridor(math.nextafter(0.0, math.inf), _CONSTRUCTION_ONLY_SOURCE)


class _UnverifiedFinalIdentityBackend:
    """Return explicit non-passing evidence without fabricating physical identity."""

    def __init__(self, sink: PngJsonlImageSink) -> None:
        self._sink = sink

    def capture(
        self,
        observation: dict[str, object],
        view: FinalViewPose,
        base_instance_id: str,
        installed_instance_ids: tuple[str, str, str, str],
    ) -> FinalViewIdentityEvidence:
        del base_instance_id
        self._sink.event(
            "final_identity_unverified",
            f"view={view.view_id}; no reviewed final-identity model; controller must hold before DONE",
        )
        return FinalViewIdentityEvidence(
            observation_time(observation),
            True,
            view.view_id,
            installed_instance_ids,
            view.expected_instance_roles,
            False,
            None,
        )


def _preliminary_discarded_motion_config() -> MotionConfig:
    """Supply finite compile inputs; CELL discards their deterministic waypoints."""
    return MotionConfig(
        30.0,
        10.0,
        5.0,
        10.0,
        15.0,
        6.0,
        1,
        INSPECTION_SAFE_Z_MM,
        _PRELIMINARY_MOTION_SOURCE,
    )


def _construction_only_verification_thresholds() -> VerificationThresholds:
    return VerificationThresholds(1.0, 1.0, 1.0, 0.0, 0.0)


def _wire_snapshot_bridge(
    perception: CellPerceptionPort,
    estimator: PlacementPoseEstimator,
    bridge: PlacementMeasuredPoseBridge,
    gate: DetectorPresenceGate,
    sink: PngJsonlImageSink,
) -> None:
    original_scan = perception.scan

    def scan(_self: CellPerceptionPort, observation: dict[str, object]) -> InventorySnapshot:
        estimator.activate(observation)
        detail = (
            f"{estimator.presence.status.value}: {estimator.presence.source}; "
            f"inventory={estimator.placement_source}"
        )
        sink.event("presence", detail)
        if estimator.presence.status is PresenceStatus.VERIFIED:
            snapshot = original_scan(observation)
            gate.verify(observation, snapshot)
        else:
            snapshot = estimator.reviewed_inventory_snapshot(observation)
            _self._xy.update(
                {
                    item.instance_id: item.coarse_xy_mm
                    for item in snapshot.items
                    if item.coarse_xy_mm is not None
                }
            )
            verifier = _self.expected_layout_verifier
            if verifier is None:
                raise ControllerPortError("unverified placement inventory requires the reviewed layout verifier")
            verifier(observation, snapshot)
            log.warning("UNVERIFIED operator-approved placement inventory is active: %s", detail)
        bridge.publish(snapshot)
        return snapshot

    perception.scan = MethodType(scan, perception)


def _validate_loader_inputs(
    expected_layout_verifier: ExpectedLayoutVerifier,
    final_view_registry: FinalViewRegistry,
) -> None:
    if type(expected_layout_verifier) is not ExpectedLayoutVerifier:
        raise SaturdayAdapterBlockedError("expected_layout", "exact ExpectedLayoutVerifier is required")
    if type(final_view_registry) is not FinalViewRegistry:
        raise SaturdayAdapterBlockedError("final_views", "exact FinalViewRegistry is required")


def load_saturday_runtime_adapters(
    catalog: Catalog,
    cad: CadCatalog,
    expected_layout_verifier: ExpectedLayoutVerifier,
    final_view_registry: FinalViewRegistry,
):
    """Build typed construction-only adapters; missing detector presence refuses by default."""
    _validate_loader_inputs(expected_layout_verifier, final_view_registry)
    detector, classifier, presence = _model_backends(_allow_unverified())
    geometry = CalibratedCellGeometry()
    layout = load_real_cell_layout()
    estimator = PlacementPoseEstimator(layout, cad, presence=presence)
    bridge = PlacementMeasuredPoseBridge(catalog, estimator)
    sink = PngJsonlImageSink(Path(os.environ.get(RUN_DIR_ENV, ROOT / "runs" / "saturday-cell")))
    gate = DetectorPresenceGate(catalog, layout, tolerance_mm=DETECTOR_PRESENCE_TOLERANCE_MM)
    perception = CellPerceptionPort(
        detector=detector,
        classifier=classifier,
        pose_estimator=estimator,
        geometry=geometry,
        thresholds=conservative_closeup_thresholds(),
        wrist_size=WRIST_150MM_PRELIMINARY,
        wrist_pose=PLACEMENT_POSE_UNCERTAINTY,
        table_pose=PLACEMENT_POSE_UNCERTAINTY,
        arm_corridor=conservative_corridor(),
        image_sink=sink,
        cad=cad,
        expected_layout_verifier=expected_layout_verifier,
    )
    _wire_snapshot_bridge(perception, estimator, bridge, gate, sink)
    if presence.status is PresenceStatus.UNVERIFIED:
        PlacementInspectionEvidence(cad, estimator, bridge, sink).bind(perception)
    from robot.cell.t09_real_runtime import SaturdayRuntimeAdapters

    return SaturdayRuntimeAdapters(
        perception,
        bridge,
        _preliminary_discarded_motion_config(),
        SaturdaySceneSafetySupervisor(),
        _construction_only_verification_thresholds(),
        _UnverifiedFinalIdentityBackend(sink),
    )
