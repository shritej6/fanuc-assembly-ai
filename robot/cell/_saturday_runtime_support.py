"""Typed, non-authoritative support adapters for the cell loader.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from PIL import Image

from assembly.calibrated_poses import (
    ApprovedInstalledOrientation,
    CalibratedPoseError,
    FoundationPoseFrameBoundary,
    GraspFrameRegistry,
    MeasuredPoseRegistry,
    MeasuredRecipePoses,
)
from assembly.catalog import Catalog, ProductRecipe, RecipeStep, UnknownBaseError
from assembly.controller_ports import (
    ControllerPortError,
    InventorySnapshot,
    ObservedInventoryItem,
    observation_time,
)
from assembly.inventory import InventoryItem, InventoryZone, ObjectCondition
from assembly.motion import AbsolutePoseUF0, OrientedPartPose, RestingState
from perception.port.backends import PoseFit
from perception.port.cad import CadCatalog
from perception.port.geometry import CalibratedCellGeometry
from perception.port.placement import PoseUncertainty
from perception.port.size_gate import Status
from robot.cell.real_layout import FINGERS_DOWN_W_DEG, PlacedObject, RealCellLayout

PLACEMENT_POSE_UNCERTAINTY = PoseUncertainty(
    2.0,
    2.0,
    Status.PRELIMINARY,
    "PRELIMINARY supervised hand placement tolerance; placement.yaml@434f488, pending CSI measurement",
)
DETECTOR_PRESENCE_TOLERANCE_MM = 25.0


class SaturdayAdapterBlockedError(RuntimeError):
    """A Saturday adapter cannot be constructed without explicit authority."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


class PresenceStatus(StrEnum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


@dataclass(frozen=True, slots=True)
class PresenceState:
    status: PresenceStatus
    source: str


def _yaw_matrix(yaw_deg: float) -> np.ndarray:
    cosine, sine = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    matrix = np.eye(4)
    matrix[:3, :3] = ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    return matrix


def _installed_up_face(step: RecipeStep) -> str:
    """Name the raw CAD-local face that the catalog rotates toward base +Z."""
    base_z = step.installed_rotation_matrix[2]
    axis = max(range(3), key=lambda index: abs(base_z[index]))
    if abs(abs(base_z[axis]) - 1.0) > 1e-6:
        raise CalibratedPoseError(f"{step.object_id} installed up-face is not axis-aligned")
    sign = "+" if base_z[axis] > 0 else "-"
    return f"installed:{sign}{'xyz'[axis]}"


def catalog_installed_orientations(
    catalog: Catalog,
) -> dict[str, ApprovedInstalledOrientation]:
    """Derive each up-face from its pinned CAD/catalog installation matrix."""
    result: dict[str, ApprovedInstalledOrientation] = {}
    for product in catalog.products:
        for step in product.steps:
            orientation = ApprovedInstalledOrientation(
                FINGERS_DOWN_W_DEG,
                0.0,
                0.0,
                _installed_up_face(step),
            )
            previous = result.setdefault(step.object_id, orientation)
            if previous != orientation:
                raise CalibratedPoseError(f"{step.object_id} has conflicting installed orientations")
    return result


class PlacementPoseEstimator:
    """PoseEstimator backed by the current reviewed placement rollout."""

    uncertainty = PLACEMENT_POSE_UNCERTAINTY

    def __init__(
        self,
        layout: RealCellLayout,
        cad: CadCatalog,
        *,
        presence: PresenceState | None = None,
    ) -> None:
        self._layout = layout
        self._cad = cad
        self.table_surface_z_mm = CalibratedCellGeometry.table_surface_z_mm
        self.presence = presence or PresenceState(PresenceStatus.UNVERIFIED, "no detector artifact was supplied")
        self.active_rollout = None
        self.observed_at_s: float | None = None

    def activate(self, observation: dict[str, object]) -> None:
        self.active_rollout = self._layout.rollout_for_observation(observation)
        self.observed_at_s = observation_time(observation)

    @property
    def placement_source(self) -> str:
        """Return the immutable placement artifact used by the supervised prior."""
        return self._layout.source

    def reviewed_inventory_snapshot(
        self,
        observation: dict[str, object],
    ) -> InventorySnapshot:
        """Publish operator-approved placement inventory under the explicit unverified flag."""
        self.activate(observation)
        assert self.active_rollout is not None
        items = tuple(
            ObservedInventoryItem(
                f"placement-r{self.active_rollout.index}-{placed.object_id}",
                placed.object_id,
                InventoryZone.TABLE,
                ObjectCondition.VERIFIED,
                (placed.center_x_mm, placed.center_y_mm),
            )
            for placed in self.active_rollout.objects
        )
        return InventorySnapshot(self.observed_at_s, True, items)  # type: ignore[arg-type]

    def object_pose(
        self,
        object_id: str,
        *,
        placement_object_id: str | None = None,
    ) -> np.ndarray:
        if self.active_rollout is None:
            raise CalibratedPoseError("placement pose estimator has no active reviewed trial")
        slot_object_id = placement_object_id or object_id
        matches = [item for item in self.active_rollout.objects if item.object_id == slot_object_id]
        if len(matches) != 1:
            raise CalibratedPoseError(f"reviewed trial needs one placement for {slot_object_id!r}")
        placed = matches[0]
        world_from_local = _yaw_matrix(placed.yaw_deg)
        world_from_local[:3, 3] = (
            placed.center_x_mm,
            placed.center_y_mm,
            self.table_surface_z_mm - self._cad.get(object_id).bbox_min_mm[2],
        )
        return world_from_local

    def __call__(
        self,
        object_id: str,
        camera: str,
        observation: dict[str, object],
        mask: np.ndarray,
    ) -> PoseFit:
        del camera, mask
        self.activate(observation)
        local_from_stl = np.asarray(self._cad.get(object_id).local_from_stl, dtype=float)
        return PoseFit(self.object_pose(object_id) @ local_from_stl, 0.0)


class DetectorPresenceGate:
    """Require every reviewed table object and refuse all extra table detections."""

    def __init__(self, catalog: Catalog, layout: RealCellLayout, *, tolerance_mm: float) -> None:
        if tolerance_mm != DETECTOR_PRESENCE_TOLERANCE_MM:
            raise SaturdayAdapterBlockedError(
                "presence_tolerance",
                f"detector presence box must remain {DETECTOR_PRESENCE_TOLERANCE_MM:.1f} mm per axis",
            )
        self._catalog = catalog
        self._layout = layout
        self._tolerance_mm = tolerance_mm

    def verify(self, observation: dict[str, object], snapshot: InventorySnapshot) -> None:
        if not isinstance(snapshot, InventorySnapshot) or not snapshot.complete:
            raise ControllerPortError("detector presence verification needs a complete snapshot")
        rollout = self._layout.rollout_for_observation(observation)
        candidates = [item for item in snapshot.items if item.zone is InventoryZone.TABLE]
        for expected in rollout.objects:
            candidates.remove(self._match(expected, candidates))
        if candidates:
            unexpected = ", ".join(sorted(item.object_id for item in candidates))
            raise ControllerPortError(f"unexpected table detection {unexpected}")

    def _match(self, expected: PlacedObject, candidates: list):
        compatible = [item for item in candidates if self._compatible(expected, item.object_id)]
        if not compatible:
            raise ControllerPortError(f"detector presence is missing {expected.object_id}")
        inside = [item for item in compatible if self._inside(item.coarse_xy_mm, expected)]
        if not inside:
            raise ControllerPortError(
                f"{expected.object_id} is outside the reviewed {self._tolerance_mm:.1f} mm presence box"
            )
        return min(inside, key=lambda item: self._distance(item.coarse_xy_mm, expected))

    def _compatible(self, expected: PlacedObject, object_id: str) -> bool:
        if not expected.is_base:
            return object_id == expected.object_id
        try:
            return self._catalog.base_type_for_object(object_id) == self._catalog.base_type_for_object(
                expected.object_id
            )
        except UnknownBaseError:
            return False

    def _inside(self, xy: tuple[float, float] | None, expected: PlacedObject) -> bool:
        return xy is not None and all(
            abs(actual - target) <= self._tolerance_mm
            for actual, target in zip(xy, (expected.center_x_mm, expected.center_y_mm), strict=True)
        )

    @staticmethod
    def _distance(xy: tuple[float, float] | None, expected: PlacedObject) -> float:
        if xy is None:
            return math.inf
        return math.hypot(xy[0] - expected.center_x_mm, xy[1] - expected.center_y_mm)


class PngJsonlImageSink:
    """Persist each inspection image and its immutable local metadata record."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.images_dir = self.run_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._sequence = 0

    @staticmethod
    def _slug(name: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip()).strip("-")
        return slug or "image"

    @staticmethod
    def _append(path: Path, payload: dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    def __call__(self, name: str, image: np.ndarray) -> str:
        pixels = np.asarray(image)
        if pixels.dtype != np.uint8 or pixels.ndim not in (2, 3):
            raise SaturdayAdapterBlockedError("image_sink_input", "images must be uint8 HxW or HxWxC arrays")
        self._sequence += 1
        path = self.images_dir / f"{self._sequence:06d}-{self._slug(name)}.png"
        Image.fromarray(pixels).save(path, format="PNG")
        self._append(self.run_dir / "images.jsonl", self._record(name, path, pixels))
        return path.resolve().as_uri()

    def _record(self, name: str, path: Path, pixels: np.ndarray) -> dict[str, object]:
        return {
            "dtype": str(pixels.dtype),
            "name": name,
            "path": str(path.relative_to(self.run_dir)),
            "sequence": self._sequence,
            "shape": list(pixels.shape),
        }

    def event(self, kind: str, detail: str) -> None:
        self._append(self.run_dir / "events.jsonl", {"detail": detail, "kind": kind})


class PlacementMeasuredPoseBridge:
    """Publish placement-derived poses through the real MeasuredRecipePoses API."""

    def __init__(self, catalog: Catalog, estimator: PlacementPoseEstimator) -> None:
        self._catalog = catalog
        self._estimator = estimator
        self._frames = GraspFrameRegistry.from_catalog(catalog)
        self._orientations = catalog_installed_orientations(catalog)
        self._placement_by_instance: dict[str, str] = {}
        self._registry = MeasuredPoseRegistry({}, observed_at_s=0.0, complete=False)
        self._provider = MeasuredRecipePoses(
            catalog=catalog,
            measurements=self._registry,
            installed_orientations=self._orientations,
            orientation_tolerance_deg=0.0,
            foundationpose_boundary=FoundationPoseFrameBoundary(
                "installed_local",
                "placement.yaml@434f488 transformed by pinned T29 local-from-STL frames",
            ),
            grasp_frames=self._frames,
        )

    def _placement_object_id(self, item: InventoryItem) -> str:
        matched = self._placement_by_instance.get(item.instance_id)
        if matched is not None:
            return matched
        try:
            base_type = self._catalog.base_type_for_object(item.object_id)
        except UnknownBaseError:
            return item.object_id
        rollout = self._estimator.active_rollout
        if rollout is None:
            raise CalibratedPoseError("placement pose estimator has no active reviewed trial")
        matches = [
            placed.object_id
            for placed in rollout.objects
            if placed.is_base
            and self._catalog.base_type_for_object(placed.object_id) == base_type
        ]
        if len(matches) != 1:
            raise CalibratedPoseError(
                f"reviewed trial needs one interchangeable {base_type!r} base placement"
            )
        return matches[0]

    def _pose(self, item: InventoryItem) -> OrientedPartPose:
        matrix = self._estimator.object_pose(
            item.object_id,
            placement_object_id=self._placement_object_id(item),
        )
        x, y, z = matrix[:3, 3]
        yaw = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
        approved = self._orientations.get(item.object_id)
        if approved is None:
            try:
                self._catalog.base_type_for_object(item.object_id)
            except UnknownBaseError as error:
                raise CalibratedPoseError(
                    f"{item.object_id!r} has no catalog-derived installed orientation"
                ) from error
            approved = ApprovedInstalledOrientation(
                FINGERS_DOWN_W_DEG,
                0.0,
                0.0,
                "placement-base:+z",
            )
        else:
            local = self._frames.frame_for(item.object_id).preferred_grasp.tcp_local_mm
            x, y, z = matrix[:3, :3] @ np.asarray(local) + matrix[:3, 3]
        tcp = AbsolutePoseUF0(float(x), float(y), float(z), approved.w_deg, approved.p_deg, yaw, "UF0", "calibrated")
        return OrientedPartPose(tcp, RestingState.INSTALLED, approved.up_face)

    def _publish_items(self, items: tuple[InventoryItem, ...], observed_at_s: float) -> None:
        if self._estimator.observed_at_s != observed_at_s:
            raise CalibratedPoseError("placement pose snapshot does not belong to the current observation")
        self._registry.replace_snapshot(
            {item.instance_id: self._pose(item) for item in items},
            observed_at_s=observed_at_s,
            complete=True,
        )

    def publish(self, snapshot: InventorySnapshot) -> None:
        self._placement_by_instance = {
            item.instance_id: self._match_placement(item)
            for item in snapshot.items
        }
        items = tuple(
            InventoryItem(item.instance_id, item.object_id, item.zone, item.condition)
            for item in snapshot.items
        )
        self._publish_items(items, snapshot.observed_at_s)

    def _match_placement(self, item) -> str:
        rollout = self._estimator.active_rollout
        if rollout is None:
            raise CalibratedPoseError("placement pose estimator has no active reviewed trial")
        try:
            base_type = self._catalog.base_type_for_object(item.object_id)
        except UnknownBaseError:
            return item.object_id
        candidates = [
            placed
            for placed in rollout.objects
            if placed.is_base
            and self._catalog.base_type_for_object(placed.object_id) == base_type
        ]
        if item.coarse_xy_mm is None or not candidates:
            raise CalibratedPoseError(f"{item.object_id!r} needs a measured interchangeable base slot")
        return min(
            candidates,
            key=lambda placed: math.hypot(
                item.coarse_xy_mm[0] - placed.center_x_mm,
                item.coarse_xy_mm[1] - placed.center_y_mm,
            ),
        ).object_id

    def grasp_for(self, part: InventoryItem, *, observed_at_s: float) -> OrientedPartPose:
        self._publish_items((part,), observed_at_s)
        return self._provider.grasp_for(part, observed_at_s=observed_at_s)

    def base_for(self, base: InventoryItem, *, observed_at_s: float) -> OrientedPartPose:
        self._publish_items((base,), observed_at_s)
        return self._provider.base_for(base, observed_at_s=observed_at_s)

    def target_for(
        self,
        product: ProductRecipe,
        base: InventoryItem,
        step: RecipeStep,
        *,
        observed_at_s: float,
    ) -> OrientedPartPose:
        self._publish_items((base,), observed_at_s)
        return self._provider.target_for(product, base, step, observed_at_s=observed_at_s)

    def verification_target_for(
        self,
        product: ProductRecipe,
        base: InventoryItem,
        step: RecipeStep,
        *,
        observed_at_s: float,
    ) -> OrientedPartPose:
        self._publish_items((base,), observed_at_s)
        return self._provider.verification_target_for(product, base, step, observed_at_s=observed_at_s)

    def grasp_from_foundationpose(self, *args: object, **kwargs: object) -> OrientedPartPose:
        del args, kwargs
        raise CalibratedPoseError("placement prior cannot authorize a fallen-part 6D grasp")
