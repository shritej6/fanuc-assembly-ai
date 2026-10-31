"""Real-cell recovery targets derived from reviewed zones plus pinned CAD/T29 geometry."""

from __future__ import annotations

import math
from collections.abc import Mapping

from assembly.catalog import Catalog, RecipeStep, UnknownProductError
from assembly.inventory import InventoryItem
from assembly.motion import AbsolutePoseUF0, OrientedPartPose, RestingState
from assembly.recovery import PullOutTarget, RecoveryError
from perception.cell.targets import CellTargets, Release, TargetError
from perception.port.cad import CadCatalog

FINGERS_DOWN_W_DEG = -180.0
NEUTRAL_WRIST_R_DEG = 180.0


class RealCellRecoveryTargets:
    """RecoveryTargetProvider with explicit empty-slot and calibration authority."""

    def __init__(
        self,
        *,
        catalog: Catalog,
        cad: CadCatalog,
        cell_targets: CellTargets,
        staging_slots: tuple[str, ...],
        pull_out_slots: Mapping[tuple[int, str], str],
        source: str,
    ) -> None:
        if not isinstance(source, str) or not source.strip():
            raise RecoveryError("real recovery targets require a calibration source")
        if not staging_slots or len(staging_slots) != len(set(staging_slots)):
            raise RecoveryError("real recovery targets need unique reviewed empty staging slots")
        self._catalog = catalog
        self._cad = cad
        self._targets = cell_targets
        self._staging_slots = tuple(staging_slots)
        self._pull_out_slots = dict(pull_out_slots)
        self._validate_slots()
        self.source = (
            f"{source}; sdg/configs/cell_layout.yaml; {cad.source}; "
            f"T29 {catalog.grasp_source.sha256}"
        )

    def pull_out_target_for(self, product: int, part: InventoryItem) -> PullOutTarget:
        step = self._step(product, part)
        key = (product, part.object_id)
        if key not in self._pull_out_slots:
            raise RecoveryError(
                f"{part.object_id} has no reviewed pull-out slot for product {product}"
            )
        release = self._release_for_slot(self._pull_out_slots[key], part.object_id)
        return PullOutTarget(self._pose(release, step), release.approach_z_mm, self.source)

    def staging_target_for(
        self,
        product: int,
        part: InventoryItem,
        slot: int,
    ) -> OrientedPartPose:
        step = self._step(product, part)
        if isinstance(slot, bool) or not isinstance(slot, int) or not 1 <= slot <= len(self._staging_slots):
            raise RecoveryError(f"recovery stage {slot!r} has no reviewed empty staging slot")
        release = self._release_for_slot(self._staging_slots[slot - 1], part.object_id)
        return self._pose(release, step)

    def staging_verification_target_for(
        self,
        product: int,
        part: InventoryItem,
        slot: int,
    ) -> OrientedPartPose:
        return self.staging_target_for(product, part, slot)

    def reject_target_for(self, part: InventoryItem) -> OrientedPartPose:
        step = self._step_for_object(part.object_id)
        try:
            release = self._targets.next_reject(self._held_below_tcp_mm(part.object_id))
        except TargetError as error:
            raise RecoveryError(f"real reject target is unavailable: {error}") from error
        return self._pose(release, step)

    def _validate_slots(self) -> None:
        known = set(self._targets.slots)
        configured = set(self._staging_slots) | set(self._pull_out_slots.values())
        unknown = sorted(configured - known)
        if unknown:
            raise RecoveryError(f"reviewed recovery slots are absent from cell_layout.yaml: {unknown}")
        for key in self._pull_out_slots:
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or isinstance(key[0], bool)
                or not isinstance(key[0], int)
                or not isinstance(key[1], str)
            ):
                raise RecoveryError("pull-out slot keys must be (product, object_id)")

    def _release_for_slot(self, slot_id: str, object_id: str) -> Release:
        try:
            return self._targets.slot(slot_id, self._held_below_tcp_mm(object_id))
        except TargetError as error:
            raise RecoveryError(f"real staging target is unavailable: {error}") from error

    def _held_below_tcp_mm(self, object_id: str) -> float:
        cad = self._cad.get(object_id)
        if cad.tcp_height_mm is None:
            raise RecoveryError(f"{object_id} has no pinned T29 TCP height")
        below = cad.tcp_height_mm - cad.bbox_min_mm[2]
        if not math.isfinite(below) or below < 0.0:
            raise RecoveryError(f"{object_id} has invalid pinned held-part geometry")
        return below

    def _step(self, product: int, part: InventoryItem) -> RecipeStep:
        try:
            recipe = self._catalog.product(product)
        except UnknownProductError as error:
            raise RecoveryError(f"unknown recovery product {product!r}") from error
        for step in recipe.steps:
            if step.object_id == part.object_id:
                return step
        raise RecoveryError(f"{part.object_id} does not belong to product {product}")

    def _step_for_object(self, object_id: str) -> RecipeStep:
        matches = tuple(
            step
            for product in self._catalog.products
            for step in product.steps
            if step.object_id == object_id
        )
        if len(matches) != 1:
            raise RecoveryError(f"{object_id} is not one exact movable catalog object")
        return matches[0]

    @staticmethod
    def _pose(release: Release, step: RecipeStep) -> OrientedPartPose:
        local_r = (NEUTRAL_WRIST_R_DEG - step.preferred_grasp.turn_deg) % 360.0
        tcp = AbsolutePoseUF0(
            release.x_mm,
            release.y_mm,
            release.z_mm,
            FINGERS_DOWN_W_DEG,
            0.0,
            local_r,
            "UF0",
            "calibrated",
        )
        return OrientedPartPose(tcp, RestingState.INSTALLED, "installed:+z")
