"""Unverified placement evidence that allows fixed inspection moves only.
Authored by Shritej Gunda.
"""

from __future__ import annotations

from types import MethodType

import numpy as np

from assembly.controller_ports import (
    CollisionScene,
    ControllerPortError,
    current_waypoint,
    observation_time,
)
from assembly.inventory import InventoryItem, InventoryZone, ObjectCondition
from perception.port.cad import CadCatalog
from perception.port.frames import installed_pose
from perception.port.port import CellPerceptionPort
from perception.port.scene import box_from_pose
from robot.cell._saturday_runtime_support import (
    PlacementMeasuredPoseBridge,
    PlacementPoseEstimator,
    PngJsonlImageSink,
)


class PlacementInspectionEvidence:
    """Build a placement-pinned scene that cannot authorize learned part motion."""

    def __init__(
        self,
        cad: CadCatalog,
        estimator: PlacementPoseEstimator,
        bridge: PlacementMeasuredPoseBridge,
        sink: PngJsonlImageSink,
    ) -> None:
        self._cad = cad
        self._estimator = estimator
        self._bridge = bridge
        self._sink = sink

    def bind(self, perception: CellPerceptionPort) -> None:
        """Install an inspection-only port and an explicit learned-motion refusal."""

        def inspection_scene(
            _port: CellPerceptionPort,
            observation: dict[str, object],
            moving_instance_id: str,
            base_instance_id: str,
        ) -> CollisionScene:
            return self.scene(observation, moving_instance_id, base_instance_id)

        def collision_scene(
            _port: CellPerceptionPort,
            observation: dict[str, object],
            moving_instance_id: str,
            base_instance_id: str,
        ) -> CollisionScene:
            del observation, moving_instance_id, base_instance_id
            raise ControllerPortError(
                "learned part motion requires detector evidence; placement prior is inspection-only"
            )

        perception.inspection_scene = MethodType(inspection_scene, perception)  # type: ignore[attr-defined]
        perception.collision_scene = MethodType(collision_scene, perception)

    def scene(
        self,
        observation: dict[str, object],
        moving_instance_id: str,
        base_instance_id: str,
    ) -> CollisionScene:
        observed_at = observation_time(observation)
        rollout = self._active_rollout(observed_at)
        items = self._items_by_instance(rollout.index)
        self._require_item(items, moving_instance_id, "moving")
        base = self._require_item(items, base_instance_id, "base")
        if not base.object_id.startswith("F"):
            raise ControllerPortError("inspection reference must be a reviewed placement base")
        base_pose = self._bridge.base_for(base, observed_at_s=observed_at)
        fixtures, parts = self._boxes(rollout.index, moving_instance_id)
        self._sink.event(
            "inspection_scene_unverified",
            "placement.yaml scene; no live intrusion detector; supervised fixed inspection only",
        )
        return CollisionScene(
            observed_at,
            True,
            base_instance_id,
            (base_pose.tcp.x, base_pose.tcp.y, base_pose.tcp.z, base_pose.tcp.r),
            0.0,
            moving_instance_id,
            current_waypoint(observation).gripper == 1.0,
            False,
            fixtures,
            parts,
        )

    def _active_rollout(self, observed_at: float):
        rollout = self._estimator.active_rollout
        if rollout is None or self._estimator.observed_at_s != observed_at:
            raise ControllerPortError("inspection scene requires the current placement inventory snapshot")
        return rollout

    def _items_by_instance(self, rollout_index: int) -> dict[str, InventoryItem]:
        assert self._estimator.active_rollout is not None
        return {
            self._instance_id(rollout_index, placed.object_id): InventoryItem(
                self._instance_id(rollout_index, placed.object_id),
                placed.object_id,
                InventoryZone.TABLE,
                ObjectCondition.VERIFIED,
            )
            for placed in self._estimator.active_rollout.objects
        }

    @staticmethod
    def _require_item(
        items: dict[str, InventoryItem],
        instance_id: str,
        label: str,
    ) -> InventoryItem:
        try:
            return items[instance_id]
        except KeyError as error:
            raise ControllerPortError(f"{label} instance is absent from the reviewed placement") from error

    def _boxes(self, rollout_index: int, moving_instance_id: str):
        fixtures, parts = [], []
        assert self._estimator.active_rollout is not None
        for placed in self._estimator.active_rollout.objects:
            instance_id = self._instance_id(rollout_index, placed.object_id)
            cad = self._cad.get(placed.object_id)
            world_from_local = self._estimator.object_pose(placed.object_id)
            world_from_stl = world_from_local @ np.asarray(cad.local_from_stl, dtype=float)
            box = box_from_pose(instance_id, installed_pose(world_from_stl, cad), cad)
            if placed.is_base:
                fixtures.append(box)
            elif instance_id != moving_instance_id:
                parts.append(box)
        return tuple(fixtures), tuple(parts)

    @staticmethod
    def _instance_id(rollout_index: int, object_id: str) -> str:
        return f"placement-r{rollout_index}-{object_id}"
