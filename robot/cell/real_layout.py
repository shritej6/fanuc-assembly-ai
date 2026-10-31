"""Reviewed CSI placement evidence and fixed camera poses for the real cell.

The placement artifact describes where an operator should stage objects.  It is
used only to reject a mismatched starting layout; detected poses remain the
sole source for grasp and insertion commands.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from assembly.catalog import Catalog, load_catalog
from assembly.controller_ports import ControllerPortError, InventorySnapshot
from assembly.final_views import FinalViewError, FinalViewPose, FinalViewRegistry
from assembly.fixed_inspection import InspectionPoseRegistry, InspectionSlotPose
from assembly.inventory import InventoryZone
from perception.cell.targets import MAX_HELD_BELOW_TCP_MM
from perception.port.cad import CadCatalog, load_cad
from perception.port.placement import precision_gate_tolerances

DEFAULT_PLACEMENT_PATH = Path(__file__).with_name("submission") / "placement.yaml"
CELL_LAYOUT_PATH = Path(__file__).resolve().parents[2] / "sdg" / "configs" / "cell_layout.yaml"
PLACEMENT_SOURCE_COMMIT = "1f97ff5"
START_POSITION_TOLERANCE_MM = 10.0
INSPECTION_Z_MM = -168.0
INSPECTION_SAFE_Z_MM = -180.0
FINGERS_DOWN_W_DEG = -180.0
NEUTRAL_WRIST_R_DEG = 180.0
_OBJECT_ID = re.compile(r"^(F\d{2}|M\d_\d{2})(?:\s|$)")
_SLOT_X = (("PN", 255.0), ("PF", 335.0))
_SLOT_Y = (-225.0, -135.0, -45.0, 45.0, 135.0, 225.0)
_SLOT_HALF_MM = 25.0
_INSPECTION_SOURCE = (
    "T38 placement commit 1f97ff5 slot XY and neutral R=180; "
    "PR9 measured CSI wrist inspection stop Z=-168, W=-180, P=0 and safe transit Z=-180"
)
_FINAL_VIEW_SOURCE = (
    "PRELIMINARY T38 placement commit 1f97ff5 base XY/yaw + "
    "sdg/configs/cell_layout.yaml clearance.base_gap_mm; "
    "PR9 measured CSI wrist inspection Z=-168, W=-180, P=0"
)
_BASE_SPOT_BY_Y = {-165.0: "B1", 0.0: "B2", 165.0: "B3"}


@dataclass(frozen=True, slots=True)
class PlacedObject:
    object_id: str
    center_x_mm: float
    center_y_mm: float
    yaw_deg: float

    @property
    def is_base(self) -> bool:
        return self.object_id.startswith("F")


@dataclass(frozen=True, slots=True)
class PlacementRollout:
    index: int
    prompt: str
    objects: tuple[PlacedObject, ...]

    @property
    def bases(self) -> tuple[PlacedObject, ...]:
        return tuple(item for item in self.objects if item.is_base)

    @property
    def parts(self) -> tuple[PlacedObject, ...]:
        return tuple(item for item in self.objects if not item.is_base)


@dataclass(frozen=True, slots=True)
class RealCellLayout:
    rollouts: tuple[PlacementRollout, ...]
    source: str

    def rollout_for_observation(self, observation: Mapping[str, object]) -> PlacementRollout:
        trial = observation.get("trial")
        if not isinstance(trial, Mapping):
            raise ControllerPortError("real-cell observation requires trial metadata")
        index = trial.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ControllerPortError("real-cell trial index must be a zero-based integer")
        if not 0 <= index < len(self.rollouts):
            raise ControllerPortError(f"real-cell trial index {index!r} has no reviewed rollout")
        rollout = self.rollouts[index]
        if trial.get("instruction") != rollout.prompt:
            raise ControllerPortError(f"trial {index} prompt does not match reviewed placement")
        return rollout

    def verify_expected_snapshot(
        self,
        catalog: Catalog,
        rollout: PlacementRollout,
        snapshot: InventorySnapshot,
    ) -> None:
        if not isinstance(snapshot, InventorySnapshot) or not snapshot.complete:
            raise ControllerPortError("expected-layout verification needs a complete inventory snapshot")
        candidates = [item for item in snapshot.items if item.zone is InventoryZone.TABLE]
        for expected in rollout.parts:
            match = _take_exact_candidate(candidates, expected)
            _require_position(match.coarse_xy_mm, expected)
        for expected in rollout.bases:
            match = _take_compatible_base(catalog, candidates, expected)
            _require_position(match.coarse_xy_mm, expected)


class ExpectedLayoutVerifier:
    """Verify each reviewed trial once, without becoming motion authority."""

    def __init__(self, layout: RealCellLayout, catalog: Catalog) -> None:
        self._layout = layout
        self._catalog = catalog
        self._verified_trials: set[int] = set()

    def __call__(self, observation: Mapping[str, object], snapshot: InventorySnapshot) -> None:
        rollout = self._layout.rollout_for_observation(observation)
        if rollout.index in self._verified_trials:
            return
        self._layout.verify_expected_snapshot(self._catalog, rollout, snapshot)
        self._verified_trials.add(rollout.index)

    def reset(self) -> None:
        """Require the reviewed start layout to be checked again for the next trial."""
        self._verified_trials.clear()


def load_real_cell_layout(path: Path = DEFAULT_PLACEMENT_PATH) -> RealCellLayout:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ControllerPortError(f"cannot load reviewed placement {path}: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"rollouts"}:
        raise ControllerPortError("reviewed placement must contain only rollouts")
    raw_rollouts = payload["rollouts"]
    if not isinstance(raw_rollouts, list) or len(raw_rollouts) != 3:
        raise ControllerPortError("reviewed placement must contain exactly three rollouts")
    catalog = load_catalog()
    rollouts = tuple(_parse_rollout(index, raw, catalog) for index, raw in enumerate(raw_rollouts))
    return RealCellLayout(rollouts, f"robot/cell/submission/placement.yaml@{PLACEMENT_SOURCE_COMMIT}")


def _parse_rollout(index: int, raw: object, catalog: Catalog) -> PlacementRollout:
    if not isinstance(raw, dict) or set(raw) != {"objects", "prompt"}:
        raise ControllerPortError(f"placement rollout {index} has unknown or missing keys")
    if not isinstance(raw["prompt"], str) or not raw["prompt"].strip():
        raise ControllerPortError(f"placement rollout {index} requires a prompt")
    objects = raw["objects"]
    if not isinstance(objects, dict) or len(objects) != 15:
        raise ControllerPortError(f"placement rollout {index} must contain exactly 15 objects")
    known = _known_object_ids(catalog)
    parsed = tuple(_parse_object(label, pose, known) for label, pose in objects.items())
    identifiers = tuple(item.object_id for item in parsed)
    if len(identifiers) != len(set(identifiers)):
        raise ControllerPortError(f"placement rollout {index} contains duplicate object IDs")
    rollout = PlacementRollout(index, raw["prompt"], parsed)
    if len(rollout.bases) != 3 or len(rollout.parts) != 12:
        raise ControllerPortError(f"placement rollout {index} needs three bases and twelve parts")
    return rollout


def _known_object_ids(catalog: Catalog) -> set[str]:
    bases = {object_id for object_id, _base_type in catalog._base_types_by_object}
    parts = {step.object_id for product in catalog.products for step in product.steps}
    return bases | parts


def _parse_object(label: object, pose: object, known: set[str]) -> PlacedObject:
    if not isinstance(label, str) or (match := _OBJECT_ID.match(label)) is None:
        raise ControllerPortError(f"placement object key {label!r} has no valid object ID")
    object_id = match.group(1)
    if object_id not in known:
        raise ControllerPortError(f"placement object {object_id!r} is not in the catalog")
    if not isinstance(pose, list) or len(pose) != 3:
        raise ControllerPortError(f"placement pose for {object_id} must be [x, y, yaw]")
    numbers = tuple(_finite(value, f"placement {object_id}") for value in pose)
    return PlacedObject(object_id, *numbers)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControllerPortError(f"{label} values must be finite numbers")
    number = float(value)
    if not math.isfinite(number):
        raise ControllerPortError(f"{label} values must be finite numbers")
    return number


def _take_exact_candidate(candidates: list, expected: PlacedObject):
    matches = [item for item in candidates if item.object_id == expected.object_id]
    if not matches:
        raise ControllerPortError(f"reviewed placement is missing {expected.object_id}")
    match = min(matches, key=lambda item: _distance(item.coarse_xy_mm, expected))
    candidates.remove(match)
    return match


def _take_compatible_base(catalog: Catalog, candidates: list, expected: PlacedObject):
    expected_type = catalog.base_type_for_object(expected.object_id)
    matches = [
        item
        for item in candidates
        if item.object_id.startswith("F")
        and catalog.base_type_for_object(item.object_id) == expected_type
    ]
    if not matches:
        raise ControllerPortError(
            f"reviewed placement is missing a base compatible with {expected.object_id}"
        )
    match = min(matches, key=lambda item: _distance(item.coarse_xy_mm, expected))
    candidates.remove(match)
    return match


def _distance(coarse_xy_mm: tuple[float, float] | None, expected: PlacedObject) -> float:
    if coarse_xy_mm is None:
        return math.inf
    return math.hypot(coarse_xy_mm[0] - expected.center_x_mm, coarse_xy_mm[1] - expected.center_y_mm)


def _require_position(coarse_xy_mm: tuple[float, float] | None, expected: PlacedObject) -> None:
    error = _distance(coarse_xy_mm, expected)
    if error > START_POSITION_TOLERANCE_MM:
        raise ControllerPortError(
            f"{expected.object_id} is outside the reviewed {START_POSITION_TOLERANCE_MM:.1f} mm "
            f"start tolerance ({error:.1f} mm)"
        )


def reviewed_inspection_registry() -> InspectionPoseRegistry:
    slots = tuple(
        _inspection_slot(prefix, x_mm, index, y_mm)
        for prefix, x_mm in _SLOT_X
        for index, y_mm in enumerate(_SLOT_Y, start=1)
    )
    return InspectionPoseRegistry(slots)


def reviewed_final_view_registry() -> FinalViewRegistry:
    """Derive two straight-down views for every reviewed placement base spot."""
    layout = load_real_cell_layout()
    catalog = load_catalog()
    cad = load_cad()
    baseline_mm, table_z_mm = _final_view_layout_values()
    unique = {
        (int(base.object_id[1:]), base.object_id, base.center_x_mm, base.center_y_mm, base.yaw_deg)
        for rollout in layout.rollouts
        for base in rollout.bases
    }
    views = tuple(
        view
        for product, object_id, x_mm, y_mm, yaw_deg in sorted(unique)
        for view in _final_views_for_base(
            product,
            object_id,
            x_mm,
            y_mm,
            yaw_deg,
            tuple(step.object_id for step in catalog.product(product).steps),
            baseline_mm,
        )
    )
    for view in views:
        _require_final_view_clearance(view, cad, table_z_mm)
    return FinalViewRegistry(views)


def _final_views_for_base(
    product: int,
    object_id: str,
    x_mm: float,
    y_mm: float,
    yaw_deg: float,
    roles: tuple[str, str, str, str],
    baseline_mm: float,
) -> tuple[FinalViewPose, FinalViewPose]:
    spot = _BASE_SPOT_BY_Y.get(y_mm, f"XY-{x_mm:g}-{y_mm:g}")
    view_y = (y_mm - baseline_mm, y_mm + baseline_mm)
    return tuple(
        FinalViewPose(
            product,
            object_id,
            f"{spot}-view-{index}",
            (x_mm, view_y_mm, INSPECTION_Z_MM, FINGERS_DOWN_W_DEG, 0.0, yaw_deg % 360.0),
            INSPECTION_SAFE_Z_MM,
            roles,
            _FINAL_VIEW_SOURCE,
            (x_mm, y_mm),
        )
        for index, view_y_mm in enumerate(view_y, start=1)
    )  # type: ignore[return-value]


def _final_view_layout_values() -> tuple[float, float]:
    try:
        payload = yaml.safe_load(CELL_LAYOUT_PATH.read_text(encoding="utf-8"))
        baseline_mm = float(payload["clearance"]["base_gap_mm"])
        table_z_mm = float(payload["table_z"])
    except (OSError, UnicodeError, yaml.YAMLError, KeyError, TypeError, ValueError) as error:
        raise ControllerPortError(f"cannot load final-view clearance from {CELL_LAYOUT_PATH}: {error}") from error
    if not math.isfinite(baseline_mm) or baseline_mm <= 0.0 or not math.isfinite(table_z_mm):
        raise ControllerPortError("final-view base gap and table Z must be finite, with a positive base gap")
    return baseline_mm, table_z_mm


def _require_final_view_clearance(
    view: FinalViewPose,
    cad: CadCatalog,
    table_z_mm: float,
) -> None:
    object_ids = (view.base_object_id, *view.expected_instance_roles)
    stack_top_z = table_z_mm + max(cad.get(object_id).seated_top_z_mm for object_id in object_ids)
    execution, _perception = precision_gate_tolerances()
    required_mm = MAX_HELD_BELOW_TCP_MM + execution.max_position_error_mm
    lowest_tcp_z = min(view.safe_z_mm, view.endpoint_tcp_uf0[2])
    if lowest_tcp_z - stack_top_z < required_mm:
        raise FinalViewError(
            f"{view.view_id} lacks assembled-product clearance: "
            f"{lowest_tcp_z - stack_top_z:.1f} mm < {required_mm:.1f} mm"
        )


def _inspection_slot(prefix: str, x_mm: float, index: int, y_mm: float) -> InspectionSlotPose:
    return InspectionSlotPose(
        slot_id=f"{prefix}{index}",
        selection_bounds_xy_mm=(
            x_mm - _SLOT_HALF_MM,
            x_mm + _SLOT_HALF_MM,
            y_mm - _SLOT_HALF_MM,
            y_mm + _SLOT_HALF_MM,
        ),
        endpoint_tcp=(
            x_mm,
            y_mm,
            INSPECTION_Z_MM,
            FINGERS_DOWN_W_DEG,
            0.0,
            NEUTRAL_WRIST_R_DEG,
        ),
        safe_z_mm=INSPECTION_SAFE_Z_MM,
        source=_INSPECTION_SOURCE,
    )
