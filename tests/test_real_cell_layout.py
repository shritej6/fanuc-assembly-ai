"""The real-cell layout is expected-state evidence, never grasp authority."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from assembly.catalog import load_catalog
from assembly.controller_ports import (
    ControllerPortError,
    InventorySnapshot,
    ObservedInventoryItem,
)
from assembly.inventory import InventoryZone
from robot.cell.real_layout import (
    PLACEMENT_SOURCE_COMMIT,
    RealCellLayout,
    load_real_cell_layout,
    reviewed_inspection_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def _observation(index: int, instruction: str) -> dict[str, object]:
    return {"trial": {"index": index, "instruction": instruction}, "t": 1.0}


def _snapshot(layout: RealCellLayout, rollout_index: int) -> InventorySnapshot:
    rollout = layout.rollouts[rollout_index]
    return InventorySnapshot(
        observed_at_s=1.0,
        complete=True,
        items=tuple(
            ObservedInventoryItem(
                f"seen-{placed.object_id}",
                placed.object_id,
                InventoryZone.TABLE,
                coarse_xy_mm=(placed.center_x_mm, placed.center_y_mm),
            )
            for placed in rollout.objects
        ),
    )


def test_layout_selects_zero_based_rollout_only_for_exact_prompt() -> None:
    layout = load_real_cell_layout(ROOT / "robot/cell/submission/placement.yaml")
    prompt = layout.rollouts[1].prompt

    selected = layout.rollout_for_observation(_observation(1, prompt))

    assert selected is layout.rollouts[1]
    assert len(selected.parts) == 12
    assert len(selected.bases) == 3
    assert PLACEMENT_SOURCE_COMMIT.startswith("57eff21")
    with pytest.raises(ControllerPortError, match="prompt does not match"):
        layout.rollout_for_observation(_observation(1, prompt + " changed"))


def test_expected_layout_accepts_same_shape_base_but_not_missing_part() -> None:
    layout = load_real_cell_layout(ROOT / "robot/cell/submission/placement.yaml")
    catalog = load_catalog()
    rollout = layout.rollouts[0]
    snapshot = _snapshot(layout, 0)
    items = list(snapshot.items)
    base_index = next(index for index, item in enumerate(items) if item.object_id == "F01")
    items[base_index] = ObservedInventoryItem(
        "seen-compatible-base",
        "F02",
        InventoryZone.TABLE,
        coarse_xy_mm=items[base_index].coarse_xy_mm,
    )

    layout.verify_expected_snapshot(catalog, rollout, snapshot.__class__(1.0, True, tuple(items)))

    without_part = tuple(item for item in snapshot.items if item.object_id != "M1_04")
    with pytest.raises(ControllerPortError, match="M1_04"):
        layout.verify_expected_snapshot(catalog, rollout, InventorySnapshot(1.0, True, without_part))


def test_expected_layout_uses_t38_start_tolerance_not_execution_tolerance() -> None:
    layout = load_real_cell_layout(ROOT / "robot/cell/submission/placement.yaml")
    catalog = load_catalog()
    rollout = layout.rollouts[2]
    items = list(_snapshot(layout, 2).items)
    target = next(index for index, item in enumerate(items) if item.object_id == "M3_02")
    x, y = items[target].coarse_xy_mm or (0.0, 0.0)
    items[target] = ObservedInventoryItem(
        items[target].instance_id,
        items[target].object_id,
        InventoryZone.TABLE,
        coarse_xy_mm=(x + 10.1, y),
    )

    with pytest.raises(ControllerPortError, match="outside the reviewed 10.0 mm"):
        layout.verify_expected_snapshot(catalog, rollout, InventorySnapshot(1.0, True, tuple(items)))


def test_reviewed_inspection_registry_has_exact_twelve_t38_slots() -> None:
    registry = reviewed_inspection_registry()

    assert len(registry.slots) == 12
    assert registry.resolve((255.0, -225.0)).slot_id == "PN1"
    assert registry.resolve((335.0, 225.0)).slot_id == "PF6"
    assert all(slot.endpoint_tcp[2:] == (-168.0, -180.0, 0.0, 180.0) for slot in registry.slots)
    assert all(slot.safe_z_mm == -180.0 for slot in registry.slots)
    assert all("T38" in slot.source and "PR9" in slot.source for slot in registry.slots)


def test_placement_parser_rejects_unknown_object_keys(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        (ROOT / "robot/cell/submission/placement.yaml").read_text(encoding="utf-8")
    )
    first = payload["rollouts"][0]["objects"]
    first["UNKNOWN decorative key"] = first.pop(next(iter(first)))
    path = tmp_path / "placement.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ControllerPortError, match="UNKNOWN"):
        load_real_cell_layout(path)
