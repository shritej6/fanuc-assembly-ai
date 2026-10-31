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
    ExpectedLayoutVerifier,
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
    assert len(selected.parts) == 4
    assert len(selected.bases) == 1
    assert PLACEMENT_SOURCE_COMMIT == "434f488"
    assert tuple(item.object_id for item in selected.bases) == ("F05",)
    assert selected.prompt.startswith("Assemble product 5")
    with pytest.raises(ControllerPortError, match="prompt does not match"):
        layout.rollout_for_observation(_observation(1, prompt + " changed"))


def test_expected_layout_accepts_same_shape_base_but_not_missing_part() -> None:
    layout = load_real_cell_layout(ROOT / "robot/cell/submission/placement.yaml")
    catalog = load_catalog()
    rollout = layout.rollouts[0]
    snapshot = _snapshot(layout, 0)
    items = list(snapshot.items)
    base_index = next(index for index, item in enumerate(items) if item.object_id == "F05")
    items[base_index] = ObservedInventoryItem(
        "seen-compatible-base",
        "F04",
        InventoryZone.TABLE,
        coarse_xy_mm=items[base_index].coarse_xy_mm,
    )

    layout.verify_expected_snapshot(catalog, rollout, snapshot.__class__(1.0, True, tuple(items)))

    without_part = tuple(item for item in snapshot.items if item.object_id != "M5_04")
    with pytest.raises(ControllerPortError, match="M5_04"):
        layout.verify_expected_snapshot(catalog, rollout, InventorySnapshot(1.0, True, without_part))


def test_expected_layout_uses_t38_start_tolerance_not_execution_tolerance() -> None:
    layout = load_real_cell_layout(ROOT / "robot/cell/submission/placement.yaml")
    catalog = load_catalog()
    rollout = layout.rollouts[2]
    items = list(_snapshot(layout, 2).items)
    target = next(index for index, item in enumerate(items) if item.object_id == "M5_02")
    x, y = items[target].coarse_xy_mm or (0.0, 0.0)
    items[target] = ObservedInventoryItem(
        items[target].instance_id,
        items[target].object_id,
        InventoryZone.TABLE,
        coarse_xy_mm=(x + 10.1, y),
    )

    with pytest.raises(ControllerPortError, match="outside the reviewed 10.0 mm"):
        layout.verify_expected_snapshot(catalog, rollout, InventorySnapshot(1.0, True, tuple(items)))


def test_expected_layout_reset_rechecks_the_same_trial_index() -> None:
    layout = load_real_cell_layout(ROOT / "robot/cell/submission/placement.yaml")
    verifier = ExpectedLayoutVerifier(layout, load_catalog())
    rollout = layout.rollouts[1]
    observation = _observation(1, rollout.prompt)
    snapshot = _snapshot(layout, 1)

    verifier(observation, snapshot)
    verifier(observation, InventorySnapshot(1.0, True, ()))
    verifier.reset()

    with pytest.raises(ControllerPortError, match="missing"):
        verifier(observation, InventorySnapshot(1.0, True, ()))


def test_reviewed_inspection_registry_has_exact_four_reviewed_product_five_slots() -> None:
    registry = reviewed_inspection_registry()

    assert len(registry.slots) == 4
    assert registry.resolve((358.0, 193.0)).slot_id == "P1"
    assert registry.resolve((383.0, 12.0)).slot_id == "P4"
    assert tuple(slot.selection_bounds_xy_mm for slot in registry.slots) == (
        (333.0, 383.0, 168.0, 218.0),
        (280.0, 330.0, 11.0, 61.0),
        (277.0, 327.0, 90.0, 140.0),
        (358.0, 408.0, -13.0, 37.0),
    )
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


def test_placement_parser_rejects_a_mixed_five_object_rollout(tmp_path: Path) -> None:
    payload = yaml.safe_load(
        (ROOT / "robot/cell/submission/placement.yaml").read_text(encoding="utf-8")
    )
    objects = payload["rollouts"][0]["objects"]
    service_lock = next(key for key in objects if key.startswith("M5_04 "))
    objects[service_lock.replace("M5_04", "M4_04", 1)] = objects.pop(service_lock)
    path = tmp_path / "placement.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ControllerPortError, match="one complete catalog product"):
        load_real_cell_layout(path)
