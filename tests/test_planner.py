"""Recipe planning from validated inventory into the shared plan-step contract."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from assembly.catalog import load_catalog
from assembly.inventory import Inventory
from assembly.motion import AbsolutePoseUF0, OrientedPartPose, RestingState

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"


class CalibratedPoses:
    def grasp_for(self, part, *, observed_at_s: float) -> OrientedPartPose:
        assert observed_at_s == 10.0
        step = int(part.object_id[-2:])
        return _pose(220.0 + step * 10, -80.0, -300.0, 10.0)

    def base_for(self, base, *, observed_at_s: float) -> OrientedPartPose:
        assert observed_at_s == 10.0
        return _pose(400.0, 40.0, -300.0, 90.0)

    def target_for(self, product, base, step, *, observed_at_s: float) -> OrientedPartPose:
        assert observed_at_s == 10.0
        return _pose(420.0, 40.0, -290.0 + step.step, 90.0)


def _pose(x: float, y: float, z: float, yaw: float) -> OrientedPartPose:
    return OrientedPartPose(
        tcp=AbsolutePoseUF0(x, y, z, -180.0, 0.0, yaw, "UF0", "calibrated"),
        resting_state=RestingState.INSTALLED,
        up_face="installed:+z",
    )


def _inventory(product: int = 4) -> Inventory:
    inventory = Inventory(load_catalog())
    inventory.add("base", "F04")
    for step in range(1, 5):
        inventory.add(f"part-{step}", f"M{product}_{step:02d}")
    inventory.reserve_product(product)
    return inventory


def _registry() -> Registry:
    registry = Registry()
    for path in CONTRACTS.glob("*.schema.json"):
        registry = registry.with_resource(path.name, Resource.from_contents(json.loads(path.read_text())))
    return registry


def test_next_step_is_contract_valid_and_keeps_enriched_calibrated_poses() -> None:
    from assembly.planner import RecipePlanner

    prepared = RecipePlanner(load_catalog(), CalibratedPoses()).next_step(
        _inventory(), observed_at_s=10.0, attempt=2
    )

    assert prepared is not None
    record = prepared.record.as_dict()
    schema = json.loads((CONTRACTS / "plan_step.schema.json").read_text())
    assert list(Draft202012Validator(schema, registry=_registry()).iter_errors(record)) == []
    assert record == {
        "step_id": "product-4-step-01-attempt-2",
        "op": "insert",
        "product": 4,
        "instance_id": "part-1",
        "object_id": "M4_01",
        "grasp": {"x": 230.0, "y": -80.0, "z": -300.0, "yaw": 10.0},
        "target": {"x": 420.0, "y": 40.0, "z": -289.0, "yaw": 90.0},
        "gripper_roll_deg": 90.0,
        "attempt": 2,
    }
    assert prepared.grasp.tcp.w == prepared.target.tcp.w == -180.0
    assert prepared.grasp.tcp.p == prepared.target.tcp.p == 0.0


def test_next_step_follows_recipe_order_and_finishes_after_all_four_parts() -> None:
    from assembly.planner import RecipePlanner

    inventory = _inventory()
    planner = RecipePlanner(load_catalog(), CalibratedPoses())

    for expected_step in range(1, 5):
        prepared = planner.next_step(inventory, observed_at_s=10.0)
        assert prepared is not None
        assert prepared.record.object_id == f"M4_{expected_step:02d}"
        inventory.install(f"part-{expected_step}")

    assert planner.next_step(inventory, observed_at_s=10.0) is None


def test_next_step_rejects_out_of_order_installed_inventory() -> None:
    from assembly.planner import PlanOrderError, RecipePlanner

    inventory = _inventory()
    inventory.install("part-2")

    with pytest.raises(PlanOrderError, match="recipe prefix"):
        RecipePlanner(load_catalog(), CalibratedPoses()).next_step(inventory, observed_at_s=10.0)


def test_next_step_rejects_a_base_that_is_no_longer_reserved_and_verified() -> None:
    from assembly.planner import PlannerStateError, RecipePlanner

    inventory = _inventory()
    inventory.reject("base")

    with pytest.raises(PlannerStateError, match="base"):
        RecipePlanner(load_catalog(), CalibratedPoses()).next_step(inventory, observed_at_s=10.0)


def test_next_step_requires_active_reservation_and_positive_attempt() -> None:
    from assembly.planner import PlannerStateError, RecipePlanner

    planner = RecipePlanner(load_catalog(), CalibratedPoses())
    with pytest.raises(PlannerStateError, match="active reservation"):
        planner.next_step(Inventory(load_catalog()), observed_at_s=10.0)
    with pytest.raises(PlannerStateError, match="attempt"):
        planner.next_step(_inventory(), observed_at_s=10.0, attempt=0)

    with pytest.raises(PlannerStateError, match="observed_at_s"):
        planner.next_step(_inventory(), observed_at_s=float("nan"))


def test_next_step_rejects_untrusted_or_non_installed_pose() -> None:
    from assembly.planner import PlannerPoseError, RecipePlanner

    class UntrustedPoses(CalibratedPoses):
        def grasp_for(self, part, *, observed_at_s: float) -> OrientedPartPose:
            pose = super().grasp_for(part, observed_at_s=observed_at_s)
            return replace(pose, tcp=replace(pose.tcp, source="sim_ground_truth"))

    with pytest.raises(PlannerPoseError, match="calibrated"):
        RecipePlanner(load_catalog(), UntrustedPoses()).next_step(_inventory(), observed_at_s=10.0)
