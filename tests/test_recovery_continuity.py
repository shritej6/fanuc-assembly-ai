"""Physical pose continuity across staged dismantle and rebuild."""

from assembly.catalog import load_catalog
from assembly.inventory import Inventory, InventoryItem, InventoryZone
from assembly.motion import AbsolutePoseUF0, MotionConfig, MotionPhase, OrientedPartPose, RestingState, Waypoint
from assembly.planner import RecipePlanner
from assembly.recovery import RecoveryCoordinator, RecoveryStageKind
from assembly.verification import decide_recovery


def _pose(x: float, y: float, z: float, yaw: float = 0.0) -> OrientedPartPose:
    tcp = AbsolutePoseUF0(x, y, z, -180.0, 0.0, yaw, "UF0", "calibrated")
    return OrientedPartPose(tcp, RestingState.INSTALLED, "installed:+z")


class TrackingPoses:
    def __init__(self) -> None:
        self.grasp_calls: list[str] = []

    def grasp_for(self, part: InventoryItem, *, observed_at_s: float) -> OrientedPartPose:
        assert observed_at_s == 5.0
        self.grasp_calls.append(part.instance_id)
        if part.object_id == "M1_01":
            return _pose(380.0, 80.0, -300.0)
        return _pose(240.0, -125.0, -300.0)

    def base_for(self, base: InventoryItem, *, observed_at_s: float) -> OrientedPartPose:
        assert base.instance_id == "base-1" and observed_at_s == 5.0
        return _pose(350.0, 50.0, -300.0)

    def target_for(self, product, base, step, *, observed_at_s: float) -> OrientedPartPose:
        assert product.product == 1 and base.instance_id == "base-1" and observed_at_s == 5.0
        return _pose(380.0, 80.0, -300.0 + step.step)


class RecoveryTargets:
    def staging_target_for(self, product: int, part: InventoryItem, slot: int) -> OrientedPartPose:
        assert product == 1 and part.instance_id == "p1-1" and slot == 1
        return _pose(500.0, -265.0, -300.0)

    def reject_target_for(self, part: InventoryItem) -> OrientedPartPose:
        assert part.instance_id == "p1-2"
        return _pose(550.0, -350.0, -300.0)


def _inventory() -> Inventory:
    inventory = Inventory(load_catalog())
    inventory.add("base-1", "F01")
    for step in range(1, 5):
        inventory.add(f"p1-{step}", f"M1_{step:02d}")
    inventory.add("p1-2-spare", "M1_02", zone=InventoryZone.SPARE)
    inventory.reserve_product(1)
    inventory.install("p1-1")
    return inventory


def _prepare(coordinator, inventory, catalog, poses, config, current):
    return coordinator.prepare_stage(
        inventory=inventory,
        catalog=catalog,
        poses=poses,
        current=current,
        config=config,
        observed_at_s=5.0,
    )


def test_rebuild_grasps_verified_staging_slot_and_holds_until_verification() -> None:
    catalog, inventory, poses = load_catalog(), _inventory(), TrackingPoses()
    prepared = RecipePlanner(catalog, poses).next_step(inventory, observed_at_s=5.0)
    assert prepared is not None
    decision = decide_recovery("M1_02", 2, True, installed_object_ids=("M1_01",))
    coordinator = RecoveryCoordinator(RecoveryTargets())
    coordinator.begin(inventory, prepared, decision)
    config = MotionConfig(10.0, 20.0, 10.0, 10.0, 1000.0, 360.0, 100)

    held = Waypoint(380.0, 80.0, -298.0, -180.0, 0.0, 0.0, 1.0)
    rejected = _prepare(coordinator, inventory, catalog, poses, config, held)
    assert rejected is not None and rejected.stage.kind is RecoveryStageKind.REJECT_FAILED
    coordinator.commit(inventory, rejected.stage)

    idle = Waypoint(200.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0)
    staged = _prepare(coordinator, inventory, catalog, poses, config, idle)
    assert staged is not None and staged.stage.kind is RecoveryStageKind.STAGE_INSTALLED
    assert staged.release_after_verification is True
    assert MotionPhase.RELEASE not in {chunk.phase for chunk in staged.chunks}
    coordinator.commit(inventory, staged.stage)

    rebuilt = _prepare(coordinator, inventory, catalog, poses, config, idle)
    assert rebuilt is not None and rebuilt.stage.kind is RecoveryStageKind.REBUILD_INSTALLED
    grip = next(chunk for chunk in rebuilt.chunks if chunk.phase is MotionPhase.GRIP).waypoints[-1]
    assert grip.xyz == (500.0, -265.0, -300.0)
    assert poses.grasp_calls.count("p1-1") == 1
    assert MotionPhase.RELEASE not in {chunk.phase for chunk in rebuilt.chunks}
