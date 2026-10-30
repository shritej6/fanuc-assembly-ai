"""End-to-end controller behavior through the guarded T05 CellAdapter."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

from assembly.catalog import load_catalog
from assembly.controller_ports import (
    ExecutionTolerances,
    FinalProductEvidence,
    PerceptionTolerances,
)
from assembly.controller_startup import StartupHoldConfig
from assembly.inventory import InventoryZone, ObjectCondition
from assembly.motion import (
    AbsolutePoseUF0,
    InspectionStopConfig,
    MotionConfig,
    OrientedPartPose,
    RestingState,
    Waypoint,
)
from assembly.safety import AABB
from assembly.state_machine import PackMLState
from assembly.verification import VerificationThresholds
from robot.adapter import CellAdapter
from tests.controller_fixtures import build_test_safety


class CalibratedScenarioPoses:
    def grasp_for(self, part, *, observed_at_s: float) -> OrientedPartPose:
        del observed_at_s
        product, step = (int(value) for value in part.object_id[1:].split("_"))
        if part.zone is InventoryZone.INSTALLED:
            return _oriented(360.0 + 20.0 * product, 80.0, -300.0 + step, 90.0)
        return _oriented(210.0 + 30.0 * product, -160.0 + 35.0 * step, -300.0, step * 10.0)

    def base_for(self, base, *, observed_at_s: float) -> OrientedPartPose:
        del observed_at_s
        product = int(base.instance_id.rsplit("-", maxsplit=1)[-1])
        return _oriented(400.0 + 20.0 * product, 80.0, -320.0, 90.0)

    def grasp_from_foundationpose(self, part, detection, *, observed_at_s: float) -> OrientedPartPose:
        del observed_at_s
        assert detection["pose_source"] == "foundationpose"
        pose = detection["pose"]
        oriented = _oriented(float(pose["x"]), float(pose["y"]), float(pose["z"]), float(pose["yaw"]))
        if detection.get("resting_side") in {"side", "flipped"}:
            return replace(oriented, resting_state=RestingState.FALLEN, up_face="fallen:observed")
        return oriented

    def target_for(self, product, base, step, *, observed_at_s: float) -> OrientedPartPose:
        del base, observed_at_s
        return _oriented(360.0 + 20.0 * product.product, 100.0, -290.0 + step.step, 90.0)

    def verification_target_for(self, product, base, step, *, observed_at_s: float) -> OrientedPartPose:
        del base, observed_at_s
        return _oriented(360.0 + 20.0 * product.product, 80.0, -300.0 + step.step, 90.0)


class CalibratedRecoveryTargets:
    def staging_target_for(self, product, part, slot):
        return _oriented(500.0, -300.0 + 35.0 * slot, -300.0, 0.0)

    def staging_verification_target_for(self, product, part, slot):
        return _oriented(496.0, -300.0 + 35.0 * slot, -317.0, 0.0)

    def reject_target_for(self, part):
        return _oriented(550.0, -350.0, -300.0, 0.0)


class MockFinalProductVerifier:
    """Mock-only two-view workflow fixture; it never authorizes cell motion."""

    reviewed_motion_source = "mock-only endpoint-verified final views"
    mock_only = True

    def verify(self, observation, product, base_instance_id, installed_instance_ids):
        return FinalProductEvidence(
            observed_at_s=float(observation["t"]),
            complete=True,
            product=product,
            base_instance_id=base_instance_id,
            installed_instance_ids=installed_instance_ids,
            view_ids=(f"mock://product-{product}/left", f"mock://product-{product}/right"),
            view_instance_ids=(installed_instance_ids, installed_instance_ids),
            view_passes=(True, True),
            checks_agree=True,
            motion_source=self.reviewed_motion_source,
        )


def _oriented(x: float, y: float, z: float, yaw: float) -> OrientedPartPose:
    return OrientedPartPose(
        AbsolutePoseUF0(x, y, z, -180.0, 0.0, yaw, "UF0", "calibrated"),
        RestingState.INSTALLED,
        "installed:+z",
    )


@dataclass
class ScriptedPerception:
    observed: tuple
    fallen_once: set[str]
    failed_insertions: Counter
    disagreement_once: set[str]
    obstacles: tuple[AABB, ...] = ()

    def __post_init__(self) -> None:
        self.scan_calls = 0
        self.pre_calls: Counter = Counter()
        self.post_calls: Counter = Counter()
        self.transfer_calls: Counter = Counter()
        self.collision_calls = 0
        self.staged_poses: dict[str, dict[str, float]] = {}
        self.next_staging_slot = 1

    def scan(self, observation):
        from assembly.controller_ports import InventorySnapshot

        self.scan_calls += 1
        return InventorySnapshot(float(observation["t"]), True, self.observed)

    def pre_pick(self, observation, part):
        from assembly.controller_ports import EvidencePair

        self.pre_calls[part.instance_id] += 1
        resting = "side" if part.instance_id in self.fallen_once else "upright"
        self.fallen_once.discard(part.instance_id)
        checks_agree = part.instance_id not in self.disagreement_once
        self.disagreement_once.discard(part.instance_id)
        detection = _detection(part.instance_id, part.object_id, resting=resting)
        if part.zone is InventoryZone.INSTALLED:
            product, step = (int(value) for value in part.object_id[1:].split("_"))
            detection["pose"] = {"x": 360.0 + 20.0 * product, "y": 80.0, "z": -300.0 + step, "yaw": 90.0}
        elif part.zone is InventoryZone.STAGED:
            detection["pose"] = self.staged_poses[part.instance_id]
        return EvidencePair(
            detection,
            _inspection(part.instance_id, part.object_id, checks_agree=checks_agree),
            float(observation["t"]),
        )

    def post_place(self, observation, part):
        from assembly.controller_ports import PlacementEvidence

        self.post_calls[part.instance_id] += 1
        remaining = self.failed_insertions[part.instance_id]
        if remaining:
            self.failed_insertions[part.instance_id] -= 1
        seated = remaining == 0
        product, step = (int(value) for value in part.object_id[1:].split("_"))
        detection = _detection(part.instance_id, part.object_id)
        detection["pose"] = {"x": 360.0 + 20.0 * product, "y": 80.0, "z": -300.0 + step, "yaw": 90.0}
        return PlacementEvidence(
            detection,
            _inspection(part.instance_id, part.object_id),
            float(observation["t"]),
            seated=seated,
            pose_agrees=seated,
        )

    def post_transfer(self, observation, part, target_zone):
        from assembly.controller_ports import RecoveryTransferEvidence

        self.transfer_calls[(part.instance_id, target_zone)] += 1
        if target_zone is InventoryZone.STAGED:
            target_pose = {
                "x": 496.0,
                "y": -300.0 + 35.0 * self.next_staging_slot,
                "z": -317.0,
                "yaw": 0.0,
            }
            self.staged_poses[part.instance_id] = target_pose
            self.next_staging_slot += 1
        else:
            target_pose = {"x": 540.0, "y": -340.0, "z": -310.0, "yaw": 15.0}
        detection = _detection(part.instance_id, part.object_id)
        detection["pose"] = target_pose
        return RecoveryTransferEvidence(
            detection,
            _inspection(part.instance_id, part.object_id),
            float(observation["t"]),
            target_zone,
            True,
        )

    def collision_scene(self, observation, moving_instance_id, base_instance_id):
        from assembly.controller_ports import CollisionScene

        self.collision_calls += 1
        product = int(base_instance_id.rsplit("-", maxsplit=1)[-1])
        base_pose = (400.0 + 20.0 * product, 80.0, -320.0, 90.0)
        attached = float(observation["joints"][6]) == 1.0
        return CollisionScene(
            float(observation["t"]),
            True,
            base_instance_id,
            base_pose,
            moving_instance_id,
            attached,
            False,
            fixtures=self.obstacles,
            parts=(),
        )


def _detection(instance_id: str, object_id: str, *, resting: str = "upright") -> dict[str, object]:
    product, step = (int(value) for value in object_id[1:].split("_"))
    return {
        "instance_id": instance_id,
        "camera": "wrist",
        "label": object_id,
        "object_id": object_id,
        "score": 0.95,
        "bbox_px": [10.0, 20.0, 60.0, 80.0],
        "pose": {
            "x": 210.0 + 30.0 * product,
            "y": -160.0 + 35.0 * step,
            "z": -300.0,
            "yaw": step * 10.0,
        },
        "pose_source": "foundationpose",
        "pose_fit_score": 0.9,
        "resting_side": resting,
    }


def _inspection(instance_id: str, object_id: str, *, checks_agree: bool = True) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "object_id": object_id,
        "verdict": "ok",
        "action": "use",
        "checks_agree": checks_agree,
        "evidence": {
            "classifier_score": 0.95,
            "patchcore_score": 0.05,
            "depth_residual_mm": 0.2,
            "image_ref": f"wrist://{instance_id}",
        },
    }


def _observed_three_products(*, include_missing_spare: bool = True):
    from assembly.controller_ports import ObservedInventoryItem

    items = []
    for product, base in ((1, "F01"), (3, "F03"), (4, "F04")):
        items.append(ObservedInventoryItem(f"base-{product}", base, coarse_xy_mm=(400.0, 80.0)))
        for step in range(1, 5):
            if product == 4 and step == 3:
                continue
            items.append(
                ObservedInventoryItem(
                    f"p{product}-{step}",
                    f"M{product}_{step:02d}",
                    coarse_xy_mm=(210.0 + 30.0 * product, -160.0 + 35.0 * step),
                )
            )
    items.append(
        ObservedInventoryItem("p1-2-spare", "M1_02", zone=InventoryZone.SPARE, coarse_xy_mm=(240.0, -90.0))
    )
    if include_missing_spare:
        items.append(
            ObservedInventoryItem("p4-3-spare", "M4_03", zone=InventoryZone.SPARE, coarse_xy_mm=(330.0, -55.0))
        )
    return tuple(items)


def _observed_family_c_products():
    from assembly.controller_ports import ObservedInventoryItem

    items = []
    for product in (4, 5, 6):
        items.append(ObservedInventoryItem(f"base-{product}", f"F0{product}", coarse_xy_mm=(400.0, 80.0)))
        items.extend(
            ObservedInventoryItem(
                f"p{product}-{step}",
                f"M{product}_{step:02d}",
                coarse_xy_mm=(210.0 + 30.0 * product, -160.0 + 35.0 * step),
            )
            for step in range(1, 5)
        )
    return tuple(items)


def _observed_products(products: tuple[int, ...]):
    from assembly.controller_ports import ObservedInventoryItem

    items = []
    for product in products:
        items.append(ObservedInventoryItem(f"base-{product}", f"F0{product}", coarse_xy_mm=(400.0, 80.0)))
        items.extend(
            ObservedInventoryItem(
                f"p{product}-{step}",
                f"M{product}_{step:02d}",
                coarse_xy_mm=(210.0 + 30.0 * product, -160.0 + 35.0 * step),
            )
            for step in range(1, 5)
        )
    return tuple(items)


def _controller(
    perception: ScriptedPerception,
    perception_tolerances: PerceptionTolerances | None = None,
    execution_tolerances: ExecutionTolerances | None = None,
    poses=None,
    final_verifier=None,
    *,
    allow_mock_final_verification: bool = True,
):
    from assembly.controller import AssemblyController

    motion = MotionConfig(20.0, 30.0, 15.0, 20.0, 50.0, 30.0, 8, -180.0, "mock-only test fixture")
    calibrated_poses = poses or CalibratedScenarioPoses()
    return AssemblyController(
        catalog=load_catalog(),
        perception=perception,
        poses=calibrated_poses,
        motion_config=motion,
        inspection_stop=InspectionStopConfig(
            -168.0,
            -180.0,
            0.0,
            "PR9 follow-up measured CSI wrist inspection stop",
        ),
        startup_hold=StartupHoldConfig(
            Waypoint(200.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0),
            "mock-only initial observation fixture",
        ),
        safety=build_test_safety(),
        verification_thresholds=VerificationThresholds(0.9, 0.8, 0.9, 0.1, 1.0),
        execution_tolerances=execution_tolerances or ExecutionTolerances(0.0, 0.0),
        perception_tolerances=perception_tolerances or PerceptionTolerances(0.0, 0.0),
        recovery_targets=CalibratedRecoveryTargets(),
        final_verifier=final_verifier or MockFinalProductVerifier(),
        allow_mock_final_verification=allow_mock_final_verification,
    )
def _observation(products: tuple[int, ...] = (1, 3, 4)) -> dict[str, object]:
    product_text = ", ".join(str(product) for product in products)
    return {
        "t": 0.0,
        "tcp_pose": [200.0, 0.0, -250.0, -180.0, 0.0, 0.0],
        "joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "images": {},
        "trial": {"instruction": f"assemble products {product_text}"},
    }


def _drive(
    controller,
    *,
    products: tuple[int, ...] = (1, 3, 4),
    limit: int = 800,
) -> list[list[float]]:
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation(products)
    actions: list[list[float]] = []
    for _ in range(limit):
        result = adapter.act(observation)
        rows = result if result and isinstance(result[0], list) else [result]
        actions.extend(rows)
        last = rows[-1]
        observation["tcp_pose"] = last[:6]
        observation["joints"] = [0.0] * 6 + [last[6]]
        observation["t"] = float(observation["t"]) + 1.0
        if controller.finished or controller.state_machine.state is PackMLState.HELD:
            return actions
    raise AssertionError(f"controller did not reach a terminal gate within {limit} callbacks")


def test_three_product_loop_recovers_fallen_failed_and_missing_parts() -> None:
    perception = ScriptedPerception(
        observed=_observed_three_products(),
        fallen_once={"p1-2"},
        failed_insertions=Counter({"p3-3": 1}),
        disagreement_once=set(),
    )
    controller = _controller(perception)

    actions = _drive(controller)

    assert actions
    assert controller.completed_products == (1, 3, 4)
    assert len(controller.inventory.items(InventoryZone.INSTALLED)) == 12
    assert any(event.kind == "spare" and event.detail == "p4-3-spare" for event in controller.events)
    assert controller.inventory.item("p1-2").zone is InventoryZone.REJECT
    assert controller.inventory.item("p1-2").condition is ObjectCondition.FALLEN
    assert controller.inventory.item("p1-2-spare").zone is InventoryZone.INSTALLED
    assert controller.inventory.item("p4-3-spare").zone is InventoryZone.INSTALLED
    assert perception.transfer_calls[("p1-2", InventoryZone.REJECT)] == 1
    assert perception.post_calls["p3-3"] == 2
    assert any(event.kind == "retry" and event.object_id == "M3_03" for event in controller.events)


def test_missing_part_without_spare_suspends_and_calls_andon_without_motion() -> None:
    observed = tuple(
        item
        for item in _observed_three_products(include_missing_spare=False)
        if item.object_id.startswith("M4") or item.object_id == "F04"
    )
    perception = ScriptedPerception(observed, set(), Counter(), set())
    controller = _controller(perception)

    output = CellAdapter(controller, action_mode="tcp", dt=0.1).act(_observation((4,)))

    assert output == [200.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0]
    assert controller.state_machine.state is PackMLState.SUSPENDED
    assert controller.events[-1].kind == "andon"
    assert controller.plan_records == ()


def test_independent_check_disagreement_holds_before_any_pick_motion() -> None:
    observed = tuple(
        item for item in _observed_three_products() if item.object_id.startswith("M1") or item.object_id == "F01"
    )
    perception = ScriptedPerception(observed, set(), Counter(), {"p1-1"})
    controller = _controller(perception)

    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))
    inspect = adapter.act(observation)
    rows = inspect if isinstance(inspect[0], list) else [inspect]
    observation["tcp_pose"] = rows[-1][:6]
    observation["joints"] = [0.0] * 6 + [rows[-1][6]]
    observation["t"] = 1.0

    output = adapter.act(observation)

    assert output == observation["tcp_pose"] + [0.0]
    assert controller.state_machine.state is PackMLState.HELD
    assert controller.events[-1].kind == "andon"
    assert controller.plan_records == ()
    assert "published calibrated wrist-view targets" in controller.events[-1].detail


def test_each_chunk_revalidates_from_actual_observed_tcp() -> None:
    observed = tuple(
        item for item in _observed_three_products() if item.object_id.startswith("M1") or item.object_id == "F01"
    )
    perception = ScriptedPerception(observed, set(), Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))

    assert adapter.act(observation)
    observation["tcp_pose"] = [650.0, 450.0, -250.0, -180.0, 0.0, 0.0]
    observation["t"] = 1.0
    held = adapter.act(observation)

    assert held == [650.0, 450.0, -250.0, -180.0, 0.0, 0.0, 0.0]
    assert controller.state_machine.state is PackMLState.ABORTED
    assert "previously guarded endpoint" in controller.events[-1].detail


def test_disappearing_reserved_part_holds_before_next_chunk() -> None:
    observed = tuple(
        item for item in _observed_three_products() if item.object_id.startswith("M1") or item.object_id == "F01"
    )
    perception = ScriptedPerception(observed, set(), Counter(), set())
    controller = _controller(perception)
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation((1,))
    first = adapter.act(observation)
    rows = first if isinstance(first[0], list) else [first]

    observation["tcp_pose"] = rows[-1][:6]
    observation["joints"] = [0.0] * 6 + [rows[-1][6]]
    observation["t"] = 1.0
    perception.observed = ()
    adapter.act(observation)

    assert controller.state_machine.state is PackMLState.HELD
    assert controller.inventory.active_reservation.product == 1
    assert "verified" in controller.events[-1].detail


def test_same_base_family_products_consume_distinct_completed_bases() -> None:
    perception = ScriptedPerception(_observed_family_c_products(), set(), Counter(), set())
    controller = _controller(perception)

    _drive(controller, products=(4, 5, 6), limit=1200)

    assert controller.completed_products == (4, 5, 6)
    assert {controller.inventory.item(f"base-{product}").zone for product in (4, 5, 6)} == {
        InventoryZone.OUTPUT
    }
