"""Opt-in T09 controller fixture for the official local mock only.

The perception and calibration ports in this module are deterministic contract
fixtures. They do not read simulator ground truth and are not valid for a real
cell, model evaluation, or physical safety decisions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from assembly.catalog import ProductRecipe, RecipeStep, load_catalog
from assembly.controller import AssemblyController
from assembly.controller_ports import (
    CollisionScene,
    ControllerEvent,
    EvidencePair,
    ExecutionTolerances,
    FinalProductEvidence,
    InventorySnapshot,
    ObservedInventoryItem,
    PerceptionTolerances,
    PlacementEvidence,
    RecoveryTransferEvidence,
)
from assembly.controller_startup import StartupHoldConfig
from assembly.fixed_inspection import InspectionPoseRegistry, InspectionSlotPose
from assembly.instruction import parse_requested_products
from assembly.inventory import InventoryItem, InventoryZone
from assembly.motion import (
    AbsolutePoseUF0,
    InspectionStopConfig,
    MotionConfig,
    OrientedPartPose,
    RestingState,
    Waypoint,
)
from assembly.recovery import PullOutTarget
from assembly.safety import AABBCollisionChecker, ReachabilityCapabilities, SafetyConfig, SafetySupervisor
from assembly.verification import VerificationThresholds
from robot.adapter import TCP_LIMITS, Action, Observation

BOOTSTRAP_ACTION = [200.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0]
MOCK_THRESHOLDS = VerificationThresholds(0.9, 0.8, 0.9, 0.1, 1.0)


def _oriented(x: float, y: float, z: float, yaw: float) -> OrientedPartPose:
    tcp = AbsolutePoseUF0(x, y, z, -180.0, 0.0, yaw, "UF0", "calibrated")
    return OrientedPartPose(tcp, RestingState.INSTALLED, "mock-installed:+z")


def _part_indices(object_id: str) -> tuple[int, int]:
    product, step = (int(value) for value in object_id[1:].split("_"))
    return product, step


class MockCalibratedPoses:
    """Fixed in-box poses for transport tests, never physical calibration."""

    def grasp_for(self, part: InventoryItem, *, observed_at_s: float) -> OrientedPartPose:
        del observed_at_s
        product, step = _part_indices(part.object_id)
        if part.zone is InventoryZone.INSTALLED:
            return _oriented(360.0 + 20.0 * product, 80.0, -300.0 + step, 90.0)
        return _oriented(210.0 + 30.0 * product, -160.0 + 35.0 * step, -300.0, step * 10.0)

    def base_for(self, base: InventoryItem, *, observed_at_s: float) -> OrientedPartPose:
        del observed_at_s
        product = int(base.instance_id.rsplit("-", maxsplit=1)[-1])
        return _oriented(400.0 + 20.0 * product, 80.0, -320.0, 90.0)

    def grasp_from_foundationpose(
        self,
        part: InventoryItem,
        detection: dict[str, object],
        *,
        observed_at_s: float,
    ) -> OrientedPartPose:
        del part, observed_at_s
        pose = detection["pose"]
        oriented = _oriented(float(pose["x"]), float(pose["y"]), float(pose["z"]), float(pose["yaw"]))
        if detection.get("resting_side") in {"side", "flipped"}:
            return OrientedPartPose(oriented.tcp, RestingState.FALLEN, "mock-fallen:observed")
        return oriented

    def target_for(
        self,
        product: ProductRecipe,
        base: InventoryItem,
        step: RecipeStep,
        *,
        observed_at_s: float,
    ) -> OrientedPartPose:
        del base, observed_at_s
        return _oriented(360.0 + 20.0 * product.product, 80.0, -300.0 + step.step, 90.0)

    def verification_target_for(
        self,
        product: ProductRecipe,
        base: InventoryItem,
        step: RecipeStep,
        *,
        observed_at_s: float,
    ) -> OrientedPartPose:
        return self.target_for(product, base, step, observed_at_s=observed_at_s)


class MockRecoveryTargets:
    """Fixed recovery zones used only by the local mock fixture."""

    def pull_out_target_for(self, product: int, part: InventoryItem) -> PullOutTarget:
        del product, part
        return PullOutTarget(
            _oriented(500.0, -300.0, -300.0, 0.0),
            -180.0,
            "mock-only deterministic test-bed clearance plane",
        )

    def staging_target_for(self, product: int, part: InventoryItem, slot: int) -> OrientedPartPose:
        del product, part
        return _oriented(500.0, -300.0 + 35.0 * slot, -300.0, 0.0)

    def staging_verification_target_for(
        self,
        product: int,
        part: InventoryItem,
        slot: int,
    ) -> OrientedPartPose:
        return self.staging_target_for(product, part, slot)

    def reject_target_for(self, part: InventoryItem) -> OrientedPartPose:
        del part
        return _oriented(550.0, -350.0, -300.0, 0.0)


class MockFinalProductVerifier:
    """Deterministic two-view workflow available only to the local mock."""

    reviewed_motion_source = "mock-only endpoint-verified final views"
    mock_only = True

    def verify(
        self,
        observation: Observation,
        product: int,
        base_instance_id: str,
        installed_instance_ids: tuple[str, ...],
    ) -> FinalProductEvidence:
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


class MockBoxReachability:
    """TCP-box-only reachability fixture; it is not robot-link kinematics."""

    capabilities = ReachabilityCapabilities(full_robot_kinematics=False)

    def is_reachable(self, pose: tuple[float, float, float, float, float, float]) -> bool:
        return all(lower <= value <= upper for value, (lower, upper, _) in zip(pose[:3], TCP_LIMITS[:3], strict=True))


def _detection(instance_id: str, object_id: str, *, resting: str = "upright") -> dict[str, object]:
    product, step = _part_indices(object_id)
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


def _inspection(instance_id: str, object_id: str) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "object_id": object_id,
        "verdict": "ok",
        "action": "use",
        "checks_agree": True,
        "evidence": {
            "classifier_score": 0.95,
            "patchcore_score": 0.05,
            "depth_residual_mm": 0.2,
            "image_ref": f"mock://{instance_id}",
        },
    }


def _observed_items(products: tuple[int, ...]) -> tuple[ObservedInventoryItem, ...]:
    items: list[ObservedInventoryItem] = []
    missing = (products[-1], 3) if len(products) >= 3 else None
    for product in products:
        items.append(ObservedInventoryItem(f"base-{product}", f"F{product:02d}", coarse_xy_mm=(400.0, 80.0)))
        items.extend(_observed_parts(product, missing_step=3 if missing == (product, 3) else None))
    items.append(_mock_spare(products[0], 2))
    if missing is not None:
        items.append(_mock_spare(*missing))
    return tuple(items)


def _mock_spare(product: int, step: int) -> ObservedInventoryItem:
    return ObservedInventoryItem(
        f"p{product}-{step}-spare",
        f"M{product}_{step:02d}",
        zone=InventoryZone.SPARE,
        coarse_xy_mm=(210.0 + 30.0 * product, -160.0 + 35.0 * step),
    )


def _observed_parts(product: int, *, missing_step: int | None) -> list[ObservedInventoryItem]:
    return [
        ObservedInventoryItem(
            f"p{product}-{step}",
            f"M{product}_{step:02d}",
            coarse_xy_mm=(210.0 + 30.0 * product, -160.0 + 35.0 * step),
        )
        for step in range(1, 5)
        if step != missing_step
    ]


class MockPerception:
    """Contract-valid scripted evidence; no image or ground-truth access."""

    def __init__(self) -> None:
        self._catalog = load_catalog()
        self.reset()

    def reset(self) -> None:
        self._instruction: object = None
        self.observed: tuple[ObservedInventoryItem, ...] = ()
        self.fallen_once: set[str] = set()
        self.failed_insertions: Counter[str] = Counter()
        self.staged_poses: dict[str, dict[str, float]] = {}
        self.next_staging_slot = 1

    def scan(self, observation: Observation) -> InventorySnapshot:
        self._configure_from_prompt(observation)
        return InventorySnapshot(float(observation["t"]), True, self.observed)

    def _configure_from_prompt(self, observation: Observation) -> None:
        trial = observation.get("trial")
        instruction = trial.get("instruction") if isinstance(trial, dict) else None
        if self._instruction is not None:
            return
        products = parse_requested_products(instruction, self._catalog)
        self._instruction = instruction
        self.observed = _observed_items(products)
        self.fallen_once = {f"p{products[0]}-2"}
        if len(products) >= 2:
            self.failed_insertions = Counter({f"p{products[1]}-3": 1})

    def pre_pick(self, observation: Observation, part: InventoryItem) -> EvidencePair:
        resting = "side" if part.instance_id in self.fallen_once else "upright"
        self.fallen_once.discard(part.instance_id)
        detection = _detection(part.instance_id, part.object_id, resting=resting)
        if part.zone is InventoryZone.INSTALLED:
            product, step = _part_indices(part.object_id)
            detection["pose"] = {"x": 360.0 + 20.0 * product, "y": 80.0, "z": -300.0 + step, "yaw": 90.0}
        elif part.zone is InventoryZone.STAGED:
            detection["pose"] = self.staged_poses[part.instance_id]
        return EvidencePair(
            detection,
            _inspection(part.instance_id, part.object_id),
            float(observation["t"]),
        )

    def post_place(self, observation: Observation, part: InventoryItem) -> PlacementEvidence:
        remaining = self.failed_insertions[part.instance_id]
        if remaining:
            self.failed_insertions[part.instance_id] -= 1
        seated = remaining == 0
        product, step = _part_indices(part.object_id)
        detection = _detection(part.instance_id, part.object_id)
        detection["pose"] = {"x": 360.0 + 20.0 * product, "y": 80.0, "z": -300.0 + step, "yaw": 90.0}
        return PlacementEvidence(
            detection,
            _inspection(part.instance_id, part.object_id),
            float(observation["t"]),
            seated=seated,
            pose_agrees=seated,
        )

    def post_transfer(
        self,
        observation: Observation,
        part: InventoryItem,
        target_zone: InventoryZone,
    ) -> RecoveryTransferEvidence:
        if target_zone is InventoryZone.STAGED:
            target_pose = {
                "x": 500.0,
                "y": -300.0 + 35.0 * self.next_staging_slot,
                "z": -300.0,
                "yaw": 0.0,
            }
            self.staged_poses[part.instance_id] = target_pose
            self.next_staging_slot += 1
        else:
            target_pose = {"x": 550.0, "y": -350.0, "z": -300.0, "yaw": 0.0}
        detection = _detection(part.instance_id, part.object_id)
        detection["pose"] = target_pose
        return RecoveryTransferEvidence(
            detection,
            _inspection(part.instance_id, part.object_id),
            float(observation["t"]),
            target_zone,
            True,
        )

    def collision_scene(
        self,
        observation: Observation,
        moving_instance_id: str,
        base_instance_id: str,
    ) -> CollisionScene:
        product = int(base_instance_id.rsplit("-", maxsplit=1)[-1])
        base_pose = (400.0 + 20.0 * product, 80.0, -320.0, 90.0)
        attached = float(observation["joints"][6]) == 1.0
        return CollisionScene(
            float(observation["t"]),
            True,
            base_instance_id,
            base_pose,
            0.0,
            moving_instance_id,
            attached,
            False,
        )


def _mock_safety() -> SafetySupervisor:
    config = SafetyConfig(
        0.1,
        -335.0,
        1.0,
        -180.0,
        0.0,
        0.5,
        1.0,
        10000.0,
        "mock-only deterministic runtime fixture",
    )
    return SafetySupervisor(
        config=config,
        reachability=MockBoxReachability(),
        collision=AABBCollisionChecker(),
        allow_tcp_sweep_only_for_mock=True,
    )


def _controller_kwargs(perception: MockPerception) -> dict[str, object]:
    motion = MotionConfig(
        10.0,
        10.0,
        10.0,
        10.0,
        1000.0,
        360.0,
        100,
        -180.0,
        "mock-only deterministic runtime fixture",
    )
    return {
        "catalog": load_catalog(),
        "perception": perception,
        "poses": MockCalibratedPoses(),
        "motion_config": motion,
        "inspection_stop": InspectionStopConfig(
            -168.0,
            -180.0,
            0.0,
            "PR9 follow-up measured CSI wrist inspection stop",
        ),
        "startup_hold": StartupHoldConfig(
            Waypoint(200.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0),
            "mock-only deterministic runtime fixture",
        ),
        "safety": _mock_safety(),
        "verification_thresholds": MOCK_THRESHOLDS,
        "execution_tolerances": ExecutionTolerances(0.0, 0.0),
        "perception_tolerances": PerceptionTolerances(0.0, 0.0),
        "recovery_targets": MockRecoveryTargets(),
        "final_verifier": MockFinalProductVerifier(),
        "allow_mock_final_verification": True,
    }


def _mock_inspection_poses() -> InspectionPoseRegistry:
    """Return fixed mock-grid camera poses; these are not real-cell calibration."""
    slots = []
    for product in range(1, 7):
        for step in range(1, 5):
            x = 210.0 + 30.0 * product
            y = -160.0 + 35.0 * step
            slots.append(
                InspectionSlotPose(
                    f"mock-table-p{product}-s{step}",
                    (x - 10.0, x + 10.0, y - 10.0, y + 10.0),
                    (x, y, -168.0, -180.0, 0.0, 0.0),
                    -180.0,
                    "mock-only deterministic inventory grid; not real-cell calibration",
                )
            )
    return InspectionPoseRegistry(tuple(slots))


def _build_controller(perception: MockPerception) -> AssemblyController:
    return AssemblyController(**_controller_kwargs(perception))


def build_groot_mock_controller(
    executor,
    *,
    inspection_poses: InspectionPoseRegistry | None = None,
) -> AssemblyController:
    """Software-proof fixture: real T09 states with GR00T as sole row source."""
    from assembly.cell_runtime import (
        SATURDAY_ENDPOINT_SETTLE_LIMITS,
        SATURDAY_GROOT_WATCHDOG_LIMITS,
    )
    from assembly.groot_controller import GrootAssemblyController, GrootControllerAuthority

    perception = MockPerception()
    return GrootAssemblyController(
        groot_executor=executor,
        inspection_poses=inspection_poses or _mock_inspection_poses(),
        endpoint_settle_limits=SATURDAY_ENDPOINT_SETTLE_LIMITS,
        watchdog_limits=SATURDAY_GROOT_WATCHDOG_LIMITS,
        runtime_authority=GrootControllerAuthority.OFFICIAL_MOCK_PROOF,
        **_controller_kwargs(perception),
    )


@dataclass(frozen=True, slots=True)
class MockRunRecord:
    completed_products: tuple[int, ...]
    events: tuple[ControllerEvent, ...]
    finished: bool


class T09MockController:
    """Bootstrap and archive an AssemblyController for the local mock."""

    def __init__(self) -> None:
        self._perception = MockPerception()
        self._controller = _build_controller(self._perception)
        self._archived: list[MockRunRecord] = []

    def reset(self) -> None:
        if self._controller.events:
            self._archived.append(self.current_record)
        self._perception.reset()
        self._controller.reset()

    def act(self, observation: Observation) -> Action:
        if observation.get("tcp_pose") is None:
            return list(BOOTSTRAP_ACTION)
        return self._controller.act(observation)

    @property
    def current_record(self) -> MockRunRecord:
        return MockRunRecord(
            self._controller.completed_products,
            self._controller.events,
            self._controller.finished,
        )

    @property
    def archived_runs(self) -> tuple[MockRunRecord, ...]:
        return tuple(self._archived)

    @property
    def run_records(self) -> tuple[MockRunRecord, ...]:
        current = () if not self._controller.events else (self.current_record,)
        return (*self.archived_runs, *current)

    @property
    def completed_products(self) -> tuple[int, ...]:
        return self._controller.completed_products

    @property
    def events(self) -> tuple[ControllerEvent, ...]:
        return self._controller.events

    @property
    def finished(self) -> bool:
        return self._controller.finished
