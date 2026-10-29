"""Scripted perception boundary and mock action follower for T17 scenarios."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from assembly.catalog import Catalog, ProductRecipe, RecipeStep, load_catalog
from assembly.controller import AssemblyController
from assembly.controller_ports import (
    CollisionScene,
    ControllerPortError,
    EvidencePair,
    InventorySnapshot,
    ObservedInventoryItem,
    PlacementEvidence,
    RecoveryTransferEvidence,
)
from assembly.inventory import InventoryItem, InventoryZone, ObjectCondition
from assembly.motion import AbsolutePoseUF0, OrientedPartPose, RestingState
from assembly.state_machine import PackMLState
from eval.scenario_data import FleetInputError, Scenario, ScenarioObject, parse_instruction_products
from eval.scenario_report import (
    FleetReport,
    ScenarioPhysicalEvent,
    ScenarioPhysicalEventKind,
    ScenarioResult,
    ScenarioRunStatus,
)
from robot.adapter import CellAdapter, Observation


class ControllerFactory(Protocol):
    def __call__(self, runtime: ScenarioRuntime) -> AssemblyController: ...


@dataclass(slots=True)
class _PhysicalState:
    pose_by_instance: dict[str, tuple[float, float, float, float]]
    observed_pose_by_instance: dict[str, tuple[float, float, float, float]]
    target_by_object: dict[str, tuple[float, float, float, float]]
    transfer_by_key: dict[tuple[str, InventoryZone], tuple[float, float, float, float]]


class ScriptedPerceptionPort:
    """Contract-shaped evidence generated only from the scripted camera boundary."""

    def __init__(self, scenario: Scenario, state: _PhysicalState) -> None:
        self._scenario = scenario
        self._state = state
        self._objects = {item.instance_id: item for item in scenario.objects}
        self.rejected_instances: set[str] = set()
        self.physical_events: list[ScenarioPhysicalEvent] = []

    def scan(self, observation: Observation) -> InventorySnapshot:
        self._state.observed_pose_by_instance = dict(self._state.pose_by_instance)
        items = tuple(
            ObservedInventoryItem(
                item.instance_id,
                item.object_id,
                item.zone,
                item.condition,
                coarse_xy_mm=self._state.pose_by_instance[item.instance_id][:2],
            )
            for item in self._scenario.objects
        )
        return InventorySnapshot(float(observation["t"]), True, items)

    def pre_pick(self, observation: Observation, part: InventoryItem) -> EvidencePair:
        self.physical_events.append(ScenarioPhysicalEvent(ScenarioPhysicalEventKind.PRE_PICK, part.instance_id))
        scripted = self._scripted(part.instance_id)
        return EvidencePair(
            self._detection(scripted),
            self._inspection(scripted),
            float(observation["t"]),
        )

    def post_place(self, observation: Observation, part: InventoryItem) -> PlacementEvidence:
        target = self._state.target_by_object[part.object_id]
        self._state.pose_by_instance[part.instance_id] = target
        scripted = self._scripted(part.instance_id)
        return PlacementEvidence(
            self._detection(scripted, condition=ObjectCondition.VERIFIED),
            self._inspection(scripted, condition=ObjectCondition.VERIFIED),
            float(observation["t"]),
            seated=True,
            pose_agrees=True,
        )

    def post_transfer(
        self,
        observation: Observation,
        part: InventoryItem,
        target_zone: InventoryZone,
    ) -> RecoveryTransferEvidence:
        target = self._state.transfer_by_key[(part.instance_id, target_zone)]
        self._state.pose_by_instance[part.instance_id] = target
        if target_zone is InventoryZone.REJECT:
            self.rejected_instances.add(part.instance_id)
            self.physical_events.append(
                ScenarioPhysicalEvent(ScenarioPhysicalEventKind.REJECT_TRANSFER, part.instance_id)
            )
        scripted = self._scripted(part.instance_id)
        return RecoveryTransferEvidence(
            self._detection(scripted),
            self._inspection(scripted),
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
        base_pose = _observed_pose(self._state, base_instance_id)
        attached = float(observation["joints"][6]) == 1.0  # type: ignore[index]
        return CollisionScene(
            float(observation["t"]),
            True,
            base_instance_id,
            base_pose,
            moving_instance_id,
            attached,
        )

    def _scripted(self, instance_id: str) -> ScenarioObject:
        try:
            return self._objects[instance_id]
        except KeyError as error:
            raise ControllerPortError(f"scenario has no perception evidence for {instance_id!r}") from error

    def _detection(
        self,
        item: ScenarioObject,
        *,
        condition: ObjectCondition | None = None,
    ) -> dict[str, object]:
        effective = item.condition if condition is None else condition
        if effective is ObjectCondition.FALLEN:
            resting = "side" if item.grasp_approved else "flipped"
        else:
            resting = "upright"
        x, y, z, yaw = self._state.pose_by_instance[item.instance_id]
        return {
            "instance_id": item.instance_id,
            "camera": "wrist",
            "label": item.object_id,
            "object_id": item.object_id,
            "score": 0.95,
            "bbox_px": [10.0, 20.0, 60.0, 80.0],
            "pose": {"x": x, "y": y, "z": z, "yaw": yaw},
            "pose_source": "foundationpose",
            "pose_fit_score": 0.95,
            "resting_side": resting,
        }

    @staticmethod
    def _inspection(
        item: ScenarioObject,
        *,
        condition: ObjectCondition | None = None,
    ) -> dict[str, object]:
        effective = item.condition if condition is None else condition
        verdict, action = _inspection_outcome(effective)
        return {
            "instance_id": item.instance_id,
            "object_id": item.object_id,
            "verdict": verdict,
            "action": action,
            "checks_agree": True,
            "evidence": {
                "classifier_score": 0.95,
                "patchcore_score": 0.05 if effective is ObjectCondition.VERIFIED else 0.95,
                "depth_residual_mm": 0.2,
                "image_ref": f"fleet://{item.instance_id}",
            },
        }


class ScenarioPoseProvider:
    def __init__(self, state: _PhysicalState) -> None:
        self._state = state

    def grasp_for(self, part: InventoryItem, *, observed_at_s: float) -> OrientedPartPose:
        del part, observed_at_s
        raise ControllerPortError("FoundationPose wrist evidence required for every part grasp")

    def grasp_from_foundationpose(
        self,
        part: InventoryItem,
        detection: dict[str, object],
        *,
        observed_at_s: float,
    ) -> OrientedPartPose:
        del observed_at_s
        if detection.get("instance_id") != part.instance_id or detection.get("object_id") != part.object_id:
            raise ControllerPortError("FoundationPose detection identity does not match reserved part")
        resting_side = detection.get("resting_side")
        if resting_side == "flipped":
            raise ControllerPortError(f"no approved grasp for resting side of {part.instance_id!r}")
        if resting_side not in {"upright", "side"}:
            raise ControllerPortError("FoundationPose detection resting side is unsupported")
        state = RestingState.FALLEN if resting_side == "side" else RestingState.INSTALLED
        return _oriented(_detection_pose(detection), part.object_id, state)

    def base_for(self, base: InventoryItem, *, observed_at_s: float) -> OrientedPartPose:
        del observed_at_s
        return _oriented(_observed_pose(self._state, base.instance_id), base.object_id)

    def target_for(
        self,
        product: ProductRecipe,
        base: InventoryItem,
        step: RecipeStep,
        *,
        observed_at_s: float,
    ) -> OrientedPartPose:
        del product, observed_at_s
        target = _offset_pose(_observed_pose(self._state, base.instance_id), step.insert_offset_mm)
        self._state.target_by_object[step.object_id] = target
        return _oriented(target, step.object_id)


class ScenarioRecoveryTargets:
    def __init__(self, scenario: Scenario, state: _PhysicalState) -> None:
        self._scenario = scenario
        self._state = state

    def staging_target_for(self, product: int, part: InventoryItem, slot: int) -> OrientedPartPose:
        del product
        low, high = self._scenario.bounds.staging_y_mm
        fraction = (slot % 10) / 10.0
        pose = (self._scenario.bounds.staging_x_mm, low + (high - low) * fraction, self._scenario.bounds.part_z_mm, 0.0)
        self._state.transfer_by_key[(part.instance_id, InventoryZone.STAGED)] = pose
        return _oriented(pose, part.object_id)

    def reject_target_for(self, part: InventoryItem) -> OrientedPartPose:
        pose = self._scenario.bounds.reject_pose
        self._state.transfer_by_key[(part.instance_id, InventoryZone.REJECT)] = pose
        return _oriented(pose, part.object_id)


class ScenarioRuntime:
    """Bind one generated scenario to perception, calibration, and recovery ports."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        state = _PhysicalState(
            {item.instance_id: item.pose for item in scenario.objects},
            {},
            {},
            {},
        )
        self.perception = ScriptedPerceptionPort(scenario, state)
        self.poses = ScenarioPoseProvider(state)
        self.recovery_targets = ScenarioRecoveryTargets(scenario, state)

    def observation(self, timestamp: float) -> Observation:
        home = self.scenario.bounds.home_pose
        return {
            "t": timestamp,
            "tcp_pose": list(home),
            "joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "images": {},
            "trial": {"instruction": self.scenario.instruction},
        }


class FleetRunner:
    def __init__(self, controller_factory: ControllerFactory, *, max_callbacks: int, dt_s: float) -> None:
        if isinstance(max_callbacks, bool) or not isinstance(max_callbacks, int) or max_callbacks < 1:
            raise FleetInputError("max_callbacks must be a positive integer")
        if isinstance(dt_s, bool) or not isinstance(dt_s, (int, float)) or not math.isfinite(dt_s) or dt_s <= 0:
            raise FleetInputError("dt_s must be finite and positive")
        self._controller_factory = controller_factory
        self._max_callbacks = max_callbacks
        self._dt_s = float(dt_s)
        self._catalog: Catalog = load_catalog()

    def run(self, scenarios: tuple[Scenario, ...]) -> FleetReport:
        return FleetReport(tuple(self._run_one(scenario) for scenario in scenarios))

    def _run_one(self, scenario: Scenario) -> ScenarioResult:
        runtime = ScenarioRuntime(scenario)
        observation = runtime.observation(0.0)
        instruction = _trial_instruction(observation)
        products = parse_instruction_products(instruction, self._catalog)
        if scenario.products != products:
            raise FleetInputError("scenario products disagree with trial instruction")
        controller = self._controller_factory(runtime)
        adapter = CellAdapter(controller, action_mode="tcp", dt=self._dt_s)
        status, callbacks, action_rows, error = _drive_callbacks(
            adapter,
            observation,
            controller,
            max_callbacks=self._max_callbacks,
            dt_s=self._dt_s,
        )
        return _result(scenario, instruction, products, controller, runtime, status, callbacks, action_rows, error)


def _drive_callbacks(
    adapter: CellAdapter,
    observation: Observation,
    controller: AssemblyController,
    *,
    max_callbacks: int,
    dt_s: float,
) -> tuple[ScenarioRunStatus, int, int, str | None]:
    action_rows = 0
    try:
        for callback in range(1, max_callbacks + 1):
            rows = _action_rows(adapter.act(observation))
            action_rows += len(rows)
            _advance_observation(observation, rows[-1], dt_s)
            status = _terminal_status(controller)
            if status is not None:
                return status, callback, action_rows, None
    except (ValueError, RuntimeError) as error:
        return ScenarioRunStatus.ERROR, callback, action_rows, str(error)
    return ScenarioRunStatus.CALLBACK_LIMIT, max_callbacks, action_rows, None


def _inspection_outcome(condition: ObjectCondition) -> tuple[str, str]:
    if condition is ObjectCondition.DEFECTIVE:
        return "broken", "reject"
    if condition is ObjectCondition.FALLEN:
        return "flipped", "reject"
    return "ok", "use"


def _observed_pose(state: _PhysicalState, instance_id: str) -> tuple[float, float, float, float]:
    try:
        return state.observed_pose_by_instance[instance_id]
    except KeyError as error:
        raise ControllerPortError(f"instance {instance_id!r} was not observed by perception") from error


def _detection_pose(detection: Mapping[str, object]) -> tuple[float, float, float, float]:
    pose = detection.get("pose")
    if not isinstance(pose, Mapping):
        raise ControllerPortError("FoundationPose detection pose must be a mapping")
    values = tuple(pose.get(field) for field in ("x", "y", "z", "yaw"))
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ControllerPortError("FoundationPose detection pose must contain numeric x, y, z, and yaw")
    numeric = tuple(float(value) for value in values)  # type: ignore[arg-type]
    if not all(math.isfinite(value) for value in numeric):
        raise ControllerPortError("FoundationPose detection pose must be finite")
    return numeric  # type: ignore[return-value]


def _oriented(
    pose: tuple[float, float, float, float],
    object_id: str,
    resting_state: RestingState = RestingState.INSTALLED,
) -> OrientedPartPose:
    x, y, z, yaw = pose
    tcp = AbsolutePoseUF0(x, y, z, -180.0, 0.0, yaw, "UF0", "calibrated")
    return OrientedPartPose(tcp, resting_state, f"installed:{object_id}")


def _offset_pose(
    base: tuple[float, float, float, float],
    offset: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    x, y, z, yaw = base
    angle = math.radians(yaw)
    dx, dy, dz = offset
    return (
        x + dx * math.cos(angle) - dy * math.sin(angle),
        y + dx * math.sin(angle) + dy * math.cos(angle),
        z + dz,
        yaw,
    )


def _action_rows(action: object) -> list[list[float]]:
    if not isinstance(action, list) or not action:
        raise FleetInputError("controller returned an empty or malformed action")
    rows = action if isinstance(action[0], list) else [action]
    if any(not isinstance(row, list) or len(row) != 7 for row in rows):
        raise FleetInputError("controller action rows must contain seven values")
    return rows  # type: ignore[return-value]


def _advance_observation(observation: Observation, row: list[float], dt_s: float) -> None:
    observation["tcp_pose"] = row[:6]
    observation["joints"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, row[6]]
    observation["t"] = float(observation["t"]) + dt_s


def _trial_instruction(observation: Observation) -> str:
    trial = observation.get("trial")
    if not isinstance(trial, Mapping):
        raise FleetInputError("observation trial must be a mapping")
    instruction = trial.get("instruction")
    if not isinstance(instruction, str):
        raise FleetInputError("observation trial instruction must be a string")
    return instruction


def _terminal_status(controller: AssemblyController) -> ScenarioRunStatus | None:
    if controller.finished:
        return ScenarioRunStatus.COMPLETE
    states = {
        PackMLState.HELD: ScenarioRunStatus.HELD,
        PackMLState.SUSPENDED: ScenarioRunStatus.SUSPENDED,
        PackMLState.ABORTED: ScenarioRunStatus.ABORTED,
    }
    return states.get(controller.state_machine.state)


def _result(
    scenario: Scenario,
    instruction: str,
    products: tuple[int, ...],
    controller: AssemblyController,
    runtime: ScenarioRuntime,
    status: ScenarioRunStatus,
    callbacks: int,
    action_rows: int,
    error: str | None = None,
) -> ScenarioResult:
    events = controller.events
    hazards = {item.instance_id for item in scenario.hazardous_objects}
    unrejected = hazards - runtime.perception.rejected_instances
    completed = status is ScenarioRunStatus.COMPLETE
    escapes = len(unrejected) if completed else None
    unresolved_hazards = 0 if completed else len(unrejected)
    return ScenarioResult(
        scenario.scenario_id,
        instruction,
        products,
        scenario.fault,
        status,
        controller.completed_products,
        callbacks,
        action_rows,
        sum(event.kind == "andon" for event in events),
        sum(event.kind == "retry" for event in events),
        escapes,
        unresolved_hazards,
        tuple(runtime.perception.physical_events),
        events,
        error,
    )
