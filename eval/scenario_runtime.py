"""Scripted perception boundary and mock action follower for T17 scenarios."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from assembly.catalog import Catalog, ProductRecipe, RecipeStep, load_catalog
from assembly.controller import AssemblyController
from assembly.controller_ports import (
    CollisionScene,
    ControllerEvent,
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
from eval.scenario_data import FaultKind, FleetInputError, Scenario, ScenarioObject, parse_instruction_products
from robot.adapter import CellAdapter, Observation


class ControllerFactory(Protocol):
    def __call__(self, runtime: ScenarioRuntime, products: tuple[int, ...]) -> AssemblyController: ...


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

    def scan(self, observation: Observation) -> InventorySnapshot:
        self._state.observed_pose_by_instance = dict(self._state.pose_by_instance)
        items = tuple(
            ObservedInventoryItem(item.instance_id, item.object_id, item.zone, item.condition)
            for item in self._scenario.objects
        )
        return InventorySnapshot(float(observation["t"]), True, items)

    def pre_pick(self, observation: Observation, part: InventoryItem) -> EvidencePair:
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
        resting = "side" if effective is ObjectCondition.FALLEN else "upright"
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
    def __init__(self, scenario: Scenario, state: _PhysicalState) -> None:
        self._scenario = scenario
        self._state = state
        self._objects = {item.instance_id: item for item in scenario.objects}

    def grasp_for(self, part: InventoryItem, *, observed_at_s: float) -> OrientedPartPose:
        del observed_at_s
        item = self._objects[part.instance_id]
        if not item.grasp_approved:
            raise ControllerPortError(f"no approved grasp for resting side of {part.instance_id!r}")
        state = RestingState.FALLEN if item.condition is ObjectCondition.FALLEN else RestingState.INSTALLED
        return _oriented(_observed_pose(self._state, part.instance_id), part.object_id, state)

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
        self.poses = ScenarioPoseProvider(scenario, state)
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


class ScenarioRunStatus(StrEnum):
    COMPLETE = "complete"
    HELD = "held"
    SUSPENDED = "suspended"
    ABORTED = "aborted"
    CALLBACK_LIMIT = "callback_limit"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    scenario_id: str
    instruction: str
    products: tuple[int, ...]
    fault: FaultKind
    status: ScenarioRunStatus
    completed_products: tuple[int, ...]
    callbacks: int
    action_rows: int
    assists: int
    retries: int
    escapes: int
    events: tuple[ControllerEvent, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FleetReport:
    results: tuple[ScenarioResult, ...]

    @property
    def scenario_count(self) -> int:
        return len(self.results)

    @property
    def requested_products(self) -> int:
        return sum(len(result.products) for result in self.results)

    @property
    def completed_products(self) -> int:
        return sum(len(result.completed_products) for result in self.results)

    @property
    def action_rows(self) -> int:
        return sum(result.action_rows for result in self.results)

    @property
    def assists(self) -> int:
        return sum(result.assists for result in self.results)

    @property
    def retries(self) -> int:
        return sum(result.retries for result in self.results)

    @property
    def escapes(self) -> int:
        return sum(result.escapes for result in self.results)

    @property
    def final_yield(self) -> float:
        return self.completed_products / self.requested_products if self.requested_products else 0.0

    @property
    def builds_per_assist(self) -> float | None:
        return None if self.assists == 0 else self.completed_products / self.assists

    def as_dict(self) -> dict[str, object]:
        """Serialize only counters and ratios measured by completed harness runs."""
        return {
            "scenario_count": self.scenario_count,
            "requested_products": self.requested_products,
            "completed_products": self.completed_products,
            "final_yield": self.final_yield,
            "action_rows": self.action_rows,
            "assists": self.assists,
            "builds_per_assist": self.builds_per_assist,
            "retries": self.retries,
            "escapes": self.escapes,
            "status_counts": dict(Counter(result.status.value for result in self.results)),
            "fault_counts": dict(Counter(result.fault.value for result in self.results)),
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
        products = parse_instruction_products(scenario.instruction, self._catalog)
        controller = self._controller_factory(runtime, products)
        adapter = CellAdapter(controller, action_mode="tcp", dt=self._dt_s)
        observation = runtime.observation(0.0)
        action_rows = 0
        try:
            for callback in range(1, self._max_callbacks + 1):
                output = adapter.act(observation)
                rows = _action_rows(output)
                action_rows += len(rows)
                _advance_observation(observation, rows[-1], self._dt_s)
                status = _terminal_status(controller)
                if status is not None:
                    return _result(scenario, controller, runtime, status, callback, action_rows)
        except (ValueError, RuntimeError) as error:
            return _result(scenario, controller, runtime, ScenarioRunStatus.ERROR, callback, action_rows, str(error))
        return _result(scenario, controller, runtime, ScenarioRunStatus.CALLBACK_LIMIT, self._max_callbacks, action_rows)


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
    controller: AssemblyController,
    runtime: ScenarioRuntime,
    status: ScenarioRunStatus,
    callbacks: int,
    action_rows: int,
    error: str | None = None,
) -> ScenarioResult:
    events = controller.events
    hazards = {item.instance_id for item in scenario.hazardous_objects}
    escaped = hazards - runtime.perception.rejected_instances if controller.finished else set()
    return ScenarioResult(
        scenario.scenario_id,
        scenario.instruction,
        scenario.products,
        scenario.fault,
        status,
        controller.completed_products,
        callbacks,
        action_rows,
        sum(event.kind == "andon" for event in events),
        sum(event.kind == "retry" for event in events),
        len(escaped),
        events,
        error,
    )

