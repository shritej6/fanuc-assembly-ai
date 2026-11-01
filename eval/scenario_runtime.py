"""Scripted perception boundary and mock action follower for scenario runs.
Authored by Shritej Gunda."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Protocol

from assembly.catalog import Catalog, load_catalog
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
from assembly.state_machine import PackMLState
from eval.scenario_data import FleetInputError, Scenario, ScenarioObject, parse_instruction_products
from eval.scenario_pose_runtime import (
    ScenarioPoseProvider,
    ScenarioPoseState,
    ScenarioRecoveryTargets,
    observed_pose,
)
from eval.scenario_report import (
    FleetReport,
    ScenarioPhysicalEvent,
    ScenarioPhysicalEventKind,
    ScenarioResult,
    ScenarioRunStatus,
)
from robot.adapter import CellAdapter, Observation


class ControllerFactory(Protocol):
    """Callable that builds a controller for one scenario."""
    def __call__(self, runtime: ScenarioRuntime) -> AssemblyController: ...


class ScriptedPerceptionPort:
    """Perception port that returns evidence from the scripted cameras only."""

    def __init__(self, scenario: Scenario, state: ScenarioPoseState) -> None:
        self._scenario = scenario
        self._state = state
        self._objects = {item.instance_id: item for item in scenario.objects}
        self.rejected_instances: set[str] = set()
        self.physical_events: list[ScenarioPhysicalEvent] = []

    def scan(self, observation: Observation) -> InventorySnapshot:
        """Return the scripted scan evidence."""
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
        """Return the scripted evidence before a pick."""
        self.physical_events.append(ScenarioPhysicalEvent(ScenarioPhysicalEventKind.PRE_PICK, part.instance_id))
        scripted = self._scripted(part.instance_id)
        return EvidencePair(
            self._detection(scripted),
            self._inspection(scripted),
            float(observation["t"]),
        )

    def post_place(self, observation: Observation, part: InventoryItem) -> PlacementEvidence:
        """Return the scripted evidence after a place."""
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
        """Return the scripted evidence after a transfer."""
        key = (part.instance_id, target_zone)
        expected = self._state.transfer_by_key[key]
        actual = self._state.transfer_result_by_key.get(key, expected)
        self._state.pose_by_instance[part.instance_id] = actual
        agrees = actual == expected
        observed_zone = target_zone if agrees else InventoryZone.TABLE
        if observed_zone is InventoryZone.REJECT:
            self.rejected_instances.add(part.instance_id)
            self.physical_events.append(
                ScenarioPhysicalEvent(ScenarioPhysicalEventKind.REJECT_TRANSFER, part.instance_id)
            )
        scripted = self._scripted(part.instance_id)
        return RecoveryTransferEvidence(
            self._detection(scripted),
            self._inspection(scripted),
            float(observation["t"]),
            observed_zone,
            agrees,
        )

    def collision_scene(
        self,
        observation: Observation,
        moving_instance_id: str,
        base_instance_id: str,
    ) -> CollisionScene:
        """Return the collision scene for the current state."""
        base_pose = observed_pose(self._state, base_instance_id)
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


class ScenarioRuntime:
    """Bind one generated scenario to perception, calibration and recovery ports."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        state = ScenarioPoseState(
            {item.instance_id: item.pose for item in scenario.objects},
            {},
            {},
            {},
            {},
        )
        self._state = state
        self.perception = ScriptedPerceptionPort(scenario, state)
        self.poses = ScenarioPoseProvider(state, scenario.bounds.tcp_local_mm)
        self.recovery_targets = ScenarioRecoveryTargets(scenario, state)

    def script_transfer_observation(
        self,
        instance_id: str,
        zone: InventoryZone,
        pose: tuple[float, float, float, float],
    ) -> None:
        """Script the observation seen after a transfer."""
        if instance_id not in self._state.pose_by_instance:
            raise FleetInputError(f"unknown transfer instance {instance_id!r}")
        if zone not in {InventoryZone.REJECT, InventoryZone.STAGED}:
            raise FleetInputError("scripted transfer zone must be reject or staged")
        values = _finite_pose(pose, "scripted transfer pose")
        self._state.transfer_result_by_key[(instance_id, zone)] = values

    def observation(self, timestamp: float) -> Observation:
        """Return the current scripted observation."""
        home = self.scenario.bounds.home_pose
        return {
            "t": timestamp,
            "tcp_pose": list(home),
            "joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "images": {},
            "trial": {"instruction": self.scenario.instruction},
        }


class FleetRunner:
    """Run a fleet of scenarios and collect the report."""
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
        """Run every scenario and return the fleet report."""
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


def _finite_pose(value: object, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, tuple) or len(value) != 4:
        raise FleetInputError(f"{label} must contain x, y, z, and yaw")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise FleetInputError(f"{label} must contain numeric x, y, z, and yaw")
    numeric = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in numeric):
        raise FleetInputError(f"{label} must be finite")
    return numeric  # type: ignore[return-value]


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
