"""Focused runtime behavior using the shared controller scenario fixture."""

from collections import Counter

from robot.adapter import CellAdapter
from tests.test_controller import ScriptedPerception, _controller, _observation, _observed_three_products


def test_connector_step_applies_catalog_gripper_roll() -> None:
    observed = tuple(
        item for item in _observed_three_products() if item.object_id.startswith("M4") or item.object_id == "F04"
    )
    perception = ScriptedPerception(observed, set(), Counter(), set())
    controller = _controller(perception, products=(4,))

    first = CellAdapter(controller, action_mode="tcp", dt=0.1).act(_observation())
    rows = first if isinstance(first[0], list) else [first]

    assert rows[-1][5] == 100.0


def test_collision_scene_is_refreshed_for_every_emitted_chunk() -> None:
    observed = tuple(
        item for item in _observed_three_products() if item.object_id.startswith("M1") or item.object_id == "F01"
    )
    perception = ScriptedPerception(observed, set(), Counter(), set())
    controller = _controller(perception, products=(1,))
    adapter = CellAdapter(controller, action_mode="tcp", dt=0.1)
    observation = _observation()
    first = adapter.act(observation)
    first_rows = first if isinstance(first[0], list) else [first]
    first_count = perception.collision_calls

    observation["tcp_pose"] = first_rows[-1][:6]
    observation["joints"] = [0.0] * 6 + [first_rows[-1][6]]
    observation["t"] = 1.0
    adapter.act(observation)

    assert first_count == 2
    assert perception.collision_calls == 3
