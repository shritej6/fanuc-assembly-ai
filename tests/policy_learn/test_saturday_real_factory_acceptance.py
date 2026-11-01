"""Acceptance proof for the pinned real-cell factory over mock transport.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import numpy as np
import pytest

from assembly import cell_runtime
from assembly.cell_runtime import CellTransport
from assembly.fixed_inspection import FixedInspectionMove
from assembly.state_machine import PackMLState
from robot.cell.policy import Policy
from robot.cell.real_layout import load_real_cell_layout
from robot.cell.saturday_runtime_adapters import RUN_DIR_ENV, UNVERIFIED_PRESENCE_ENV
from robot.cell.t09_real_runtime import build_groot_real_controller
from tests.policy_learn.test_fake_groot_cell_e2e import (
    CELL_DIR,
    _FakeGrootServer,
    _MockCellServer,
)


def test_pinned_real_factory_queues_product_five_inspection_over_official_mock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Explicit unverified presence must reach M5_01's guarded fixed inspection."""
    original_validate = cell_runtime._validate_controller
    monkeypatch.setenv(UNVERIFIED_PRESENCE_ENV, "1")
    monkeypatch.setenv(RUN_DIR_ENV, str(tmp_path))
    monkeypatch.syspath_prepend(str(CELL_DIR))
    monkeypatch.setattr(
        cell_runtime,
        "_resolve_factory",
        lambda transport: build_groot_real_controller,
    )
    monkeypatch.setattr(
        cell_runtime,
        "_validate_controller",
        lambda controller, executor, transport: original_validate(
            controller,
            executor,
            CellTransport.REAL,
        ),
    )
    rollout = load_real_cell_layout().rollouts[0]

    with _FakeGrootServer() as groot, _MockCellServer(trial_s=10.0) as cell:
        policy = Policy(
            runtime="cell",
            checkpoint=groot.url,
            cell_transport=CellTransport.OFFICIAL_MOCK,
        )
        try:
            trial = cell.ready_trial()
            with cell.mock.lock:
                cell.mock.trials[0]["instruction"] = rollout.prompt
            client = cell.client
            client.start(trial["trial_id"])
            policy.reset()
            observation = client.observe(
                policy.CAMERAS,
                policy.IMAGE_SIZE,
                policy.IMAGE_FORMAT,
                depth=policy.DEPTH,
            )

            action = policy.act(observation)
            controller = policy.controller
            record = controller.state_machine.current_record
            assert record.state is not PackMLState.HELD, record.reason

            queue = controller._motion_queue
            move = queue._intent
            assert isinstance(move, FixedInspectionMove)
            assert controller.inventory.item(move.instance_id).object_id == "M5_01"
            assert controller._coarse_xy_by_instance[move.instance_id] == (358.0, 193.0)
            assert move.slot_id == "P1"
            assert move.endpoint_tcp == (358.0, 193.0, -168.0, -180.0, 0.0, 180.0)
            assert queue.active is True
            assert queue.last_authority_label == "non_learned_inspection"
            assert float(observation["joints"][6]) == 0.0
            assert float(np.asarray(action, dtype=float)[-1]) == 0.0
            assert "get_action" not in [request["endpoint"] for request in groot.requests]

            rows = np.atleast_2d(np.asarray(action, dtype=float))
            client.act_tcp(rows, policy.DT)
            assert cell.mock.rejections == []
        finally:
            policy.close()
