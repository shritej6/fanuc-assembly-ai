"""The official CELL entrypoint stays unarmed until the semantic stack is complete."""

from __future__ import annotations

import pytest

from assembly.cell_runtime import (
    CellRuntimeBlockedError,
    create_cell_policy,
    runtime_blockers,
)


def test_runtime_blockers_name_each_missing_physical_authority() -> None:
    blockers = runtime_blockers()
    codes = {blocker.code for blocker in blockers}

    assert codes == {
        "semantic_t09_executor",
        "served_checkpoint_attestation",
        "measured_motion_limits",
        "live_perception_backends",
        "reviewed_physical_safety",
    }
    assert all(blocker.owner and blocker.detail for blocker in blockers)


@pytest.mark.parametrize("checkpoint", [None, "stub", "groot://127.0.0.1:5555"])
def test_cell_factory_never_falls_back_while_runtime_is_unarmed(checkpoint: str | None) -> None:
    with pytest.raises(CellRuntimeBlockedError, match="CELL runtime is not armed") as captured:
        create_cell_policy(checkpoint=checkpoint)

    message = str(captured.value)
    assert "semantic_t09_executor" in message
    assert "reviewed_physical_safety" in message
    assert "joint hold" not in message.lower()
