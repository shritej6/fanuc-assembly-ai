# T21 operator interface

The operator layer compiles a deliberately small set of plain-language phrases into typed T09 task requests. It does
not create waypoints, call a robot adapter, interpret arbitrary instructions, or use a language model to guess intent.

## Approved commands

- `status`, `show status`, `show cell status`, `what is the cell status?`
- `assemble everything on the table`, `assemble all products on the table`, `build everything on the table`
- `assemble product 1` through `assemble product 6` (`build product N` is also accepted)

Everything else is rejected. In particular, free-form TCP commands, stop/resume commands, multiple-product phrases,
extra clauses, control characters, and out-of-range product IDs cannot reach the task port.

`assemble everything on the table` resolves a fresh, injected visible-product provider. This provider is where the
reviewed detector/grouping output will connect; it must return a non-empty unique tuple of product IDs. Explicit
single-product commands do not use that scan.

## Runtime boundary

`T09TaskCoordinator` accepts four injected boundaries: the source label (`mock`, `sim`, `replay`, or `cell`), a T09
controller factory, a visible-product provider, and optional clock/task-ID factories. It creates a T09 controller but
never calls `act()`. The official runtime retains physical authority and consumes the controller through its existing
guarded adapter path.

Live reports contain:

- the exact PackML state record and active fault;
- selected and mechanically completed products;
- current T09 plan-step ID and event count;
- explicit data source and report timestamp;
- controller completion, final-check decisions, quality-release state, and `execution_authorized`.

Each final-check decision preserves the active task ID, evidence request ID and observation time, recipe result, Cosmos
result, exact evidence source, model ID, and explanatory detail. A decision for another task is rejected so a stub,
replay, or unavailable endpoint cannot be mistaken for current production Cosmos evidence.

The runtime must stop requesting subsequent physical actions whenever `execution_authorized` is false. T09 currently
records mechanical completion independently; production wiring must add a pause point before automatic advancement to
the next product. Until that reviewed hook and the real Cosmos endpoint land, this interface is mock/replay-ready but
not a real-cell quality-release authorization.

## Final build check

`FinalBuildGate` implements the required two independent checks in order:

1. deterministic recipe check: installed IDs must be exactly `M<product>_01` through `_04` in order;
2. Cosmos Reason 2 (2B): an injected endpoint checks the current image bundle.

Recipe mismatch returns `rebuild` without calling Cosmos. A Cosmos rejection returns `rebuild`. Missing, malformed,
stale, mismatched, substituted-model, or unexpected endpoint evidence returns `hold_andon`. `done` requires both checks.
The included deterministic stub is explicitly labelled `stub` and can never quality-release a product, even when its
configured answer is true.

## Minimal mock wiring

```python
from agent import ControllerSource, FinalBuildGate, OperatorAgent, T09TaskCoordinator
from agent.cosmos_reason import DeterministicCosmosReasonStub

tasks = T09TaskCoordinator(
    source=ControllerSource.MOCK,
    controller_factory=build_mock_t09_controller,
    visible_products=lambda: (1, 3, 4),
)
operator = OperatorAgent(tasks, FinalBuildGate(DeterministicCosmosReasonStub({})))
response = operator.handle("assemble everything on the table")
```

Run the focused verification with:

```bash
uv run pytest -q tests/test_operator_commands.py tests/test_cosmos_reason.py tests/test_operator_agent.py
```
