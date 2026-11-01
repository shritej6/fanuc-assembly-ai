# Operator interface

Authored by Shritej Gunda.

The operator layer compiles a deliberately small set of plain-language phrases into typed controller task requests. It does
not choose products, create waypoints, call a robot adapter, interpret arbitrary instructions, or use a language model
to guess intent.

## Approved commands and product authority

- `status`, `show status`, `show cell status`, `what is the cell status?`
- `assemble everything on the table`, `assemble all products on the table`, `build everything on the table`
- `assemble product 1` through `assemble product 6` (`build product N` is also accepted)

Everything else is rejected. Free-form TCP commands, stop/resume commands, multiple-product phrases, extra clauses,
control characters, and out-of-range product IDs never reach the task port.

The command is operator intent, not product authority. Every submission obtains a fresh, typed
`TrialInstructionEvidence` from the reviewed observation boundary and parses only its
`obs["trial"]["instruction"]` text with the controller's catalog-aware parser. The evidence carries run, rollout, observation,
capture/expiry times, and source. Missing, untyped, stale, or future evidence holds the task. An explicit one-product
command must exactly equal the prompt-derived sequence; an "everything" command uses the prompt-derived sequence
without a visible-product or fixed-location override.

## Runtime boundary

`T09TaskCoordinator` accepts a source, assembly controller factory, trial-instruction provider, trusted final-build evidence
provider, validated catalog, and optional clock/task-ID factories. It never calls `act()`. For mock, sim, and replay it
verifies the controller's `requested_products`: empty is permitted before the first observation binds the prompt, while any bound
sequence must exactly equal the coordinator's fresh prompt evidence.

`ControllerSource.CELL` is intentionally unavailable. The operator interface has no reviewed runtime capability that physically inhibits
`act()` between mechanically completed products. The `execution_authorized` report field is diagnostic only and cannot
be used as a safety gate. A real-cell integration must inject a reviewed capability that owns the act loop and enforces
the pause; a boolean, source label, or ordinary mock cannot claim that authority.

Live reports contain:

- the exact PackML state record and active fault;
- prompt-selected and mechanically completed products;
- current controller plan-step ID and event count;
- explicit mock/sim/replay source and report timestamp;
- controller completion, final-check decisions, quality-release state, and diagnostic `execution_authorized`.

The read boundary enforces the controller's PackML ownership invariant: Idle has no product, every non-Idle state owns one selected
product, and a controller-state timestamp cannot be later than the operator report clock.

## Trusted final-build evidence

The public operator method accepts only a product ID. It asks the task port to prepare evidence; callers cannot submit
installed IDs, timestamps, image paths, or a prebuilt `FinalBuildRequest`.

The trusted provider receives a query containing the active run, rollout, task, product, latest mechanical timestamp,
and recipe object IDs derived from the validated catalog. It must return immutable evidence containing:

- the same run, rollout, task, and product;
- unique physical installed-instance lineage mapped to the exact catalog recipe order;
- an image capture ID and SHA-256 digest, not a caller-provided URI;
- capture and expiry times no older than the mechanical state; and
- provider provenance.

The coordinator owns prepare, then `FinalBuildGate`, then record under one locked operation. It exposes no public prepare or
record method, verifies identity again before recording, rejects reused request/image evidence, and rechecks expiry. A
caller therefore cannot substitute a later run, rollout, product, capture, digest, or physical instance, or fabricate a
label-only `done` decision without invoking the configured Cosmos port.

`FinalBuildGate` performs the two independent checks in order:

1. deterministic recipe check against the validated catalog (there is no `M<product>_NN` naming formula authority);
2. an injected Cosmos Reason port checks the same captured image identity and digest.

Recipe mismatch returns `rebuild` without calling Cosmos. Cosmos rejection returns `rebuild`. Missing, malformed,
mismatched, substituted-model, or unexpected evidence returns `hold_andon`. `done` requires both checks. The included
deterministic stub is explicitly labelled `stub` and can never quality-release a product, even when configured true.
The operator layer supplies an interface, not a deployed Cosmos endpoint; production provenance must come from a separately reviewed
port.

## Integration shape

```python
catalog = load_catalog()
tasks = T09TaskCoordinator(
    source=ControllerSource.MOCK,
    controller_factory=build_mock_t09_controller,
    trial_instruction=trusted_trial_port.capture,
    final_evidence=trusted_final_build_port,
    final_gate=FinalBuildGate(reviewed_cosmos_port, catalog),
    catalog=catalog,
)
operator = OperatorAgent(tasks)
```

The names prefixed `trusted_` and `reviewed_` above are integration capabilities, not implementations supplied by the operator interface.
Omitting the final-build provider holds quality release. Selecting `CELL` fails closed until the physical act-gate
contract is available and reviewed.

Run focused verification with:

```bash
uv run pytest -q tests/test_operator_commands.py tests/test_cosmos_reason.py \
  tests/test_operator_agent.py tests/test_operator_authority.py \
  tests/test_operator_t09_compatibility.py tests/test_operator_prepared_decision.py
```
