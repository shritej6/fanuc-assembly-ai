# Brief for Codex: T35 Cosmos Reason 2 8B (2026-10-30, from Claude; the user approved T35)

If another branch already has this file, merge this section into it.

## Pin the final-build gate to Cosmos Reason 2 8B (T21 files, Codex-owned)

The user chose Cosmos Reason 2 **8B** for the build check on 2026-10-28 (`coord/DECISIONS.md`, commit 1419d23 on
the T11 line; `docs/PLAN.md` section 4). T21 was branched before that and pins **2B** as the only production
provenance, so the real 8B endpoint (`agent/cosmos_reason2_endpoint.py`, this branch) can never release a product:
every 8B pass is turned into Hold/Andon ("production evidence must come from Cosmos Reason 2 (2B)"). Verified on
this branch: `FinalBuildGate(CosmosReason2Endpoint(...)).evaluate(...)` with an 8B pass returns `hold_andon`.

Requested change (no check weakened: still exactly one accepted production model ID):

1. `agent/cosmos_reason.py` `_validate_result` (line 241-242):
   `result.model_id != "Cosmos-Reason2-2B"` → `!= "Cosmos-Reason2-8B"`, message "… Cosmos Reason 2 (8B)".
   Better: import `MODEL_ID` from `agent/cosmos_reason2_endpoint.py` (or move one constant `COSMOS_REASON_2_MODEL_ID =
   "Cosmos-Reason2-8B"` into `agent/cosmos_reason.py` and have the endpoint import it) so the two can never drift again.
2. `agent/task_validation.py` `_validate_cosmos_provenance` (lines 162-167): same constant, messages "(8B)".
3. Tests that build production results with `"Cosmos-Reason2-2B"`: `tests/test_cosmos_reason.py` (lines 60, 94),
   `tests/test_operator_agent.py` (57), `tests/test_operator_authority.py` (106),
   `tests/test_operator_prepared_decision.py` (65) → `"Cosmos-Reason2-8B"`. Add one regression test that a
   `Cosmos-Reason2-2B` result is now rejected (held), so a stale 2B server cannot release a product.
4. Then delete the `xfail` mark on `test_gate_releases_a_recipe_complete_build_on_8b_pass_evidence` in
   `tests/test_cosmos_reason2_endpoint.py` (it is `strict=True`, so it turns red as soon as the pin is fixed).
5. Text: `docs/OPERATOR_INTERFACE.md` and the T21 `coord/HANDOFF.md` entries say `Cosmos-Reason2-2B`; new entries
   should say 8B.

## Wiring the real port (when the act gate and evidence provider exist)

`CosmosReason2Endpoint(EndpointConfig(base_url=...), images, catalog)` implements `CosmosReasonPort`. `images` must be
the trusted capture store: `image_bytes(capture_id) -> bytes` (JPEG or PNG) for the capture the final-evidence provider
recorded; the port re-checks its SHA-256 against `FinalBuildRequest.image_sha256` before sending anything. It raises
only `CosmosReasonUnavailable` for unverifiable cases (bad digest, unreachable after 3 tries, truncated / unparseable /
wrong-model answer), which the gate already maps to Hold/Andon. The prompt names the product, the recipe parts from
`installed_object_ids` and asks about the top part; please send the capture of the finished product cropped to that
product (the bench crops set D the same way). If the planner takes the "2-angle close-up" of a finished product,
call `check` once per angle with its own request and require both to pass.

## Not a mismatch

`policy_learn/groot/pins.env` `BACKBONE_ID=nvidia/Cosmos-Reason2-2B` is GR00T N1.7's own VLM backbone, not the
build check. Leave it.
