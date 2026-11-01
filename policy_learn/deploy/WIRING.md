# Planner to GR00T physical-cell wiring

This is the reviewed runtime boundary. It is intentionally fail-closed:
the mock controller remains available for software tests, while `runtime="cell"` cannot
emit motion until every physical dependency is present and reviewed.

## Runtime path

```text
official observation
  -> planner state machine
  -> identity-bound MotionIntent
  -> T09GrootExecutor
  -> GrootPolicy (one fresh row)
  -> learned-policy ChunkGuard
  -> live-scene SafetySupervisor
  -> CellAdapter
  -> official FANUC client
```

There is no scripted or joint-hold fallback in CELL mode. A missing dependency raises
`CellRuntimeBlockedError`; it does not select the mock path.

## Semantic contract

The deployed checkpoint must attest these skills:

- `inspect_view`
- `pick_insert_hold`
- `release_retreat`
- `pull_out`
- `place_free`
- `defect_drop`

`pick_insert_hold` must finish with the gripper closed. The planner performs its independent
post-place verification before issuing `release_retreat`. The policy consumes exactly
one action row per fresh observation, so no prefetched trajectory can cross that gate.

## Checkpoint identity

`policy_learn/groot/serve_policy.sh` serves GR00T through the attested proxy. The proxy:

1. hashes the checkpoint tree selected for launch;
2. compares the hash with a reviewed deployment manifest;
3. derives `server_verified=true`; and
4. exposes the versioned `describe` endpoint.

That is necessary but not sufficient for physical authorization. The current proxy
serves a static envelope and forwards to the upstream port; it does not authenticate
each response as coming from the original child process, and the upstream checkpoint
directory is not sealed read-only after verification. Before CELL can be armed, the
launcher must supervise and authenticate the exact child process and serve from an
immutable content-addressed checkpoint (or an equivalently reviewed lifecycle).

The CELL client also checks the dataset, catalog, reviewed grasp source, exact guard
calibration, RGB model interface, embodiment, 5 Hz timing, one-row horizon, evaluation
approval, and release protocol. Table/wrist depth remains an independent planner perception
dependency; the GR00T wire format does not claim to consume it. A sidecar file cannot
self-assert server verification. `APPROVED_DEPLOYMENT_PINS` is deliberately empty on
this branch; a real checkpoint cannot be selected until a dual-reviewed manifest digest
is added to that registry.

## Physical authorization gates

`assembly.cell_runtime.runtime_blockers()` is the canonical list. At minimum, CELL
requires:

- a planner semantic queue that preserves the hold-before-release protocol;
- a checkpoint with all six attested skills and reviewed evaluation;
- measured guard limits with an immutable source;
- live perception and collision-scene backends with reviewed artifacts; and
- reviewed full IK, joint limits, robot-link sweeps, and held-part sweeps.

Production limits must be loaded with `GuardLimits.from_calibration(path, sha256)`.
Setting `status="MEASURED"` in Python is not sufficient: the loader binds every numeric
limit to the exact reviewed calibration-report bytes.

`SDGReachabilityChecker` and `CADCollisionChecker` are preflight diagnostics only. They
intentionally advertise incomplete production capabilities, so `SafetySupervisor`
rejects them for physical authorization. A future CELL provider must seal the complete
transitive physical sources (solver, joint limits, robot links, UTOOL, fingers, held
part geometry, and validation evidence) before those capability flags can be true.

The current branch implements and tests these boundaries but deliberately does not
invent the missing measurements, physical providers, or model capabilities. Therefore
`create_cell_policy()` still raises `CellRuntimeBlockedError` and produces no real-cell
action.

The current `MotionIntent.release_authorized` flag exercises the executor protocol in
software tests only. Before the CELL gate can open, the planner must replace it with a one-shot
controller-issued token bound to trial, step, part/base identities, fresh observation
time, release TCP, and independent inspection evidence.

## Safe verification

```bash
# Software-only official-client loop
cd robot/cell
uv run run.py --mock --trials 1 --trial-s 0.5

# Check an attestation without starting a server
PYTHONPATH=. python -m policy_learn.groot.attested_server \
  --checkpoint /reviewed/checkpoint \
  --attestation /reviewed/attestation.json \
  --verify-only
```

Do not run `runtime="cell"` until the factory reports no blockers and the CSI physical
readiness review has approved the measured sources.
