# T17 seeded scenario fleet

`eval.scenario_fleet` generates deterministic, randomized three-product prompts and table layouts from the validated
catalog. It does not read `placement.yaml`, choose a fixed product trio, or pass simulator ground truth to the
controller. Each generated object enters T09 through `ScriptedPerceptionPort`; calibrated mock poses remain unavailable
until a scan has observed them.

The caller supplies mock bounds, recovery targets, and a controller factory. That keeps cell-specific coordinates and
safety limits out of the fleet. `FleetRunner` follows each guarded TCP action as the official mock does and returns a
`FleetReport`. `FleetReport.as_dict()` contains only counters and ratios measured during those runs: requested and
completed products, final yield, action rows, assists, retries, escapes, and per-status/per-fault counts.

Fault injection covers clean runs, detected defects, fallen parts, look-alike traps, parts foreign to the requested
products, and exact-part spare shortages. Fallen cases include both approved and unavailable resting-side grasps; an
unavailable grasp must remain fail-closed.

Run the regression fleet with:

```bash
uv run pytest -q tests/test_scenario_fleet.py
```

The scripted evidence is a contract fixture, not model evaluation or real-cell calibration. It must never be selected
as the `runtime="cell"` perception provider.

