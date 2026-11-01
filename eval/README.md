# Seeded scenario fleet

`eval.scenario_fleet` generates deterministic, randomized three-product prompts and table layouts from the validated
catalog. It does not read `placement.yaml`, choose a fixed product trio, or pass simulator ground truth to the
controller. Each generated object enters the planner through `ScriptedPerceptionPort`; calibrated mock poses remain unavailable
until a scan has observed them. A 20-scenario run covers every unordered three-of-six product combination exactly once;
a 60-scenario run covers each combination three times. Seeded shuffling changes order, faults, and XY layouts without
changing that coverage.

The caller supplies mock bounds, recovery targets, and a controller factory. That keeps cell-specific coordinates and
safety limits out of the fleet. `FleetRunner` follows each guarded TCP action as the official mock does and returns a
`FleetReport`. Requested products and KPI denominators come only from `obs["trial"]["instruction"]`; contradictory
scenario metadata is rejected. `FleetReport.as_dict()` contains only counters and ratios measured during those runs:
requested and completed products, final yield, action rows, assists, retries, escapes, unresolved hazards, and
per-status/per-fault counts. Escape totals are `null` and `acceptance_ready` is false whenever any run is Held,
Suspended, Aborted, callback-limited, or errored; `known_escapes` and `unresolved_hazards` remain separately reported.

Fault injection covers clean runs, detected defects, fallen parts, look-alike traps, parts foreign to the requested
products, and exact-part spare shortages. Fallen cases include both approved and unavailable resting-side grasps; an
approved grasp must produce a verified physical reject before spare selection, while an unavailable grasp must enter
Held without reject motion. Mock grasp commands are derived from the FoundationPose-shaped detection payload rather
than hidden scenario state.

Run the regression fleet with:

```bash
uv run pytest -q tests/test_scenario_fleet.py
```

The scripted evidence is a contract fixture, not model evaluation or real-cell calibration. It must never be selected
as the `runtime="cell"` perception provider.
