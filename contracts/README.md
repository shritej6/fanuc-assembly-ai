# Contracts

Shared JSON Schemas (draft 2020-12) that Claude- and Codex-owned code both read and write.
Change them only through a PR reviewed by both agents. All poses use the controller frame **UF0** in mm/deg
(the same frame as `obs["tcp_pose"]` and `"tcp"` actions; table surface at z ~ -335 mm).

| File | Used for |
|---|---|
| `common.schema.json` | shared ids, families, base types, UF0 pose |
| `taxonomy.schema.json` | the 30 objects, families, look-alike groups |
| `detection.schema.json` | perception output per object instance |
| `product_group.schema.json` | grouping output per product |
| `inspection.schema.json` | inspection verdict + action per part |
| `plan_step.schema.json` | state machine → motion layer |
| `packml_state.schema.json` | cell state (PackML) |
| `birth_certificate.schema.json` | traceability record per product |
| `catalog.schema.json` | product recipe with CAD fixture-frame installed transforms, T29 grasp-roll offsets, and hash-pinned source provenance (filled in T04) |

## Error codes (`common.schema.json#/$defs/error_code`)

Adopted from the Team V spec (user decision 2026-10-28). Inspection lists every code that applies
(`error_codes`); a plan step says which code it answers (`error_code`). The arm picks and places straight down
only, so a part that is not standing as installed is never reoriented: it goes to reject and a spare is taken.

| Code | Meaning |
|---|---|
| `WRONG_MODEL` | part belongs to a different product model than the work order asks for |
| `WRONG_PART` | right model, wrong part for this assembly stage |
| `WRONG_ORIENTATION` | part is not standing in its installed orientation (lying or upside down); the arm cannot reorient it |
| `WRONG_FRONT_BACK` | front/rear (yaw) of the part or target does not match the fixture's known front/rear axis |
| `LOOKALIKE_UNRESOLVED` | fewer than two independent cues separate the part from its look-alikes |
| `GUARD_TOPPLED` | a guard (M*_04) has fallen over; no tested recovery, take a spare |
| `GRIP_FEATURE_INACCESSIBLE` | no validated grasp envelope is free on the part |
| `HOUSING_NOT_SEATED` | after insertion the part sits above its CAD seat or is tilted |
| `OCCLUDED` | the discriminating features are not visible in any available view |
| `COLLISION_RISK` | the approach or place path is not collision free |
| `UNVERIFIED_MODEL3_GEOMETRY` | the decision depends on Model 3 geometry that has no verified template |
