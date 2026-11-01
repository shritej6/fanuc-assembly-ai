# Contracts

Shared JSON Schemas (draft 2020-12) that every package in the cell reads and writes.
Treat them as stable interfaces: change a schema only together with all of its producers and consumers.
All poses use the controller frame **UF0** in mm and degrees (the same frame as `obs["tcp_pose"]` and
`"tcp"` actions; the table surface is at about z = -335 mm).

| File | Used for |
|---|---|
| `common.schema.json` | shared ids, families, base types and the UF0 pose |
| `taxonomy.schema.json` | the 30 objects, their families and look-alike groups |
| `detection.schema.json` | perception output for each object instance |
| `product_group.schema.json` | grouping output for each product |
| `inspection.schema.json` | inspection verdict and action for each part |
| `plan_step.schema.json` | steps passed from the state machine to the motion layer |
| `packml_state.schema.json` | cell state (PackML) |
| `birth_certificate.schema.json` | traceability record for each product |
| `catalog.schema.json` | product recipes with CAD fixture-frame installed transforms, grasp-roll offsets and hash-pinned source provenance |

## Error codes (`common.schema.json#/$defs/error_code`)

The codes follow the Team V specification. An inspection lists every code that applies (`error_codes`);
a plan step names the code it answers (`error_code`). The arm picks and places straight down only, so a part
that is not standing in its installed orientation is never reoriented: it goes to reject and a spare is used.

| Code | Meaning |
|---|---|
| `WRONG_MODEL` | the part belongs to a different product model than the work order asks for |
| `WRONG_PART` | right model, wrong part for this assembly stage |
| `WRONG_ORIENTATION` | the part is not standing in its installed orientation (lying or upside down); the arm cannot reorient it |
| `WRONG_FRONT_BACK` | the front/rear (yaw) of the part or target does not match the fixture's known front/rear axis |
| `LOOKALIKE_UNRESOLVED` | fewer than two independent cues separate the part from its look-alikes |
| `GUARD_TOPPLED` | a guard (M*_04) has fallen over; there is no tested recovery, so a spare is used |
| `GRIP_FEATURE_INACCESSIBLE` | no validated grasp envelope is free on the part |
| `HOUSING_NOT_SEATED` | after insertion the part sits above its CAD seat or is tilted |
| `OCCLUDED` | the discriminating features are not visible in any available view |
| `COLLISION_RISK` | the approach or place path is not collision free |
| `UNVERIFIED_MODEL3_GEOMETRY` | the decision depends on Model 3 geometry that has no verified template |
