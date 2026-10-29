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
| `catalog.schema.json` | product recipe with CAD fixture-frame installed translation/rotation and immutable source provenance (filled in T04) |
