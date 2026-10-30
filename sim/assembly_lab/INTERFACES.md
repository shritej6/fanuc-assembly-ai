# T12 / T18 interfaces (shared by the 6 agents)

Spec: `docs/ROBOT_BRAIN.md`. Change this file only through the main session (it keeps the agents' parts compatible).

## Units and frames
- API units are mm and degrees. The sim's internal units (m, rad) never leave a module.
- Robot frame UF0 = the official client's frame (`robot/cell/cell_client.py`, `calibration/geometry.py`). TCP = UTOOL1,
  the fingertip point. Fingers straight down are `w, p, r = -180, 0, r`.
- Workspace box (PLAN §3.1): x 150–650, y −450 to 450, z −334 to 300 mm. The table is at z ≈ −335.
- Action row (identical to `CellClient.act_tcp`): `[x, y, z, w, p, r, gripper]`, gripper 0 open / 1 closed, chunk
  1–100 rows, dt 0.1–1.0 s.
- Observation (identical keys to `CellClient.observe`): `images` {table, side, wrist: HxWx3 uint8}, optional `depth`,
  `joints` (7: J1..J6 deg, gripper), `tcp_pose`, `instruction`.
  - Sim-only extras under `sim`: `object_poses` {id: [x, y, z, w, p, r]} and `contact` {id: {"normal_N", "lateral_N",
    "torque_Nm"}}.
  - These extras are used only for the force guard and for eval. A policy never reads them.

## Modules
| Path | Owner (agent) | Main entry points |
|---|---|---|
| `sim/assembly_lab/env/` | 1 Sim cell | `cell_env.AssemblyCellEnv(cfg)`: `.reset(layout) -> obs`, `.step_tcp(rows, dt) -> obs`, `.close()`. Layout dict: `{objects: [{id, x, y, yaw, state: installed\|fallen\|on:<base>}], spares: [...], defects: [...]}` |
| `sim/assembly_lab/objects/` | 2 Object master | `objects.yaml` (one entry per object id) and `planner.py`: `plan_pick(scene, id) -> GraspPlan`, `plan_place(scene, id, base) -> PlacePlan`, `blockers(scene, id)` |
| `sim/assembly_lab/physics/` | 2 Object master | `contact.yaml` (PhysX material and contact settings), `limits.yaml` `{axial_N, lateral_N, torque_Nm, status: PRELIMINARY\|SIM_MEASURED, source}` |
| `sim/assembly_lab/skills/` | 3 Skill demos | `pick(plan)`, `insert(plan)`, `pull_out(plan)`, `place_free(plan)` -> tcp rows; `record.py` writes Isaac Lab HDF5 demos |
| `sim/assembly_lab/mimic/` | 4 Mimic | Isaac Lab Mimic config + generation; `export_lerobot.py` -> LeRobot dataset |
| `policy_learn/groot/` | 5 GR00T trainer | GR00T N1.7 data config, fine-tune driver (Brev H100), checkpoint eval hook |
| `sim/assembly_lab/eval/`, `policy_learn/deploy/` | 6 Evaluator | scenario runner + KPIs; `groot_policy.py` (`act(obs)`) + real-arm guards |

## Data on fa-sim (`~/t12_data/`)
- `demos/<skill>/*.hdf5`: scripted demos. Only those that passed the force guard and the success check are kept.
- `mimic/<skill>/*.hdf5`: generated demos.
- `lerobot/<dataset>/`: the LeRobot export for GR00T.
- `eval/<run>/`: KPIs JSON + videos.

## Objects
- `objects.yaml` fields, all measured from CAD (`cad/products`, `cad/grasp_frames.json`, `cad/fingers/`):
  - `id`, `product`, `role` (base / 01..04), `sheet_order`
  - `bbox_mm`, `distinguishing_dims_mm` (the sizes that tell look-alikes apart), `mass_g` (PLA 1.24 g/cm³ × CAD volume),
    `com_mm`
  - `grips` (pad, closing axis, TCP, roll offset), `allowed_rolls_deg` (inside J6 and the finger check)
  - `pegs` / `holes` (position, diameter, height, chamfer), `mates_with`, `seated_height_mm`, `insert_axis`,
    `pull_out_axis`
  - `max_force` (from `limits.yaml`)

## Precision gates (user: "like heart surgery"; derived from CAD)
- Why: pegs and holes have 0.5 mm clearance per side; the 1.5 mm entry chamfer captures only within about 1.5 mm. The
  pegs sit about 15.6 mm from the part centre (peg A at -12, -10), so yaw error must be under 0.5/15.6 rad ≈ 1.8°;
  the gate is 1°.
- A demo is KEPT only if all of these hold:
  - final part xy within 0.3 mm of the CAD installed pose;
  - yaw within 1°;
  - seated height within ±0.3 mm (and tilt within 1°);
  - peak axial/lateral/torque under `physics/limits.yaml`;
  - no non-target object moved more than 1 mm or 2°;
  - arm and fingers touched only the target;
  - grip on the CAD pad at the planned roll;
  - every decision matches `docs/ROBOT_BRAIN.md` §3.
- Eval (agent 6) uses the same gates on unseen layouts and reports measured pass rates per product and case.
