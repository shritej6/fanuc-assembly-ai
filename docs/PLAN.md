# System design: Self-Inspecting Robotic Assembly Cell

Authored by Shritej Gunda.

Rockwell Automation and NVIDIA Physical AI Hackathon 2026, Challenge 3, FANUC ER-4iA cell. Official client:
`SIDCLab/PhysicalAI-Hackathon-2026`.

## 1. The idea

The parts for 3 of the 6 products stand on the table at random positions and yaw, each in the orientation it has in
its product: the arm only picks and places straight down and cannot turn a part over. All parts are black PLA. Some
are damaged, fallen over or from the wrong product. The cell:

- finds every part and base,
- groups them into products,
- inspects each part,
- replaces bad parts from a spare store,
- assembles each product on its base, checking every step,
- takes a product apart and rebuilds it if the result is wrong.

The aim is all 3 products finished with far fewer human touches, plus a per-part defect record that tells the plant
which spares to stock.

**Industrial framing.** The cell combines ViDi-style inspection, an Emulate3D-style fault-injected twin and CAD-driven
motion. Learned models are used where they improve on the scripted baseline. It is judged by first-pass yield,
escapes (target 0), builds per assist, final yield after self-repair, cycle time and localization error.

## 2. Design decisions

| Topic | Decision |
|---|---|
| Scope | All 3 products in simulation and on the real cell |
| Bases | Start at random positions and never move. The product is assembled on the base where it lies. |
| Table accessories | Free-standing; the camera finds their position every run |
| Zones | Work area, optional spare store, reject zone (tray or camera-defined area) |
| Order | One product at a time |
| Spares | High-risk parts only |
| Defects | Broken or missing feature; fallen over (reject and use a spare); print defect; wrong or foreign part |
| Defect probabilities | Engineering ranking, then cycle tests, then real run counts; labelled as estimates. Never derived from synthetic data. |
| Operator | Plain-language command plus a live report |
| Part colour | All black PLA |
| Restock pause | An Andon call: the cell stops safely, tells the operator which part to put in which store slot, and resumes when the camera sees it. Each call counts as one assist. |
| Real-cell client | Official client: `policy.py` implements `act(obs)`, checked with `uv run run.py --mock`, sent with `--submit` |

## 3. Facts that shape the design

### 3.1 The official client

- Actions are absolute targets: `"tcp"` = `[x, y, z, w, p, r, gripper]` in mm and degrees (UF0), or `"joints"`.
  - `act` returns a chunk of 1 to 100 targets, each with a time step `dt` of 0.1 to 1.0 s.
  - `act` has no time limit; the arm holds still while it computes, so heavy perception per call is acceptable.
- Workspace box: x 150 to 650, y -450 to 450, z -334 to 300 mm.
  - The table is at z of about -335, so the lowest fingertip target is about 1 mm above the table.
  - Fingers pointing straight down: `w, p, r = -180, 0, R`.
- Observations:
  - RGB from table, side and wrist cameras (native 1920x1080, 1920x1080, 1280x720).
  - Joints, TCP pose and the instruction.
  - Optional depth: table 640x400, wrist 848x480.
  - `calibration/geometry.py` gives `pixel_to_base` and `depth_to_base`.
- A run is 10 rollouts by default, in a 30-minute slot.
- For every rollout, the operator places the objects where `placement.yaml` says (x, y, yaw per object, plus a
  prompt). The true positions also give a direct localization-error measurement.

### 3.2 Measured look-alikes

Every STL pair was compared by surface distance:

| Group | Max shape difference |
|---|---|
| F01 = F02 | 0.0 mm, identical |
| F04 = F05 = F06 | 0.0 mm, identical |
| F03 vs others | 1.2 to 1.4 mm |
| Locks M4/M5/M6_04 | about 1.0 mm |
| Guides M4/M5/M6_03 | 1.5 mm, same volume |
| M1_02 vs M2_02 | 1.0 mm |
| Housings | 1.4 to 3 mm |
| Others | 2 to 5 mm |

Consequences:

1. There are only 3 base types: A = F01/F02, B = F03, C = F04/F05/F06. A base belongs to any product of its type, and
   the planner assigns bases to products. This is a real mixed-model line feature (shared fixtures).
2. The table camera sees about 0.8 mm per pixel on black parts. It separates part families, but not a lock from a
   similar lock. Exact identity therefore uses a wrist close-up at 0.24 to 0.49 mm per pixel. "Wrong part" defects are
   exactly these look-alike swaps.

### 3.3 Black PLA

- Silhouettes: strong contrast on the white table, good for detection.
- Surface detail: weak, and depth may be noisy.
- Colour: "wrong colour" is not a defect type; a non-black foreign object still is.

## 4. Technology stack

| Job | Tool |
|---|---|
| Simulated cell | Isaac Sim 6.0 starter plus the 30 STL parts |
| Training images | Replicator in Isaac Sim (black PLA, blur, noise, clutter) |
| Extra realism | Cosmos Transfer, spot-checked for shape drift |
| Find parts and fixtures | RT-DETR detector (Faster R-CNN as alternative) |
| Exact pose | FoundationPose with STL files and depth |
| Look-alikes | Wrist close-up classifier |
| Damaged parts | anomalib PatchCore |
| Missing layers | Table and wrist depth |
| Decisions | Recipe state machine plus live inventory |
| Build check | Recipe check, then Cosmos Reason 2 (8B) |
| Motion | Detected pose plus CAD heights (planner) |
| Learned motion | GR00T N1.7 via LeRobot (main), SmolVLA and ACT (baselines); trained on simulated demonstrations multiplied by Isaac Lab Mimic |
| Deployment | Official client `policy.py` |
| Compute | Cloud H100 and A100 GPUs, an HPC cluster with T4 and H100 nodes, and a DGX Spark (GB10) for Cosmos Reason 2 and the operator agent |

## 5. System pipeline

```
Operator: "assemble everything on the table"   (or obs["trial"]["instruction"])
1 SCAN      detector on table and side cameras -> type + location; FoundationPose (STL + depth) -> exact pose
2 IDENTIFY  wrist close-up classifier -> exact part ID for the look-alikes
3 GROUP     part ID -> product; base type -> assign a free base of the right type
4 INSPECT   PatchCore (damage) + depth (missing layers): OK | fallen / defect / foreign -> reject
5 REPLAN    bad or missing part -> spare from the store; if none, Andon call (pause, tell operator, resume)
6 ASSEMBLE  01 -> 04: yaw each part so its underside holes meet the pegs
            (sdg/data/insertion_tolerance.json: keyed parts +-1.4 to 2.4 deg, +-0.5 mm) -> VERIFY
7 RECOVER   not seated -> pull out and reinsert x2 -> then disassemble from the top, re-inspect, swap, rebuild
8 FINAL QC  recipe check, then Cosmos Reason 2 second opinion -> DONE / REBUILD
9 LEARN     log defects, jams, slips and retries per part -> risk table -> stocking advice
```

The state machine and perception code run identically against Isaac Sim, the mock and the real cell through one
adapter (`robot/adapter.py`).

## 6. Synthetic data

### 6.1 Models and their data

| Model | Learns | Camera | Images (target) |
|---|---|---|---|
| A. Detector | part and fixture type and location | table, side | 16,000 |
| B. Look-alike classifier | exact part of 30 (wrist close-up) | wrist | 24,000 |
| C. PatchCore | good-part renders for training; defect renders for evaluation | wrist | 8,000 |
| D. Build-check set | per product and step: seated, high, tilted, wrong part | wrist, side | 6,000 |
| E. GR00T demonstrations | scripted simulation demos multiplied by Isaac Lab Mimic, LeRobot format | all | sized from the scripted baseline |

### 6.2 Scene randomization

- Any 3 of 6 products; all 20 combinations equally often.
- Bases of the right type at random position and yaw.
- Parts standing as installed at any yaw; a share fallen over; touching and clustered.
- Partly assembled products (steps 0 to 4).
- 0 to 2 defects per scene from CAD defect variants: snapped tab, missing pin or lock, 97% or 103% scale, warp, layer
  split.
- A look-alike from a product that is not present (a wrong-part trap).
- Spares in the store, items in the reject tray, foreign objects, the arm partly in view.
- Black PLA material with realistic sheen, lighting changes, blur and noise, all 3 cameras.

Real-cell images are used for evaluation only.

## 7. Real-cell runs

- `placement.yaml` lists the rollouts, each with object poses (x, y, yaw) and a prompt. Defects can be planned per
  rollout (a broken lock, a look-alike swap, an empty store).
- Before every run: `uv run run.py --mock` must pass, every target must stay inside the TCP box, and the same
  `placement.yaml` is dry-run in the twin.

## 8. Physical test parts (all black)

| Print | Qty | Purpose |
|---|---|---|
| Deliberate defects | 4 | snapped lock, missing pin, 103% scale, warped or split |
| Look-alike traps | 1 to 2 | a lock or guide from a product not in the rollout |
| High-risk spares | 5 to 6 | chosen from the CAD-based risk ranking |
| Cycle-test copies | 2 to 3 | about 25 assemble and disassemble cycles; log cracks, jams, slips |

## 9. Industry layer

| # | Feature | What it does |
|---|---|---|
| 1 | PackML state model (ISA-TR88) | Idle, Starting, Execute, Held (Andon call), Suspended (waiting for a spare), Completing, Complete; Stopping and Aborted on faults |
| 2 | Product birth certificate | One record per product: part placed at each step, inspection image and score, retries, rebuilds, spare swaps, assists, cycle time. JSON plus a one-page PDF |
| 3 | Two independent checks (poka-yoke) | A part is installed only if the look-alike classifier and FoundationPose's fit score agree. A product is done only if the recipe check and Cosmos Reason 2 agree. Disagreement leads to reject or Held. |
| 4 | Live OEE and Andon dashboard | Availability x performance x quality, first-pass yield, escapes, builds per assist, red/amber/green status, live from the run log |
| 5 | OPC UA tags (`asyncua`) | Publishes PackML state, current product and step, good/reject/assist counts and the active fault for a PLC or HMI. Integration-ready, not a Rockwell product integration. |
| 6 | Kanban spares and defect Pareto | Using a spare raises a reorder card with the part's estimated defect rate. A Pareto chart of defect causes per part feeds back into print orientation or design. |
| 7 | Factory acceptance test (FAT) report | The fault-injected scenario fleet written up as an acceptance document: every injected fault, whether it was caught, and the result against the acceptance bars |

No number is reported unless it was measured, and defect probabilities stay labelled as estimates.

## 10. Evaluation

Metrics: first-pass yield, escapes (target 0), builds per assist, final yield, cycle time per step, localization error
against `placement.yaml`.

Baselines:

1. Blind scripted assembly, no inspection.
2. Inspection without spare swap or rebuild.

## 11. Engineering risks and mitigations

1. Slot time versus 3 products per rollout: rollouts are planned to fit the slot.
2. Black PLA depth and detail: measured on the real cell early.
3. 0.5 mm insertion clearance: wrist alignment and the reinsert loop.
4. A free-standing base can slide: optional pads, and the base pose is re-checked before each insert.
5. Cluster job interruptions: render chunks are idempotent and can be re-run.
