# Self-Inspecting Robotic Assembly Cell

Shritej Gunda (Team V)

Rockwell Automation and NVIDIA Physical AI Hackathon 2026, Challenge 3, FANUC ER-4iA cell.

This repository contains a robotic assembly cell that builds small 3D-printed products part after part and checks
its own work at every step. A FANUC ER-4iA takes a plain-language command, finds the parts on the table, inspects
each one, rejects parts that are defective, fallen or from another product, and inserts the good parts onto fixed
bases in assembly-sheet order. Motion comes from NVIDIA GR00T N1.7 fine-tuned on simulated demonstrations, combined
with a CAD-based grasp and insertion planner, and every motion passes through an independent safety guard before it
reaches the robot.

## 1. Problem and business case

Small-batch assembly of printed or molded parts is still largely manual. Three issues drive the cost:

- **Look-alike parts.** Parts from different products often differ by 1 to 1.5 mm. A wrong part looks right until
  the product fails a functional test.
- **Late defect detection.** Cracks, warping and missing features are usually found at end-of-line inspection, after
  labour and other parts have already been spent on the product.
- **Reprogramming cost.** A classical robot program is written per product. With many small batches, programming
  time exceeds run time.

The cell addresses all three: it inspects every part before it is installed, it records evidence per product, and
it is retargeted to a new product by a language command and CAD data instead of a new robot program.

## 2. System design

**Hardware.** FANUC ER-4iA with an SMC MHZL2-16D2 parallel gripper and custom printed fingers. Three cameras: a table
camera, a side camera and a wrist camera.

**Task.** Six product designs exist; each trial uses three. Each product has a fixed base (F01 to F06) and four parts
(`M<product>_01` to `M<product>_04`), installed in assembly-sheet order 01 to 04. Parts stand in their installed
orientation at any yaw, because the arm picks and places straight down. The planner aligns the yaw so that the
bottom holes of each part meet the pegs of the base.

**Inspection.** Every part is checked before it is used:

| Check | Method |
|---|---|
| Find parts and bases | RT-DETR detector (Faster R-CNN as alternative) |
| Six-degree-of-freedom pose | FoundationPose |
| Exact identity of look-alikes | Wrist close-up classifier (parts that differ by 1 to 1.5 mm) |
| Surface defects | PatchCore anomaly detection |
| Finished build | Recipe check plus Cosmos Reason 2 build check |

**Recovery.** Defective, fallen or foreign parts go to a reject zone and a spare is taken from the store. A part that
does not seat gets two retries, after which the part and the holes are inspected. Dropped parts are re-detected and
re-picked. When a wrong part sits under others, only the parts above it are removed, in reverse order. When the cell
is unsure, it calls Andon and waits for the operator.

**Insertion.** Insertion follows the way a person does it: approach with a slight tilt, touch the chamfer, level the
part, apply a small wiggle and finish with a light push. The arm has no force sensor, so force limits are learned in
simulation. On the real arm, insertion uses a slow final approach and a depth cap.

**Decision making.** NVIDIA GR00T N1.7 is fine-tuned on Isaac Lab Mimic demonstrations and works together with a
CAD-based grasp and insertion planner. ACT and SmolVLA are trained as baselines. An operator agent turns language
commands into typed tasks and reports progress live.

**Industry layer.** PackML (ISA-TR88) states, a product birth certificate per product, two independent checks before
a part or product is accepted, a live OEE and Andon dashboard, OPC UA tags for PLC and HMI integration, Kanban spare
replenishment with a defect Pareto, and a factory acceptance test (FAT) report.

## 3. Architecture

```
 table / side / wrist cameras
              |
              v
   perception (detect, pose, classify, anomaly)
              |
              v
   planner + PackML state machine  <---  operator agent (language command)
              |
              v
   GR00T N1.7 policy (action chunks)
              |
              v
   safety guard (chunk guard + runtime checks)
              |
              v
   FANUC ER-4iA cell (official client)
```

| Folder | Role |
|---|---|
| `cad/` | Product CAD, STL exports, defect variants, gripper fingers, grasp frames, assembly sheets, part risk ranking |
| `sdg/` | Synthetic data generation: scene sampling, Isaac Sim Replicator configs, Cosmos augmentation, coverage and QA |
| `sim/` | Isaac Lab assembly cell (`assembly_lab/`: environment, physics seating, skills, Mimic generation, evaluation), demos, starter scene |
| `perception/` | Detector, look-alike classifier, FoundationPose interface, PatchCore anomaly, depth, dataset manifests and metrics |
| `policy_learn/` | GR00T N1.7 dataset preparation and fine-tuning, ACT and SmolVLA baselines, deployment runtimes and guards |
| `serving/` | Cosmos Reason 2 build-check endpoint and health checks |
| `eval/` | Scenario fleet, fault-injection runtime, scenario reports, Cosmos Reason 2 benchmark |
| `assembly/` | Product catalog, controller (startup, inspection, placement, recovery, completion), birth certificate |
| `agent/` | Operator agent: language commands, task validation, authority rules, Cosmos Reason client |
| `industry/` | PackML lifecycle, run log, OEE metrics and the Andon dashboard |
| `robot/` | Robot adapter, OPC UA server and tag model; `robot/cell/` holds the official cell client and the runtime programs |
| `contracts/` | JSON schemas shared across modules (detection, inspection, plan step, PackML state, birth certificate) |
| `tools/` | Placement file generation and validation, rehearsal scenarios against a delayed mock cell |
| `tests/` | Unit and integration tests for all of the above |
| `docs/` | Design notes, interfaces, assembly instructions and measured perception results |

## 4. Safety

Every action chunk passes a chunk guard (`policy_learn/deploy/guards.py`, `policy_learn/deploy/cell_guards.py`)
before it reaches the cell:

- workspace box (x 150 to 650 mm, y -450 to 450 mm, z -334 to 300 mm), matching the official client;
- tilt limit that keeps the fingers pointing down;
- contact floor near the insertion target, so the arm cannot drive a part below its seated height;
- speed limits per step, with a slower limit in the final 5 mm above the contact target;
- gripper changes only while the arm is still, and no gripper chatter;
- malformed chunks (shape, non-finite values, time step outside 0.1 to 1.0 s) are held whole.

On top of the guard, the runtime checks grasp tolerance before lifting, confirms that a part is seated, detects jams,
enforces time budgets per step and keeps the carry path inside a corridor. A refused chunk becomes a safe hold at the
current pose with the current gripper state, so a held part stays held.

## 5. Real-cell runtime

The programs in `robot/cell/` wrap the official client without modifying `run.py`, `policy.py` or `cell_client.py`:

| Program | Behaviour |
|---|---|
| `policy_planner.py` | Planner-driven: the CAD planner drives approach, grasp, carry and insertion from `placement.yaml` |
| `policy_hybrid.py` | GR00T drives approach and carry; the planner takes over for the final grasp and insertion |
| `policy_direct.py` | GR00T drives every motion, through the chunk guard |

`submission/submission.yaml` describes the team, policy and task. `submission/placement.yaml` gives the object poses
and the prompt for each rollout; it is generated by `tools/placement/single_product.py` and checked by
`tools/placement/validate.py`.

```bash
cd robot/cell
uv run --with pyzmq --with msgpack --with msgpack-numpy python policy_planner.py --mock
uv run --with pyzmq --with msgpack --with msgpack-numpy python policy_planner.py --checkpoint groot://127.0.0.1:5557
```

The same flags apply to `policy_hybrid.py` and `policy_direct.py`. The `groot://` checkpoint points to the GR00T
policy server over ZMQ. Each run writes a report of every step event and guard hold.

## 6. Training pipeline

```
Isaac Lab cell -> Mimic demonstrations -> LeRobot export (5 Hz, 0.2 s actions)
  -> GR00T N1.7 fine-tune -> policy server over ZMQ -> evaluation in simulation
```

1. **Simulation cell.** `sim/assembly_lab/` builds the ER-4iA, gripper, bases and parts in Isaac Lab, with contact
   physics tuned against measured seating tolerances.
2. **Demonstrations.** Scripted skills (pick, insert, pull out, place) record seed episodes; Isaac Lab Mimic
   multiplies them across layouts, yaws and products.
3. **Export.** Episodes are written in LeRobot format at 5 Hz, one row per 0.2 s action step, matching the runtime.
4. **Fine-tuning.** `policy_learn/groot/` prepares the dataset and fine-tunes GR00T N1.7 with an ER-4iA embodiment
   configuration. ACT and SmolVLA baselines use the same data.
5. **Serving.** The checkpoint is served over ZMQ and queried by the runtime in action chunks.
6. **Evaluation.** `sim/assembly_lab/eval/` and `eval/` replay scenarios, including injected faults, and score them.

Datasets and checkpoints are hosted on Hugging Face.

## 7. Results

All numbers below were measured.

- **Simulation.** The hybrid runtime (GR00T with planner takeover for the final grasp and insertion) seated the first
  part of product 5.
- **Real cell.** The planner-driven runtime picked, carried and placed all four parts of product 5 at the planned
  seat poses in three of four recorded trials, releasing each part within about 1 mm of its planned seat pose.
- **Perception.** Pose and anomaly measurements are in `docs/results/`; synthetic data coverage is in
  `docs/SDG_COVERAGE.md`.

## 8. Roadmap

1. Camera-based part localisation, so parts can be placed anywhere on the table.
2. Fine-tuning on real images, with camera matching and domain randomisation.
3. Closed-loop insertion with a wiggle search driven by the seated check.
4. Full three-product trials in one rollout.
5. Full inspection, reject handling and Andon on the real cell.

## 9. Setup and tests

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

Isaac Lab, GR00T and the perception models need GPU environments; their install scripts are in `sim/`,
`policy_learn/` and `perception/`.

## 10. References

- NVIDIA Isaac Lab and Isaac Lab Mimic: https://github.com/isaac-sim/IsaacLab
- NVIDIA Isaac GR00T N1.7: https://github.com/NVIDIA/Isaac-GR00T
- LeRobot (ACT and SmolVLA baselines, dataset format): https://github.com/huggingface/lerobot
- NVIDIA Cosmos Reason 2 (build check)
- anomalib (PatchCore): https://github.com/openvinotoolkit/anomalib
- FoundationPose: https://github.com/NVlabs/FoundationPose
- RT-DETR: https://github.com/lyuwenyu/RT-DETR
- ACT (Zhao et al., Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, 2023)
- OPC UA for Python (asyncua): https://github.com/FreeOpcUa/opcua-asyncio
- PackML (ISA-TR88.00.02)
- Hackathon cell client: https://github.com/SIDCLab/PhysicalAI-Hackathon-2026

## Acknowledgements

Rockwell Automation, NVIDIA, the UWM Connected Systems Institute and MSOE, for the cell, the compute and the
challenge.

## Author

Shritej Gunda (Team V)
