# Plan: Self-Inspecting Robotic Assembly Cell (v2)

Rockwell Automation + NVIDIA Physical AI Hackathon, Fall 2026, Challenge 3, on the FANUC ER-4iA cell.
- **v1:** 2026-10-27.
- **v2:** 2026-10-27, follows the Team V Tech Stack exactly, adds, the official client (`SIDCLab/PhysicalAI-Hackathon-2026`, commit ff7534e) and measured part look-alikes.
- **Final submission:** **2026-11-01, 11:59 PM CT.**

---

## 1. The idea

The parts for 3 of the 6 products stand on the table at random positions and yaw, each in the orientation it has in its product: the arm only picks and places straight down and cannot turn a part over (user, 2026-10-28). All parts are black PLA. Some are damaged, fallen over or from the wrong product. The cell:
- finds every part and base,
- groups them into products,
- inspects each part,
- replaces bad parts from a spare store,
- assembles each product on its base, checking every step,
- takes a product apart and rebuilds it if the result is wrong.

The aim is all 3 products finished with far fewer human touches, plus a per-part failure record that tells the plant which spares to stock.

**Industrial framing**, taken from your tech stack doc:
- It works like Cognex ViDi-style inspection, an Emulate3D-style fault-injected twin, and CAD-driven scripted motion.
- Learned models are only used where they beat the script.
- It is judged by first-pass yield, escapes (target 0), builds per assist, final yield after self-repair, cycle time and localization error.

## 2. Decisions locked

| Topic | Decision |
|---|---|
| Real-cell scope | All 3 products in sim and on the real cell |
| Bases | Start at random positions and never move. Assemble on the base where it lies; the product stays there. |
| Table accessories | Free-standing (no tape or glue); the camera finds their position every run |
| Zones | Work area, optional spare store, reject zone (tray or camera-defined area) |
| Order | One product at a time |
| Spares | High-risk parts only |
| Defects | Broken or missing feature; fallen over (arm cannot right it: reject, use a spare); print defect; wrong or foreign part |
| Failure probabilities | Engineering ranking, then cycle tests, then real run counts; labelled PRELIMINARY. Never derived from synthetic data. |
| Operator | Plain-language command plus a live report |
| Non-slip base pads | Optional |
| Part colour | **All black PLA** |
| Restock pause | Handled the industrial way: an **Andon call**. The cell stops safely, tells the operator exactly which part to put in which store slot, and resumes when the camera sees it. Each call counts as one assist in "builds per assist". |
| Real-cell client | Official repo: `uv sync`, our code goes in `policy.py` → `act(obs)`, checked with `uv run run.py --mock`, sent with `--submit`, run with `uv run run.py` |
| Repo / agents / team | New GitHub repo (no AdaptiveVision code); Claude and Codex split by domain, acting only on your instruction; you work alone |

## 3. Facts that change the design

### 3.1 The official client (verified in the repo, not from memory)

- **Actions are absolute targets**, not small steps: `"tcp"` = `[x, y, z, w, p, r, gripper]` in mm/deg (UF0), or `"joints"`.
  - `act` can return a chunk of 1–100 targets, each with a time step `dt` of 0.1–1.0 s.
  - **`act` has no time limit**; the arm holds still while it thinks. So heavy perception per call is fine.
- **Workspace box:** x 150–650, y −450 to 450, z −334 to 300 mm.
  - The table is at z ≈ −335, so the lowest fingertip target is about 1 mm above the table.
  - Fingers pointing straight down = `w, p, r = −180, 0, R`.
- **Observations:**
  - RGB from table, side and wrist, at whatever size we request (native 1920×1080 / 1920×1080 / 1280×720).
  - Joints, TCP pose and the instruction.
  - Optional depth: table 640×400, wrist 848×480.
  - `calibration/geometry.py` gives `pixel_to_base` and `depth_to_base`.
- **A run is 10 rollouts by default**, booked as a **30-minute slot** on the hackathon calendar. The organizers keep adding slots.
- **Who places the objects:** for every rollout, the **operator places the objects where `placement.yaml` says** (x, y, yaw per object, plus a prompt).
  - So *we* write the 10 random layouts, including which defects appear.
  - These true positions also give us a free localization-error measurement.
- **Old notes no longer apply:** the "8 mm / 5° per step" and "3 runs" notes in my memory are superseded by this repo.

### 3.2 Measured look-alikes: grouping can't rely on the table camera alone

I compared every STL pair by surface distance:

| Group | Max shape difference |
|---|---|
| **F01 = F02** | **0.0 mm, identical** |
| **F04 = F05 = F06** | **0.0 mm, identical** |
| F03 vs others | 1.2–1.4 mm |
| Locks M4/M5/M6_04 | about 1.0 mm |
| Guides M4/M5/M6_03 | 1.5 mm, same volume |
| M1_02 vs M2_02 | 1.0 mm |
| Housings | 1.4–3 mm |
| Others | 2–5 mm |

**What this means:**
1. **There are only 3 base types:** A = F01/F02, B = F03, C = F04/F05/F06.
   - A base belongs to *any* product of its type. The planner assigns bases to products; it doesn't search for "the F05 base".
   - Use this in the pitch: it is a real mixed-model line feature (shared fixtures).
2. **The table camera sees about 0.8 mm per pixel** on black parts. It can tell the *part family* apart, but not a lock from the lock next to it.
   - **Exact identity therefore needs a wrist close-up** at 0.24–0.49 mm per pixel. Your stack already has this "look-alike" step; it is now mandatory, not optional.
   - "Wrong part" defects (a knob on the indicator, an IO insert in the power housing) are exactly these look-alike swaps.

### 3.3 Black PLA

- **Silhouettes:** strong contrast on the white table, good for detection.
- **Surface detail:** weak, and depth may be noisy. Your stack flags this risk; **test it early on the cell.**
- **Colour checks:** "wrong colour" can't be a defect type any more. A **non-black foreign object** still can.

## 4. Tech stack (Team V Tech Stack, as given; followed exactly)

| Job | Tool |
|---|---|
| Simulated cell | Isaac Sim 6.0 starter + our 30 STL parts |
| Training images | Replicator in Isaac Sim (~800 renders per part: black PLA, blur, noise, clutter) |
| Extra realism | Cosmos Transfer (required, user 2026-10-28; spot-check for shape drift) |
| Find parts, fixtures | Detector: RT-DETR (user 2026-10-28; Faster R-CNN was the alternative; avoid YOLO's AGPL) |
| Exact pose | FoundationPose with STL files + depth |
| Look-alikes | Wrist-camera close-up classifier |
| Damaged parts | anomalib (PatchCore) |
| Missing layers | Table and wrist depth |
| Decisions | Recipe state machine + live inventory |
| Build check | Recipe check, then Cosmos Reason 2 (8B, user 2026-10-28; was 2B) |
| Motion | Detected pose + CAD heights (scripted baseline) |
| Learned picking | GR00T N1.7 via LeRobot (main), SmolVLA (backup), ACT (baseline); trained on sim demos multiplied by Isaac Lab Mimic; used only if it beats the script |
| Deployment | Official client `policy.py` |
| Compute | RunPod (user 2026-10-28, $100 cap): H100 SXM for the detector, A100 SXM for the classifier, PatchCore and Cosmos; GB10 for Cosmos too; Rosie's H100 when the queue starts it; GR00T on an H100/H200 later |
| Left out | ROS 2, MoveIt, Jetson, Isaac OS, Gemini Robotics, Cosmos 3 Nano/Super, π0.5 |

## 5. System pipeline

```
Operator: "assemble everything on the table"   (or obs["trial"]["instruction"])
1 SCAN      detector (Faster R-CNN / RT-DETR) on table → type + location; FoundationPose (STL + depth) → exact pose
2 IDENTIFY  wrist close-up classifier → exact part ID for the look-alikes
3 GROUP     part ID → product; base type → assign a free base of the right type
            per product: base + 01..04 present? Draw a boundary around each group in the report
4 INSPECT   anomalib PatchCore (damage) + table/wrist depth (missing layers): OK | fallen over / defect / foreign → reject
5 REPLAN    bad or missing part → spare from the store; if none, Andon call (pause, tell operator, resume)
6 ASSEMBLE  01→04: scripted motion from detected pose + CAD heights; yaw each part so its underside holes meet the
            pegs (sdg/data/insertion_tolerance.json: keyed parts ±1.4–2.4°, ±0.5 mm) (GR00T N1.7 only if it beats the script)
            → VERIFY (depth height + wrist check)
7 RECOVER   not seated → pull out and reinsert ×2 → still bad: disassemble 04→01, re-inspect, swap, rebuild
8 FINAL QC  recipe check, then Cosmos Reason 2 (8B) second opinion → DONE / REBUILD
9 LEARN     log defects, jams, slips and retries per part → risk table → stocking advice
```

The state machine and all perception code run **identically** against Isaac Sim, the mock and the real cell through one adapter (`robot/adapter.py`): `SimAdapter`, `MockAdapter` and `CellAdapter` (which wraps `policy.py`).

## 6. Synthetic data plan

### 6.1 Models and their data

| Model | Learns | Camera | Images (target) |
|---|---|---|---|
| A. Detector (Faster R-CNN / RT-DETR) | part and fixture type + location | table, side | 16,000 |
| B. Look-alike classifier | exact part of 30 (wrist close-up) | wrist | 24,000 (~800 per part, per the stack) |
| C. PatchCore good-part renders + defect test set | anomalib learns from good parts; defect renders used to evaluate it | wrist | 8,000 |
| D. Build-check set | per product and step: seated / high / tilted / wrong part (recipe check + Cosmos Reason 2 evaluation) | wrist, side | 6,000 |
| E. GR00T demos | scripted sim demos multiplied by Isaac Lab Mimic, LeRobot format | all | sized after the scripted baseline works |
| Pilot | all of the above, label check on | all | 1,500 |
| **Total** | | | **about 55,500** |

### 6.2 Scenes (Replicator randomization)

- Any 3 of 6 products; all 20 combinations equally often.
- Bases of the right type, at random position and yaw.
- Parts standing as installed at any yaw (sets N R O P Q); a share fallen over (reject); touching and clustered.
  Sets A–M drew random drop poses and still teach what a fallen part looks like.
- Partly assembled products (steps 0–4).
- 0–2 defects per scene, from **CAD defect variants**:
  - snapped tab, missing pin or lock,
  - 97% or 103% scale,
  - warp, layer split.
- A look-alike from a product that isn't present (a wrong-part trap).
- Spares in the store, items in the reject tray, foreign objects (screws, tape, a coloured cube), the arm partly in view.
- **Black PLA material with realistic sheen**, lighting changes, blur and noise, all 3 cameras.

### 6.3 Compute and batches (estimates; the pilot measures the real speed)

The basis for the time estimate is **5.5 s per image on a T4** (real-time render), from my notes.

- **Total:** about 55.5k images ≈ **85 T4 GPU-hours** ≈ **about 11 h on 8 T4s** (4 jobs × 2 GPUs).
- **Batches:** 111 chunks of 500. Every chunk can be re-run and finished ones are skipped, because Rosie T4 nodes can kill jobs when a neighbour job ends.
- **If the pilot is slower than 5.5 s per image:** cut B to about 500 per object first, and overflow to Brev.

| Resource | Job |
|---|---|
| Rosie T4 | Isaac/Replicator rendering, Slurm arrays |
| Rosie dgxh100 / Brev H100–A100 | Train A–D; Cosmos Transfer pilot (50 images, measure time and credits, then about 10% of set A) |
| Brev | GR00T training, Cosmos, overflow when Rosie's H100 cannot start in time. Cap $50 for this phase (user 2026-10-28). Stop every instance as soon as its job ends. |
| GB10 / DGX Spark | Cosmos Reason 2 build-check endpoint + operator agent |
| Omniverse / Isaac Sim | USD assets of all 30 objects, zones, insertion physics, the fault-injected scenario fleet |

**Real cell images** (LeRobot recordings) are for **evaluation only**, never for training.

## 7. Real-cell runs

- **Write 10 rollouts in `placement.yaml`** with layout and objects `different`. Each rollout gets 3 random products with random x/y/yaw, plus a prompt.
  - Defects are planned per rollout: for example rollout 4 has a broken lock, rollout 7 a look-alike swap, rollout 9 an empty store. This mirrors your "challenge parts" test table.
- **Time is the real limit:** the slot is 30 minutes for setup + rollouts, and the operator must place about 15–20 objects per rollout.
  - Building 3 products per rollout may need several minutes.
  - **Ask the organizers** whether we can book fewer, longer rollouts (`rollouts: 3`, each assembling 3 products) or book back-to-back slots.
- **Before the first slot:**
  - `uv run run.py --mock` must end with "✓ Ready".
  - Every real-cell target must stay inside the TCP box.
  - Do a dry run in the twin with the same `placement.yaml`.

## 8. Agents (Claude Code + Codex)

| Claude Code: CAD, sim, data, docs | Codex: models, logic, robot, tests |
|---|---|
| `cad-engineer`: defect CAD variants, assembly instruction sheets, grasp/insert frames, risk ranking from the CAD | `perception-trainer`: detector, look-alike classifier, PatchCore, FoundationPose setup |
| `sim-builder`: USD conversion, black PLA material, zones, insertion physics | `assembly-planner`: recipe state machine, inventory, recovery, Andon, risk table |
| `sdg-engineer`: Replicator generators, Slurm jobs, label QA | `robot-adapter`: Sim/Mock/Cell adapters, `policy.py`, TCP-box guard |
| `cosmos-augment`: Cosmos Transfer and Cosmos Reason endpoint | `operator-agent`: plain-language command → task, live report |
| `doc-writer`: video script, summary, references | `test-engineer`: unit, integration, scenario-fleet regression |
| `demo-generator`: Isaac Lab Mimic demos for GR00T | `policy-learner`: GR00T N1.7 / SmolVLA / ACT fine-tune on H100/Brev, compared against the script |

**Rules that keep them from colliding:**
- **One board:** `coord/TASKS.md` lists `ID | owner | status | branch | depends on`. A task starts only when **you** mark it READY.
- **Shared formats first:** both sides code against the JSON schemas in `contracts/`. A schema changes only through a PR reviewed by both.
- **One branch per task**, reviewed by the *other* agent, merged by you.
- **Logs:** `coord/HANDOFF.md` gets an entry after every task; `coord/DECISIONS.md` records decisions. `~/.ai-shared/memory/current_task.md` points to them.

**Repo layout:**

```
fanuc-assembly-ai/
├── AGENTS.md  CLAUDE.md  README.md  docs/PLAN.md  docs/ASSEMBLY_INSTRUCTIONS/
├── coord/  contracts/  cad/  sim/  sdg/  perception/  assembly/  agent/  eval/  tests/
├── robot/ (adapter.py, sim_adapter.py, mock_adapter.py, cell/ ← vendored copy of the official client, policy.py)
```

## 9. Physical plan (10–15 extra prints, all black)

| Print | Qty | Purpose |
|---|---|---|
| Deliberate defects | 4 | snapped lock, missing pin, 103% scale, warped/split |
| Look-alike traps | 1–2 | a spare lock/guide from a product not in the rollout |
| High-risk spares | 5–6 | from the CAD-based ranking I'll produce before you print |
| Cycle-test copies | 2–3 | about 25 assemble/disassemble cycles; log cracks, jams, slips |

## 10. Day by day

| Day | You | Claude | Codex |
|---|---|---|---|
| **Mon Oct 27** | Approve v2; ask organizers about rollout length (section 7) | Repo, coord files, contracts; vendor the client; assembly instruction sheets; CAD risk ranking | Catalog + recipe graph; Mock/Cell adapter skeleton; `run.py --mock` passing |
| **Tue Oct 28** | Print defects and spares; depth test on black PLA if a slot is free | Defect variants; USD + black material; Isaac zones; pilot 1.5k | State machine + recovery + Andon on the sim adapter; tests |
| **Wed Oct 29** | Cycle tests → risk table v1 | Full sets A–D on T4; Cosmos pilot; Mimic demos | Train detector + classifier; FoundationPose + PatchCore setup; inference service |
| **Thu Oct 30** | **Real slot 1** | Fix sim-to-real gaps; b-roll | GR00T/SmolVLA fine-tune vs script; scenario fleet (3 products × 20 layouts × faults); KPIs |
| **Fri Oct 31** | **Real slot 2 (+3)** | Summary draft, renders | Fixes; baseline comparison |
| **Sat Nov 1** | Video | Summary, references, README | Final tests, clean repo → **submit by 11:59 PM CT** |

**If time runs short:** real-cell scope changes only with your OK. Cosmos Transfer is required (user 2026-10-28).

## 10a. Industry-level layer (approved 2026-10-27; adds to the stack, replaces nothing)

| # | Feature | What it does | Owner | When |
|---|---|---|---|---|
| 1 | **PackML state model** (ISA-TR88) | The recipe state machine uses the standard states: Idle → Starting → Execute → Held (Andon call) / Suspended (waiting for a spare) → Completing → Complete; Stopping/Aborted on faults | Codex `assembly-planner` | Tue |
| 2 | **Product birth certificate** | One record per product: part placed at each step, its inspection image and score, retries, rebuilds, spare swaps, assists, cycle time. Output: `eval/records/<run>/<product>.json` + a 1-page PDF | Codex (data), Claude (PDF layout) | Wed |
| 3 | **Two independent checks (poka-yoke)** | A part is installed only if the look-alike classifier **and** FoundationPose's fit score agree. A product is DONE only if the recipe check **and** Cosmos Reason 2 pass. Disagreement → reject or Held (ask operator). | Codex `assembly-planner` | Tue–Wed |
| 4 | **Live OEE + Andon dashboard** | Availability × performance × quality, first-pass yield, escapes, builds per assist, red/amber/green status, live from the run log; video-ready | Claude `doc-writer` + Codex (metrics) | Wed–Thu |
| 5 | **OPC UA PLC-style tags** (`asyncua`) | Publishes PackML state, current product/step, good/reject/assist counts and the active fault, so a PLC or HMI could read the cell. Pitched as "integration-ready", not as a Rockwell product integration. | Codex `robot-adapter` | Thu (if on track) |
| 6 | **Kanban spare replenishment + defect Pareto** | Using a spare raises a reorder card that includes the part's PRELIMINARY failure rate. A Pareto chart of defect causes per part (crack at a layer line, jam, slip, extraction failure) feeds back into print orientation or design. | Codex (logic), Claude (charts) | Thu |
| 7 | **Factory acceptance test (FAT) report** | The fault-injected scenario fleet written up as an acceptance document: every injected fault, whether it was caught, the result, and pass/fail against the bars in the stack (zero escapes, every swap caught) | Claude `doc-writer` | Fri |

The rules from section 2 still apply: no number is reported unless it was measured, and failure probabilities stay labelled PRELIMINARY.

## 11. Proof (adopted from your doc)

**Metrics:** first-pass yield · escapes (target 0) · builds per assist · final yield · cycle time per step · localization error vs. `placement.yaml`.

**Baselines:**
1. Blind scripted assembly, no inspection.
2. Inspection without spare swap or rebuild.

**Reporting rule:** report the failures too, and never claim a number we didn't measure.

## 12. Risks

1. **Slot time vs. 3 products × 10 rollouts** (section 7).
2. **Black PLA depth and detail.** Test on the first slot.
3. **0.5 mm insertion clearance.** Wrist alignment and the reinsert loop handle it.
4. **A free-standing base slides.** Optional pads, and the base pose is re-checked before each insert.
5. **Rosie queue and job kills.** Chunks can be re-run safely; Brev is the overflow.
6. **One person, 5 days.** Hence the drop order.

## 13. Open questions

1. **Organizers:** can a rollout be about 10 minutes (3 products) with `rollouts: 3`, or can we book several slots in a row?
2. **Organizers:** may a few real *good-part* reference photos be used as an inspection reference, or is that training on real data?
