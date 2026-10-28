# Synthetic image generation plan (T06 / T07 / T08 / T10)

Status 2026-10-28. Extends `docs/PLAN.md` section 6. Stack unchanged: **Isaac Sim + Replicator** renders, Cosmos Transfer
optional (T11). Real-cell images are for evaluation only, never training.

## 1. What is ready and what is not

| Piece | File | State |
|---|---|---|
| CAD geometry: installed pose of every part on its base + resting poses when loose | `sdg/tools/object_geometry.py` → `sdg/data/object_geometry.json` | **Done, verified.** All 30 STLs register onto the CAD assembly within 0.003 mm. Every resting pose is checked to sit on z = 0 |
| Table zones (work area, spare store, reject tray) | `sdg/configs/cell_layout.yaml` | **Proposed.** Must be checked on the real cell before T22 |
| Set sizes and randomization | `sdg/configs/sets.yaml` | Done (sizes from PLAN 6.1) |
| Scene sampler (layout, defects, look-alike traps, build states, wrist poses) | `sdg/scene_sampler.py` | **Done, tested.** All 54,000 scenes lay out in about 16 s on a laptop; 16 behaviour tests |
| Post-processing + label assembly | `sdg/postprocess.py` | Done, tested (6 tests) |
| Isaac stage builder + render loop | `sdg/isaac/stage.py`, `sdg/isaac/generate.py` | **Runs on the GB10** (Isaac 6.0.0): images, masks and labels come out; boxes now come from the masks. Rosie T4 run in progress |
| Render hosts | `sdg/rosie/*` (T4, Slurm, Isaac 6.0.0 sandbox), `sdg/spark/render_chunk.sh` (GB10, Docker) | Both start Isaac 6.0.0; see section 5 |
| Starter cell (robot, gripper, cameras, lights) | `sim/starter/` | Vendored unmodified; all 38 files match the starter's own SHA-256 list (`MANIFEST.json`) |
| Defect meshes | `sdg/tools/defect_variants.py` → `cad/defects/` | **Done (T06).** 144 variants (24 parts × 6), regenerated on each host and checked against `manifest.json` (identical on Mac and GB10) |
| Coverage report | `sdg/coverage.py` → `docs/SDG_COVERAGE.md` | **Done.** Every planned case present; smallest is part × defect at 51 |
| QA gate | `sdg/qa.py` | Done: files, box = mask extent, depth range, overlay sheet; `--sample N` for spot checks |
| Arm reachability | `sdg/wrist_ik.py` | **Done.** The sampler only emits arm poses the starter's own IK solves (section 3) |
| Watchdog | `sdg/watchdog.py` | **Done.** Sample QA of every finished chunk, bounded retries, hang kill (section 12) |
| Black PLA look check | `sdg/isaac/stage.py` (UsdPreviewSurface) | Parameters drafted; T07 compares a render against a phone photo of a real print |

## 2. Sets

A scene renders one image per listed camera. Chunks hold 500 images.

| Set | Model | Cameras | Images | Scenes | Chunks | Scene content |
|---|---|---|---|---|---|---|
| A | Detector (Faster R-CNN / RT-DETR) | table + side | 16,000 | 8,000 | 32 | 3 products, steps 0–4, 0–2 defects, look-alike trap 30%, spares, rejects, foreign objects, clusters 35%, arm in view 40% |
| B | Wrist look-alike classifier | wrist | 24,000 | 24,000 | 48 | Exactly 800 per object (all 30). The target is always loose; wrist 120–260 mm above it, tilt up to 20° |
| C | PatchCore good parts + defect test set | wrist | 8,000 | 8,000 | 16 | Cycles the 24 parts; 75% good (training set), 25% one defect on the target (test set only) |
| D | Build check (recipe check + Cosmos Reason 2) | wrist + side | 6,000 | 3,000 | 12 | Top installed part is seated / high 1.5–4 mm / tilted 4–12° / wrong part (same family, other product) |
| Pilot | all | all | 1,500 | — | — | First 375 images of chunk 0 of each set. **The full run reuses them** |

Total: **54,000 images in 108 chunks**, plus the edge-case sets F–J (40,000) and the close-ups K–L (4,000) in
section 11. E (GR00T demos) is sized after the scripted baseline works (T12).

## 3. Scene rules (what the sampler guarantees, and the tests check)

- 3 different products. Each product's own base type, at a random position and yaw, never touching another base
  (25 mm gap). Bases never move, matching the real cell.
- Every part of each product appears exactly once: installed on its base (steps 1..k, exact CAD pose) or loose.
- Loose parts lie in a CAD resting pose, weighted by how likely that pose is. Parts may touch but never overlap.
- Look-alike trap: a part from a product that is **not** on the table but shares a family with a present part.
- Spares sit in the 6 store slots (the PRELIMINARY high-risk list). Rejects go in the tray (fewer when it is full).
- Foreign objects: M3 screw, tape roll, 10 mm cube.
- Every object centre sits 180–490 mm from the robot base. Wrist TCP poses stay inside the cell's TCP box.
- Every arm pose (wrist view or arm in view) has an IK solution with the starter's solver, joint limits and seeds.
  The sampler redraws an unreachable pose; the arm is the last draw, so nothing else in the scene changes. Found
  2026-10-28 when Rosie set B stopped at unreachable poses (1.6 % of B, 0.5 % of D, about 4 % of A's arm-in-view
  poses). `uv run python -m sdg.tools.check_wrist_ik <sets>` re-checks a set before it is queued.
- Same (set, index) always gives the identical spec, so a killed chunk re-renders identically.

## 4. Outputs and labels

```
$DATA_ROOT/
  specs/<set>/chunk_NNNN.jsonl                 scene specs (input to the renderer)
  synthetic/<set>/chunk_NNNN/
    rgb/<cam>/<index>.jpg                      JPEG, quality 80–98 (the cell streams JPEG)
    depth/<cam>/<index>.png                    uint16 mm, 0 = no return; table 640×400, wrist 848×480 (real sizes)
    seg/<cam>/<index>.png                      uint16 instance mask: uid number (o07 → 8), 0 = everything else
    labels/<index>.json                        full spec + per-camera boxes: uid, object_id, kind, state,
                                               variant, build_state, bbox_xyxy, occlusion
    _DONE.json                                 scene count, wall time, measured s/img, container used
```

Converters for training formats (COCO for the detector, crops for the classifier, good-only folders for anomalib)
read `labels/`. They are **T13 / T14 (Codex)** work. The label fields above are a proposal for Codex to review in
this PR.

## 5. Compute and storage (estimates; the pilot replaces them with measurements)

| Item | Estimate | Basis |
|---|---|---|
| Render speed | 5.5 s / image on a T4 | My earlier notes (real-time render). Not measured for this scene yet |
| GPU time | about 83 T4 GPU-hours | 54,000 × 5.5 s |
| Wall time | about 10–11 h | 8 array tasks in parallel (`MAX_PARALLEL=8`); Rosie had 12 idle T4 nodes on 2026-10-27 |
| Disk | about 25 GB images + about 5 GB cache | JPEG about 200–400 KB, depth PNG about 300 KB, masks about 50 KB |

Render hosts (checked 2026-10-27):

| Host | Isaac | Notes |
|---|---|---|
| Rosie T4 (`teaching`, 20 nodes × 4 T4) | 6.0.0 sandbox at `~/containers/isaac-sim-6.0.0`, built by `sdg/rosie/build_isaac_sandbox.sbatch` | The shared `isaac-sim-6.1.0.sif` cannot be used: `/isaac-sim` inside it is root-only. Each task gets its own overlay file (Isaac outgrows the 64 MB tmpfs). Compute nodes reach NVIDIA's asset server |
| GB10 (`spark`, DGX Spark, arm64) | Official `nvcr.io/nvidia/isaac-sim:6.0.0` (arm64), Docker | Runs as the image's own user (uid 1234) in our group. 1.6 TB free |
| Brev A100 / H100 | — | **Not for rendering**: A100 / H100 have no RT cores, which Isaac RTX rendering needs. Use them for training (section 10) |

Both render hosts run **Isaac Sim 6.0.0, the version in the stack**. The T4s take sets A and B; the GB10 takes C
and D, which need defects and depth. GB10 output is copied to Rosie with `rsync` for training.

## 6. How to run (after the user approves the Isaac container, section 8)

```bash
# laptop: regenerate geometry only if CAD changes
~/cad-workspace/.venv/bin/python sdg/tools/object_geometry.py --cad-dir ~/cad-workspace/FANUC_PRINT_FIRST/cad
uv run pytest -q

# Rosie login node, once
git clone https://github.com/shritej6/fanuc-assembly-ai.git ~/fanuc-assembly-ai
~/fanuc-assembly-ai/sdg/rosie/setup.sh

# pilot, then the full sets
export ISAAC_SIF=/data/containers/<approved container>.sif
export DATA_ROOT=~/fanuc-assembly-data          # later: the shared team folder
~/fanuc-assembly-ai/sdg/rosie/submit.sh pilot
~/fanuc-assembly-ai/sdg/rosie/submit.sh A       # then B, C, D
```

## 7. Pilot gates (all must pass before T10 starts the full sets)

1. Every output exists for every scene and camera; `_DONE.json` holds the measured s/img.
2. Boxes and masks agree: for 50 random images, each box encloses its mask pixels (scripted check).
3. Overlay sheet of 40 images (boxes + ids) checked by eye: no missing, shifted or mislabelled parts.
4. Depth: table surface reads 0.5–1.5 m from the table camera, with no zeros on the part tops.
5. The black PLA renders as dark grey with visible edges and layer sheen, not a flat black silhouette. Compare
   against one phone photo of a real print (T07).
6. ~~The wrist view shows the fingertips, as on the real cell.~~ Corrected 2026-10-28: this was an assumption, not
   an observation. The starter's calibrated wrist camera sits 98 mm beside the tool and looks along it, so the
   fingertips are out of frame in every render. Confirm with one real wrist frame; the camera is not changed.
7. Measured s/img × remaining images fits the schedule; otherwise cut B to 500 per object first (PLAN 6.3).

## 8. Open items for the user

| Item | Why it matters | Proposed |
|---|---|---|
| ~~Isaac version~~ | Solved: both hosts run 6.0.0 (section 5) | — |
| Data location | 30 GB in a crowded home folder | Use the shared folder as soon as karriv grants access (`DATA_ROOT=/data/ai_club/...`) |
| Robot asset on compute nodes | The ER-4iA USD loads from Isaac's asset server; compute nodes may have no internet | The pilot shows it. If it fails, copy the asset once and set `FANUC_ROBOT_USD` |
| Zone positions | The spare store and reject tray must be reachable and visible on the real cell | Check during the first real slot |

## 9. Order of work

T07 USD + look check → T06 defect meshes → pilot (1.5k) → gates → T10 full sets (108 chunks) → T11 Cosmos Transfer
pilot. T13 / T14 (Codex) start training on set A as soon as its 32 chunks are done.

## 10. Training (owned by Codex: T13 detector + classifier, T14 PatchCore + FoundationPose, T18 policy)

Training starts as soon as a set's chunks pass the QA gate. Compute, per the stack:

| Model | Data | Compute |
|---|---|---|
| Detector (Faster R-CNN or RT-DETR) | Set A (table + side) | Brev A100 or Rosie `dgxh100` |
| Wrist look-alike classifier | Set B crops | Brev A100 / Rosie `dgxh100` (small; a T4 also works) |
| anomalib PatchCore | Set C good-only (train), C defects (test) | Rosie T4 is enough |
| Build check | Set D + Cosmos Reason 2 (2B) | GB10 (endpoint) |
| GR00T N1.7 / SmolVLA / ACT | Set E (Isaac Lab Mimic, T12) | Brev H100 |

Every model is evaluated per scenario from `labels/` (per product, per defect type, per build state, traps), not
only on overall accuracy. Real-cell images are the final test, never training data.

## 11. Edge-case and close-up sets (added 2026-10-28)

The coverage gate (section 3) shows every planned case exists, but A–D sample nominal conditions. These sets push
one axis each past the nominal ranges, so the models also learn the bad cases the real cell can produce. Labels
and folders are the same as A–D; training uses the union.

| Set | For | Images | What is pushed | Host |
|---|---|---|---|---|
| F | Detector | 8,000 | Light 0.1–3.5× and 2700–7500 K, noise up to 4 %, motion blur 40 %, JPEG 45–95, camera mount drift 15 mm / 2.5°, wider PLA look, rare resting poses | Rosie T4 |
| G | Detector | 8,000 | Clutter: 90 % touching clusters, arm in view 80 %, 4–6 spares, 1–3 rejects, 1–3 foreign objects, up to 3 defects, traps 60 % | GB10 |
| H | Wrist classifier | 12,000 | Rare resting poses as often as common ones, a look-alike always present, touching neighbours, target up to 40 mm off-centre, tilt up to 30°, 90–320 mm standoff, hard look | Rosie T4 |
| I | PatchCore | 6,000 | Good parts under hard light and rare poses (must not alarm); 40 % defect test images | GB10 |
| J | Build check | 6,000 | The decision boundary: high 0.4–1.5 mm, tilt 1–4°, seated within 0.3 mm print tolerance. Labels carry `build_offset` (mm / deg) so the threshold is chosen from data | GB10 |
| K | Classifier + detector | 2,400 | Every object close up (70–180 mm), every resting pose, 80 per object, hard look | GB10 |
| L | Build check | 1,600 | Every product at every step close up, all build states (high 0.4–4 mm, tilt 1–12°) | GB10 |
| M | Build check | 6,000 | Mis-assembly from the 200-case edge package: shifted 0.5–4 mm off the pegs (16 directions), turned 90/180/270° (only turns that change the visible shape by ≥ 1 mm), tilted 1–12°, look-alike swaps. `build_offset` adds `shift_mm`, `yaw_deg` | Brev |

Every scene of F–L lays out and every arm pose solves (checked 2026-10-28). A–C specs are byte-identical to before
the change apart from redrawn unreachable arm poses; D gains the `build_offset` field.

**Wrist camera jitter stays nominal (fixed 2026-10-28)**: the 15 mm mount-drift jitter of the hard-look sets first
also moved the wrist camera, whose eye then sat inside its own lens housing (`WristCamLens`): 22 of 367 set I frames
showed only the housing. Found by the depth check of the sampled QA. Drift now applies to the table / side cameras
only; set I was re-rendered and set H re-specified before rendering. Sets without depth (B, H) cannot show this in
QA, so the regression test checks the spec instead.

**Mis-built parts rest on the stack (fixed 2026-10-28)**: a tilt about the part's bottom centre pushed half the
part into the fixture (up to 1,350 mm³ at 8° for a housing), and some look-alike swaps overlapped the part below.
`sdg/tools/build_clearance.py` measures (manifold3d booleans, 0.01 mm) how far each part must rise to clear the
fixture plus the parts under it, per tilt direction and angle, shift, turn and swap; the sampler lifts the part by
that amount. Sampled D, J, L, M specs: overlap ≤ 0.05 mm³, resting gap ~0.1 mm. `build_offset.high_mm` is now the
real lift for every state. A–C and E–K specs are byte-identical; the 1,155 changed D scenes are re-rendered
(`sdg/tools/supersede_changed.py`).

**From the 200-case edge package (reviewed 2026-10-28)**: wrong-part swaps, the six defect classes, glare / colour
temperature, blur, clutter, rare poses and mount drift were already rendered; shifts and wrong-way turns became
set M. Not images: grasp failures, drops, recipe changes, pauses, stale sensor data (GR00T / Cosmos demos, set E).
Training-time augmentation, not rendering: low resolution, depth holes on glossy faces, a missing camera.

**Path tracing tried and dropped (2026-10-28)**: a 3-scene K smoke test with Isaac's path tracer on the GB10 took
164 s/img, and the capture did not follow the arm pose and returned empty instance masks and depth. Real-time ray
tracing on the same scenes: 0.6 s/img, 0 QA errors. Realism comes from the randomized look plus Cosmos Transfer (T11).

**Not covered, and why**: new defect geometry beyond the six types (needs new CAD booleans, T06 follow-up); a part
held in the gripper (no grasp simulation in the renderer); real-camera lens distortion (the starter cameras are
ideal pinholes; Cosmos Transfer, T11, is the planned realism step).

## 12. Watchdog, and what happens when the images are done

**Watchdog** (`sdg/watchdog.py`, one pass every 10 min per host):
- Every finished chunk: 40 random scenes go through the QA gate and the scene count is compared with the specs.
  Result in `_QA.json`; overlay sheet in `$DATA_ROOT/watchdog/sheets/` for the by-eye check.
- A failed chunk (`_FAILED.txt`) is retried up to 3 times (Rosie: re-submitted; GB10: re-run). The generator skips
  scenes that have labels, so a retry renders only what is missing. After 3 failures it reports `GAVE UP`.
- GB10: when nothing renders it starts the next unfinished chunks of its plan; a render that writes no label for
  30 min is killed and retried.
- `$DATA_ROOT/watchdog/status.json` lists per set: chunks done, scenes, failures, QA failures and
  `ready_for_training` (every chunk done and checked). One JSON line per pass in `watchdog/log.txt`.

**Run order (rebalanced 2026-10-28 02:25)**: Rosie B → F (Slurm, 4 at a time). GB10 C → D → A → K → L. Brev 2x L40S
(`sdg/brev/render_queue.sh`, $35 cap) G, I, J → H. All three are expected to finish around 05:00–06:00 CDT.

**Rosie denoiser (fixed 2026-10-28)**: Isaac logged `Failed to create NGX context` on every T4 task, so the RTX
denoiser never ran and every Rosie frame kept its ray-tracing speckle (seen on the set B overlay sheets; the sampled
QA checks labels, not image noise). The job script now also mounts `libnvidia-ngx.so.1`, `libnvoptix.so.*` and
`/usr/share/nvidia` (OptiX weights). Verified clean on a T4; 1.6 s/img instead of 2.7. The 19 noisy set B chunks
are kept in `$DATA_ROOT/rejected/B_no_denoiser_20261028/` and re-rendered; they are not used for training.

**When a set is `ready_for_training`** (Claude):
1. Rosie: `sdg/rosie/collect_spark.sh` pulls the GB10's finished chunks (direct Rosie → GB10 SSH, safe to repeat).
2. `uv run python -m sdg.coverage` on the final specs; read three overlay sheets per set by eye.
3. Write `coord/HANDOFF.md`: set, chunk count, QA result, data path. Codex starts from that entry.

**Training (Codex, T13 / T14, section 10)** starts per set, not after everything:

| As soon as | Codex starts | Data |
|---|---|---|
| C ready | PatchCore, good-only | C good (+ I good later); C and I defects are the test set |
| B ready | Wrist look-alike classifier | B crops (+ H, K later) |
| A ready | Detector | A (+ F, G, K later) |
| D ready | Build-check thresholds + Cosmos Reason 2 prompts | D (+ J, L later) |

Each model is reported per scenario from `labels/` (product, defect type, build state, trap, edge set), never only
as one overall number. Real-cell images are the final test, never training data.
