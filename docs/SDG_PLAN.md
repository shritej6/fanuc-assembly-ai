# Synthetic image generation plan (T06 / T07 / T08 / T10)

Status 2026-10-27. Extends `docs/PLAN.md` section 6. Stack unchanged: **Isaac Sim + Replicator** renders, Cosmos Transfer
optional (T11). Real-cell images are for evaluation only, never training.

## 1. What is ready and what is not

| Piece | File | State |
|---|---|---|
| CAD geometry: installed pose of every part on its base + resting poses when loose | `sdg/tools/object_geometry.py` → `sdg/data/object_geometry.json` | **Done, verified.** All 30 STLs register onto the CAD assembly within 0.003 mm. Every resting pose is checked to sit on z = 0 |
| Table zones (work area, spare store, reject tray) | `sdg/configs/cell_layout.yaml` | **Proposed.** Must be checked on the real cell before T22 |
| Set sizes and randomization | `sdg/configs/sets.yaml` | Done (sizes from PLAN 6.1) |
| Scene sampler (layout, defects, look-alike traps, build states, wrist poses) | `sdg/scene_sampler.py` | **Done, tested.** All 54,000 scenes lay out in about 16 s on a laptop; 16 behaviour tests |
| Post-processing + label assembly | `sdg/postprocess.py` | Done, tested (6 tests) |
| Isaac stage builder + render loop | `sdg/isaac/stage.py`, `sdg/isaac/generate.py` | **Written, UNTESTED.** Runs only inside Isaac; the pilot is its first run |
| Rosie jobs | `sdg/rosie/{setup.sh,submit.sh,render_chunk.sbatch}` | Written, UNTESTED (syntax-checked) |
| Starter cell (robot, gripper, cameras, lights) | `sim/starter/` | Vendored unmodified; all 38 files match the starter's own SHA-256 list (`MANIFEST.json`) |
| Defect meshes | `cad/defects/<id>__<variant>.stl` | **Not started (T06).** The generator refuses to render a defect label without its mesh |
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

Total: **54,000 images in 108 chunks**. E (GR00T demos) is sized after the scripted baseline works (T12).

## 3. Scene rules (what the sampler guarantees, and the tests check)

- 3 different products. Each product's own base type, at a random position and yaw, never touching another base
  (25 mm gap). Bases never move, matching the real cell.
- Every part of each product appears exactly once: installed on its base (steps 1..k, exact CAD pose) or loose.
- Loose parts lie in a CAD resting pose, weighted by how likely that pose is. Parts may touch but never overlap.
- Look-alike trap: a part from a product that is **not** on the table but shares a family with a present part.
- Spares sit in the 6 store slots (the PRELIMINARY high-risk list). Rejects go in the tray (fewer when it is full).
- Foreign objects: M3 screw, tape roll, 10 mm cube.
- Every object centre sits 180–490 mm from the robot base. Wrist TCP poses stay inside the cell's TCP box.
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

Rosie facts checked 2026-10-27:
- `teaching` partition with 20 T4 nodes, 4 GPUs each.
- Containers in `/data/containers`: `isaac-sim-6.1.0.sif`, `isaac-sim_5.0.0.sif` and `isaac-sim-20231103.sif`.
  **There is no 6.0 container.**
- `/home` is 87% full cluster-wide, and the user's home folder already holds 201 GB.

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
6. The wrist view shows the fingertips, as on the real cell.
7. Measured s/img × remaining images fits the schedule; otherwise cut B to 500 per object first (PLAN 6.3).

## 8. Open items for the user

| Item | Why it matters | Proposed |
|---|---|---|
| **Isaac version**: the stack says 6.0; Rosie has 6.1.0 | The starter was validated on 6.0. A 6.0 container would have to be pulled (about 10 GB) into a `/home` that is 87% full | Run the pilot on `isaac-sim-6.1.0.sif`. If the starter cell fails to build on it, pull 6.0. **Your call; nothing runs until you choose** |
| Data location | 30 GB in a crowded home folder | Use the shared folder as soon as karriv grants access (`DATA_ROOT=/data/ai_club/...`) |
| Robot asset on compute nodes | The ER-4iA USD loads from Isaac's asset server; compute nodes may have no internet | The pilot shows it. If it fails, copy the asset once and set `FANUC_ROBOT_USD` |
| Zone positions | The spare store and reject tray must be reachable and visible on the real cell | Check during the first real slot |

## 9. Order of work

T07 USD + look check → T06 defect meshes → pilot (1.5k) → gates → T10 full sets (108 chunks) → T11 Cosmos Transfer
pilot. T13 / T14 (Codex) start training on set A as soon as its 32 chunks are done.
