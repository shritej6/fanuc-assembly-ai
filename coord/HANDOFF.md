# Handoff log

Newest first. One entry per finished task.

## 2026-10-28 · Old sets auto-labelled installed / fallen (for CODEX) · CLAUDE · read before T13 training

- Old sets A-M have no `rest` field and 71 % of their loose parts lie in poses the real test-bed only shows as fallen
  (e.g. rear guards M1-3_04 flat). Using them unlabelled would teach the `fallen` flag that a sleeping part is fine.
- `sdg/tools/rest_labels.py` wrote `_REST.json` into all 209 old chunks on fa-r4 (`~/old-sets/synthetic`):
  `{"images": {"<index>": {"<uid>": "installed" | "fallen"}}}`. Same rule as the new sets; tests show it reproduces the
  rendered N / Q labels exactly. Totals: 204,126 installed, 503,125 fallen over 78,200 images.
- Training: new sets read `objects[].rest`; old sets read `_REST.json` by image index + uid. Weight the new sets higher
  (they are 91 % installed like the real bed). Objects absent from the sidecar (installed, mis-builds, bases, foreign)
  have no rest state: mask the fallen loss for them.

## 2026-10-28 · Guide orientation check (for CODEX) · CLAUDE · T25 labels verified, training block can be lifted

- Codex flagged `N/chunk_0009/002252` (flat guides labelled `rest: installed`) and blocked H100 training. Checked:
  the guides M4/M5/M6_03 are 5 mm U frames that sit FLAT on the housing (CAD `connector_family.py` stack: frame z 18-23,
  back block up, lock legs through it). The user compared our assembled render with the printed parts: correct.
- Measured over every rendered N/R/O/P/Q label on fa-r4 (19,000 images): 167,571 loose parts labelled `installed` all
  have the product's up axis (< 2 deg), 16,420 labelled `fallen` all differ; 0 wrong labels. Fallen share N 7.2 %,
  R 6.9 %, O 9.9 %, P 3.0 %, Q 31.8 % (as configured).
- New test `test_installed_rest_has_the_same_up_face_as_the_part_inside_its_product` ties the bed pose to the product
  pose for all 24 parts (shown to fail when M4_03 is stood on its edge).
- Next: lift the training block; no re-render, no `pick_ready_pose`. If a real part ever differs from CAD, fix the CAD
  and this test, then re-render only that product.

## 2026-10-28 · Model behaviour decided (for CODEX) · CLAUDE · read before T13 / T14 / T18 training

The user answered the model questions before training (all rows dated 2026-10-28 in `coord/DECISIONS.md`):
- Detector: exact part of 30 + location + `fallen` flag; look-alikes need the wrist classifier to agree before a pick.
- Unsure look-alike after a second wrist view → spare (LOOKALIKE_UNRESOLVED). PatchCore flag → reject + spare.
- Identity re-checked after placing (a wrong housing seats fully in F03); seated = within `insertion_tolerance.json`.
- Accuracy first. H100 order: RT-DETR → wrist classifier → set-P model → PatchCore (per object, parallel) →
  FoundationPose validation (pretrained) → copy artifacts, close the box. PatchCore's dependency audit must pass first.
- Data: new sets N R O P Q render on Brev `fa-r4` (+ GB10) with Cosmos for 50 % of scenes (`rgb_cosmos/`, `_COSMOS.json`);
  per-chunk QA in `watchdog/status.json` there. Weight the new installed-orientation sets up. Ready ~01:40 CT Oct 29.
- Spares: any of the 24 parts, never a base (`spares_any_part` in sets N-Q). Set Q re-rendered with it; the old Q is
  in `superseded/Q_oldspares/` on fa-r4: do not train on it. 40 earlier N / R / P chunks keep the six-part spare list.

## 2026-10-28 · T25 test-bed data (in progress) · CLAUDE (for CODEX) · parts stand as installed, any yaw

- Rule from the user: the arm only picks/places straight down and cannot turn a part over, so every loose part and
  spare stands in its installed orientation at any yaw. A part that has fallen over must go to the reject zone and be
  replaced from the spare store (no "flip -> rework"). Planner work for you: reject on `rest == "fallen"`-like evidence.
- Old sets A-M drew random drop poses: only 14-27 % of loose parts were in installed orientation and M1/M2/M3_04
  (standing on an 8 mm edge) never were. Keep A-M in training (they teach fallen parts) but weight the new sets up.
- New sets (sdg/configs/sets.yaml, rendering next): N 20k + R 10k (detector, table+side, installed, 10 % fallen),
  O 24k (wrist identity close-ups at ONE height, 147-153 mm above the part's top face, so mm-per-pixel is fixed and
  1-1.5 mm look-alike size gaps stay visible: the planner should take identity close-ups at 150 mm above the part top),
  P 12k (wrist insertion alignment: top part on/over its pegs with labelled `build_offset` shift/yaw/tilt/height),
  Q 4k (50 % fallen + spare store).
- New label fields: loose work-area parts and spares carry `rest`: `installed` | `fallen`. `stable_pose` stays an int:
  the matching drop pose, or -1 when the installed orientation is not a drop pose (M1/M2/M3_04 standing).
- `sdg/data/insertion_tolerance.json` (from CAD): per part the yaws it fits at (11 one way only, 11 every 180 deg,
  2 any), yaw tolerance (keyed parts +-1.4-2.4 deg, caps 6.8) and sideways tolerance (0.38-0.50 mm). Use it to yaw-align
  before the straight-down insert and to size the small search when a part does not seat.

## 2026-10-28 · T11 Cosmos Transfer · CLAUDE (for CODEX) · 10 % of A, F, G, B, H, K restyled tonight

- Do not wait for Cosmos: smoke-test and benchmark on the renders now. The production prepare gate (10-20 % accepted
  Cosmos images in A, F, G, B, H, K) passes only after the import below.
- What lands: 10 % of every scene in A, F, G (table + side, ~1,600 images) and B, H, K (wrist, 3,840 images) at
  `<chunk>/rgb_cosmos/<cam>/<idx>.jpg`, same size and same labels as `<chunk>/rgb/<cam>/<idx>.jpg`. Per-image results
  in `<chunk>/_COSMOS.json`: `accepted` (false = not written), `edge_fscore`, `worst_object_fscore`, and
  `cosmos_objects` / `rendered_objects` (how many parts kept the Cosmos look).
- How they are made: Cosmos Transfer 2.5 (edge very_low + vis 0.5) restyles the frame. A part keeps its Cosmos pixels
  only if its own edge F-score is >= 0.85 and its mean colour is within 35 grey levels of the relit render; every
  other part is pasted back from the render, relit and feathered (`sdg/cosmos/composite.py`), so labels stay exact.
  Calibration (60 images, 1,285 parts): 73.9 % of parts keep the Cosmos look. Image gate: whole-image edge F-score of
  the Cosmos frame >= 0.85 (calibration min 0.872, median 0.948).
- Where: A, F, G 720 images on GB10 (`spark:~/fanuc-assembly-data/cosmos/prodAFG`); the other 5,093 on two RunPod
  A100 pods (`fa-cosmos-1/2`, shards built on Rosie in `~/fanuc-assembly-data/cosmos/shards`). Import on Rosie:
  `python3 -m sdg.cosmos.restyle import <bundle> --min-fscore 0.85`.
- C, I, D, J, L, M (PatchCore / build-check) are not restyled.

## 2026-10-28 · T13 + T14 READY · CLAUDE (for CODEX) · start training on the finished synthetic sets

- User said go (2026-10-28 ~08:00): T13 and T14 are READY for Codex.
- Label decision (user, `coord/DECISIONS.md`): all 30 objects are separate classes, in the detector and the wrist
  classifier. Identical shapes (`contracts/taxonomy.json` look-alike groups, e.g. F01=F02, F04=F05=F06) stay separate;
  report their confusion per group in the eval so it is visible.
- Data: Rosie `~/fanuc-assembly-data/synthetic/<set>/chunk_NNNN/`, read `labels/` only (never `superseded/`).
  Detector A, F, G · wrist classifier B, H, K · PatchCore C, I (good targets train, defect targets test only) ·
  build check D, J, L, M. 78,200 scenes / 104,000 images, every chunk sample-checked, coverage 0 thin cases.
- Real-cell photos are evaluation only, never training.
- Update (user, 2026-10-28): detector is **RT-DETR**. Train on Rosie's V100 partition `dgx` (2 of 3 nodes idle);
  the H100 partition `dgxh100` is booked, `sbatch --test-only` estimated a start on 2026-11-06 (after the deadline).
  Cosmos-restyled copies of 10–20 % of each set will be added by Claude (T11) once they pass the shape-drift check.

## 2026-10-28 · T06 + T10 (running) · CLAUDE · defect meshes, production rendering, edge sets, watchdog

- Changed: `sdg/tools/defect_variants.py` + `cad/defects/manifest.json` (144 variants, regenerated per host),
  `sdg/coverage.py`, `sdg/qa.py`, `sdg/watchdog.py`, `sdg/wrist_ik.py`, `sdg/tools/check_wrist_ik.py`, sets F–L in
  `sdg/configs/sets.yaml`, `sdg/rosie/collect_spark.sh`.
- Running: Rosie B → F → H (Slurm 289876 / 289881 / 289883, 4 at a time); GB10 C → D → A → G → I → J → K → L.
  Data: Rosie `~/fanuc-assembly-data/synthetic/<set>/chunk_NNNN/` (GB10 output pulled in by `collect_spark.sh`).
- Monitor (Codex too): `ssh rosie cat ~/fanuc-assembly-data/watchdog/status.json` and the same on the GB10
  (`hackathon1@spark1`). Per set: `chunks_done`, `failed`, `qa_bad`, `ready_for_training`. A chunk's `_QA.json`
  holds its sample check; sheets in `watchdog/sheets/`. Report anything in `failed` / `qa_bad` / `GAVE UP` events.
- Facts (measured): GB10 0.41–0.56 s/img; Rosie T4 2.7 s/img; every checked chunk so far 0 QA errors. Every arm pose
  of every set solves with the starter IK. The wrist camera does not see the fingertips (calibrated mount).
- For Codex: start T13 / T14 per set the moment `ready_for_training` is true (table in `docs/SDG_PLAN.md` section
  12). New label field: `build_offset` {high_mm, tilt_deg} on build-check targets (D, J, L). Evaluate per scenario.
- Next: Brev (after `brev login`) can take chunks of any set; T07 PLA look check needs one phone photo of a real print.
- **Set C READY FOR TRAINING (2026-10-28 01:35)**: 16/16 chunks, 8,000 wrist images, every chunk sample-checked with
  0 errors, on Rosie at `~/fanuc-assembly-data/synthetic/C/`. Codex can start PatchCore (T14): train on scenes whose
  wrist target has `variant == "ok"`, keep the defect scenes as the test set.
- Brev 2x L40S (`fa-render-l40s`) renders G, I, J with `sdg/brev/render_queue.sh`; deleted when done ($35 cap).
- **ALL SETS RENDERED (2026-10-28 06:45)**: A-D and F-M, 78,200 scenes / 104,000 images, all on Rosie under
  `~/fanuc-assembly-data/synthetic/<set>/chunk_NNNN/` (read `labels/`; ignore `superseded/`). Every chunk
  sample-checked with 0 errors. Coverage over all sets: 0 thin cases (`docs/SDG_COVERAGE.md`). Brev deleted ($21).
- **Set A READY FOR TRAINING**: 32/32 chunks, 8,000 scenes (16,000 table + side images, detector nominal).
  Rosie copy identical to the GB10 (48,065 files incl. one retry note).
- **Set M READY FOR TRAINING**: 12/12 chunks, 3,000 scenes (wrist + side, mis-assembly: shifted / rotated / tilted /
  wrong part, all resting on the stack). Rosie copy identical to Brev (18,024 files).
- **Set B READY FOR TRAINING (2026-10-28 06:30)**: 48/48 chunks, 24,000 wrist images (~800 per object), chunks 0-41
  rendered on Rosie with the denoiser fix, 42-47 on the GB10 (same specs, byte-identical; copy verified). Every chunk
  sample-checked including the render-noise check, 0 errors. The 19 noisy pre-fix chunks stay in
  `~/fanuc-assembly-data/rejected/` and are not part of the set.
- **Set F READY FOR TRAINING (2026-10-28 06:20)**: 16/16 chunks, 4,000 scenes (8,000 table + side images, detector
  hard look). Every chunk sample-checked, 0 errors; Rosie copy identical to the GB10 (24,032 files, same bytes).
- **Set H READY FOR TRAINING (2026-10-28 06:05)**: 24/24 chunks, 12,000 wrist images (classifier hard cases: rare
  poses, a look-alike always present, off-centre / tilted / far views, hard look). Every chunk sample-checked, 0
  errors; sample sheets checked by eye (no depth in H, so a blocked camera can only be seen): none blocked. Rosie copy
  identical to Brev (36,048 files, same bytes).
- **Set D READY FOR TRAINING again (2026-10-28 05:45)**: re-render with the resting fix done, 12/12 chunks, 3,000
  scenes (wrist + side). Every stored spec equals the current sampler's on both the GB10 and Rosie (0 stale scenes),
  every chunk sample-checked, 0 errors. Read `labels/` only: `superseded/` holds the old labels for the record.
- **Sets K and L READY FOR TRAINING (2026-10-28 04:50)**, on Rosie (collected from the GB10 with
  `sdg/rosie/collect_spark.sh`): K 5/5 chunks (2,400 wrist close-ups, every object and resting pose); L 4/4 chunks
  (800 scenes, wrist + side, build states close up, resting fix applied). Every chunk sample-checked, 0 errors.
- **Set J READY FOR TRAINING (2026-10-28 04:25)**: 12/12 chunks, 3,000 scenes (6,000 wrist + side images), build
  check at the decision boundary, rendered with the resting fix (no clipping). All chunks sample-checked, 0 errors;
  Rosie copy identical to Brev (18,024 files, same bytes). Labels: `build_state`, `build_offset` {high_mm, tilt_deg}.
- **Sets G and I READY FOR TRAINING (2026-10-28 04:00)**, on Rosie under `~/fanuc-assembly-data/synthetic/`:
  G 16/16 chunks (4,000 scenes, 8,000 table + side images, detector clutter); I 12/12 chunks (6,000 wrist images,
  PatchCore hard look: good targets extend the training set, defect targets are test only). Every chunk sample-checked
  with 0 errors; Rosie copy verified file-for-file against Brev (same file count and bytes).
- **Build-check labels changed (2026-10-28 03:00)**: mis-built parts now rest ON the stack (no clipping into the
  fixture); `build_offset.high_mm` is the real lift for tilted / swapped / shifted / turned parts. Set D: 1,155 scenes
  re-rendering on the GB10 — do not train on D until it is marked ready again. New set M (6,000, Brev): states
  `shifted`, `rotated` added; `build_offset` there also has `shift_mm`, `yaw_deg`. See `docs/SDG_PLAN.md` section 11.

## 2026-10-27 · T08 (prep) · CLAUDE · synthetic data plan, scene sampler, Isaac generator draft

- Changed: `docs/SDG_PLAN.md`; `sdg/` (configs, CAD geometry tool + data, scene sampler, post-processing, Isaac stage
  builder and render loop, Rosie Slurm scripts); `cad/stl/` (the 30 print STLs); `sim/starter/` (organizers' starter,
  unmodified, hashes verified); `numpy` added to the dependencies.
- Run: `uv sync && uv run pytest -q` (34 pass). `uv run python -m sdg.scene_sampler --set A --chunk 0 --out /tmp/a.jsonl`.
- Facts: every STL registers onto the CAD assembly within 0.003 mm; every resting pose sits on z = 0; all 54,000
  scenes (108 chunks) lay out. Rosie has Isaac 6.1.0 / 5.0.0 containers, **no 6.0**; `/home` is 87% full.
- Not tested: `sdg/isaac/*` and `sdg/rosie/*`; they need Isaac on Rosie. The pilot is their first run.
- For Codex (review): the label fields in `docs/SDG_PLAN.md` section 4 are what T13/T14 will read.
- Next: U07 (user picks the container), T07 USD + black PLA look check, T06 defect meshes, then the pilot.

## 2026-10-27 · T01 · CLAUDE · repo scaffold

- Changed: repo layout, `CLAUDE.md` / `AGENTS.md` rules, `coord/` (TASKS, DECISIONS, HANDOFF), `contracts/` v1 (9 schemas
  + `taxonomy.json` with measured look-alike groups), root `pyproject.toml`, `tests/test_contracts.py`, vendored official
  client in `robot/cell/` (upstream commit in `UPSTREAM_COMMIT`, unmodified).
- Run: `uv sync && uv run pytest` (12 pass); `cd robot/cell && uv sync && uv run run.py --mock` → "✓ Ready for the real robot"
  with the placeholder policy.
- Facts: F01=F02 and F04=F05=F06 are identical shapes (0.0 mm); look-alike groups (≤ 2.5 mm) are listed in
  `contracts/taxonomy.json`. All contract poses are UF0 mm/deg.
- Next: T02 assembly sheets, T03 CAD risk ranking (CLAUDE); T04 catalog and T05 adapters are ready for Codex once the user
  marks them READY.
