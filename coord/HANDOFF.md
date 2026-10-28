# Handoff log

Newest first. One entry per finished task.

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
