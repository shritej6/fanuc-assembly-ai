# Handoff log

Newest first. One entry per finished task.

## 2026-10-30 · T33 · CODEX · M1_01/M3_01 size-twin contract

- Changed: `contracts/taxonomy.json` now marks M1_01 and M3_01 as symmetric look-alikes, so both require the wrist
  identity check despite their surface-distance pair exceeding the original near-surface threshold. The contract
  description records that same-envelope size twins are also look-alikes.
- Evidence: Claude T12 printed-STL analysis reports equal 36 x 49 x 16 mm outer dimensions and different internal
  pockets; no real-cell measurement is claimed.
- Regression: `test_same_size_control_housings_require_a_lookalike_check` plus the existing symmetric taxonomy gate.
- Run: `UV_CACHE_DIR=/private/tmp/fanuc-t33-uv-cache uv run pytest -q` (13 passed),
  `UV_CACHE_DIR=/private/tmp/fanuc-t33-uv-cache uv run ruff check .`, and `git diff --check`.
- Next: both agents review the contract PR before merge; consumers then regenerate any derived look-alike tables.

## 2026-10-30 · T09 atomic T29 turn/TCP consumption · CODEX

- Merged T04 `efd698a` and replaced T09's independent scalar-TCP parser with the typed
  `RecipeStep.preferred_grasp`. Pick, insert, reject, release, retry, and staged recovery now carry the same atomic
  source turn/TCP pair; the algebraic alternate turn is not CELL-authorized.
- `PreparedPlanStep` rejects identity mismatches and any plan record whose turn differs from the authorized pair.
  The raw T29 snapshot remains hash-verified, while motion code receives only the catalog-authorized pairing.
- Updated the 24-part x 3-yaw motion matrix for the turn-indexed flange-axis TCP. Exact M1_01 and non-identity M1_04
  regressions prove the former pad point is never commanded and the 1.8 mm lateral flange offset is preserved.
- Full gate: `uv run pytest -q` = 630 passed; `uv run ruff check .`, `git diff --check`, and the protected official
  `robot/cell/cell_client.py` check passed. No real-cell command was issued.
- PR #9 remains blocked for CELL on PR #15 physical/provenance evidence and the reviewed GR00T semantic executor,
  live perception, collision, measured-limit, view-motion, and release-authorization dependencies.

## 2026-10-30 · T04 T29 turn/TCP atomic pin · CODEX

- Repinned the immutable T29 grasp snapshot to `fac05905338b82a90744e934268f4a30bf412fc7` with SHA-256
  `dbd00b5082042e8a326b380d237890449bfb2206243c1753b7ebe2d940435977`.
- Rejected the superseded single-vector pad point and require the turn-indexed flange-axis TCP schema. Canonical keys,
  strict numeric finite values, and exact source fields now fail closed on malformed data.
- Added `PreferredGrasp(turn_deg, tcp_local_mm)` to every typed `RecipeStep`. This binds the source-tested wrist turn to
  its matching TCP and intentionally withholds the algebraic 180-degree alternative from CELL authorization.
- Regression coverage spans all 24 movable parts plus legacy vectors, missing/unknown turns, NaN, strings, and booleans.
  Full gate: `uv run pytest -q` = 76 passed; `uv run ruff check .` and `git diff --check` passed. Independent Codex
  re-review found no remaining code issue.
- Merge remains blocked on PR #15's immutable CAD/finger/STL and physical-validation provenance. T09 must consume
  `RecipeStep.preferred_grasp` atomically; it must not reopen the raw T29 JSON or authorize the alternate turn.

## 2026-10-30 · T30 perception port · CLAUDE

- **What changed.**
  - Merged `origin/feature/T09-state-machine` (2204663) into this branch. Codex's `assembly/*` files are unchanged.
  - Added `perception/port/`: `CellPerceptionPort` implements the T09 `PerceptionPort` (`scan`, `pre_pick`,
    `post_place`, `post_transfer`, `collision_scene`). Every failure is a typed `PerceptionPortError` (a
    `ControllerPortError`).
  - Detection and sizing:
    - class-agnostic duplicate suppression with a top-1 label;
    - FoundationPose STL → installed-local normalisation (`world_from_stl @ inv(T_local_from_stl)`);
    - the look-alike size gate against the pinned T12 `objects.yaml` @74d749a (sha256-checked), using footprint,
      height, CAD top cues, size twins (M1_01/M3_01, M1_02/M2_03) and all 30 objects, so a piece that fits nothing is
      reported as foreign.
  - The user's careful defect band (reject ≥ 0.994, second look 0.5–0.994) maps to `print_defect`/reject.
  - Unsure → typed `CloseupReport`: second look with a `ViewRequest` (150 mm, 30° oblique, LOOKALIKES side), then
    Andon with both pictures.
  - Post-place seat (±0.3 mm) and tilt (1°) are certified only with a MEASURED bound, refuted when clearly off, and
    otherwise sent to the operator.
  - `precision_gate_tolerances()` returns the INTERFACES gates.
  - `contracts/`, `assembly/` and `robot/cell/cell_client.py` are unchanged. Contract proposals are in
    `perception/port/README.md`.
- **Measured on real captures `r20261029-995508`** (evaluation only):
  - Suppression cut T31 table+side duplicates from 125 to 36 against the agent visual count (412 objects) and lost
    no object.
  - Wrist depth noise at 364 mm is 7.64 mm std per pixel (80 patches). Scaled to 150 mm that is 1.3 mm
    (PRELIMINARY), so depth alone cannot separate the 1 mm M1_02/M2_02 difference.
- **Run.**
  - `uv run pytest -q`: 958 passed, 1 skipped. Of those, 71 are the new `tests/test_port_*.py`, all on fakes with no
    GPU.
  - `uv run ruff check .`: clean.
- **Not done.**
  - No real-weight smoke: Rosie was unreachable from the Mac (DNS for dh-mgmt2 failed).
  - No controller-in-the-loop run with real images.
- **Next / blocked.**
  - A FoundationPose `PoseEstimator` service (T28/main).
  - Measured `CloseupThresholds` (T27/T28).
  - Merge T27 `d869bad` for the defect-head loader.
  - A measured FoundationPose repeatability to certify the 0.3 mm seat (first real-cell minutes).
  - An arm-corridor radius or a robot-link model for intrusion (Codex).
  - Controller motion for the second look and an Andon hook that reads `port.closeup()` (Codex).

## 2026-10-30 · T09 robot-brain case-rule hardening · CODEX

- Implemented bounded case handling from `docs/ROBOT_BRAIN.md` section 3 without adding unsourced motion targets:
  missing products are deferred while another requested product can run; a spare is not consumed before that work;
  Dropped parts use a fresh post-drop scan: upright re-plans, fallen/defective rejects then uses a spare, and unobserved
  Holds. Work-area intrusion clears the stale queue, requires a fresh scan, performs only a safe state-specific
  replan, and Holds during pending install or staged recovery states.
- Added a separate typed `FinalProductVerifier` trust boundary. Product completion requires fresh, complete evidence
  bound to the exact product, base, and four installed instance IDs; two distinct view IDs; the same exact four
  identities in both views; two Boolean results; identity agreement; and reviewed physical-motion provenance. The
  deterministic mock implementation is explicitly `mock_only` and requires an opt-in flag, so it restores three-product
  mock completion without authorizing cell motion. Construction rejects every callback-only production verifier before
  the first action: T30 must first supply controller-owned, safety-approved physical view motions with independently
  verified endpoints; synchronous camera calls and self-reported capture metadata are not proof. Verified final failure
  also Holds until a reviewed installed-part
  grasp/staging sequence supports reverse removal. Unsure pre-pick remains blocked on calibrated alternate-view motion.
  Crowded relocation remains blocked on typed blocker identity plus a calibrated remembered empty staging target.
- Won't-seat recovery now executes the
  user-approved 3.0 mm lift from `docs/ROBOT_BRAIN.md@2a751da` section 3, then carries the still-held part through the
  configured safe transit to the measured set-O inspection stop at UF0 z -168 mm before re-running `pre_pick` plus
  FoundationPose. It recomputes the candidate/base/target and performs one guarded reinsert. A second failure lifts and
  re-inspects the held part: a typed defect goes physically to REJECT before spare lookup/Andon; a good part Holds until
  T30 publishes a typed calibrated hole-clearance verdict. No generic safe-Z lift or cached target substitutes for the
  measured retry. Operator acknowledge/resume now also refuses every unresolved physical pending state, including a
  release or +3 mm recovery lift interrupted by intrusion; a fresh scan cannot silently commit or reinsert it.
- Observed gripper state is now accepted only when it is exactly the official-client value 0 or 1. Fractional or
  out-of-range feedback fails closed before perception, planning, or motion instead of being interpreted as detached.
- Removed the now-unreachable staged-rebuild hooks from `AssemblyController` instead of retaining dead production
  branches. The independently tested recovery planner/library remains intact; wiring reverse removal back into the
  controller still requires the typed offending-instance, hole verdict, installed grasp, and staging dependencies
  listed above.
- T12 contact-force dependency remains open: the real arm has no force sensor, so real-cell insertion is blocked until
  T12 supplies reviewed, measured per-part limits and the production seated-height/final-approach guards. Preliminary
  physics values are not promoted to controller limits. No contract or `robot/cell/cell_client.py` file changed, and
  no real-cell command was issued.
- Run: focused completion/controller/brain-rule/runtime/T29 suite `130 passed`; full `uv run pytest -q` `620 passed`; full
  Ruff and `git diff --check` passed; contracts and `robot/cell/cell_client.py` are unchanged from the task base. The
  official default mock passed 3 rollouts / 602 actions / 0 rejected and printed `Ready for the real robot`; the opt-in
  T09 mock passed 1 rollout / 200 actions / 0 rejected and completed all three requested products while explicitly
  reporting mock-only orchestration, not real-robot readiness. Next: Claude reviews this branch head;
  T28/T30/T12 publish the typed motion/perception/force dependencies above before the final gate can be enabled.

## 2026-10-29 · T09 critical fingertip-release follow-up · CODEX

- Recovered Claude's deleted `pr9_grasp.md` review from its session log and verified it against the pinned T29
  artifact. PR #9 head already composes `tcp_local_mm` and applies the T29 roll once for pick/insert; the stale T30
  worktree does not and must merge/rebase PR #9 before it can be considered for cell use.
- Closed the remaining release boundary: `compile_release_retreat()` now refuses to open unless the held TCP agrees
  with the sourced fingertip target within the caller's measured execution position/orientation tolerances. It no
  longer accepts a camera canonical object origin as the release point. The M1_04 regression rejects the object
  origin that is 27.9 mm below the T29 fingertip target and has a 90-degree yaw error.
- Added an end-to-end T29 matrix over all 24 parts and fixture yaw 0/90/-135 degrees. It checks canonical
  FoundationPose origin -> fingertip grasp -> CAD/T29 insertion -> roll-adjusted emitted waypoints -> verified
  release, including the nonidentity rear-guard mesh transform and the Y-closing connector housings.
- Run: `uv run ruff check .` (passed); `uv run pytest -q` (586 passed); `git diff --check` (passed). Official default
  mock completed 3 rollouts / 602 actions / 0 rejected and printed `Ready for the real robot`; opt-in T09 mock
  completed 1 rollout / 201 actions / 0 rejected with its mock-only verdict. No contract or official client file
  changed, and no real-cell command was issued.
- Next: Claude re-reviews PR #9. T30 must consume this head and implement/prove raw FoundationPose STL-frame to
  installed-local normalization before any explicit cell runtime can be enabled.

## 2026-10-29 · T27 test-bed training · CLAUDE (user override of T13 ownership, for CODEX review)

- User: train only on the new test-bed sets, and Claude makes the change now (Codex reviews; DECISIONS 2026-10-28).
- Changed (on top of T13 `ae6c1ae`): `--profile testbed` freezes N R Q (detector) + O (wrist classifier) with one
  `<part>__fallen` detector class per product part (57 classes: 30 + 3 foreign + 24 fallen), `rest` joined from the
  scene objects by uid; `--profile testbed_build` freezes set P with the six build states
  (seated / shifted / rotated / tilted / high / wrong_part) as the ResNet-18 classes. Annotations and crops carry the
  trained `label`. Cosmos gates per set use the measured shares (N R Q 0.20-0.30: one camera of half the scenes;
  O P 0.45-0.55). `perception/brev/h100_testbed.sh` runs env -> prepare+gate x2 -> detector -> classifier -> set-P.
- Verified: 147 perception tests; Rosie T4 smoke with real N / O / P chunks passed the runbook gate (finite loss,
  resume 2 -> 3 steps) for the detector (jobs 290693/290695), identity classifier (290694/290696, 290704) and set-P
  classifier (290702/290703). Smoke COCO: 141 fallen boxes in 24 fallen classes.
- Run: Claude's watcher creates `fa-h100` (8xH100, own $118 guard) at fa-r4 PIPELINE_DONE, copies code + sets,
  starts the launcher, copies results to Rosie `~/fanuc-t27-results`, deletes the box after the copy matches.
- PatchCore (`perception/anomaly/testbed.py`, `train_patchcore.py`, `perception/brev/h100_t14.sh`): one model per part,
  trained on good O wrist close-ups (test-bed, installed orientation); measured on C / I wrist close-ups of good vs
  defective parts lying in installed orientation (old-set `_REST.json`; test only, never training), half of them
  setting the threshold (anomalib `from_test` split); false alarms measured on good O test-split close-ups.
  Strict pip-audit gates its own env: Codex's pins (torch 2.10, pillow 12.3, filelock 4.1, pip 22, setuptools 59)
  FAILED it on Rosie job 290706 (81 known vulnerabilities); torch 2.14.1 + anomalib 2.7.0 + pillow 13.0.0 +
  filelock 4.2.4 + pip 26.2.2 + setuptools 84.2.1 pass with 0 and run anomalib (CPU fit/test checked).
- FoundationPose (pretrained, `perception/pose/foundationpose.Dockerfile` + `validate_foundationpose.py`, one H100
  `fa-fp`, deleted after): 438 ok loose parts in 80 set-N table scenes (23 parts), STL + RGB + 640x400 depth + the
  rendered instance mask, K scaled per axis from the USD camera, ground truth from the label transform and the jitter
  re-applied with pxr.Gf (self-check: 2749/2749 origins project inside their boxes). Measured ADD-S median 1.09 mm,
  92.0 % under 2 mm, 97.9 % under 5 mm, translation median 1.66 mm; per-part medians 0.92-1.66 mm; standing 1.09 mm
  (397), fallen 1.06 mm (41). Synthetic, perfect masks and noise-free depth: an upper bound, not a real-cell number.
  Results + freeze: Rosie `~/fanuc-t27-results/foundationpose`. Image fixes: pip pybind11 2.13 (Ubuntu's 2.9
  segfaults on NumPy 2), warp-lang 1.9.1 (1.10+ needs a CUDA 13 driver), libusb for open3d, usd-core 26.8.
- For Codex: please review the PR; the legacy profile is unchanged (default), old tests untouched except one message.
## 2026-10-30 16:55 · T12 Mimic generation + LeRobot export (agent 7, running) · CLAUDE · on `feature/T12-mimic`

- Changed: `sim/assembly_lab/mimic/` (`mimicgen.py` reads agent 3's recorder files, `generate.py` pick_insert +
  pull_out with agent 3's gates, `run_generation.py` per-host workers, `scale_generation.sh` controller,
  `stage_seeds.py` closed seeds + drop tags, `export_lerobot.py` / `validate_lerobot.py`), tests in
  `tests/assembly_lab/test_mimic_*.py`.
- Run (Mac controller): `M=sim/assembly_lab/mimic/scale_generation.sh; HOSTS="..." $M repo; $M seeds;
  MIMIC_GROUPS="pick_insert:5 pick_insert:6" TARGET=300 MAXATT=600 $M launch g1 fa-g21:0,1:1; $M status g1;
  SKILLS="pick_insert pull_out" $M export g1`.
- Gates: generated demos pass the same gates as agent 3's sources, plus carry drift and pull drift <= 0.5 mm. Seeds
  are only closed files (run10_closed.txt), minus drop tags. Every demo carries `env_manifest_md5` (repo_sync2:
  dbcfe5fd48d73cde763ffdc4316291a5).
- Measured (1 env per GPU, 3 cameras): pull_out p4 4/4 kept, 563 s per attempt; pick_insert p4 2/3 kept on the
  current code. Cameras off: 1.49 vs 1.87 s/step, so rendering is about 20 % of the step.
- Not training data: runs t*, r0, r1, p1 and tnc (the old env, or tests).
- Next: more GPUs as each gets GEN_READY, incremental export and prepare_dataset at 150 / 300 accepted, num_envs > 1.

## 2026-10-30 01:30 · T18 evaluator + real-arm guard (agent 6, in progress) · CLAUDE · not yet committed

- Changed (untracked on `feature/T12-mimic`): `policy_learn/deploy/` (`guards.py`, `groot_policy.py` `GrootPolicy`,
  `mock_check.py`, `WIRING.md` for Codex T05), `sim/assembly_lab/eval/` (22 scenarios / 110 unseen layouts, seeds from
  910000; oracle director; KPIs on the INTERFACES precision gates; `run_remote.sh`), tests in `tests/policy_learn/` and
  `tests/assembly_lab/test_eval_*.py` (69 pass, 1 skipped).
- Run: `cd robot/cell && GROOT_POLICY=stub uv run python ../../policy_learn/deploy/mock_check.py` (measured: "✓ Ready for
  the real robot", 601 actions, 0 rejected). Sim: `SKIP_READY=1 IMAGE=isaac-lab-base:3.0.0b2p1-sim6.0.0
  sim/assembly_lab/eval/run_remote.sh scripted_v1 scripted` on fa-mimic GPU 3.
- Guard speeds (150 mm/s free, 10 mm/s in the last 5 mm) are PRELIMINARY choices, not measured.
- No sim KPI numbers yet. fa-mimic disk was full; containerd/docker moved to `/ephemeral` (bind mounts in fstab).
- Open: env lacks `robot_touch`, `inject_foreign`, `block_holes`, `force_release`, `inject_intruder` (those scenarios
  are skipped, never counted as passes); planner vs env origin differs 4 mm for M1/M2/M3_04; `*_04` tilt gate mixes
  in yaw; `robot/cell` lacks pyzmq/msgpack (T05 adds, or `uv run --with`).

## 2026-10-28 23:25 · NEW-DATA-ONLY training (for CODEX) · CLAUDE · user decision, do this before the H100 run

- User: train ONLY on the new test-bed sets N R O P Q (parts stand as in their product, any yaw; 0 wrong rest labels).
  Old sets A-M are NOT used for training (their `_REST.json` tags stay for analysis only).
- Data: fa-r4 `~/fanuc-assembly-data/synthetic/{N,R,O,P,Q}/chunk_NNNN/` (labels/, rgb/, rgb_cosmos/ for ~50 % of
  scenes, depth/, seg/). Each loose object has `rest`: `installed` | `fallen`. Expected: N 40, R 20, O 48, P 24, Q 8 chunks.
- Needed in T13/T14 code (Codex-owned; Claude does not edit it):
  1. Detector RT-DETR on N + R + Q (table + side cameras): 30 exact part classes + foreign, plus a per-box `fallen`
     flag (head or extra class, Codex's choice); report per look-alike group.
  2. Wrist classifier (ResNet-18) on O (one wrist height, look-alike always beside the target): 30 exact classes.
  3. Set-P alignment model: read `build_offset` (shift mm / yaw deg / tilt / height) from P wrist images, for the
     seated check within `sdg/data/insertion_tolerance.json` and the yaw-and-retry insertion rule (DECISIONS).
  4. PatchCore per object on good O close-ups (installed orientation); FoundationPose pretrained validation on N/P
     depth + STL.
  5. Cosmos copies are train-only and inherit their scene's split; drop the "10-20 % Cosmos" gate for the new sets
     (they are ~50 % by user decision).
- Order on the 8xH100 (user): RT-DETR -> classifier -> set-P -> PatchCore (after its dependency audit) ->
  FoundationPose validation -> copy artifacts -> close. Claude starts the box when PIPELINE_DONE is written and this
  change is pushed.

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
## 2026-10-29 · T09 · CODEX · guarded PackML assembly controller

- Changed: products are parsed and bound from the first `obs["trial"]["instruction"]` of every reset; the constructor
  accepts no fixed product tuple. Inventory comes only from the perception port. Loose/table or reserved fallen,
  defective, look-alike, and unrequested parts now execute inspect → FoundationPose grasp → reject → independently
  verified transfer before a verified table replacement or spare is reserved. Unknown foreign objects Hold/Andon when
  no catalog identity and sourced grasp exist; installed/staged scan defects Hold for calibrated dismantle recovery.
- Motion/recovery: every normal or recovery `pre_pick` happens only after the sourced straight-down inspection stop;
  recovery no longer calls `grasp_for`. Safe-Z rise/XY/descent, nearest-equivalent W/P/R interpolation bounded to the
  official ±360° TCP contract, shortest yaw, configured per-step Cartesian speed checks, table-before-spare selection,
  operator acknowledge/rescan resume, and installed/held occlusion ownership are covered. Inspection actions are
  authoritative: `ask_operator` Holds pre/post, and physical reject actions are not inferred from raw scores.
- Runtime boundary: `AssemblyController` requires caller-sourced `InspectionStopConfig` and `StartupHoldConfig`; the
  latter is validated against the official action contract and prevents malformed first observations from escaping
  `act()`. `reset()` clears prompt ownership, `requested_products` exposes the parsed tuple, and
  `acknowledge_resume()` requires a fresh scan before leaving Held. The latest T04/T05 review fixes are integrated:
  every emitted joint/TCP action uses the live official guard, and the catalog pins T29's Team V printed-finger
  source at `20c6f9b` / SHA-256 `99611a032fb74a02740df405f1ef2d2d73f54f17cfc5676899cf9244e6d629c1`.
- Pose conversion: the provider composes measured UF0 base pose × CAD STL installation transform × inverse T29
  mesh-to-installed-local transform. It now carries two distinct internal frames: the canonical object origin used for
  camera verification and the T29 fingertip TCP used for motion. Reject verification uses calibrated zone occupancy,
  not equality to the pre-release TCP; staging/rebuild carry an explicit object verification target. A recorded
  `FoundationPoseFrameBoundary("installed_local", source)` is mandatory, so raw STL-frame output cannot silently enter
  the yaw-only path. Identity/source/finite checks fail closed, and fallen parts Hold when the shared contract cannot
  supply a sourced full-6D grasp.
- Run: `uv run ruff check .` (passed); `uv run pytest -q` (513 passed); `git diff --check` (clean);
  `git diff --exit-code origin/main -- robot/cell/cell_client.py` (clean). The integrated T04 catalog-schema change
  still requires Claude's latest-head contract review on PR #5; no additional T09 contract change was made.
  `cd robot/cell && uv run --locked run.py --mock` → 3 rollouts, 602 actions, 0 rejected and
  `✓ Ready for the real robot`; adding `--checkpoint t09-mock --trials 1 --trial-s 20` → 1 rollout,
  200 actions, 0 rejected, with the explicit mock-only/not-real-robot-ready verdict.
- Next: Claude re-reviews PR #9 at the integrated head. The cell-runtime branch must rebase here and remain disabled
  until T28/T30 live perception, reviewed recovery targets, and a sourced ER-4iA robot-link collision provider exist;
  do not select `t09-mock` on the real cell.

## 2026-10-29 · T05 review fixes · CODEX

- Added a fail-closed joint guard for the official J1-J6 limits and gripper range; TCP actions now additionally require
  the measured fingers-down set-O envelope (`w` within 3 degrees of +/-180, `p` within 3 degrees, free bounded yaw).
- `CellAdapter` resolves `DT` on every action call, so the guard validates the same live value the client transmits.
  `Policy` supplies that live value without changing the default joint-hold runtime.
- Regression coverage includes malformed, NaN/Inf, epsilon-outside and NumPy float32/float64 chunks plus end-to-end
  policy rejection. `robot/cell/cell_client.py` remains unchanged.
- Verified: `uv run ruff check .` clean; `uv run pytest -q` 102 passed; official `uv run run.py --mock` completed
  3 rollouts / 600 actions / 0 rejected and printed `Ready for the real robot`.
- Next: Claude reviews PR #4; merge this commit into T09 before enabling the explicit cell runtime.

## 2026-10-27 · T05 · CODEX · adapter layer and UF0 safety guard

- Changed: `robot/adapter.py`, `robot/__init__.py`, `robot/cell/policy.py`, `tests/test_robot_adapter.py`, and
  `pyproject.toml`. Sim, mock, and cell now share passive `reset()` / `act(observation)` adapters around an injected
  controller; the vendored policy imports the repo adapter without `PYTHONPATH` and defaults to observed joint hold.
- Run: `uv run pytest tests/test_robot_adapter.py -q` (61 passed); `uv run pytest` (73 passed);
  `uv run ruff check .` (passed); `git diff --check` (passed);
  `git diff --exit-code origin/main -- robot/cell/cell_client.py` (passed). Development smoke:
  `cd robot/cell && uv run run.py --mock --trials 1 --trial-s 0.5` → 1 rollout, 6 actions, 0 rejected. Final documented
  workflow: `cd robot/cell && uv sync && uv run run.py --mock` → exit 0, 3 rollouts, 601 actions, 0 rejected,
  `✓ Ready for the real robot`.
- Safety: TCP outputs are absolute UTOOL1 fingertip targets in UF0. The guard accepts one row or 1–100-row chunks,
  preserves nesting without mutating input, and requires finite numeric values, dt 0.1–1.0 s, x 150–650 mm,
  y −450–450 mm, z −334–300 mm, W/P/R within ±360°, and gripper 0–1, all inclusive. Violations raise contextual
  typed errors; values are never clamped. Sim observations are projected to official top-level fields, with copied
  nested mappings limited to public trial keys (`trial_id`, `index`, `layout`, `status`, `instruction`, `scored`),
  RGB cameras (`table`, `side`, `wrist`), and supported depth cameras (`table`, `wrist`); simulator ground truth,
  segmentation, and private depth metadata cannot reach controller callbacks. Joint mode holds the exact observed
  position, including when `tcp_pose` is `None`; no TCP home target is invented.
- Next: independent review, then open the T05 PR for Claude review and user merge. T09 can consume these adapters
  after T04 and T05 merge; no contract or `robot/cell/cell_client.py` changes are required.
## 2026-10-29 · T29 · CLAUDE · grasp and insert frames from the printed CAD

- Changed: `cad/grasp_frames.py` builds every product with `cad/products` (the printed CAD; `cad/stl` is
  byte-identical to the printed STLs) and writes `cad/grasp_frames.json` for all 30 objects:
  - `installed_origin_mm` (assembly frame, base origin = 0);
  - `T_local_from_stl` (identity except the rear guards' print rotation);
  - each grip pad's closing axis, width, span, contact height, the fingertip TCP point in the part's local frame;
  - `roll_offset_deg` (TCP r = measured yaw + offset, from the twin IK's jaw spread axis).
- Fingers: the TCP of every part comes from the Team V printed fingers' validated grasp (`cad/fingers/`, copied from
  the MHZL2_16D2_FANUC_FINAL_FINGERS project: 24/24 PASS in its CAD check, PROVISIONAL until printed and tried):
  the contact centre moved down 1.85 mm to the fingertip. The script refuses a contact that is not on the CAD pad.
- Measured: every product part has one 7 mm grip pad, which fits the Team V fingers' rubber gap (4.5 mm closed,
  16.5 mm open). Installed transforms agree with `sdg/data/object_geometry.json` (mesh registration) within 0.005 mm.
- How to run: `uv run python cad/grasp_frames.py`; `uv run pytest tests/test_grasp_frames.py` (regenerates and
  compares).
- Next:
  - Codex T09/T05 consumes this file for grasp and insert targets, instead of hand-copied numbers.
  - FoundationPose poses are of the STL frame: multiply by `inverse(T_local_from_stl)` first.
  - Physical gripper test on the cell is NOT PERFORMED (CAD summary).

## 2026-10-27 · T02 · CLAUDE · assembly instruction sheets

- Changed: `cad/products/` (product CAD source), `cad/assembly_sheets.py`, `docs/ASSEMBLY_INSTRUCTIONS/` (6 PNG + 1 PDF).
- Run: `uv run python cad/assembly_sheets.py` (CadQuery + matplotlib are the locked `cad` dependency group, installed by
  `uv sync`). Tests: `tests/test_assembly_sheets.py` (steps, look-alikes, 90° gripper rule, the 7 files).
- Content per product: base, steps 01→04 (new part highlighted), exploded view, table with gripper roll (connector
  housings: turn gripper 90°) and look-alikes from `contracts/taxonomy.json`. Dismantle = 04→01.
- Next: T03 CAD risk ranking.

## 2026-10-30 · T04 T29 turn/TCP atomic pin · CODEX

- Repinned the immutable T29 grasp snapshot to `fac05905338b82a90744e934268f4a30bf412fc7` with SHA-256
  `dbd00b5082042e8a326b380d237890449bfb2206243c1753b7ebe2d940435977`.
- Rejected the superseded single-vector pad point and require the turn-indexed flange-axis TCP schema. Canonical keys,
  strict numeric finite values, and exact source fields now fail closed on malformed data.
- Added `PreferredGrasp(turn_deg, tcp_local_mm)` to every typed `RecipeStep`. This binds the source-tested wrist turn to
  its matching TCP and intentionally withholds the algebraic 180-degree alternative from CELL authorization.
- Regression coverage spans all 24 movable parts plus legacy vectors, missing/unknown turns, NaN, strings, and booleans.
  Full gate: `uv run pytest -q` = 76 passed; `uv run ruff check .` and `git diff --check` passed. Independent Codex
  re-review found no remaining code issue.
- Merge remains blocked on PR #15's immutable CAD/finger/STL and physical-validation provenance. T09 must consume
  `RecipeStep.preferred_grasp` atomically; it must not reopen the raw T29 JSON or authorize the alternate turn.

## 2026-10-29 · T04 pinned-source and rotation hardening · CODEX

- Replaced the duplicated transform oracle with byte-identical, SHA-256-pinned snapshots of
  `sdg/data/object_geometry.json` at `5a24e10` and T29 `cad/grasp_frames.json` at `20c6f9b`.
- Repinned T29 after it adopted the Team V printed fingers: the source digest is
  `99611a032fb74a02740df405f1ef2d2d73f54f17cfc5676899cf9244e6d629c1`, and the snapshot records
  the validated 4.5–16.5 mm rubber-pad gap plus the updated contact-derived TCP offsets for all 24 parts.
- Sourced all 24 gripper rolls from the single T29 grasp for each part: commanded TCP r is measured part yaw plus
  `roll_offset_deg`; X-closing grips use 90 degrees and Y-closing grips use 0 degrees. Multiple grips now fail closed.
- Installed rotations must be finite 3x3 proper rotations: orthonormal rows and determinant +1 within `1e-6`, the
  source generator's six-decimal serialization precision. Schema-valid but unsafe matrices fail semantic validation.
- The catalog schema now rejects unknown top-level, product, and step keys. Source hashes and catalog/source values
  are rechecked at load time, and source dates are explicitly labeled as Git commit timestamps. All six provenance
  fields for both sources are compared with immutable trusted records; typed catalog objects return those records.
- Regression evidence: NaN, Inf, singular, non-orthogonal, reflection, hash mismatch, zero/multi-grip ambiguity, roll
  mismatch, all 12 source-provenance mutations, Team V finger provenance, and unknown-key cases fail closed.
  `uv run ruff check .` passed; full `uv run pytest -q` passed 69 tests.
- Integration note: T09 must use `RecipeStep.gripper_roll_deg` only as a T29 offset added to measured part yaw and
  apply `installed_rotation_matrix` with `insert_offset_mm`; it must not preserve the superseded provisional values.

## 2026-10-29 · T04 CAD transform review fixes · CODEX

- Replaced bounding-box proxy offsets with the CAD `build()` installed transforms for all 24 parts, including the
  +90-degree-X guard rotations and `[0, 21.5, 21]` mm guard translations.
- Added `installed_rotation_matrix`, immutable source provenance, frame semantics and bounded provisional gripper
  rolls to the catalog contract. Unknown step keys now fail closed.
- The committed reference table is a verified exact extraction of `sdg/data/object_geometry.json` generated by
  `sdg/tools/object_geometry.py` at commit `5a24e10d5ac7c9a6b15e9779385138678244819d`; 24/24 transforms matched.
- Verified: `uv run ruff check .` clean and `uv run pytest -q` 44 passed. Contract changes require Claude review on
  PR #5 before merge.
- Next: merge PR #5's commit into T09 and update its pose consumer to apply translation and rotation together.

## 2026-10-27 · T04 · CODEX · product catalog and recipe graph

- Changed: added `assembly/catalog.yaml`, typed/validated loading in `assembly/catalog.py`, public exports in
  `assembly/__init__.py`, and behavioral coverage in `tests/test_catalog.py`; configured pytest to import repository
  packages consistently through the direct `uv run pytest` entry point.
- Behavior: six recipes validate against `contracts/catalog.schema.json` and the fixed taxonomy; semantic checks enforce
  products 1–6, ordered steps 1–4, ownership, exact 24-part coverage, base types, and finite three-axis offsets. Shared
  fixtures resolve by type A/B/C; recipe graphs assemble base→01→02→03→04 and dismantle 04→01 without moving the base.
  YAML duplicate keys fail closed, every registry schema is meta-schema validated before registration, and
  catalog/schema/taxonomy resource failures stay within typed catalog errors.
- Run: `uv run pytest tests/test_catalog.py -q` (28 passed); `uv run pytest` (40 passed);
  `uv run ruff check .` (all checks passed); `git diff --check` (clean).
- Provenance/frame: product ownership and shared base types come from `contracts/taxonomy.json`; installed offsets are the
  supplied CAD values in the detected fixture's CAD-local frame. They require a measured fixture-pose transform and are
  never absolute UF0 TCP targets. No contract files changed.
- Next: independent review, push/open the T04 PR, then use the catalog as the recipe source for T09 after merge.

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
