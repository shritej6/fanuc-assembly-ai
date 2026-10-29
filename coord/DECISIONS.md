# Decisions

All decisions locked on 2026-10-27 are listed in `docs/PLAN.md` sections 2, 4 and 10a. New decisions are appended
here with a date and who made them (the user decides; agents propose).

| Date | Decision | By |
|---|---|---|
| 2026-10-27 | Follow the Team V tech stack exactly | user |
| 2026-10-27 | Bases never move; assemble in place; one product at a time | user |
| 2026-10-27 | Industry layer: PackML, birth certificate, two independent checks, OEE/Andon, OPC UA, Kanban + Pareto, FAT report | user |
| 2026-10-27 | All poses in contracts use the controller frame UF0 in mm/deg (same as `obs["tcp_pose"]` and `"tcp"` actions) | Claude (proposal, part of T01) |
| 2026-10-27 | Synthetic data root: `/data/ai_club/Innovation_lab_team_v` on Rosie (pending group write access); interim `~/fanuc-assembly-data` | user |
| 2026-10-28 | Detector and wrist classifier label all 30 objects separately (no merging of identical shapes such as F01/F02, F04/F05/F06); confusion between identical shapes is reported per group, not hidden | user |
| 2026-10-28 | Cosmos Transfer is required, not optional | user |
| 2026-10-28 | GR00T and Cosmos run on Rosie's H100 (`dgxh100`) first; Brev only if the H100 cannot start in time | user |
| 2026-10-28 | Detector is RT-DETR (not Faster R-CNN) | user |
| 2026-10-28 | Build check uses Cosmos Reason 2 8B instead of 2B | user |
| 2026-10-28 | Cosmos Transfer restyles 10–20 % of each set; shape drift is checked before any restyled image is used | user |
| 2026-10-28 | Brev cap for training, Cosmos and GR00T: $50 (separate from the ~$21 spent on rendering) | user |
| 2026-10-28 | Detector, classifier and PatchCore train on Rosie's V100 (`dgx`, idle) since the H100 queue estimates a start after the deadline | Claude (proposal, user said "do everything") |
| 2026-10-28 | Use imitation learning where it helps: GR00T N1.7 (LeRobot), SmolVLA, ACT trained on Isaac Lab Mimic demos, kept only if it beats the scripted baseline | user |
| 2026-10-28 | Compute for Cosmos and training moves to RunPod (H100 SXM for the detector, A100 SXM for the classifier, PatchCore and Cosmos); Rosie's H100s stay queued (Priority, est. Nov 9). Cap $100 for this phase | user |
| 2026-10-28 | Cosmos covers 10 % of A, F, G, B, H, K (Codex's production gate needs 10-20 %); C, I, D, J, L, M are not restyled. Cosmos part pixels are kept only for objects that pass a per-object edge check, the rest are pasted back from the render | user (approved plan) |
| 2026-10-28 | On the H100 the detector trains at 1280 px input with bf16 instead of the T4 setting (640 px, fp16, batch 2), so small parts keep their detail | user (approved plan) |
| 2026-10-28 | Production training runs DDP on one RunPod pod with 4× H100 SXM (same GPU-hours, ~3.5× shorter wall time; Codex's 4-GPU NCCL path is validated). The single H100 (fa-train-det) does smoke + throughput benchmark first; the 4× pod is created ~19:00 CT once the benchmark projects the run, and the classifier shares the H100s instead of an A100 pod. Same $100 cap | user (approved plan) |
| 2026-10-28 | Parts stand on the test-bed in their installed orientation at any yaw; the arm cannot turn a part over, so a fallen part is rejected and replaced from the spare store (replaces "flipped → rework") | user |
| 2026-10-28 | ~70,000 more images in installed orientation (sets N R O P Q: test-bed detector nominal + hard, identity close-ups at one wrist height, insertion alignment, fallen/spares), rendered on GB10 + Brev | user |
| 2026-10-28 | Brev ($400 credit) carries the new rendering, Cosmos and training; RunPod finishes the running Cosmos run and is the fallback. Brev guard proposed at $350 | user ($350 guard: Claude proposal) |
| 2026-10-28 | Brev split: $270 for render + Cosmos (guard deletes Cosmos boxes at $265, never the render box), $120 kept for one 8× H100 training run. Cosmos restyles 50 % of the scenes of the new sets only (N R O P Q); old-set Cosmos stopped at 4,990 / 5,093 | user |
| 2026-10-28 | Detector (RT-DETR, table/side) reports the exact part (all 30) + location; look-alikes must be confirmed by the wrist close-up before a pick (two independent cues) | user |
| 2026-10-28 | Look-alike still unresolved after a second wrist look from another angle → set the part aside and take a known spare (LOOKALIKE_UNRESOLVED), never guess | user |
| 2026-10-28 | Seated check passes only within the CAD insertion tolerance (`sdg/data/insertion_tolerance.json`: yaw ±1.4–2.4°, caps ±6.8°, shift 0.4–0.5 mm, no lift) | user |
| 2026-10-28 | Accuracy first: bigger / slower models and extra checks are acceptable | user |
| 2026-10-28 | PatchCore-flagged part → reject zone + spare | user |
| 2026-10-28 | Identity checked before the pick AND re-checked on the stack after placing (fit ≠ identity: M1_01 / M2_01 seat fully in F03); mismatch → WRONG_PART, remove and replace | user |
| 2026-10-28 | Detector has a `fallen` flag per part (sets N / Q teach it); planner rejects fallen parts and takes a spare | user |
| 2026-10-28 | H100 order: 1 RT-DETR production, 2 wrist classifier, 3 set-P alignment model if ready, 4 PatchCore (single-GPU per object, object jobs in parallel over the 8 GPUs ≈ 3 waves; its dependency-security audit must pass first), 5 FoundationPose validation (pretrained, one H100, own pinned environment), 6 copy artifacts and close the box | user |
| 2026-10-28 | The spare area holds a few spares (up to the 6 slots) of any of the 24 parts, never a base. New sets use it from the change on; set Q (the spare-store set) is re-rendered whole with it; older N / R / P chunks keep the old six-part list | user |
| 2026-10-28 | "Standing" = the part's assembled orientation (same up face, roll, pitch); only xy and yaw vary on the bed. The guides M4/M5/M6_03 are flat U frames with the back block up in the real product (user checked the assembled render against the printed parts), so they lie flat on the bed too; a flat guide is NOT a fallen guide. No `pick_ready_pose` separate from the installed pose; flat-vs-standing changes appear only as labelled `fallen` | user |
| 2026-10-28 | Old sets A-M are used for training after auto-labelling: every loose part / trap / spare is tagged `installed` (product up axis, yaw free) or `fallen` from its exact label pose (`sdg/tools/rest_labels.py`, sidecar `_REST.json` per chunk, labels untouched). Old sets are 71 % fallen, so the new sets N/R/O/P/Q are weighted higher | user |
| 2026-10-28 | Insertion: before placing, yaw the part so its bottom holes line up over the pegs (CAD hole/peg positions + detected pose); place straight down; if the seated check fails (outside CAD tolerance), lift, re-yaw in small steps and retry until it seats, then give up to reject + spare after a bounded number of tries. Every slot, peg and spare position comes from the CAD/cell layout, never guessed. Look-alikes differ by mm: identity always from detector + wrist close-up, never from shape alone; grasp width from CAD per part (parts are small and can slip or fall) | user |
| 2026-10-28 | Training uses ONLY the new test-bed sets N R O P Q; old sets A-M are not trained on (supersedes the auto-label-then-use row) | user |
