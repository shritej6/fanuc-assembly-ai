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
| 2026-10-29 | Rendered sets A–D keep their sampled product combinations (measured per-combination spread A 372–427, B 1169–1252, C 362–434, D 129–188) instead of the equal frequency in PLAN 6.2; re-rendering 43,000 scenes would change the specs behind data already trained on. Sets rendered from now on stratify the combinations | Claude (user delegated the call: "you have to know it") |
| 2026-10-29 | The CSI-slot client and the T27 models run on a RunPod A100 cloud pod (FoundationPose needs CUDA); the user watches from the laptop | user |
| 2026-10-29 | Controller rules for the slot: no hardcoded scenarios (products from the instruction, parts from perception), a fallen part is picked as it lies and rejected (operator hold if no approved grasp), bases of one shape are interchangeable (F01=F02, F04=F05=F06 are the same mesh, `contracts/taxonomy.json`), so a product goes on any free base of its type | user |
