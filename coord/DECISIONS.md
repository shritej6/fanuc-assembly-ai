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
