# Codex role: perception-trainer

**Owns:** Detector (Faster R-CNN or RT-DETR), wrist look-alike classifier, FoundationPose (STL + depth), anomalib PatchCore, depth missing-layer check; training on Rosie dgxh100 / Brev, evaluation on held-out synthetic and real eval images.
**Tasks:** T13, T14
**Code area:** perception/

Rules (same as AGENTS.md):
1. Read `docs/PLAN.md`, `coord/TASKS.md`, `coord/DECISIONS.md`, `contracts/`, newest `coord/HANDOFF.md` entries first.
2. Start only a task owned by CODEX with status READY (set by the user). Set IN PROGRESS, branch
   `feature/T<id>-<slug>`, tests with the code, PR, status REVIEW, `coord/HANDOFF.md` entry.
3. Use only the formats in `contracts/`; contract changes need a PR reviewed by both agents.
4. Follow the Team V tech stack exactly. Never weaken a check, never use sim ground truth to command grasps, never
   report an unmeasured number, never train on real-cell data.
