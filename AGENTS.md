# Codex rules for this repo

Read first: `docs/PLAN.md`, `coord/TASKS.md`, `coord/DECISIONS.md`, `contracts/`.

- Domain: model training/evaluation (detector, look-alike classifier, PatchCore, FoundationPose), recipe state
  machine (PackML), inventory, recovery, adapters and `policy.py`, operator agent, OPC UA, GR00T/SmolVLA/ACT, tests.
- Start a task only when its status in `coord/TASKS.md` is READY and its owner is CODEX.
- Never edit files owned by a CLAUDE task; never edit `robot/cell/cell_client.py` (official client).
- One branch per task (`feature/T<id>-<slug>`), PR into `main`, reviewed by Claude, merged by the user.
- Contracts change only through a PR reviewed by both agents.
- After every task add an entry to `coord/HANDOFF.md` (what changed, how to run, what's next).
- Follow the Team V tech stack exactly (PLAN section 4). Do not substitute tools.
- Never weaken a check to make it pass. Never use simulator ground truth to command grasps. Never report a
  number that was not measured. Real-cell data is for evaluation only, never training.
