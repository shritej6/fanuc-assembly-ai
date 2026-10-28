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

## Codex roles

Role briefs live in `agents/codex/`. When you take a task, act as the role that owns it:
`perception-trainer`, `assembly-planner`, `robot-adapter`, `operator-agent`, `policy-learner`, `test-engineer`,
and `reviewer` (reviews every Claude PR with `gh pr review`). Claude's agents are in `.claude/agents/`.

## Working loop (every session)

1. `git fetch --all && git status`; read the newest `coord/HANDOFF.md` entries and `coord/TASKS.md`.
2. Review open Claude PRs (`gh pr list`), comment or approve; never push to a Claude branch.
3. Work the READY CODEX task(s) you were given; finish with PR + HANDOFF entry.
4. Stop and report when nothing is READY. Do not start BACKLOG tasks on your own.
