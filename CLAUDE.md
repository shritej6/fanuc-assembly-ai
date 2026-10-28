# Claude Code rules for this repo

Read first: `docs/PLAN.md`, `coord/TASKS.md`, `coord/DECISIONS.md`, `contracts/`.

- Domain: CAD, Isaac Sim scenes, synthetic data (Replicator), Cosmos, Isaac Lab Mimic demos, docs/video.
- Start a task only when its status in `coord/TASKS.md` is READY and its owner is CLAUDE.
- Never edit files owned by a CODEX task; never edit `robot/cell/cell_client.py` (official client).
- One branch per task (`feature/T<id>-<slug>`), PR into `main`, reviewed by Codex, merged by the user.
- Contracts change only through a PR reviewed by both agents.
- After every task add an entry to `coord/HANDOFF.md` (what changed, how to run, what's next).
- Follow the Team V tech stack exactly (PLAN section 4). Do not substitute tools.
- Never weaken a check to make it pass. Never report a number that was not measured. Failure
  probabilities stay labelled PRELIMINARY. Real-cell data is for evaluation only, never training.

Claude agents: `.claude/agents/` (cad-engineer, sim-builder, sdg-engineer, cosmos-engineer, demo-generator,
doc-writer). Codex roles: `agents/codex/`. Review every Codex PR with `gh pr review`.
