# Codex role: assembly-planner

**Owns:** Product catalog and recipe graph, PackML recipe state machine, live inventory, recovery (reinsert x2, then disassemble and rebuild), Andon, two-independent-check gates, risk table (design prior from cad/risk_ranking_v0.json, PRELIMINARY).
**Tasks:** T04, T09, T15 (data), T20 (logic)
**Code area:** assembly/

Rules (same as AGENTS.md):
1. Read `docs/PLAN.md`, `coord/TASKS.md`, `coord/DECISIONS.md`, `contracts/`, newest `coord/HANDOFF.md` entries first.
2. Start only a task owned by CODEX with status READY (set by the user). Set IN PROGRESS, branch
   `feature/T<id>-<slug>`, tests with the code, PR, status REVIEW, `coord/HANDOFF.md` entry.
3. Use only the formats in `contracts/`; contract changes need a PR reviewed by both agents.
4. Follow the Team V tech stack exactly. Never weaken a check, never use sim ground truth to command grasps, never
   report an unmeasured number, never train on real-cell data.
