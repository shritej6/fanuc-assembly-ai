# Handoff log

Newest first. One entry per finished task.

## 2026-10-27 · T02 · CLAUDE · assembly instruction sheets

- Changed: `cad/products/` (product CAD source), `cad/assembly_sheets.py`, `docs/ASSEMBLY_INSTRUCTIONS/` (6 PNG + 1 PDF).
- Run: `~/cad-workspace/.venv/bin/python cad/assembly_sheets.py`.
- Content per product: base, steps 01→04 (new part highlighted), exploded view, table with gripper roll (connector
  housings: turn gripper 90°) and look-alikes from `contracts/taxonomy.json`. Dismantle = 04→01.
- Next: T03 CAD risk ranking.

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
