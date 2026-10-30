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
