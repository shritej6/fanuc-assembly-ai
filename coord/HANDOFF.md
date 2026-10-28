# Handoff log

Newest first. One entry per finished task.

## 2026-10-27 · T03 · CLAUDE · CAD risk ranking v0 (PRELIMINARY)

- Changed: `cad/risk_ranking.py`, `cad/risk_ranking_v0.json`, `cad/RISK_RANKING_v0.md`.
- Run: `~/cad-workspace/.venv/bin/python cad/risk_ranking.py` (needs trimesh + rtree + shapely; about 5 s).
- Result: top risks are M2_03, M2_02, M1_03, M6_02, M3_03; locks have the weakest layer (15.3 mm²). The 13-part
  print list for U04 is in the .md. Codex's `assembly/risk_table.json` (T09) should use these scores as the design
  prior and keep the PRELIMINARY label.
- Next: T06 defect variants (sim) and T07 USD scene, once the user marks them READY.

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
