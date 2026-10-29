# Handoff log

Newest first. One entry per finished task.

## 2026-10-29 · T29 · CLAUDE · grasp and insert frames from the printed CAD

- Changed: `cad/grasp_frames.py` builds every product with `cad/products` (the printed CAD; `cad/stl` is
  byte-identical to the printed STLs) and writes `cad/grasp_frames.json` for all 30 objects:
  - `installed_origin_mm` (assembly frame, base origin = 0);
  - `T_local_from_stl` (identity except the rear guards' print rotation);
  - each grip pad's closing axis, width, span, contact height, the fingertip TCP point in the part's local frame;
  - `roll_offset_deg` (TCP r = measured yaw + offset, from the twin IK's jaw spread axis).
- Measured: every product part has one 7 mm grip pad, which fits the SMC MHZL2-16D2 fingertip gap (4 +- 1 closed,
  15 +- 2 open). Installed transforms agree with `sdg/data/object_geometry.json` (mesh registration) within 0.005 mm.
- How to run: `uv run python cad/grasp_frames.py`; `uv run pytest tests/test_grasp_frames.py` (regenerates and
  compares).
- Next:
  - Codex T09/T05 consumes this file for grasp and insert targets, instead of hand-copied numbers.
  - FoundationPose poses are of the STL frame: multiply by `inverse(T_local_from_stl)` first.
  - Physical gripper test on the cell is NOT PERFORMED (CAD summary).

## 2026-10-27 · T02 · CLAUDE · assembly instruction sheets

- Changed: `cad/products/` (product CAD source), `cad/assembly_sheets.py`, `docs/ASSEMBLY_INSTRUCTIONS/` (6 PNG + 1 PDF).
- Run: `uv run python cad/assembly_sheets.py` (CadQuery + matplotlib are the locked `cad` dependency group, installed by
  `uv sync`). Tests: `tests/test_assembly_sheets.py` (steps, look-alikes, 90° gripper rule, the 7 files).
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
