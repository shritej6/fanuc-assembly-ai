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

## 2026-10-29 · T05 review fixes · CODEX

- Added a fail-closed joint guard for the official J1-J6 limits and gripper range; TCP actions now additionally require
  the measured fingers-down set-O envelope (`w` within 3 degrees of +/-180, `p` within 3 degrees, free bounded yaw).
- `CellAdapter` resolves `DT` on every action call, so the guard validates the same live value the client transmits.
  `Policy` supplies that live value without changing the default joint-hold runtime.
- Regression coverage includes malformed, NaN/Inf, epsilon-outside and NumPy float32/float64 chunks plus end-to-end
  policy rejection. `robot/cell/cell_client.py` remains unchanged.
- Verified: `uv run ruff check .` clean; `uv run pytest -q` 102 passed; official `uv run run.py --mock` completed
  3 rollouts / 600 actions / 0 rejected and printed `Ready for the real robot`.
- Next: Claude reviews PR #4; merge this commit into T09 before enabling the explicit cell runtime.

## 2026-10-27 · T05 · CODEX · adapter layer and UF0 safety guard

- Changed: `robot/adapter.py`, `robot/__init__.py`, `robot/cell/policy.py`, `tests/test_robot_adapter.py`, and
  `pyproject.toml`. Sim, mock, and cell now share passive `reset()` / `act(observation)` adapters around an injected
  controller; the vendored policy imports the repo adapter without `PYTHONPATH` and defaults to observed joint hold.
- Run: `uv run pytest tests/test_robot_adapter.py -q` (61 passed); `uv run pytest` (73 passed);
  `uv run ruff check .` (passed); `git diff --check` (passed);
  `git diff --exit-code origin/main -- robot/cell/cell_client.py` (passed). Development smoke:
  `cd robot/cell && uv run run.py --mock --trials 1 --trial-s 0.5` → 1 rollout, 6 actions, 0 rejected. Final documented
  workflow: `cd robot/cell && uv sync && uv run run.py --mock` → exit 0, 3 rollouts, 601 actions, 0 rejected,
  `✓ Ready for the real robot`.
- Safety: TCP outputs are absolute UTOOL1 fingertip targets in UF0. The guard accepts one row or 1–100-row chunks,
  preserves nesting without mutating input, and requires finite numeric values, dt 0.1–1.0 s, x 150–650 mm,
  y −450–450 mm, z −334–300 mm, W/P/R within ±360°, and gripper 0–1, all inclusive. Violations raise contextual
  typed errors; values are never clamped. Sim observations are projected to official top-level fields, with copied
  nested mappings limited to public trial keys (`trial_id`, `index`, `layout`, `status`, `instruction`, `scored`),
  RGB cameras (`table`, `side`, `wrist`), and supported depth cameras (`table`, `wrist`); simulator ground truth,
  segmentation, and private depth metadata cannot reach controller callbacks. Joint mode holds the exact observed
  position, including when `tcp_pose` is `None`; no TCP home target is invented.
- Next: independent review, then open the T05 PR for Claude review and user merge. T09 can consume these adapters
  after T04 and T05 merge; no contract or `robot/cell/cell_client.py` changes are required.

## 2026-10-29 · T29 · CLAUDE · grasp and insert frames from the printed CAD

- Changed: `cad/grasp_frames.py` builds every product with `cad/products` (the printed CAD; `cad/stl` is
  byte-identical to the printed STLs) and writes `cad/grasp_frames.json` for all 30 objects:
  - `installed_origin_mm` (assembly frame, base origin = 0);
  - `T_local_from_stl` (identity except the rear guards' print rotation);
  - each grip pad's closing axis, width, span, contact height, the fingertip TCP point in the part's local frame;
  - `roll_offset_deg` (TCP r = measured yaw + offset, from the twin IK's jaw spread axis).
- Fingers: the TCP of every part comes from the Team V printed fingers' validated grasp (`cad/fingers/`, copied from
  the MHZL2_16D2_FANUC_FINAL_FINGERS project: 24/24 PASS in its CAD check, PROVISIONAL until printed and tried):
  the contact centre moved down 1.85 mm to the fingertip. The script refuses a contact that is not on the CAD pad.
- Measured: every product part has one 7 mm grip pad, which fits the Team V fingers' rubber gap (4.5 mm closed,
  16.5 mm open). Installed transforms agree with `sdg/data/object_geometry.json` (mesh registration) within 0.005 mm.
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
