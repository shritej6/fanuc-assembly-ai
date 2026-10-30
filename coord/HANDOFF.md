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

## 2026-10-30 · T04 T29 turn/TCP atomic pin · CODEX

- Repinned the immutable T29 grasp snapshot to `fac05905338b82a90744e934268f4a30bf412fc7` with SHA-256
  `dbd00b5082042e8a326b380d237890449bfb2206243c1753b7ebe2d940435977`.
- Rejected the superseded single-vector pad point and require the turn-indexed flange-axis TCP schema. Canonical keys,
  strict numeric finite values, and exact source fields now fail closed on malformed data.
- Added `PreferredGrasp(turn_deg, tcp_local_mm)` to every typed `RecipeStep`. This binds the source-tested wrist turn to
  its matching TCP and intentionally withholds the algebraic 180-degree alternative from CELL authorization.
- Regression coverage spans all 24 movable parts plus legacy vectors, missing/unknown turns, NaN, strings, and booleans.
  Full gate: `uv run pytest -q` = 76 passed; `uv run ruff check .` and `git diff --check` passed. Independent Codex
  re-review found no remaining code issue.
- Merge remains blocked on PR #15's immutable CAD/finger/STL and physical-validation provenance. T09 must consume
  `RecipeStep.preferred_grasp` atomically; it must not reopen the raw T29 JSON or authorize the alternate turn.

## 2026-10-29 · T04 pinned-source and rotation hardening · CODEX

- Replaced the duplicated transform oracle with byte-identical, SHA-256-pinned snapshots of
  `sdg/data/object_geometry.json` at `5a24e10` and T29 `cad/grasp_frames.json` at `20c6f9b`.
- Repinned T29 after it adopted the Team V printed fingers: the source digest is
  `99611a032fb74a02740df405f1ef2d2d73f54f17cfc5676899cf9244e6d629c1`, and the snapshot records
  the validated 4.5–16.5 mm rubber-pad gap plus the updated contact-derived TCP offsets for all 24 parts.
- Sourced all 24 gripper rolls from the single T29 grasp for each part: commanded TCP r is measured part yaw plus
  `roll_offset_deg`; X-closing grips use 90 degrees and Y-closing grips use 0 degrees. Multiple grips now fail closed.
- Installed rotations must be finite 3x3 proper rotations: orthonormal rows and determinant +1 within `1e-6`, the
  source generator's six-decimal serialization precision. Schema-valid but unsafe matrices fail semantic validation.
- The catalog schema now rejects unknown top-level, product, and step keys. Source hashes and catalog/source values
  are rechecked at load time, and source dates are explicitly labeled as Git commit timestamps. Insert offsets must
  contain exactly three axes at the contract boundary. All six provenance fields for both sources are compared with
  immutable trusted records; typed catalog objects return those records.
- Regression evidence: NaN, Inf, singular, non-orthogonal, reflection, hash mismatch, zero/multi-grip ambiguity, roll
  mismatch, all 12 source-provenance mutations, Team V finger provenance, and unknown-key cases fail closed.
  Contract/provenance and recipe-graph tests are split into focused modules (461/64 lines) to preserve the 500-line
  gate. `uv run ruff check .` passed; full `uv run pytest -q` passed 70 tests.
- Integration note: T09 must use `RecipeStep.gripper_roll_deg` only as a T29 offset added to measured part yaw and
  apply `installed_rotation_matrix` with `insert_offset_mm`; it must not preserve the superseded provisional values.

## 2026-10-29 · T04 CAD transform review fixes · CODEX

- Replaced bounding-box proxy offsets with the CAD `build()` installed transforms for all 24 parts, including the
  +90-degree-X guard rotations and `[0, 21.5, 21]` mm guard translations.
- Added `installed_rotation_matrix`, immutable source provenance, frame semantics and bounded provisional gripper
  rolls to the catalog contract. Unknown step keys now fail closed.
- The committed reference table is a verified exact extraction of `sdg/data/object_geometry.json` generated by
  `sdg/tools/object_geometry.py` at commit `5a24e10d5ac7c9a6b15e9779385138678244819d`; 24/24 transforms matched.
- Verified: `uv run ruff check .` clean and `uv run pytest -q` 44 passed. Contract changes require Claude review on
  PR #5 before merge.
- Next: merge PR #5's commit into T09 and update its pose consumer to apply translation and rotation together.

## 2026-10-27 · T04 · CODEX · product catalog and recipe graph

- Changed: added `assembly/catalog.yaml`, typed/validated loading in `assembly/catalog.py`, public exports in
  `assembly/__init__.py`, and behavioral coverage in `tests/test_catalog.py`; configured pytest to import repository
  packages consistently through the direct `uv run pytest` entry point.
- Behavior: six recipes validate against `contracts/catalog.schema.json` and the fixed taxonomy; semantic checks enforce
  products 1–6, ordered steps 1–4, ownership, exact 24-part coverage, base types, and finite three-axis offsets. Shared
  fixtures resolve by type A/B/C; recipe graphs assemble base→01→02→03→04 and dismantle 04→01 without moving the base.
  YAML duplicate keys fail closed, every registry schema is meta-schema validated before registration, and
  catalog/schema/taxonomy resource failures stay within typed catalog errors.
- Run: `uv run pytest tests/test_catalog.py -q` (28 passed); `uv run pytest` (40 passed);
  `uv run ruff check .` (all checks passed); `git diff --check` (clean).
- Provenance/frame: product ownership and shared base types come from `contracts/taxonomy.json`; installed offsets are the
  supplied CAD values in the detected fixture's CAD-local frame. They require a measured fixture-pose transform and are
  never absolute UF0 TCP targets. No contract files changed.
- Next: independent review, push/open the T04 PR, then use the catalog as the recipe source for T09 after merge.

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
