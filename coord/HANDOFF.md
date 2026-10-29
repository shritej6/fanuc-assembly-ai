# Handoff log

Newest first. One entry per finished task.

## 2026-10-29 · T16 · CODEX · live OEE and Andon dashboard

- Changed: added run/rollout-scoped append-only events, a locked and bounded controller log writer/recorder,
  evidence-backed OEE calculations, first/final yield, builds per assist, average cycle time, escapes, live
  PackML/Andon status, and a responsive read-only loopback dashboard. The recorder handles each actual T09 reset as a
  new rollout history, validates terminal records through T15 plus its semantic result meaning, records explicit
  evaluation escapes, and emits scoped liveness heartbeats without changing state or metrics.
- Integrity and safety: missing measurements remain unavailable rather than estimated; reporting windows are half-open;
  stale/malformed/mixed-source evidence fails red; live state continues after a reporting window closes; same-time
  transitions preserve controller order. Terminal identities are unique in both the writer and exported OEE boundary,
  timestamp-ordered, and require the matching PackML lifecycle. Good results require four recipe-consistent passing
  steps, a base, positive attempts, true final checks, and result-consistent recovery evidence; failed requires Aborted,
  not merely Stopped, while blocked requires active Held/Andon evidence. Passed steps require matching embedded
  inspection identity plus `checks_agree=true`, `verdict=ok`, and `action=use`. The writer and public OEE boundary share
  the same latest-earlier lifecycle validator, so stale, later-aborted, wrong-identity, and out-of-order evidence fails.
  Shared/exclusive locks prevent partial reads; write/fsync failures roll back and double-fsync
  failures report uncertain durability. The UI uses matching CSP hashes and safe text rendering, with no command route.
- Base and run: stacked on T15 `5b598d7` (including accepted T09 `d1bfd9c`); `uv run pytest -q tests/test_oee.py
  tests/test_run_log.py tests/test_dashboard.py` (78 passed); `uv run pytest -q` (625 passed); `uv run ruff check .`,
  `uv lock --check`, `uv pip check`, `git diff --check`, file/function gates, and unchanged-contract / unchanged-official
  client checks passed. A 24-process append stress recorded 24 unique rollouts. Browser HTTP tests ran outside the
  restricted loopback sandbox.
- Next: Claude reviews the T16 PR. A live integration must schedule `record_heartbeat` more frequently than the chosen
  stale threshold and supply only measured ideal/cycle times; use mock/replay sources until real-cell evaluation.

## 2026-10-29 · T15 · CODEX · immutable product birth certificates

- Changed: added `assembly/birth_certificate.py` to assemble one contract-valid traceability record from T09
  `PlanStepRecord` / `ControllerEvent` streams, archived post-placement inspections, and an explicit measured terminal
  outcome. Added `assembly/birth_certificate_pdf.py` for a one-page ReportLab layout and exported the public API from
  `assembly/__init__.py`. Each rollout/product now publishes as one atomic read-only directory under
  `eval/records/<run_id>/rollout-<number>/product-<number>/`, containing `certificate.json`, `certificate.pdf`, and a
  SHA-256 manifest. The manifest binds canonical JSON and exact PDF bytes; the canonical JSON digest is printed in the
  PDF, and `verify_birth_certificate_artifacts(...)` detects either artifact changing. PDF dependencies load only when
  an artifact is requested, so controller imports do not require ReportLab or Pillow. Validation, writing, and
  verification all reapply nonnegative-rollout, product 1–6, and base-identity rules; verification also requires the
  three typed artifact paths to use fixed filenames in one directory before any file is read.
- Behavior: product/inspection identities, recipe membership, terminal events, recovery classification, two final
  checks, four passing seated checks for good products, safe run IDs, finite JSON values, paired timing measurements,
  and an actively held terminal Andon for blocked outcomes all fail closed. Records are capped at the four-step product
  recipe so the one-page layout cannot overlap its final checks. Attempts, spare use, rebuilds, and
  deduplicated Andon assist episodes derive from T09 evidence. Unmeasured cycle time and Cosmos checks remain `null`;
  no value is inferred. No contract file or official
  `robot/cell/cell_client.py` file changed.
- PDF QA: rendered the generated PDF with `pdftoppm` at 144 DPI and inspected the 1224×1584 PNG. The single page had
  no clipped or overlapping text, all four step rows and evidence references were legible, status/final-check badges
  were aligned, and the full canonical JSON SHA-256, footer, and page number rendered correctly.
- Run: `uv run --locked pytest tests/test_birth_certificate.py -q` (34 passed); `uv run --locked pytest -q`
  (547 passed) after rebasing onto accepted T09 `d1bfd9c`; `uv run --locked ruff check .` (passed);
  `uv lock --check`, `uv pip check`, `git diff --check`, and
  unchanged-contract / unchanged-official-client checks (passed). Production files are 422, 151, 214, and 16 lines;
  no changed function exceeds 40 lines. Root dependencies are bounded to `reportlab>=4.2,<6` and `pypdf>=5,<7`.
- Next: Claude reviews the T15 PR. Runtime callers archive each final post-placement inspection as
  `StepInspectionEvidence`, call `build_birth_certificate(...)`, then publish the validated record with
  `write_birth_certificate(...)`. Training weights are not required; the same T27-shaped inspection payloads can be
  supplied by deterministic stubs until the reviewed models land.

## 2026-10-29 · T09 · CODEX · guarded PackML assembly controller

- Changed: products are parsed and bound from the first `obs["trial"]["instruction"]` of every reset; the constructor
  accepts no fixed product tuple. Inventory comes only from the perception port. Loose/table or reserved fallen,
  defective, look-alike, and unrequested parts now execute inspect → FoundationPose grasp → reject → independently
  verified transfer before a verified table replacement or spare is reserved. Unknown foreign objects Hold/Andon when
  no catalog identity and sourced grasp exist; installed/staged scan defects Hold for calibrated dismantle recovery.
- Motion/recovery: every normal or recovery `pre_pick` happens only after the sourced straight-down inspection stop;
  recovery no longer calls `grasp_for`. Safe-Z rise/XY/descent, nearest-equivalent W/P/R interpolation bounded to the
  official ±360° TCP contract, shortest yaw, configured per-step Cartesian speed checks, table-before-spare selection,
  operator acknowledge/rescan resume, and installed/held occlusion ownership are covered. Inspection actions are
  authoritative: `ask_operator` Holds pre/post, and physical reject actions are not inferred from raw scores.
- Runtime boundary: `AssemblyController` requires caller-sourced `InspectionStopConfig` and `StartupHoldConfig`; the
  latter is validated against the official action contract and prevents malformed first observations from escaping
  `act()`. `reset()` clears prompt ownership, `requested_products` exposes the parsed tuple, and
  `acknowledge_resume()` requires a fresh scan before leaving Held. The latest T04/T05 review fixes are integrated:
  every emitted joint/TCP action uses the live official guard, and the catalog pins T29's Team V printed-finger
  source at `20c6f9b` / SHA-256 `99611a032fb74a02740df405f1ef2d2d73f54f17cfc5676899cf9244e6d629c1`.
- Pose conversion: the provider composes measured UF0 base pose × CAD STL installation transform × inverse T29
  mesh-to-installed-local transform. It now carries two distinct internal frames: the canonical object origin used for
  camera verification and the T29 fingertip TCP used for motion. Reject verification uses calibrated zone occupancy,
  not equality to the pre-release TCP; staging/rebuild carry an explicit object verification target. A recorded
  `FoundationPoseFrameBoundary("installed_local", source)` is mandatory, so raw STL-frame output cannot silently enter
  the yaw-only path. Identity/source/finite checks fail closed, and fallen parts Hold when the shared contract cannot
  supply a sourced full-6D grasp.
- Run: `uv run ruff check .` (passed); `uv run pytest -q` (513 passed); `git diff --check` (clean);
  `git diff --exit-code origin/main -- robot/cell/cell_client.py` (clean). The integrated T04 catalog-schema change
  still requires Claude's latest-head contract review on PR #5; no additional T09 contract change was made.
  `cd robot/cell && uv run --locked run.py --mock` → 3 rollouts, 602 actions, 0 rejected and
  `✓ Ready for the real robot`; adding `--checkpoint t09-mock --trials 1 --trial-s 20` → 1 rollout,
  200 actions, 0 rejected, with the explicit mock-only/not-real-robot-ready verdict.
- Next: Claude re-reviews PR #9 at the integrated head. The cell-runtime branch must rebase here and remain disabled
  until T28/T30 live perception, reviewed recovery targets, and a sourced ER-4iA robot-link collision provider exist;
  do not select `t09-mock` on the real cell.

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
  are rechecked at load time, and source dates are explicitly labeled as Git commit timestamps. All six provenance
  fields for both sources are compared with immutable trusted records; typed catalog objects return those records.
- Regression evidence: NaN, Inf, singular, non-orthogonal, reflection, hash mismatch, zero/multi-grip ambiguity, roll
  mismatch, all 12 source-provenance mutations, Team V finger provenance, and unknown-key cases fail closed.
  `uv run ruff check .` passed; full `uv run pytest -q` passed 69 tests.
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
