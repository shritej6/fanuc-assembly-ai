# Brief for Codex (2026-10-30, from Claude; the user asked for it)

The robot brain is now GR00T-only on the real arm (`coord/DECISIONS.md`, 2026-10-29). Read
`docs/ROBOT_BRAIN.md` (every case rule the user chose) and `sim/assembly_lab/INTERFACES.md` (units, frames, precision
gates) first. Branch with all of it: `feature/T12-mimic` (local, not pushed yet). Ask the user for the push if you
need it.

## 1. T09 controller (PR #9): implement the user's case rules
`docs/ROBOT_BRAIN.md` §3 covers these cases, each with an Andon reason:
- unknown piece → defect area
- damaged part → defect area + spare
- unsure → second close-up, then Andon
- missing part → build the other products first, then the spare, then Andon
- crowded part → move the blocker first
- won't seat → 2 tries, then pull out and inspect the part and the holes
- dropped part → re-pick if upright, reject if fallen
- fix by reverse removal → good parts go to an empty, remembered test-bed spot
- finished product → 2-angle close-up check
- intrusion → freeze, re-scan, re-plan

Also, from the earlier review (PR #9):
- Targets must use `tcp_local_mm`, not the part origin (it can be up to 27.9 mm too low for M1_04).
- Roll = yaw + `roll_offset_deg` (+90 for X pads, +0 for Y pads). Source: `cad/grasp_frames.json`, T29 PR #15.

## 2. PerceptionPort tolerances: measured, never defaults
`ExecutionTolerances` / `PerceptionTolerances` must come from measured values:
- `sim/assembly_lab/objects/objects.yaml` → `insertion.{xy_clearance_mm, xy_capture_mm, yaw_tol_deg}`. The smallest
  yaw window is 1.367°.
- The precision gates: seated xy ≤ 0.3 mm, yaw ≤ 1°, z ±0.3 mm, tilt ≤ 1°.

Force caps are per part: `objects.yaml` `max_force` (M5_04/M6_04 axial 2.5 N, sim-measured; the other limits are
PRELIMINARY).

Look-alikes are decided by measured size and features from the wrist close-up:
- Two views at 150 mm: top-down and oblique.
- See `sim/assembly_lab/objects/LOOKALIKES.md`.
- F01=F02 and F04=F05=F06 are byte-identical and interchangeable.
- The guides M4/5/6_03 differ only in their bottom keying.

Claude implements the PerceptionPort adapter (T30), which adds a good/defective verdict from the wrist classifier.

## 3. T05 (PR #4): wire GR00T into `policy.py`
- `policy_learn/deploy/groot_policy.py` exposes `act(obs)`, with every chunk passed through
  `policy_learn/deploy/guards.py`.
- The wiring note `policy_learn/deploy/WIRING.md` comes from Claude's evaluator agent.
- The instruction must come from `sim/assembly_lab/mimic/instructions.instruction(plan_ids, plan_target)`, the same
  text the demos were trained on.

## 4. Contract PR (both agents review)
- `contracts/taxonomy.json`: add the look-alike group M1_01/M3_01. They are size twins (36×49×16 mm; volume 16.7 vs
  18.4 cm³) and are missing from the groups.
- A possible inspection-schema field for the good/defective verdict. The defect agent writes the proposal; nothing is
  edited without the PR.

## 5. Reviews wanted
PR #15 (T29 grasp frames), PR #7 (T27 training), then the T31 sets and the T12/T18 branch once they are pushed.
