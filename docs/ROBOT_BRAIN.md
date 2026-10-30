# Robot brain: how the cell builds the products (T12 + T18)

The user's decisions of 2026-10-29 (night), for the real run on Sat 2026-11-01. GR00T N1.7 moves the real arm, with no
scripted fallback (`coord/DECISIONS.md`). Scripted motion exists only in Isaac Sim, to record demos. Nothing here
changes a contract; where the state machine (T09, Codex) must change, this file is the spec it follows.

## 1. The loop

1. **Scan**: table and side cameras with the T27 detector, and the wrist camera at the home pose (it sees the whole
   table).
2. **Identify and measure** every object:
   - Run a wrist close-up for every part that the plan needs, and for any object that "looks different".
   - **Look-alikes differ in dimensions.** Measure the size at this first step, using wrist depth plus the CAD fit
     (FoundationPose against the STL). Match the measured size to the right product, not the look.
   - Classify each object as good part / damaged part / fallen part / unknown piece (see §3).
3. **Plan** (the planner decides; GR00T executes):
   - The product order follows the parts available.
   - The part order follows the assembly sheets (01 → 04).
   - For every pick and place, the planner sets:
     - the grip faces (CAD grip pads, `cad/grasp_frames.json`);
     - the wrist turn (`r = measured yaw + roll_offset`, whichever of ±180 is nearest the current r and inside J6);
     - the approach and retreat paths, checked for finger clearance against every neighbour.
4. **Act**: GR00T gets the cameras, the joints and an instruction that carries the plan's target, for example
   "pick M4_02 at (x, y, yaw) gripping X, insert on F04 at (x, y, yaw)". It outputs `act_tcp` targets.
5. **Check after every step**:
   - **Pre-pick**: is the part in the fingers?
   - **Post-place**: is it the right part, fully down, not crooked? Compare the seated height with the CAD.
   - **Finished product**: a close-up from 2 wrist angles.

## 2. Contact: grip, insert, pull out

- **Grip**:
  - Use only the CAD grip pad, which is 4.5–16.5 mm and fits the Team V fingers (24/24 PASS in
    `cad/fingers/24_part_grasp_report.csv`).
  - A part whose pad is blocked at every allowed wrist turn is **crowded**. Move the blocking part first. Never push or
    bump.
- **Insert, human style**:
  1. Come in tilted about 3°.
  2. Touch one edge into the 1.5 mm entry chamfer.
  3. Level out while sliding in.
  4. Twist ±2°.
  5. Give a light push to the seated height.
  - All pegs and holes have 0.5 mm clearance per side (`cad/products/parameters.py`).
- **Pull out (dismantle)**: go straight up and slowly, with a ±1° wiggle. Plan the path clear of every other installed
  part.
- **Force**: the real arm has no force sensor.
  - In sim, contact force is measured. A demo that exceeds a limit is thrown away, so GR00T only ever sees gentle
    contact.
  - On the real arm, a guard stops any target lower than 1 mm below the seated height, and the last 5 mm is slow.
  - The limits come from the physics study (agent 2). The calculated starting point is PRELIMINARY: ≤ 15 N straight
    down and ≤ 5 N sideways, because the Ø3 mm, 2.5 mm tall PLA peg's layers fail at about 16 N sideways.

## 3. Every case (user answers)

| Case | What the robot does |
|---|---|
| Unknown or foreign piece (not one of the 30) | Move it to the defect area if the fingers can grip it safely. Otherwise leave it as an obstacle, and Andon if it blocks. |
| Damaged part (snapped tab, missing peg, warp, wrong size) | Defect area, then take a spare. No spare: Andon. |
| Fallen part (not in its installed orientation) | Reject, then take a spare (the arm cannot reorient). |
| Checks disagree or are unsure | Second close-up from another angle. Still unsure: Andon, showing both pictures (use it, or reject?). |
| Part needed but missing | Build the other products first, then come back. Then the spare store. Then Andon (which part, which slot). |
| Crowded: no finger fit at any wrist turn | Move the blocking part first (to an empty, remembered test-bed spot if it is not needed yet). |
| Won't seat | Retry once: lift 3 mm, re-measure, align again. Then pull out gently and inspect the part and the holes. Defect: defect area + spare. Holes blocked: Andon. |
| Drops during carry | Find it. Upright: re-plan and re-pick. Fallen: reject + spare. Not found: Andon. |
| Wrong part or crooked after placing, with parts above it | Remove only the parts above it, in reverse sheet order. Put the good ones on an empty, remembered test-bed spot. Fix, then rebuild. |
| Finished product | Close-up from 2 wrist angles. Pass: leave it in place. Fail: fix as above. |
| Something enters the work area during a move | Freeze (keep holding), re-scan, re-plan. Still blocked: Andon. |

## 4. Who builds it (6 agents, branch `feature/T12-mimic`, then `feature/T18-policy-learn`)

| # | Agent | Owns | Delivers |
|---|---|---|---|
| 1 | Sim cell | `sim/assembly_lab/env/` | Isaac Lab cell: ER-4iA, Team V fingers, 30 PLA parts (mass from CAD volume), test bed, spare store, defect area, CSI camera poses |
| 2 | Object master + contact physics | `sim/assembly_lab/objects/`, `physics/` | Per-object knowledge (sizes that tell look-alikes apart, mass, grip pads, allowed wrist turns, pegs and holes, mates, seated height, pull-out direction). Grasp and path planner with finger clearance. Contact tuning and the measured force limits. |
| 3 | Skill demos | `sim/assembly_lab/skills/` | Scripted sim-only pick, carry, insert (tilt-touch-level-wiggle) and pull-out, with the force guard. Recovery demos for the §3 cases. |
| 4 | Mimic | `sim/assembly_lab/mimic/` | Isaac Lab Mimic: about 10 demos per skill become thousands (positions, yaw, light, camera jitter). LeRobot export. |
| 5 | GR00T trainer | `policy_learn/groot/` | GR00T N1.7 fine-tune on Brev H100s (instruction + cameras + joints → `act_tcp` chunks), with a sim eval each epoch |
| 6 | Evaluator + safety | `sim/assembly_lab/eval/`, `policy_learn/deploy/` | Unseen sim layouts and every §3 case: success rate, force peaks, collisions. Real-arm guards. The `policy.py` adapter, checked with `run.py --mock`. |

Budget: up to $250 of Brev credit (user, 2026-10-29).
