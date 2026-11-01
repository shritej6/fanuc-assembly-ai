# Robot brain: how the cell builds the products

Authored by Shritej Gunda.

GR00T N1.7 moves the real arm, and a CAD-based planner decides what it does: which part, which grip faces, which
wrist turn and which path. Scripted motion in Isaac Sim records the training demonstrations.

## 1. The loop

1. **Scan**: table and side cameras with the detector, and the wrist camera at the home pose (it sees the whole
   table).
2. **Identify and measure** every object:
   - Run a wrist close-up for every part that the plan needs, and for any object that "looks different".
   - **Look-alikes differ in dimensions.** Measure the size at this first step, using wrist depth plus the CAD fit
     (FoundationPose against the STL). Match the measured size to the right product, not the look.
   - Classify each object as good part / damaged part / fallen part / unknown piece (see section 3).
3. **Plan** (the planner decides; GR00T executes):
   - The product order follows the parts available.
   - The part order follows the assembly sheets (01 to 04).
   - For every pick and place, the planner sets:
     - the grip faces (CAD grip pads, `cad/grasp_frames.json`);
     - the wrist turn (`r = measured yaw + roll_offset`, whichever of +-180 is nearest the current r and inside J6);
     - the approach and retreat paths, checked for finger clearance against every neighbour.
4. **Act**: GR00T gets the cameras, the joints and an instruction that carries the plan's target, for example
   "pick M4_02 at (x, y, yaw) gripping X, insert on F04 at (x, y, yaw)". It outputs `act_tcp` targets.
5. **Check after every step**:
   - **Pre-pick**: is the part in the fingers?
   - **Post-place**: is it the right part, fully down, not crooked? Compare the seated height with the CAD.
   - **Finished product**: a close-up from 2 wrist angles.

## 2. Contact: grip, insert, pull out

- **Grip**:
  - Use only the CAD grip pad, which is 4.5 to 16.5 mm and fits the Team V fingers (all 24 parts pass in
    `cad/fingers/24_part_grasp_report.csv`).
  - A part whose pad is blocked at every allowed wrist turn is **crowded**. Move the blocking part first. Never push or
    bump.
- **Insert, human style**:
  1. Come in tilted about 3 deg.
  2. Touch one edge into the 1.5 mm entry chamfer.
  3. Level out while sliding in.
  4. Twist +-2 deg.
  5. Give a light push to the seated height.
  - All pegs and holes have 0.5 mm clearance per side (`cad/products/parameters.py`).
- **Pull out (dismantle)**: go straight up and slowly, with a +-1 deg wiggle. Plan the path clear of every other installed
  part.
- **Force**: the real arm has no force sensor.
  - In sim, contact force is measured. A demo that exceeds a limit is thrown away, so GR00T only ever sees gentle
    contact.
  - On the real arm, a guard stops any target lower than 1 mm below the seated height, and the last 5 mm is slow.
  - The limits come from the physics study (`sim/assembly_lab/physics/`). The calculated starting point is an estimate: at most 15 N straight
    down and at most 5 N sideways, because the 3 mm diameter, 2.5 mm tall PLA peg's layers separate at about 16 N sideways.
  - On the real arm, safety relies on the software guards, slow insertion, the FANUC native collision stop and the
    physical emergency stop. The force values above filter simulation demonstrations; without a sensor they are
    not measured on the real arm.

## 3. Every case

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
