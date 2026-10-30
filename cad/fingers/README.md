# Team V printed fingers (source copy)

Copied 2026-10-29 from `MHZL2_16D2_FANUC_FINAL_FINGERS` (built 2026-10-27, CadQuery source
`source/export_all.py`, 9/9 CAD gates). These are the fingers mounted for the CSI slot, not the organizers' aluminium
fingers. Status there: **PROVISIONAL CAD, cell measurements required**; no physical test performed.

- `product_grasp_frames.json`: per product part, the gripper pose and the contact centre in the product's assembly
  frame (mm), from that project's 24-part grasp validation (`24_part_grasp_report.csv`: approach, closure, seated
  grip, carry, insertion, retreat and release checked against the fixture and every installed part, 24/24 PASS).
- Key numbers (that project's README): finger 76 x 12 mm (the CSI TCP length is kept); NBR pad 6.8 x 3.7 x 1.5 mm at
  the absolute tip; fingertip gaps rubber 16.5 open / 4.5 closed, rigid stop 17.3 / 5.3; contact centre at Z 84.25 and
  tip plane at Z 86.10 in the gripper STEP frame, so the fingertip is 1.85 mm below the contact centre.
