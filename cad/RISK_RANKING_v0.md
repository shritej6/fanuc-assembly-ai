# CAD risk ranking v0: PRELIMINARY

These scores come only from design factors measured on the delivered STLs: thin walls, the smallest layer
cross-section in the print orientation, and slenderness. They are **not** failure probabilities. They decide what to
cycle-test and which spares to stock first. Cycle tests (U05) and real runs replace them. The full table is in
`risk_ranking_v0.json`; run `cad/risk_ranking.py` to regenerate it.

## Top 10 product parts

| Rank | Part | Risk score | Why |
|---|---|---|---|
| 1 | M2_03 Indicator Cap | 0.90 | thinnest walls (5th percentile 0.5 mm), 7.9 % of the surface under 1.6 mm, very slender |
| 2 | M2_02 Diffuser Carrier | 0.87 | 1.0 mm walls, 5.6 % thin |
| 3 | M1_03 Rotary Knob | 0.81 | 1.2 mm walls, 7.7 % thin |
| 4 | M6_02 Power Insert | 0.70 | 2.0 mm walls, 2.4 % thin |
| 5 | M3_03 Button Cap | 0.69 | 3.9 % thin |
| 6 | M5_02 IO Insert | 0.62 | 3.1 % thin |
| 7 | M4_04 Top Lock | 0.60 | smallest layer cross-section (15.3 mm²): may split along a layer line |
| 8 | M4_02 Sensor Insert | 0.57 | 2.2 % thin |
| 9 | M5_04 Service Lock | 0.57 | 15.3 mm² weakest layer |
| 10 | M6_04 Power Lock | 0.54 | 15.3 mm² weakest layer |

The housings, rear guards and bases rank lowest.

## Print list for U04 (13 extra black PLA parts, within the 10–15 budget)

| Use | Parts | Qty |
|---|---|---|
| High-risk spares (spare store) | M2_03, M2_02, M1_03, M6_02, M3_03, M4_04 | 6 |
| Defect blanks, damaged by hand after printing (more realistic than CAD-made defects) | M2_03: snap one thin feature · M1_03: crack / chip · M4_04: snap the lock arm · M5_02: print at **103 % scale** in the slicer (oversize defect) | 4 |
| Cycle-test copies (about 25 assemble/disassemble cycles each, log cracks, jams, slips, failed extractions) | M2_03, M1_03, M4_04 | 3 |

**Caveat on look-alikes:** M4_04, M5_04 and M6_04 differ by only about 1 mm, so a lock spare fits only its own
product. A spare M5_04 or M6_04 placed in a rollout that doesn't need it doubles as a **wrong-part trap**.
