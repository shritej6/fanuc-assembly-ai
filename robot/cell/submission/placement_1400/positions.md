# CSI real-cell placement, 2026-10-31 14:00 CT

x, y = centre of the part's footprint (mm, robot frame UF0, as obs["tcp_pose"]). x grows towards the front edge (away from the robot); +y is to the right in table_plan.png and in the PNGs here.
yaw = direction of the part's CAD local +x axis (red arrow in rollout_N.png), degrees from +x turning towards +y. Every part stands in its installed orientation. Match the silhouette in the PNG.

Bases sit on spots B1 (420, -165), B2 (420, 0), B3 (420, 165); loose parts on a fixed 2 x 6 grid (rows x 255 / 335, columns y -225, -135, -45, 45, 135, 225). Reject zone and spare store stay empty.

## Rollout 1: products 1, 2, 3

Prompt: `Assemble products 1, 2 and 3 on their bases. Reject any part that is defective, fallen or not in these products, and use a spare from the store instead.`

| # | object | spot | x | y | yaw | pick roll r | insert roll r |
|---|---|---|---|---|---|---|---|
| 1 | F01 base (Rotary Fixture, product 1) | B1 | 420 | -165 | 90 | - | - |
| 2 | M1_01 Housing | PF2 | 335 | -135 | 75 | 165 | 180 |
| 3 | M1_02 Spindle Carrier | PN2 | 255 | -135 | 110 | 200 | 180 |
| 4 | M1_03 Rotary Knob | PN1 | 255 | -225 | 70 | 160 | 180 |
| 5 | M1_04 Rear Guard | PF1 | 335 | -225 | 100 | 190 | 180 |
| 6 | F02 base (Indicator Fixture, product 2) | B2 | 420 | 0 | 90 | - | - |
| 7 | M2_01 Housing | PF3 | 335 | -45 | 105 | 195 | 180 |
| 8 | M2_02 Diffuser Carrier | PN3 | 255 | -45 | 80 | 170 | 180 |
| 9 | M2_03 Indicator Cap | PF4 | 335 | 45 | 70 | 160 | 180 |
| 10 | M2_04 Rear Guard | PN4 | 255 | 45 | 110 | 200 | 180 |
| 11 | F03 base (Pushbutton Fixture, product 3) | B3 | 420 | 165 | 90 | - | - |
| 12 | M3_01 Housing | PF6 | 335 | 225 | 100 | 190 | 180 |
| 13 | M3_02 Actuator Stem | PN6 | 255 | 225 | 75 | 165 | 180 |
| 14 | M3_03 Button Cap | PN5 | 255 | 135 | 110 | 200 | 180 |
| 15 | M3_04 Rear Guard | PF5 | 335 | 135 | 80 | 170 | 180 |

## Rollout 2: products 4, 5, 6

Prompt: `Assemble products 4, 5 and 6 on their bases. Reject any part that is defective, fallen or not in these products, and use a spare from the store instead.`

| # | object | spot | x | y | yaw | pick roll r | insert roll r |
|---|---|---|---|---|---|---|---|
| 1 | F04 base (Sensor Connector Fixture, product 4) | B1 | 420 | -165 | 45 | - | - |
| 2 | M4_01 Housing | PF2 | 335 | -135 | 25 | 205 | 225 |
| 3 | M4_02 Sensor Insert | PN2 | 255 | -135 | 60 | 150 | 135 |
| 4 | M4_03 Upper Guide | PN1 | 255 | -225 | 70 | 160 | 135 |
| 5 | M4_04 Top Lock | PF1 | 335 | -225 | 55 | 145 | 135 |
| 6 | F05 base (IO Connector Fixture, product 5) | B2 | 420 | 0 | 45 | - | - |
| 7 | M5_01 Housing | PF3 | 335 | -45 | 30 | 210 | 225 |
| 8 | M5_02 IO Insert | PN3 | 255 | -45 | 65 | 155 | 135 |
| 9 | M5_03 Alignment Guide | PF4 | 335 | 45 | 55 | 145 | 135 |
| 10 | M5_04 Service Lock | PN4 | 255 | 45 | 70 | 160 | 135 |
| 11 | F06 base (Power Connector Fixture, product 6) | B3 | 420 | 165 | 45 | - | - |
| 12 | M6_01 Housing | PF6 | 335 | 225 | 20 | 200 | 225 |
| 13 | M6_02 Power Insert | PN5 | 255 | 135 | 55 | 145 | 135 |
| 14 | M6_03 Power Guide | PN6 | 255 | 225 | 65 | 155 | 135 |
| 15 | M6_04 Power Lock | PF5 | 335 | 135 | 60 | 150 | 135 |

## Rollout 3: products 1, 3, 5

Prompt: `Assemble products 1, 3 and 5 on their bases. Reject any part that is defective, fallen or not in these products, and use a spare from the store instead.`

| # | object | spot | x | y | yaw | pick roll r | insert roll r |
|---|---|---|---|---|---|---|---|
| 1 | F01 base (Rotary Fixture, product 1) | B3 | 420 | 165 | 90 | - | - |
| 2 | M1_01 Housing | PF5 | 335 | 135 | 110 | 200 | 180 |
| 3 | M1_02 Spindle Carrier | PN5 | 255 | 135 | 75 | 165 | 180 |
| 4 | M1_03 Rotary Knob | PF6 | 335 | 225 | 100 | 190 | 180 |
| 5 | M1_04 Rear Guard | PN6 | 255 | 225 | 70 | 160 | 180 |
| 6 | F03 base (Pushbutton Fixture, product 3) | B1 | 420 | -165 | 90 | - | - |
| 7 | M3_01 Housing | PF2 | 335 | -135 | 80 | 170 | 180 |
| 8 | M3_02 Actuator Stem | PN2 | 255 | -135 | 105 | 195 | 180 |
| 9 | M3_03 Button Cap | PF1 | 335 | -225 | 110 | 200 | 180 |
| 10 | M3_04 Rear Guard | PN1 | 255 | -225 | 75 | 165 | 180 |
| 11 | F05 base (IO Connector Fixture, product 5) | B2 | 420 | 0 | 45 | - | - |
| 12 | M5_01 Housing | PF4 | 335 | 45 | 25 | 205 | 225 |
| 13 | M5_02 IO Insert | PN4 | 255 | 45 | 65 | 155 | 135 |
| 14 | M5_03 Alignment Guide | PN3 | 255 | -45 | 55 | 145 | 135 |
| 15 | M5_04 Service Lock | PF3 | 335 | -45 | 70 | 160 | 135 |
