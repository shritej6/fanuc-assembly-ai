# CSI real-cell placement, 2026-10-31 14:00 CT (one product per rollout)

x, y = centre of the part's footprint (mm, robot frame UF0, as obs["tcp_pose"]). x grows towards the front edge (away from the robot); +y is to the right in table_plan.png and in the PNGs here.
yaw = direction of the part's CAD local +x axis (red arrow in rollout_N.png), degrees from +x turning towards +y. Every part stands in its installed orientation. Nothing else on the table.

## Rollout 1: product 5

Prompt: `Assemble product 5 on its base. Reject any part that is defective, fallen or not in this product, and use a spare from the store instead.`

| # | object | x | y | yaw | pick roll r | insert roll r |
|---|---|---|---|---|---|---|
| 1 | F05 base (IO Connector Fixture, product 5) | 275 | -123 | -34 | - | - |
| 2 | M5_01 Housing | 358 | 193 | -30 | 150 | 146 |
| 3 | M5_02 IO Insert | 305 | 36 | -86 | 184 | 236 |
| 4 | M5_03 Alignment Guide | 302 | 115 | -92 | 178 | 236 |
| 5 | M5_04 Service Lock | 383 | 12 | -79 | 191 | 236 |

## Rollout 2: product 5

Prompt: `Assemble product 5 on its base. Reject any part that is defective, fallen or not in this product, and use a spare from the store instead.`

| # | object | x | y | yaw | pick roll r | insert roll r |
|---|---|---|---|---|---|---|
| 1 | F05 base (IO Connector Fixture, product 5) | 275 | -123 | -34 | - | - |
| 2 | M5_01 Housing | 358 | 193 | -30 | 150 | 146 |
| 3 | M5_02 IO Insert | 305 | 36 | -86 | 184 | 236 |
| 4 | M5_03 Alignment Guide | 302 | 115 | -92 | 178 | 236 |
| 5 | M5_04 Service Lock | 383 | 12 | -79 | 191 | 236 |

## Rollout 3: product 5

Prompt: `Assemble product 5 on its base. Reject any part that is defective, fallen or not in this product, and use a spare from the store instead.`

| # | object | x | y | yaw | pick roll r | insert roll r |
|---|---|---|---|---|---|---|
| 1 | F05 base (IO Connector Fixture, product 5) | 275 | -123 | -34 | - | - |
| 2 | M5_01 Housing | 358 | 193 | -30 | 150 | 146 |
| 3 | M5_02 IO Insert | 305 | 36 | -86 | 184 | 236 |
| 4 | M5_03 Alignment Guide | 302 | 115 | -92 | 178 | 236 |
| 5 | M5_04 Service Lock | 383 | 12 | -79 | 191 | 236 |
