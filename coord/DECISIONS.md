# Decisions

All decisions locked on 2026-10-27 are listed in `docs/PLAN.md` sections 2, 4 and 10a. New decisions are appended
here with a date and who made them (the user decides; agents propose).

| Date | Decision | By |
|---|---|---|
| 2026-10-27 | Follow the Team V tech stack exactly | user |
| 2026-10-27 | Bases never move; assemble in place; one product at a time | user |
| 2026-10-27 | Industry layer: PackML, birth certificate, two independent checks, OEE/Andon, OPC UA, Kanban + Pareto, FAT report | user |
| 2026-10-27 | All poses in contracts use the controller frame UF0 in mm/deg (same as `obs["tcp_pose"]` and `"tcp"` actions) | Claude (proposal, part of T01) |
| 2026-10-27 | Synthetic data root: `/data/ai_club/Innovation_lab_team_v` on Rosie (pending group write access); interim `~/fanuc-assembly-data` | user |
| 2026-10-29 | Cell TCP targets are fail-closed to the measured set-O inspection envelope: fingers down with `w` within 3 degrees of +/-180 and `p` within 3 degrees; yaw remains free within the official +/-360 limit | user/T28 measured follow-up on PR #9 |
