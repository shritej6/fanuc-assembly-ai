# FANUC Self-Inspecting Assembly Cell

Rockwell Automation + NVIDIA Physical AI Hackathon (Fall 2026), Challenge 3, FANUC ER-4iA cell.

The cell finds the parts of 3 (of 6) black 3D-printed products lying at random on the table, groups them into
products, inspects every part, swaps defective parts for fresh spares, assembles each product on its base with a
check after every step, and disassembles and rebuilds when a build is wrong.

- Plan and locked decisions: [docs/PLAN.md](docs/PLAN.md)
- Task board (who does what): [coord/TASKS.md](coord/TASKS.md)
- Shared data formats: [contracts/](contracts/)
- Official real-cell client (vendored, unmodified): [robot/cell/](robot/cell/) — upstream commit in `robot/cell/UPSTREAM_COMMIT`

## Quick check of the real-cell client

```bash
cd robot/cell && uv sync && uv run run.py --mock
```
