# Handoff log

Newest first. One entry per finished task.

## 2026-10-27 · T08 (prep) · CLAUDE · synthetic data plan, scene sampler, Isaac generator draft

- Changed: `docs/SDG_PLAN.md`; `sdg/` (configs, CAD geometry tool + data, scene sampler, post-processing, Isaac stage
  builder and render loop, Rosie Slurm scripts); `cad/stl/` (the 30 print STLs); `sim/starter/` (organizers' starter,
  unmodified, hashes verified); `numpy` added to the dependencies.
- Run: `uv sync && uv run pytest -q` (34 pass). `uv run python -m sdg.scene_sampler --set A --chunk 0 --out /tmp/a.jsonl`.
- Facts: every STL registers onto the CAD assembly within 0.003 mm; every resting pose sits on z = 0; all 54,000
  scenes (108 chunks) lay out. Rosie has Isaac 6.1.0 / 5.0.0 containers, **no 6.0**; `/home` is 87% full.
- Not tested: `sdg/isaac/*` and `sdg/rosie/*`; they need Isaac on Rosie. The pilot is their first run.
- For Codex (review): the label fields in `docs/SDG_PLAN.md` section 4 are what T13/T14 will read.
- Next: U07 (user picks the container), T07 USD + black PLA look check, T06 defect meshes, then the pilot.

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
