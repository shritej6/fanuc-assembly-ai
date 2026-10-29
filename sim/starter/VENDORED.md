# Vendored: FANUC ER-4iA hackathon starter

- Source: the organizers' starter zip (`fanuc_er4ia_sim/`), vendored 2026-10-27.
- Unmodified. All 38 files match the SHA-256 hashes in `MANIFEST.json`. Re-check with:
  `python3 -c "import json,hashlib,pathlib as p;r=p.Path('sim/starter');print([f['path'] for f in json.load(open(r/'MANIFEST.json'))['files'] if hashlib.sha256((r/f['path']).read_bytes()).hexdigest()!=f['sha256']])"`
- Not vendored: `.venv/`, `textures/` (generated locally by `bootstrap.sh`).
- We import `cell.py` / `scene_config.py` / `ik.py` from here (cell, robot, gripper, cameras, lights). Our own code lives
  in `sdg/` and `sim/` outside this folder. Do not edit files here; wrap them instead.
- Validated upstream on Isaac Sim 6.0 (RTX A6000). Rosie has `isaac-sim-6.1.0.sif`, not 6.0 (see `docs/SDG_PLAN.md`).
