"""PhysX contact / material settings: agent 2's sim/assembly_lab/physics/contact.yaml mapped onto the env's keys,
with the env defaults (labelled, not tuned) for anything it does not set."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
CONTACT_YAML = REPO / "sim" / "assembly_lab" / "physics" / "contact.yaml"
# Defaults used when agent 2's contact.yaml is absent. NOT tuned: textbook-range PLA / NBR friction, SDF settings
# from the PhysX SDF guide scaled to 20-60 mm parts (256 -> 0.08-0.23 mm cells, below the 0.5 mm clearance).
DEFAULT_CONTACT = {
    "finger_collider": "sdf",
    "sleep_threshold": 5e-5,          # PhysX defaults (mass-normalised kinetic energy, m^2/s^2); 0 = never sleep
    "stabilization_threshold": 1e-5,       # Team V finger mesh collider: sdf (exact to ~0.25 mm) or convexDecomposition
    "physics_dt": 1.0 / 240.0,
    "solver_position_iterations": 32,
    "solver_velocity_iterations": 1,
    "contact_offset_m": 0.0005,
    "rest_offset_m": 0.0,
    "max_depenetration_velocity": 0.05,
    "sdf_resolution": 256,
    "sdf_subgrid_resolution": 6,
    "sdf_margin": 0.01,              # fraction of the mesh bbox diagonal (PhysX); sdf_margin_m overrides it per part
    "sdf_margin_m": None,
    "sdf_narrow_band_thickness": 0.01,
    "friction_offset_threshold": 0.001,
    "friction_correlation_distance": 0.0005,
    "bounce_threshold_velocity": 0.05,
    "materials": {
        "pla": {"static_friction": 0.45, "dynamic_friction": 0.35, "restitution": 0.0, "friction_combine": "average"},
        "nbr_pad": {"static_friction": 1.2, "dynamic_friction": 1.0, "restitution": 0.0, "friction_combine": "max"},
        "finger": {"static_friction": 0.45, "dynamic_friction": 0.35, "restitution": 0.0, "friction_combine": "average"},
        "table": {"static_friction": 0.4, "dynamic_friction": 0.3, "restitution": 0.0, "friction_combine": "average"},
    },
}


# Env-level settings on top of physics/contact.yaml, each with the measurement behind it (demo_check.py, M4_01 CAD
# grasp, 34 N jaws): 32 position iterations let the pads sink ~0.9 mm per side into the 7 mm grip faces (jaw gap
# 5.2 mm) and the part pitched 6-11 deg in the grip on lift; 128 gave gap 6.3 mm and 0.003 deg pitch.
# max_depenetration_velocity: contact.yaml's 0.05 m/s caps how fast PhysX pushes a part back out of a finger, so the
# 34 N pads sank 0.25-0.4 mm into a held part and, with the old jaw drive, ratcheted 1-1.8 mm in onto the jaw stop
# (grasp_diag.py M1_02: pinch collapsed to 3-10 N, 1.8-2.4 mm slip in carry). 5.0 m/s is what Isaac Lab's own SDF
# peg-in-hole tasks use for their parts (isaaclab_tasks direct/factory/factory_tasks_cfg.py, direct/automate); with it
# the pads stop within 0.05 mm of the CAD contact.
ENV_OVERRIDES = {"solver_position_iterations": 128, "max_depenetration_velocity": 5.0}


def load_contact(path: Path, overrides: dict) -> tuple[dict, str]:
    cfg = copy.deepcopy(DEFAULT_CONTACT)
    source = "defaults (sim/assembly_lab/physics/contact.yaml absent)"
    if path.exists():
        _merge(cfg, from_agent2_schema(yaml.safe_load(path.read_text()) or {}))
        source = str(path)
    _merge(cfg, ENV_OVERRIDES)
    env_set = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in os.environ.get("T12_CONTACT_SET", "").split(",") if kv)}
    _merge(cfg, env_set)
    _merge(cfg, overrides)
    if env_set:
        source += f" + T12_CONTACT_SET {env_set}"
    return cfg, source


AGENT2_SOLVER = {"physics_dt_s": "physics_dt", "position_iterations": "solver_position_iterations",
                 "velocity_iterations": "solver_velocity_iterations",
                 "max_depenetration_velocity_m_s": "max_depenetration_velocity",
                 "bounce_threshold_velocity_m_s": "bounce_threshold_velocity"}
AGENT2_COLLISION = {"sdf_resolution": "sdf_resolution", "sdf_margin_m": "sdf_margin_m",
                    "contact_offset_m": "contact_offset_m", "rest_offset_m": "rest_offset_m"}
AGENT2_MATERIALS = {"pla_printed": "pla", "nbr_fingertip": "nbr_pad", "table": "table"}


def from_agent2_schema(doc: dict) -> dict:
    """Map agent 2's contact.yaml (materials / collision / solver sections) onto this env's flat keys."""
    out = {k: v for k, v in doc.items() if k not in ("materials", "collision", "solver", "pairs")}
    for src, dst in AGENT2_SOLVER.items():
        if src in doc.get("solver", {}):
            out[dst] = doc["solver"][src]
    for src, dst in AGENT2_COLLISION.items():
        if src in doc.get("collision", {}):
            out[dst] = doc["collision"][src]
    mats = {}
    for src, dst in AGENT2_MATERIALS.items():
        m = doc.get("materials", {}).get(src)
        if m:
            mats[dst] = {k: m[k] for k in ("static_friction", "dynamic_friction", "restitution") if k in m}
            if "friction_combine_mode" in m:
                mats[dst]["friction_combine"] = m["friction_combine_mode"]
    if mats:
        out["materials"] = mats
    return out


def _merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        elif k in DEFAULT_CONTACT or k in dst:
            dst[k] = v
        else:
            print(f"[cell_env] contact.yaml: ignoring unknown key {k!r}", file=sys.stderr)
