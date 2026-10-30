"""Isaac Lab Mimic data multiplier for the T12 assembly cell (agent 4).

Agent 1's AssemblyCellEnv is a single-env SimulationContext wrapper, not the ManagerBasedRLMimicEnv that Isaac Lab's
annotate_demos.py / generate_dataset.py need, so `mimicgen.py` + `generate.py` run the same Mimic algorithm directly
on it (object-centric subtask segments, nearest-neighbour source selection, interpolation, action noise) and keep
only demos that pass the precision gates.

Host-safe: catalog, frames, force_guard, gates, instructions, spec, mimicgen, lerobot_writer, export_lerobot,
validate_lerobot, run_generation. Needs Isaac Lab (or the kinematic double): generate.
"""
