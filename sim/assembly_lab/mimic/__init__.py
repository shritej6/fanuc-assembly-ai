"""Isaac Lab Mimic data multiplier for the T12 assembly cell (agents 4 / 7).

`generate.py` runs the MimicGen algorithm (object-centric subtask segments, nearest-neighbour source selection,
connectors, tapered action noise; `mimicgen.py`) over agent 1's env through agent 3's IsaacBackend + EpisodeRecorder,
so every generated demo passes exactly the gates a scripted source demo must pass and is written by the same Isaac Lab
recorder. `run_generation.py` launches workers on one host; `scale_generation.sh` drives several hosts, then
`export_lerobot.py` / `validate_lerobot.py` produce the GR00T LeRobot v2.1 dataset.

Host-safe: catalog, frames, force_guard, gates, instructions, spec, mimicgen, lerobot_writer, export_lerobot,
validate_lerobot, run_generation. Needs Isaac Lab: generate.
"""
