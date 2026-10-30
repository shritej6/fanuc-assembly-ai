"""Scripted, sim-only skills that record demos for GR00T (docs/ROBOT_BRAIN.md). The real arm never runs these."""

from .insert import insert
from .pick import pick
from .place_free import place_free
from .pull_out import pull_out

__all__ = ["insert", "pick", "place_free", "pull_out"]
