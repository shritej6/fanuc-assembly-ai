"""Typed perception port failures; all are ControllerPortError so the controller holds.
Authored by Shritej Gunda."""

from __future__ import annotations

from assembly.controller_ports import ControllerPortError


class PerceptionPortError(ControllerPortError):
    """The port cannot produce trustworthy evidence for this observation; nothing is guessed."""


class ObservationInputError(PerceptionPortError):
    """The observation lacks an image, depth map, joint vector or time stamp the call needs."""


class ModelOutputError(PerceptionPortError):
    """A model backend returned malformed output (bad box, probability outside [0, 1], non-rigid pose)."""


class CadSourceError(PerceptionPortError):
    """The pinned CAD snapshot is missing, altered (sha256) or lacks a field for this object."""


class UnknownInstanceError(PerceptionPortError):
    """The controller named an instance the port never published in a scan."""
