"""SHA-pinned CAD footprints for the learned-motion contact-floor guard."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256

from assembly.catalog import OBJECT_GEOMETRY_SOURCE_SNAPSHOT_PATH, TRUSTED_GEOMETRY_SOURCE


class ContactGeometryError(ValueError):
    """Pinned geometry cannot safely define a contact envelope."""


@dataclass(frozen=True, slots=True)
class ObjectFootprint:
    center_xy_mm: tuple[float, float]
    radius_mm: float


@lru_cache(maxsize=1)
def _pinned_footprints() -> dict[str, ObjectFootprint]:
    contents = OBJECT_GEOMETRY_SOURCE_SNAPSHOT_PATH.read_bytes()
    if sha256(contents).hexdigest() != TRUSTED_GEOMETRY_SOURCE.sha256:
        raise ContactGeometryError("contact geometry does not match the pinned CAD SHA-256")
    try:
        objects = json.loads(contents)["objects"]
        footprints = {
            object_id: ObjectFootprint(
                tuple(float(value) for value in entry["upright"]["footprint_xy"]),
                float(entry["upright"]["footprint_r"]),
            )
            for object_id, entry in objects.items()
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ContactGeometryError("pinned CAD contact geometry is malformed") from error
    if any(
        len(item.center_xy_mm) != 2
        or not all(math.isfinite(value) for value in (*item.center_xy_mm, item.radius_mm))
        or item.radius_mm <= 0.0
        for item in footprints.values()
    ):
        raise ContactGeometryError("pinned CAD contact footprints must be finite and positive")
    return footprints


def contact_envelope_radius_mm(
    endpoint_xy_mm: tuple[float, float],
    objects: tuple[tuple[str, tuple[float, float], float], ...],
) -> float:
    """Return a TCP-centred circle enclosing each rotated CAD footprint."""
    footprints = _pinned_footprints()
    extents: list[float] = []
    for object_id, origin_xy, yaw_deg in objects:
        try:
            footprint = footprints[object_id]
        except KeyError as error:
            raise ContactGeometryError(f"{object_id!r} has no pinned CAD footprint") from error
        yaw = math.radians(yaw_deg)
        local_x, local_y = footprint.center_xy_mm
        center = (
            origin_xy[0] + math.cos(yaw) * local_x - math.sin(yaw) * local_y,
            origin_xy[1] + math.sin(yaw) * local_x + math.cos(yaw) * local_y,
        )
        extents.append(math.hypot(endpoint_xy_mm[0] - center[0], endpoint_xy_mm[1] - center[1]) + footprint.radius_mm)
    if not extents:
        raise ContactGeometryError("contact envelope requires at least one physical footprint")
    return max(extents)


def contact_geometry_source() -> str:
    return f"{TRUSTED_GEOMETRY_SOURCE.snapshot} sha256={TRUSTED_GEOMETRY_SOURCE.sha256}"
