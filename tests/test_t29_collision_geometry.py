"""Held-part collision geometry comes only from pinned T29 frames and CAD STLs."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from assembly.motion import Waypoint
from assembly.safety import MotionOwner


def _binary_stl(path: Path, vertices: tuple[tuple[float, float, float], ...]) -> None:
    triangle = struct.pack("<12fH", 0.0, 0.0, 1.0, *vertices[0], *vertices[1], *vertices[2], 0)
    path.write_bytes(b"source-cad".ljust(80, b"\0") + struct.pack("<I", 1) + triangle)


def _frames(path: Path, *, transform: list[list[float]] | None = None) -> None:
    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    raw = {
        "source": "reviewed T29 printed CAD",
        "units": "mm, deg",
        "objects": {
            "M1_01": {
                "T_local_from_stl": transform or identity,
                "grips": [{"tcp_local_mm": [1.0, 1.0, 1.0], "roll_offset_deg": 90.0}],
            }
        },
    }
    path.write_text(json.dumps(raw))


def _point(x: float, y: float, z: float) -> Waypoint:
    return Waypoint(x, y, z, -180.0, 0.0, 0.0, 1.0)


def test_t29_held_part_sweep_is_source_derived_and_conservative(tmp_path: Path) -> None:
    from assembly.t29_collision_geometry import HeldPartSTLSweepSource

    frames = tmp_path / "grasp_frames.json"
    mesh = tmp_path / "M1_01.stl"
    _frames(frames)
    _binary_stl(mesh, ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)))
    source = HeldPartSTLSweepSource(frames, {"M1_01": mesh})

    bounds = source.swept_bounds(
        _point(10.0, 20.0, 30.0),
        _point(12.0, 25.0, 30.0),
        MotionOwner("part-1", "M1_01", True),
    )

    radius = math.sqrt(3.0)
    assert bounds.name == "held:M1_01"
    assert bounds.minimum == pytest.approx((10.0 - radius, 20.0 - radius, 30.0 - radius))
    assert bounds.maximum == pytest.approx((12.0 + radius, 25.0 + radius, 30.0 + radius))
    assert "sha256=" in source.source
    assert str(frames.resolve()) in source.source


def test_t29_transform_is_applied_before_the_tcp_radius(tmp_path: Path) -> None:
    from assembly.t29_collision_geometry import HeldPartSTLSweepSource

    frames = tmp_path / "grasp_frames.json"
    mesh = tmp_path / "M1_01.stl"
    translated = [[1.0, 0.0, 0.0, 10.0], [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    _frames(frames, transform=translated)
    _binary_stl(mesh, ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)))

    source = HeldPartSTLSweepSource(frames, {"M1_01": mesh})
    bounds = source.swept_bounds(
        _point(10.0, 20.0, 30.0), _point(10.0, 20.0, 30.0), MotionOwner("part-1", "M1_01", True)
    )

    assert bounds.minimum[0] < 0.5
    assert bounds.maximum[0] > 19.5


def test_t29_sweep_rejects_unknown_or_detached_parts(tmp_path: Path) -> None:
    from assembly.cell_runtime_safety import ProductionSafetyUnavailable
    from assembly.t29_collision_geometry import HeldPartSTLSweepSource

    frames = tmp_path / "grasp_frames.json"
    mesh = tmp_path / "M1_01.stl"
    _frames(frames)
    _binary_stl(mesh, ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)))
    source = HeldPartSTLSweepSource(frames, {"M1_01": mesh})

    with pytest.raises(ProductionSafetyUnavailable, match="unknown held part"):
        source.swept_bounds(_point(0.0, 0.0, 0.0), _point(1.0, 0.0, 0.0), MotionOwner("x", "M2_01", True))
    with pytest.raises(ProductionSafetyUnavailable, match="attached"):
        source.swept_bounds(_point(0.0, 0.0, 0.0), _point(1.0, 0.0, 0.0), MotionOwner("x", "M1_01", False))


def test_t29_sweep_rejects_malformed_frames(tmp_path: Path) -> None:
    from assembly.cell_runtime_safety import ProductionSafetyConfigurationError
    from assembly.t29_collision_geometry import HeldPartSTLSweepSource

    frames = tmp_path / "grasp_frames.json"
    mesh = tmp_path / "M1_01.stl"
    frames.write_text(json.dumps({"source": "", "units": "mm, deg", "objects": {}}))
    _binary_stl(mesh, ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)))

    with pytest.raises(ProductionSafetyConfigurationError, match="source"):
        HeldPartSTLSweepSource(frames, {"M1_01": mesh})


def test_t29_sweep_wraps_invalid_ascii_stl_as_configuration_error(tmp_path: Path) -> None:
    from assembly.cell_runtime_safety import ProductionSafetyConfigurationError
    from assembly.t29_collision_geometry import HeldPartSTLSweepSource

    frames = tmp_path / "grasp_frames.json"
    mesh = tmp_path / "M1_01.stl"
    _frames(frames)
    mesh.write_text("solid bad\nfacet normal 0 0 1\nouter loop\nvertex x 0 0\nendsolid bad\n")

    with pytest.raises(ProductionSafetyConfigurationError, match="STL vertex"):
        HeldPartSTLSweepSource(frames, {"M1_01": mesh})
