"""Stop box colliders come out of the base STL exactly (stop_colliders.py)."""

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import stop_colliders as S

STL = HERE.parents[2] / "cad/stl/F04_Sensor_Connector_Fixture.stl"


def _read(path: Path) -> np.ndarray:
    b = path.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return np.frombuffer(b[84:84 + n * 50], dtype=rec)["v"].astype(float)


def test_f04_stops_become_six_exact_boxes_and_pegs_stay_sdf():
    boxes = sorted(S.sharp_boxes(_read(STL)))
    # two 3 x 6 mm side stops (objects.yaml: centre x +-21, y 10) and two L-shaped front stops, two boxes each;
    # the chamfered round pegs (peg_A at -12, -10; peg_B at 11, 9) are not boxed
    assert len(boxes) == 6
    assert (-22.5, -19.5, 7.0, 13.0, -0.5, 3.0) in boxes and (19.5, 22.5, 7.0, 13.0, -0.5, 3.0) in boxes
    assert (14.5, 22.5, -18.5, -15.5, -0.5, 3.0) in boxes and (19.5, 22.5, -15.5, -10.5, -0.5, 3.0) in boxes
    for x0, x1, y0, y1, _, _ in boxes:
        for px, py in ((-12.0, -10.0), (11.0, 9.0)):
            assert not (x0 <= px <= x1 and y0 <= py <= y1)


def test_largest_rectangle_splits_an_l():
    m = np.zeros((6, 6), bool)
    m[0:2, :] = True
    m[:, 0:2] = True
    area, r0, r1, c0, c1 = S._largest_rectangle(m)
    assert area == 12 and (r1 - r0 + 1) * (c1 - c0 + 1) == 12
