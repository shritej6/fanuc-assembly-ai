import json

import numpy as np
from PIL import Image

from sdg.cosmos.drift import edge_fscore
from sdg.cosmos.restyle import export, import_results, parse_name, pick


def scene(shift: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A white table with two black parts; seg marks the parts at their unshifted place."""
    rgb = np.full((120, 160, 3), 235, np.uint8)
    seg = np.zeros((120, 160), np.uint16)
    for uid, (y, x) in enumerate([(30, 30), (60, 90)]):
        rgb[y + shift:y + shift + 30, x + shift:x + shift + 40] = 20
        seg[y:y + 30, x:x + 40] = uid + 1
    return rgb, seg


def test_restyle_that_keeps_edges_passes_and_moved_parts_fail():
    rgb, seg = scene()
    relit = (rgb.astype(np.float32) * 0.7 + 30).astype(np.uint8)          # new light, same shapes
    moved, _ = scene(shift=8)                                            # parts drawn 8 px away
    assert edge_fscore(rgb, relit, seg) > 0.9
    assert edge_fscore(rgb, moved, seg) < 0.5


def write_set(root, set_name="A", n=10):
    chunk = root / "synthetic" / set_name / "chunk_0000"
    for sub in ("labels", "rgb/table", "rgb/side", "seg/table", "seg/side"):
        (chunk / sub).mkdir(parents=True, exist_ok=True)
    rgb, seg = scene()
    for i in range(n):
        idx = f"{i:06d}"
        (chunk / "labels" / f"{idx}.json").write_text(json.dumps({"cameras": {"table": [], "side": []}}))
        for cam in ("table", "side"):
            Image.fromarray(rgb).save(chunk / "rgb" / cam / f"{idx}.jpg", quality=98)
            Image.fromarray(seg).save(chunk / "seg" / cam / f"{idx}.png")
    return chunk


def test_pick_is_deterministic_and_uses_each_camera(tmp_path):
    write_set(tmp_path)
    a, b = pick(tmp_path, "A", 0.6, 0), pick(tmp_path, "A", 0.6, 0)
    assert a == b and len(a) == 6
    assert {cam for _, _, cam, _ in a} == {"table", "side"}
    assert len(pick(tmp_path, "A", 4, 0)) == 4 and len(pick(tmp_path, "A", 50, 0)) == 10


def test_import_keeps_only_images_that_pass_the_drift_check(tmp_path):
    chunk = write_set(tmp_path, n=4)
    bundle = tmp_path / "bundle"
    assert export(tmp_path, ["A"], 0.5, 0, bundle) == 2
    names = sorted(p.stem for p in (bundle / "specs").glob("*.json"))
    (bundle / "out").mkdir()
    rgb, _ = scene()
    moved, _ = scene(shift=8)
    for name, img in ((names[0], rgb), (names[1], moved)):              # Cosmos output size differs from the render
        Image.fromarray(img).resize((320, 240)).save(bundle / "out" / f"{name}.jpg")
    assert import_results(tmp_path, bundle, min_fscore=0.7) == {"checked": 2, "accepted": 1, "min_fscore": 0.7}
    _, _, cam, idx = parse_name(names[0])
    kept = Image.open(chunk / "rgb_cosmos" / cam / f"{idx}.jpg")
    assert kept.size == (160, 120)
    record = json.loads((chunk / "_COSMOS.json").read_text())
    assert [r["accepted"] for r in record.values()] == [True, False]
