import json
import shutil

import numpy as np
from PIL import Image

from sdg.cosmos.composite import box_blur, composite, light_change
from sdg.cosmos.drift import edge_fscore, worst_object
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


def test_one_reshaped_object_is_caught_by_the_worst_object_score():
    rgb, seg = scene()
    reshaped = rgb.copy()
    reshaped[60:90, 90:130] = 235
    reshaped[60:90, 100:120] = 20                                        # second part drawn half as wide
    assert edge_fscore(rgb, reshaped, seg) > 0.6                         # the whole image still looks fine
    score, uid = worst_object(rgb, reshaped, seg)
    assert uid == 2 and score < 0.5


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


def test_pick_can_take_one_rendered_chunk(tmp_path):
    chunk = write_set(tmp_path)
    shutil.copytree(chunk, chunk.parent / "chunk_0001")
    picked = pick(tmp_path, "A", 0.5, 0, chunks=["chunk_0001"])
    assert len(picked) == 5 and {c for _, c, _, _ in picked} == {"chunk_0001"}
    assert len(pick(tmp_path, "A", 0.5, 0)) == 10

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
    assert import_results(tmp_path, bundle, 0.7) == {"checked": 2, "accepted": 1, "min_fscore": 0.7}
    _, _, cam, idx = parse_name(names[0])
    kept = Image.open(chunk / "rgb_cosmos" / cam / f"{idx}.jpg")
    assert kept.size == (160, 120)
    record = json.loads((chunk / "_COSMOS.json").read_text())
    assert [r["accepted"] for r in record.values()] == [True, False]


def test_composite_keeps_rendered_parts_relit_and_takes_the_rest_from_cosmos():
    rgb, seg = scene()
    cosmos = (rgb.astype(np.float32) * np.array([0.85, 0.8, 0.7])).astype(np.uint8)   # dimmer, warmer light
    cosmos[60:90, 90:130] = 235                                           # and it erased the second part
    cosmos[5:15, 5:15] = 120                                              # and added a real-looking smudge
    out, kept = composite(rgb, cosmos, seg)
    parts = seg > 0
    expected = rgb.astype(np.float32) * np.array([0.85, 0.8, 0.7])
    assert np.abs(out[parts].astype(np.float32) - expected[parts]).max() <= 1.5   # both parts right, relit
    assert (out[5:15, 5:15] == 120).all()                                 # background stays Cosmos
    assert kept == [1]                                                    # only the unchanged part kept Cosmos


def test_composite_keeps_the_cosmos_look_of_a_part_whose_shape_is_unchanged():
    rgb, seg = scene()
    cosmos = rgb.copy()
    cosmos[30:60, 30:70] = 45                                             # part 1: new material, same outline
    cosmos[60:90, 90:130] = 235                                           # part 2: erased
    out, kept = composite(rgb, cosmos, seg)
    assert kept == [1]
    assert (out[32:58, 32:68] == 45).all()                                # Cosmos look kept
    assert (out[62:88, 92:128] == 20).all()                               # erased part restored from the render


def test_light_change_is_clamped_and_box_blur_keeps_flat_images_flat():
    src = np.random.default_rng(1).integers(0, 255, (40, 40, 3)).astype(np.uint8)
    gain = light_change(src, np.clip(src.astype(np.float32) * 3, 0, 255).astype(np.uint8), np.ones((40, 40), bool))
    assert (gain <= 1.5).all()
    flat = np.full((20, 30), 7.0)
    assert np.allclose(box_blur(flat, 3), 7.0)


def test_composite_restores_a_part_cosmos_recoloured_with_its_outline_intact():
    rgb, seg = scene()
    rgb[30:60, 30:70] = (220, 60, 50)                                     # a red foreign object
    cosmos = rgb.copy()
    cosmos[30:60, 30:70] = (40, 35, 30)                                   # Cosmos painted it black, same outline
    out, kept = composite(rgb, cosmos, seg)
    assert kept == [2]
    assert (np.abs(out[32:58, 32:68].astype(int) - (220, 60, 50)) <= 1).all()
