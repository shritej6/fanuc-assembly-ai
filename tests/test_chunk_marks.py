"""Behavior tests for chunk completion: only a valid, complete chunk rendered from the same specs counts as done."""

from __future__ import annotations

import json
from pathlib import Path

from sdg import chunk_marks as cm

DEPTH = {"table", "wrist"}


def _spec(index: int) -> dict:
    return {"index": index, "cameras": ["table", "side"], "outputs": ["rgb", "depth"], "seed": index}


def _render(out: Path, spec: dict, skip: str | None = None) -> None:
    for f in cm.scene_files(spec, DEPTH):
        if f != skip:
            (out / f).parent.mkdir(parents=True, exist_ok=True)
            (out / f).write_bytes(b"x")
    cm.write_json_atomic(out / "labels" / f"{spec['index']:06d}.json", {"spec": spec, "cameras": {"table": [], "side": []}})


def _chunk(tmp_path: Path, n: int = 3) -> tuple[Path, Path, list[dict]]:
    out, specs = tmp_path / "chunk_0000", tmp_path / "chunk_0000.jsonl"
    rows = [_spec(i) for i in range(n)]
    specs.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        _render(out, r)
    cm.write_json_atomic(out / cm.DONE, {"scenes": n, "specs_sha256": cm.specs_digest(specs), "depth_cams": sorted(DEPTH)})
    return out, specs, rows


def test_a_complete_chunk_from_the_same_specs_is_done(tmp_path: Path) -> None:
    out, specs, _ = _chunk(tmp_path)

    assert cm.check(out, specs) == []


def test_depth_is_expected_only_for_depth_cameras(tmp_path: Path) -> None:
    assert cm.scene_files(_spec(7), DEPTH) == [
        "rgb/table/000007.jpg", "seg/table/000007.png", "depth/table/000007.png",
        "rgb/side/000007.jpg", "seg/side/000007.png"]


def test_a_truncated_done_marker_does_not_count(tmp_path: Path) -> None:
    out, specs, _ = _chunk(tmp_path)
    (out / cm.DONE).write_text('{"scenes": 3, "specs_sha')

    assert cm.chunk_done(out, specs) is None
    assert cm.check(out, specs)


def test_a_chunk_rendered_from_other_specs_is_not_reused(tmp_path: Path) -> None:
    out, specs, rows = _chunk(tmp_path)
    rows[1]["seed"] = 99
    specs.write_text("".join(json.dumps(r) + "\n" for r in rows))

    assert cm.chunk_done(out, specs) is None
    assert not cm.scene_complete(out, rows[1], DEPTH)   # its label was made from the old spec
    assert cm.scene_complete(out, rows[0], DEPTH)


def test_a_scene_with_a_label_but_a_missing_or_empty_image_is_rendered_again(tmp_path: Path) -> None:
    out = tmp_path / "chunk"
    _render(out, _spec(0), skip="seg/side/000000.png")
    _render(out, _spec(1))
    (out / "depth/table/000001.png").write_bytes(b"")

    assert not cm.scene_complete(out, _spec(0), DEPTH)
    assert not cm.scene_complete(out, _spec(1), DEPTH)


def test_a_truncated_label_is_rendered_again(tmp_path: Path) -> None:
    out = tmp_path / "chunk"
    _render(out, _spec(0))
    (out / "labels" / "000000.json").write_text('{"spec": {"ind')

    assert not cm.scene_complete(out, _spec(0), DEPTH)


def test_check_names_the_incomplete_scenes(tmp_path: Path) -> None:
    out, specs, _ = _chunk(tmp_path, n=4)
    (out / "rgb/side/000002.jpg").unlink()

    assert cm.check(out, specs) == ["1 scenes incomplete, first [2]"]


def test_an_atomic_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    cm.write_json_atomic(tmp_path / "a.json", {"ok": 1})

    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]
