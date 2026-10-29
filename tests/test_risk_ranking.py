"""Behavior tests for the CAD risk ranking: complete input or nothing written, rank direction, real print layers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import trimesh

from cad import risk_ranking as rr

IDS = [obj["object_id"] for obj in json.loads(rr.TAXONOMY.read_text())["objects"]]


def _stl_set(root: Path, ids: list[str] = IDS) -> Path:
    """One watertight box per taxonomy ID, from 3 mm tall (interior layers beyond the skins) and thicker by index."""
    for i, oid in enumerate(ids):
        folder = root / f"Product_{oid[1]}"
        folder.mkdir(parents=True, exist_ok=True)
        trimesh.creation.box(extents=(20, 20, 3 + i)).export(folder / f"{oid}_Part.stl")
    return root


def test_ranks_every_taxonomy_part_and_puts_the_thinnest_first(tmp_path: Path) -> None:
    out = rr.ranking(_stl_set(tmp_path / "stl"))

    assert sorted(row["object_id"] for row in out["parts"]) == sorted(IDS)
    assert out["parts"][0]["object_id"] == IDS[0]   # 3 mm tall: thinnest wall
    assert all(0 <= row["risk_score"] <= 1 for row in out["parts"])
    assert out["status"].startswith("PRELIMINARY")


def test_equal_parts_share_a_rank(tmp_path: Path) -> None:
    rows = [{"object_id": str(i), **dict.fromkeys(rr.FACTORS, 1.0)} for i in range(3)]

    rr.rank_scores(rows)

    assert {row["risk_score"] for row in rows} == {0.5}


def test_a_missing_part_writes_nothing(tmp_path: Path) -> None:
    stl = _stl_set(tmp_path / "stl")
    next(stl.glob("*/M2_03_*.stl")).unlink()
    out = tmp_path / "ranking.json"
    out.write_text("previous evidence\n")

    with pytest.raises(rr.RiskInputError, match=r"missing \['M2_03'\]"):
        rr.main(["--stl-dir", str(stl), "--out", str(out)])

    assert out.read_text() == "previous evidence\n"


def test_an_empty_folder_is_refused(tmp_path: Path) -> None:
    (tmp_path / "stl").mkdir()

    with pytest.raises(rr.RiskInputError, match="missing"):
        rr.ranking(tmp_path / "stl")


def test_extra_and_duplicate_stls_are_refused(tmp_path: Path) -> None:
    stl = _stl_set(tmp_path / "stl")
    shutil.copy(next(stl.glob("*/M1_01_*.stl")), stl / "Product_1" / "M1_01_Copy.stl")
    trimesh.creation.box(extents=(5, 5, 5)).export(stl / "Product_1" / "M9_01_Unknown.stl")

    with pytest.raises(rr.RiskInputError, match=r"not in taxonomy \['M9_01'\], more than one STL \['M1_01'\]"):
        rr.ranking(stl)


def test_an_open_mesh_is_refused(tmp_path: Path) -> None:
    stl = _stl_set(tmp_path / "stl")
    path = next(stl.glob("*/F01_*.stl"))
    mesh = trimesh.load_mesh(path)
    trimesh.Trimesh(mesh.vertices, mesh.faces[1:]).export(path)   # one triangle missing: a hole

    with pytest.raises(rr.RiskInputError, match="F01_Part.stl is not watertight"):
        rr.ranking(stl)


def test_sections_are_cut_at_the_printed_layer_mid_heights(monkeypatch: pytest.MonkeyPatch) -> None:
    part = trimesh.creation.box(extents=(10, 10, 1.0))
    part.apply_translation((0, 0, 3.5))   # z 3.0 to 4.0: five 0.20 mm layers
    cuts = []
    section = part.section
    monkeypatch.setattr(part, "section", lambda plane_origin, plane_normal: cuts.append(plane_origin[2]) or section(
        plane_origin=plane_origin, plane_normal=plane_normal))

    areas = rr.layer_areas(part)

    assert cuts == pytest.approx([3.1, 3.3, 3.5, 3.7, 3.9])
    assert areas == pytest.approx([100.0] * 5)


def test_the_weakest_layer_ignores_the_solid_top_and_bottom_skins() -> None:
    areas = [10.0] + [90.0] * 4 + [70.0, 40.0, 80.0] + [90.0] * 4 + [5.0]   # chamfered first and last layers

    assert rr.weakest_layer(areas) == 40.0


def test_a_part_with_no_interior_layer_is_refused(tmp_path: Path) -> None:
    stl = _stl_set(tmp_path / "stl")
    path = next(stl.glob("*/F01_*.stl"))
    trimesh.creation.box(extents=(20, 20, 2)).export(path)   # 10 layers, all solid skin

    with pytest.raises(rr.RiskInputError, match="F01_Part.stl: impossible weakest_layer_mm2"):
        rr.ranking(stl)
