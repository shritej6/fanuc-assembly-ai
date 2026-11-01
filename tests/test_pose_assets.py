"""Tests for pose assets.
Authored by Shritej Gunda.
"""

import json
from pathlib import Path

import pytest

from perception.pose.assets import PoseAssetError, load_pose_assets

ROOT = Path(__file__).resolve().parents[1]


def test_pose_assets_cover_taxonomy_with_explicit_metric_scale() -> None:
    taxonomy = json.loads((ROOT / "contracts/taxonomy.json").read_text())
    expected_ids = [item["object_id"] for item in taxonomy["objects"]]

    assets = load_pose_assets(ROOT)

    assert list(assets) == expected_ids
    assert len(assets) == 30
    assert all(asset.mesh_path.is_file() for asset in assets.values())
    assert {asset.mesh_scale_to_m for asset in assets.values()} == {0.001}


def test_pose_assets_reject_geometry_taxonomy_mismatch(tmp_path: Path) -> None:
    (tmp_path / "contracts").mkdir()
    (tmp_path / "sdg/data").mkdir(parents=True)
    (tmp_path / "contracts/taxonomy.json").write_text(
        json.dumps({"objects": [{"object_id": "M1_01"}]})
    )
    (tmp_path / "sdg/data/object_geometry.json").write_text(
        json.dumps({"units": "mm", "objects": {}})
    )

    with pytest.raises(PoseAssetError, match="object IDs"):
        load_pose_assets(tmp_path)


def test_pose_assets_reject_mesh_path_outside_repository(tmp_path: Path) -> None:
    (tmp_path / "contracts").mkdir()
    (tmp_path / "sdg/data").mkdir(parents=True)
    (tmp_path / "contracts/taxonomy.json").write_text(
        json.dumps({"objects": [{"object_id": "M1_01"}]})
    )
    (tmp_path / "sdg/data/object_geometry.json").write_text(
        json.dumps(
            {
                "units": "mm",
                "objects": {"M1_01": {"stl": "../outside.stl"}},
            }
        )
    )

    with pytest.raises(PoseAssetError, match="inside the repository"):
        load_pose_assets(tmp_path)


def test_pose_assets_wrap_malformed_geometry_entry(tmp_path: Path) -> None:
    (tmp_path / "contracts").mkdir()
    (tmp_path / "sdg/data").mkdir(parents=True)
    (tmp_path / "contracts/taxonomy.json").write_text(
        json.dumps({"objects": [{"object_id": "M1_01"}]})
    )
    (tmp_path / "sdg/data/object_geometry.json").write_text(
        json.dumps({"units": "mm", "objects": {"M1_01": []}})
    )

    with pytest.raises(PoseAssetError, match="geometry entry"):
        load_pose_assets(tmp_path)
