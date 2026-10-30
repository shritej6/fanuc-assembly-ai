"""Boundary regressions for immutable catalog source provenance."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import assembly.catalog as catalog_module
from assembly.catalog import CatalogResourceError, CatalogSemanticError, load_catalog

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "assembly" / "catalog.yaml"
GRASP_SOURCE_PATH = ROOT / "assembly" / "sources" / "grasp_frames.fac0590.json"
GRASP_SOURCE_SHA256 = "dbd00b5082042e8a326b380d237890449bfb2206243c1753b7ebe2d940435977"


def _catalog_data() -> dict:
    return yaml.safe_load(CATALOG_PATH.read_text())


def _write_catalog(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _write_json_source(path: Path, data: dict) -> str:
    contents = json.dumps(data, indent=1)
    path.write_text(contents)
    return hashlib.sha256(contents.encode()).hexdigest()


@pytest.mark.parametrize("source_key", ["geometry_source", "grasp_source"])
@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("file", "untrusted/source.json"),
        ("generator", "untrusted/generator.py"),
        ("commit", "0" * 40),
        ("commit_timestamp", "2026-10-29T00:00:00-05:00"),
        ("snapshot", "assembly/sources/untrusted.json"),
        ("sha256", "0" * 64),
    ],
)
def test_load_catalog_rejects_mutated_source_provenance(
    tmp_path: Path,
    source_key: str,
    field: str,
    replacement: str,
) -> None:
    invalid = deepcopy(_catalog_data())
    invalid[source_key][field] = replacement
    path = _write_catalog(tmp_path / "mutated-provenance.yaml", invalid)

    with pytest.raises(CatalogSemanticError, match=f"{source_key}.*trusted provenance"):
        load_catalog(path)


def test_load_catalog_rejects_zero_grips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contents = GRASP_SOURCE_PATH.read_bytes()
    assert hashlib.sha256(contents).hexdigest() == GRASP_SOURCE_SHA256
    source = json.loads(contents)
    source["objects"]["M1_01"]["grips"] = []
    source_path = tmp_path / "zero-grip-source.json"
    source_sha256 = _write_json_source(source_path, source)
    catalog_data = _catalog_data()
    catalog_data["grasp_source"]["sha256"] = source_sha256
    catalog_path = _write_catalog(tmp_path / "zero-grip-catalog.yaml", catalog_data)
    trusted_source = catalog_module.GraspSource(**catalog_data["grasp_source"])
    monkeypatch.setattr(catalog_module, "TRUSTED_GRASP_SOURCE", trusted_source, raising=False)
    monkeypatch.setattr(catalog_module, "GRASP_SOURCE_SNAPSHOT_PATH", source_path)

    with pytest.raises(CatalogSemanticError, match="M1_01.*exactly one grip"):
        load_catalog(catalog_path)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"-90": [0.0, -22.86, 8.4]}, "validated turn 90"),
        (
            {"90": [0.0, -19.26, 8.4], "-90": [0.0, -22.86, 8.4], "90.0": [0.0, -19.26, 8.4]},
            "canonical turn keys",
        ),
        ({"90": [0.0, float("nan"), 8.4], "-90": [0.0, -22.86, 8.4]}, "finite values"),
        ({"90": [0.0, "1.8", 8.4], "-90": [0.0, -22.86, 8.4]}, "numeric values"),
        ({"90": [0.0, True, 8.4], "-90": [0.0, -22.86, 8.4]}, "numeric values"),
    ],
)
def test_load_catalog_rejects_unusable_preferred_turn_tcp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: object,
    message: str,
) -> None:
    source = json.loads(GRASP_SOURCE_PATH.read_text())
    source["objects"]["M1_01"]["grips"][0]["tcp_local_mm"] = replacement
    source_path = tmp_path / "bad-turn-source.json"
    source_sha256 = _write_json_source(source_path, source)
    catalog_data = _catalog_data()
    catalog_data["grasp_source"]["sha256"] = source_sha256
    catalog_path = _write_catalog(tmp_path / "bad-turn-catalog.yaml", catalog_data)
    trusted_source = catalog_module.GraspSource(**catalog_data["grasp_source"])
    monkeypatch.setattr(catalog_module, "TRUSTED_GRASP_SOURCE", trusted_source)
    monkeypatch.setattr(catalog_module, "GRASP_SOURCE_SNAPSHOT_PATH", source_path)

    with pytest.raises(CatalogResourceError, match=message):
        load_catalog(catalog_path)
