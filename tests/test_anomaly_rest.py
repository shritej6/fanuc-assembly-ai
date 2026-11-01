import hashlib
import json
from pathlib import Path

import pytest

from perception.anomaly.rest_provenance import (
    RestProvenanceError,
    derive_rest_sidecars,
    verify_rest_sidecars,
)

TOOL = """
RULE = "installed = same up axis as inside its product (yaw free); anything else = fallen"
MAX_TILT_DEG = 2.0

def product_up_axes(geometry):
    return {"M1_01": [0.0, 0.0, 1.0]}

def rest_states(spec, up_axes):
    return {obj["uid"]: obj["expected_rest"] for obj in spec["objects"]}
"""


def _sources(root: Path) -> None:
    for set_name, rest in (("C", "installed"), ("I", "fallen")):
        labels = root / set_name / "chunk_0000/labels"
        labels.mkdir(parents=True)
        payload = {
            "spec": {
                "objects": [
                    {
                        "uid": "o01",
                        "object_id": "M1_01",
                        "expected_rest": rest,
                    }
                ]
            }
        }
        (labels / "000000.json").write_text(json.dumps(payload))


def test_derivation_uses_hash_pinned_tool_without_mutating_sources(tmp_path: Path) -> None:
    tool = tmp_path / "rest_labels.py"
    geometry = tmp_path / "geometry.json"
    tool.write_text(TOOL)
    geometry.write_text('{"objects": {}}')
    old_root, rest_root = tmp_path / "old", tmp_path / "derived"
    _sources(old_root)

    summary = derive_rest_sidecars(
        old_root,
        rest_root,
        tool_path=tool,
        geometry_path=geometry,
        expected_tool_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
        expected_geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),
    )

    assert not list(old_root.rglob("_REST.json"))
    assert json.loads((rest_root / "C/chunk_0000/_REST.json").read_text())["images"] == {"000000": {"o01": "installed"}}
    assert summary.counts == {"fallen": 1, "installed": 1}
    assert len(summary.sidecar_sha256) == 2


def test_derivation_rejects_unreviewed_tool_before_writing(tmp_path: Path) -> None:
    tool = tmp_path / "rest_labels.py"
    geometry = tmp_path / "geometry.json"
    tool.write_text(TOOL)
    geometry.write_text('{"objects": {}}')

    with pytest.raises(RestProvenanceError, match="tool SHA-256"):
        derive_rest_sidecars(
            tmp_path / "old",
            tmp_path / "derived",
            tool_path=tool,
            geometry_path=geometry,
            expected_tool_sha256="0" * 64,
            expected_geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),
        )

    assert not (tmp_path / "derived").exists()


def test_derivation_refuses_to_write_inside_source_renders(tmp_path: Path) -> None:
    tool = tmp_path / "rest_labels.py"
    geometry = tmp_path / "geometry.json"
    tool.write_text(TOOL)
    geometry.write_text('{"objects": {}}')
    old_root = tmp_path / "old"
    _sources(old_root)

    with pytest.raises(RestProvenanceError, match="outside the immutable source"):
        derive_rest_sidecars(
            old_root,
            old_root / "derived",
            tool_path=tool,
            geometry_path=geometry,
            expected_tool_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
            expected_geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),
        )

    assert not (old_root / "derived").exists()


def test_final_verification_rejects_sidecar_drift(tmp_path: Path) -> None:
    tool = tmp_path / "rest_labels.py"
    geometry = tmp_path / "geometry.json"
    tool.write_text(TOOL)
    geometry.write_text('{"objects": {}}')
    old_root, rest_root = tmp_path / "old", tmp_path / "derived"
    _sources(old_root)
    summary = derive_rest_sidecars(
        old_root,
        rest_root,
        tool_path=tool,
        geometry_path=geometry,
        expected_tool_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
        expected_geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),
    )
    (rest_root / "C/chunk_0000/_REST.json").write_text("changed")

    with pytest.raises(RestProvenanceError, match="changed after derivation"):
        verify_rest_sidecars(rest_root, summary)
