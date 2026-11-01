"""Tests for rest labels.
Authored by Shritej Gunda.
"""

import json

import pytest

from sdg.scene_sampler import Inputs, sample_scene
from sdg.tools.rest_labels import label_chunk, product_up_axes, rest_states


@pytest.fixture(scope="module")
def inp() -> Inputs:
    return Inputs.load()


@pytest.fixture(scope="module")
def up_axes(inp) -> dict:
    return product_up_axes(inp.geometry)


def test_reproduces_the_rest_label_the_test_bed_sets_were_rendered_with(inp, up_axes):
    checked = 0
    for set_name in ("N", "Q"):
        for i in range(60):
            spec = sample_scene(inp, set_name, i)
            truth = {o["uid"]: o["rest"] for o in spec["objects"] if "rest" in o}
            got = rest_states(spec, up_axes)
            assert {u: got[u] for u in truth} == truth
            checked += len(truth)
    assert checked > 500


def test_old_set_scenes_get_both_states_and_skip_what_has_no_rest_state(inp, up_axes):
    states = {}
    for i in range(40):
        spec = sample_scene(inp, "A", i)
        got = rest_states(spec, up_axes)
        kinds = {o["uid"]: o for o in spec["objects"]}
        assert all(kinds[u]["kind"] in ("part", "trap", "spare") and kinds[u]["state"] != "installed" for u in got)
        states.update(got)
    assert set(states.values()) == {"installed", "fallen"}


def test_chunk_sidecar_lists_every_image_and_leaves_labels_untouched(inp, up_axes, tmp_path):
    labels = tmp_path / "chunk_0000" / "labels"
    labels.mkdir(parents=True)
    for i in range(3):
        (labels / f"{i:06d}.json").write_text(json.dumps({"spec": sample_scene(inp, "B", i)}))
    before = {f.name: f.read_text() for f in labels.iterdir()}
    counts = label_chunk(tmp_path / "chunk_0000", up_axes)
    sidecar = json.loads((tmp_path / "chunk_0000" / "_REST.json").read_text())
    assert sorted(sidecar["images"]) == ["000000", "000001", "000002"]
    assert sum(counts.values()) == sum(len(v) for v in sidecar["images"].values()) > 0
    assert {f.name: f.read_text() for f in labels.iterdir()} == before
