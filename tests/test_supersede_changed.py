"""Re-render bookkeeping: only scenes whose sampled spec changed are superseded; renderer notes do not count.
Authored by Shritej Gunda.
"""

import json

from sdg.scene_sampler import Inputs, sample_scene
from sdg.tools import supersede_changed as sc


def _chunk(tmp_path, specs):
    chunk = tmp_path / "synthetic" / "D" / "chunk_0000"
    (chunk / "labels").mkdir(parents=True)
    for spec in specs:
        (chunk / "labels" / f"{spec['index']:06d}.json").write_text(json.dumps({"spec": spec}))
    (tmp_path / "specs" / "D").mkdir(parents=True)
    (tmp_path / "specs" / "D" / "chunk_0000.jsonl").write_text("{}\n" * len(specs))
    (chunk / "_DONE.json").write_text("{}")
    return chunk


def test_renderer_rest_height_is_not_a_spec_change_and_a_wrong_move_is_restored(tmp_path):
    inputs = Inputs.load()
    current = [json.loads(json.dumps(sample_scene(inputs, "D", i))) for i in range(3)]
    current[0]["objects"][0]["render_dz_mm"] = 0.4          # added by sdg/isaac/stage.py at render time
    stale = json.loads(json.dumps(current[2]))
    stale["objects"][-1]["transform_uf0"][2][3] += 5.0       # rendered from an older sampler
    chunk = _chunk(tmp_path, [current[0], current[1], stale])
    assert sc.supersede(chunk, inputs) == 1
    assert not (chunk / "_DONE.json").exists() and (chunk / "superseded" / "000002.json").exists()

    (chunk / "superseded" / "000002.json").write_text(json.dumps({"spec": current[2]}))   # re-rendered, then moved
    assert sc.restore(chunk, inputs) == 1
    assert (chunk / "labels" / "000002.json").exists() and (chunk / "_DONE.json").exists()
