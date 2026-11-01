"""T36 ACT / SmolVLA baselines on CPU: the LeRobot view of GR00T's data, the GR00T-protocol server, the guard path.

LeRobot itself is not a repo dependency; the one test that loads the view with LeRobot 0.3.3 is skipped without it.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from test_groot_pipeline import make_export  # same directory: pytest puts it on sys.path

from policy_learn.baselines import lerobot_view as view
from policy_learn.baselines import policy_server as srv
from policy_learn.groot import prepare_dataset as prep
from policy_learn.groot.tcp_convert import tcp_to_eef9d

TCP = [400.0, 0.0, -300.0, -180.0, 0.0, 30.0]
JOINTS = [0.0, 10.0, -20.0, 0.0, -70.0, 30.0, 1.0]


@pytest.fixture
def built(tmp_path):
    for name in ("pick_demo", "pull_demo"):
        src = make_export(tmp_path / "src" / name, n_eps=5, n_frames=20)
        if name == "pull_demo":   # a second dataset with its own instructions: tasks must merge, not collide
            (src / "meta/tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": f"pull out {i}"}) + "\n"
                                                          for i in range(2)))
        prep.prepare(src, tmp_path / "prep", val_fraction=0.2, seed=0)
    reports = view.build(tmp_path / "prep", tmp_path / "view")
    return tmp_path, reports


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_view_keeps_grootss_split_and_merges_datasets(built):
    tmp, reports = built
    assert [(r["split"], r["episodes"], r["frames"]) for r in reports] == [("train", 8, 160), ("val", 2, 40)]
    tasks = [t["task"] for t in _jsonl(tmp / "view/train/meta/tasks.jsonl")]
    assert len(tasks) == len(set(tasks)) == 4 and "pull out 0" in tasks
    eps = _jsonl(tmp / "view/train/meta/episodes.jsonl")
    assert [e["episode_index"] for e in eps] == list(range(8))
    prep_train = sum(len(_jsonl(tmp / f"prep/train/{n}/meta/episodes.jsonl")) for n in ("pick_demo", "pull_demo"))
    assert prep_train == 8


def test_view_columns_are_grootss_state_and_action_keys(built):
    tmp, _ = built
    src = pd.read_parquet(tmp / "prep/train/pick_demo" / prep.DATA_PATH.format(episode_chunk=0, episode_index=0))
    out = pd.read_parquet(tmp / "view/train" / prep.DATA_PATH.format(episode_chunk=0, episode_index=0))
    state, action = np.stack(out["observation.state"]), np.stack(out["action"])
    raw_s, raw_a = np.stack(src["observation.state"]), np.stack(src["action"])
    np.testing.assert_allclose(state[:, :9], tcp_to_eef9d(raw_s[:, 7:13]), atol=1e-5)
    np.testing.assert_array_equal(state[:, 9], raw_s[:, 6])
    np.testing.assert_array_equal(state[:, 10:], raw_s[:, :6])
    np.testing.assert_allclose(action[:, :9], tcp_to_eef9d(raw_a[:, :6]), atol=1e-5)
    np.testing.assert_array_equal(action[:, 9], raw_a[:, 6])
    task_text = {t["task_index"]: t["task"] for t in _jsonl(tmp / "view/train/meta/tasks.jsonl")}
    src_text = {t["task_index"]: t["task"] for t in _jsonl(tmp / "prep/train/pick_demo/meta/tasks.jsonl")}
    assert [task_text[t] for t in out["task_index"]] == [src_text[t] for t in src["task_index"]]
    info = json.loads((tmp / "view/train/meta/info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [16] and info["features"]["action"]["shape"] == [10]


def test_view_stats_and_videos(built):
    tmp, _ = built
    stats = _jsonl(tmp / "view/train/meta/episodes_stats.jsonl")
    assert len(stats) == 8
    s0 = stats[0]["stats"]
    assert len(s0["observation.state"]["mean"]) == 16 and s0["action"]["count"] == [20]
    assert np.array(s0["observation.images.wrist"]["mean"]).shape == (3, 1, 1)
    link = tmp / "view/train" / prep.VIDEO_PATH.format(episode_chunk=0, episode_index=0,
                                                       video_key="observation.images.table")
    assert link.is_symlink() and link.resolve().is_relative_to(tmp / "src")


def test_view_loads_in_lerobot(built):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata("local/test", root=built[0] / "view/train")
    assert meta.total_episodes == 8 and meta.stats["observation.state"]["mean"].shape == (16,)


def _groot_obs(hw=(360, 640)):
    gc = pytest.importorskip("policy_learn.groot.groot_client")
    img = np.full((*hw, 3), 255, np.uint8)
    return gc.obs_to_groot({"images": {c: img for c in gc.CAMERAS}, "joints": JOINTS, "tcp_pose": TCP,
                            "instruction": "pull out M6_01 at (338, -229, 2)"})


def test_server_observation_matches_training_layout():
    frame = srv.groot_obs_to_lerobot(_groot_obs(), (360, 640))
    np.testing.assert_allclose(frame["observation.state"][:9], tcp_to_eef9d(TCP), atol=1e-6)
    assert frame["observation.state"][9] == 1.0
    np.testing.assert_allclose(frame["observation.state"][10:], JOINTS[:6])
    img = frame["observation.images.side"]
    assert img.shape == (3, 360, 640) and img.dtype == np.float32 and img.max() == 1.0
    assert frame["task"] == "pull out M6_01 at (338, -229, 2)"
    with pytest.raises(srv.ObservationError, match="trained on"):
        srv.groot_obs_to_lerobot(_groot_obs((180, 320)), (360, 640))


class FakeChunkPolicy:
    """Stands in for LeRobotChunkPolicy: a 16-row chunk 1 mm down per row from the observed TCP, gripper held."""

    def __init__(self):
        self.resets = 0

    def reset(self, options=None):
        self.resets += 1
        return {}

    def get_action(self, observation, options=None):
        state = srv.groot_obs_to_lerobot(observation)["observation.state"]
        chunk = np.tile(np.concatenate([state[:9], state[9:10]]), (20, 1))
        chunk[:, 2] -= np.arange(1, 21)
        return [srv.chunk_to_groot_action(chunk, 16), {}]


def test_server_drives_grootpolicy_through_the_guard():
    zmq = pytest.importorskip("zmq")
    from policy_learn.deploy.cell_groot_policy import GrootPolicy
    from policy_learn.groot.groot_client import PolicyServerError

    with zmq.Context() as ctx, ctx.socket(zmq.REP) as probe:
        port = probe.bind_to_random_port("tcp://127.0.0.1")
        probe.unbind(probe.getsockopt_string(zmq.LAST_ENDPOINT))
    fake = FakeChunkPolicy()
    thread = threading.Thread(target=srv.serve, args=(fake, "127.0.0.1", port), daemon=True)
    thread.start()
    policy = GrootPolicy(f"groot://127.0.0.1:{port}")   # exactly what run_eval.py builds for --policy groot://
    policy.reset()
    img = np.zeros((360, 640, 3), np.uint8)
    rows = policy.act({"images": {c: img for c in policy.CAMERAS}, "joints": np.array(JOINTS), "tcp_pose": TCP,
                       "instruction": "pull out M6_01"})
    assert rows.shape == (16, 7) and not policy.guard_log
    np.testing.assert_allclose(rows[:, 2], TCP[2] - np.arange(1, 17), atol=1e-3)
    assert set(rows[:, 6]) == {1.0} and fake.resets == 1
    client = policy.model.client
    with pytest.raises(PolicyServerError, match="unknown endpoint"):
        client.call("nope")
    assert client.call("kill") == {"status": "ok"}
    thread.join(5)
    assert not thread.is_alive()


def test_bad_chunk_is_refused():
    with pytest.raises(ValueError, match="bad chunk"):
        srv.chunk_to_groot_action(np.full((16, 10), np.nan), 16)
    with pytest.raises(ValueError, match="bad chunk"):
        srv.chunk_to_groot_action(np.zeros((16, 7)), 16)
