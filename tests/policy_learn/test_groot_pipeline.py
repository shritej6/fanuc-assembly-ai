"""GR00T pipeline on CPU: pose conversion, dataset prepare/checks, client wire format.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from policy_learn.groot import prepare_dataset as prep
from policy_learn.groot.tcp_convert import eef9d_to_tcp, matrix_to_wpr_pair, tcp_to_eef9d, wpr_to_matrix

GROOT_SRC = os.environ.get("GROOT_SRC")
needs_groot = pytest.mark.skipif(not GROOT_SRC, reason="set GROOT_SRC to an Isaac-GR00T checkout")
RNG = np.random.default_rng(0)


def _random_tcp(n: int) -> np.ndarray:
    xyz = RNG.uniform([150, -450, -334], [650, 450, 300], size=(n, 3))
    wpr = np.column_stack([RNG.uniform(-180, 180, n), RNG.uniform(-85, 85, n), RNG.uniform(-180, 180, n)])
    return np.column_stack([xyz, wpr])


# ---------- pose conversion ----------


def test_matrix_matches_deploy_guard_convention():
    guards = pytest.importorskip("policy_learn.deploy.guards")
    for w, p, r in _random_tcp(20)[:, 3:]:
        np.testing.assert_allclose(wpr_to_matrix([w, p, r]), guards.wpr_to_matrix(w, p, r), atol=1e-12)


def test_tcp_eef9d_round_trip_is_the_same_rotation():
    tcp = _random_tcp(200)
    back = eef9d_to_tcp(tcp_to_eef9d(tcp), tcp[0, 3:])
    np.testing.assert_allclose(back[:, :3], tcp[:, :3], atol=1e-3)
    np.testing.assert_allclose(wpr_to_matrix(back[:, 3:]), wpr_to_matrix(tcp[:, 3:]), atol=1e-5)


def test_fingers_down_w_plus_minus_180_give_one_eef9d():
    a = tcp_to_eef9d([400, 0, -300, -180, 0, 30])
    b = tcp_to_eef9d([400, 0, -300, 180, 0, 30])
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_chunk_stays_continuous_near_the_reference():
    # r sweeps through 180 and w sits at the +-180 seam: the returned angles must not jump by 360.
    r = np.linspace(170, 190, 21)
    tcp = np.column_stack([np.full(21, 400), np.zeros(21), np.full(21, -300), np.full(21, -180), np.zeros(21), r])
    out = eef9d_to_tcp(tcp_to_eef9d(tcp), reference_wpr=[-180, 0, 170])
    assert np.all(np.abs(np.diff(out[:, 3:], axis=0)) < 1.5)
    np.testing.assert_allclose(out[:, 5], r, atol=1e-3)
    np.testing.assert_allclose(out[:, 3], -180, atol=1e-3)


def test_both_euler_solutions_are_the_same_rotation():
    m = wpr_to_matrix(_random_tcp(50)[:, 3:])
    first, second = matrix_to_wpr_pair(m)
    np.testing.assert_allclose(wpr_to_matrix(first), m, atol=1e-9)
    np.testing.assert_allclose(wpr_to_matrix(second), m, atol=1e-9)


# ---------- fake LeRobot export in DATA_FORMAT.md ----------


def _episode(ep: int, n: int, start: int) -> pd.DataFrame:
    t = np.arange(n)
    tcp = np.column_stack([400 + t, -50 + 0.5 * t, -300 + 0.2 * t, np.full(n, -180.0), np.zeros(n), 30 + t * 2.0])
    grip = (t >= n // 2).astype(np.float64)
    joints = np.column_stack([RNG.normal(0, 20, (n, 6)), grip])
    action = np.column_stack([np.vstack([tcp[1:], tcp[-1:]]), np.append(grip[1:], grip[-1])])
    return pd.DataFrame(
        {
            "observation.state": list(np.column_stack([joints, tcp]).astype(np.float32)),
            "action": list(action.astype(np.float32)),
            "timestamp": (t / prep.FPS).astype(np.float32),
            "frame_index": t,
            "episode_index": np.full(n, ep),
            "index": start + t,
            "task_index": np.full(n, ep % 2),
        }
    )


def make_export(root: Path, n_eps: int = 6, n_frames: int = 30) -> Path:
    (root / "meta").mkdir(parents=True)
    video_feat = {"dtype": "video", "shape": [360, 640, 3], "names": ["height", "width", "channels"]}
    info = {
        "codebase_version": "v2.1",
        "robot_type": "fanuc_er4ia",
        "fps": prep.FPS,
        "chunks_size": 1000,
        "data_path": prep.DATA_PATH,
        "video_path": prep.VIDEO_PATH,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [13], "names": None},
            "action": {"dtype": "float32", "shape": [7], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            **{f"observation.images.{c}": video_feat for c in prep.CAMERAS},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    tasks = ["pick M4_02 at (412.0, -87.5, yaw 33) gripping X", "insert M4_02 on F04 at (300.0, 120.0, yaw 90)"]
    (root / "meta/tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": s}) + "\n" for i, s in enumerate(tasks))
    )
    eps = []
    for ep in range(n_eps):
        df = _episode(ep, n_frames, ep * n_frames)
        path = root / prep.DATA_PATH.format(episode_chunk=0, episode_index=ep)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        for cam in prep.CAMERAS:
            vid = root / prep.VIDEO_PATH.format(
                episode_chunk=0, episode_index=ep, video_key=f"observation.images.{cam}"
            )
            vid.parent.mkdir(parents=True, exist_ok=True)
            vid.write_bytes(b"not decoded by the prepare step")
        eps.append({"episode_index": ep, "tasks": [tasks[ep % 2]], "length": n_frames})
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    return root


@pytest.fixture
def prepared(tmp_path):
    src = make_export(tmp_path / "src" / "pick_demo")
    report = prep.prepare(src, tmp_path / "out", val_fraction=0.2, seed=1)
    return src, tmp_path / "out", report


def test_prepare_writes_train_and_val_splits(prepared):
    src, out, report = prepared
    assert (report["train_episodes"], report["val_episodes"]) == (5, 1)
    train = out / "train" / "pick_demo"
    eps = [json.loads(line) for line in (train / "meta/episodes.jsonl").read_text().splitlines()]
    assert [e["episode_index"] for e in eps] == list(range(5))
    assert json.loads((train / "meta/modality.json").read_text()) == prep.MODALITY
    info = json.loads((train / "meta/info.json").read_text())
    assert info["total_episodes"] == 5 and info["features"]["action.eef_9d"]["shape"] == [9]
    df = pd.read_parquet(train / prep.DATA_PATH.format(episode_chunk=0, episode_index=4))
    assert (df["episode_index"] == 4).all() and df["index"].iloc[0] == 4 * 30
    old = eps[4]["source_episode_index"]
    src_df = pd.read_parquet(src / prep.DATA_PATH.format(episode_chunk=0, episode_index=old))
    action = np.stack(src_df["action"].to_numpy())
    np.testing.assert_allclose(np.stack(df["action.eef_9d"].to_numpy()), tcp_to_eef9d(action[:, :6]), atol=1e-4)
    link = train / prep.VIDEO_PATH.format(episode_chunk=0, episode_index=4, video_key="observation.images.wrist")
    assert link.is_symlink() and link.resolve().name == f"episode_{old:06d}.mp4"


def test_modality_slices_pick_the_right_fields(prepared):
    _, out, _ = prepared
    df = pd.read_parquet(out / "train/pick_demo" / prep.DATA_PATH.format(episode_chunk=0, episode_index=0))
    state = np.stack(df["observation.state"].to_numpy())
    m = prep.MODALITY["state"]
    np.testing.assert_array_equal(state[:, m["gripper"]["start"] : m["gripper"]["end"]], state[:, 6:7])
    action = np.stack(df["action"].to_numpy())
    a = prep.MODALITY["action"]["gripper"]
    assert set(action[:, a["start"] : a["end"]].ravel()) <= {0.0, 1.0}


@pytest.mark.parametrize(
    "breakage, message",
    [
        (lambda root: _edit_info(root, fps=10), "fps"),
        (lambda root: _edit_action(root, 0, 0, 700.0), "leaves"),
        (lambda root: _edit_action(root, 1, 6, np.nan), "NaN"),
        (
            lambda root: next(root.glob("videos/chunk-000/observation.images.side/episode_000002.mp4")).unlink(),
            "video side",
        ),
    ],
)
def test_prepare_rejects_bad_exports(tmp_path, breakage, message):
    src = make_export(tmp_path / "src" / "bad")
    breakage(src)
    with pytest.raises(prep.DatasetError, match=message):
        prep.prepare(src, tmp_path / "out", 0.2, 0)
    assert not (tmp_path / "out").exists()


def _edit_info(root: Path, **changes) -> None:
    info = json.loads((root / "meta/info.json").read_text())
    info.update(changes)
    (root / "meta/info.json").write_text(json.dumps(info))


def _edit_action(root: Path, ep: int, col: int, value: float) -> None:
    path = root / prep.DATA_PATH.format(episode_chunk=0, episode_index=ep)
    df = pd.read_parquet(path)
    action = np.stack(df["action"].to_numpy())
    action[3, col] = value
    df["action"] = list(action.astype(np.float32))
    df.to_parquet(path, index=False)


# ---------- client: observation / action conversion and wire format ----------


def _obs() -> dict:
    img = np.zeros((360, 640, 3), np.uint8)
    return {
        "images": {c: img for c in ("table", "side", "wrist")},
        "joints": [0, 10, -20, 0, -70, 30, 0],
        "tcp_pose": [400, 0, -300, -180, 0, 30],
        "trial": {"instruction": "pick M4_02 at (412.0, -87.5, yaw 33)"},
    }


def test_obs_to_groot_shapes_and_dtypes():
    gc = pytest.importorskip("policy_learn.groot.groot_client")
    g = gc.obs_to_groot(_obs())
    assert all(v.shape == (1, 1, 360, 640, 3) and v.dtype == np.uint8 for v in g["video"].values())
    assert {k: v.shape for k, v in g["state"].items()} == {
        "eef_9d": (1, 1, 9),
        "gripper": (1, 1, 1),
        "joints": (1, 1, 6),
    }
    assert all(v.dtype == np.float32 for v in g["state"].values())
    assert g["language"] == {gc.LANGUAGE_KEY: [["pick M4_02 at (412.0, -87.5, yaw 33)"]]}
    with pytest.raises(ValueError, match="instruction"):
        gc.obs_to_groot({**_obs(), "trial": {}})


def _fake_action(obs: dict, horizon: int = 16) -> dict:
    tcp = np.asarray(obs["tcp_pose"], float)
    rows = np.tile(tcp, (horizon, 1))
    rows[:, 2] -= np.arange(horizon)  # go down 1 mm per row
    rows[:, 5] = 175 + np.arange(horizon) * 1.0  # r crosses 180
    return {"eef_9d": tcp_to_eef9d(rows)[None], "gripper": np.linspace(0, 1, horizon, dtype=np.float32)[None, :, None]}


def test_groot_to_rows_gives_absolute_continuous_act_tcp():
    gc = pytest.importorskip("policy_learn.groot.groot_client")
    rows = np.array(gc.groot_to_rows(_fake_action(_obs()), [400, 0, -300, -180, 0, 170], execution_horizon=8))
    assert rows.shape == (8, 7)
    np.testing.assert_allclose(rows[:, 2], -300 - np.arange(8), atol=1e-3)
    np.testing.assert_allclose(rows[:, 5], 175 + np.arange(8), atol=1e-3)
    np.testing.assert_allclose(rows[:, 3], -180, atol=1e-3)
    np.testing.assert_allclose(rows[:, 6], np.linspace(0, 1, 16)[:8], atol=1e-7)


def _serve_once(socket, unpack, pack, handler) -> None:
    for _ in range(2):
        request = unpack(socket.recv())
        socket.send(pack(handler(request)))


def test_client_round_trip_over_zmq():
    zmq = pytest.importorskip("zmq")
    gc = pytest.importorskip("policy_learn.groot.groot_client")
    ctx = zmq.Context()
    server = ctx.socket(zmq.REP)
    port = server.bind_to_random_port("tcp://127.0.0.1")
    seen = {}

    def handler(request):
        if request["endpoint"] == "ping":
            return {"status": "ok"}
        seen.update(request["data"]["observation"])
        return [_fake_action(_obs()), {}]

    thread = threading.Thread(target=_serve_once, args=(server, gc.unpack, gc.pack, handler))
    thread.start()
    client = gc.GrootTcpClient(port=port, timeout_ms=5000)
    assert client.ping()
    rows = client.act(_obs())
    thread.join(5)
    client.close()
    server.close()
    ctx.term()
    assert len(rows) == 16 and seen["state"]["eef_9d"].shape == (1, 1, 9)
    assert seen["language"][gc.LANGUAGE_KEY] == [["pick M4_02 at (412.0, -87.5, yaw 33)"]]


def test_client_describe_requires_the_versioned_attestation_envelope():
    gc = pytest.importorskip("policy_learn.groot.groot_client")

    class Calling:
        def __init__(self, reply):
            self.reply = reply

        def call(self, endpoint, data=None):
            assert endpoint == "describe" and data is None
            return self.reply

    valid = {
        "protocol": "fanuc-groot-attestation-v1",
        "attestation": {"checkpoint_sha256": "a" * 64},
    }
    client = object.__new__(gc.GrootTcpClient)
    client.call = Calling(valid).call
    assert client.describe() == valid["attestation"]

    for bad in (
        {},
        {"protocol": "wrong", "attestation": {}},
        {
            "protocol": "fanuc-groot-attestation-v1",
            "attestation": [],
        },
    ):
        client.call = Calling(bad).call
        with pytest.raises(gc.PolicyServerError, match="attestation"):
            client.describe()


# ---------- cross-checks against the real Isaac-GR00T code ----------


@pytest.fixture(scope="module")
def groot():
    sys.path.insert(0, GROOT_SRC)
    if "gr00t.policy" not in sys.modules:  # skip gr00t/policy/__init__ (it imports torch)
        pkg = types.ModuleType("gr00t.policy")
        pkg.__path__ = [str(Path(GROOT_SRC) / "gr00t/policy")]
        sys.modules["gr00t.policy"] = pkg
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    if "new_embodiment" not in MODALITY_CONFIGS:
        importlib.import_module("policy_learn.groot.fanuc_er4ia_config")
    return MODALITY_CONFIGS


@needs_groot
def test_rot6d_matches_groot_end_effector_pose(groot):
    from gr00t.data.state_action.pose import EndEffectorPose
    from gr00t.data.types import ActionFormat

    for row in tcp_to_eef9d(_random_tcp(20)):
        pose = EndEffectorPose.from_action_format(row.astype(np.float64), ActionFormat.XYZ_ROT6D)
        np.testing.assert_allclose(pose.xyz_rot6d, row, atol=1e-5)


@needs_groot
def test_groot_stats_and_processor_round_trip(groot, prepared):
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.state_action.state_action_processor import StateActionProcessor
    from gr00t.data.stats import generate_rel_stats, generate_stats

    _, out, _ = prepared
    ds = out / "train" / "pick_demo"
    generate_stats(ds)
    generate_rel_stats(ds, EmbodimentTag.NEW_EMBODIMENT)
    cfg = {k: v for k, v in groot["new_embodiment"].items() if k in ("state", "action")}
    loader = LeRobotEpisodeLoader(ds, cfg)
    stats = loader.get_dataset_statistics()
    assert set(stats["relative_action"]) == {"eef_9d"}
    proc = StateActionProcessor(
        {"new_embodiment": cfg}, {"new_embodiment": stats}, use_percentiles=True, use_relative_action=True
    )
    df = loader[0]
    horizon = len(cfg["action"].delta_indices)
    state = {k: np.stack(df[f"state.{k}"].to_numpy())[:1] for k in cfg["state"].modality_keys}
    action = {k: np.stack(df[f"action.{k}"].to_numpy())[:horizon] for k in cfg["action"].modality_keys}
    normed = proc.apply_action(action, "new_embodiment", state=state)
    assert all(np.abs(v).max() <= 1.0 for v in normed.values())
    back = proc.unapply_action(normed, "new_embodiment", state=state)
    np.testing.assert_allclose(back["eef_9d"][:, :3], action["eef_9d"][:, :3], atol=0.5)  # mm, after q01/q99 clip
    np.testing.assert_allclose(back["gripper"], action["gripper"], atol=1e-3)


@needs_groot
def test_client_bytes_decode_with_groot_serializer(groot):
    gc = pytest.importorskip("policy_learn.groot.groot_client")
    from gr00t.policy.server_client import MsgSerializer

    request = MsgSerializer.from_bytes(
        gc.pack({"endpoint": "get_action", "data": {"observation": gc.obs_to_groot(_obs()), "options": None}})
    )
    obs = request["data"]["observation"]
    assert obs["video"]["wrist"].shape == (1, 1, 360, 640, 3) and obs["state"]["joints"].dtype == np.float32
    reply = gc.unpack(MsgSerializer.to_bytes([_fake_action(_obs()), {}]))
    assert reply[0]["eef_9d"].shape == (1, 16, 9)
