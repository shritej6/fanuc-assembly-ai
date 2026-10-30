"""Sim-eval / deploy hook: talk to a GR00T policy server with the cell's observation dict, get `act_tcp` rows.

Needs only numpy, msgpack, msgpack-numpy and pyzmq (no torch, no Isaac-GR00T install), so Agent 6's evaluator on
fa-sim and `policy_learn/deploy/` can import it. The wire format is Isaac-GR00T's `PolicyServer`
(`gr00t/policy/server_client.py`, ZMQ REQ/REP + msgpack-numpy); start the server with `serve_policy.sh`.

    from policy_learn.groot.groot_client import GrootTcpClient
    policy = GrootTcpClient(host="127.0.0.1", port=5555)   # ssh -N -L 5555:127.0.0.1:5555 <training box>
    rows = policy.act(obs)        # list of [x, y, z, w, p, r, gripper], absolute, mm / deg, play at dt 0.2 s

`obs` uses the INTERFACES.md keys: images {table, side, wrist: HxWx3 uint8, the whole frame resized to 640x360 as
in the training videos}, joints (7: J1..J6 deg, gripper), tcp_pose (6), instruction (str; or obs["trial"]
["instruction"] as in the official client). The rows are NOT guarded: the caller runs them through
`policy_learn/deploy/guards.py` before they reach the sim or the arm.
"""

from __future__ import annotations

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

from policy_learn.groot.tcp_convert import eef9d_to_tcp, tcp_to_eef9d

CAMERAS = ("table", "side", "wrist")
LANGUAGE_KEY = "annotation.human.task_description"
DT_S = 0.2  # the training data's control step (5 Hz)


class PolicyServerError(RuntimeError):
    """The server answered with an error payload."""


def instruction_of(obs: dict) -> str:
    text = obs.get("instruction") or obs.get("trial", {}).get("instruction")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("observation has no instruction (obs['instruction'] or obs['trial']['instruction'])")
    return text


def obs_to_groot(obs: dict) -> dict:
    """Cell observation -> GR00T Policy API observation (batch 1, one time step)."""
    images = obs["images"]
    video = {}
    for cam in CAMERAS:
        img = np.asarray(images[cam])
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"images[{cam}] must be HxWx3 uint8, got {img.dtype} {img.shape}")
        video[cam] = img[None, None]
    joints = np.asarray(obs["joints"], dtype=np.float64)
    tcp = np.asarray(obs["tcp_pose"], dtype=np.float64)
    if joints.shape != (7,) or tcp.shape != (6,):
        raise ValueError(f"joints must be 7 values and tcp_pose 6, got {joints.shape} / {tcp.shape}")
    state = {
        "eef_9d": tcp_to_eef9d(tcp)[None, None],
        "gripper": joints[None, None, 6:7].astype(np.float32),
        "joints": joints[None, None, :6].astype(np.float32),
    }
    return {"video": video, "state": state, "language": {LANGUAGE_KEY: [[instruction_of(obs)]]}}


def groot_to_rows(action: dict, current_tcp: np.ndarray, execution_horizon: int | None = None) -> list[list[float]]:
    """GR00T action dict {eef_9d: (1, T, 9), gripper: (1, T, 1)} -> absolute act_tcp rows."""
    eef = np.asarray(action["eef_9d"], dtype=np.float64)[0]
    grip = np.asarray(action["gripper"], dtype=np.float64)[0, :, 0]
    if eef.ndim != 2 or eef.shape[1] != 9 or len(grip) != len(eef):
        raise ValueError(f"unexpected action shapes eef_9d {eef.shape} gripper {grip.shape}")
    n = len(eef) if execution_horizon is None else min(execution_horizon, len(eef))
    tcp = eef9d_to_tcp(eef[:n], np.asarray(current_tcp, dtype=np.float64)[3:6])
    gripper = (grip[:n] >= 0.5).astype(np.float64)  # binary in every demo: 0 open / 1 closed
    return np.column_stack([tcp, gripper]).tolist()


def pack(data) -> bytes:
    return msgpack.packb(data, default=mnp.encode)


def unpack(raw: bytes):
    return msgpack.unpackb(raw, object_hook=mnp.decode, raw=False)


class GrootTcpClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 30000,
                 execution_horizon: int | None = None):
        self.address = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self.execution_horizon = execution_horizon
        self.context = zmq.Context()
        self._connect()

    def _connect(self) -> None:
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.address)

    def call(self, endpoint: str, data: dict | None = None):
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self.socket.send(pack(request))
            reply = unpack(self.socket.recv())
        except zmq.error.Again:
            self.socket.close()
            self._connect()  # a timed-out REQ socket cannot send again
            raise TimeoutError(f"no reply from {self.address} in {self.timeout_ms} ms ({endpoint})") from None
        if isinstance(reply, dict) and "error" in reply:
            raise PolicyServerError(f"{endpoint}: {reply['error']}")
        return reply

    def ping(self) -> bool:
        try:
            return self.call("ping").get("status") == "ok"
        except TimeoutError:
            return False

    def reset(self) -> None:
        self.call("reset", {"options": None})

    def predict(self, obs: dict) -> dict:
        action, _info = self.call("get_action", {"observation": obs_to_groot(obs), "options": None})
        return action

    def act(self, obs: dict) -> list[list[float]]:
        return groot_to_rows(self.predict(obs), obs["tcp_pose"], self.execution_horizon)

    def close(self) -> None:
        self.socket.close()
        self.context.term()
