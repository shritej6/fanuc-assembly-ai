"""Client for a GR00T policy server: send the cell observation, get act_tcp rows.
Authored by Shritej Gunda."""

from __future__ import annotations

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

from policy_learn.groot.tcp_convert import eef9d_to_tcp, tcp_to_eef9d

CAMERAS = ("table", "side", "wrist")
LANGUAGE_KEY = "annotation.human.task_description"
DT_S = 0.2  # the training data's control step (5 Hz)
ATTESTATION_PROTOCOL = "fanuc-groot-attestation-v1"


class PolicyServerError(RuntimeError):
    """Raised when the server answers with an error payload."""


def instruction_of(obs: dict) -> str:
    """Return the instruction text from an observation."""
    text = obs.get("instruction") or obs.get("trial", {}).get("instruction")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("observation has no instruction (obs['instruction'] or obs['trial']['instruction'])")
    return text


def obs_to_groot(obs: dict) -> dict:
    """Return the GR00T observation for one cell observation."""
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
    """Return absolute act_tcp rows from a GR00T action dict."""
    eef = np.asarray(action["eef_9d"], dtype=np.float64)[0]
    grip = np.asarray(action["gripper"], dtype=np.float64)[0, :, 0]
    if eef.ndim != 2 or eef.shape[1] != 9 or len(grip) != len(eef):
        raise ValueError(f"unexpected action shapes eef_9d {eef.shape} gripper {grip.shape}")
    n = len(eef) if execution_horizon is None else min(execution_horizon, len(eef))
    tcp = eef9d_to_tcp(eef[:n], np.asarray(current_tcp, dtype=np.float64)[3:6])
    # Keep the raw value: only ChunkGuard decides to snap or refuse the gripper.
    return np.column_stack([tcp, grip[:n]]).tolist()


def pack(data) -> bytes:
    """Return the msgpack bytes for a message."""
    return msgpack.packb(data, default=mnp.encode)


def unpack(raw: bytes):
    """Return the message decoded from msgpack bytes."""
    return msgpack.unpackb(raw, object_hook=mnp.decode, raw=False)


class GrootTcpClient:
    """ZMQ client for a GR00T policy server."""
    def __init__(
        self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 30000, execution_horizon: int | None = None
    ):
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
        """Send one request and return the reply."""
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
        """Return True when the server answers."""
        try:
            return self.call("ping").get("status") == "ok"
        except TimeoutError:
            return False

    def describe(self) -> dict:
        """Return the deployment attestation reported by the server."""
        reply = self.call("describe")
        valid = (
            isinstance(reply, dict)
            and reply.get("protocol") == ATTESTATION_PROTOCOL
            and isinstance(reply.get("attestation"), dict)
        )
        if not valid:
            raise PolicyServerError("describe: invalid or missing versioned attestation envelope")
        return dict(reply["attestation"])

    def reset(self) -> None:
        """Reset the server session."""
        self.call("reset", {"options": None})

    def predict(self, obs: dict) -> dict:
        """Return the raw action dict for one observation."""
        action, _info = self.call("get_action", {"observation": obs_to_groot(obs), "options": None})
        return action

    def act(self, obs: dict) -> list[list[float]]:
        """Return act_tcp rows for one observation."""
        return groot_to_rows(self.predict(obs), obs["tcp_pose"], self.execution_horizon)

    def close(self) -> None:
        """Close the socket."""
        self.socket.close()
        self.context.term()
