"""Serve a LeRobot ACT or SmolVLA checkpoint behind the GR00T PolicyServer protocol (sim comparison only).
Authored by Shritej Gunda."""

from __future__ import annotations

import argparse
import logging
import traceback
from pathlib import Path

import numpy as np

from policy_learn.groot.groot_client import CAMERAS, LANGUAGE_KEY, pack, unpack

log = logging.getLogger("baseline_server")
STATE_DIM, ACTION_DIM = 16, 10   # policy_learn/baselines/lerobot_view.py STATE_NAMES / ACTION_NAMES


class ObservationError(ValueError):
    """Raised when the observation does not match what the checkpoint was trained on."""


def groot_obs_to_lerobot(observation: dict, image_hw: tuple[int, int] | None = None) -> dict:
    """Return a LeRobot frame built from one GR00T observation."""
    frame = {}
    for cam in CAMERAS:
        img = np.asarray(observation["video"][cam])
        if img.dtype != np.uint8 or img.ndim != 5 or img.shape[:2] != (1, 1) or img.shape[-1] != 3:
            raise ObservationError(f"video[{cam}] must be (1, 1, H, W, 3) uint8, got {img.dtype} {img.shape}")
        if image_hw is not None and img.shape[2:4] != tuple(image_hw):
            raise ObservationError(f"video[{cam}] is {img.shape[2:4]} but the checkpoint was trained on {image_hw}")
        frame[f"observation.images.{cam}"] = np.ascontiguousarray(img[0, 0].transpose(2, 0, 1), np.float32) / 255.0
    state = observation["state"]
    parts = [np.asarray(state[k], np.float32).reshape(-1) for k in ("eef_9d", "gripper", "joints")]
    if [len(p) for p in parts] != [9, 1, 6]:
        raise ObservationError(f"state eef_9d/gripper/joints sizes {[len(p) for p in parts]}, expected [9, 1, 6]")
    frame["observation.state"] = np.concatenate(parts)
    text = observation["language"][LANGUAGE_KEY][0][0]
    if not isinstance(text, str) or not text.strip():
        raise ObservationError("empty instruction")
    frame["task"] = text
    return frame


def chunk_to_groot_action(chunk: np.ndarray, n_rows: int) -> dict:
    """Return a GR00T action dict holding the first n_rows rows of a chunk."""
    chunk = np.asarray(chunk, np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or not np.isfinite(chunk).all():
        raise ValueError(f"policy returned a bad chunk {chunk.shape}")
    rows = chunk[:n_rows]
    return {"eef_9d": rows[None, :, :9], "gripper": rows[None, :, 9:10]}


class LeRobotChunkPolicy:
    """LeRobot ACT or SmolVLA checkpoint that answers one chunk per observation."""

    def __init__(self, checkpoint: Path, device: str):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device = device
        self.policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg).to(device).eval()
        self.torch, self.device, self.kind = torch, device, cfg.type
        self.n_rows = cfg.n_action_steps
        shapes = {k: tuple(f.shape) for k, f in cfg.input_features.items()}
        if shapes.get("observation.state") != (STATE_DIM,) or tuple(cfg.output_features["action"].shape) != (
                ACTION_DIM,):
            raise ValueError(f"{checkpoint}: features {shapes} / {cfg.output_features} are not the baseline view's")
        self.image_hw = shapes[f"observation.images.{CAMERAS[0]}"][1:]
        log.info("loaded %s from %s: %d rows per chunk, images %s", self.kind, checkpoint, self.n_rows, self.image_hw)

    def reset(self, options=None) -> dict:
        """Clear the policy state for a new episode."""
        self.policy.reset()
        return {}

    def get_action(self, observation: dict, options=None) -> list:
        """Return the action chunk for one observation."""
        frame = groot_obs_to_lerobot(observation, self.image_hw)
        batch = {k: v if k == "task" else self.torch.from_numpy(v)[None].to(self.device) for k, v in frame.items()}
        batch["task"] = [frame["task"]]
        with self.torch.inference_mode():
            chunk = self.policy.predict_action_chunk(batch)[0].float().cpu().numpy()
        return [chunk_to_groot_action(chunk, self.n_rows), {"policy": self.kind}]


def serve(policy, host: str, port: int) -> None:
    """Serve the policy until interrupted."""
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{host}:{port}")
    handlers = {"ping": lambda **_: {"status": "ok", "message": "Server is running"},
                "reset": policy.reset, "get_action": policy.get_action}
    log.info("serving on %s:%d", host, port)
    running = True
    while running:
        request = unpack(socket.recv())
        endpoint = request.get("endpoint", "get_action") if isinstance(request, dict) else None
        try:
            if endpoint == "kill":
                running, reply = False, {"status": "ok"}
            elif endpoint in handlers:
                reply = handlers[endpoint](**request.get("data", {}))
            else:
                raise ValueError(f"unknown endpoint {endpoint!r}")
        except Exception as exc:  # noqa: BLE001 - the request boundary: the client gets the error, the log gets the trace
            log.error("%s failed: %s\n%s", endpoint, exc, traceback.format_exc())
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        socket.send(pack(reply))
    socket.close(linger=1000)
    context.term()


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and start the server."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="a LeRobot .../pretrained_model directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5561)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    serve(LeRobotChunkPolicy(args.checkpoint.expanduser(), args.device), args.host, args.port)


if __name__ == "__main__":
    main()
