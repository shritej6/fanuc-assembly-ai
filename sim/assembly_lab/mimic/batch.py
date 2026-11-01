"""Run Mimic generation on several cloned cells in one process, one lane thread per cell.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


class LaneFailed(RuntimeError):
    """The hub could not serve a lane's request (the env raised): every waiting lane gets this error."""


@dataclass
class _Lane:
    pending: tuple | None = None
    result: object = None
    done: bool = False
    layout: dict | None = None
    frames_fn: object = None


@dataclass
class Hub:
    """Serves lane requests on the env from the thread that calls serve() (Isaac Sim must stay on one thread)."""

    env: object                       # AssemblyCellEnv(num_envs=N): step_tcp_many / end_episode / begin_reset ...
    n: int
    on_reset_all: object = None       # callable run before a lockstep reset_many (no episode in progress)
    maybe_rotate: object = None       # per-env reset: callable run between calls while no end is in flight
    lock: threading.Lock = field(default_factory=threading.Lock)  # held by a lane around meta write + end

    def __post_init__(self):
        self.cv = threading.Condition()
        self.lanes = [_Lane() for _ in range(self.n)]
        self.live = set(range(self.n))
        self.resetting: set[int] = set()
        self.error: BaseException | None = None
        self.steps = 0
        self.started = False  # the first reset is always reset_many: Isaac Lab's env.reset() builds the obs buffers

    # --- lane side ------------------------------------------------------------------------------------------
    def request(self, k: int, req: tuple):
        """Send a request from lane k to the env thread and return its result."""
        with self.cv:
            if self.error is not None:
                raise LaneFailed(f"env failed: {self.error}") from self.error
            lane = self.lanes[k]
            lane.pending, lane.done = req, False
            self.cv.notify_all()
            self.cv.wait_for(lambda: lane.done or self.error is not None)
            if not lane.done:
                raise LaneFailed(f"env failed: {self.error}") from self.error
            return lane.result

    def retire(self, k: int) -> None:
        """Remove lane k from the live lanes."""
        with self.cv:
            self.live.discard(k)
            self.cv.notify_all()

    # --- env side -------------------------------------------------------------------------------------------
    def serve(self) -> None:
        """Run until every lane retired (call from the main thread, after starting the lane threads)."""
        while True:
            with self.cv:
                self.cv.wait_for(self._ready)
                if not self.live:
                    return
                batch = {k: self.lanes[k].pending for k in self.live - self.resetting
                         if self.lanes[k].pending is not None}
            try:
                results = self._serve(batch)
                self._rotate_if_idle()
            except BaseException as e:
                with self.cv:
                    self.error = e
                    self.cv.notify_all()
                raise
            with self.cv:
                for k, r in results.items():
                    lane = self.lanes[k]
                    lane.pending, lane.result, lane.done = None, r, True
                self.cv.notify_all()

    def _ready(self) -> bool:
        if not self.live:
            return True
        reqs = [self.lanes[k].pending for k in self.live - self.resetting]
        if any(r is not None and r[0] == "end" for r in reqs):
            return True
        if self._per_env_reset() and self.started and any(r is not None and r[0] == "reset" for r in reqs):
            return True
        return all(r is not None for r in reqs)  # per-env: [] when every live cell settles -> one hold step

    def _per_env_reset(self) -> bool:
        return hasattr(self.env, "begin_reset")

    def _serve(self, batch: dict) -> dict:
        ends = {k: r for k, r in batch.items() if r[0] == "end"}
        if ends:
            for k, (_, ok) in ends.items():
                self.env.end_episode(k, ok)
            return dict.fromkeys(ends)
        resets = {k: r for k, r in batch.items() if r[0] == "reset"}
        if resets and self._per_env_reset() and self.started:
            self.env.begin_reset({k: (r[1], r[2]) for k, r in resets.items()})
            with self.cv:  # the lanes stay blocked until ready_resets() hands them their obs
                self.resetting |= set(resets)
                for k in resets:
                    self.lanes[k].pending = None
            return {}
        steps = {k: (r[1], r[2]) for k, r in batch.items() if r[0] == "step"}
        if steps or self.resetting:
            self.steps += 1
            out = dict(self.env.step_tcp_many(steps))
            if self._per_env_reset():
                ready = self.env.ready_resets()
                with self.cv:
                    self.resetting -= set(ready)
                out.update(ready)
            return out
        return self._reset_all(resets)

    def _rotate_if_idle(self) -> None:
        if self.maybe_rotate is None or not self._per_env_reset() or not self.lock.acquire(blocking=False):
            return
        try:
            self.maybe_rotate()
        finally:
            self.lock.release()

    def _reset_all(self, resets: dict) -> dict:
        """Every live lane waits for a reset: rotate (no episode in progress), reset every cell together."""
        if self.on_reset_all is not None:
            self.on_reset_all()
        for k, r in resets.items():
            self.lanes[k].layout, self.lanes[k].frames_fn = r[1], r[2]
        idle = [k for k in range(self.n) if k not in resets]
        if any(self.lanes[k].layout is None for k in idle):
            raise LaneFailed("a retired lane never reset: no layout to give its cell")
        obs = self.env.reset_many([self.lanes[k].layout for k in range(self.n)],
                                  [self.lanes[k].frames_fn if k in resets else None for k in range(self.n)])
        self.started = True
        for k in idle:  # retired: nothing will step this cell; its new episode is ended (empty) at once
            self.env.end_episode(k, False)
        return {k: obs[k] for k in resets}


class LaneEnv:
    """The env as one lane sees it: AssemblyCellEnv's single-cell API (reset / step_tcp), served by the hub."""

    def __init__(self, hub: Hub, k: int):
        self.hub, self.k = hub, k

    def reset(self, layout: dict, frames_fn=None) -> dict:
        """Reset this lane's cell to the layout and return the first observation."""
        return self.hub.request(self.k, ("reset", layout, frames_fn))

    def step_tcp(self, rows, dt: float) -> dict:
        """Step this lane's cell with TCP rows and return the observation."""
        obs = self.hub.request(self.k, ("step", rows, dt))
        if obs.get("ik_error"):  # with one env AssemblyCellEnv raises TcpError (a ValueError); keep that contract
            raise ValueError(f"env {self.k}: {obs['ik_error']}")
        return obs

    def end_episode(self, ok: bool) -> None:
        """End this lane's episode as passed or failed."""
        self.hub.request(self.k, ("end", bool(ok)))

    @property
    def placements(self) -> dict:
        """Return the object placements of this lane's cell."""
        return self.hub.env.env.eps[self.k].placements

    def close(self) -> None:
        """The hub owns the env; a lane never closes it."""


def lane_backend(hub: Hub, k: int, scenario: str, product: int, record_path: str):
    """Return a backend for one lane that drives the shared env through its LaneEnv."""
    from ..objects import planner
    from ..skills.frames import LocalFrameEnv
    from ..skills.isaac_backend import IsaacBackend

    class LaneBackend(IsaacBackend):
        """Backend for one lane that shares the hub's env."""
        def __init__(self):  # no env of its own (IsaacBackend.__init__ would build one)
            self.planner, self.scenario, self.products = planner, scenario, (product,)
            self.raw = LaneEnv(hub, k)
            self.env = LocalFrameEnv(self.raw)
            self.record_path = record_path
            self.installed, self.plans, self.refusal = {}, None, None

        def finish(self, ok: bool) -> None:
            """End this cell's episode."""
            self.raw.end_episode(ok)

        def rotate(self, next_path: str) -> None:
            """Refuse to rotate: the hub rotates the shared recorder file."""
            raise RuntimeError("a lane does not rotate the shared recorder file: the hub does at lockstep resets")

        def close(self, annotate_path=None) -> None:
            """Retire this lane from the hub."""
            hub.retire(k)

    return LaneBackend()
