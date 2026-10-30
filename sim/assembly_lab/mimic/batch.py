"""Mimic generation on N cloned cells in one process (agent 9's AssemblyCellEnv(num_envs=N), BATCH_API.md).

Each env is a lane: one thread running the same per-env attempt loop as a single-env worker (agent 3's
EpisodeRecorder, gates and force guard; this package's source choice, subtask progress, accept / reject). Lanes
talk to the cells only through the Hub, which owns the env on the main thread:

    step    a lane's EpisodeRecorder plays one act_tcp chunk; the hub waits until every live lane that is not
            resetting asked for a step (or an end / reset), then makes ONE step_tcp_many call. A cell that is
            recording is never given a hold step it did not ask for (BATCH_API hub rules 1 and 2).
    end     the lane's episode is over: meta line + end_episode(k, ok) are served at once, before the next step, in
            the order the lanes ask (Hub.lock), so <file>.meta.jsonl lines follow the recorder's export order.
    reset   the lane wants a new layout. With per-env reset (env.begin_reset) the cell starts settling at once and
            the others keep stepping; the lane gets its obs from ready_resets() after a later step_tcp_many ({} when
            every live cell is settling). The first reset of all is always a lockstep reset_many, which builds the
            env's observation buffers. Without begin_reset the hub resets every cell together when every live lane waits
            for one (reset_many) and rotates the recorder file there, when no cell has an episode in progress.

With per-env reset the recorder file rotates between calls whenever no lane is between its meta line and its end
(Hub.lock is free): in-progress episodes stay in memory until end_episode (BATCH_API rotate). A lane that finished its
target retires: its cell holds; at a lockstep reset it gets its previous layout and is ended at once.
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
        return self.hub.request(self.k, ("reset", layout, frames_fn))

    def step_tcp(self, rows, dt: float) -> dict:
        obs = self.hub.request(self.k, ("step", rows, dt))
        if obs.get("ik_error"):  # with one env AssemblyCellEnv raises TcpError (a ValueError); keep that contract
            raise ValueError(f"env {self.k}: {obs['ik_error']}")
        return obs

    def end_episode(self, ok: bool) -> None:
        self.hub.request(self.k, ("end", bool(ok)))

    @property
    def placements(self) -> dict:
        return self.hub.env.env.eps[self.k].placements

    def close(self) -> None:
        """The hub owns the env; a lane never closes it."""


def lane_backend(hub: Hub, k: int, scenario: str, product: int, record_path: str):
    """agent 3's IsaacBackend for one lane: its own layout sampling, planning and frames (plans per lane), on the
    shared env through LaneEnv; finish() ends this cell's episode, rotate()/close() are the hub's business."""
    from ..objects import planner
    from ..skills.frames import LocalFrameEnv
    from ..skills.isaac_backend import IsaacBackend

    class LaneBackend(IsaacBackend):
        def __init__(self):  # no env of its own (IsaacBackend.__init__ would build one)
            self.planner, self.scenario, self.products = planner, scenario, (product,)
            self.raw = LaneEnv(hub, k)
            self.env = LocalFrameEnv(self.raw)
            self.record_path = record_path
            self.installed, self.plans, self.refusal = {}, None, None

        def finish(self, ok: bool) -> None:
            self.raw.end_episode(ok)

        def rotate(self, next_path: str) -> None:
            raise RuntimeError("a lane does not rotate the shared recorder file: the hub does at lockstep resets")

        def close(self, annotate_path=None) -> None:
            hub.retire(k)

    return LaneBackend()
