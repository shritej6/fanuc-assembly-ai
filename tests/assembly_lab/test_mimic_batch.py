"""mimic/batch.py against a fake batched env: lanes in threads, the env on the test's thread."""

import threading

import pytest

from sim.assembly_lab.mimic.batch import Hub, LaneEnv, LaneFailed


class FakeEnv:
    """reset_many / step_tcp_many / end_episode with the documented semantics; records who was stepped when."""

    def __init__(self, n, fail_at=None, ik_env=None):
        self.n, self.calls, self.ends, self.resets = n, [], [], 0
        self.t = [0] * n
        self.in_episode = [False] * n
        self.fail_at, self.ik_env = fail_at, ik_env

    def reset_many(self, layouts, fns):
        assert len(layouts) == self.n and not any(self.in_episode)
        self.resets += 1
        self.in_episode = [True] * self.n
        return [{"env": k, "layout": layouts[k], "t": 0} for k in range(self.n)]

    def step_tcp_many(self, reqs):
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("physics blew up")
        self.calls.append(sorted(reqs))
        out = {}
        for k in reqs:
            assert self.in_episode[k], f"env {k} stepped after its episode ended"
            self.t[k] += 1
            out[k] = {"env": k, "t": self.t[k], **({"ik_error": "no IK"} if k == self.ik_env else {})}
        return out

    def end_episode(self, k, ok):
        self.in_episode[k] = False
        self.ends.append((k, ok))


class PerEnvResetFake(FakeEnv):
    """begin_reset / ready_resets: a cell settles for `settle` step_tcp_many calls, then its episode starts. Records
    every hold step given to a recording cell that did not ask for it (it would land in that cell's demo)."""

    def __init__(self, n, settle=2, **kw):
        super().__init__(n, **kw)
        self.settle, self.settling, self.ready, self.unasked = settle, {}, {}, []

    def begin_reset(self, reqs):
        for k, (lay, _) in reqs.items():
            assert not self.in_episode[k] and k not in self.settling
            self.settling[k] = [self.settle, lay]

    def ready_resets(self):
        out, self.ready = self.ready, {}
        return out

    def step_tcp_many(self, reqs):
        assert not set(reqs) & set(self.settling), "stepped a settling cell"
        self.unasked += [k for k in range(self.n) if self.in_episode[k] and k not in reqs]
        out = super().step_tcp_many(reqs)
        for k, s in list(self.settling.items()):
            s[0] -= 1
            if s[0] <= 0:
                del self.settling[k]
                self.in_episode[k], self.t[k] = True, 0
                self.ready[k] = {"env": k, "layout": s[1], "t": 0}
        return out


def _run(env, n, lengths, episodes=2, on_reset_all=None, maybe_rotate=None):
    hub = Hub(env, n, on_reset_all=on_reset_all, maybe_rotate=maybe_rotate)
    metas, errors = [], []

    def lane(k):
        e = LaneEnv(hub, k)
        try:
            for ep in range(episodes):
                e.reset({"lane": k, "ep": ep})
                for _ in range(lengths[k]):
                    e.step_tcp([[0] * 7], 0.1)
                with hub.lock:  # meta line, then end: the order the recorder exports in
                    metas.append(k)
                    e.end_episode(True)
        except Exception as ex:  # noqa: BLE001 - collected for the assertions
            errors.append(ex)
        finally:
            hub.retire(k)

    threads = [threading.Thread(target=lane, args=(k,)) for k in range(n)]
    for th in threads:
        th.start()
    try:
        hub.serve()
    finally:
        for th in threads:
            th.join(timeout=10)
    return hub, metas, errors


def test_lockstep_lanes_step_together_and_meta_follows_export_order():
    env = FakeEnv(3)
    rotations = []
    _, metas, errors = _run(env, 3, [5, 2, 8], on_reset_all=lambda: rotations.append(list(env.in_episode)))
    assert not errors
    assert [k for k, _ in env.ends] == metas                  # meta line order == end_episode (export) order
    assert env.resets == 2 and rotations == [[False] * 3] * 2  # rotation only while no cell is mid-episode
    assert env.calls[0] == [0, 1, 2] and env.calls[2] == [0, 2]  # lane 1 ended after 2 steps: never stepped again
    assert sum(len(c) for c in env.calls) == 2 * (5 + 2 + 8)   # every step was asked for by its lane


def test_per_env_reset_does_not_wait_for_the_slowest_lane():
    env = PerEnvResetFake(2)
    _, metas, errors = _run(env, 2, [1, 6], episodes=3)
    assert not errors and env.resets == 0
    assert metas.count(0) == 3 and metas.index(0) < metas.index(1)  # lane 0 cycled on while lane 1 still ran
    assert env.unasked == []  # no recording cell was ever held without asking
    assert env.calls[:2] == [[], []]  # both cells settle first: two hold steps, nothing recorded
    assert sum(len(c) for c in env.calls) == 3 * (1 + 6)


def test_per_env_rotation_only_while_no_end_is_in_flight():
    env = PerEnvResetFake(3, settle=1)
    hub_ref, seen = [], []

    def rotate():
        seen.append(hub_ref[0].lock.locked())

    hub = Hub(env, 3, maybe_rotate=rotate)
    hub_ref.append(hub)
    lanes = []
    for k in range(3):
        def lane(k=k):
            e = LaneEnv(hub, k)
            for ep in range(2):
                e.reset({"lane": k, "ep": ep})
                for _ in range(k + 1):
                    e.step_tcp([[0] * 7], 0.1)
                with hub.lock:
                    e.end_episode(True)
            hub.retire(k)
        lanes.append(threading.Thread(target=lane))
    for th in lanes:
        th.start()
    hub.serve()
    for th in lanes:
        th.join(timeout=10)
    assert seen and all(seen)  # the hub held the lane lock for every rotation: no lane between meta and end
    assert env.unasked == [] and len(env.ends) == 6


def test_ik_error_raises_in_that_lane_only():
    env = FakeEnv(2, ik_env=1)
    _, metas, errors = _run(env, 2, [3, 3], episodes=1)
    assert metas == [0] and len(errors) == 1 and "no IK" in str(errors[0])


def test_env_failure_reaches_every_lane_and_the_caller():
    env = FakeEnv(2, fail_at=1)
    with pytest.raises(RuntimeError, match="physics"):
        _run(env, 2, [4, 4], episodes=1)


def test_retired_lane_cell_is_ended_at_lockstep_resets():
    env = FakeEnv(2)
    hub = Hub(env, 2)
    hub.lanes[0].layout = {"old": 0}
    hub.live = {1}
    out = hub._reset_all({1: ("reset", {"new": 1}, None)})
    assert set(out) == {1} and env.ends[-1] == (0, False)  # the retired cell's new episode is ended (empty) at once


def test_lane_failed_is_raised_after_env_error():
    hub = Hub(FakeEnv(1), 1)
    hub.error = RuntimeError("boom")
    with pytest.raises(LaneFailed):
        hub.request(0, ("step", [[0] * 7], 0.1))


def test_batched_cells_share_the_worker_chunks(tmp_path):
    from sim.assembly_lab.mimic.generate import lane_out
    from sim.assembly_lab.mimic.run_generation import closed_chunks

    d = tmp_path / "pick_insert" / "p4"
    d.mkdir(parents=True)
    out = d / "h_g0_w0.hdf5"
    for k in range(3):
        (d / f"{lane_out(out, k).name}.stats.json").write_text("{}")
    for c in range(2):
        (d / f"h_g0_w0_c{c:03d}.hdf5").write_bytes(b"")
        (d / f"h_g0_w0_c{c:03d}.hdf5.meta.jsonl").write_text("")
    assert lane_out(out, 3).name == "h_g0_w0_e03.hdf5"
    assert [p.split("/")[-1] for p in closed_chunks(tmp_path, "")] == ["h_g0_w0_c000.hdf5", "h_g0_w0_c001.hdf5"]
    assert [p.split("/")[-1] for p in closed_chunks(tmp_path, f"--out {out}")] == ["h_g0_w0_c000.hdf5"]


def test_rotate_shared_uses_the_batched_env_rotate_and_waits_for_enough_demos(tmp_path):
    from types import SimpleNamespace

    from sim.assembly_lab.mimic.generate import _rotate_shared

    out = tmp_path / "h_g0_w0.hdf5"
    calls = []
    raw = SimpleNamespace(rotate=lambda nxt: calls.append(("rotate", nxt)) or "closed.hdf5")
    owner = SimpleNamespace(raw=raw, _annotate=lambda p: calls.append(("annotate", p)), record_path=None,
                            rotate=lambda nxt: calls.append(("backend_rotate", nxt)))
    run_cfg = SimpleNamespace(path=tmp_path / "h_g0_w0_c000.hdf5", metas=[{}, {}])
    _rotate_shared(owner, run_cfg, out, min_metas=3)
    assert calls == [] and len(run_cfg.metas) == 2  # fewer demos than lanes: keep recording into this file
    run_cfg.metas.append({})
    _rotate_shared(owner, run_cfg, out, min_metas=3)
    nxt = str(tmp_path / "h_g0_w0_c001.hdf5")
    assert calls == [("rotate", nxt), ("annotate", "closed.hdf5")]  # never IsaacBackend.rotate (exports all)
    assert run_cfg.path.name == "h_g0_w0_c001.hdf5" and run_cfg.metas == [] and owner.record_path == nxt
