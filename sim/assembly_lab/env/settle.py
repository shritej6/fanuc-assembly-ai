"""Reset settling for several cells at once: step physics until every object of every resetting env is at rest.

Criteria (coordinator spec, mimic_env.SETTLE): motion is the displacement over a sliding window_s window (position
change / window < 1 mm/s, rotation change / window < 0.5 deg/s), so a one-step solver velocity spike is not motion but
sustained motion still fails; every object's net contact force must change < 0.05 N between physics steps. An env is
settled once it has been calm for hold_s and at least min_s has passed; it gives up at cap_s. Each env is judged on its
own objects only; the loop runs until every env settled or the cap. On failure an env's detail names the worst object
and value for each quantity. Robot joint targets are held (whatever the caller wrote) during the loop.
"""

from __future__ import annotations

import math

import torch


def _detail(lin, ang, dfo, best: int, fails: dict, dt: float, pool: list[str], names: list[str],
            force_order: list[str]) -> dict:
    worst = lambda v: (pool[int(v.argmax())], float(v.max()))
    return {"max_lin_mm_s": [worst(lin)[0], round(worst(lin)[1] * 1000, 3)],
            "max_ang_deg_s": [worst(ang)[0], round(math.degrees(worst(ang)[1]), 3)],
            "max_dforce_N": [pool[names.index(force_order[int(dfo.argmax())])], round(float(dfo.max()), 4)],
            "longest_calm_s": round(best * dt, 3), "failed_steps": dict(fails)}


def settle(env, env_ids: list[int], spec: dict, min_s: float) -> dict[int, tuple[bool, float, dict]]:
    """{env_id: (settled, simulated seconds, detail)} for every env in env_ids (detail is {} when settled)."""
    import isaaclab.utils.math as M

    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    objs = [env.scene[n] for n in env.names]
    net = env.scene["object_contact"]
    dt = env.physics_dt
    need, cap = max(1, round(spec["hold_s"] / dt)), round(spec["cap_s"] / dt)
    w = max(1, round(spec["window_s"] / dt))
    hist_p, hist_q = [], []
    prev_f, lin, ang, dfo = None, None, None, None
    calm = {e: 0 for e in env_ids}
    best = dict(calm)
    fails = {e: {"lin": 0, "ang": 0, "dforce": 0} for e in env_ids}
    out: dict[int, tuple[bool, float, dict]] = {}
    for k in range(1, cap + 1):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=dt)
        hist_p.append(torch.stack([o.data.root_link_pos_w.torch[ids] for o in objs], dim=1).clone())   # (E, N, 3)
        hist_q.append(torch.stack([o.data.root_link_quat_w.torch[ids] for o in objs], dim=1).clone())
        f = net.data.net_forces_w.torch[ids].clone()
        dfo = None if prev_f is None else (f - prev_f).norm(dim=-1)
        prev_f = f
        if len(hist_p) <= w or dfo is None:
            continue
        p0, q0 = hist_p.pop(0), hist_q.pop(0)
        lin = (hist_p[-1] - p0).norm(dim=-1) / (w * dt)
        ang = M.quat_error_magnitude(hist_q[-1], q0) / (w * dt)
        ok_lin = (lin.max(dim=1).values < spec["lin_m_s"]).tolist()
        ok_ang = (ang.max(dim=1).values < spec["ang_rad_s"]).tolist()
        ok_f = (dfo.max(dim=1).values < spec["dforce_n"]).tolist()
        for i, e in enumerate(env_ids):
            if e in out:
                continue
            good = {"lin": ok_lin[i], "ang": ok_ang[i], "dforce": ok_f[i]}
            for key, g in good.items():
                fails[e][key] += 0 if g else 1
            calm[e] = calm[e] + 1 if all(good.values()) else 0
            best[e] = max(best[e], calm[e])
            if calm[e] >= need and k * dt >= min_s:
                out[e] = (True, k * dt, {})
        if len(out) == len(env_ids):
            return out
    force_order = list(net.body_names)
    for i, e in enumerate(env_ids):
        if e in out:
            continue
        if lin is None:                                          # cap shorter than one window: nothing measured
            out[e] = (False, cap * dt, {"longest_calm_s": 0.0, "failed_steps": fails[e]})
        else:
            out[e] = (False, cap * dt, _detail(lin[i], ang[i], dfo[i], best[e], fails[e], dt, env.pool, env.names,
                                               force_order))
    return out


class SettleTracker:
    """Settle some envs inside the normal env.step while the others keep running (per-env reset).

    start(env_id) after the cell was placed; tick() once per physics step (TcpAction.apply_actions, i.e. after the
    previous physics step); envs that settled or hit the cap come back from tick() as (env_id, settled, sim seconds,
    detail). Same criteria and counting as settle(): the first tick after start is the placement itself and is skipped,
    so sample k is the state after k physics steps. One GPU->CPU sync per physics step while any env settles."""

    def __init__(self, env, spec: dict):
        self.env, self.spec = env, spec
        self.dt = env.physics_dt
        self.w = max(1, round(spec["window_s"] / self.dt))
        self.need, self.cap = max(1, round(spec["hold_s"] / self.dt)), round(spec["cap_s"] / self.dt)
        self.active: dict[int, dict] = {}
        self._buf_p = self._buf_q = self._prev_f = None
        self._slot = 0

    def start(self, env_id: int, min_s: float) -> None:
        self.active[int(env_id)] = {"k": -1, "calm": 0, "best": 0, "min_s": min_s,
                                    "fails": {"lin": 0, "ang": 0, "dforce": 0}}

    def _read(self):
        objs = [self.env.scene[n] for n in self.env.names]
        p = torch.stack([o.data.root_link_pos_w.torch for o in objs], dim=1)          # (E, N, 3)
        q = torch.stack([o.data.root_link_quat_w.torch for o in objs], dim=1)
        f = self.env.scene["object_contact"].data.net_forces_w.torch.clone()           # (E, B, 3)
        return p, q, f

    def tick(self) -> list[tuple[int, bool, float, dict]]:
        if not self.active:
            return []
        import isaaclab.utils.math as M

        p, q, f = self._read()
        if self._buf_p is None:
            self._buf_p = p.unsqueeze(1).repeat(1, self.w + 1, 1, 1)
            self._buf_q = q.unsqueeze(1).repeat(1, self.w + 1, 1, 1)
            self._prev_f = f
        self._slot = (self._slot + 1) % (self.w + 1)
        self._buf_p[:, self._slot], self._buf_q[:, self._slot] = p, q
        old = (self._slot + 1) % (self.w + 1)                    # the sample w physics steps ago
        lin = (p - self._buf_p[:, old]).norm(dim=-1) / (self.w * self.dt)
        ang = M.quat_error_magnitude(q, self._buf_q[:, old]) / (self.w * self.dt)
        dfo = (f - self._prev_f).norm(dim=-1)
        self._prev_f = f
        ok = torch.stack([lin.max(dim=1).values < self.spec["lin_m_s"], ang.max(dim=1).values < self.spec["ang_rad_s"],
                          dfo.max(dim=1).values < self.spec["dforce_n"]], dim=1).tolist()
        done = []
        for e, st in list(self.active.items()):
            st["k"] += 1
            k = st["k"]
            if k <= self.w:                       # sample 0 is the placement; need w + 1 samples for one window
                continue
            good = dict(zip(("lin", "ang", "dforce"), ok[e]))
            for key, g in good.items():
                st["fails"][key] += 0 if g else 1
            st["calm"] = st["calm"] + 1 if all(good.values()) else 0
            st["best"] = max(st["best"], st["calm"])
            if st["calm"] >= self.need and k * self.dt >= st["min_s"]:
                done.append((e, True, k * self.dt, {}))
            elif k >= self.cap:
                done.append((e, False, k * self.dt, _detail(lin[e], ang[e], dfo[e], st["best"], st["fails"], self.dt,
                                                           self.env.pool, self.env.names,
                                                           list(self.env.scene["object_contact"].body_names))))
        for e, *_ in done:
            del self.active[e]
        return done
