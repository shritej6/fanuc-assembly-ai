"""Record scripted sim demos for Isaac Lab Mimic / GR00T.

    # on fa-sim, inside Isaac Lab's python (see ~/ISAACLAB_READY), GPU 2:
    CUDA_VISIBLE_DEVICES=2 python -m sim.assembly_lab.skills.record --backend isaac --scenario build \\
        --products 1 --target 10
    # any host, pipeline check only (KINEMATIC_DRY_RUN, never written under ~/t12_data/demos):
    python -m sim.assembly_lab.skills.record --backend kinematic --scenario build --target 3 --out /tmp/dry

Scenarios (docs/ROBOT_BRAIN.md section 3):
    build      pick_insert for every part of each product in sheet order (01 -> 04), one episode per part
    wont_seat  pick_insert whose first try is staged off by more than the chamfer catches; the retry must seat it
    dropped    pick_insert of a part lying where it fell (re-pick after a drop)
    fix_above  pull_out of an installed part to an empty test-bed spot (reverse sheet order)
    crowded    place_free of the planner's blocker to an empty remembered test-bed spot (move the blocker first)
    defect     defect_drop of a part into the defect area

Every attempt is counted. Only episodes that pass every success check AND the force guard are written; the rest are
counted with their reason in <out>/<seed_dir>/product_<n>_stats.json. Nothing is retried silently.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import episodes, hdf5_io
from .backends import IsaacBackend, KinematicBackend, NoCase, base_of
from .force_guard import ForceGuard, LimitsError, load_limits
from .gates import GateFailed
from .runner import EpisodeFailed, EpisodeRecorder

DEMO_ROOT = Path.home() / "t12_data" / "demos"
SEED_DIR = {"build": "pick_insert", "wont_seat": "pick_insert", "dropped": "pick_insert",
            "fix_above": "pull_out", "crowded": "place_free", "defect": "defect_drop"}
STAGED_BEYOND_CAPTURE_MM = 1.0  # the staged first try misses by this much more than the part's chamfer capture


def _staged(rng: np.random.Generator, capture_mm: float) -> tuple[float, float, float]:
    """A first-try error the chamfer cannot catch (objects.yaml xy_capture_mm + 1 mm), in a random direction."""
    a, d = rng.uniform(0, 2 * math.pi), capture_mm + STAGED_BEYOND_CAPTURE_MM
    return (d * math.cos(a), d * math.sin(a), 0.0)


def _episode(backend, rec: EpisodeRecorder, rng, scenario: str, product: int, part: str, obs: dict) -> None:
    if scenario in ("build", "wont_seat", "dropped"):
        pick, ins = backend.pick_insert(obs, part, base_of(product))
        case = {"build": "nominal"}.get(scenario, scenario)
        miss = _staged(rng, ins.capture_mm) if scenario == "wont_seat" else (0.0, 0.0, 0.0)
        episodes.pick_insert(rec, pick, ins, case, miss)
    elif scenario == "fix_above":
        pull, spot = backend.pull_out(obs, part, rng)
        episodes.pull_out(rec, pull, spot)
    elif scenario == "crowded":
        pick, spot = backend.crowded(obs, part, rng)
        episodes.place_free(rec, pick, spot)
    elif scenario == "defect":
        pick, area = backend.defect(obs, part, rng)
        episodes.defect_drop(rec, pick, area)
    else:
        raise ValueError(f"unknown scenario {scenario!r}")


def _installed_before(backend, product: int, part: str) -> dict[str, str]:
    parts = backend.parts(product)
    return {p: base_of(product) for p in parts[:parts.index(part)]}


def record(backend, scenario: str, products: list[int], target: int, out: Path, seed: int, max_attempts: int,
           limits_path: Path | None = None, allow_touch_proxy: bool = False) -> dict:
    limits = load_limits(limits_path) if limits_path else load_limits()
    rng = np.random.default_rng(seed)
    summary = {}
    for product in products:
        for part in backend.parts(product):
            path = out / SEED_DIR[scenario] / f"product_{product}.hdf5"
            stats = _record_part(backend, scenario, product, part, target, path, rng, limits, max_attempts,
                                 allow_touch_proxy)
            summary[f"{product}:{part}"] = stats
            _write_stats(path, part, scenario, stats)
            print(json.dumps({"product": product, "part": part,
                              **{k: v for k, v in stats.items() if k not in ("runs", "limits", "gate_max")}}))
    return summary


def _record_part(backend, scenario, product, part, target, path, rng, limits, max_attempts, allow_touch_proxy) -> dict:
    saved, attempts, skipped, reasons, runs = 0, 0, 0, Counter(), []
    while saved < target and attempts < max_attempts:
        attempts += 1
        installed = _installed_before(backend, product, part)
        if scenario == "fix_above":  # the part to remove is installed, with everything below it
            installed[part] = base_of(product)
        obs = _reset(backend, rng, product, scenario, installed, part)
        t0 = time.time()
        rec = EpisodeRecorder(backend.env, ForceGuard(limits), obs, allow_touch_proxy=allow_touch_proxy)
        try:
            _episode(backend, rec, rng, scenario, product, part, obs)
        except NoCase as e:
            attempts -= 1
            skipped += 1
            if skipped > 5 * max_attempts:
                raise RuntimeError(f"{scenario} for {part}: the layouts never produced the case ({e})") from e
            continue
        except (EpisodeFailed, GateFailed) as e:
            reasons[str(e).split(":")[0]] += 1
            runs.append({"ok": False, "reason": str(e), "gates": rec.gate_log.values, "peaks": rec.guard.episode_peaks(),
                         "events": rec.events, "s": round(time.time() - t0, 1)})
            continue
        attrs = {"backend": backend.name, "scenario": scenario, "seed": int(rng.integers(1 << 31)),
                 "limits_status": limits.status, "gates": rec.gate_log.values, "touch_gate": rec.gate_log.touch_method}
        if backend.name == "kinematic":
            attrs["KINEMATIC_DRY_RUN"] = True
        key = hdf5_io.append_demo(path, rec, attrs, f"T12-{backend.name}")
        saved += 1
        runs.append({"ok": True, "demo": key, "steps": len(rec.frames), "peaks": rec.guard.episode_peaks(),
                     "gates": rec.gate_log.values, "events": rec.events, "s": round(time.time() - t0, 1)})
    return {"attempts": attempts, "saved": saved, "success_rate": saved / attempts if attempts else None,
            "discarded": dict(reasons), "peak_max": _peak_max(runs), "gate_max": _gate_max(runs),
            "limits": limits.__dict__, "runs": runs}


def _gate_max(runs: list[dict]) -> dict[str, dict]:
    """Per gate over every attempt (kept and rejected): worst |value|, how often it was measured and failed."""
    out: dict[str, dict] = {}
    for r in runs:
        for name, g in r.get("gates", {}).items():
            s = out.setdefault(name, {"worst": 0.0, "measured": 0, "failed": 0, "limit": g["limit"]})
            s["worst"] = max(s["worst"], abs(g["value"]))
            s["measured"] += 1
            s["failed"] += 0 if g["pass"] else 1
    return out


def _reset(backend, rng, product, scenario, installed, target):
    if isinstance(backend, IsaacBackend):
        return backend.reset(rng, product, scenario, installed, target)
    return backend.reset(rng, product, scenario)


def _peak_max(runs: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in runs:
        for k, v in r.get("peaks", {}).items():
            out[k] = max(out.get(k, 0.0), v)
    return out


def _write_stats(path: Path, part: str, scenario: str, stats: dict) -> None:
    f = path.with_name(path.stem + "_stats.json")
    old = json.loads(f.read_text()) if f.exists() else {}
    old.setdefault(scenario, {})[part] = stats
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(old, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("isaac", "kinematic"), required=True)
    ap.add_argument("--scenario", choices=tuple(SEED_DIR), default="build")
    ap.add_argument("--products", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--target", type=int, default=10, help="clean demos to keep per part")
    ap.add_argument("--max-attempts", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEMO_ROOT)
    ap.add_argument("--limits", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cameras", nargs="*", default=[], choices=("table", "side", "wrist"),
                    help="render and store these cameras in the seed demos (default none)")
    ap.add_argument("--allow-touch-proxy", action="store_true",
                    help="accept the gripper_obj proxy for the 'touched only the target' gate when the env has no "
                         "sim.robot_touch (stamped on every demo as touch_gate)")
    a = ap.parse_args(argv)
    if a.backend == "kinematic" and a.out.resolve().is_relative_to(DEMO_ROOT.resolve()):
        ap.error("the kinematic dry run never writes under ~/t12_data/demos")
    if a.backend == "kinematic":
        backend = KinematicBackend()
    else:
        backend = IsaacBackend(a.device, tuple(a.products), tuple(a.cameras))
    products = [p for p in a.products if p in backend.products]
    try:
        record(backend, a.scenario, products, a.target, a.out, a.seed, a.max_attempts, a.limits, a.allow_touch_proxy)
    except LimitsError as e:
        ap.error(str(e))
    finally:
        if getattr(backend.env, "close", None) and backend.name == "isaac":
            backend.env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
