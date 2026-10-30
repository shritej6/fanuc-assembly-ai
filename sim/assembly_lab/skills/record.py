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
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import episodes, frames, hdf5_io
from .backends import KinematicBackend, NoCase, base_of, sheet_parts
from .force_guard import ForceGuard, LimitsError, load_limits
from .gates import GateFailed
from .motion import MotionError
from .runner import EpisodeFailed, EpisodeRecorder, ResetInvalid

DEMO_ROOT = Path.home() / "t12_data" / "demos"
SEED_DIR = {"build": "pick_insert", "wont_seat": "pick_insert", "dropped": "pick_insert",
            "fix_above": "pull_out", "crowded": "place_free", "defect": "defect_drop"}
STAGED_BEYOND_CAPTURE_MM = 1.0  # the staged first try misses by this much more than the part's chamfer capture


def isaac_out_path(out: Path, scenario: str, product: int, seed: int) -> Path:
    """The Isaac recorder's HDF5 for one job. Scenario and seed are in the name: build and wont_seat share the
    pick_insert dir, and two such jobs that start in the same second must not open the same file (run10 lock clash)."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return out / SEED_DIR[scenario] / f"product_{product}_{scenario}_{stamp}_s{seed}.hdf5"


def select_parts(parts: list[str], only: list[str] | None) -> list[str]:
    """`parts` in sheet order, or just `only` (each must be one of them: a typo must not record nothing)."""
    if only is None:
        return list(parts)
    unknown = [p for p in only if p not in parts]
    if unknown:
        raise ValueError(f"--parts {unknown} are not among {list(parts)}")
    return [p for p in parts if p in only]


def part_path(path: Path, part: str) -> Path:
    """One part's recorder file next to the job's base name (scenario + seed + part: unique per job and part)."""
    return path.with_name(f"{path.stem}_{part}{path.suffix}")


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


@dataclass
class Run:
    """Settings shared by every attempt of one recording run."""

    limits: object
    per_object: dict
    max_attempts: int
    allow_touch_proxy: bool = False
    measure_only: bool = False
    path: Path | None = None  # the HDF5 this run writes (kinematic: hdf5_io; isaac: the Isaac Lab recorder's file)
    metas: list = field(default_factory=list)  # one per recorder episode, in order (isaac)


def per_object_limits() -> dict:
    """objects.yaml max_force per object id (agent 2; e.g. M5_04 / M6_04 axial 2.5 N, SIM_MEASURED)."""
    return {oid: {k: v for k, v in (e.get("max_force") or {}).items() if k in ("axial_N", "lateral_N", "torque_Nm")}
            for oid, e in frames.catalog().items()}


def record(backend, scenario: str, products: list[int], target: int, out: Path, seed: int, max_attempts: int,
           limits_path: Path | None = None, allow_touch_proxy: bool = False, measure_only: bool = False,
           path: Path | None = None, parts: list[str] | None = None) -> dict:
    limits = load_limits(limits_path) if limits_path else load_limits()
    run = Run(limits, per_object_limits(), max_attempts, allow_touch_proxy, measure_only)
    rng = np.random.default_rng(seed)
    summary = {}
    for product in products:
        for i, part in enumerate(select_parts(backend.parts(product), parts)):
            if not hasattr(backend, "rotate"):  # kinematic: hdf5_io appends every part to one file
                run.path = path or out / SEED_DIR[scenario] / f"product_{product}.hdf5"
            else:  # isaac: one closed file per part, the recorder moves on to the next part's file
                run.path = part_path(path, part)
                if i > 0:
                    backend.rotate(run.path)
                    run.metas = []
            stats = _record_part(backend, scenario, product, part, target, rng, run)
            summary[f"{product}:{part}"] = stats
            _write_stats(run.path, part, scenario, stats)
            print(json.dumps({"product": product, "part": part,
                              **{k: v for k, v in stats.items() if k not in ("runs", "limits", "gate_max")}},
                             default=_jsonable), flush=True)
    return summary


def _record_part(backend, scenario, product, part, target, rng, run: Run) -> dict:
    limits, max_attempts = run.limits, run.max_attempts
    saved, attempts, skipped, invalid, reasons, runs = 0, 0, 0, 0, Counter(), []
    invalid_run, aborted = 0, None
    while saved < target and attempts < max_attempts:
        attempts += 1
        installed = _installed_before(backend, product, part)
        if scenario == "fix_above":  # the part to remove is installed, with everything below it
            installed[part] = base_of(product)
        t0 = time.time()
        rec = None
        try:
            obs = _reset(backend, rng, product, scenario, installed, part)  # NoCase: the layout lacks the case
            rec = EpisodeRecorder(backend.env, ForceGuard(limits, per_object=run.per_object), obs,
                                  allow_touch_proxy=run.allow_touch_proxy, measure_only=run.measure_only)
            _episode(backend, rec, rng, scenario, product, part, obs)
        except ResetInvalid as e:
            attempts -= 1
            invalid += 1
            runs.append({"ok": False, "status": "reset_invalid", "reason": str(e),
                         "steps": len(rec.frames) if rec else 0,
                         "pre_touch_peaks": rec.guard.pre_touch_peaks if rec else {}, "s": round(time.time() - t0, 1)})
            if rec is not None:
                _finish(backend, run, rec, False, scenario, f"reset_invalid: {e}")
            elif backend.name == "isaac":
                backend.finish(False)
            invalid_run += 1
            if invalid_run > MAX_INVALID_IN_A_ROW:
                aborted = f"{invalid_run} invalid resets in a row ({e})"
                print(f"[record] {scenario} {part}: stopping this part: {aborted}", flush=True)
                break
            continue
        except NoCase as e:
            attempts -= 1
            skipped += 1
            if skipped > 5 * max_attempts:
                aborted = f"the layouts never produced the case in {skipped} resets ({e})"
                print(f"[record] {scenario} {part}: stopping this part: {aborted}", flush=True)
                break
            continue
        except (EpisodeFailed, GateFailed, MotionError) as e:  # MotionError: the plan breaks a motion rule
            invalid_run = 0
            reasons[str(e).split(":")[0]] += 1
            runs.append({"ok": False, "status": "robot_failed", "reason": str(e), "steps": len(rec.frames),
                         "gates": rec.gate_log.values, "peaks": rec.guard.episode_peaks(),
                         "pre_touch_peaks": rec.guard.pre_touch_peaks, "events": rec.events,
                         "seat_tries": rec.seat_tries, "twist_follow_deg": rec.twist_follow_deg, "in_hand_drift_mm": rec.in_hand_drift_mm, "others_trace": rec.others_trace, "seat_contacts": rec.seat_contacts, "substep_trace": rec.substep_trace, "level_corrections": rec.level_corrections, "settled": rec.settled,
                         "s": round(time.time() - t0, 1)})
            _finish(backend, run, rec, False, scenario, str(e))
            continue
        invalid_run = 0
        key = _finish(backend, run, rec, True, scenario, "ok")
        saved += 1
        runs.append({"ok": True, "status": "kept" if not run.measure_only else "would_keep", "demo": key,
                     "steps": len(rec.frames), "peaks": rec.guard.episode_peaks(), "gates": rec.gate_log.values,
                     "pre_touch_peaks": rec.guard.pre_touch_peaks, "events": rec.events, "settled": rec.settled,
                     "seat_tries": rec.seat_tries, "twist_follow_deg": rec.twist_follow_deg, "in_hand_drift_mm": rec.in_hand_drift_mm, "others_trace": rec.others_trace, "seat_contacts": rec.seat_contacts, "substep_trace": rec.substep_trace, "level_corrections": rec.level_corrections,
                     "s": round(time.time() - t0, 1)})
    return {"attempts": attempts, "kept_or_would_keep": saved, "robot_failed": attempts - saved,
            "reset_invalid": invalid, "planner_skipped": skipped,
            "success_rate": saved / attempts if attempts else None,
            "discarded": dict(reasons), "peak_max": _peak_max(runs), "gate_max": _gate_max(runs),
            "seat": _seat_summary(runs), "level": _level_summary(runs), "aborted": aborted,
            "limits": limits.__dict__, "runs": runs}


def _level_summary(runs: list[dict]) -> dict:
    """Distribution of the measured in-hand tilt and of the TCP level correction applied, per insert try."""
    out = {}
    for name in ("in_hand_tilt_deg", "level_deg"):
        for k in (1, 2):
            vals = sorted(c[name] for r in runs for c in r.get("level_corrections", []) if c["try"] == k)
            if vals:
                out[f"{name}_try{k}"] = {"n": len(vals), "min": vals[0], "median": vals[len(vals) // 2],
                                         "max": vals[-1]}
    return out


def _seat_summary(runs: list[dict]) -> dict:
    """Measured seat numbers of every attempt that reached the seat check (kept or not)."""
    out = {}
    for name in ("seat_xy_mm", "seat_yaw_deg", "seat_z_mm", "seat_tilt_deg"):
        vals = [r["gates"][name]["value"] for r in runs if name in r.get("gates", {})]
        if vals:
            out[name] = {"n": len(vals), "min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}
    return out


def _touch_gate(rec: EpisodeRecorder, run: Run) -> str:
    method = rec.gate_log.touch_method
    if method != "robot_touch":
        return "missing" if run.measure_only else method
    return "robot_touch_unverified" if run.measure_only else "robot_touch"


def _finish(backend, run: Run, rec: EpisodeRecorder, ok: bool, scenario: str, reason: str) -> str | None:
    """Kinematic: append kept episodes with hdf5_io. Isaac: mark the recorder episode and log its metadata (in order,
    kept or not: every non-empty attempt is one recorder episode)."""
    attrs = {"backend": backend.name, "scenario": scenario, "limits_status": run.limits.status,
             "gates": rec.gate_log.values, "touch_gate": _touch_gate(rec, run), "measure_only": run.measure_only,
             "kept": bool(ok and not run.measure_only), "reason": reason}
    if backend.name == "kinematic":
        if not ok:
            return None
        return hdf5_io.append_demo(run.path, rec, attrs | {"KINEMATIC_DRY_RUN": True}, "T12-kinematic")
    backend.finish(ok and not run.measure_only)
    if not rec.frames:
        return None
    meta = attrs | {k: v for k, v in rec.meta.items()} | {
        "steps": len(rec.frames), "instruction": rec.instructions[0] if rec.instructions else "",
        "marks": rec.marks, "events": rec.events, "peaks": rec.guard.episode_peaks(),
        "train_mask": [bool(f.seg.learn) for f in rec.frames], "would_keep": bool(ok),
        "seat_tries": rec.seat_tries, "twist_follow_deg": rec.twist_follow_deg, "in_hand_drift_mm": rec.in_hand_drift_mm, "others_trace": rec.others_trace, "seat_contacts": rec.seat_contacts, "substep_trace": rec.substep_trace, "level_corrections": rec.level_corrections}
    run.metas.append(meta)
    with open(str(run.path) + ".meta.jsonl", "a") as f:
        f.write(json.dumps(meta, default=_jsonable) + "\n")
    return f"demo_{len(run.metas) - 1}"


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
    if backend.name == "isaac":
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
    f.write_text(json.dumps(old, indent=1, default=_jsonable))


def _jsonable(v):
    """numpy / torch scalars (np.bool_, np.float32, 0-d tensors) as plain Python values."""
    if hasattr(v, "item"):
        return v.item()
    if hasattr(v, "tolist"):
        return v.tolist()
    raise TypeError(f"{type(v).__name__} is not JSON serialisable")


MAX_INVALID_IN_A_ROW = 30  # resets that never settle: the env is the problem, stop this part and report it

MEASURE_ROOT = Path.home() / "t12_data" / "measure"


def _args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("isaac", "kinematic"), required=True)
    ap.add_argument("--scenario", choices=tuple(SEED_DIR), default="build")
    ap.add_argument("--products", type=int, nargs="+", default=[1])
    ap.add_argument("--parts", nargs="+", default=None,
                    help="record only these parts of the product (short jobs that hand over cleanly); default all")
    ap.add_argument("--target", type=int, default=10, help="clean demos to keep per part")
    ap.add_argument("--max-attempts", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="default ~/t12_data/demos (~/t12_data/measure if "
                                                          "--measure-only)")
    ap.add_argument("--limits", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cameras", nargs="*", default=[], choices=("table", "side", "wrist"),
                    help="render and store these cameras in the seed demos (default none)")
    ap.add_argument("--measure-only", action="store_true",
                    help="robot_touch not verified yet: log every gate (touch included) but keep nothing; writes "
                         "under ~/t12_data/measure, episodes marked touch_gate=missing / robot_touch_unverified")
    ap.add_argument("--allow-touch-proxy", action="store_true",
                    help="accept the gripper_obj proxy for the 'touched only the target' gate (stamped on every demo)")
    a = ap.parse_args(argv)
    a.out = a.out or (MEASURE_ROOT if a.measure_only else DEMO_ROOT)
    if a.measure_only and a.out.resolve().is_relative_to(DEMO_ROOT.resolve()):
        ap.error("measure-only runs never write under ~/t12_data/demos")
    if a.backend == "kinematic" and a.out.resolve().is_relative_to(DEMO_ROOT.resolve()):
        ap.error("the kinematic dry run never writes under ~/t12_data/demos")
    if a.backend == "isaac" and len(a.products) != 1:
        ap.error("the isaac backend records one product per process (one env, one recorder file)")
    return ap, a


def main(argv=None) -> int:
    ap, a = _args(argv)
    path = None
    if a.backend == "kinematic":
        backend = KinematicBackend()
    else:
        from .isaac_backend import IsaacBackend

        path = isaac_out_path(a.out, a.scenario, a.products[0], a.seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        first = part_path(path, select_parts(sheet_parts(a.products[0]), a.parts)[0])
        backend = IsaacBackend(a.scenario, a.products[0], str(first), a.device, tuple(a.cameras))
    products = [p for p in a.products if p in backend.products]
    try:
        record(backend, a.scenario, products, a.target, a.out, a.seed, a.max_attempts, a.limits,
               a.allow_touch_proxy, a.measure_only, path, a.parts)
    except LimitsError as e:
        ap.error(str(e))
    except Exception:
        traceback.print_exc()  # before the app shuts down: SimulationApp.close() ends the process
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        if backend.name == "isaac":
            backend.close(backend.record_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
