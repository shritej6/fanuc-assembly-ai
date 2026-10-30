"""Episode KPIs against the precision gates (sim/assembly_lab/INTERFACES.md) and physics/limits.yaml (numpy only).

Per step: peak axial / lateral / torque contact per object (obs["sim"]["contact_peak"], else "contact"), disturbed
non-target objects (> 1 mm or 2 deg since the current planner step began), wrong-object grips, and robot contact with
non-targets when the env reports it (obs["sim"]["robot_touch"] {uid: N}, agent 3's key; otherwise UNMEASURED,
never passed). Only objects on the table count (obs["sim"]["on_table"]; parked pool objects are ignored).
At the end: products built to the precision gates, defect area, test-bed spots, untouched decoys, Andon and order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import cad
from .director import spot_error

LIMITS_YAML = cad.REPO / "sim/assembly_lab/physics/limits.yaml"
OBJECTS_YAML = cad.REPO / "sim/assembly_lab/objects/objects.yaml"
FORCE_KEYS = (("normal_N", "axial_N"), ("lateral_N", "lateral_N"), ("torque_Nm", "torque_Nm"))


def load_limits(path: Path = LIMITS_YAML) -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    missing = [k for k in ("axial_N", "lateral_N", "torque_Nm") if not isinstance(raw.get(k), (int, float))]
    if missing:
        raise ValueError(f"{path}: no numeric {missing}")
    out = {"axial_N": float(raw["axial_N"]), "lateral_N": float(raw["lateral_N"]),
           "torque_Nm": float(raw["torque_Nm"]), "status": raw.get("status", "UNKNOWN"), "source": str(path)}
    out["per_object"] = per_object_limits(out)
    return out


def per_object_limits(global_lim: dict, path: Path = OBJECTS_YAML) -> dict:
    """objects.yaml max_force per object (e.g. M5_04 / M6_04 axial 2.5 N), never looser than limits.yaml."""
    objs = yaml.safe_load(Path(path).read_text())["objects"]
    out = {}
    for oid, o in objs.items():
        mf = o.get("max_force") or {}
        out[oid] = {k: min(float(mf.get(k, global_lim[k])), global_lim[k]) for k in ("axial_N", "lateral_N", "torque_Nm")}
    return out


def limit_for(limits: dict, uid: str, key: str) -> float:
    return limits.get("per_object", {}).get(cad.type_id(uid), limits)[key]


@dataclass
class EpisodeKPI:
    limits: dict
    initial: dict                                  # uid -> pose at reset
    peaks: dict = field(default_factory=lambda: {"normal_N": 0.0, "lateral_N": 0.0, "torque_Nm": 0.0})
    peak_obj: dict = field(default_factory=dict)   # force key -> uid with that peak
    damaged: dict = field(default_factory=dict)    # uid -> first over-limit reading
    disturbed: list = field(default_factory=list)
    wrong_grips: list = field(default_factory=list)
    jams: list = field(default_factory=list)       # parts stuck high on their mate (reported apart from damage)
    robot_contacts: list = field(default_factory=list)
    robot_contact_measured: bool = True
    _ref: dict = field(default_factory=dict)
    _step_key: tuple | None = None

    def on_step(self, obs: dict, task, t_s: float) -> None:
        sim = obs["sim"]
        self._forces(sim.get("contact_peak") or sim.get("contact") or {}, t_s)
        target = task.uid if task is not None else None
        key = (target, getattr(task, "started_s", None), getattr(task, "attempts", None))
        on_table = sim.get("on_table")
        poses = {u: p for u, p in sim["object_poses"].items() if on_table is None or on_table.get(u, True)}
        if key != self._step_key:  # a new planner step: non-targets are measured from here
            self._ref, self._step_key = {u: np.asarray(p, float) for u, p in poses.items()}, key
        for uid, pose in poses.items():
            if uid == target or uid not in self._ref:
                continue
            d = cad.pose_delta(self._ref[uid], pose)
            if d.mm > cad.DISTURB_MM or d.deg > cad.DISTURB_DEG:
                self.disturbed.append({"t": round(t_s, 2), "uid": uid, "mm": round(d.mm, 2), "deg": round(d.deg, 2),
                                       "during": target})
                self._ref[uid] = np.asarray(pose, float)  # count each new disturbance once
        held = cad.held_uid(obs)
        if held is not None and held != target:
            self.wrong_grips.append({"t": round(t_s, 2), "held": held, "target": target})
        if "robot_touch" not in sim:
            self.robot_contact_measured = False
        else:
            for uid, f in (sim["robot_touch"] or {}).items():
                if uid != target and float(f) > cad.TOUCH_N:
                    self.robot_contacts.append({"t": round(t_s, 2), "uid": uid, "N": float(f), "target": target})

    def _forces(self, contact: dict, t_s: float) -> None:
        for uid, c in contact.items():
            for k, lim_key in FORCE_KEYS:
                v = float(c.get(k, 0.0))
                if v > self.peaks[k]:
                    self.peaks[k], self.peak_obj[k] = v, uid
                lim = limit_for(self.limits, uid, lim_key)
                if v > lim and uid not in self.damaged:
                    self.damaged[uid] = {"t": round(t_s, 2), k: round(v, 3), "limit": lim}

    def summary(self) -> dict:
        return {"peak_axial_N": round(self.peaks["normal_N"], 3), "peak_lateral_N": round(self.peaks["lateral_N"], 3),
                "peak_torque_Nm": round(self.peaks["torque_Nm"], 4), "peak_objects": self.peak_obj,
                "limits": self.limits, "parts_damaged": len(self.damaged), "damaged": self.damaged,
                "disturbed": self.disturbed, "wrong_grips": self.wrong_grips, "jams": self.jams,
                "robot_contacts": self.robot_contacts if self.robot_contact_measured else "UNMEASURED",
                "collisions": len(self.disturbed) + len(self.wrong_grips) + len(self.robot_contacts)}


def jam_check(uid: str, base: str, poses: dict) -> dict | None:
    """A part caught on its mate: within the chamfer reach (xy <= 3 mm) but sitting above the seated height by more
    than the 0.3 mm gate and no more than the engagement depth + 2 mm (physics/limits.yaml: housings jam at > 0.5 mm
    offset). None when it is seated or nowhere near."""
    if uid not in poses or base not in poses:
        return None
    ok, err = cad.seat_check(uid, poses[uid], cad.seated_pose(uid, base, poses[base]))
    eng = float(cad.objects_yaml()[cad.type_id(uid)].get("insertion", {}).get("engagement_mm") or 0.0)
    if not ok and err["xy"] <= 3.0 and cad.SEAT_Z_MM < err["dz"] <= eng + 2.0:
        return {"uid": uid, "base": base, **{k: round(v, 3) for k, v in err.items()}}
    return None


def _seated_anywhere(uid: str, poses: dict) -> bool:
    return any(cad.seat_check(uid, poses[uid], cad.seated_pose(uid, b, poses[b]))[0]
               for b in poses if b.startswith("F"))


def built_check(prod: str, base: str, poses: dict) -> dict:
    """Every role of the product seated on `base` to the precision gates, by any instance of the right type."""
    parts = {}
    for pid in cad.product_parts(prod):
        best = None
        for uid in (u for u in poses if cad.type_id(u) == pid):
            ok, err = cad.seat_check(pid, poses[uid], cad.seated_pose(pid, base, poses[base]))
            if best is None or ok or err["xy"] < best["err"]["xy"]:
                best = {"uid": uid, "ok": ok, "err": {k: round(v, 3) for k, v in err.items()}}
            if ok:
                break
        parts[pid] = best or {"uid": None, "ok": False, "err": "no instance in the scene"}
    return {"ok": all(p["ok"] for p in parts.values()), "parts": parts}


def final_checks(expect: dict, poses: dict, initial: dict, events: list) -> dict:
    """Gate -> {ok, detail}; a gate that does not apply is left out."""
    g: dict = {}
    for prod, base in expect["built"].items():
        g[f"built_{prod}"] = built_check(prod, base, poses)
    zone, area = cad.cell_layout()["reject_zone"], cad.cell_layout()["work_area"]
    for uid in expect["in_defect_area"]:
        g[f"defect_area_{uid}"] = {"ok": uid in poses and cad.in_zone(poses[uid], zone)}
    for uid in expect["on_test_bed_upright"]:  # on its remembered spot, to the skills/gates.py spot gates
        spots = [e["spot"] for e in events if e["type"] == "step_done" and e["uid"] == uid and e.get("spot")]
        err = spot_error(uid, poses[uid], spots[-1]) if spots and uid in poses else None
        ok = (err is not None and err["xy"] <= cad.SPOT_XY_MM and err["yaw"] <= cad.SPOT_YAW_DEG
              and err["tilt"] <= cad.SPOT_TILT_DEG and cad.in_zone(poses[uid], area)
              and not _seated_anywhere(uid, poses))
        g[f"test_bed_{uid}"] = {"ok": ok, "err": err}
    for uid in expect["untouched"]:
        d = cad.pose_delta(initial[uid], poses[uid])
        g[f"untouched_{uid}"] = {"ok": d.mm <= cad.DISTURB_MM and d.deg <= cad.DISTURB_DEG, "mm": d.mm, "deg": d.deg}
    for uid in expect["not_installed"]:
        g[f"not_installed_{uid}"] = {"ok": not _seated_anywhere(uid, poses)}
    andons = [e for e in events if e["type"] == "andon"]
    g["andon"] = {"ok": bool(andons) == bool(expect["andon"]), "expected": expect["andon"], "raised": andons}
    done = [e for e in events if e["type"] == "step_done"]
    if expect.get("build_order_last") and done:
        last_prod = cad.type_id(done[-1]["uid"])[:2]
        g["build_order"] = {"ok": last_prod == expect["build_order_last"], "last": last_prod}
    if expect.get("move_before"):  # crowded: the blocker is moved before the blocked part is picked
        first, then = expect["move_before"]
        starts = [e["uid"] for e in events if e["type"] == "step_start"]
        ok = first in starts and (then not in starts or starts.index(first) < starts.index(then))
        g["blocker_first"] = {"ok": ok and then in starts, "order": starts}
    grips = [e["grip_error"] for e in done if e.get("grip_error")]
    g["grip_on_pad"] = {"ok": all(e["xyz_mm"] <= cad.GRIP_XYZ_MM and e["roll_deg"] <= cad.GRIP_ROLL_DEG
                                  for e in grips), "worst": max(grips, key=lambda e: e["xyz_mm"]) if grips else None}
    for name in expect.get("events", []):
        g[f"event_{name}"] = {"ok": any(e["type"] == name for e in events)}
    return g
