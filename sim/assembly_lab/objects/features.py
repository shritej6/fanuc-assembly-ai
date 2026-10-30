"""Pegs, holes and mates of the 30 printed objects, from the CAD that was printed (needs cadquery: cad group).

Every number comes from cad/products/parameters.py or a model module's constants, i.e. the inputs of the `build()`
that wrote the STLs. Positions are in the object's installed local frame (origin = centre of its lowest seating
plane, the product frame shifted by `installed_origin_mm`). `check_features` then tests each feature against the
printed mesh (a peg centre must be inside material, a hole centre must be empty), so a CAD change that moves a
feature fails the build instead of silently writing a wrong position.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "cad" / "products"))

import control_family as KF
import parameters as P

MODULES = {1: "model_01_rotary", 2: "model_02_indicator", 3: "model_03_pushbutton",
           4: "model_04_sensor_connector", 5: "model_05_io_connector", 6: "model_06_power_connector"}
BASE_TYPE = {1: "A", 2: "A", 3: "B", 4: "C", 5: "C", 6: "C"}


def _round(name, d, xy, z0, h, chamfer, kind):
    return {"name": name, "shape": "round", "diameter_mm": d, "xy_mm": list(xy), "z0_mm": z0, "height_mm": h,
            "chamfer_mm": chamfer, "kind": kind}


def _rect(name, size, xy, z0, h, chamfer, kind, note=None):
    out = {"name": name, "shape": "rect", "size_mm": list(size), "xy_mm": list(xy), "z0_mm": z0, "height_mm": h,
           "chamfer_mm": chamfer, "kind": kind}
    if note:
        out["note"] = note
    return out


def _mate(with_ids, pairs, clearance, engagement, capture, seat, note=None):
    out = {"with": with_ids, "features": pairs, "clearance_per_side_mm": clearance, "engagement_mm": engagement,
           "capture_mm": capture, "seats_on": seat}
    if note:
        out["note"] = note
    return out


def housing_dims(n: int) -> tuple[float, float]:
    mod = importlib.import_module(MODULES[n])
    return (P.CONN_HOUSING_X, P.CONN_HOUSING_Y) if n >= 4 else (mod.HOUSING_X, mod.HOUSING_Y)


def fixture_features(n: int) -> dict:
    hx, hy = housing_dims(n)
    c, side_cx = P.LOCATING_CLEARANCE_PER_SIDE, hx / 2 + P.SIDE_STOP_GAP
    pegs = [_round("peg_A", P.PEG_A_D, (P.PEG_A_X, P.PEG_A_Y), 0.0, P.PEG_HEIGHT, P.PEG_TOP_CHAMFER, "peg"),
            _round("peg_B", P.PEG_B_D, (P.PEG_B_X, P.PEG_B_Y), 0.0, P.PEG_HEIGHT, P.PEG_TOP_CHAMFER, "peg")]
    stops = [_rect(f"side_stop_{s}", (P.SIDE_STOP_X, P.SIDE_STOP_Y), (sx * side_cx, P.SIDE_STOP_CENTER_Y), 0.0,
                   P.STOP_HEIGHT, 0.0, "stop") for s, sx in (("left", -1), ("right", 1))]
    stops.append({"name": "front_L_stops", "kind": "stop", "inner_corner_xy_mm": [[-(hx / 2 + c), -(hy / 2 + c)],
                  [hx / 2 + c, -(hy / 2 + c)]], "wall_mm": P.FRONT_STOP_WALL, "leg_mm": P.FRONT_STOP_LEG,
                  "height_mm": P.STOP_HEIGHT})
    return {"pegs": pegs + stops, "holes": [], "mates_with": [], "housing_opening_mm": [hx + 2 * c, hy + 2 * c]}


def _sockets() -> list[dict]:
    return [_round("socket_A", P.SOCKET_A_D, (P.PEG_A_X, P.PEG_A_Y), 0.0, P.SOCKET_DEPTH, P.SOCKET_MOUTH_CHAMFER,
                   "hole_from_below"),
            _round("socket_B", P.SOCKET_B_D, (P.PEG_B_X, P.PEG_B_Y), 0.0, P.SOCKET_DEPTH, P.SOCKET_MOUTH_CHAMFER,
                   "hole_from_below")]


def _base_mate(n: int) -> dict:
    return _mate(f"base_type_{BASE_TYPE[n]}", ["socket_A<-peg_A", "socket_B<-peg_B", "footprint<-stops"],
                 P.LOCATING_CLEARANCE_PER_SIDE, P.STOP_HEIGHT, P.LOCATING_CLEARANCE_PER_SIDE, "base top z 0",
                 "the 3 mm stops engage before the 2.5 mm pegs and have no chamfer: lateral capture is the 0.5 mm "
                 "clearance only")


def control_housing_features(n: int, ids: dict) -> dict:
    mod = importlib.import_module(MODULES[n])
    depth = P.CTRL_HOUSING_Z - P.CTRL_POCKET_FLOOR_Z
    if n == 3:
        pocket = _rect("pocket", (mod.POCKET_X, mod.POCKET_Y), (0, 0), P.CTRL_POCKET_FLOOR_Z, depth, P.ENTRY_CHAMFER,
                       "hole_from_above", "key block fills the rear-right corner (anti-180)")
    else:
        pocket = _round("pocket", 19.0, (0, 0), P.CTRL_POCKET_FLOOR_Z, depth, P.ENTRY_CHAMFER, "hole_from_above")
    slots = [_rect(f"guard_socket_{s}", (P.GUARD_SOCKET_X, P.GUARD_SOCKET_Y), (sx * mod.GUARD_ARM_CX, P.GUARD_CENTER_Y),
                   P.GUARD_SOCKET_BOTTOM_Z, P.CTRL_HOUSING_Z - P.GUARD_SOCKET_BOTTOM_Z, 0.0, "hole_from_above")
             for s, sx in (("left", -1), ("right", 1))]
    mates = [_base_mate(n),
             _mate(ids[2], ["pocket<-plug"], P.LOCATING_CLEARANCE_PER_SIDE, depth,
                   P.ENTRY_CHAMFER + P.CARRIER_BOTTOM_CHAMFER + P.LOCATING_CLEARANCE_PER_SIDE, "pocket floor z 7"),
             _mate(ids[4], ["guard_socket_left<-arm_left", "guard_socket_right<-arm_right"],
                   P.LOCATING_CLEARANCE_PER_SIDE, P.CTRL_HOUSING_Z - P.GUARD_SOCKET_BOTTOM_Z,
                   P.GUARD_FOOT_CHAMFER + P.LOCATING_CLEARANCE_PER_SIDE, "socket bottom z 4")]
    return {"pegs": [], "holes": _sockets() + [pocket] + slots, "mates_with": mates}


def carrier_features(n: int, ids: dict) -> dict:
    mod = importlib.import_module(MODULES[n])
    plug_h = P.CARRIER_FLANGE_Z0 - P.CTRL_POCKET_FLOOR_Z
    post_z0 = KF.CAP_Z0 - P.CTRL_POCKET_FLOOR_Z
    if n == 3:
        plug = _rect("plug", (mod.STEM_X, mod.STEM_Y), (0, 0), 0.0, plug_h, P.CARRIER_BOTTOM_CHAMFER, "peg",
                     f"{mod.RELIEF} mm relieved corner at +X +Y clears the pocket key block")
    else:
        plug = _round("plug", 18.0, (0, 0), 0.0, plug_h, P.CARRIER_BOTTOM_CHAMFER, "peg")
        plug["note"] = "1 mm front flat at y = -8"
    post = _rect("key_post", (P.KEY_POST_X, P.KEY_POST_Y), (0, 0), post_z0, mod.POST_H, P.GRIP_TOP_CHAMFER, "peg")
    cap_capture = P.SOCKET_ENTRY_CHAMFER + P.GRIP_TOP_CHAMFER + P.LOCATING_CLEARANCE_PER_SIDE
    mates = [_mate(ids[1], ["plug->pocket"], P.LOCATING_CLEARANCE_PER_SIDE, plug_h,
                   P.ENTRY_CHAMFER + P.CARRIER_BOTTOM_CHAMFER + P.LOCATING_CLEARANCE_PER_SIDE, "pocket floor z 7"),
             _mate(ids[3], ["key_post->socket"], P.LOCATING_CLEARANCE_PER_SIDE, mod.POST_H, cap_capture,
                   "cap seats on this flange")]
    return {"pegs": [plug, post], "holes": [], "mates_with": mates}


def cap_features(n: int, ids: dict) -> dict:
    mod = importlib.import_module(MODULES[n])
    sock = _rect("socket", (P.KEY_POST_X + 1.0, P.KEY_POST_Y + 1.0), (0, 0), 0.0, mod.POST_H + P.SOCKET_HEADROOM,
                 P.SOCKET_ENTRY_CHAMFER, "hole_from_below")
    capture = P.SOCKET_ENTRY_CHAMFER + P.GRIP_TOP_CHAMFER + P.LOCATING_CLEARANCE_PER_SIDE
    return {"pegs": [], "holes": [sock], "mates_with": [
        _mate(ids[2], ["socket<-key_post"], P.LOCATING_CLEARANCE_PER_SIDE, mod.POST_H, capture,
              "carrier flange top z 18 (0.5 mm socket headroom above the post)")]}


def guard_features(n: int, ids: dict) -> dict:
    mod = importlib.import_module(MODULES[n])
    eng = P.CTRL_HOUSING_Z - P.GUARD_SOCKET_BOTTOM_Z
    arms = [_rect(f"arm_{s}", (P.GUARD_ARM_X, P.GUARD_ARM_Y), (sx * mod.GUARD_ARM_CX, 0.0), 0.0, eng,
                  P.GUARD_FOOT_CHAMFER, "peg") for s, sx in (("left", -1), ("right", 1))]
    return {"pegs": arms, "holes": [], "mates_with": [
        _mate(ids[1], ["arm_left->guard_socket_left", "arm_right->guard_socket_right"], P.LOCATING_CLEARANCE_PER_SIDE,
              eng, P.GUARD_FOOT_CHAMFER + P.LOCATING_CLEARANCE_PER_SIDE, "socket bottom z 4")]}


def connector_housing_features(n: int, ids: dict) -> dict:
    mod = importlib.import_module(MODULES[n])
    depth = P.CONN_HOUSING_Z - P.CONN_FLOOR_Z
    cavity = _rect("cavity", (mod.CAV_X, mod.CAV_Y), (0, 0), P.CONN_FLOOR_Z, depth, 0.0, "hole_from_above",
                   "open to the front; no entry chamfer (the insert carries a 1.5 mm bottom lead-in)")
    if n == 4:
        cavity["guide_ribs"] = [{"side": s, "width_y_mm": w, "center_y_mm": y, "projection_mm": 1.0}
                                for s, w, y in mod.GUIDE_RIBS]
    pegs = [_rect(f"guide_peg_{s}", (P.GUIDE_PEG, P.GUIDE_PEG), (sx * P.GUIDE_HOLE_X, P.GUIDE_HOLE_Y),
                  P.CONN_HOUSING_Z, P.GUIDE_PEG_H, P.SMALL_LEAD_IN, "peg") for s, sx in (("left", -1), ("right", 1))]
    mates = [_base_mate(n),
             _mate(ids[2], ["cavity<-insert_body"], P.LOCATING_CLEARANCE_PER_SIDE, depth,
                   P.INSERT_BOTTOM_CHAMFER + P.LOCATING_CLEARANCE_PER_SIDE, "cavity floor z 5"),
             _mate(ids[3], ["guide_peg_left->guide_hole_left", "guide_peg_right->guide_hole_right"],
                   P.LOCATING_CLEARANCE_PER_SIDE, P.GUIDE_PEG_H, 2 * P.SMALL_LEAD_IN + P.LOCATING_CLEARANCE_PER_SIDE,
                   "housing top z 18"),
             _mate(ids[4], ["top<-lock_legs"], None, 0.0, None, "lock legs stand on the housing top z 18")]
    return {"pegs": pegs, "holes": _sockets() + [cavity], "mates_with": mates}


def insert_features(n: int, ids: dict) -> dict:
    mod = importlib.import_module(MODULES[n])
    body = _rect("insert_body", (mod.INSERT_X, mod.INSERT_Y), (0, 0), 0.0, P.INSERT_H, P.INSERT_BOTTOM_CHAMFER, "peg")
    if n == 4:
        body["note"] = "open-bottom grooves ride on the housing guide ribs (both at y +4): blocks a 180 deg turn"
    eng = P.CONN_HOUSING_Z - P.CONN_FLOOR_Z
    return {"pegs": [body], "holes": [], "mates_with": [
        _mate(ids[1], ["insert_body->cavity"], P.LOCATING_CLEARANCE_PER_SIDE, eng,
              P.INSERT_BOTTOM_CHAMFER + P.LOCATING_CLEARANCE_PER_SIDE, "cavity floor z 5")]}


def guide_features(n: int, ids: dict) -> dict:
    holes = [_rect(f"guide_hole_{s}", (P.GUIDE_HOLE, P.GUIDE_HOLE), (sx * P.GUIDE_HOLE_X, P.GUIDE_HOLE_Y), 0.0,
                   P.GUIDE_T, P.SMALL_LEAD_IN, "hole_from_below") for s, sx in (("left", -1), ("right", 1))]
    holes += [_rect(f"lock_hole_{s}", (P.LOCK_HOLE + 0.5, P.LOCK_HOLE), (sx * (P.LOCK_X - 0.25), P.LOCK_Y), 0.0,
                    P.GUIDE_T, 0.0, "through", "4 x 4 hole extended 0.5 mm into the U opening")
              for s, sx in (("left", -1), ("right", 1))]
    return {"pegs": [], "holes": holes, "mates_with": [
        _mate(ids[1], ["guide_hole_left<-guide_peg_left", "guide_hole_right<-guide_peg_right"],
              P.LOCATING_CLEARANCE_PER_SIDE, P.GUIDE_PEG_H, 2 * P.SMALL_LEAD_IN + P.LOCATING_CLEARANCE_PER_SIDE,
              "housing top z 18"),
        _mate(ids[4], ["lock_hole_left<-leg_left", "lock_hole_right<-leg_right"], P.LOCATING_CLEARANCE_PER_SIDE,
              P.GUIDE_T, P.SMALL_LEAD_IN + P.LOCATING_CLEARANCE_PER_SIDE, "lock bridge on the guide top z 23")]}


def lock_features(n: int, ids: dict) -> dict:
    legs = [_rect(f"leg_{s}", (P.LOCK_LEG, P.LOCK_LEG), (sx * P.LOCK_X, 0.0), 0.0, P.GUIDE_T, P.SMALL_LEAD_IN, "peg")
            for s, sx in (("left", -1), ("right", 1))]
    return {"pegs": legs, "holes": [], "mates_with": [
        _mate(ids[3], ["leg_left->lock_hole_left", "leg_right->lock_hole_right"], P.LOCATING_CLEARANCE_PER_SIDE,
              P.GUIDE_T, P.SMALL_LEAD_IN + P.LOCATING_CLEARANCE_PER_SIDE, "legs on the housing top z 18"),
        _mate(ids[1], ["lock_legs->top"], None, 0.0, None, "legs stand on the housing top z 18")]}


PART_FEATURES = {1: (control_housing_features, connector_housing_features), 2: (carrier_features, insert_features),
                 3: (cap_features, guide_features), 4: (guard_features, lock_features)}


def object_features(n: int, role: int, ids: dict) -> dict:
    """role 0 = base, 1..4 = sheet step; ids = {0..4: object id} of product n."""
    if role == 0:
        return fixture_features(n)
    control, connector = PART_FEATURES[role]
    return (control if n <= 3 else connector)(n, ids)


def _inside(tri: np.ndarray, point: np.ndarray) -> bool:
    """Point-in-mesh by vertical ray parity (point nudged off symmetry lines so the ray misses edges)."""
    p = point + np.array([0.0123, 0.0071, 0.0])
    a, b, c = tri[:, 0, :2], tri[:, 1, :2], tri[:, 2, :2]
    d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(d) > 1e-12
    d = np.where(ok, d, 1.0)
    l1 = ((b[:, 1] - c[:, 1]) * (p[0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (p[1] - c[:, 1])) / d
    l2 = ((c[:, 1] - a[:, 1]) * (p[0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (p[1] - c[:, 1])) / d
    l3 = 1 - l1 - l2
    hit = ok & (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
    z = l1 * tri[:, 0, 2] + l2 * tri[:, 1, 2] + l3 * tri[:, 2, 2]
    return int(np.count_nonzero(hit & (z > p[2]))) % 2 == 1


def check_features(name: str, tri: np.ndarray, feats: dict) -> None:
    """Peg centres must be inside the printed mesh and hole centres must be empty, at mid height."""
    for f in feats["pegs"] + feats["holes"]:
        if "xy_mm" not in f or f["kind"] == "stop" and f["name"].startswith("front"):
            continue
        point = np.array([*f["xy_mm"], f["z0_mm"] + f["height_mm"] / 2])
        want_material = f["kind"] in ("peg", "stop")
        if _inside(tri, point) != want_material:
            raise ValueError(f"{name}: feature {f['name']} at {point.tolist()} does not match the printed mesh")


__all__ = ["BASE_TYPE", "MODULES", "check_features", "housing_dims", "object_features"]
