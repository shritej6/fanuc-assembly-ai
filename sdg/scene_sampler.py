"""Deterministic scene specs for the Isaac Sim Replicator generator.

A scene spec is plain JSON: which 3 products are on the table, where every base, part, spare, reject and foreign
object sits (4x4 transform, UF0 mm), which parts are defective, the camera / light / material draws and, for wrist
sets, the TCP pose the arm moves to. The same (set, index) always gives the same spec, so any chunk can be re-run on
Rosie and produce identical labels. No Isaac imports here: this runs and is tested on any laptop.

    uv run python -m sdg.scene_sampler --set A --chunk 0 --out specs/A/chunk_0000.jsonl
    uv run python -m sdg.scene_sampler --set A --n-chunks
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from sdg.transforms import about_point, footprint, rot_axis, rot_z, translation, world_aabb, yaw_deg
from sdg.wrist_ik import reachable

REPO = Path(__file__).resolve().parents[1]
SPEC_VERSION = 1
MAX_TRIES = 400
FOREIGN_RADIUS_MM = {"screw_m3": 9.0, "tape_roll": 26.0, "cube_10mm": 8.0}
TCP_BOX = {"x": (150.0, 650.0), "y": (-450.0, 450.0), "z": (-334.0, 300.0)}
REST_MARGIN_MM = 0.05   # a mis-built part rests this far above the contact found by sdg/tools/build_clearance.py
MAX_REST_LIFT_MM = 15.0   # higher than this the part would be balanced on a point: such a yaw is not used


class SceneSamplingError(RuntimeError):
    """A scene could not be laid out within MAX_TRIES (zones too small for the requested objects)."""


@dataclass(frozen=True)
class Inputs:
    layout: dict
    sets: dict
    geometry: dict
    taxonomy: dict
    clearance: dict

    @classmethod
    def load(cls, repo: Path = REPO) -> Inputs:
        return cls(
            layout=yaml.safe_load((repo / "sdg/configs/cell_layout.yaml").read_text()),
            sets=yaml.safe_load((repo / "sdg/configs/sets.yaml").read_text()),
            geometry=json.loads((repo / "sdg/data/object_geometry.json").read_text())["objects"],
            taxonomy={o["object_id"]: o for o in json.loads((repo / "contracts/taxonomy.json").read_text())["objects"]},
            clearance=json.loads((repo / "sdg/data/build_clearance.json").read_text()),
        )

    def product_parts(self, product: int) -> list[str]:
        ids = [k for k in self.geometry if k.startswith(f"M{product}_")]
        return sorted(ids, key=lambda k: self.geometry[k]["install_step"])

    def family_mates(self, object_id: str) -> list[str]:
        fam, prod = self.taxonomy[object_id]["family"], self.taxonomy[object_id]["product"]
        return sorted(k for k, o in self.taxonomy.items() if o["family"] == fam and o["product"] != prod)


def scene_seed(set_seed: int, index: int) -> int:
    return set_seed * 1_000_003 + index


class SceneBuilder:
    """Lays out one scene. Use `sample_scene`; the builder is single-use."""

    def __init__(self, inputs: Inputs, set_name: str, index: int):
        self.inp, self.set_name, self.index = inputs, set_name, index
        self.cfg = inputs.sets["sets"][set_name]
        self.sc = self.cfg["scene"]
        self.defaults = inputs.sets["defaults"]
        self.seed = scene_seed(self.cfg["seed"], index)
        self.rng = random.Random(self.seed)
        self.table_z = inputs.layout["table_z"]
        self.objects: list[dict] = []
        self._occupied: list[tuple[np.ndarray, float, str]] = []

    # ------------------------------------------------------------------ top level
    def build(self) -> dict:
        target_id = self._planned_target()
        products = self._choose_products(target_id)
        steps = {p: self._choose_step(p, target_id) for p in products}
        self._add_spares()                                    # fixed slots first, then bases (they need the most room)
        bases = {p: self._place_base(p) for p in products}
        for p in products:
            self._place_parts(p, steps[p], bases[p])
        self._add_trap(products)
        self._add_rejects()
        self._add_foreign()
        target = self._resolve_target(target_id)
        self._apply_defects(target)
        if "build_states" in self.cfg:
            self._apply_build_state(target)
        return self._spec(products, steps, target)

    def _spec(self, products: list[int], steps: dict, target: dict | None) -> dict:
        rnd = {**self.inp.sets["randomization"], **self.cfg.get("randomization", {})}   # edge sets widen ranges
        spec = {
            "spec_version": SPEC_VERSION, "set": self.set_name, "index": self.index, "seed": self.seed,
            "table_z_uf0": self.table_z,
            "products": products, "assembly_step": {str(p): steps[p] for p in products},
            "cameras": self.cfg["cameras"], "outputs": self.cfg["outputs"], "objects": self.objects,
            "camera_jitter": {c: self._camera_jitter(self._jitter_range(c, rnd)) for c in self.cfg["cameras"]},
            "lighting": self._draw(rnd["lighting"]), "material": self._draw_material(rnd["black_pla"]),
            "post": self._draw_post(rnd["post"]), "arm": self._arm(target),
        }
        if target is not None:
            spec["wrist_target_uid"] = target["uid"]
        return spec

    # ------------------------------------------------------------------ products and parts
    def _planned_target(self) -> str | None:
        """`all_objects` balances all 30 objects; `parts` cycles the 24 parts. Others pick after layout (or none)."""
        plan = self.cfg.get("target_plan")
        if plan == "all_objects":
            return sorted(self.inp.geometry)[self.index % len(self.inp.geometry)]
        if plan == "parts":
            parts = sorted(k for k in self.inp.geometry if k.startswith("M"))
            return parts[self.index % len(parts)]
        return None

    def _choose_products(self, target_id: str | None) -> list[int]:
        n = self.sc["products"]
        forced = [self.inp.taxonomy[target_id]["product"]] if target_id else []
        rest = [p for p in range(1, 7) if p not in forced]
        return sorted(forced + self.rng.sample(rest, n - len(forced)))

    def _choose_step(self, product: int, target_id: str | None) -> int:
        step = self.rng.choices(range(5), weights=self.sc["assembly_step_weights"])[0]
        if target_id and target_id.startswith(f"M{product}_"):
            step = min(step, self.inp.geometry[target_id]["install_step"] - 1)   # wrist target stays loose
        return step

    def _place_base(self, product: int) -> dict:
        base_id = f"F0{product}"
        geo = self.inp.geometry[base_id]
        lift = translation(0, 0, -geo["bounds_mm"][0][2])
        base_tf = self._place(lift, geo["upright"], self.inp.layout["work_area"], "base")
        return self._add("base", base_id, base_tf, state="on_table", product=product)

    def _place_parts(self, product: int, step: int, base: dict) -> None:
        base_tf = np.array(base["transform_uf0"])
        for part_id in self.inp.product_parts(product):
            pgeo = self.inp.geometry[part_id]
            if pgeo["install_step"] <= step:
                tf = base_tf @ np.array(pgeo["installed_in_base_frame"])
                self._add("part", part_id, tf, state="installed", product=product, on_base=base["uid"])
            else:
                self._add_loose("part", part_id, product=product)

    def _add_loose(self, kind: str, object_id: str, product: int | None = None, zone: str = "work_area",
                   required: bool = True) -> dict | None:
        geo = self.inp.geometry[object_id]
        if self.sc.get("pose_sampling") == "installed" and zone == "work_area":
            pose, k, rest = self._test_bed_pose(geo)
        else:
            poses = geo["stable_poses"]
            # `uniform` shows rare resting poses (on a side, upside down) as often as the likely ones.
            weights = None if self.sc.get("pose_sampling") == "uniform" else [p["probability"] for p in poses]
            k = self.rng.choices(range(len(poses)), weights=weights)[0]
            pose, rest = poses[k], None
        cluster = self._cluster_centre() if zone == "work_area" else None
        tf = self._place(np.array(pose["transform"]), pose, self.inp.layout[zone], "part", cluster, required)
        if tf is None:
            return None
        state = "in_reject" if zone == "reject_zone" else "on_table"
        extra = {"rest": rest} if rest else {}
        return self._add(kind, object_id, tf, state=state, product=product, stable_pose=k, **extra)

    def _test_bed_pose(self, geo: dict) -> tuple[dict, int, str]:
        """Test-bed placement: the arm cannot turn a part over, so parts stand as installed (yaw comes from _place).
        A `fallen_prob` share lies in another resting pose instead; the planner must reject those and use a spare.
        stable_pose is the matching drop pose, or -1 when the installed orientation is not one (an edge stand)."""
        installed = geo["installed_rest"]
        up = np.array(installed["transform"])[2, :3]
        same = [np.allclose(np.array(p["transform"])[2, :3], up, atol=1e-2) for p in geo["stable_poses"]]
        fallen = [k for k, s in enumerate(same) if not s]
        if fallen and self.rng.random() < self.sc.get("fallen_prob", 0.0):
            k = self.rng.choices(fallen, weights=[geo["stable_poses"][i]["probability"] for i in fallen])[0]
            return geo["stable_poses"][k], k, "fallen"
        return installed, (same.index(True) if True in same else -1), "installed"

    def _add_trap(self, products: list[int]) -> None:
        """A look-alike of a present part, from a product that is not on the table."""
        if self.rng.random() >= self.sc["wrong_part_trap_prob"]:
            return
        present = [p for prod in products for p in self.inp.product_parts(prod)]
        mates = sorted({m for p in present for m in self.inp.family_mates(p)
                        if m.startswith("M") and self.inp.taxonomy[m]["product"] not in products})
        if mates:
            trap_id = self.rng.choice(mates)
            self._add_loose("trap", trap_id, product=self.inp.taxonomy[trap_id]["product"])

    def _add_spares(self) -> None:
        slots = self.inp.layout["spare_store"]["slots"]
        n = self.rng.randint(*self.sc["spares_in_store"])
        test_bed = self.sc.get("pose_sampling") == "installed"   # spares are picked straight down too
        for slot, part_id in zip(self.rng.sample(slots, n), self.rng.sample(self.defaults["spares"], n)):
            geo = self.inp.geometry[part_id]
            pose = geo["installed_rest"] if test_bed else geo["stable_poses"][0]
            at = translation(slot["x"], slot["y"], self.table_z) @ rot_z(self.rng.uniform(-10, 10))
            self._occupied.append((*self._world_footprint(at, pose), "part"))
            extra = {"rest": "installed"} if test_bed else {}
            self._add("spare", part_id, at @ np.array(pose["transform"]), state="in_store", slot=slot["id"],
                      stable_pose=0, **extra)

    def _add_rejects(self) -> None:
        parts = sorted(k for k in self.inp.geometry if k.startswith("M"))
        for _ in range(self.rng.randint(*self.sc["rejects"])):
            # The tray is small: when it is full the scene simply has fewer rejects.
            if self._add_loose("reject", self.rng.choice(parts), zone="reject_zone", required=False) is None:
                return

    def _add_foreign(self) -> None:
        for _ in range(self.rng.randint(*self.sc["foreign_objects"])):
            kind = self.rng.choice(self.defaults["foreign_objects"])
            fp = {"footprint_xy": [0.0, 0.0], "footprint_r": FOREIGN_RADIUS_MM[kind]}
            tf = self._place(np.eye(4), fp, self.inp.layout["work_area"], "part", self._cluster_centre())
            self._add("foreign", None, tf, state="on_table", foreign_type=kind)

    # ------------------------------------------------------------------ targets, defects, build states
    def _resolve_target(self, target_id: str | None) -> dict | None:
        if target_id:
            return next(o for o in self.objects if o["object_id"] == target_id and o["kind"] in ("base", "part"))
        if "build_states" in self.cfg:
            tops = {}
            for o in self.objects:
                if o["state"] == "installed":
                    tops[o["product"]] = o   # parts are added in install order, so the last one is the top
            return tops[self.rng.choice(sorted(tops))]
        return None

    def _apply_defects(self, target: dict | None) -> None:
        types = self.defaults["defect_types"]
        if "good_fraction" in self.cfg:
            if target is not None and self.rng.random() >= self.cfg["good_fraction"]:
                target["variant"] = self.rng.choice(types)
            return
        candidates = [o for o in self.objects if o["kind"] in ("part", "trap")]
        n = min(self.rng.randint(*self.sc["defects_per_scene"]), len(candidates))
        for obj in self.rng.sample(candidates, n):
            obj["variant"] = self.rng.choice(types)

    def _apply_build_state(self, target: dict) -> None:
        """Mis-builds rest ON the fixture stack: each is lifted by the precomputed clearance, never passes through."""
        state = self.rng.choices(self.cfg["build_states"], weights=self.cfg["build_state_weights"])[0]
        base = next(o for o in self.objects if o["uid"] == target["on_base"])
        base_tf = np.array(base["transform_uf0"])
        part_id = target["object_id"]
        clear = self.inp.clearance["parts"][part_id]
        installed = np.array(self.inp.geometry[part_id]["installed_in_base_frame"])
        box = world_aabb(installed, self.inp.geometry[part_id]["bounds_mm"])
        pivot = np.array([*(box[0, :2] + box[1, :2]) / 2, box[0, 2]])
        offset = {"high_mm": 0.0, "tilt_deg": 0.0}   # magnitude in the label, so a threshold can be chosen later
        if "shift_mm" in self.cfg:
            offset.update(shift_mm=0.0, yaw_deg=0.0)
        if state == "rotated":
            yaws = [y for y, v in clear["yaw"].items()
                    if v["deviation_mm"] >= self.inp.clearance["yaw_visible_mm"] and v["lift_mm"] <= MAX_REST_LIFT_MM]
            state = state if yaws else "shifted"   # a symmetric part turned looks seated: shift it instead
        lift = 0.0
        if state == "seated" and "seated_noise_mm" in self.cfg:   # print tolerance: still seated
            lift = offset["high_mm"] = self.rng.uniform(0, self.cfg["seated_noise_mm"])
        elif state == "high":
            lift = offset["high_mm"] = self.rng.uniform(*self.cfg["high_mm"])
        elif state == "tilted":
            az = self.rng.uniform(0, 360)
            offset["tilt_deg"] = self.rng.uniform(*self.cfg["tilt_deg"])
            installed = about_point(rot_axis((math.cos(math.radians(az)), math.sin(math.radians(az)), 0),
                                             offset["tilt_deg"]), pivot) @ installed
            lift = _rest_lift(clear["tilt_mm"], az, offset["tilt_deg"], self.inp.clearance["tilt_deg"])
        elif state == "shifted":   # peg contact changes abruptly with direction: use exact table entries only
            n_dirs, grid = len(clear["shift_mm"]), self.inp.clearance["shift_mm"]
            d = self.rng.randrange(n_dirs)
            k = self.rng.choice([i for i, v in enumerate(grid) if self.cfg["shift_mm"][0] <= v <= self.cfg["shift_mm"][1]])
            az, offset["shift_mm"] = 360.0 * d / n_dirs, grid[k]
            installed = translation(grid[k] * math.cos(math.radians(az)), grid[k] * math.sin(math.radians(az)), 0) \
                @ installed
            lift = clear["shift_mm"][d][k] + REST_MARGIN_MM * (clear["shift_mm"][d][k] > 0)
        elif state == "rotated":
            yaw = self.rng.choice(yaws)
            offset["yaw_deg"] = float(yaw)
            installed = about_point(rot_axis((0, 0, 1), float(yaw)), pivot) @ installed
            lift = clear["yaw"][yaw]["lift_mm"] + REST_MARGIN_MM * (clear["yaw"][yaw]["lift_mm"] > 0)
        elif state == "wrong_part":
            mate = self.rng.choice(self.inp.family_mates(part_id))
            self._set_object(target, mate)
            lift = clear["wrong_mm"][mate] + REST_MARGIN_MM * (clear["wrong_mm"][mate] > 0)
        if state in ("tilted", "shifted", "rotated", "wrong_part"):
            offset["high_mm"] = lift
        target["build_state"] = state
        target["build_offset"] = {k: round(v, 3) for k, v in offset.items()}
        self._set_transform(target, base_tf @ translation(0, 0, lift) @ installed)

    # ------------------------------------------------------------------ arm, camera, look
    def _arm(self, target: dict | None) -> dict:
        """Drawn last in the scene, so redrawing an unreachable pose changes nothing else in the spec."""
        if target is not None and "wrist" in self.cfg["cameras"]:
            return {"mode": "wrist_view", "tcp_uf0": self._reachable(lambda: self._wrist_tcp(target))}
        if self.rng.random() < self.sc["arm_in_view_prob"]:
            return {"mode": "in_view", "tcp_uf0": self._reachable(self._in_view_tcp)}
        return {"mode": "parked"}

    def _reachable(self, draw) -> list[float]:
        for _ in range(MAX_TRIES):
            tcp = draw()
            if reachable(tcp):
                return tcp
        raise SceneSamplingError(f"{self.set_name}[{self.index}]: no reachable arm pose after {MAX_TRIES} tries")

    def _in_view_tcp(self) -> list[float]:
        wa = self.inp.layout["work_area"]
        tcp = [self.rng.uniform(*wa["x"]), self.rng.uniform(*wa["y"]), self.rng.uniform(-200, 100),
               -180.0, 0.0, self.rng.uniform(-90, 90)]
        return _check_tcp([round(v, 3) for v in tcp])

    def _wrist_tcp(self, target: dict) -> list[float]:
        w = self.cfg["wrist"]
        box = world_aabb(np.array(target["transform_uf0"]), self.inp.geometry[target["object_id"]]["bounds_mm"])
        centre, _ = footprint(box)
        off_a, off_r = self.rng.uniform(0, 2 * math.pi), self.rng.uniform(0, w["target_offset_mm"])
        tilt, tilt_az = self.rng.uniform(*w["tilt_deg"]), self.rng.uniform(0, 2 * math.pi)
        tcp = [centre[0] + off_r * math.cos(off_a), centre[1] + off_r * math.sin(off_a),
               box[1, 2] + self.rng.uniform(*w["standoff_mm"]),
               -180.0 + tilt * math.cos(tilt_az), tilt * math.sin(tilt_az), self.rng.uniform(-90, 90)]
        return _check_tcp([round(float(v), 3) for v in tcp])

    def _jitter_range(self, camera: str, rnd: dict) -> dict:
        """Mount drift applies to the fixed table / side cameras. The wrist camera is bolted to the tool: a large
        offset would put its eye inside its own lens housing, so it always keeps the nominal range."""
        return self.inp.sets["randomization"]["camera_jitter"] if camera == "wrist" else rnd["camera_jitter"]

    def _camera_jitter(self, cfg: dict) -> dict:
        t, r = cfg["translate_mm"], cfg["rotate_deg"]
        return {"translate_mm": [round(self.rng.uniform(-t, t), 3) for _ in range(3)],
                "rotate_deg": [round(self.rng.uniform(-r, r), 4) for _ in range(3)]}

    def _draw(self, ranges: dict) -> dict:
        return {k: round(self.rng.uniform(*v), 4) for k, v in ranges.items()}

    def _draw_material(self, cfg: dict) -> dict:
        lo, hi = cfg["base_color_rgb"]
        grey = self.rng.uniform(lo[0], hi[0])
        out = self._draw({k: v for k, v in cfg.items() if k != "base_color_rgb"})
        return {"name": self.defaults["material"], "base_color_rgb": [round(grey, 4)] * 3, **out}

    def _draw_post(self, cfg: dict) -> dict:
        return {"motion_blur": self.rng.random() < cfg["motion_blur_prob"],
                "gaussian_noise_sigma": round(self.rng.uniform(*cfg["gaussian_noise_sigma"]), 4),
                "jpeg_quality": self.rng.randint(*cfg["jpeg_quality"])}

    # ------------------------------------------------------------------ layout helpers
    def _cluster_centre(self) -> tuple[float, float] | None:
        if self.rng.random() >= self.sc["cluster_prob"]:
            return None
        wa = self.inp.layout["work_area"]
        return self.rng.uniform(*wa["x"]), self.rng.uniform(*wa["y"])

    def _place(self, local: np.ndarray, fp: dict, region: dict, kind: str, cluster=None,
               required=True) -> np.ndarray | None:
        """Drop `local` (an object already resting on z=0) at a free spot in `region`; returns its UF0 transform."""
        for attempt in range(MAX_TRIES):
            if cluster is not None and attempt < MAX_TRIES // 4:   # a full cluster falls back to uniform
                x, y = self.rng.gauss(cluster[0], 45.0), self.rng.gauss(cluster[1], 45.0)
            else:
                x, y = self.rng.uniform(*region["x"]), self.rng.uniform(*region["y"])
            at = translation(x, y, self.table_z) @ rot_z(self.rng.uniform(-180.0, 180.0))
            centre, radius = self._world_footprint(at, fp)
            if self._fits(centre, radius, region, kind):
                self._occupied.append((centre, radius, kind))
                return at @ local
        if not required:
            return None
        raise SceneSamplingError(f"{self.set_name}[{self.index}]: no room for a {kind} after {MAX_TRIES} tries")

    @staticmethod
    def _world_footprint(at: np.ndarray, fp: dict) -> tuple[np.ndarray, float]:
        """Footprint circle measured on the posed mesh (object_geometry.py), moved by the table placement `at`."""
        cx, cy = fp["footprint_xy"]
        return (at @ np.array([cx, cy, 0.0, 1.0]))[:2], fp["footprint_r"]

    def _fits(self, centre: np.ndarray, radius: float, region: dict, kind: str) -> bool:
        lay = self.inp.layout
        if not (region["x"][0] <= centre[0] <= region["x"][1] and region["y"][0] <= centre[1] <= region["y"][1]):
            return False
        if not lay["reach_radius_min"] <= math.hypot(*centre) <= lay["reach_radius_max"]:
            return False
        gaps = lay["clearance"]
        for c, r, k in self._occupied:
            gap = gaps["base_gap_mm"] if k == kind == "base" else (
                gaps["part_gap_mm"] if k == kind == "part" else gaps["base_to_part_gap_mm"])
            if np.linalg.norm(centre - c) < radius + r + gap:
                return False
        return True

    def _add(self, kind: str, object_id: str | None, tf: np.ndarray, **fields) -> dict:
        obj = {"uid": f"o{len(self.objects):02d}", "kind": kind, "object_id": None, "product": None, "variant": "ok",
               "build_state": None, **fields}
        self._set_object(obj, object_id)
        self._set_transform(obj, tf)
        self.objects.append(obj)
        return obj

    def _set_object(self, obj: dict, object_id: str | None) -> None:
        obj["object_id"] = object_id
        obj["stl"] = self.inp.geometry[object_id]["stl"] if object_id else None

    @staticmethod
    def _set_transform(obj: dict, tf: np.ndarray) -> None:
        obj["transform_uf0"] = np.round(tf, 4).tolist()
        obj["pose_uf0"] = {"x": round(float(tf[0, 3]), 3), "y": round(float(tf[1, 3]), 3),
                           "z": round(float(tf[2, 3]), 3), "yaw": round(yaw_deg(tf), 3)}


def _rest_lift(table: list[list[float]], az_deg: float, value: float, grid: list[float]) -> float:
    """Clearance for a tilt / shift of `value` toward azimuth `az_deg`: the larger of the two neighbouring azimuth
    rows (so the part may float a hair, never sink), interpolated along `grid` from 0 at 0."""
    n = len(table)
    a0 = int(az_deg % 360 / 360 * n) % n
    lifts = [float(np.interp(value, [0.0, *grid], [0.0, *table[a % n]])) for a in (a0, a0 + 1)]
    return max(lifts) + REST_MARGIN_MM * (max(lifts) > 0)


def _check_tcp(tcp: list[float]) -> list[float]:
    for v, axis in zip(tcp[:3], "xyz"):
        lo, hi = TCP_BOX[axis]
        if not lo <= v <= hi:
            raise SceneSamplingError(f"wrist TCP {axis}={v} outside the cell box [{lo}, {hi}]")
    return tcp


def sample_scene(inputs: Inputs, set_name: str, index: int) -> dict:
    return SceneBuilder(inputs, set_name, index).build()


def scenes_per_chunk(inputs: Inputs, set_name: str) -> int:
    """Chunks are sized in images; a scene renders one image per camera."""
    return inputs.sets["defaults"]["chunk_size"] // len(inputs.sets["sets"][set_name]["cameras"])


def n_chunks(inputs: Inputs, set_name: str) -> int:
    cfg = inputs.sets["sets"][set_name]
    return math.ceil(cfg["images"] / len(cfg["cameras"]) / scenes_per_chunk(inputs, set_name))


def chunk_range(inputs: Inputs, set_name: str, chunk: int) -> range:
    cfg = inputs.sets["sets"][set_name]
    total = cfg["images"] // len(cfg["cameras"])
    per = scenes_per_chunk(inputs, set_name)
    if not 0 <= chunk < n_chunks(inputs, set_name):
        raise ValueError(f"set {set_name} has chunks 0..{n_chunks(inputs, set_name) - 1}, got {chunk}")
    return range(chunk * per, min((chunk + 1) * per, total))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", required=True, help="a set name from sdg/configs/sets.yaml")
    ap.add_argument("--chunk", type=int, help="write the specs of one chunk")
    ap.add_argument("--n-chunks", action="store_true", help="print how many chunks the set has")
    ap.add_argument("--out", type=Path, help="JSONL file for --chunk")
    args = ap.parse_args()
    inputs = Inputs.load()
    if args.set not in inputs.sets["sets"]:
        ap.error(f"unknown set {args.set}; sets.yaml has {sorted(inputs.sets['sets'])}")
    if args.n_chunks:
        print(n_chunks(inputs, args.set))
        return
    if args.chunk is None or args.out is None:
        ap.error("--chunk and --out are required unless --n-chunks is given")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    indices = chunk_range(inputs, args.set, args.chunk)
    with args.out.open("w") as f:
        for i in indices:
            f.write(json.dumps(sample_scene(inputs, args.set, i)) + "\n")
    print(f"wrote {len(indices)} specs ({args.set} chunk {args.chunk}) to {args.out}")


if __name__ == "__main__":
    main()
