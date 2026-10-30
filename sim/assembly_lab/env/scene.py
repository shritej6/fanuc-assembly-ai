"""Isaac Lab InteractiveScene for the cell. Runs inside Isaac Lab (after AppLauncher).

The whole USD (starter cell + Team V fingers + pool objects + Mimic frame markers) is written by one custom spawner,
`spawn_cell`, the first scene entry; every Isaac Lab asset / sensor after it wraps prims that spawner made
(`spawn=None`). The cell is a per-env template: spawn_cell builds it under /World as the starter does, then moves
every cell prim into /World/envs/env_0/Cell, and Isaac Lab copies env_0 to env_1..N-1 on a grid (env_spacing, the
6 m floor fits inside one cell). Only the shared look / physics materials and the dome light stay global (one sky,
not N). Cells never touch, so collision filtering stays off; every pose a caller sees is relative to its env origin.

Scene keys: `robot`; one RigidObject per pool instance named by `prim_name(uid)` (e.g. M4_01, M4_03_c2), so the
recorder's states read `rigid_object/<name>`; Mimic frame markers `grasp_frame`, `seat_frame`, `spot_frame`
(kinematic, no collider, invisible); `robot_contact_<link>` (that robot link vs every object), `object_contact` (net
force on every object), `object_env_contact_<name>` (that object vs every non-robot collider); cameras `table`,
`side`, `wrist`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
import layout as L
import numpy as np
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import configclass

REPO = Path(__file__).resolve().parents[3]
FRAMES = ("grasp_frame", "seat_frame", "spot_frame")
ENV_SPACING_M = 7.0                 # the starter floor is 6 x 6 m about the robot base: neighbouring floors never overlap
GLOBAL_PRIMS = ("Looks", "Physics", "DomeLight")   # /World children every cell shares; everything else is per env
CELL = "{ENV_REGEX_NS}/Cell"        # per-env cell root; the starter's /World/<name> becomes <env>/Cell/<name>
OBJECTS_ROOT = CELL + "/Objects"
FRAMES_ROOT = CELL + "/Frames"
EXTRAS_ROOT = CELL + "/Extras"
ROBOT_PATH = CELL + "/ER4iA"
STATIC_COLLIDERS = tuple(f"{CELL}/{n}" for n in ("Table", "Mat", "Floor"))   # every non-robot static collider
                                                                              # (mimic_check.py --list_colliders)
EXTRAS_PARK_M = (2.6, 1.6, 0.0)     # env-local m, behind the table and side cameras (layout.PARK_ORIGIN_MM region)
# Every robot rigid body (the robot_contact sensors; ContactMetrics checks it against the articulation). PhysX filters
# only pair one sensor body per env with each filter, so each link (and each object, below) is its own sensor.
ROBOT_LINKS = ("robot_base", "J1_link", "J2_link", "J3_link", "J4_link", "J5_link", "J6_link", "JawL", "JawR")
CAMERA_PATHS = {"table": CELL + "/TableCam", "side": CELL + "/SideCam", "wrist": CELL + "/ER4iA/J6_link/WristCam"}
# Position-controlled arm like the real FANUC: stiff joint PD, gravity off on the links (the real controller
# compensates it). Gains are chosen for < 0.05 mm steady-state TCP error, not identified from the real arm.
ARM_STIFFNESS = 2.0e5     # N m / rad
ARM_DAMPING = 2.0e3       # N m s / rad
ARM_EFFORT = 1.0e4        # N m (no saturation: the real arm is position controlled)
# Jaw drives: the squeeze is a constant feed-forward force (actions.TcpAction); these position drives only centre the
# jaw pair on its mean closing depth (rack and pinion). Their implicit stiffness absorbs part of the feed-forward force
# while the contact is solved (grasp_diag.py, M1_02 at 34 N: 10 x -> 27.4 N on the part, 1 x -> 33.3 N but 0.03 mm
# centring slip); 3 x: 32-34 N on the part and 4.5 N lateral / (2 x stiffness) = 0.019 mm off-centre.
JAW_STIFF_X = 3.0         # x scene_config.JAW_DRIVE_STIFF
JAW_EFFORT_X = 2.0        # centring drive force cap, x the per-finger force


def prim_name(uid: str) -> str:
    return uid.replace("#", "_c")


def env_path(template: str, env_id: int = 0) -> str:
    """A CELL-relative path template resolved for one env (USD prim path, no regex)."""
    return template.format(ENV_REGEX_NS=f"/World/envs/env_{env_id}")


def _starter_path(template: str) -> str:
    """Where the starter / assets.py write a cell prim before spawn_cell moves it into env_0 (/World/<name>)."""
    return template.replace(CELL, "/World")


def _move_cell_prims(stage, cell_root: str, built: list[str]) -> None:
    """Move every /World child the cell build created (except GLOBAL_PRIMS) under cell_root. Sdf.CopySpec re-targets
    relationships inside the moved subtree (joint bodies, the gripper's jaw joints); bindings to the global looks /
    physics materials keep their absolute targets."""
    from pxr import Sdf

    layer = stage.GetEditTarget().GetLayer()
    for name in built:
        src, dst = Sdf.Path(f"/World/{name}"), Sdf.Path(f"{cell_root}/{name}")
        if not layer.GetPrimAtPath(src):
            raise RuntimeError(f"cell prim {src} is not authored in the edit layer; cannot move it into {cell_root}")
        if not Sdf.CopySpec(layer, src, layer, dst):
            raise RuntimeError(f"could not copy {src} to {dst}")
        stage.RemovePrim(src)


@configclass
class CellSpawnerCfg(SpawnerCfg):
    """Builds the whole cell. `func` writes USD; contact / pool / jaw force come from CellEnvCfg."""

    func: Callable = MISSING
    contact: dict = MISSING
    pool: list = MISSING
    jaw_force_n: float = MISSING
    extras: bool = False


def spawn_cell(prim_path: str, cfg: CellSpawnerCfg, translation=None, orientation=None, **kwargs):
    import assets
    import scene_config as C

    stage = sim_utils.get_current_stage()
    world = stage.GetPrimAtPath("/World")
    before = {c.GetName() for c in world.GetChildren()} if world.IsValid() else set()
    assets.build_cell(stage, cfg.contact)
    assets.set_jaw_force(stage, JAW_EFFORT_X * cfg.jaw_force_n)
    assets.robot_physics(stage, cfg.contact)
    solver = L.LayoutSolver()
    cache = assets.object_cache_dir(cfg.contact)
    floor_z = -(C.TABLE_TOP_Z + L.UF0_ABOVE_BASE_M) * 1000.0
    for pl in solver.solve({}, list(cfg.pool), floor_z):
        usd = cache / f"{pl.type_id}.usda"
        if not usd.exists():
            assets.write_object_usd(usd, REPO / solver.geometry[pl.type_id]["stl"], cfg.contact)
        assets.add_object(stage, f"{_starter_path(OBJECTS_ROOT)}/{prim_name(pl.uid)}", usd,
                          L.uf0_to_world_m(pl.tf_uf0, C.TABLE_TOP_Z))
    if cfg.extras:
        assets.add_extras(stage, _starter_path(EXTRAS_ROOT), EXTRAS_PARK_M)
    for k, name in enumerate(FRAMES):
        assets.add_marker(stage, f"{_starter_path(FRAMES_ROOT)}/{name}", (-1.0, 1.0 + 0.1 * k, 0.05))
    root = stage.DefinePrim(prim_path, "Xform")
    built = [c.GetName() for c in stage.GetPrimAtPath("/World").GetChildren()
             if c.GetName() not in before and c.GetName() not in GLOBAL_PRIMS]
    _move_cell_prims(stage, prim_path, built)
    return root


def make_scene_cfg(pool: list[str], contact: dict, jaw_force_n: float, home_rad: np.ndarray,
                   cameras: tuple[str, ...], image_size: tuple[int, int], depth: tuple[str, ...],
                   extras: bool = False, num_envs: int = 1) -> InteractiveSceneCfg:
    import scene_config as C

    scene = InteractiveSceneCfg(num_envs=num_envs, env_spacing=ENV_SPACING_M, replicate_physics=False,
                                filter_collisions=False, lazy_sensor_update=True)
    scene.cell = AssetBaseCfg(prim_path=CELL, spawn=CellSpawnerCfg(
        func=spawn_cell, contact=contact, pool=list(pool), jaw_force_n=jaw_force_n, extras=extras))
    joint_pos = {f"J{i}": float(home_rad[i - 1]) for i in range(1, 7)}
    joint_pos.update({"jaw_l": 0.0, "jaw_r": 0.0})
    scene.robot = ArticulationCfg(
        prim_path=ROBOT_PATH, spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, C.TABLE_TOP_Z), joint_pos=joint_pos),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["J[1-6]"], stiffness=ARM_STIFFNESS, damping=ARM_DAMPING,
                                       effort_limit_sim=ARM_EFFORT),
            # two independent drives, synchronised in software every physics step (actions.TcpAction)
            "jaws": ImplicitActuatorCfg(joint_names_expr=["jaw_.*"], stiffness=JAW_STIFF_X * C.JAW_DRIVE_STIFF,
                                        damping=C.JAW_DRIVE_DAMP, effort_limit_sim=JAW_EFFORT_X * jaw_force_n)})
    names = [prim_name(u) for u in pool]
    for n in names:
        setattr(scene, n, RigidObjectCfg(prim_path=f"{OBJECTS_ROOT}/{n}", spawn=None))
    for f in FRAMES:
        setattr(scene, f, RigidObjectCfg(prim_path=f"{FRAMES_ROOT}/{f}", spawn=None))
    if extras:
        import assets

        for name in assets.EXTRAS:
            setattr(scene, f"extra_{name}", RigidObjectCfg(prim_path=f"{EXTRAS_ROOT}/{name}", spawn=None))
    # each robot link against every pool object (robot_contact_<link>; contact_metrics.SensorGroup reads them as one)
    for link in ROBOT_LINKS:
        setattr(scene, f"robot_contact_{link}", ContactSensorCfg(
            prim_path=f"{ROBOT_PATH}/{link}", track_contact_points=True, max_contact_data_count_per_prim=512,
            filter_prim_paths_expr=[f"{OBJECTS_ROOT}/{n}" for n in names]))
    scene.object_contact = ContactSensorCfg(prim_path=f"{OBJECTS_ROOT}/.*")
    # each object against everything that is NOT the robot (other objects, table, mat, floor), per contact point:
    # the seating torque comes only from these (contact_metrics.py)
    for n in names:
        setattr(scene, f"object_env_contact_{n}", ContactSensorCfg(
            prim_path=f"{OBJECTS_ROOT}/{n}", track_contact_points=True, track_friction_forces=True,
            max_contact_data_count_per_prim=512,
            filter_prim_paths_expr=[f"{OBJECTS_ROOT}/{m}" for m in names] + list(STATIC_COLLIDERS)))
    w, h = image_size
    for cam in cameras:
        setattr(scene, cam, CameraCfg(prim_path=CAMERA_PATHS[cam], spawn=None, width=w, height=h, update_period=0.0,
                                      update_latest_camera_pose=True,
                                      data_types=["rgb"] + (["distance_to_image_plane"] if cam in depth else [])))
    return scene
