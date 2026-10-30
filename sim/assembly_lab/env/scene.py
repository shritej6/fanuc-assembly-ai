"""Isaac Lab InteractiveScene for the cell. Runs inside Isaac Lab (after AppLauncher).

The whole USD (starter cell + Team V fingers + pool objects + Mimic frame markers) is written by one custom spawner,
`spawn_cell`, the first scene entry; every Isaac Lab asset / sensor after it wraps prims that spawner made
(`spawn=None`). Single environment: all prims are global (/World/...), so physics cloning and collision filtering
are off.

Scene keys: `robot`; one RigidObject per pool instance named by `prim_name(uid)` (e.g. M4_01, M4_03_c2), so the
recorder's states read `rigid_object/<name>`; Mimic frame markers `grasp_frame`, `seat_frame`, `spot_frame`
(kinematic, no collider, invisible); `robot_contact` (every robot link vs every object) and `object_contact` (net
force on every object); cameras `table`, `side`, `wrist`.
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
OBJECTS_ROOT = "/World/Objects"
FRAMES_ROOT = "/World/Frames"
EXTRAS_ROOT = "/World/Extras"
STATIC_COLLIDERS = ("/World/Table", "/World/Mat", "/World/Floor")   # every non-robot static collider (mimic_check.py
                                                                    # --list_colliders, measured)
EXTRAS_PARK_M = (2.6, 1.6, 0.0)     # world m, behind the table and side cameras (layout.PARK_ORIGIN_MM region)
CAMERA_PATHS = {"table": "/World/TableCam", "side": "/World/SideCam", "wrist": "/World/ER4iA/J6_link/WristCam"}
# Position-controlled arm like the real FANUC: stiff joint PD, gravity off on the links (the real controller
# compensates it). Gains are chosen for < 0.05 mm steady-state TCP error, not identified from the real arm.
ARM_STIFFNESS = 2.0e5     # N m / rad
ARM_DAMPING = 2.0e3       # N m s / rad
ARM_EFFORT = 1.0e4        # N m (no saturation: the real arm is position controlled)


def prim_name(uid: str) -> str:
    return uid.replace("#", "_c")


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
    assets.build_cell(stage, cfg.contact)
    assets.set_jaw_force(stage, cfg.jaw_force_n)
    assets.robot_physics(stage, cfg.contact)
    solver = L.LayoutSolver()
    cache = assets.object_cache_dir(cfg.contact)
    floor_z = -(C.TABLE_TOP_Z + L.UF0_ABOVE_BASE_M) * 1000.0
    for pl in solver.solve({}, list(cfg.pool), floor_z):
        usd = cache / f"{pl.type_id}.usda"
        if not usd.exists():
            assets.write_object_usd(usd, REPO / solver.geometry[pl.type_id]["stl"], cfg.contact)
        assets.add_object(stage, f"{OBJECTS_ROOT}/{prim_name(pl.uid)}", usd,
                          L.uf0_to_world_m(pl.tf_uf0, C.TABLE_TOP_Z))
    if cfg.extras:
        assets.add_extras(stage, EXTRAS_ROOT, EXTRAS_PARK_M)
    for k, name in enumerate(FRAMES):
        assets.add_marker(stage, f"{FRAMES_ROOT}/{name}", (-1.0, 1.0 + 0.1 * k, 0.05))
    return stage.DefinePrim(prim_path, "Xform")


def make_scene_cfg(pool: list[str], contact: dict, jaw_force_n: float, home_rad: np.ndarray,
                   cameras: tuple[str, ...], image_size: tuple[int, int], depth: tuple[str, ...],
                   extras: bool = False) -> InteractiveSceneCfg:
    import scene_config as C

    scene = InteractiveSceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=False, filter_collisions=False,
                                lazy_sensor_update=True)
    scene.cell = AssetBaseCfg(prim_path="/World/CellBuild", spawn=CellSpawnerCfg(
        func=spawn_cell, contact=contact, pool=list(pool), jaw_force_n=jaw_force_n, extras=extras))
    joint_pos = {f"J{i}": float(home_rad[i - 1]) for i in range(1, 7)}
    joint_pos.update({"jaw_l": 0.0, "jaw_r": 0.0})
    scene.robot = ArticulationCfg(
        prim_path=C.ROBOT_PRIM, spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, C.TABLE_TOP_Z), joint_pos=joint_pos),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["J[1-6]"], stiffness=ARM_STIFFNESS, damping=ARM_DAMPING,
                                       effort_limit_sim=ARM_EFFORT),
            # jaw_r follows jaw_l through a mimic joint (assets.set_jaw_force): one drive, 2 x the per-finger force
            "jaws": ImplicitActuatorCfg(joint_names_expr=["jaw_l"], stiffness=2.0 * C.JAW_DRIVE_STIFF,
                                        damping=2.0 * C.JAW_DRIVE_DAMP, effort_limit_sim=2.0 * jaw_force_n),
            "jaw_follower": ImplicitActuatorCfg(joint_names_expr=["jaw_r"], stiffness=0.0, damping=0.0,
                                                effort_limit_sim=2.0 * jaw_force_n)})
    names = [prim_name(u) for u in pool]
    for n in names:
        setattr(scene, n, RigidObjectCfg(prim_path=f"{OBJECTS_ROOT}/{n}", spawn=None))
    for f in FRAMES:
        setattr(scene, f, RigidObjectCfg(prim_path=f"{FRAMES_ROOT}/{f}", spawn=None))
    if extras:
        import assets

        for name in assets.EXTRAS:
            setattr(scene, f"extra_{name}", RigidObjectCfg(prim_path=f"{EXTRAS_ROOT}/{name}", spawn=None))
    scene.robot_contact = ContactSensorCfg(
        prim_path=C.ROBOT_PRIM + "/.*", track_contact_points=True, max_contact_data_count_per_prim=512,
        filter_prim_paths_expr=[f"{OBJECTS_ROOT}/{n}" for n in names])
    scene.object_contact = ContactSensorCfg(prim_path=f"{OBJECTS_ROOT}/.*")
    # every object against everything that is NOT the robot (other objects, table, mat, floor), per contact point:
    # the seating torque comes only from these (contact_metrics.py)
    scene.object_env_contact = ContactSensorCfg(
        prim_path=f"{OBJECTS_ROOT}/.*", track_contact_points=True, track_friction_forces=True,
        max_contact_data_count_per_prim=512,
        filter_prim_paths_expr=[f"{OBJECTS_ROOT}/{n}" for n in names] + list(STATIC_COLLIDERS))
    w, h = image_size
    for cam in cameras:
        setattr(scene, cam, CameraCfg(prim_path=CAMERA_PATHS[cam], spawn=None, width=w, height=h, update_period=0.0,
                                      update_latest_camera_pose=True,
                                      data_types=["rgb"] + (["distance_to_image_plane"] if cam in depth else [])))
    return scene
