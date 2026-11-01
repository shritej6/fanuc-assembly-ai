"""Build the USD assets of the sim cell inside Isaac Sim.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


def _starter_shims() -> None:
    """cell.py imports isaacsim.core.utils.stage.add_reference_to_stage, which Isaac Sim 6.0.0 no longer ships
    (only isaacsim.core.experimental.*), and isaacsim.storage.native, which Isaac Lab's kit app does not enable."""
    import types

    import omni.kit.app
    import omni.usd

    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled("isaacsim.storage.native"):
        manager.set_extension_enabled_immediate("isaacsim.storage.native", True)
    if "isaacsim.core.utils.stage" in sys.modules:
        return

    def add_reference_to_stage(usd_path: str, prim_path: str, prim_type: str = "Xform"):
        prim = omni.usd.get_context().get_stage().DefinePrim(prim_path, prim_type)
        prim.GetReferences().AddReference(usd_path)
        return prim

    for name in ("isaacsim.core.utils", "isaacsim.core.utils.stage", "isaacsim.core.api",
                 "isaacsim.core.api.materials"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["isaacsim.core.utils.stage"].add_reference_to_stage = add_reference_to_stage
    sys.modules["isaacsim.core.utils"].stage = sys.modules["isaacsim.core.utils.stage"]
    sys.modules["isaacsim.core.api.materials"].OmniPBR = _OmniPBR


class _OmniPBR:
    """The part of isaacsim.core.api.materials.OmniPBR (gone in Isaac Sim 6.0) that cell.init_materials uses:
    an OmniPBR.mdl material with constant colour / metallic / roughness inputs."""

    def __init__(self, prim_path: str, name: str = "", color=None):
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        self.material = UsdShade.Material.Define(stage, prim_path)
        shader = UsdShade.Shader.Define(stage, f"{prim_path}/Shader")
        shader.GetImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        self.material.CreateSurfaceOutput("mdl").ConnectToSource(out)
        self.material.CreateVolumeOutput("mdl").ConnectToSource(out)
        self.material.CreateDisplacementOutput("mdl").ConnectToSource(out)
        self.shaders_list = [shader]
        if color is not None:
            shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))

    def set_metallic_constant(self, value: float) -> None:
        self.shaders_list[0].CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(value))

    def set_reflection_roughness(self, value: float) -> None:
        self.shaders_list[0].CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(value))

    def set_project_uvw(self, flag: bool) -> None:
        self.shaders_list[0].CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(bool(flag))


_starter_shims()
import cell
import layout as L
import scene_config as C
from layout import pla_mass_kg, read_binary_stl

HERE = Path(__file__).resolve().parent
FINGER_DIR = HERE / "assets" / "fingers"
# Team V STEP frame: jaws close along Y, fingers point +Z, tip plane Z = 86.1 mm, contact centre X = -1.8 mm.
# The STEP "Right" finger sits on the +Y jaw (= starter JawL, sign +1); "Left" is its mirror on the -Y jaw.
TEAMV_FINGER = {"L": "MHZL2_16D2_Right_FANUC_FINAL_Finger.stl", "R": "MHZL2_16D2_Left_FANUC_FINAL_Finger.stl"}
TEAMV_PAD = "NBR_Pad_6p8x3p7x1p5_REFERENCE.stl"     # the +Y pad, open position
TEAMV_TIP_Z_MM = 86.1
STARTER_FINGER_CHILDREN = ("Finger", "SoftTip", "Detail", "Collision", "OuterCrossBolt", "SensorCarrier",
                           "SensorRedNose", "Tape")
PLA_MAT = "/World/Physics/PLA"
NBR_MAT = "/World/Physics/NBRPad"
FINGER_MAT = "/World/Physics/FingerPETG"
LOOK_BLACK_PLA = "/World/Looks/BlackPLA"


# ---------------------------------------------------------------- materials
def physics_material(stage, path: str, mat: dict) -> UsdShade.Material:
    m = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(mat["static_friction"]))
    api.CreateDynamicFrictionAttr().Set(float(mat["dynamic_friction"]))
    api.CreateRestitutionAttr().Set(float(mat["restitution"]))
    px = PhysxSchema.PhysxMaterialAPI.Apply(m.GetPrim())
    if "friction_combine" in mat:
        px.CreateFrictionCombineModeAttr().Set(mat["friction_combine"])
    return m


def black_pla_look(stage) -> UsdShade.Material:
    """Same look the SDG renders used by default (UsdPreviewSurface, near-black, mid roughness)."""
    mat = UsdShade.Material.Define(stage, LOOK_BLACK_PLA)
    sh = UsdShade.Shader.Define(stage, LOOK_BLACK_PLA + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.03, 0.03, 0.032))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind_physics(prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                                                 materialPurpose="physics")


def contact_offsets(prim, contact: dict) -> None:
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    api.CreateContactOffsetAttr().Set(float(contact["contact_offset_m"]))
    api.CreateRestOffsetAttr().Set(float(contact["rest_offset_m"]))


def mesh_prim(stage, path: str, tris_mm: np.ndarray, xform_mm: np.ndarray | None = None) -> UsdGeom.Mesh:
    """Welded triangle mesh (shared vertices). The STL is a triangle soup; written unwelded, every triangle was its
    own island, which the SDF cooker cannot sign reliably (inside vs outside of small features such as pegs)."""
    verts = tris_mm.reshape(-1, 3)
    if xform_mm is not None:
        verts = verts @ xform_mm[:3, :3].T + xform_mm[:3, 3]
    uniq, index = np.unique(np.round(verts, 5), axis=0, return_inverse=True)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((uniq / 1000.0).astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris_mm)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in index.reshape(-1)]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return mesh


# ---------------------------------------------------------------- cell + Team V fingers
def build_cell(stage, contact: dict) -> None:
    cell.build(stage, with_objects=False)
    materials = contact["materials"]
    physics_material(stage, PLA_MAT, materials["pla"])
    physics_material(stage, NBR_MAT, materials["nbr_pad"])
    physics_material(stage, FINGER_MAT, materials["finger"])
    black_pla_look(stage)
    table_mat = physics_material(stage, "/World/Physics/Table", materials["table"])
    for prim_path in ("/World/Mat", "/World/Table"):
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            bind_physics(prim, table_mat)
            contact_offsets(prim, contact)
    install_teamv_fingers(stage, contact)


def install_teamv_fingers(stage, contact: dict) -> None:
    """Replace the starter fingers on JawL / JawR with the Team V finger + NBR pad meshes."""
    carriage_x, _ = cell.gripper_layout()
    step_to_j6 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    shift_x_mm = C.TOOL_LENGTH * 1000.0 - TEAMV_TIP_Z_MM
    pad = read_binary_stl(FINGER_DIR / TEAMV_PAD)
    finger_mat = UsdShade.Material(stage.GetPrimAtPath(FINGER_MAT))
    pad_mat = UsdShade.Material(stage.GetPrimAtPath(NBR_MAT))
    for tag, sign in (("L", +1), ("R", -1)):
        jaw = cell.JAW_PRIMS[tag]
        for child in stage.GetPrimAtPath(jaw).GetChildren():
            name = child.GetName()
            if name in STARTER_FINGER_CHILDREN or name.startswith(("MountScrew", "ScrewSocket")):
                stage.RemovePrim(child.GetPath())
        jaw_origin_mm = np.array([carriage_x, sign * (C.JAW_OPEN + C.FINGER_WIDTH / 2), 0.0]) * 1000.0
        tf = np.eye(4)
        tf[:3, :3] = step_to_j6
        tf[:3, 3] = np.array([shift_x_mm, 0.0, 0.0]) - jaw_origin_mm
        finger = mesh_prim(stage, f"{jaw}/TeamVFinger", read_binary_stl(FINGER_DIR / TEAMV_FINGER[tag]), tf)
        pad_tris = pad if sign > 0 else pad * np.array([1.0, -1.0, 1.0])
        if sign < 0:
            pad_tris = pad_tris[:, ::-1]                 # mirrored: keep outward normals
        pad_mesh = mesh_prim(stage, f"{jaw}/TeamVPad", pad_tris, tf)
        finger_approx = contact.get("finger_collider", "sdf")
        for prim, approx, mat in ((finger.GetPrim(), finger_approx, finger_mat),
                                  (pad_mesh.GetPrim(), "convexHull", pad_mat)):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(approx)
            if approx == "sdf":       # Team V clearances are 0.1-0.9 mm (VALIDATION/24_part_grasp_report.csv)
                sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
                sdf.CreateSdfResolutionAttr().Set(int(contact["sdf_resolution"]))
                sdf.CreateSdfSubgridResolutionAttr().Set(int(contact["sdf_subgrid_resolution"]))
            if approx == "convexDecomposition":
                cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                cd.CreateMaxConvexHullsAttr().Set(32)
                cd.CreateVoxelResolutionAttr().Set(300000)
                cd.CreateShrinkWrapAttr().Set(True)
            contact_offsets(prim, contact)
            bind_physics(prim, mat)
        cell.bind_mat(finger.GetPrim(), "black_plastic")
        cell.bind_mat(pad_mesh.GetPrim(), "soft_tip")
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(jaw)).CreateThresholdAttr().Set(0.0)


def set_jaw_force(stage, max_force_n: float) -> None:
    """Cap both jaw drives at max_force_n; the action term keeps them in sync in software."""
    for side in ("L", "R"):
        joint = stage.GetPrimAtPath(f"{cell.FLANGE}/Gripper/{cell.JAW_JOINTS[side]}")
        UsdPhysics.DriveAPI.Get(joint, "linear").GetMaxForceAttr().Set(float(max_force_n))


# ---------------------------------------------------------------- objects
def write_object_usd(out: Path, stl: Path, contact: dict) -> float:
    """One rigid body (defaultPrim /Object) with an SDF collider and the black PLA look."""
    tris = read_binary_stl(stl)
    mass, com_mm = pla_mass_kg(stl)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.{uuid.uuid4().hex}.tmp{out.suffix}")    # pids repeat across containers
    stage = Usd.Stage.CreateNew(str(tmp))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Object").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*(com_mm / 1000.0)))
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    rb.CreateSolverPositionIterationCountAttr().Set(int(contact["solver_position_iterations"]))
    rb.CreateSolverVelocityIterationCountAttr().Set(int(contact["solver_velocity_iterations"]))
    rb.CreateMaxDepenetrationVelocityAttr().Set(float(contact["max_depenetration_velocity"]))
    rb.CreateSleepThresholdAttr().Set(float(contact["sleep_threshold"]))
    rb.CreateStabilizationThresholdAttr().Set(float(contact["stabilization_threshold"]))
    PhysxSchema.PhysxContactReportAPI.Apply(root).CreateThresholdAttr().Set(0.0)
    mesh = mesh_prim(stage, "/Object/mesh", tris).GetPrim()
    UsdPhysics.CollisionAPI.Apply(mesh)
    UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr().Set("sdf")
    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh)
    sdf.CreateSdfResolutionAttr().Set(int(contact["sdf_resolution"]))
    sdf.CreateSdfSubgridResolutionAttr().Set(int(contact["sdf_subgrid_resolution"]))
    # The SDF margin only sets how far outside the surface the field is stored; never below the PhysX default
    # fraction (0.01 of the bbox diagonal, what physics/measure_seating.py measured with): a 0.2 mm margin let the
    # 34 N pads sink ~0.9 mm into a 7 mm tab (measured jaw closure).
    margin = contact["sdf_margin"]
    if contact.get("sdf_margin_m"):
        diag_m = float(np.linalg.norm(np.ptp(tris.reshape(-1, 3), axis=0))) / 1000.0
        margin = max(margin, float(contact["sdf_margin_m"]) / diag_m)
    sdf.CreateSdfMarginAttr().Set(float(margin))
    sdf.CreateSdfNarrowBandThicknessAttr().Set(float(contact["sdf_narrow_band_thickness"]))
    contact_offsets(mesh, contact)
    if stl.name.startswith("F"):
        add_stop_boxes(stage, tris, contact)
    stage.GetRootLayer().Save()
    os.replace(tmp, out)
    return mass


def add_stop_boxes(stage, tris: np.ndarray, contact: dict) -> None:
    """Exact, invisible box colliders over a base's unchamfered stops (stop_colliders.py), next to the SDF: the SDF
    rounds their corners, and parts offset 1-1.5 mm slid off them and seated where CAD jams beyond 0.5 mm."""
    import stop_colliders

    for k, (x0, x1, y0, y1, z0, z1) in enumerate(stop_colliders.sharp_boxes(tris)):
        cube = UsdGeom.Cube.Define(stage, f"/Object/stop_{k}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d((x0 + x1) / 2000.0, (y0 + y1) / 2000.0, (z0 + z1) / 2000.0))
        cube.AddScaleOp().Set(Gf.Vec3f((x1 - x0) / 1000.0, (y1 - y0) / 1000.0, (z1 - z0) / 1000.0))
        cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        contact_offsets(cube.GetPrim(), contact)


def add_object(stage, prim_path: str, usd: Path, pose_world_m: np.ndarray) -> None:
    """Reference an object USD and bind the scene's PLA physics material and black PLA look."""
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(str(usd))
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(np.asarray(pose_world_m, float).T.tolist()))
    mesh = stage.GetPrimAtPath(prim_path + "/mesh")
    bind_physics(mesh, UsdShade.Material(stage.GetPrimAtPath(PLA_MAT)))
    for child in prim.GetChildren():
        if child.GetName().startswith("stop_"):
            bind_physics(child, UsdShade.Material(stage.GetPrimAtPath(PLA_MAT)))
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(UsdShade.Material(stage.GetPrimAtPath(LOOK_BLACK_PLA)))


OBJECT_USD_VERSION = 5      # bump when write_object_usd changes (v2 SDF margin floor, v3 sleep, v4 welded, v5 stop boxes)


def object_cache_dir(contact: dict) -> Path:
    """Cache keyed by the collision settings and the writer version, so a contact.yaml change rebuilds the USDs."""
    key = f"v{OBJECT_USD_VERSION}_" + "_".join(str(contact[k]) for k in ("sdf_resolution", "sdf_subgrid_resolution",
                                                                        "sdf_margin", "sdf_margin_m",
                                                                        "sdf_narrow_band_thickness",
                                                                        "sleep_threshold", "stabilization_threshold",
                                              "contact_offset_m", "rest_offset_m", "solver_position_iterations",
                                              "solver_velocity_iterations", "max_depenetration_velocity"))
    root = Path(os.environ.get("T12_ASSET_CACHE", Path.home() / "t12_data" / "env_assets"))
    return root / key.replace(".", "p")


def robot_physics(stage, contact: dict) -> None:
    """Every robot link reports contacts (robot_touch gate) and ignores gravity (the real controller compensates)."""
    root = stage.GetPrimAtPath(C.ROBOT_PRIM)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(True)
    add_gripper_body_mass(stage)


# SMC MHZL2-16D: 135 g from catalog listings.
# The robot USD gives J6_link only the bare flange (13.8 g) and each actuator jaw C.JAW_MASS (blade + carriage); the
# body carries the rest. A 14 g J6_link under two 32 g jaws left the PhysX articulation ill-conditioned: jaw_r's
# drive force partly vanished (14 N reached the part for a 53 N drive) and a mimic joint put ~500 N m on J5.
GRIPPER_TOTAL_KG = 0.135


def add_gripper_body_mass(stage) -> None:
    """Give J6_link the MHZL2 body mass (total minus the two modelled jaws) with the body mesh's inertia."""
    body = UsdGeom.Mesh(stage.GetPrimAtPath(f"{cell.FLANGE}/Gripper/SMCBody"))
    pts = np.array(body.GetPointsAttr().Get(), float)
    tris = np.array(body.GetFaceVertexIndicesAttr().Get(), int).reshape(-1, 3)
    xf = UsdGeom.Xformable(body.GetPrim())
    to_link = np.array(xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T @ np.linalg.inv(
        np.array(UsdGeom.Xformable(stage.GetPrimAtPath(cell.FLANGE)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())).T)
    pts = pts @ to_link[:3, :3].T + to_link[:3, 3]
    vol, com_b, inertia_b = L.mesh_mass_properties(pts, tris)
    if vol <= 0:
        raise RuntimeError(f"MHZL2 body mesh is not closed / outward ({vol * 1e6:.2f} cm3)")
    m_b = GRIPPER_TOTAL_KG - 2 * C.JAW_MASS
    inertia_b *= m_b / vol
    api = UsdPhysics.MassAPI(stage.GetPrimAtPath(cell.FLANGE))
    m0, com0 = api.GetMassAttr().Get(), np.array(api.GetCenterOfMassAttr().Get(), float)
    d0 = np.array(api.GetDiagonalInertiaAttr().Get(), float)
    q = api.GetPrincipalAxesAttr().Get()
    if m0 is None or not m0 > 0 or not d0.any():
        raise RuntimeError(f"{cell.FLANGE} has no authored mass properties to add the gripper body to")
    q = Gf.Quatd(q.GetReal(), Gf.Vec3d(q.GetImaginary())) if q is not None else Gf.Quatd(1.0)
    r0 = np.array(Gf.Matrix3d(Gf.Rotation(q)).GetTranspose(), float)
    i0 = r0 @ np.diag(d0) @ r0.T
    m = m0 + m_b
    com = (m0 * com0 + m_b * com_b) / m
    shift = lambda mass, d: mass * (d @ d * np.eye(3) - np.outer(d, d))
    total = i0 + shift(m0, com0 - com) + inertia_b + shift(m_b, com_b - com)
    w, vec = np.linalg.eigh(total)
    if np.linalg.det(vec) < 0:
        vec[:, 0] *= -1
    rot = Gf.Matrix3d(*vec.T.flatten().tolist())             # Gf rows = principal axes in the link frame
    api.GetMassAttr().Set(float(m))
    api.GetCenterOfMassAttr().Set(Gf.Vec3f(*com.tolist()))
    api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*w.tolist()))
    axes = rot.ExtractRotation().GetQuat()
    api.CreatePrincipalAxesAttr().Set(Gf.Quatf(axes.GetReal(), Gf.Vec3f(axes.GetImaginary())))
    print(f"[mass] J6_link + MHZL2 body {m_b * 1000:.1f} g (mesh {vol * 1e6:.1f} cm3): mass {m * 1000:.1f} g, "
          f"com {np.round(com * 1000, 2).tolist()} mm, principal inertia {w.tolist()}", flush=True)


def add_marker(stage, prim_path: str, pos_m) -> None:
    """A Mimic frame marker: kinematic rigid body without collider or geometry, so its pose is scene state."""
    prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*pos_m))
    UsdGeom.Xformable(prim).AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr().Set(True)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(0.001)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-8, 1e-8, 1e-8))


# ---------------------------------------------------------------- disturbance props (eval)
EXTRAS = {   # name: (size_mm, kinematic) -- boxes; parked out of every camera until a scenario injects them
    "Foreign": ((30.0, 20.0, 15.0), False),      # an unknown printed piece (not one of the 30), PLA density
    "Plug": ((8.0, 8.0, 4.0), True),             # blocks a seat: the part stops 4 mm high (won't-seat case)
    "Intruder": ((100.0, 100.0, 300.0), True),   # an operator's hand / arm in the work area
}


def add_extras(stage, root: str, park_m) -> None:
    for k, (name, (size, kinematic)) in enumerate(EXTRAS.items()):
        cube = UsdGeom.Cube.Define(stage, f"{root}/{name}")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(park_m[0], park_m[1] + 0.4 * k, park_m[2] + size[2] / 2000.0))
        xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        xf.AddScaleOp().Set(Gf.Vec3f(*(s / 1000.0 for s in size)))
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.35, 0.1) if name == "Foreign" else Gf.Vec3f(0.6, 0.6, 0.6)])
        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateKinematicEnabledAttr().Set(bool(kinematic))
        vol_cm3 = size[0] * size[1] * size[2] / 1000.0
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.24e-3 * vol_cm3)
        bind_physics(prim, UsdShade.Material(stage.GetPrimAtPath(PLA_MAT)))
