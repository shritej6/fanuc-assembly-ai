"""Turn a scene spec into USD prims on the starter cell. Runs inside Isaac Sim (imports pxr / starter modules).

UNTESTED until the Rosie pilot (T08). Written against the Isaac Sim 6.x Python API.

Frames: specs are UF0 mm. The starter stage is metres with the robot base mount at (0, 0, TABLE_TOP_Z), and UF0's
origin is 330 mm above the base mount, so  world = (x/1000, y/1000, TABLE_TOP_Z + (z + 330)/1000).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdSemantics, UsdShade, Vt

UF0_ABOVE_BASE_M = 0.330
LIB_ROOT = "/_sdg_lib"           # class prim: abstract, never rendered, safe to reference
SCENE_ROOT = "/World/SdgScene"
MATERIAL_PATH = "/World/SdgLooks/BlackPLA"
STARTER_LIGHTS = ("/World/DomeLight", "/World/KeyLight", "/World/FillLight", "/World/RimLight")


class MissingVariantMesh(FileNotFoundError):
    """A defect variant is requested but its mesh (T06) is absent. Never render the good part under a defect label."""


def read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    (n,) = struct.unpack_from("<I", data, 80)
    if len(data) != 84 + 50 * n:
        raise ValueError(f"{path}: not a binary STL")
    rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
    verts = rec["v"].reshape(-1, 3).astype(np.float64)
    return verts, np.arange(len(verts)).reshape(-1, 3)


def uf0_to_world(tf_mm: np.ndarray, table_top_z: float) -> Gf.Matrix4d:
    m = np.array(tf_mm, dtype=np.float64)
    m[:3, 3] = m[:3, 3] / 1000.0 + (0.0, 0.0, table_top_z + UF0_ABOVE_BASE_M)
    return Gf.Matrix4d(m.T.tolist())    # USD is row-vector: transpose


class SceneStage:
    def __init__(self, stage, repo: Path, table_top_z: float):
        self.stage, self.repo, self.table_top_z = stage, repo, table_top_z
        self._lib: set[str] = set()
        stage.CreateClassPrim(LIB_ROOT)
        self.material = self._make_material()
        self._light_base = {p: stage.GetPrimAtPath(p).GetAttribute("inputs:intensity").Get() for p in STARTER_LIGHTS}

    # ------------------------------------------------------------------ meshes
    def _mesh_key(self, obj: dict) -> tuple[str, Path]:
        if obj["variant"] == "ok":
            return obj["object_id"], self.repo / obj["stl"]
        path = self.repo / "cad" / "defects" / f"{obj['object_id']}__{obj['variant']}.stl"
        if not path.exists():
            raise MissingVariantMesh(f"{path} (run T06 first)")
        return f"{obj['object_id']}__{obj['variant']}", path

    def _library_mesh(self, key: str, path: Path) -> str:
        prim_path = f"{LIB_ROOT}/{key}"
        if key not in self._lib:
            verts, faces = read_binary_stl(path)
            mesh = UsdGeom.Mesh.Define(self.stage, prim_path)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((verts / 1000.0).astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).tolist()))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.material)
            self._lib.add(key)
        return prim_path

    # ------------------------------------------------------------------ per scene
    def load(self, spec: dict) -> dict[str, str]:
        """Rebuild the scene objects; returns {prim path: uid} for label lookup."""
        if self.stage.GetPrimAtPath(SCENE_ROOT).IsValid():
            self.stage.RemovePrim(SCENE_ROOT)
        UsdGeom.Xform.Define(self.stage, SCENE_ROOT)
        paths = {}
        for obj in spec["objects"]:
            xf = UsdGeom.Xform.Define(self.stage, f"{SCENE_ROOT}/{obj['uid']}")
            xf.AddTransformOp().Set(uf0_to_world(obj["transform_uf0"], self.table_top_z))
            geom = self.stage.DefinePrim(f"{xf.GetPath()}/geom")
            if obj["kind"] == "foreign":
                self._foreign(geom.GetPath(), obj["foreign_type"])
            else:
                geom.GetReferences().AddInternalReference(self._library_mesh(*self._mesh_key(obj)))
            label = obj["object_id"] or f"foreign_{obj['foreign_type']}"
            _add_class_label(xf.GetPrim(), label)
            paths[str(xf.GetPath())] = obj["uid"]
        self._apply_look(spec)
        return paths

    def _foreign(self, path: Sdf.Path, kind: str) -> None:
        if kind == "cube_10mm":
            cube = UsdGeom.Cube.Define(self.stage, path)
            cube.CreateSizeAttr(0.010)
            cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.005))
            cube.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.08, 0.06)])
            return
        radius, height, colour = {"screw_m3": (0.0028, 0.003, (0.7, 0.7, 0.72)),
                                  "tape_roll": (0.026, 0.018, (0.82, 0.78, 0.6))}[kind]
        cyl = UsdGeom.Cylinder.Define(self.stage, path)
        cyl.CreateRadiusAttr(radius)
        cyl.CreateHeightAttr(height)
        cyl.CreateAxisAttr("Z")
        cyl.AddTranslateOp().Set(Gf.Vec3d(0, 0, height / 2))
        cyl.CreateDisplayColorAttr([Gf.Vec3f(*colour)])

    # ------------------------------------------------------------------ look
    def _make_material(self) -> UsdShade.Material:
        mat = UsdShade.Material.Define(self.stage, MATERIAL_PATH)
        shader = UsdShade.Shader.Define(self.stage, f"{MATERIAL_PATH}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        for name, typ in (("diffuseColor", Sdf.ValueTypeNames.Color3f), ("roughness", Sdf.ValueTypeNames.Float),
                          ("useSpecularWorkflow", Sdf.ValueTypeNames.Int), ("specularColor", Sdf.ValueTypeNames.Color3f)):
            shader.CreateInput(name, typ)
        shader.GetInput("useSpecularWorkflow").Set(1)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self._shader = shader
        return mat

    def _apply_look(self, spec: dict) -> None:
        m = spec["material"]
        self._shader.GetInput("diffuseColor").Set(Gf.Vec3f(*m["base_color_rgb"]))
        self._shader.GetInput("roughness").Set(float(m["roughness"]))
        self._shader.GetInput("specularColor").Set(Gf.Vec3f(*([m["specular"] * 0.08] * 3)))
        light = spec["lighting"]
        for path, base in self._light_base.items():
            prim = self.stage.GetPrimAtPath(path)
            scale = light["dome_intensity_scale"] if path.endswith("DomeLight") else light["rect_intensity_scale"]
            prim.GetAttribute("inputs:intensity").Set(base * scale)
            lux = UsdLux.LightAPI(prim)
            lux.CreateEnableColorTemperatureAttr(True)
            lux.CreateColorTemperatureAttr(float(light["color_temperature_k"]))


def _add_class_label(prim, label: str) -> None:
    """Semantic class for Replicator annotators (UsdSemantics.LabelsAPI, the Isaac Sim 5+ API)."""
    labels = UsdSemantics.LabelsAPI.Apply(prim, "class")
    labels.CreateLabelsAttr([label])
