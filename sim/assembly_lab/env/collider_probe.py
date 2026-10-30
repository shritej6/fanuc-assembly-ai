"""Print every collider under the objects root at runtime: approximation, SDF settings, offsets, mesh size, filters.

    isaaclab_run.sh sim/assembly_lab/env/collider_probe.py [--pool F04,M4_01]

For each collision prim: approximation (sdf / convexHull / ... / none), SDF resolution / margin / narrow band, contact
and rest offset, mesh point and triangle counts with the z extent in mm (pegs present in the collision mesh?), and
any collision-group or filtered-pair relationship that could switch object-object contact off.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="F04,M4_01")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

from cell_env import AssemblyCellEnv, CellEnvCfg


def attr(prim, name):
    a = prim.GetAttribute(name)
    return a.Get() if a and a.HasAuthoredValue() else None


def describe(prim) -> dict:
    from pxr import UsdGeom, UsdPhysics

    out = {"path": str(prim.GetPath()), "type": prim.GetTypeName(),
           "approx": attr(prim, "physics:approximation"),
           "sdf_res": attr(prim, "physxSDFMeshCollision:sdfResolution"),
           "sdf_subgrid": attr(prim, "physxSDFMeshCollision:sdfSubgridResolution"),
           "sdf_margin": attr(prim, "physxSDFMeshCollision:sdfMargin"),
           "sdf_band": attr(prim, "physxSDFMeshCollision:sdfNarrowBandThickness"),
           "contact_offset": attr(prim, "physxCollision:contactOffset"),
           "rest_offset": attr(prim, "physxCollision:restOffset"),
           "enabled": attr(prim, "physics:collisionEnabled"),
           "api": [str(s) for s in prim.GetAppliedSchemas()]}
    if prim.IsA(UsdGeom.Mesh):
        pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        idx = UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get()
        out.update(points=len(pts), tris=len(idx) // 3, z_mm=[round(float(pts[:, 2].min()) * 1000, 2),
                                                               round(float(pts[:, 2].max()) * 1000, 2)])
        from pxr import Gf

        m = Gf.Transform(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))
        out["world_scale"] = [round(float(v), 4) for v in m.GetScale()]
    fp = UsdPhysics.FilteredPairsAPI(prim)
    if fp and fp.GetFilteredPairsRel():
        out["filtered_pairs"] = [str(t) for t in fp.GetFilteredPairsRel().GetTargets()]
    return out


def main() -> None:
    env = AssemblyCellEnv(CellEnvCfg(device=args.device, pool=args.pool.split(","), cameras=()))
    from pxr import Usd, UsdPhysics

    stage = env.env.scene.stage
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and "/Objects/" in str(prim.GetPath()):
            print(f"[probe] {describe(prim)}", flush=True)
        if prim.IsA(UsdPhysics.CollisionGroup):
            g = UsdPhysics.CollisionGroup(prim)
            print(f"[probe] group {prim.GetPath()} members "
                  f"{g.GetCollidersCollectionAPI().GetIncludesRel().GetTargets()} filtered "
                  f"{g.GetFilteredGroupsRel().GetTargets()}", flush=True)
        fp = UsdPhysics.FilteredPairsAPI(prim)
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI) and fp.GetFilteredPairsRel().GetTargets():
            print(f"[probe] filtered pairs on {prim.GetPath()}: {fp.GetFilteredPairsRel().GetTargets()}", flush=True)
    print("[probe] PASS", flush=True)
    env.close()


if __name__ == "__main__":
    main()
