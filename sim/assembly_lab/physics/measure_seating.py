"""Measure in PhysX the force needed to seat every mating pair, aligned and misaligned.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np

ARGS = argparse.ArgumentParser()
ARGS.add_argument("--jobs", required=True)
ARGS.add_argument("--repo", default=".")
ARGS.add_argument("--out", required=True)
ARGS.add_argument("--only", default="")
ARGS.add_argument("--conditions", default="")
ARGS.add_argument("--caps", default="")
OPT = ARGS.parse_args()

from isaacsim import SimulationApp

APP = SimulationApp({"headless": True})

import carb
import omni.usd
from omni.physx import get_physx_interface, get_physx_simulation_interface
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

DT = 0.002
DESCENT_M_S = 0.010
HOLD_S = 0.2
K_Z, C_Z = 20000.0, 20.0       # N/m, N s/m (stiff, force-capped)
K_XY, C_XY = 2000.0, 5.0       # 2 N/mm pad shear (assumption)
SEATED_MM = 0.3


def read_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    (n,) = struct.unpack("<I", data[80:84])
    rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
    return rec["v"].astype(np.float64)


def local_mesh(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Welded vertices (m, part-local installed frame) and triangle indices."""
    t = np.asarray(rec["T_local_from_stl"], float)
    tri = read_stl(Path(OPT.repo) / rec["stl"]) @ t[:3, :3].T + t[:3, 3]
    pts, inv = np.unique(np.round(tri.reshape(-1, 3), 5), axis=0, return_inverse=True)
    return pts / 1000.0, inv.reshape(-1).astype(np.int32)


def quat_tilt(deg: float) -> Gf.Quatf:
    h = math.radians(deg) / 2
    return Gf.Quatf(math.cos(h), math.sin(h), 0.0, 0.0)


def add_body(stage, path: str, rec: dict, mat, sdf_res: int, contact: dict, kinematic: bool):
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*[v / 1000.0 for v in rec["origin_mm"]]))
    xf.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
    pts, idx = local_mesh(rec)
    mesh = UsdGeom.Mesh.Define(stage, f"{path}/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(idx) // 3)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(idx))
    prim, mprim = xf.GetPrim(), mesh.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(kinematic)
    UsdPhysics.CollisionAPI.Apply(mprim)
    UsdPhysics.MeshCollisionAPI.Apply(mprim).CreateApproximationAttr("sdf")
    PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mprim).CreateSdfResolutionAttr(sdf_res)
    col = PhysxSchema.PhysxCollisionAPI.Apply(mprim)
    col.CreateContactOffsetAttr(contact["collision"]["contact_offset_m"])
    col.CreateRestOffsetAttr(contact["collision"]["rest_offset_m"])
    UsdShade.MaterialBindingAPI.Apply(mprim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(rec["mass_g"] / 1000.0)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateSolverPositionIterationCountAttr(contact["solver"]["position_iterations"])
    rb.CreateSolverVelocityIterationCountAttr(contact["solver"]["velocity_iterations"])
    rb.CreateSleepThresholdAttr(0.0)
    rb.CreateMaxDepenetrationVelocityAttr(contact["solver"]["max_depenetration_velocity_m_s"])
    return xf


def build_stage(job: dict, contact: dict):
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    ps = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    ps.CreateEnableGPUDynamicsAttr(False)
    ps.CreateEnableCCDAttr(False)
    ps.CreateSolverTypeAttr("TGS")
    ps.CreateTimeStepsPerSecondAttr(round(1 / DT))
    pla = contact["materials"]["pla_printed"]
    mat = UsdShade.Material.Define(stage, "/World/PLA")
    m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    m.CreateStaticFrictionAttr(pla["static_friction"])
    m.CreateDynamicFrictionAttr(pla["dynamic_friction"])
    m.CreateRestitutionAttr(pla["restitution"])
    sdf = contact["collision"]["sdf_resolution"]
    for i, rec in enumerate(job["receivers"]):
        add_body(stage, f"/World/recv{i}", rec, mat, sdf, contact, kinematic=True)
    part = add_body(stage, "/World/part", job["moving"], mat, sdf, contact, kinematic=False)
    PhysxSchema.PhysxContactReportAPI.Apply(part.GetPrim()).CreateThresholdAttr(0.0)
    joint = UsdPhysics.Joint.Define(stage, "/World/hand")
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/part")])
    jp = joint.GetPrim()
    for axis in ("rotX", "rotY", "rotZ"):
        lim = UsdPhysics.LimitAPI.Apply(jp, axis)
        lim.CreateLowAttr(1.0)
        lim.CreateHighAttr(-1.0)
    for axis, k, c in (("transX", K_XY, C_XY), ("transY", K_XY, C_XY), ("transZ", K_Z, C_Z)):
        d = UsdPhysics.DriveAPI.Apply(jp, axis)
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(k)
        d.CreateDampingAttr(c)
        d.CreateMaxForceAttr(100.0)
        d.CreateTargetPositionAttr(0.0)
    return stage, part, joint


def set_condition(stage, part, joint, job: dict, cond: dict, cap: float) -> dict:
    seat = np.array(job["moving"]["origin_mm"], float) / 1000.0
    start = seat + [cond["offset_mm"][0] / 1000.0, cond["offset_mm"][1] / 1000.0,
                    (job["engagement_mm"] + 3.0) / 1000.0]
    q = quat_tilt(cond["tilt_deg"])
    part.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(*start))
    part.GetPrim().GetAttribute("xformOp:orient").Set(q)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*start))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr().Set(q.GetInverse())
    jp = joint.GetPrim()
    jp.GetAttribute("drive:transZ:physics:maxForce").Set(cap)
    jp.GetAttribute("drive:transZ:physics:targetPosition").Set(0.0)
    return {"seat": seat, "start": start}


def contact_wrench(part_int: int, origin: np.ndarray) -> np.ndarray:
    heads, data = get_physx_simulation_interface().get_contact_report()
    f, tz = np.zeros(3), 0.0
    for h in heads:
        if part_int not in (h.actor0, h.actor1):
            continue
        s = 1.0 if h.actor0 == part_int else -1.0
        for k in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
            c = data[k]
            fi = s * np.array([c.impulse[0], c.impulse[1], c.impulse[2]]) / DT
            r = np.array([c.position[0], c.position[1], c.position[2]]) - origin
            f += fi
            tz += r[0] * fi[1] - r[1] * fi[0]
    return np.array([f[0], f[1], f[2], tz])


def run(stage, part, joint, job: dict, cond: dict, cap: float) -> dict:
    geo = set_condition(stage, part, joint, job, cond, cap)
    sim, phx = get_physx_simulation_interface(), get_physx_interface()
    stage_id = omni.usd.get_context().get_stage_id()
    sim.attach_stage(stage_id)
    part_int = PhysicsSchemaTools.sdfPathToInt(Sdf.Path("/World/part"))
    tgt = joint.GetPrim().GetAttribute("drive:transZ:physics:targetPosition")
    floor = geo["seat"][2] - 0.001
    travel = geo["start"][2] - floor
    steps = int((travel / DESCENT_M_S + HOLD_S) / DT)
    log, t0, first_seat = [], time.time(), None
    for i in range(steps):
        tgt.Set(-min(DESCENT_M_S * i * DT, travel))
        sim.simulate(DT, i * DT)
        sim.fetch_results()
        pose = phx.get_rigidbody_transformation("/World/part")
        pos = np.array(pose["position"], float)
        w = contact_wrench(part_int, pos)
        dz = (pos[2] - geo["seat"][2]) * 1000.0
        if first_seat is None and abs(dz) <= SEATED_MM:
            first_seat = i
        log.append((dz, abs(w[2]), math.hypot(w[0], w[1]), abs(w[3])))
    sim.detach_stage()
    arr = np.array(log)
    pre = arr[:first_seat] if first_seat else arr
    final_xy = (pos[:2] - geo["seat"][:2]) * 1000.0
    return {"pair": job["pair"], "condition": cond["name"], "cap_N": cap,
            "seated": bool(abs(arr[-1, 0]) <= SEATED_MM), "final_dz_mm": round(float(arr[-1, 0]), 4),
            "final_xy_err_mm": [round(float(v), 4) for v in final_xy],
            "peak_axial_N": round(float(arr[:, 1].max()), 4), "peak_lateral_N": round(float(arr[:, 2].max()), 4),
            "peak_torque_Nm": round(float(arr[:, 3].max()), 6),
            "pre_seat_peak_axial_N": round(float(pre[:, 1].max()) if len(pre) else 0.0, 4),
            "pre_seat_peak_lateral_N": round(float(pre[:, 2].max()) if len(pre) else 0.0, 4),
            "pre_seat_peak_torque_Nm": round(float(pre[:, 3].max()) if len(pre) else 0.0, 6),
            "steps": steps, "wall_s": round(time.time() - t0, 2)}


def main() -> None:
    spec = json.loads(Path(OPT.jobs).read_text())
    carb.settings.get_settings().set("/physics/updateToUsd", False)
    carb.settings.get_settings().set("/physics/fabricEnabled", False)
    wanted = [s for s in OPT.only.split(",") if s]
    conds = [c for c in spec["conditions"] if not OPT.conditions or c["name"] in OPT.conditions.split(",")]
    caps = [float(c) for c in OPT.caps.split(",")] if OPT.caps else spec["caps_N"]
    out = Path(OPT.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for job in spec["jobs"]:
            if wanted and not any(w in job["pair"] for w in wanted):
                continue
            stage, part, joint = build_stage(job, spec["contact"])
            for cond in conds:
                for cap in caps:
                    res = run(stage, part, joint, job, cond, cap)
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                    print(json.dumps(res), flush=True)
                    if not res["seated"]:
                        break
    APP.close()


if __name__ == "__main__":
    main()
