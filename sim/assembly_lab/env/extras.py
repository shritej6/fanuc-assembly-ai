"""Mixin for CellMimicEnv: per-episode camera / light jitter (Mimic diversity) and the eval disturbance props
(inject_foreign, block_holes, force_release, inject_intruder). Kept apart so mimic_env.py stays the Mimic core."""

from __future__ import annotations

import math

import layout as L
import numpy as np
import torch


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


class CellExtrasMixin:
    # ------------------------------------------------------------------ visual jitter (Mimic diversity)
    def _nominal_views(self) -> dict:
        """Nominal camera poses (fixed cameras: world; wrist: J6 mount) and light intensities, read once."""
        from pxr import UsdLux

        views = {}
        for cam in ("table", "side"):
            if cam in self.scene.sensors:
                d = self.scene[cam].data
                views[cam] = (d.pos_w.torch.clone(), d.quat_w_opengl.torch.clone())
        lights = []
        for prim in self.scene.stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI):
                attr = UsdLux.LightAPI(prim).GetIntensityAttr()
                if attr.HasAuthoredValue():
                    lights.append((attr, float(attr.Get())))
        return {"views": views, "wrist": None if self._wrist is None else self._wrist[1].clone(), "lights": lights}

    def apply_jitter(self) -> None:
        """Per-episode camera / light jitter from cfg.t12["jitter"] = {cam_pos_mm, cam_rot_deg, light_frac} (0 = off).

        Each fixed camera and the wrist mount move by N(0, cam_pos_mm) per axis and rotate by N(0, cam_rot_deg) about a
        random axis; every authored light intensity is scaled by U(1 - light_frac, 1 + light_frac). Nominal values are
        restored first, so jitter never accumulates."""
        j = self.cfg.t12.get("jitter") or {}
        pos_m, rot_rad, frac = j.get("cam_pos_mm", 0.0) / 1000.0, math.radians(j.get("cam_rot_deg", 0.0)), j.get("light_frac", 0.0)
        if not (pos_m or rot_rad or frac):
            return
        import isaaclab.utils.math as M

        if self._nominal is None:
            self._nominal = self._nominal_views()

        def delta() -> torch.Tensor:
            d = torch.eye(4, device=self.device)
            axis = self.rng.normal(size=3)
            d[:3, :3] = torch.tensor(_axis_angle(axis / np.linalg.norm(axis), self.rng.normal(0.0, rot_rad)),
                                     dtype=torch.float32, device=self.device)
            d[:3, 3] = torch.tensor(self.rng.normal(0.0, pos_m, 3), dtype=torch.float32, device=self.device)
            return d

        for cam, (pos, quat) in self._nominal["views"].items():
            d = delta()
            self.scene[cam].set_world_poses(pos + d[:3, 3], M.quat_from_matrix(d[:3, :3] @ M.matrix_from_quat(quat)),
                                            convention="opengl")
        if self._nominal["wrist"] is not None:
            self._wrist = (self._wrist[0], self._nominal["wrist"] @ delta())
            self.sync_wrist_camera()
        for attr, value in self._nominal["lights"]:
            attr.Set(value * float(self.rng.uniform(1.0 - frac, 1.0 + frac)))

    # ------------------------------------------------------------------ disturbances (eval scenarios, agent 6)
    def _extra(self, name: str):
        key = f"extra_{name}"
        if key not in self.scene.rigid_objects:
            raise RuntimeError("disturbance props are off: build the env with CellEnvCfg(disturbances=True)")
        return self.scene[key]

    def _place_extra(self, name: str, tf_mm: np.ndarray) -> None:
        import isaaclab.utils.math as M

        world = torch.tensor(L.uf0_to_world_m(tf_mm, self._top()), dtype=torch.float32, device=self.device)
        pose = torch.cat([world[:3, 3], M.quat_from_matrix(world[:3, :3])]).unsqueeze(0)
        obj = self._extra(name)
        obj.write_root_pose_to_sim_index(root_pose=pose)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=self.device))

    def park_extras(self) -> None:
        import assets

        if any(k.startswith("extra_") for k in self.scene.rigid_objects):
            for name in assets.EXTRAS:
                self._park_extra(name)

    def inject_foreign(self, x_mm: float, y_mm: float, yaw_deg: float = 0.0) -> None:
        """Drop the unknown 30 x 20 x 15 mm piece (not one of the 30) flat on the table at UF0 (x, y)."""
        import assets

        h = assets.EXTRAS["Foreign"][0][2]
        self._place_extra("Foreign", L.translation(x_mm, y_mm, self.solver.table_z + h / 2 + 0.2) @ L.rot_z(yaw_deg))

    def block_holes(self, part: str, base: str) -> None:
        """Put a kinematic 8 x 8 x 4 mm plug where `part` (type or uid) seats on base instance `base`, on the part's
        CAD local origin (its bottom), so the part stops 4 mm high: the won't-seat case."""
        import assets

        tid = L.type_id(part)
        seat = self.solver.seat_tf_for(tid, base, self.object_poses_mm())
        origin = seat @ self.solver.origin_in_stl(tid)
        h = assets.EXTRAS["Plug"][0][2]
        self._place_extra("Plug", L.translation(origin[0], origin[1], origin[2] + h / 2))

    def inject_intruder(self, x_mm: float, y_mm: float) -> None:
        """Stand the 100 x 100 x 300 mm 'operator arm' box on the table at UF0 (x, y); clear_intruder() removes it."""
        import assets

        h = assets.EXTRAS["Intruder"][0][2]
        self._place_extra("Intruder", L.translation(x_mm, y_mm, self.solver.table_z + h / 2))

    def clear_intruder(self) -> None:
        self._park_extra("Intruder")

    def _park_extra(self, name: str) -> None:
        import assets
        from scene import EXTRAS_PARK_M

        k = list(assets.EXTRAS).index(name)
        size = assets.EXTRAS[name][0]
        world = L.translation(EXTRAS_PARK_M[0] * 1000, (EXTRAS_PARK_M[1] + 0.4 * k) * 1000, EXTRAS_PARK_M[2] * 1000
                              + size[2] / 2)
        world[:3, 3] /= 1000.0
        self._place_extra(name, L.world_m_to_uf0(world, self._top()))

    def force_release(self) -> None:
        """Open the gripper now whatever the policy commands (dropped-in-carry case); the override ends when the policy
        itself commands open (gripper < 0.5) or at reset."""
        self.action_manager.get_term("tcp").force_open = True
