"""Renderer settings shared by the render scripts.
Authored by Shritej Gunda.
"""

import carb


def enable_path_tracing(spp=64, max_bounces=6):
    s = carb.settings.get_settings()
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/spp", spp)
    s.set("/rtx/pathtracing/totalSpp", spp)
    s.set("/rtx/pathtracing/maxBounces", max_bounces)
    # Inner and outer walls of a clear tube need several transmission bounces.
    s.set("/rtx/pathtracing/maxBounces", max(max_bounces, 16))
    s.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 16)
    s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    s.set("/rtx/ambientOcclusion/enabled", True)
    print(f"[render] path tracing enabled ({spp} spp, {max_bounces} bounces)")


def realtime():
    carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")
