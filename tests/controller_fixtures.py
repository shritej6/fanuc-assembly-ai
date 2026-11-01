"""Shared deterministic controller fixtures that are not production safety providers.
Authored by Shritej Gunda.
"""

from assembly.safety import AABBCollisionChecker, ReachabilityCapabilities, SafetyConfig, SafetySupervisor


class AlwaysReachable:
    capabilities = ReachabilityCapabilities(full_robot_kinematics=False)

    def is_reachable(self, pose: tuple[float, float, float, float, float, float]) -> bool:
        return True


def build_test_safety() -> SafetySupervisor:
    return SafetySupervisor(
        config=SafetyConfig(
            0.1,
            -335.0,
            1.0,
            -180.0,
            0.0,
            0.5,
            1.0,
            10000.0,
            "mock-only test fixture",
        ),
        reachability=AlwaysReachable(),
        collision=AABBCollisionChecker(),
        allow_tcp_sweep_only_for_mock=True,
    )
