"""Manager terms (observations, events, terminations) of the cell env. Each takes the CellMimicEnv instance."""

from __future__ import annotations

import torch
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from scene import FRAMES
from task import SKILL_SIGNALS


def _vec(env, values) -> torch.Tensor:
    return torch.tensor([list(values)], dtype=torch.float32, device=env.device)


# ---------------------------------------------------------------- observations (policy group)
def joints(env) -> torch.Tensor:
    """CellClient.observe 'joints': J1..J6 deg + gripper 0 open .. 1 closed."""
    return _vec(env, env.joints_obs())


def tcp_pose(env) -> torch.Tensor:
    """CellClient.observe 'tcp_pose': UF0 [x y z w p r] mm / deg (UTOOL1 fingertip)."""
    return _vec(env, env.tcp_obs())


def eef_pos(env) -> torch.Tensor:
    return env.eef_matrix_m()[:, :3, 3]


def eef_quat(env) -> torch.Tensor:
    """TCP orientation as a quaternion (x, y, z, w): the Isaac Lab 3.x order."""
    import isaaclab.utils.math as M

    return M.quat_from_matrix(env.eef_matrix_m()[:, :3, :3])


def frame_pos(env, name: str) -> torch.Tensor:
    return env.frame_matrix_m(name)[:, :3, 3]


def frame_quat(env, name: str) -> torch.Tensor:
    import isaaclab.utils.math as M

    return M.quat_from_matrix(env.frame_matrix_m(name)[:, :3, :3])


def camera_rgb(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[sensor_cfg.name].data.output["rgb"].torch[..., :3]


def contact_peak(env) -> torch.Tensor:
    """[normal_N, lateral_N, torque_Nm, robot_touch_N] maxima over all objects during the last env step."""
    return env.step_peak_vector()


def signal(env, name: str) -> torch.Tensor:
    """Subtask term signal, latched: 0 until the subtask ended, then 1."""
    return _vec(env, [float(env.latched_signal(name))])


# ---------------------------------------------------------------- events / terminations
def reset_episode(env, env_ids) -> None:
    env.sample_episode()


def success(env) -> torch.Tensor:
    return torch.tensor([env.episode_success()], dtype=torch.bool, device=env.device)


def time_out(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


# ---------------------------------------------------------------- cfg builders
def policy_group(cameras: tuple[str, ...]) -> ObservationGroupCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        concatenate_terms: bool = False
        enable_corruption: bool = False

    g = PolicyCfg()
    g.joints = ObservationTermCfg(func=joints)
    g.tcp_pose = ObservationTermCfg(func=tcp_pose)
    g.eef_pos = ObservationTermCfg(func=eef_pos)
    g.eef_quat = ObservationTermCfg(func=eef_quat)
    for f in FRAMES:
        setattr(g, f"{f}_pos", ObservationTermCfg(func=frame_pos, params={"name": f}))
        setattr(g, f"{f}_quat", ObservationTermCfg(func=frame_quat, params={"name": f}))
    g.sim_contact_peak = ObservationTermCfg(func=contact_peak)
    for cam in cameras:
        setattr(g, f"{cam}_cam", ObservationTermCfg(func=camera_rgb, params={"sensor_cfg": SceneEntityCfg(cam)}))
    return g


def subtask_group(skill: str) -> ObservationGroupCfg:
    @configclass
    class SubtaskCfg(ObservationGroupCfg):
        concatenate_terms: bool = False
        enable_corruption: bool = False

    g = SubtaskCfg()
    for s in SKILL_SIGNALS[skill]:
        setattr(g, s, ObservationTermCfg(func=signal, params={"name": s}))
    return g


@configclass
class EventsCfg:
    reset_episode = EventTermCfg(func=reset_episode, mode="reset")


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=time_out, time_out=True)
    success = TerminationTermCfg(func=success)


@configclass
class RewardsCfg:
    pass
