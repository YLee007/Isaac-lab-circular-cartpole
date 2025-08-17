from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude, quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def rk_get_body_link_quat_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    output = asset.data.body_link_quat_w[:, 1:].flatten(start_dim=1)
    return output


def rk_get_body_link_ang_vel_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]
    output = asset.data.body_link_ang_vel_w[:, 1:, 1].flatten(start_dim=1)
    return output


def rk_get_pole_rotation_error(
    env: ManagerBasedEnv,
    pole_index: int,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    # Convert target position to tensor with proper device and dtype
    target_pos_tensor = torch.tensor(
        target_pos, device=asset.device, dtype=asset.data.body_quat_w.dtype
    )

    target_quat = quat_from_euler_xyz(
        target_pos_tensor[0], target_pos_tensor[1], target_pos_tensor[2]
    )

    current_quat = asset.data.body_link_quat_w[:, pole_index, :]
    target_quat_expanded = target_quat.unsqueeze(0).expand_as(current_quat)
    angular_error = quat_error_magnitude(current_quat, target_quat_expanded)

    angular_error = torch.clamp(angular_error, min=0.0, max=torch.pi)
    angular_error = torch.where(
        torch.isnan(angular_error), torch.zeros_like(angular_error), angular_error
    )
    return angular_error.unsqueeze(-1)
